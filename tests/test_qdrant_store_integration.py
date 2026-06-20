import os
import subprocess
import sys
import unittest
import uuid

from embeddings import HashEmbeddingProvider
from rag_cli import build_rag_pipeline_from_index, run_single_question
from vector_store import InMemoryVectorStore, QdrantVectorStore, VectorRecord, is_qdrant_client_available


class FakeChatClient:
    """Fake chat client for integration tests that should not call the network."""

    def chat(self, message, system_prompt=None):
        """Return a deterministic answer payload."""
        return {"choices": [{"message": {"content": "Qdrant persisted answer [1]."}}]}


@unittest.skipUnless(
    os.getenv("QDRANT_URL") and is_qdrant_client_available(),
    "Set QDRANT_URL and install qdrant-client to run Qdrant integration tests.",
)
class QdrantVectorStoreIntegrationTests(unittest.TestCase):
    def build_records(self):
        return [
            VectorRecord(
                id="alpha",
                content="alpha content",
                embedding=[1.0, 0.0],
                metadata={"source": "a.txt", "file_type": "txt", "chunk_index": 0},
            ),
            VectorRecord(
                id="beta",
                content="beta content",
                embedding=[0.0, 1.0],
                metadata={"source": "b.txt", "file_type": "txt", "chunk_index": 1},
            ),
        ]

    def build_qdrant_store(self, recreate=False):
        return QdrantVectorStore(
            collection_name=os.getenv("QDRANT_COLLECTION", "ai_qa_assistant_test"),
            dimension=2,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            recreate=recreate,
        )

    def test_qdrant_matches_memory_top_k_and_persists_records(self):
        records = self.build_records()
        memory_store = InMemoryVectorStore(dimension=2)
        memory_store.add_records(records)

        qdrant_store = self.build_qdrant_store(recreate=True)
        qdrant_store.add_records(records)
        qdrant_store.add_records(records)

        query_embedding = [1.0, 0.0]
        memory_ids = [result.record.id for result in memory_store.similarity_search(query_embedding, top_k=2)]
        qdrant_ids = [result.record.id for result in qdrant_store.similarity_search(query_embedding, top_k=2)]

        self.assertEqual(memory_ids, qdrant_ids)
        self.assertEqual(qdrant_store.count(), len(records))

        restarted_store = self.build_qdrant_store(recreate=False)
        persisted_ids = [
            result.record.id for result in restarted_store.similarity_search(query_embedding, top_k=2)
        ]

        self.assertEqual(persisted_ids, memory_ids)

    def test_qdrant_acl_match_any_filters_before_top_k(self):
        records = [
            VectorRecord(
                id="unauthorized-nearest",
                content="nearest but forbidden",
                embedding=[1.0, 0.0],
                metadata={"source": "secret.txt", "acl": ["role:secret"], "chunk_index": 0},
            ),
            VectorRecord(
                id="authorized-farther",
                content="farther but allowed",
                embedding=[0.0, 1.0],
                metadata={"source": "finance.txt", "acl": ["role:finance"], "chunk_index": 1},
            ),
        ]
        qdrant_store = self.build_qdrant_store(recreate=True)
        qdrant_store.add_records(records)

        results = qdrant_store.similarity_search(
            [1.0, 0.0],
            top_k=1,
            metadata_filter={"acl": ["role:finance"]},
        )
        denied = qdrant_store.similarity_search(
            [1.0, 0.0],
            top_k=1,
            metadata_filter={"acl": ["role:legal"]},
        )

        self.assertEqual([result.record.id for result in results], ["authorized-farther"])
        self.assertEqual(denied, [])

    def test_rag_cli_ingest_and_query_split_persists_across_processes(self):
        collection_name = f"ai_qa_t03_{uuid.uuid4().hex}"
        ingest_script = f"""
from rag_cli import ingest_documents

ingest_documents(
    'tests/fixtures',
    clean=True,
    recursive=True,
    chunk_size=100,
    chunk_overlap=10,
    embedding_provider_name='hash',
    embedding_dimension=64,
    vector_store_name='qdrant',
    parent_child_enabled=False,
    vector_store_overrides={{
        'collection_name': '{collection_name}',
        'url': '{os.getenv("QDRANT_URL")}',
        'api_key': {os.getenv("QDRANT_API_KEY")!r},
        'recreate': True,
    }},
)
"""
        subprocess.run(
            [sys.executable, "-c", ingest_script],
            cwd=os.getcwd(),
            check=True,
            capture_output=True,
            text=True,
        )

        provider = HashEmbeddingProvider(dimension=64)
        query_store = QdrantVectorStore(
            collection_name=collection_name,
            dimension=64,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            recreate=False,
        )
        pipeline = build_rag_pipeline_from_index(
            chat_client=FakeChatClient(),
            embedding_provider=provider,
            vector_store=query_store,
            rerank_provider_name="none",
            parent_child_enabled=False,
            top_k=2,
            max_context_chars=1000,
        )

        response = run_single_question(pipeline, "What is this project?", top_k=2)

        self.assertEqual(response.answer, "Qdrant persisted answer [1].")
        self.assertGreaterEqual(len(response.sources), 1)
        self.assertGreater(query_store.count(), 0)

        query_store.clear()


if __name__ == "__main__":
    unittest.main()
