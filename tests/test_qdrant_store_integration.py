import os
import unittest

from vector_store import InMemoryVectorStore, QdrantVectorStore, VectorRecord, is_qdrant_client_available


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


if __name__ == "__main__":
    unittest.main()

