import json
import tempfile
import unittest
from pathlib import Path

from config import Config
from document_loader import Document, ParentChildSplitter
from embeddings.base import EmbeddingConfig, EmbeddingProvider
from eval.run import main as eval_main
from parent_store import InMemoryParentStore
from rag import RAGPipeline
from vector_store import InMemoryVectorStore, VectorRecord


class StaticEmbeddingProvider(EmbeddingProvider):
    """Deterministic query embedding provider for parent-child tests."""

    def __init__(self):
        super().__init__(EmbeddingConfig(model_name="static-test", dimension=2))

    def embed_texts(self, texts):
        """Return the same query vector for every text."""
        return [[1.0, 0.0] for _ in list(texts)]


class FakeChatClient:
    """Chat client placeholder for pipeline construction."""

    def chat(self, message, system_prompt=None):
        """Return a fixed response."""
        return {"choices": [{"message": {"content": "ok [1]"}}]}


class ParentChildSplitterTests(unittest.TestCase):
    def test_parent_child_splitter_maps_every_child_to_one_covering_parent(self):
        document = Document(
            content=(
                "Alpha introduction for the handbook.\n\n"
                "Beta target detail explains reimbursement and receipts.\n\n"
                "Gamma closing paragraph keeps the policy readable."
            ),
            metadata={"source": "policy.md", "file_type": "markdown"},
        )
        splitter = ParentChildSplitter(
            parent_chunk_size=90,
            parent_chunk_overlap=10,
            child_chunk_size=45,
            child_chunk_overlap=5,
        )

        result = splitter.split_document(document)
        parents_by_id = {parent.metadata["parent_id"]: parent for parent in result.parents}

        self.assertGreaterEqual(len(result.parents), 1)
        self.assertGreaterEqual(len(result.children), 2)
        for child in result.children:
            parent = parents_by_id[child.metadata["parent_id"]]
            self.assertIn(child.content, parent.content)
            self.assertGreaterEqual(child.metadata["start_char"], parent.metadata["start_char"])
            self.assertLessEqual(child.metadata["end_char"], parent.metadata["end_char"])
            self.assertEqual(child.metadata["parent_index"], parent.metadata["parent_index"])

    def test_parent_child_splitter_validates_sizes(self):
        with self.assertRaises(ValueError):
            ParentChildSplitter(parent_chunk_size=100, child_chunk_size=100)
        with self.assertRaises(ValueError):
            ParentChildSplitter(parent_chunk_size=100, parent_chunk_overlap=100, child_chunk_size=40)


class ParentStoreTests(unittest.TestCase):
    def test_parent_store_is_key_value_only(self):
        parent = Document(
            content="Full parent content.",
            metadata={"parent_id": "parent-1", "source": "a.md", "parent_index": 0},
        )
        store = InMemoryParentStore()

        ids = store.add_parents([parent])

        self.assertEqual(ids, ["parent-1"])
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.get_parent("parent-1").content, "Full parent content.")
        self.assertFalse(hasattr(store, "similarity_search"))

    def test_parent_store_requires_parent_id(self):
        store = InMemoryParentStore()

        with self.assertRaises(ValueError):
            store.add_parents([Document(content="missing id", metadata={})])


class ParentExpansionTests(unittest.TestCase):
    def build_pipeline(self, expand_parent_context=True):
        parent = Document(
            content="Alpha intro. Beta target detail. Gamma closing sentence.",
            metadata={
                "parent_id": "parent-1",
                "source": "policy.md",
                "parent_index": 0,
                "start_char": 0,
                "end_char": 56,
            },
        )
        parent_store = InMemoryParentStore()
        parent_store.add_parents([parent])

        vector_store = InMemoryVectorStore(dimension=2)
        vector_store.add_records(
            [
                VectorRecord(
                    id="child-1",
                    content="Beta target detail.",
                    embedding=[1.0, 0.0],
                    metadata={
                        "id": "child-1",
                        "source": "policy.md",
                        "chunk_index": 0,
                        "start_char": 13,
                        "end_char": 32,
                        "parent_id": "parent-1",
                    },
                ),
                VectorRecord(
                    id="child-2",
                    content="Gamma closing sentence.",
                    embedding=[0.9, 0.1],
                    metadata={
                        "id": "child-2",
                        "source": "policy.md",
                        "chunk_index": 1,
                        "start_char": 33,
                        "end_char": 56,
                        "parent_id": "parent-1",
                    },
                ),
            ]
        )
        return RAGPipeline(
            embedding_provider=StaticEmbeddingProvider(),
            vector_store=vector_store,
            chat_client=FakeChatClient(),
            top_k=2,
            parent_store=parent_store,
            expand_parent_context=expand_parent_context,
        )

    def test_parent_expansion_collapses_sibling_children_to_one_parent(self):
        pipeline = self.build_pipeline(expand_parent_context=True)

        sources = pipeline.retrieve("target detail", top_k=2)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].index, 1)
        self.assertEqual(sources[0].content, "Alpha intro. Beta target detail. Gamma closing sentence.")
        self.assertEqual(sources[0].metadata["child_id"], "child-1")
        self.assertEqual(sources[0].metadata["collapsed_child_count"], 2)
        self.assertEqual(sources[0].metadata["collapsed_child_ids"], ["child-1", "child-2"])

    def test_parent_expansion_can_be_disabled_for_same_child_ranked_list(self):
        pipeline = self.build_pipeline(expand_parent_context=False)

        sources = pipeline.retrieve("target detail", top_k=2)

        self.assertEqual([source.content for source in sources], ["Beta target detail.", "Gamma closing sentence."])
        self.assertNotIn("parent_expanded", sources[0].metadata)


class ParentChildEvalTests(unittest.TestCase):
    def test_parent_child_eval_report_is_reproducible(self):
        def run_report_json():
            with tempfile.TemporaryDirectory() as temp_dir:
                eval_main(
                    [
                        "--golden-set",
                        "eval/golden_set.jsonl",
                        "--knowledge-path",
                        "eval/fixtures/knowledge_base",
                        "--report-dir",
                        temp_dir,
                        "--parent-child",
                        "--quiet",
                    ]
                )
                json_path = next(Path(temp_dir).glob("*.json"))
                return json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(run_report_json(), run_report_json())

    def test_config_validates_parent_child_values(self):
        original = {
            "PARENT_CHUNK_SIZE": Config.PARENT_CHUNK_SIZE,
            "PARENT_CHUNK_OVERLAP": Config.PARENT_CHUNK_OVERLAP,
            "CHILD_CHUNK_SIZE": Config.CHILD_CHUNK_SIZE,
            "CHILD_CHUNK_OVERLAP": Config.CHILD_CHUNK_OVERLAP,
        }
        try:
            Config.PARENT_CHUNK_SIZE = 400
            Config.PARENT_CHUNK_OVERLAP = 80
            Config.CHILD_CHUNK_SIZE = 400
            Config.CHILD_CHUNK_OVERLAP = 40

            with self.assertRaises(ValueError):
                Config.validate_parent_child()
        finally:
            for key, value in original.items():
                setattr(Config, key, value)


if __name__ == "__main__":
    unittest.main()
