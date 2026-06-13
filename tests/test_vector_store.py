import unittest

from document_loader import load_and_split_document
from embeddings import EmbeddedDocument, HashEmbeddingProvider
from tests.test_document_ingestion import FIXTURES_DIR
from vector_store import InMemoryVectorStore, VectorRecord


class InMemoryVectorStoreTests(unittest.TestCase):
    def test_add_records_stores_records_and_returns_ids(self):
        store = InMemoryVectorStore(dimension=2)
        ids = store.add_records(
            [
                VectorRecord(
                    id="record-1",
                    content="alpha",
                    embedding=[1.0, 0.0],
                    metadata={"source": "a.txt"},
                )
            ]
        )

        self.assertEqual(ids, ["record-1"])
        self.assertEqual(store.count(), 1)
        self.assertEqual(store.get_record("record-1").content, "alpha")

    def test_add_records_rejects_dimension_mismatch(self):
        store = InMemoryVectorStore(dimension=2)

        with self.assertRaises(ValueError):
            store.add_records(
                [
                    VectorRecord(
                        id="bad-record",
                        content="bad",
                        embedding=[1.0, 0.0, 0.0],
                    )
                ]
            )

    def test_similarity_search_returns_top_results_by_score(self):
        store = InMemoryVectorStore(dimension=2)
        store.add_records(
            [
                VectorRecord(id="x", content="x-axis", embedding=[1.0, 0.0]),
                VectorRecord(id="y", content="y-axis", embedding=[0.0, 1.0]),
            ]
        )

        results = store.similarity_search([1.0, 0.0], top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record.id, "x")
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_similarity_search_supports_metadata_filter(self):
        store = InMemoryVectorStore(dimension=2)
        store.add_records(
            [
                VectorRecord(
                    id="public",
                    content="public content",
                    embedding=[1.0, 0.0],
                    metadata={"visibility": "public"},
                ),
                VectorRecord(
                    id="private",
                    content="private content",
                    embedding=[1.0, 0.0],
                    metadata={"visibility": "private"},
                ),
            ]
        )

        results = store.similarity_search(
            [1.0, 0.0],
            top_k=5,
            metadata_filter={"visibility": "private"},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record.id, "private")

    def test_similarity_search_rejects_invalid_top_k(self):
        store = InMemoryVectorStore(dimension=2)

        with self.assertRaises(ValueError):
            store.similarity_search([1.0, 0.0], top_k=0)

    def test_delete_and_clear_records(self):
        store = InMemoryVectorStore(dimension=2)
        store.add_records([VectorRecord(id="record-1", content="alpha", embedding=[1.0, 0.0])])

        self.assertTrue(store.delete("record-1"))
        self.assertFalse(store.delete("record-1"))
        self.assertEqual(store.count(), 0)

        store.add_records([VectorRecord(id="record-2", content="beta", embedding=[0.0, 1.0])])
        store.clear()
        self.assertEqual(store.count(), 0)

    def test_add_documents_converts_embedded_documents_to_records(self):
        store = InMemoryVectorStore(dimension=2)
        embedded_documents = [
            EmbeddedDocument(
                content="alpha",
                metadata={"source": "a.txt", "chunk_index": 0},
                embedding=[1.0, 0.0],
            )
        ]

        ids = store.add_documents(embedded_documents)

        self.assertEqual(store.count(), 1)
        self.assertEqual(store.get_record(ids[0]).metadata["source"], "a.txt")
        self.assertEqual(store.get_record(ids[0]).metadata["id"], ids[0])

    def test_local_ingestion_embedding_and_search_pipeline(self):
        chunks = load_and_split_document(
            str(FIXTURES_DIR / "sample.txt"),
            clean=True,
            chunk_size=80,
            chunk_overlap=10,
        )
        provider = HashEmbeddingProvider(dimension=64)
        embedded_documents = provider.embed_documents(chunks)

        store = InMemoryVectorStore(dimension=64)
        store.add_documents(embedded_documents)

        query_embedding = provider.embed_text("RAG ingestion prototype")
        results = store.similarity_search(query_embedding, top_k=3)

        self.assertEqual(store.count(), len(embedded_documents))
        self.assertGreaterEqual(len(results), 1)
        self.assertIn("source", results[0].metadata)
        self.assertIn("chunk_index", results[0].metadata)
        self.assertEqual(len(results[0].record.embedding), 64)


if __name__ == "__main__":
    unittest.main()
