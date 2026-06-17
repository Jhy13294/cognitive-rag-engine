import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vector_store import QdrantVectorStore, QdrantVectorStoreError, VectorRecord, create_vector_store
from vector_store.qdrant_store import INDEXED_PAYLOAD_FIELDS


class RetryableQdrantError(Exception):
    status_code = 429


class NonRetryableQdrantError(Exception):
    status_code = 400


class FakeQdrantClient:
    def __init__(self, exists=False, vector_size=2):
        self.exists = exists
        self.vector_size = vector_size
        self.created_collections = []
        self.deleted_collections = []
        self.payload_indexes = []
        self.upserts = []
        self.search_calls = []
        self.retrieve_calls = []
        self.scroll_calls = []
        self.delete_calls = []
        self.count_value = 0
        self.search_hits = []
        self.retrieve_points = []
        self.scroll_pages = []
        self.fail_upsert_attempts = 0
        self.fail_upsert_with = RetryableQdrantError("rate limited")

    def collection_exists(self, collection_name):
        return self.exists

    def get_collection(self, collection_name):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=self.vector_size),
                )
            )
        )

    def create_collection(self, collection_name, vectors_config):
        self.created_collections.append(
            {
                "collection_name": collection_name,
                "vectors_config": vectors_config,
            }
        )
        self.exists = True

    def delete_collection(self, collection_name):
        self.deleted_collections.append(collection_name)
        self.exists = False

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.payload_indexes.append(
            {
                "collection_name": collection_name,
                "field_name": field_name,
                "field_schema": field_schema,
            }
        )

    def upsert(self, collection_name, points, wait):
        if self.fail_upsert_attempts > 0:
            self.fail_upsert_attempts -= 1
            raise self.fail_upsert_with

        self.upserts.append(
            {
                "collection_name": collection_name,
                "points": points,
                "wait": wait,
            }
        )

    def search(self, collection_name, query_vector, query_filter, limit, with_payload):
        self.search_calls.append(
            {
                "collection_name": collection_name,
                "query_vector": query_vector,
                "query_filter": query_filter,
                "limit": limit,
                "with_payload": with_payload,
            }
        )
        return self.search_hits

    def retrieve(self, collection_name, ids, with_payload, with_vectors):
        self.retrieve_calls.append(
            {
                "collection_name": collection_name,
                "ids": ids,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        return self.retrieve_points

    def scroll(self, collection_name, limit, offset, with_payload, with_vectors):
        self.scroll_calls.append(
            {
                "collection_name": collection_name,
                "limit": limit,
                "offset": offset,
                "with_payload": with_payload,
                "with_vectors": with_vectors,
            }
        )
        if self.scroll_pages:
            return self.scroll_pages.pop(0)
        return [], None

    def delete(self, collection_name, points_selector, wait):
        self.delete_calls.append(
            {
                "collection_name": collection_name,
                "points_selector": points_selector,
                "wait": wait,
            }
        )

    def count(self, collection_name, exact):
        return SimpleNamespace(count=self.count_value)


class QdrantVectorStoreMockTests(unittest.TestCase):
    def build_store(self, client=None, **kwargs):
        return QdrantVectorStore(
            collection_name="kb",
            dimension=2,
            client=client or FakeQdrantClient(),
            batch_size=kwargs.pop("batch_size", 2),
            base_delay=0,
            max_delay=0,
            **kwargs,
        )

    def test_initializes_collection_and_payload_indexes(self):
        client = FakeQdrantClient(exists=False)

        self.build_store(client)

        self.assertEqual(len(client.created_collections), 1)
        indexed_fields = {item["field_name"] for item in client.payload_indexes}
        self.assertEqual(indexed_fields, set(INDEXED_PAYLOAD_FIELDS))

    def test_rejects_existing_collection_dimension_mismatch(self):
        client = FakeQdrantClient(exists=True, vector_size=3)

        with self.assertRaises(ValueError):
            self.build_store(client)

    def test_point_id_mapping_is_deterministic(self):
        store = self.build_store()

        first = store._to_point_id("record-1")
        second = store._to_point_id("record-1")
        other = store._to_point_id("record-2")

        self.assertEqual(first, second)
        self.assertNotEqual(first, other)

    def test_add_records_batches_and_preserves_payload(self):
        client = FakeQdrantClient()
        store = self.build_store(client, batch_size=2)
        records = [
            VectorRecord(
                id=f"record-{index}",
                content=f"content-{index}",
                embedding=[1.0, 0.0],
                metadata={"source": "sample.txt", "chunk_index": index},
            )
            for index in range(3)
        ]

        ids = store.add_records(records)

        self.assertEqual(ids, ["record-0", "record-1", "record-2"])
        self.assertEqual(len(client.upserts), 2)
        first_point = client.upserts[0]["points"][0]
        self.assertEqual(first_point.payload["content"], "content-0")
        self.assertEqual(first_point.payload["record_id"], "record-0")
        self.assertEqual(first_point.payload["source"], "sample.txt")

    def test_similarity_search_translates_filter_and_restores_results(self):
        client = FakeQdrantClient()
        client.search_hits = [
            SimpleNamespace(
                id="point-1",
                score=0.91,
                payload={"content": "alpha", "record_id": "record-1", "source": "a.txt"},
            )
        ]
        store = self.build_store(client)

        results = store.similarity_search(
            [1.0, 0.0],
            top_k=3,
            metadata_filter={"file_type": "txt", "acl": ["role:admin", "user:1"]},
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record.id, "record-1")
        self.assertEqual(results[0].content, "alpha")
        self.assertAlmostEqual(results[0].score, 0.91)

        query_filter = client.search_calls[0]["query_filter"]
        conditions = {condition.key: condition.match for condition in query_filter.must}
        self.assertEqual(conditions["file_type"].value, "txt")
        self.assertEqual(conditions["acl"].any, ["role:admin", "user:1"])

    def test_get_delete_clear_and_count(self):
        client = FakeQdrantClient()
        client.retrieve_points = [
            SimpleNamespace(
                id="point-1",
                vector=[1.0, 0.0],
                payload={"content": "alpha", "record_id": "record-1", "source": "a.txt"},
            )
        ]
        client.count_value = 1
        store = self.build_store(client)

        record = store.get_record("record-1")
        deleted = store.delete("record-1")
        count = store.count()
        store.clear()

        self.assertEqual(record.id, "record-1")
        self.assertEqual(record.embedding, [1.0, 0.0])
        self.assertTrue(deleted)
        self.assertEqual(count, 1)
        self.assertEqual(len(client.delete_calls), 1)
        self.assertEqual(client.deleted_collections, ["kb"])

    def test_list_records_scrolls_and_restores_records(self):
        client = FakeQdrantClient()
        client.scroll_pages = [
            (
                [
                    SimpleNamespace(
                        id="point-1",
                        vector=[1.0, 0.0],
                        payload={"content": "alpha", "record_id": "record-1", "source": "a.txt"},
                    )
                ],
                "next-page",
            ),
            (
                [
                    SimpleNamespace(
                        id="point-2",
                        vector=[0.0, 1.0],
                        payload={"content": "beta", "record_id": "record-2", "source": "b.txt"},
                    )
                ],
                None,
            ),
        ]
        store = self.build_store(client)

        records = store.list_records()

        self.assertEqual([record.id for record in records], ["record-1", "record-2"])
        self.assertEqual([call["offset"] for call in client.scroll_calls], [None, "next-page"])

    def test_delete_returns_false_when_record_is_missing(self):
        client = FakeQdrantClient()
        client.retrieve_points = []
        store = self.build_store(client)

        self.assertFalse(store.delete("missing"))
        self.assertEqual(client.delete_calls, [])

    def test_add_records_retries_rate_limit(self):
        client = FakeQdrantClient()
        client.fail_upsert_attempts = 1
        store = self.build_store(client)

        with patch("vector_store.qdrant_store.time.sleep") as sleep_mock:
            store.add_records([VectorRecord(id="record-1", content="alpha", embedding=[1.0, 0.0])])

        self.assertEqual(len(client.upserts), 1)
        self.assertTrue(sleep_mock.called)

    def test_add_records_raises_non_retryable_error(self):
        client = FakeQdrantClient()
        client.fail_upsert_attempts = 1
        client.fail_upsert_with = NonRetryableQdrantError("bad request")
        store = self.build_store(client)

        with self.assertRaises(QdrantVectorStoreError):
            store.add_records([VectorRecord(id="record-1", content="alpha", embedding=[1.0, 0.0])])

        self.assertEqual(len(client.upserts), 0)

    def test_factory_can_create_memory_store(self):
        store = create_vector_store(provider_name="memory", dimension=2)

        self.assertEqual(store.dimension, 2)

    def test_factory_can_create_qdrant_store_with_injected_client(self):
        client = FakeQdrantClient()

        store = create_vector_store(
            provider_name="qdrant",
            dimension=2,
            client=client,
            collection_name="kb",
        )

        self.assertIsInstance(store, QdrantVectorStore)
        self.assertEqual(len(client.created_collections), 1)


if __name__ == "__main__":
    unittest.main()
