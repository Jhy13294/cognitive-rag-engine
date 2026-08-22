import asyncio
import json
import math
import os
import time
import unittest
import uuid
from dataclasses import asdict

from cache import CachingEmbeddingProvider, CachingRAGPipeline
from cache.redis_store import CacheSettings, RedisCacheStore
from cache.serialization import decode_vector, encode_vector
from config import Config
from embeddings import EmbeddingConfig, EmbeddingProvider, HashEmbeddingProvider
from rag import RAGPipeline

Config.EMBEDDING_PROVIDER = "hash"

from service.app import ServiceState, create_app, stream_query_events
from service.models import QueryRequest
from vector_store import InMemoryVectorStore, VectorRecord


class FakeRedis:
    """Async in-memory Redis double."""

    def __init__(self, fail=False, delay=0.0, shared=None):
        self.fail = fail
        self.delay = delay
        self.data = shared if shared is not None else {}

    async def get(self, key):
        await self._maybe_wait()
        return self.data.get(key)

    async def set(self, key, value, ex=None):
        await self._maybe_wait()
        self.data[key] = value
        return True

    async def setnx(self, key, value):
        await self._maybe_wait()
        if key in self.data:
            return False
        self.data[key] = str(value).encode("utf-8")
        return True

    async def incr(self, key):
        await self._maybe_wait()
        raw_value = self.data.get(key, b"0")
        if isinstance(raw_value, bytes):
            raw_value = raw_value.decode("utf-8")
        value = int(raw_value) + 1
        self.data[key] = str(value).encode("utf-8")
        return value

    async def _maybe_wait(self):
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("planned redis failure")


class FixedEmbeddingProvider(EmbeddingProvider):
    """Embedding provider with exact floating-point values for serialization tests."""

    def __init__(self):
        super().__init__(EmbeddingConfig(model_name="fixed", dimension=4, normalize=False))
        self.embed_text_calls = 0

    def embed_text(self, text):
        self.embed_text_calls += 1
        return [0.1, math.pi, 1e-300, 1.2345678901234567]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


class CountingHashEmbeddingProvider(HashEmbeddingProvider):
    """Hash provider that counts query embedding calls."""

    def __init__(self, dimension=16):
        super().__init__(dimension=dimension)
        self.embed_text_calls = 0

    def embed_text(self, text):
        self.embed_text_calls += 1
        return super().embed_text(text)


class GateEmbeddingProvider(EmbeddingProvider):
    """Semantic test provider whose rejection query is orthogonal to the corpus."""

    def __init__(self):
        super().__init__(EmbeddingConfig(model_name="semantic-cache-test", dimension=2))
        self.embed_text_calls = 0

    def embed_text(self, text):
        self.embed_text_calls += 1
        return [0.0, 1.0] if "reject" in text else [1.0, 0.0]

    def embed_texts(self, texts):
        return [self.embed_text(text) for text in texts]


class ExplodingGateChatClient:
    """Make any generation call on a rejected retrieval fail the test."""

    def __init__(self):
        self.chat_calls = 0

    def chat(self, message, system_prompt=None):
        self.chat_calls += 1
        raise AssertionError("Retrieval rejection must not call chat generation")


class CountingVectorStore(InMemoryVectorStore):
    """In-memory vector store that counts similarity searches."""

    def __init__(self, dimension):
        super().__init__(dimension=dimension)
        self.search_calls = 0

    def similarity_search(self, query_embedding, top_k=5, metadata_filter=None):
        self.search_calls += 1
        return super().similarity_search(
            query_embedding, top_k=top_k, metadata_filter=metadata_filter
        )


class CountingChatClient:
    """Chat client that counts sync and streaming calls."""

    def __init__(self, answer="cached answer", delay=0.0):
        self.answer = answer
        self.delay = delay
        self.chat_calls = 0
        self.stream_calls = 0

    def chat(self, message, system_prompt=None):
        self.chat_calls += 1
        if self.delay:
            time.sleep(self.delay)
        return {"choices": [{"message": {"content": self.answer}}]}

    async def stream_chat(self, message, system_prompt=None):
        self.stream_calls += 1
        for token in ["live ", "answer"]:
            yield token


def build_cache_store(fake_redis=None, **overrides):
    settings = CacheSettings(
        enabled=True,
        namespace="test-cache",
        embedding_enabled=True,
        retrieval_enabled=True,
        answer_enabled=True,
        embedding_ttl=3600,
        retrieval_ttl=60,
        answer_ttl=60,
        timeout_seconds=1.0,
    )
    for key, value in overrides.items():
        setattr(settings, key, value)
    return RedisCacheStore(fake_redis or FakeRedis(), settings)


async def delete_redis_namespace(redis_url, namespace):
    """Delete keys created by live Redis tests."""
    import redis.asyncio as redis

    client = redis.from_url(redis_url, socket_connect_timeout=1.0, socket_timeout=1.0)
    try:
        keys = [key async for key in client.scan_iter(match=f"{namespace}:*")]
        if keys:
            await client.delete(*keys)
    finally:
        await client.aclose()


def build_cached_pipeline(cache_store=None):
    provider = CountingHashEmbeddingProvider(dimension=16)
    vector_store = CountingVectorStore(dimension=16)
    records = []
    for index, content in enumerate(["alpha policy", "beta handbook"], start=1):
        records.append(
            VectorRecord(
                id=f"record-{index}",
                content=content,
                embedding=provider.embed_text(content),
                metadata={
                    "id": f"record-{index}",
                    "source": f"doc-{index}.md",
                    "group": f"g{index}",
                    "acl": [f"role:{index}"],
                },
            )
        )
    provider.embed_text_calls = 0
    vector_store.add_records(records)
    chat_client = CountingChatClient()
    pipeline = RAGPipeline(
        embedding_provider=provider,
        vector_store=vector_store,
        chat_client=chat_client,
        top_k=1,
        max_context_chars=1000,
    )
    cached = CachingRAGPipeline(
        pipeline, cache_store or build_cache_store(), corpus_identity="unit-corpus"
    )
    return cached, provider, vector_store, chat_client


def build_gated_cached_pipeline(cache_store=None):
    provider = GateEmbeddingProvider()
    vector_store = CountingVectorStore(dimension=2)
    vector_store.add_records(
        [
            VectorRecord(
                id="gate-record",
                content="supported cache policy",
                embedding=[1.0, 0.0],
                metadata={"id": "gate-record", "source": "gate.md"},
            )
        ]
    )
    chat_client = ExplodingGateChatClient()
    pipeline = RAGPipeline(
        embedding_provider=provider,
        vector_store=vector_store,
        chat_client=chat_client,
        top_k=1,
        relevance_gate_enabled=True,
        relevance_gate_min_dense_cosine=0.5,
    )
    cached = CachingRAGPipeline(
        pipeline, cache_store or build_cache_store(), corpus_identity="gate-corpus"
    )
    return cached, provider, vector_store, chat_client


def parse_sse(raw_event):
    lines = [line for line in raw_event.strip().splitlines() if line]
    return lines[0].split(":", 1)[1].strip(), json.loads(lines[1].split(":", 1)[1].strip())


class CacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_l1_embedding_cache_preserves_float_values_exactly(self):
        provider = FixedEmbeddingProvider()
        cached_provider = CachingEmbeddingProvider(provider, build_cache_store())

        cold = cached_provider.embed_text("Precise Vector")
        hot = cached_provider.embed_text("  precise   vector ")

        self.assertEqual(cold, hot)
        self.assertEqual(provider.embed_text_calls, 1)
        self.assertEqual(cached_provider.cache_store.stats()["hits"]["l1"], 1)

    async def test_l2_retrieval_hit_short_circuits_similarity_search_and_returns_copies(self):
        pipeline, provider, vector_store, _chat = build_cached_pipeline()

        cold = pipeline.retrieve("alpha", top_k=1)
        hot = pipeline.retrieve("alpha", top_k=1)
        hot[0].metadata["mutated"] = True
        hot_again = pipeline.retrieve("alpha", top_k=1)

        self.assertEqual(asdict(cold[0]), asdict(hot_again[0]))
        self.assertEqual(provider.embed_text_calls, 1)
        self.assertEqual(vector_store.search_calls, 1)
        self.assertEqual(pipeline.cache_stats()["hits"]["l2"], 2)

    async def test_l2_caches_rejected_empty_results_with_gate_provenance(self):
        fake_redis = FakeRedis()
        store = build_cache_store(fake_redis, answer_enabled=False)
        pipeline, provider, vector_store, chat = build_gated_cached_pipeline(store)

        cold = pipeline.retrieve("reject unrelated", top_k=1)
        hot = pipeline.retrieve("reject unrelated", top_k=1)
        first_answer = pipeline.answer("reject unrelated", top_k=1)
        second_answer = pipeline.answer("reject unrelated", top_k=1)

        self.assertEqual(cold, [])
        self.assertEqual(hot, [])
        self.assertTrue(cold.relevance_gate_decision.rejected)
        self.assertTrue(hot.relevance_gate_decision.rejected)
        self.assertEqual(first_answer.answer, second_answer.answer)
        self.assertEqual(chat.chat_calls, 0)
        self.assertEqual(provider.embed_text_calls, 1)
        self.assertEqual(vector_store.search_calls, 1)
        cached_payloads = []
        for raw_value in fake_redis.data.values():
            try:
                payload = json.loads(raw_value)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and "relevance_gate" in payload:
                cached_payloads.append(payload)
        self.assertEqual(len(cached_payloads), 1)
        self.assertEqual(cached_payloads[0]["sources"], [])
        self.assertEqual(cached_payloads[0]["relevance_gate"]["action"], "rejected")

    async def test_l3_caches_deterministic_gate_abstention(self):
        pipeline, provider, vector_store, chat = build_gated_cached_pipeline()

        cold = pipeline.answer("reject unrelated", top_k=1)
        hot = pipeline.answer("reject unrelated", top_k=1)

        self.assertEqual(asdict(cold), asdict(hot))
        self.assertEqual(chat.chat_calls, 0)
        self.assertEqual(provider.embed_text_calls, 1)
        self.assertEqual(vector_store.search_calls, 1)
        self.assertEqual(pipeline.cache_stats()["hits"]["l3"], 1)

    async def test_l2_key_includes_top_k_and_metadata_filter(self):
        pipeline, _provider, vector_store, _chat = build_cached_pipeline()

        pipeline.retrieve("alpha", top_k=1)
        pipeline.retrieve("alpha", top_k=2)
        pipeline.retrieve("alpha", top_k=1, metadata_filter={"group": "g1"})

        self.assertEqual(vector_store.search_calls, 3)

    async def test_l2_l3_keys_include_allowed_acl_metadata_filter(self):
        pipeline, _provider, vector_store, chat = build_cached_pipeline()

        first = pipeline.answer("alpha", top_k=1, metadata_filter={"acl": ["role:1"]})
        second = pipeline.answer("alpha", top_k=1, metadata_filter={"acl": ["role:2"]})
        hot_first = pipeline.answer("alpha", top_k=1, metadata_filter={"acl": ["role:1"]})

        self.assertEqual(first.answer, hot_first.answer)
        self.assertNotEqual(first.sources[0].metadata["acl"], second.sources[0].metadata["acl"])
        self.assertEqual(vector_store.search_calls, 2)
        self.assertEqual(chat.chat_calls, 2)
        self.assertGreaterEqual(pipeline.cache_stats()["hits"]["l3"], 1)

    async def test_l3_answer_hit_short_circuits_chat_and_retrieval(self):
        pipeline, provider, vector_store, chat = build_cached_pipeline()

        cold = pipeline.answer("alpha", top_k=1)
        hot = pipeline.answer("alpha", top_k=1)

        self.assertEqual(asdict(cold), asdict(hot))
        self.assertEqual(chat.chat_calls, 1)
        self.assertEqual(vector_store.search_calls, 1)
        self.assertEqual(provider.embed_text_calls, 1)
        self.assertEqual(pipeline.cache_stats()["hits"]["l3"], 1)

    async def test_cache_fail_open_returns_cold_answer(self):
        failing_store = build_cache_store(FakeRedis(fail=True))
        pipeline, _provider, _vector_store, chat = build_cached_pipeline(failing_store)

        first = pipeline.answer("alpha", top_k=1)
        second = pipeline.answer("alpha", top_k=1)

        self.assertEqual(first.answer, "cached answer")
        self.assertEqual(second.answer, "cached answer")
        self.assertEqual(chat.chat_calls, 2)
        self.assertGreater(pipeline.cache_stats()["errors"], 0)

    async def test_corpus_version_persists_and_increments_across_store_instances(self):
        shared = {}
        first_store = build_cache_store(FakeRedis(shared=shared))
        second_store = build_cache_store(FakeRedis(shared=shared))

        self.assertEqual(await first_store.get_corpus_version(), 0)
        self.assertEqual(await first_store.bump_corpus_version(), 1)
        self.assertEqual(await second_store.get_corpus_version(), 1)

        state = ServiceState(cache_store=second_store)
        await state.invalidate()

        self.assertEqual(await first_store.get_corpus_version(), 2)

    async def test_stream_replays_l3_cached_answer_without_live_stream_call(self):
        pipeline, _provider, _vector_store, chat = build_cached_pipeline()
        cached_response = pipeline.answer("alpha", top_k=1)
        chat.stream_calls = 0
        app = create_app(ServiceState(cache_store=pipeline._cache_store))
        iterator = stream_query_events(
            app, QueryRequest(question="alpha", top_k=1), pipeline=pipeline
        )

        first_event = await iterator.__anext__()
        second_event = await iterator.__anext__()
        done_event = await iterator.__anext__()

        first_name, first_payload = parse_sse(first_event)
        second_name, second_payload = parse_sse(second_event)
        done_name, done_payload = parse_sse(done_event)

        self.assertEqual(first_name, "sources")
        self.assertTrue(first_payload["cached"])
        self.assertTrue(first_payload["stream_replay"])
        self.assertEqual(second_name, "token")
        self.assertTrue(second_payload["cached"])
        self.assertEqual(done_name, "done")
        self.assertEqual(done_payload["answer"], cached_response.answer)
        self.assertEqual(chat.stream_calls, 0)

    async def test_slow_cache_calls_do_not_serialize_threadpool_queries(self):
        store = build_cache_store(FakeRedis(delay=0.02))
        pipeline, _provider, _vector_store, _chat = build_cached_pipeline(store)

        started = time.perf_counter()
        await asyncio.gather(
            asyncio.to_thread(pipeline.answer, "alpha", 1),
            asyncio.to_thread(pipeline.answer, "beta", 1),
            asyncio.to_thread(pipeline.answer, "gamma", 1),
        )
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.75)

    @unittest.skipUnless(os.getenv("REDIS_URL"), "REDIS_URL is not set; live Redis smoke is gated")
    async def test_live_redis_smoke_incr_ttl_and_binary_payload(self):
        try:
            import redis.asyncio as redis
        except ModuleNotFoundError:
            self.skipTest("redis package is not installed in this environment")

        redis_url = os.environ["REDIS_URL"]
        namespace = f"test-cache-live-{uuid.uuid4().hex}"
        client = redis.from_url(redis_url, socket_connect_timeout=1.0, socket_timeout=1.0)
        settings = CacheSettings(
            enabled=True,
            redis_url=redis_url,
            namespace=namespace,
            embedding_enabled=True,
            retrieval_enabled=True,
            answer_enabled=True,
            embedding_ttl=2,
            retrieval_ttl=2,
            answer_ttl=2,
            timeout_seconds=2.0,
        )
        store = RedisCacheStore(client, settings)
        json_key = f"{namespace}:json"
        bytes_key = f"{namespace}:bytes"
        version_key = f"{namespace}:corpus:version"

        try:
            self.assertTrue(await client.ping())
            self.assertEqual(await store.get_corpus_version(), 0)
            self.assertEqual(await store.bump_corpus_version(), 1)

            await store.set_json("l2", json_key, {"value": 1}, settings.retrieval_ttl)
            self.assertEqual(await store.get_json("l2", json_key), {"value": 1})
            self.assertGreater(await client.ttl(json_key), 0)

            vector = [0.1, math.pi, 1e-300, 1.2345678901234567]
            await store.set_bytes("l1", bytes_key, encode_vector(vector), settings.embedding_ttl)
            self.assertEqual(decode_vector(await store.get_bytes("l1", bytes_key)), vector)
            self.assertGreater(await client.ttl(bytes_key), 0)
        finally:
            await client.delete(version_key, json_key, bytes_key)
            await client.aclose()

    @unittest.skipUnless(
        os.getenv("REDIS_URL"), "REDIS_URL is not set; live Redis cross-loop test is gated"
    )
    async def test_live_redis_cross_loop_sync_calls_hit_l1_l2_l3_without_errors(self):
        try:
            import redis  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("redis package is not installed in this environment")

        redis_url = os.environ["REDIS_URL"]
        namespace = f"test-cache-loop-{uuid.uuid4().hex}"
        settings = CacheSettings(
            enabled=True,
            redis_url=redis_url,
            namespace=namespace,
            embedding_enabled=True,
            retrieval_enabled=True,
            answer_enabled=True,
            embedding_ttl=30,
            retrieval_ttl=30,
            answer_ttl=30,
            timeout_seconds=2.0,
        )
        store = RedisCacheStore.from_url(settings)

        try:
            fixed_provider = FixedEmbeddingProvider()
            cached_provider = CachingEmbeddingProvider(fixed_provider, store)

            first_vector = cached_provider.embed_text("same question")
            second_vector = await asyncio.to_thread(cached_provider.embed_text, "same question")

            self.assertEqual(first_vector, second_vector)
            self.assertEqual(fixed_provider.embed_text_calls, 1)
            stats_after_l1 = store.stats()
            self.assertEqual(stats_after_l1["hits"]["l1"], 1)
            self.assertEqual(stats_after_l1["errors"], 0)
            self.assertTrue(stats_after_l1["background_loop_enabled"])
            self.assertTrue(stats_after_l1["background_loop_alive"])
            self.assertIsNotNone(stats_after_l1["background_loop_thread_id"])

            pipeline, provider, vector_store, chat = build_cached_pipeline(store)

            first_sources = await asyncio.to_thread(pipeline.retrieve, "alpha", 1)
            second_sources = await asyncio.to_thread(pipeline.retrieve, "alpha", 1)

            self.assertEqual(asdict(first_sources[0]), asdict(second_sources[0]))
            self.assertEqual(provider.embed_text_calls, 1)
            self.assertEqual(vector_store.search_calls, 1)

            first_answer = await asyncio.to_thread(pipeline.answer, "beta", 1)
            second_answer = await asyncio.to_thread(pipeline.answer, "beta", 1)

            self.assertEqual(asdict(first_answer), asdict(second_answer))
            self.assertEqual(chat.chat_calls, 1)

            await store.bump_corpus_version()
            invalidated_answer = await asyncio.to_thread(pipeline.answer, "beta", 1)

            self.assertEqual(invalidated_answer.answer, first_answer.answer)
            self.assertEqual(chat.chat_calls, 2)

            gated_pipeline, gated_provider, gated_store, gated_chat = build_gated_cached_pipeline(
                store
            )
            first_empty = await asyncio.to_thread(gated_pipeline.retrieve, "reject unrelated", 1)
            second_empty = await asyncio.to_thread(gated_pipeline.retrieve, "reject unrelated", 1)
            first_abstention = await asyncio.to_thread(gated_pipeline.answer, "reject unrelated", 1)
            second_abstention = await asyncio.to_thread(
                gated_pipeline.answer, "reject unrelated", 1
            )

            self.assertEqual(first_empty, [])
            self.assertEqual(second_empty, [])
            self.assertTrue(first_empty.relevance_gate_decision.rejected)
            self.assertTrue(second_empty.relevance_gate_decision.rejected)
            self.assertEqual(asdict(first_abstention), asdict(second_abstention))
            self.assertEqual(gated_provider.embed_text_calls, 1)
            self.assertEqual(gated_store.search_calls, 1)
            self.assertEqual(gated_chat.chat_calls, 0)

            final_stats = store.stats()
            self.assertGreaterEqual(final_stats["hits"]["l2"], 1)
            self.assertGreaterEqual(final_stats["hits"]["l3"], 1)
            self.assertEqual(final_stats["errors"], 0)
            self.assertEqual(
                final_stats["background_loop_thread_id"],
                stats_after_l1["background_loop_thread_id"],
            )
        finally:
            await store.aclose()
            await delete_redis_namespace(redis_url, namespace)


if __name__ == "__main__":
    unittest.main()
