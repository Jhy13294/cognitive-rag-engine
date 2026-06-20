import asyncio
import json
import tempfile
import time
import unittest
from pathlib import Path

import httpx

from access import StaticACLResolver
from config import Config
from logger import LOG_FORMAT, get_request_id, reset_request_id, set_request_id, setup_logger
from observability import JSONLineAuditSink, MetricsRegistry, ObservabilityManager, StructuredAuditEmitter
from observability.factory import create_observability_manager
from rag import RAGResponse, RetrievedSource
from service.app import ServiceState, create_app, stream_query_events
from service.models import QueryRequest


class MemoryAuditSink:
    """Collect audit records without filesystem I/O."""

    def __init__(self):
        """Initialize an empty record list."""
        self.records = []

    def emit(self, record):
        """Store a defensive copy of one record."""
        self.records.append(json.loads(json.dumps(record)))


class ThrowingAuditSink:
    """Raise on every audit write for fail-open tests."""

    def emit(self, record):
        """Raise a deterministic sink error."""
        raise RuntimeError("audit sink unavailable")


class ThrowingMetricsRegistry:
    """Raise on every metric operation for fail-open tests."""

    def __getattr__(self, name):
        """Return a callable that raises for any registry operation."""
        def fail(*args, **kwargs):
            raise RuntimeError(f"metrics unavailable: {name}")

        return fail


class FixedTokenCounter:
    """Return deterministic token estimates for tests."""

    encoding_name = "test"

    def count(self, text):
        """Return one token per whitespace-delimited word."""
        return max(1, len(str(text).split()))


class FakeCacheStore:
    """Expose true-shaped cache counters to the metrics adapter."""

    def __init__(self):
        """Initialize fixed L1/L2/L3 counters."""
        self.payload = {
            "hits": {"l1": 2, "l2": 3, "l3": 4},
            "misses": {"l1": 5, "l2": 6, "l3": 7},
            "errors": 0,
        }

    def stats(self):
        """Return copied cache counters."""
        return json.loads(json.dumps(self.payload))


class FakeChatClient:
    """Provide synchronous and streaming chat behavior."""

    def __init__(self, token_delay=0.0):
        """Initialize optional streaming token delay."""
        self.token_delay = token_delay

    def chat(self, message, system_prompt=None):
        """Return a provider-shaped response with reported token usage."""
        return {
            "choices": [{"message": {"content": "safe answer"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
        }

    async def stream_chat(self, message, system_prompt=None):
        """Yield two delayed token chunks."""
        for token in ("safe", " answer"):
            if self.token_delay:
                await asyncio.sleep(self.token_delay)
            yield token


class FakePipeline:
    """Provide an existing-pipeline-shaped service test double."""

    system_prompt = "system"

    def __init__(self, delay=0.0, chat_client=None, sources=None):
        """Initialize deterministic answer and retrieval behavior."""
        self.delay = delay
        self.chat_client = chat_client or FakeChatClient()
        self.sources = sources if sources is not None else [make_source()]

    def answer(self, question, top_k=None, metadata_filter=None):
        """Return a complete response after an optional blocking delay."""
        if self.delay:
            time.sleep(self.delay)
        raw_response = self.chat_client.chat("prompt", system_prompt=self.system_prompt)
        return RAGResponse(
            question=question,
            answer=raw_response["choices"][0]["message"]["content"],
            sources=list(self.sources),
            prompt="prompt with private context",
            raw_response=raw_response,
        )

    def retrieve(self, question, top_k=None, metadata_filter=None):
        """Return deterministic sources."""
        return list(self.sources)

    def _build_prompt_and_sources(self, question, sources):
        """Return a prompt and unchanged source list."""
        return "prompt with private context", list(sources)


def make_source():
    """Build a source containing fields that must not leak to audit logs."""
    return RetrievedSource(
        index=1,
        content="classified document body",
        score=0.91,
        metadata={"id": "record-1", "source": "secret/path.md", "acl": ["role:secret"]},
    )


class AuditAndMetricsUnitTests(unittest.TestCase):
    def test_logger_request_context_adds_id_without_call_site_changes(self):
        request_logger = setup_logger("tests.request_context")
        self.assertTrue(all(handler.formatter._fmt == LOG_FORMAT for handler in request_logger.handlers))
        token = set_request_id("request-log")
        try:
            with self.assertLogs("tests.request_context", level="INFO") as captured:
                request_logger.info("query completed | outcome=success")
        finally:
            reset_request_id(token)

        output = "\n".join(captured.output)
        self.assertIn("request_id=request-log", output)
        self.assertIn("query completed | outcome=success", output)

    def test_audit_record_is_traceable_without_sensitive_values(self):
        sink = MemoryAuditSink()
        emitter = StructuredAuditEmitter(sink=sink, hash_salt="unit-test-salt")

        emitted = emitter.emit_query(
            request_id="request-1",
            route="query",
            principal="alice@example.com",
            question="show the private roadmap with sk-secret",
            top_k=5,
            retrieval_mode="dense",
            sources=[make_source()],
            latency_ms=12.5,
            cache_outcome="miss",
            status_code=200,
            outcome="success",
        )

        self.assertTrue(emitted)
        self.assertEqual(len(sink.records), 1)
        record = sink.records[0]
        serialized = json.dumps(record, ensure_ascii=False)
        self.assertEqual(record["request_id"], "request-1")
        self.assertTrue(record["principal_id"].startswith("p_"))
        self.assertTrue(record["query_hash"].startswith("q_"))
        self.assertEqual(record["source_ids"], ["record-1"])
        self.assertNotIn("alice@example.com", serialized)
        self.assertNotIn("show the private roadmap", serialized)
        self.assertNotIn("sk-secret", serialized)
        self.assertNotIn("role:secret", serialized)
        self.assertNotIn("classified document body", serialized)
        self.assertNotIn("secret/path.md", serialized)
        self.assertNotIn("query_text", record)

    def test_audit_sink_failure_is_contained(self):
        emitter = StructuredAuditEmitter(sink=ThrowingAuditSink(), hash_salt="unit-test-salt")

        with self.assertLogs("observability.audit", level="WARNING") as captured:
            emitted = emitter.emit_query(
                request_id="request-2",
                route="query",
                principal="alice",
                question="question",
                top_k=5,
                retrieval_mode="dense",
                sources=[],
                latency_ms=1.0,
                cache_outcome="disabled",
                status_code=403,
                outcome="acl_denied",
            )

        self.assertFalse(emitted)
        self.assertIn("alert_code=audit_emit_failed", "\n".join(captured.output))

    def test_path_like_source_identifier_is_hashed(self):
        sink = MemoryAuditSink()
        emitter = StructuredAuditEmitter(sink=sink, hash_salt="unit-test-salt")
        source = make_source()
        source.metadata["id"] = "tenant/private/source.md"

        emitter.emit_query(
            request_id="request-path-id",
            route="query",
            principal=None,
            question="question",
            top_k=5,
            retrieval_mode="dense",
            sources=[source],
            latency_ms=1.0,
            cache_outcome="disabled",
            status_code=200,
            outcome="success",
        )

        self.assertTrue(sink.records[0]["source_ids"][0].startswith("s_"))
        self.assertNotIn("tenant/private/source.md", json.dumps(sink.records[0]))

    def test_query_text_requires_opt_in_and_is_length_bounded(self):
        sink = MemoryAuditSink()
        emitter = StructuredAuditEmitter(
            sink=sink,
            hash_salt="unit-test-salt",
            log_query_text=True,
            query_text_max_length=8,
        )

        emitter.emit_query(
            request_id="request-query-text",
            route="query",
            principal=None,
            question="sensitive query text",
            top_k=5,
            retrieval_mode="dense",
            sources=[],
            latency_ms=1.0,
            cache_outcome="disabled",
            status_code=200,
            outcome="empty",
        )

        self.assertEqual(sink.records[0]["query_text"], "sensitiv")
        self.assertTrue(sink.records[0]["query_hash"].startswith("q_"))

    def test_json_line_sink_writes_one_record_per_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            sink = JSONLineAuditSink(str(path), max_bytes=4096, backup_count=1, queue_size=10)
            emitter = StructuredAuditEmitter(sink=sink, hash_salt="unit-test-salt")
            emitter.emit_query(
                request_id="request-jsonl",
                route="query",
                principal="alice",
                question="question",
                top_k=5,
                retrieval_mode="dense",
                sources=[make_source()],
                latency_ms=1.0,
                cache_outcome="disabled",
                status_code=200,
                outcome="success",
            )
            emitter.close()

            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["request_id"], "request-jsonl")

    def test_background_audit_write_failure_emits_stable_warning(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.jsonl"
            sink = JSONLineAuditSink(str(path), max_bytes=4096, backup_count=1, queue_size=10)

            def fail_write(record):
                """Raise a deterministic filesystem error."""
                raise OSError("disk unavailable")

            sink._handler.emit = fail_write
            emitter = StructuredAuditEmitter(sink=sink, hash_salt="unit-test-salt")
            with self.assertLogs("observability.audit", level="WARNING") as captured:
                emitter.emit_query(
                    request_id="request-write-failure",
                    route="query",
                    principal="alice",
                    question="question",
                    top_k=5,
                    retrieval_mode="dense",
                    sources=[],
                    latency_ms=1.0,
                    cache_outcome="disabled",
                    status_code=403,
                    outcome="acl_denied",
                )
                emitter.close()

            self.assertIn("alert_code=audit_sink_write_failed", "\n".join(captured.output))

    def test_metrics_expose_five_bounded_families_and_true_cache_counters(self):
        registry = MetricsRegistry(namespace="rag")
        registry.observe_request("query", "success", 0.125)
        registry.add_token_usage("chat", "reported", 9)
        registry.add_token_usage("embedding", "estimated", 3)
        registry.observe_scores("query", "dense", [0.91, 0.45])
        registry.increment_empty_retrieval("query_stream", "hybrid")

        payload = registry.render(cache_stats=FakeCacheStore().stats())

        for metric_name in (
            "rag_request_duration_seconds",
            "rag_token_usage_total",
            "rag_retrieval_score",
            "rag_empty_retrieval_total",
            "rag_cache_operations_total",
        ):
            self.assertIn(metric_name, payload)
        self.assertIn('rag_retrieval_score_bucket{route="query",retrieval_mode="dense",le="0.5"} 1', payload)
        self.assertIn('rag_cache_operations_total{cache_layer="L3",outcome="hit"} 4', payload)
        labels = {label for names in registry.label_names().values() for label in names}
        self.assertTrue(labels.isdisjoint({"principal", "request_id", "query", "source_id"}))
        with self.assertRaises(ValueError):
            registry.observe_request("/query/alice", "success", 0.1)

    def test_manager_marks_reported_and_estimated_tokens_honestly(self):
        registry = MetricsRegistry(namespace="rag")
        manager = ObservabilityManager(
            metrics_registry=registry,
            token_counter=FixedTokenCounter(),
        )
        observation = manager.start_query(
            request_id="request-reported",
            route="query",
            principal=None,
            question="one two",
            top_k=5,
            retrieval_mode="dense",
        )
        manager.complete_query(
            observation,
            sources=[make_source()],
            raw_response={"usage": {"total_tokens": 9}},
            prompt="one two three",
            answer="four",
        )
        estimated = manager.start_query(
            request_id="request-estimated",
            route="query",
            principal=None,
            question="one two",
            top_k=5,
            retrieval_mode="dense",
        )
        manager.complete_query(
            estimated,
            sources=[make_source()],
            raw_response={},
            prompt="one two three",
            answer="four",
        )

        payload = registry.render()
        self.assertIn('rag_token_usage_total{kind="chat",source="reported"} 9', payload)
        self.assertIn('rag_token_usage_total{kind="chat",source="estimated"} 4', payload)
        self.assertIn('rag_token_usage_total{kind="embedding",source="estimated"} 4', payload)

    def test_manager_reads_reported_embedding_token_delta_from_provider(self):
        class UsageEmbeddingProvider:
            """Expose the cumulative counter used by OpenAIEmbeddingProvider."""

            request_count = 1
            total_prompt_tokens = 10

        class UsagePipeline:
            """Expose an embedding provider without changing retrieval APIs."""

            embedding_provider = UsageEmbeddingProvider()

        registry = MetricsRegistry(namespace="rag")
        manager = ObservabilityManager(metrics_registry=registry, token_counter=FixedTokenCounter())
        pipeline = UsagePipeline()
        observation = manager.start_query(
            request_id="request-embedding-usage",
            route="query",
            principal=None,
            question="one two",
            top_k=5,
            retrieval_mode="dense",
        )
        manager.bind_pipeline(observation, pipeline)
        pipeline.embedding_provider.request_count = 2
        pipeline.embedding_provider.total_prompt_tokens = 16
        manager.complete_query(
            observation,
            sources=[make_source()],
            raw_response={"usage": {"total_tokens": 9}},
            prompt="prompt",
            answer="answer",
        )

        payload = registry.render()
        self.assertIn('rag_token_usage_total{kind="embedding",source="reported"} 6', payload)

    def test_provider_watermark_does_not_double_count_overlapping_requests(self):
        class UsageEmbeddingProvider:
            """Expose mutable cumulative usage for overlap simulation."""

            request_count = 0
            total_prompt_tokens = 0

        class UsagePipeline:
            """Share one provider across overlapping observations."""

            embedding_provider = UsageEmbeddingProvider()

        registry = MetricsRegistry(namespace="rag")
        manager = ObservabilityManager(metrics_registry=registry, token_counter=FixedTokenCounter())
        pipeline = UsagePipeline()
        observations = [
            manager.start_query(
                request_id=f"overlap-{index}",
                route="query",
                principal=None,
                question="one",
                top_k=5,
                retrieval_mode="dense",
            )
            for index in range(2)
        ]
        for observation in observations:
            manager.bind_pipeline(observation, pipeline)

        pipeline.embedding_provider.request_count = 2
        pipeline.embedding_provider.total_prompt_tokens = 11
        for observation in reversed(observations):
            manager.complete_query(
                observation,
                sources=[make_source()],
                raw_response={"usage": {"total_tokens": 1}},
                prompt="prompt",
                answer="answer",
            )

        payload = registry.render()
        self.assertIn('rag_token_usage_total{kind="embedding",source="reported"} 11', payload)

    def test_cached_raw_usage_is_not_counted_as_a_new_chat_request(self):
        class UsageChatClient:
            """Expose cumulative chat usage while cached raw responses are replayed."""

            request_count = 0
            total_tokens = 0

            def usage_stats(self):
                """Return current cumulative usage."""
                return {"request_count": self.request_count, "total_tokens": self.total_tokens}

        class UsagePipeline:
            """Expose one shared chat client."""

            chat_client = UsageChatClient()

        registry = MetricsRegistry(namespace="rag")
        manager = ObservabilityManager(metrics_registry=registry, token_counter=FixedTokenCounter())
        pipeline = UsagePipeline()
        cold = manager.start_query(
            request_id="chat-cold",
            route="query",
            principal=None,
            question="question",
            top_k=5,
            retrieval_mode="dense",
        )
        manager.bind_pipeline(cold, pipeline)
        pipeline.chat_client.request_count = 1
        pipeline.chat_client.total_tokens = 10
        manager.complete_query(
            cold,
            sources=[make_source()],
            raw_response={"usage": {"total_tokens": 10}},
            prompt="prompt",
            answer="answer",
        )

        replay = manager.start_query(
            request_id="chat-replay",
            route="query",
            principal=None,
            question="question",
            top_k=5,
            retrieval_mode="dense",
        )
        manager.bind_pipeline(replay, pipeline)
        manager.complete_query(
            replay,
            sources=[make_source()],
            raw_response={"usage": {"total_tokens": 10}},
            prompt="prompt",
            answer="answer",
        )

        payload = registry.render()
        self.assertIn('rag_token_usage_total{kind="chat",source="reported"} 10', payload)
        self.assertNotIn('rag_token_usage_total{kind="chat",source="reported"} 20', payload)

    def test_observability_configuration_rejects_unsafe_or_incomplete_values(self):
        names = (
            "OBSERVABILITY_ENABLED",
            "AUDIT_ENABLED",
            "METRICS_ENABLED",
            "AUDIT_HASH_SALT",
            "AUDIT_PRINCIPAL_MODE",
            "METRICS_NAMESPACE",
        )
        original = {name: getattr(Config, name) for name in names}
        try:
            Config.OBSERVABILITY_ENABLED = True
            Config.AUDIT_ENABLED = True
            Config.METRICS_ENABLED = False
            Config.AUDIT_HASH_SALT = None
            with self.assertRaisesRegex(ValueError, "AUDIT_HASH_SALT"):
                Config.validate_observability()

            Config.AUDIT_HASH_SALT = "test-salt"
            Config.AUDIT_PRINCIPAL_MODE = "unknown"
            with self.assertRaisesRegex(ValueError, "AUDIT_PRINCIPAL_MODE"):
                Config.validate_observability()

            Config.AUDIT_PRINCIPAL_MODE = "hash"
            Config.AUDIT_ENABLED = False
            Config.METRICS_ENABLED = True
            Config.METRICS_NAMESPACE = "1invalid"
            with self.assertRaisesRegex(ValueError, "METRICS_NAMESPACE"):
                Config.validate_observability()
        finally:
            for name, value in original.items():
                setattr(Config, name, value)


class ObservabilityServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        """Enable safe test observability settings and preserve global config."""
        self.config_names = (
            "OBSERVABILITY_ENABLED",
            "AUDIT_ENABLED",
            "METRICS_ENABLED",
            "AUDIT_HASH_SALT",
            "AUDIT_PRINCIPAL_MODE",
            "AUDIT_LOG_QUERY_TEXT",
            "ACL_ENABLED",
            "ACL_ALLOW_BODY_PRINCIPAL",
        )
        self.original_config = {name: getattr(Config, name) for name in self.config_names}
        Config.OBSERVABILITY_ENABLED = True
        Config.AUDIT_ENABLED = True
        Config.METRICS_ENABLED = True
        Config.AUDIT_HASH_SALT = "service-test-salt"
        Config.AUDIT_PRINCIPAL_MODE = "hash"
        Config.AUDIT_LOG_QUERY_TEXT = False
        Config.ACL_ENABLED = False
        Config.ACL_ALLOW_BODY_PRINCIPAL = False

    def tearDown(self):
        """Restore shared Config values after each service test."""
        for name, value in self.original_config.items():
            setattr(Config, name, value)

    async def open_client(self, app):
        """Return an in-process ASGI client."""
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://testserver")

    async def test_request_context_propagates_through_threadpool_offload(self):
        token = set_request_id("request-thread")
        try:
            observed = await asyncio.to_thread(get_request_id)
        finally:
            reset_request_id(token)

        self.assertEqual(observed, "request-thread")

    def build_manager(self, sink=None, metrics=None, cache_store=None):
        """Create an injected manager without filesystem or network I/O."""
        return create_observability_manager(
            Config,
            cache_store=cache_store,
            audit_sink=sink or MemoryAuditSink(),
            metrics_registry=metrics or MetricsRegistry("rag"),
        )

    async def test_query_emits_one_sanitized_audit_record_and_five_metrics(self):
        sink = MemoryAuditSink()
        cache_store = FakeCacheStore()
        manager = self.build_manager(sink=sink, cache_store=cache_store)
        state = ServiceState(
            pipeline_builder=lambda **kwargs: FakePipeline(),
            chat_client_factory=lambda: FakeChatClient(),
            cache_store=cache_store,
            observability=manager,
        )
        app = create_app(state)

        async with await self.open_client(app) as client:
            response = await client.post(
                "/query",
                headers={"X-Principal": "alice@example.com", "X-Request-ID": "attacker-controlled"},
                json={"question": "show the private roadmap"},
            )
            metrics = await client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertRegex(response.headers["X-Request-ID"], r"^[0-9a-f]{32}$")
        self.assertNotEqual(response.headers["X-Request-ID"], "attacker-controlled")
        self.assertEqual(len(sink.records), 1)
        self.assertEqual(sink.records[0]["request_id"], response.headers["X-Request-ID"])
        self.assertNotIn("alice@example.com", json.dumps(sink.records[0]))
        self.assertEqual(metrics.status_code, 200)
        self.assertIn("rag_request_duration_seconds", metrics.text)
        self.assertIn('rag_cache_operations_total{cache_layer="L1",outcome="hit"} 2', metrics.text)

    async def test_acl_denial_emits_one_opaque_audit_record(self):
        Config.ACL_ENABLED = True
        sink = MemoryAuditSink()
        manager = self.build_manager(sink=sink)
        state = ServiceState(
            pipeline_builder=lambda **kwargs: FakePipeline(),
            chat_client_factory=lambda: FakeChatClient(),
            acl_resolver=StaticACLResolver({"bob": ["role:finance"]}),
            observability=manager,
        )
        app = create_app(state)

        async with await self.open_client(app) as client:
            response = await client.post(
                "/query",
                headers={"X-Principal": "alice"},
                json={"question": "restricted question"},
            )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(len(sink.records), 1)
        record = sink.records[0]
        self.assertEqual(record["outcome"], "acl_denied")
        self.assertEqual(record["n_sources"], 0)
        self.assertTrue(record["principal_id"].startswith("p_"))
        self.assertNotIn("alice", json.dumps(record))

    async def test_empty_retrieval_emits_one_audit_record_and_counter(self):
        sink = MemoryAuditSink()
        manager = self.build_manager(sink=sink)
        state = ServiceState(
            pipeline_builder=lambda **kwargs: FakePipeline(sources=[]),
            chat_client_factory=lambda: FakeChatClient(),
            cache_store=FakeCacheStore(),
            observability=manager,
        )
        app = create_app(state)

        async with await self.open_client(app) as client:
            response = await client.post("/query", json={"question": "no match"})
            metrics = await client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(sink.records), 1)
        self.assertEqual(sink.records[0]["outcome"], "empty")
        self.assertEqual(sink.records[0]["n_sources"], 0)
        self.assertIn(
            'rag_empty_retrieval_total{route="query",retrieval_mode="dense"} 1',
            metrics.text,
        )

    async def test_server_error_emits_one_sanitized_audit_record(self):
        class ErrorPipeline(FakePipeline):
            """Raise a private server error from the existing answer seam."""

            def answer(self, question, top_k=None, metadata_filter=None):
                """Raise an error containing text that must not reach audit."""
                raise RuntimeError("private document body sk-secret")

        sink = MemoryAuditSink()
        manager = self.build_manager(sink=sink)
        state = ServiceState(
            pipeline_builder=lambda **kwargs: ErrorPipeline(),
            chat_client_factory=lambda: FakeChatClient(),
            cache_store=FakeCacheStore(),
            observability=manager,
        )
        app = create_app(state)

        async with await self.open_client(app) as client:
            response = await client.post("/query", json={"question": "failing query"})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(len(sink.records), 1)
        record = sink.records[0]
        self.assertEqual(record["outcome"], "server_error")
        serialized = json.dumps(record)
        self.assertNotIn("private document body", serialized)
        self.assertNotIn("sk-secret", serialized)

    async def test_throwing_audit_metrics_and_alert_hooks_do_not_break_query(self):
        manager = ObservabilityManager(
            audit_emitter=StructuredAuditEmitter(ThrowingAuditSink(), hash_salt="test-salt"),
            metrics_registry=ThrowingMetricsRegistry(),
            token_counter=FixedTokenCounter(),
            alert_hooks=[lambda code, payload: (_ for _ in ()).throw(RuntimeError("alert failed"))],
        )
        state = ServiceState(
            pipeline_builder=lambda **kwargs: FakePipeline(sources=[]),
            chat_client_factory=lambda: FakeChatClient(),
            cache_store=FakeCacheStore(),
            observability=manager,
        )
        app = create_app(state)

        async with await self.open_client(app) as client:
            response = await client.post("/query", json={"question": "still answer"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "safe answer")

    async def test_observability_keeps_query_concurrency_below_gate(self):
        sink = MemoryAuditSink()
        manager = self.build_manager(sink=sink)
        state = ServiceState(
            pipeline_builder=lambda **kwargs: FakePipeline(delay=0.25),
            chat_client_factory=lambda: FakeChatClient(),
            cache_store=FakeCacheStore(),
            observability=manager,
        )
        app = create_app(state)

        async with await self.open_client(app) as client:
            started = time.perf_counter()
            responses = await asyncio.gather(
                client.post("/query", json={"question": "q1"}),
                client.post("/query", json={"question": "q2"}),
                client.post("/query", json={"question": "q3"}),
            )
            elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.55)
        self.assertTrue(all(response.status_code == 200 for response in responses))
        self.assertEqual(len(sink.records), 3)

    async def test_stream_sources_stay_fast_and_finalize_one_audit_record(self):
        sink = MemoryAuditSink()
        manager = self.build_manager(sink=sink)
        chat_client = FakeChatClient(token_delay=0.05)
        pipeline = FakePipeline(chat_client=chat_client)
        state = ServiceState(
            pipeline_builder=lambda **kwargs: pipeline,
            chat_client_factory=lambda: chat_client,
            cache_store=FakeCacheStore(),
            observability=manager,
        )
        app = create_app(state)
        request = QueryRequest(question="stream safely")
        observation = manager.start_query(
            request_id="stream-request",
            route="query_stream",
            principal=None,
            question=request.question,
            top_k=5,
            retrieval_mode="dense",
        )
        manager.bind_pipeline(observation, pipeline)
        iterator = stream_query_events(app, request, pipeline=pipeline, observation=observation)

        started = time.perf_counter()
        first_event = await iterator.__anext__()
        first_elapsed = time.perf_counter() - started
        remaining = [event async for event in iterator]

        self.assertLess(first_elapsed, 0.04)
        self.assertTrue(first_event.startswith("event: sources"))
        self.assertTrue(remaining[-1].startswith("event: done"))
        self.assertEqual(len(sink.records), 1)
        self.assertEqual(sink.records[0]["request_id"], "stream-request")


if __name__ == "__main__":
    unittest.main()
