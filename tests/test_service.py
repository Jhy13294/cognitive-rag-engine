import asyncio
import json
import time
import unittest
from dataclasses import dataclass

import httpx

from api_client import APIError
from access import ACLAccessError, StaticACLResolver
from config import Config
from rag import EmbeddingSpaceMismatchError, IndexNotReadyError
from rag import RAGResponse, RetrievedSource
from service.app import ServiceState, create_app, resolve_request_metadata_filter, stream_query_events
from service.models import QueryRequest


def make_source(index=1, content="source text"):
    return RetrievedSource(
        index=index,
        content=content,
        score=0.9,
        metadata={"source": "fixture.txt", "chunk_index": index - 1},
    )


class FakeChatClient:
    """Chat client double with both sync and streaming methods."""

    def __init__(self, answer="complete answer", stream_tokens=None, token_delay=0.0):
        self.answer = answer
        self.stream_tokens = stream_tokens or ["complete", " answer"]
        self.token_delay = token_delay

    def chat(self, message, system_prompt=None):
        return {"choices": [{"message": {"content": self.answer}}]}

    async def stream_chat(self, message, system_prompt=None):
        for token in self.stream_tokens:
            if self.token_delay:
                await asyncio.sleep(self.token_delay)
            yield token


class FakePipeline:
    """RAGPipeline-compatible test double."""

    system_prompt = "system"

    def __init__(self, answer="complete answer", answer_delay=0.0, error=None, chat_client=None):
        self.answer_text = answer
        self.answer_delay = answer_delay
        self.error = error
        self.chat_client = chat_client or FakeChatClient(answer=answer)
        self.answer_filters = []
        self.retrieve_filters = []

    def answer(self, question, top_k=None, metadata_filter=None):
        self.answer_filters.append(metadata_filter)
        if self.answer_delay:
            time.sleep(self.answer_delay)
        if self.error:
            raise self.error
        source = make_source()
        return RAGResponse(
            question=question,
            answer=self.answer_text,
            sources=[source],
            prompt="prompt",
            raw_response={"choices": [{"message": {"content": self.answer_text}}]},
        )

    def retrieve(self, question, top_k=None, metadata_filter=None):
        self.retrieve_filters.append(metadata_filter)
        if self.error:
            raise self.error
        return [make_source()]

    def _build_prompt_and_sources(self, question, sources):
        return f"prompt for {question}", sources


@dataclass
class FakeIngestResult:
    ids: list
    embedding_model: str = "hash-embedding-local"
    embedding_dimension: int = 8
    parent_count: int = 0


def parse_sse(raw_event):
    lines = [line for line in raw_event.strip().splitlines() if line]
    event_name = lines[0].split(":", 1)[1].strip()
    payload = json.loads(lines[1].split(":", 1)[1].strip())
    return event_name, payload


class FastAPIServiceTests(unittest.IsolatedAsyncioTestCase):
    async def open_client(self, app):
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url="http://testserver")

    async def test_create_app_validates_cache_configuration(self):
        original_ttl = Config.CACHE_EMBEDDING_TTL
        try:
            Config.CACHE_EMBEDDING_TTL = 0
            with self.assertRaisesRegex(ValueError, "CACHE_EMBEDDING_TTL"):
                create_app(ServiceState(cache_store=object()))
        finally:
            Config.CACHE_EMBEDDING_TTL = original_ttl

    async def test_query_offloads_sync_answer_work_for_concurrent_requests(self):
        state = ServiceState(
            pipeline_builder=lambda **kwargs: FakePipeline(answer="ok", answer_delay=0.25),
            chat_client_factory=lambda: FakeChatClient(answer="ok"),
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
        self.assertGreaterEqual(elapsed, 0.20)
        self.assertTrue(all(response.status_code == 200 for response in responses))

    async def test_stream_query_sends_sources_before_tokens_and_matches_complete_answer(self):
        chat_client = FakeChatClient(stream_tokens=["Hel", "lo"], token_delay=0.05)
        state = ServiceState(
            pipeline_builder=lambda **kwargs: FakePipeline(chat_client=chat_client),
            chat_client_factory=lambda: chat_client,
        )
        app = create_app(state)
        iterator = stream_query_events(app, QueryRequest(question="stream me"))

        started = time.perf_counter()
        first_event = await iterator.__anext__()
        first_elapsed = time.perf_counter() - started
        second_event = await iterator.__anext__()
        first_token_elapsed = time.perf_counter() - started
        third_event = await iterator.__anext__()
        done_event = await iterator.__anext__()
        done_elapsed = time.perf_counter() - started

        first_name, first_payload = parse_sse(first_event)
        second_name, second_payload = parse_sse(second_event)
        third_name, third_payload = parse_sse(third_event)
        done_name, done_payload = parse_sse(done_event)

        self.assertEqual(first_name, "sources")
        self.assertEqual(first_payload["sources"][0]["index"], 1)
        self.assertLess(first_elapsed, 0.04)
        self.assertEqual(second_name, "token")
        self.assertEqual(third_name, "token")
        self.assertEqual(done_name, "done")
        self.assertGreaterEqual(first_token_elapsed, 0.04)
        self.assertGreater(done_elapsed, first_token_elapsed)
        self.assertEqual(second_payload["delta"] + third_payload["delta"], done_payload["answer"])
        self.assertEqual(done_payload["answer"], "Hello")

    async def test_ingest_invalidates_cached_pipeline(self):
        build_count = {"value": 0}

        def build_pipeline(**kwargs):
            build_count["value"] += 1
            return FakePipeline(answer=f"build-{build_count['value']}")

        state = ServiceState(
            ingest_callable=lambda *args, **kwargs: FakeIngestResult(ids=["record-1"]),
            pipeline_builder=build_pipeline,
            chat_client_factory=lambda: FakeChatClient(),
        )
        app = create_app(state)

        async with await self.open_client(app) as client:
            first = await client.post("/query", json={"question": "what"})
            second = await client.post("/query", json={"question": "what"})
            ingest = await client.post("/ingest", json={"path": "tests/fixtures/sample.txt"})
            third = await client.post("/query", json={"question": "what"})

        self.assertEqual(first.json()["answer"], "build-1")
        self.assertEqual(second.json()["answer"], "build-1")
        self.assertEqual(ingest.status_code, 200)
        self.assertTrue(ingest.json()["cache_invalidated"])
        self.assertEqual(third.json()["answer"], "build-2")
        self.assertEqual(build_count["value"], 2)

    async def test_error_mapping_is_sanitized(self):
        cases = [
            (IndexNotReadyError("Vector store collection is empty or unavailable: qa"), 409, "index_not_ready"),
            (
                EmbeddingSpaceMismatchError("Embedding dimension mismatch: indexed=8 requested=4"),
                422,
                "embedding_space_mismatch",
            ),
            (APIError("Rate limit exceeded", status_code=429, retryable=True), 429, "upstream_rate_limited"),
            (APIError("Authentication failed for sk-secret", status_code=401), 502, "upstream_auth_failed"),
            (APIError("Request timed out", status_code=0, retryable=True, error_kind="timeout"), 504, "upstream_timeout"),
            (RuntimeError("boom sk-secret traceback"), 500, "internal_error"),
        ]

        for error, expected_status, expected_code in cases:
            with self.subTest(expected_code=expected_code):
                state = ServiceState(
                    pipeline_builder=lambda **kwargs: FakePipeline(error=error),
                    chat_client_factory=lambda: FakeChatClient(),
                )
                app = create_app(state)
                async with await self.open_client(app) as client:
                    response = await client.post("/query", json={"question": "what"})

                body = response.json()
                self.assertEqual(response.status_code, expected_status)
                self.assertEqual(body["detail"]["error"]["code"], expected_code)
                self.assertNotIn("sk-secret", json.dumps(body))
                self.assertNotIn("traceback", json.dumps(body).lower())

    async def test_query_injects_server_acl_filter_and_intersects_client_acl(self):
        original_enabled = Config.ACL_ENABLED
        original_key = Config.ACL_METADATA_KEY
        original_default_deny = Config.ACL_DEFAULT_DENY
        original_header = Config.ACL_PRINCIPAL_HEADER
        original_allow_body = Config.ACL_ALLOW_BODY_PRINCIPAL
        pipeline = FakePipeline()
        try:
            Config.ACL_ENABLED = True
            Config.ACL_METADATA_KEY = "acl"
            Config.ACL_DEFAULT_DENY = True
            Config.ACL_PRINCIPAL_HEADER = "X-Principal"
            Config.ACL_ALLOW_BODY_PRINCIPAL = False
            state = ServiceState(
                pipeline_builder=lambda **kwargs: pipeline,
                chat_client_factory=lambda: FakeChatClient(),
                acl_resolver=StaticACLResolver({"alice": ["role:finance", "role:admin"]}),
            )
            app = create_app(state)
            async with await self.open_client(app) as client:
                response = await client.post(
                    "/query",
                    headers={"X-Principal": "alice"},
                    json={
                        "question": "what",
                        "metadata_filter": {
                            "source": "finance.md",
                            "acl": ["role:finance", "role:public"],
                        },
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                pipeline.answer_filters[0],
                {"source": "finance.md", "acl": ["role:finance"]},
            )
        finally:
            Config.ACL_ENABLED = original_enabled
            Config.ACL_METADATA_KEY = original_key
            Config.ACL_DEFAULT_DENY = original_default_deny
            Config.ACL_PRINCIPAL_HEADER = original_header
            Config.ACL_ALLOW_BODY_PRINCIPAL = original_allow_body

    async def test_query_missing_principal_fails_closed_before_pipeline_build(self):
        original_enabled = Config.ACL_ENABLED
        original_allow_body = Config.ACL_ALLOW_BODY_PRINCIPAL
        try:
            Config.ACL_ENABLED = True
            Config.ACL_ALLOW_BODY_PRINCIPAL = False
            build_count = {"value": 0}

            def build_pipeline(**kwargs):
                build_count["value"] += 1
                return FakePipeline()

            state = ServiceState(
                pipeline_builder=build_pipeline,
                chat_client_factory=lambda: FakeChatClient(),
                acl_resolver=StaticACLResolver({"alice": ["role:finance"]}),
            )
            app = create_app(state)
            async with await self.open_client(app) as client:
                response = await client.post("/query", json={"question": "what"})

            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["detail"]["error"]["code"], "acl_forbidden")
            self.assertEqual(build_count["value"], 0)
        finally:
            Config.ACL_ENABLED = original_enabled
            Config.ACL_ALLOW_BODY_PRINCIPAL = original_allow_body

    async def test_query_client_acl_widening_is_rejected(self):
        original_enabled = Config.ACL_ENABLED
        original_allow_body = Config.ACL_ALLOW_BODY_PRINCIPAL
        try:
            Config.ACL_ENABLED = True
            Config.ACL_ALLOW_BODY_PRINCIPAL = False
            pipeline = FakePipeline()
            state = ServiceState(
                pipeline_builder=lambda **kwargs: pipeline,
                chat_client_factory=lambda: FakeChatClient(),
                acl_resolver=StaticACLResolver({"alice": ["role:secret"]}),
            )
            app = create_app(state)
            async with await self.open_client(app) as client:
                response = await client.post(
                    "/query",
                    headers={"X-Principal": "alice"},
                    json={
                        "question": "what",
                        "metadata_filter": {"acl": ["role:public"]},
                    },
                )

            self.assertEqual(response.status_code, 403)
            self.assertEqual(pipeline.answer_filters, [])
        finally:
            Config.ACL_ENABLED = original_enabled
            Config.ACL_ALLOW_BODY_PRINCIPAL = original_allow_body

    async def test_query_rejects_body_principal_without_trusted_header(self):
        original_enabled = Config.ACL_ENABLED
        original_allow_body = Config.ACL_ALLOW_BODY_PRINCIPAL
        pipeline = FakePipeline()
        try:
            Config.ACL_ENABLED = True
            Config.ACL_ALLOW_BODY_PRINCIPAL = False
            state = ServiceState(
                pipeline_builder=lambda **kwargs: pipeline,
                chat_client_factory=lambda: FakeChatClient(),
                acl_resolver=StaticACLResolver({"alice": ["role:finance"]}),
            )
            app = create_app(state)
            async with await self.open_client(app) as client:
                response = await client.post(
                    "/query",
                    json={"question": "what", "principal": "alice"},
                )

            self.assertEqual(response.status_code, 403)
            self.assertEqual(response.json()["detail"]["error"]["code"], "acl_forbidden")
            self.assertEqual(pipeline.answer_filters, [])
        finally:
            Config.ACL_ENABLED = original_enabled
            Config.ACL_ALLOW_BODY_PRINCIPAL = original_allow_body

    def test_acl_filter_resolution_never_falls_back_to_body_principal(self):
        original_enabled = Config.ACL_ENABLED
        try:
            Config.ACL_ENABLED = True
            state = ServiceState(
                chat_client_factory=lambda: FakeChatClient(),
                acl_resolver=StaticACLResolver({"alice": ["role:finance"]}),
            )
            app = create_app(state)
            request = QueryRequest(question="what", principal="alice")

            with self.assertRaisesRegex(ACLAccessError, "Trusted principal is required"):
                resolve_request_metadata_filter(app, request)
        finally:
            Config.ACL_ENABLED = original_enabled

    async def test_stream_query_maps_pipeline_build_errors_before_sse_starts(self):
        state = ServiceState(
            pipeline_builder=lambda **kwargs: (_ for _ in ()).throw(
                IndexNotReadyError("Vector store collection is empty or unavailable: qa")
            ),
            chat_client_factory=lambda: FakeChatClient(),
        )
        app = create_app(state)

        async with await self.open_client(app) as client:
            response = await client.post("/query/stream", json={"question": "what"})

        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["detail"]["error"]["code"], "index_not_ready")

    async def test_openapi_schema_exposes_service_routes(self):
        app = create_app(
            ServiceState(
                pipeline_builder=lambda **kwargs: FakePipeline(),
                chat_client_factory=lambda: FakeChatClient(),
            )
        )

        async with await self.open_client(app) as client:
            response = await client.get("/openapi.json")

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertIn("/health", body["paths"])
        self.assertIn("/ingest", body["paths"])
        self.assertIn("/query", body["paths"])
        self.assertIn("/query/stream", body["paths"])
        self.assertIn("/cache/stats", body["paths"])


if __name__ == "__main__":
    unittest.main()
