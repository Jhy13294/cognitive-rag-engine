"""Gated true-backend collaboration tests for the service stack.

These tests are intentionally skipped by default. They require real Qdrant,
Redis, and MySQL so they can catch cross-component failures that unit fakes
cannot represent.

Run:
    RUN_E2E_INTEGRATION=1 QDRANT_URL=... REDIS_URL=... METADATA_DB_URL=... \
    CACHE_ENABLED=true ACL_ENABLED=true OBSERVABILITY_ENABLED=true METRICS_ENABLED=true AUDIT_ENABLED=true \
        ./.venv/Scripts/python.exe -m unittest tests.test_e2e_integration -v
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from access.mysql import MYSQL_SCHEMA_SQL, _parse_mysql_url
from api_client import run_async_blocking
from config import Config
from observability.metrics import (
    ALLOWED_OUTCOMES,
    ALLOWED_RETRIEVAL_MODES,
    ALLOWED_ROUTES,
    ALLOWED_TOKEN_KINDS,
    ALLOWED_TOKEN_SOURCES,
    METRIC_LABEL_NAMES,
)
from vector_store import QdrantVectorStore, is_qdrant_client_available


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _mysql_settings_present() -> bool:
    if os.getenv("METADATA_DB_URL"):
        return True
    return bool(os.getenv("MYSQL_HOST") and os.getenv("MYSQL_USER") and os.getenv("MYSQL_DATABASE"))


def _backends_present() -> bool:
    return bool(
        _env_truthy("RUN_E2E_INTEGRATION")
        and os.getenv("QDRANT_URL")
        and os.getenv("REDIS_URL")
        and _mysql_settings_present()
    )


def _mysql_options() -> Dict[str, Any]:
    if os.getenv("METADATA_DB_URL"):
        return _parse_mysql_url(os.environ["METADATA_DB_URL"])
    return {
        "host": os.getenv("MYSQL_HOST"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE"),
    }


class DeterministicE2EChatClient:
    """Deterministic local chat client used only to avoid external LLM calls."""

    def __init__(self) -> None:
        self.request_count = 0
        self.total_tokens = 0
        self._lock = threading.Lock()

    def chat(self, message: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Return an OpenAI-compatible response with deterministic usage."""
        answer = self._answer_from_prompt(message)
        usage = self._record_usage(message, answer)
        return {
            "choices": [{"message": {"content": answer}}],
            "usage": usage,
            "model": "deterministic-e2e-chat",
        }

    async def stream_chat(self, message: str, system_prompt: Optional[str] = None):
        """Yield a deterministic answer in several chunks."""
        answer = self._answer_from_prompt(message)
        self._record_usage(message, answer)
        for token in self._split_answer(answer):
            await asyncio.sleep(0)
            yield token

    def usage_stats(self) -> Dict[str, int]:
        """Return cumulative usage for the observability watermark."""
        with self._lock:
            return {
                "request_count": self.request_count,
                "total_tokens": self.total_tokens,
            }

    def _record_usage(self, prompt: str, answer: str) -> Dict[str, int]:
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(answer.split()))
        total_tokens = prompt_tokens + completion_tokens
        with self._lock:
            self.request_count += 1
            self.total_tokens += total_tokens
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    @staticmethod
    def _answer_from_prompt(prompt: str) -> str:
        if "No relevant context was retrieved." in prompt:
            return "The answer is not available in the knowledge base."
        if "E2EPRIVATEFINANCETOKEN" in prompt:
            return "The finance-only source states E2EPRIVATEFINANCETOKEN [1]."
        if "E2ESTREAMBODY" in prompt:
            return "The streaming source confirms the cached replay answer [1]."
        if "E2ELEXICALFORBIDDENTOKEN" in prompt:
            return "The lexical-only source is reachable without ACL filtering [1]."
        if "E2ELEGALALLOWEDTOKEN" in prompt:
            return "The legal source is available to the legal principal [1]."
        return "The answer is supported by the retrieved test source [1]."

    @staticmethod
    def _split_answer(answer: str) -> Iterable[str]:
        midpoint = max(1, len(answer) // 2)
        return (answer[:midpoint], answer[midpoint:])


@unittest.skipUnless(
    _backends_present(),
    "Set RUN_E2E_INTEGRATION=1 plus QDRANT_URL, REDIS_URL, and METADATA_DB_URL or MYSQL_HOST/USER/DATABASE.",
)
class E2ECollaborationIntegrationTests(unittest.TestCase):
    """Exercise Qdrant, Redis, MySQL, ACL, cache, streaming, and observability together."""

    DIMENSION = 64
    PRINCIPAL_HEADER = "X-Principal"
    TOP_K = 3

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import mysql.connector  # noqa: F401
            import redis  # noqa: F401
            from fastapi.testclient import TestClient  # noqa: F401
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(f"Missing integration dependency: {error.name}") from error
        if not is_qdrant_client_available():
            raise unittest.SkipTest("qdrant-client is not installed")

        cls.root_dir = Path(__file__).resolve().parents[1]
        cls.run_id = f"e2e_{uuid.uuid4().hex[:12]}"
        cls.collection_name = f"ai_qa_{cls.run_id}"
        cls.redis_namespace = f"rag-cache-{cls.run_id}"
        cls.kb_name = f"{cls.run_id}_kb"
        cls.principal_prefix = f"{cls.run_id}_principal"
        cls.temp_dir = Path(tempfile.mkdtemp(prefix=f"{cls.run_id}_"))
        cls.audit_dir = cls.temp_dir / "audit"
        cls.audit_dir.mkdir(parents=True, exist_ok=True)
        cls.query_rewrite_fixture = cls.temp_dir / "query_rewrites.jsonl"
        cls.mysql_options = _mysql_options()

        cls._original_config = cls._patch_config(audit_path=str(cls.audit_dir / "bootstrap.jsonl"))
        cls._connect_mysql()
        cls._apply_schema_and_seed_mysql()
        cls.qdrant_store = cls._build_qdrant_store(recreate=True)
        cls.redis_client = cls._build_sync_redis_client()

        from fastapi.testclient import TestClient

        from cache import reset_default_cache_store
        from rag_cli import build_rag_pipeline_from_index
        from service.app import ServiceState, create_app, resolve_request_metadata_filter
        from service.models import QueryRequest

        cls.ServiceState = ServiceState
        cls.create_app = staticmethod(create_app)
        cls.QueryRequest = QueryRequest
        cls.TestClient = TestClient
        cls.resolve_request_metadata_filter = staticmethod(resolve_request_metadata_filter)
        cls.build_rag_pipeline_from_index = staticmethod(build_rag_pipeline_from_index)
        cls.reset_default_cache_store = staticmethod(reset_default_cache_store)

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls._delete_qdrant_collection()
        finally:
            try:
                cls._clear_redis_namespace()
                cls.redis_client.close()
            finally:
                try:
                    cls._delete_mysql_seed()
                finally:
                    cls._restore_config()
                    shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def setUp(self) -> None:
        self.case_id = f"{self.run_id}_{self._testMethodName}"
        self.docs_dir = self.temp_dir / self._testMethodName
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.audit_dir / f"{self._testMethodName}.jsonl"
        Config.AUDIT_LOG_PATH = str(self.audit_path)
        self._clear_redis_namespace()
        self.qdrant_store.clear()
        self._delete_mysql_documents()
        self.reset_default_cache_store()

    def tearDown(self) -> None:
        self.reset_default_cache_store()

    def test_acl_cache_keys_prevent_cross_user_l2_l3_poisoning(self) -> None:
        client, app, state = self._build_client()
        question = f"Who can see {self.case_id} finance data?"
        finance_doc = self._ingest_text(
            client,
            "finance_private.md",
            f"E2EPRIVATEFINANCETOKEN {self.case_id} is available only to finance staff.",
            acl=["role:finance"],
        )

        alice_payload = self._query_payload(question)
        alice_response = client.post(
            "/query",
            json=alice_payload,
            headers={self.PRINCIPAL_HEADER: self._principal("alice")},
        )
        self.assertEqual(alice_response.status_code, 200, alice_response.text)
        alice_body = alice_response.json()
        self.assertIn("E2EPRIVATEFINANCETOKEN", alice_body["answer"])
        self.assertTrue(self._response_has_source(alice_body, finance_doc))

        before_bob = self._cache_stats(client)
        bob_response = client.post(
            "/query",
            json=self._query_payload(question),
            headers={self.PRINCIPAL_HEADER: self._principal("bob")},
        )
        self.assertEqual(bob_response.status_code, 200, bob_response.text)
        bob_body = bob_response.json()
        self.assertNotIn("E2EPRIVATEFINANCETOKEN", bob_body["answer"])
        self.assertFalse(self._response_has_source(bob_body, finance_doc))
        after_bob = self._cache_stats(client)
        self.assertEqual(
            self._layer_count(before_bob, "hits", "l3"),
            self._layer_count(after_bob, "hits", "l3"),
            "Bob must not hit Alice's cached L3 answer.",
        )
        self.assertGreater(
            self._layer_count(after_bob, "misses", "l3"),
            self._layer_count(before_bob, "misses", "l3"),
        )

        pipeline = self._single_cached_pipeline(state)
        corpus_version = run_async_blocking(state.cache_store.get_corpus_version())
        self.assertIsInstance(corpus_version, int)
        alice_filter = self._effective_filter(app, alice_payload, self._principal("alice"))
        bob_filter = self._effective_filter(app, alice_payload, self._principal("bob"))
        self.assertNotEqual(
            pipeline._answer_key(question, self.TOP_K, alice_filter, corpus_version),
            pipeline._answer_key(question, self.TOP_K, bob_filter, corpus_version),
        )
        self.assertEqual(
            pipeline._answer_key(question, self.TOP_K, None, corpus_version),
            pipeline._answer_key(question, self.TOP_K, None, corpus_version),
            "If metadata_filter is removed from the key payload, both principals collide.",
        )

    def test_cross_process_corpus_version_makes_old_l2_l3_unreachable(self) -> None:
        client, _app, state = self._build_client()
        question = f"How does freshness work for {self.case_id}?"
        self._ingest_text(
            client,
            "freshness.md",
            f"E2ELEGALALLOWEDTOKEN {self.case_id} validates cross-process freshness.",
            acl=["role:legal"],
        )

        first = client.post(
            "/query",
            json=self._query_payload(question),
            headers={self.PRINCIPAL_HEADER: self._principal("bob")},
        )
        self.assertEqual(first.status_code, 200, first.text)
        stats_after_prime = self._cache_stats(client)
        version_before = run_async_blocking(state.cache_store.get_corpus_version())
        generation_before = state.cache_generation

        bumped_version, subprocess_pid = self._bump_corpus_version_in_subprocess()
        self.assertNotEqual(subprocess_pid, os.getpid())
        self.assertEqual(bumped_version, version_before + 1)
        self.assertEqual(
            state.cache_generation,
            generation_before,
            "Process one must not self-invalidate for the cross-process freshness proof.",
        )

        second = client.post(
            "/query",
            json=self._query_payload(question),
            headers={self.PRINCIPAL_HEADER: self._principal("bob")},
        )
        self.assertEqual(second.status_code, 200, second.text)
        stats_after_bump = self._cache_stats(client)
        self.assertEqual(
            self._layer_count(stats_after_prime, "hits", "l3"),
            self._layer_count(stats_after_bump, "hits", "l3"),
        )
        self.assertGreater(
            self._layer_count(stats_after_bump, "misses", "l3"),
            self._layer_count(stats_after_prime, "misses", "l3"),
        )
        self.assertGreater(
            self._layer_count(stats_after_bump, "writes", "l3"),
            self._layer_count(stats_after_prime, "writes", "l3"),
        )

    def test_l3_replay_does_not_double_count_chat_tokens_under_concurrency(self) -> None:
        client, _app, _state = self._build_client()
        question = f"What is the token accounting answer for {self.case_id}?"
        self._ingest_text(
            client,
            "token_accounting.md",
            f"E2ELEGALALLOWEDTOKEN {self.case_id} supports token accounting.",
            acl=["role:legal"],
        )

        prime = client.post(
            "/query",
            json=self._query_payload(question),
            headers={self.PRINCIPAL_HEADER: self._principal("bob")},
        )
        self.assertEqual(prime.status_code, 200, prime.text)
        self._wait_for_audit_records(1)
        token_total_before = self._chat_token_total(client)
        audit_count_before = len(self._audit_records())

        def run_query() -> int:
            response = client.post(
                "/query",
                json=self._query_payload(question),
                headers={self.PRINCIPAL_HEADER: self._principal("bob")},
            )
            return response.status_code

        parallel_count = 4
        with ThreadPoolExecutor(max_workers=parallel_count) as executor:
            statuses = list(executor.map(lambda _: run_query(), range(parallel_count)))

        self.assertEqual(statuses, [200] * parallel_count)
        self._wait_for_audit_records(audit_count_before + parallel_count)
        token_total_after = self._chat_token_total(client)
        self.assertEqual(token_total_after, token_total_before)
        replay_records = self._audit_records()[-parallel_count:]
        self.assertEqual(
            [record["cache_outcome"] for record in replay_records], ["hit_l3"] * parallel_count
        )

    def test_hybrid_multi_query_respects_acl_on_dense_and_bm25_paths(self) -> None:
        client, _app, _state = self._build_client()
        question = f"Find lexical-only forbidden runbook {self.case_id}"
        forbidden_doc = self._ingest_text(
            client,
            "forbidden_lexical.md",
            (
                f"E2ELEXICALFORBIDDENTOKEN {self.case_id} forbidden runbook "
                "lexical-only access phrase repeated repeated."
            ),
            acl=["role:finance"],
        )
        self._ingest_text(
            client,
            "legal_filler.md",
            f"E2ELEGALALLOWEDTOKEN {self.case_id} is unrelated legal material.",
            acl=["role:legal"],
        )
        self._write_query_rewrite_fixture(
            question,
            [
                f"E2ELEXICALFORBIDDENTOKEN {self.case_id} forbidden runbook",
                "lexical-only access phrase repeated",
            ],
        )

        response = client.post(
            "/query",
            json=self._query_payload(
                question,
                hybrid=True,
                hybrid_fetch_k=5,
                multi_query=True,
                query_rewrite_provider="deterministic",
                query_rewrite_fixture=str(self.query_rewrite_fixture),
                query_rewrite_num_queries=3,
            ),
            headers={self.PRINCIPAL_HEADER: self._principal("bob")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse(self._response_has_source(response.json(), forbidden_doc))
        for source in response.json().get("sources", []):
            self.assertNotIn("E2ELEXICALFORBIDDENTOKEN", source.get("content", ""))

        pipeline = self.build_rag_pipeline_from_index(
            chat_client=DeterministicE2EChatClient(),
            embedding_provider_name="hash",
            embedding_dimension=self.DIMENSION,
            vector_store_name="qdrant",
            rerank_provider_name="none",
            hybrid_enabled=True,
            hybrid_fetch_k=5,
            query_rewrite_enabled=True,
            query_rewrite_provider_name="deterministic",
            query_rewrite_fixture_path=str(self.query_rewrite_fixture),
            query_rewrite_num_queries=3,
            top_k=5,
        )
        unfiltered_sources = pipeline.retrieve(question, top_k=5, metadata_filter=None)
        self.assertTrue(
            any(source.metadata.get("source") == forbidden_doc for source in unfiltered_sources),
            "The forbidden document must be reachable when ACL filtering is removed.",
        )
        self.assertTrue(
            any(source.metadata.get("query_rewrite_enabled") for source in unfiltered_sources),
            "The negative control must exercise the multi-query path, not only single-query BM25.",
        )

    def test_stream_replays_l3_with_cached_flags_and_sanitized_single_audit(self) -> None:
        client, _app, _state = self._build_client()
        question = f"Stream cached answer for {self.case_id}?"
        source_path = self._ingest_text(
            client,
            "stream.md",
            f"E2ESTREAMBODY {self.case_id} should not appear in audit logs.",
            acl=["role:legal"],
        )

        prime = client.post(
            "/query",
            json=self._query_payload(question),
            headers={self.PRINCIPAL_HEADER: self._principal("bob")},
        )
        self.assertEqual(prime.status_code, 200, prime.text)
        cached_answer = prime.json()["answer"]
        audit_before_stream = len(self._audit_records())

        events = self._post_stream(
            client,
            self._query_payload(question),
            principal=self._principal("bob"),
        )
        self.assertEqual(events[0][0], "sources")
        self.assertTrue(events[0][1].get("cached"))
        self.assertTrue(events[0][1].get("stream_replay"))
        streamed_answer = "".join(
            payload.get("delta", "") for name, payload in events if name == "token"
        )
        self.assertEqual(streamed_answer, cached_answer)
        self.assertTrue(events[-1][1].get("cached"))

        self._wait_for_audit_records(audit_before_stream + 1)
        audit_record = self._audit_records()[-1]
        self.assertEqual(audit_record["cache_outcome"], "hit_l3")
        audit_text = json.dumps(audit_record, ensure_ascii=False)
        self.assertNotIn(question, audit_text)
        self.assertNotIn("E2ESTREAMBODY", audit_text)
        self.assertNotIn(source_path, audit_text)
        self.assertNotIn(self._principal("bob"), audit_text)

        miss_events = self._post_stream(
            client,
            self._query_payload(f"Unprimed stream miss for {self.case_id}?"),
            principal=self._principal("bob"),
        )
        self.assertFalse(
            any(payload.get("cached") is True for _name, payload in miss_events),
            "A cache miss stream must not advertise cached replay flags.",
        )

    def test_metrics_labels_remain_bounded_with_many_principals_and_questions(self) -> None:
        client, _app, _state = self._build_client()
        self._ingest_text(
            client,
            "metrics.md",
            f"E2ELEGALALLOWEDTOKEN {self.case_id} supports bounded metrics.",
            acl=["role:legal"],
        )
        metric_principals = [self._principal(f"metrics{i}") for i in range(6)]
        self._seed_principal_memberships(
            [(principal, "role:legal") for principal in metric_principals]
        )

        first_questions = [
            f"Metrics cardinality first {index} {self.case_id}?" for index in range(3)
        ]
        second_questions = [
            f"Metrics cardinality second {index} {self.case_id}?" for index in range(3)
        ]
        for principal, question in zip(metric_principals[:3], first_questions, strict=True):
            response = client.post(
                "/query",
                json=self._query_payload(question),
                headers={self.PRINCIPAL_HEADER: principal},
            )
            self.assertEqual(response.status_code, 200, response.text)
        metrics_after_first = client.get(Config.METRICS_PATH).text
        series_after_first = self._metric_series(metrics_after_first)

        for principal, question in zip(metric_principals[3:], second_questions, strict=True):
            response = client.post(
                "/query",
                json=self._query_payload(question),
                headers={self.PRINCIPAL_HEADER: principal},
            )
            self.assertEqual(response.status_code, 200, response.text)
        metrics_after_second = client.get(Config.METRICS_PATH).text
        series_after_second = self._metric_series(metrics_after_second)

        self.assertEqual(series_after_second, series_after_first)
        self._assert_metric_labels_bounded(metrics_after_second)
        forbidden_fragments = (
            metric_principals
            + first_questions
            + second_questions
            + ["E2ELEGALALLOWEDTOKEN", str(self.docs_dir)]
        )
        for fragment in forbidden_fragments:
            self.assertNotIn(fragment, metrics_after_second)

    @classmethod
    def _patch_config(cls, *, audit_path: str) -> Dict[str, Any]:
        names = [
            "MODEL_NAME",
            "TEMPERATURE",
            "MAX_TOKENS",
            "EMBEDDING_PROVIDER",
            "EMBEDDING_DIMENSION",
            "VECTOR_STORE_PROVIDER",
            "VECTOR_STORE_URL",
            "VECTOR_STORE_API_KEY",
            "VECTOR_STORE_COLLECTION",
            "VECTOR_STORE_RECREATE",
            "VECTOR_STORE_BATCH_SIZE",
            "RERANK_ENABLED",
            "RERANK_PROVIDER",
            "HYBRID_ENABLED",
            "QUERY_REWRITE_ENABLED",
            "QUERY_REWRITE_PROVIDER",
            "CACHE_ENABLED",
            "CACHE_NAMESPACE",
            "CACHE_EMBEDDING_ENABLED",
            "CACHE_RETRIEVAL_ENABLED",
            "CACHE_ANSWER_ENABLED",
            "CACHE_TIMEOUT",
            "REDIS_URL",
            "ACL_ENABLED",
            "ACL_METADATA_KEY",
            "ACL_DEFAULT_DENY",
            "ACL_PRINCIPAL_HEADER",
            "ACL_ALLOW_BODY_PRINCIPAL",
            "ACL_INGEST_BINDINGS_ENABLED",
            "METADATA_DB_URL",
            "OBSERVABILITY_ENABLED",
            "AUDIT_ENABLED",
            "METRICS_ENABLED",
            "AUDIT_LOG_PATH",
            "AUDIT_LOG_QUERY_TEXT",
            "AUDIT_PRINCIPAL_MODE",
            "AUDIT_HASH_SALT",
            "AUDIT_QUEUE_SIZE",
            "METRICS_NAMESPACE",
            "METRICS_PATH",
        ]
        original = {name: getattr(Config, name) for name in names}
        overrides = {
            "MODEL_NAME": "deterministic-e2e-chat",
            "TEMPERATURE": 0.0,
            "MAX_TOKENS": 512,
            "EMBEDDING_PROVIDER": "hash",
            "EMBEDDING_DIMENSION": cls.DIMENSION,
            "VECTOR_STORE_PROVIDER": "qdrant",
            "VECTOR_STORE_URL": os.getenv("QDRANT_URL"),
            "VECTOR_STORE_API_KEY": os.getenv("QDRANT_API_KEY"),
            "VECTOR_STORE_COLLECTION": cls.collection_name,
            "VECTOR_STORE_RECREATE": False,
            "VECTOR_STORE_BATCH_SIZE": 16,
            "RERANK_ENABLED": False,
            "RERANK_PROVIDER": "none",
            "HYBRID_ENABLED": False,
            "QUERY_REWRITE_ENABLED": False,
            "QUERY_REWRITE_PROVIDER": "deterministic",
            "CACHE_ENABLED": True,
            "CACHE_NAMESPACE": cls.redis_namespace,
            "CACHE_EMBEDDING_ENABLED": True,
            "CACHE_RETRIEVAL_ENABLED": True,
            "CACHE_ANSWER_ENABLED": True,
            "CACHE_TIMEOUT": 2.0,
            "REDIS_URL": os.getenv("REDIS_URL"),
            "ACL_ENABLED": True,
            "ACL_METADATA_KEY": "acl",
            "ACL_DEFAULT_DENY": True,
            "ACL_PRINCIPAL_HEADER": cls.PRINCIPAL_HEADER,
            "ACL_ALLOW_BODY_PRINCIPAL": False,
            "ACL_INGEST_BINDINGS_ENABLED": False,
            "METADATA_DB_URL": os.getenv("METADATA_DB_URL"),
            "OBSERVABILITY_ENABLED": True,
            "AUDIT_ENABLED": True,
            "METRICS_ENABLED": True,
            "AUDIT_LOG_PATH": audit_path,
            "AUDIT_LOG_QUERY_TEXT": False,
            "AUDIT_PRINCIPAL_MODE": "hash",
            "AUDIT_HASH_SALT": f"{cls.run_id}-audit-salt",
            "AUDIT_QUEUE_SIZE": 1000,
            "METRICS_NAMESPACE": "rag",
            "METRICS_PATH": "/metrics",
        }
        for name, value in overrides.items():
            setattr(Config, name, value)
        return original

    @classmethod
    def _restore_config(cls) -> None:
        for name, value in getattr(cls, "_original_config", {}).items():
            setattr(Config, name, value)

    @classmethod
    def _connect_mysql(cls):
        import mysql.connector

        return mysql.connector.connect(connection_timeout=5, **cls.mysql_options)

    @classmethod
    def _apply_schema_and_seed_mysql(cls) -> None:
        connection = cls._connect_mysql()
        cursor = connection.cursor()
        try:
            for statement in MYSQL_SCHEMA_SQL.split(";"):
                if statement.strip():
                    cursor.execute(statement)
            cls._delete_mysql_seed_with_cursor(cursor)
            cursor.execute("INSERT INTO knowledge_base (name) VALUES (%s)", (cls.kb_name,))
            cls.kb_id = cursor.lastrowid
            cls._seed_principal_memberships_with_cursor(
                cursor,
                [
                    (cls._principal("alice"), "role:finance"),
                    (cls._principal("bob"), "role:legal"),
                    (cls._principal("carol"), "role:admin"),
                ],
            )
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    @classmethod
    def _delete_mysql_seed(cls) -> None:
        connection = cls._connect_mysql()
        cursor = connection.cursor()
        try:
            cls._delete_mysql_seed_with_cursor(cursor)
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    @classmethod
    def _delete_mysql_seed_with_cursor(cls, cursor) -> None:
        cursor.execute(
            "DELETE FROM principal_acl_membership WHERE principal LIKE %s ESCAPE '\\\\'",
            (cls._mysql_like_prefix(cls.principal_prefix),),
        )
        cursor.execute("SELECT id FROM knowledge_base WHERE name = %s", (cls.kb_name,))
        kb_ids = [row[0] for row in cursor.fetchall()]
        if not kb_ids:
            return
        placeholders = ", ".join(["%s"] * len(kb_ids))
        cursor.execute(
            f"SELECT id FROM document WHERE knowledge_base_id IN ({placeholders})", kb_ids
        )
        doc_ids = [row[0] for row in cursor.fetchall()]
        if doc_ids:
            doc_placeholders = ", ".join(["%s"] * len(doc_ids))
            cursor.execute(
                f"SELECT id FROM chunk WHERE document_id IN ({doc_placeholders})", doc_ids
            )
            chunk_ids = [row[0] for row in cursor.fetchall()]
            if chunk_ids:
                chunk_placeholders = ", ".join(["%s"] * len(chunk_ids))
                cursor.execute(
                    f"DELETE FROM acl_binding WHERE chunk_id IN ({chunk_placeholders})", chunk_ids
                )
                cursor.execute(f"DELETE FROM chunk WHERE id IN ({chunk_placeholders})", chunk_ids)
            cursor.execute(
                f"DELETE FROM acl_binding WHERE document_id IN ({doc_placeholders})", doc_ids
            )
            cursor.execute(f"DELETE FROM document WHERE id IN ({doc_placeholders})", doc_ids)
        cursor.execute(f"DELETE FROM knowledge_base WHERE id IN ({placeholders})", kb_ids)

    @staticmethod
    def _mysql_like_prefix(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"{escaped}%"

    def _delete_mysql_documents(self) -> None:
        connection = self._connect_mysql()
        cursor = connection.cursor()
        try:
            cursor.execute("SELECT id FROM document WHERE knowledge_base_id = %s", (self.kb_id,))
            doc_ids = [row[0] for row in cursor.fetchall()]
            if doc_ids:
                placeholders = ", ".join(["%s"] * len(doc_ids))
                cursor.execute(
                    f"SELECT id FROM chunk WHERE document_id IN ({placeholders})", doc_ids
                )
                chunk_ids = [row[0] for row in cursor.fetchall()]
                if chunk_ids:
                    chunk_placeholders = ", ".join(["%s"] * len(chunk_ids))
                    cursor.execute(
                        f"DELETE FROM acl_binding WHERE chunk_id IN ({chunk_placeholders})",
                        chunk_ids,
                    )
                    cursor.execute(
                        f"DELETE FROM chunk WHERE id IN ({chunk_placeholders})", chunk_ids
                    )
                cursor.execute(
                    f"DELETE FROM acl_binding WHERE document_id IN ({placeholders})", doc_ids
                )
                cursor.execute(f"DELETE FROM document WHERE id IN ({placeholders})", doc_ids)
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    @classmethod
    def _seed_principal_memberships_with_cursor(cls, cursor, pairs: List[Tuple[str, str]]) -> None:
        cursor.executemany(
            "INSERT IGNORE INTO principal_acl_membership (principal, acl_subject) VALUES (%s, %s)",
            pairs,
        )

    def _seed_principal_memberships(self, pairs: List[Tuple[str, str]]) -> None:
        connection = self._connect_mysql()
        cursor = connection.cursor()
        try:
            self._seed_principal_memberships_with_cursor(cursor, pairs)
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    def _seed_source_acl(self, source: str, acl_values: List[str]) -> None:
        connection = self._connect_mysql()
        cursor = connection.cursor()
        try:
            cursor.execute(
                "INSERT INTO document (knowledge_base_id, source, checksum) VALUES (%s, %s, %s)",
                (self.kb_id, source, f"checksum-{uuid.uuid4().hex}"),
            )
            document_id = cursor.lastrowid
            cursor.executemany(
                "INSERT INTO acl_binding (document_id, acl_subject) VALUES (%s, %s)",
                [(document_id, acl_value) for acl_value in acl_values],
            )
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    @classmethod
    def _build_qdrant_store(cls, *, recreate: bool = False) -> QdrantVectorStore:
        return QdrantVectorStore(
            collection_name=cls.collection_name,
            dimension=cls.DIMENSION,
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            recreate=recreate,
            batch_size=16,
            timeout=10.0,
        )

    @classmethod
    def _delete_qdrant_collection(cls) -> None:
        try:
            cls.qdrant_store._client.delete_collection(collection_name=cls.collection_name)
        except Exception:
            pass

    @classmethod
    def _build_sync_redis_client(cls):
        import redis

        return redis.Redis.from_url(os.environ["REDIS_URL"], decode_responses=False)

    @classmethod
    def _clear_redis_namespace(cls) -> None:
        if not hasattr(cls, "redis_client"):
            return
        keys = list(cls.redis_client.scan_iter(match=f"{cls.redis_namespace}:*"))
        if keys:
            cls.redis_client.delete(*keys)

    @classmethod
    def _principal(cls, suffix: str) -> str:
        return f"{cls.principal_prefix}_{suffix}"

    def _build_client(self):
        from cache import get_default_cache_store

        self.reset_default_cache_store()
        chat_clients: List[DeterministicE2EChatClient] = []

        def chat_factory() -> DeterministicE2EChatClient:
            client = DeterministicE2EChatClient()
            chat_clients.append(client)
            return client

        cache_store = get_default_cache_store()
        state = self.ServiceState(
            chat_client_factory=chat_factory,
            cache_store=cache_store,
        )
        state.e2e_chat_clients = chat_clients
        app = self.create_app(state)
        client = self.TestClient(app)
        self.addCleanup(client.close)
        return client, app, state

    def _query_payload(self, question: str, **overrides: Any) -> Dict[str, Any]:
        payload = {
            "question": question,
            "top_k": self.TOP_K,
            "embedding_provider": "hash",
            "embedding_dimension": self.DIMENSION,
            "vector_store": "qdrant",
            "rerank_provider": "none",
            "max_context_chars": 4000,
        }
        payload.update(overrides)
        return payload

    def _ingest_text(self, client, name: str, content: str, *, acl: List[str]) -> str:
        path = self.docs_dir / name
        path.write_text(content, encoding="utf-8")
        source = str(path)
        self._seed_source_acl(source, acl)
        response = client.post(
            "/ingest",
            json={
                "path": source,
                "acl": acl,
                "clean": False,
                "recursive": True,
                "chunk_size": 4096,
                "chunk_overlap": 0,
                "embedding_provider": "hash",
                "embedding_dimension": self.DIMENSION,
                "vector_store": "qdrant",
                "parent_child": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(response.json()["records_added"], 0)
        return source

    def _write_query_rewrite_fixture(self, question: str, rewrites: List[str]) -> None:
        payload = {"question": question, "rewrites": rewrites}
        self.query_rewrite_fixture.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    @staticmethod
    def _response_has_source(response_body: Dict[str, Any], source_path: str) -> bool:
        return any(
            source.get("metadata", {}).get("source") == source_path
            for source in response_body.get("sources", [])
        )

    @staticmethod
    def _layer_count(stats: Dict[str, Any], section: str, layer: str) -> int:
        return int((stats.get(section) or {}).get(layer, 0) or 0)

    @staticmethod
    def _cache_stats(client) -> Dict[str, Any]:
        response = client.get("/cache/stats")
        response.raise_for_status()
        return response.json()

    def _effective_filter(self, app, payload: Dict[str, Any], principal: str) -> Dict[str, Any]:
        request = self.QueryRequest(**payload)
        return self.resolve_request_metadata_filter(app, request, principal)

    @staticmethod
    def _single_cached_pipeline(state):
        pipelines = list(state.pipeline_cache.values())
        if not pipelines:
            raise AssertionError("No cached pipeline was built.")
        pipeline = pipelines[0]
        if not hasattr(pipeline, "_answer_key"):
            raise AssertionError("Expected a CachingRAGPipeline with _answer_key.")
        return pipeline

    def _bump_corpus_version_in_subprocess(self) -> Tuple[int, int]:
        script = (
            "from cache.redis_store import CacheSettings, RedisCacheStore\n"
            "from api_client import run_async_blocking\n"
            f"settings = CacheSettings(enabled=True, redis_url={Config.REDIS_URL!r}, "
            f"namespace={Config.CACHE_NAMESPACE!r}, timeout_seconds=2.0)\n"
            "store = RedisCacheStore.from_url(settings)\n"
            "try:\n"
            "    print(run_async_blocking(store.bump_corpus_version()))\n"
            "finally:\n"
            "    store.close()\n"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script],
            cwd=str(self.root_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout, stderr = process.communicate(timeout=20)
        self.assertEqual(process.returncode, 0, stderr)
        return int(stdout.strip().splitlines()[-1]), int(process.pid)

    def _post_stream(
        self, client, payload: Dict[str, Any], *, principal: str
    ) -> List[Tuple[str, Dict[str, Any]]]:
        with client.stream(
            "POST",
            "/query/stream",
            json=payload,
            headers={self.PRINCIPAL_HEADER: principal},
        ) as response:
            text = "".join(response.iter_text())
            self.assertEqual(response.status_code, 200, text)
        return self._parse_sse(text)

    @staticmethod
    def _parse_sse(text: str) -> List[Tuple[str, Dict[str, Any]]]:
        events: List[Tuple[str, Dict[str, Any]]] = []
        for block in text.strip().split("\n\n"):
            if not block.strip():
                continue
            event_name = None
            data_lines = []
            for line in block.splitlines():
                if line.startswith("event: "):
                    event_name = line[len("event: ") :]
                elif line.startswith("data: "):
                    data_lines.append(line[len("data: ") :])
            if event_name is None:
                continue
            events.append((event_name, json.loads("\n".join(data_lines))))
        return events

    def _audit_records(self) -> List[Dict[str, Any]]:
        if not self.audit_path.exists():
            return []
        records = []
        for line in self.audit_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records

    def _wait_for_audit_records(
        self, expected_count: int, timeout_seconds: float = 3.0
    ) -> List[Dict[str, Any]]:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            records = self._audit_records()
            if len(records) >= expected_count:
                return records
            time.sleep(0.05)
        records = self._audit_records()
        self.assertGreaterEqual(len(records), expected_count)
        return records

    def _chat_token_total(self, client) -> float:
        metrics = client.get(Config.METRICS_PATH).text
        total = 0.0
        pattern = re.compile(r'^rag_token_usage_total\{kind="chat",source="[^"]+"\}\s+([0-9.]+)$')
        for line in metrics.splitlines():
            match = pattern.match(line)
            if match:
                total += float(match.group(1))
        return total

    @staticmethod
    def _metric_series(metrics_text: str) -> set[str]:
        series = set()
        for line in metrics_text.splitlines():
            if not line or line.startswith("#"):
                continue
            series.add(line.rsplit(" ", 1)[0])
        return series

    def _assert_metric_labels_bounded(self, metrics_text: str) -> None:
        allowed_names = {name for labels in METRIC_LABEL_NAMES.values() for name in labels} | {"le"}
        allowed_values = {
            "route": ALLOWED_ROUTES,
            "outcome": ALLOWED_OUTCOMES | {"hit", "miss"},
            "retrieval_mode": ALLOWED_RETRIEVAL_MODES,
            "kind": ALLOWED_TOKEN_KINDS,
            "source": ALLOWED_TOKEN_SOURCES,
            "cache_layer": {"L1", "L2", "L3"},
        }
        label_pattern = re.compile(r"\{([^}]*)\}")
        for line in metrics_text.splitlines():
            if not line or line.startswith("#"):
                continue
            match = label_pattern.search(line)
            if not match:
                continue
            for raw_label in match.group(1).split(","):
                name, value = raw_label.split("=", 1)
                value = value.strip('"')
                self.assertIn(name, allowed_names)
                if name == "le":
                    self.assertTrue(value == "+Inf" or re.fullmatch(r"-?\d+(\.\d+)?", value))
                elif name in allowed_values:
                    self.assertIn(value, allowed_values[name])


if __name__ == "__main__":
    unittest.main()
