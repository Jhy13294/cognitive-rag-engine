import json
import unittest
from types import SimpleNamespace

from config import Config
from embeddings import EmbeddingConfig, EmbeddingProvider, HashEmbeddingProvider
from eval.ragas_live import answer_is_abstention
from rag import RAGPipeline, RelevanceGate, RelevanceGateConfig, RetrievedSource
from rag.prompt_builder import CANONICAL_ABSTENTION_RESPONSE
from rag.retrieval_orchestrator import RetrievalOrchestrator
from rerank import RerankResult

Config.EMBEDDING_PROVIDER = "hash"

from service.app import ServiceState, create_app, stream_query_events
from service.models import QueryRequest
from vector_store import InMemoryVectorStore, VectorRecord


class QueryEmbeddingProvider(EmbeddingProvider):
    """Return orthogonal vectors for explicit accept/reject queries."""

    def __init__(self):
        super().__init__(EmbeddingConfig(model_name="semantic-test", dimension=2))

    def embed_texts(self, texts):
        return [[0.0, 1.0] if "reject" in text else [1.0, 0.0] for text in texts]


class CountingChatClient:
    def __init__(self, answer="generated answer"):
        self.answer = answer
        self.chat_calls = 0
        self.stream_calls = 0

    def chat(self, message, system_prompt=None):
        self.chat_calls += 1
        return {"choices": [{"message": {"content": self.answer}}]}

    async def stream_chat(self, message, system_prompt=None):
        self.stream_calls += 1
        yield self.answer


class ExplodingChatClient:
    def __init__(self):
        self.chat_calls = 0
        self.stream_calls = 0

    def chat(self, message, system_prompt=None):
        self.chat_calls += 1
        raise AssertionError("Retrieval rejection must not call chat generation")

    async def stream_chat(self, message, system_prompt=None):
        self.stream_calls += 1
        raise AssertionError("Retrieval rejection must not call streaming generation")
        yield "unreachable"


class HybridScoreOverwritingReranker:
    """Mimic rerank providers that label their RRF input as dense_score."""

    model_name = "rerank-v3.5"

    def rerank(self, query, candidates, top_n=None):
        return [
            RerankResult(
                index=0,
                score=0.9,
                content=candidates[0].content,
                metadata={
                    "rerank_provider": "cohere",
                    "dense_score": candidates[0].score,
                    "dense_rank": 1,
                },
            )
        ]


def build_gate_pipeline(chat_client, enabled=True):
    provider = QueryEmbeddingProvider()
    store = InMemoryVectorStore(dimension=2)
    store.add_records(
        [
            VectorRecord(
                id="source-1",
                content="Supported policy content.",
                embedding=[1.0, 0.0],
                metadata={"id": "source-1", "source": "policy.md", "acl": ["role:reader"]},
            )
        ]
    )
    return RAGPipeline(
        embedding_provider=provider,
        vector_store=store,
        chat_client=chat_client,
        top_k=1,
        relevance_gate_enabled=enabled,
        relevance_gate_min_dense_cosine=0.5,
    )


def make_source(score, metadata=None):
    return RetrievedSource(
        index=1,
        content="candidate",
        score=score,
        metadata=dict(metadata or {}),
    )


def parse_sse(raw_event):
    lines = [line for line in raw_event.strip().splitlines() if line]
    return lines[0].split(":", 1)[1].strip(), json.loads(lines[1].split(":", 1)[1].strip())


class RelevanceGateScoreResolutionTests(unittest.TestCase):
    def test_hybrid_rerank_parent_matrix_resolves_all_eight_combinations(self):
        for hybrid in (False, True):
            for rerank in (False, True):
                for parent in (False, True):
                    with self.subTest(hybrid=hybrid, rerank=rerank, parent=parent):
                        metadata = {}
                        if hybrid:
                            metadata.update({"retrieval_mode": "hybrid_rrf", "dense_score": 0.8})
                        if rerank:
                            metadata.update(
                                {
                                    "rerank_model": "rerank-v3.5",
                                    "rerank_provider": "cohere",
                                }
                            )
                        if parent:
                            metadata.update(
                                {
                                    "parent_expanded": True,
                                    "child_score": 99.0,
                                }
                            )

                        gate = RelevanceGate(
                            RelevanceGateConfig(
                                enabled=True,
                                min_dense_cosine=0.5,
                                min_cohere_rerank_score=0.5,
                                dense_score_space="dense_cosine",
                                hybrid_enabled=hybrid,
                                reranker_provider="cohere" if rerank else None,
                            )
                        )
                        result = gate.apply([make_source(0.2, metadata)])
                        decision = result.relevance_gate_decision

                        if hybrid:
                            self.assertEqual(decision.action, "accepted")
                            self.assertEqual(decision.score_space, "dense_cosine")
                            self.assertEqual(decision.score, 0.8)
                            self.assertEqual(decision.score_source, 'metadata["dense_score"]')
                        elif rerank:
                            self.assertEqual(decision.action, "rejected")
                            self.assertEqual(decision.score_space, "cohere_rerank_score")
                            self.assertEqual(decision.score_source, "source.score")
                        else:
                            self.assertEqual(decision.action, "rejected")
                            self.assertEqual(decision.score_space, "dense_cosine")
                            self.assertEqual(decision.score_source, "source.score")

    def test_deterministic_lexical_rerank_is_observable_noop(self):
        gate = RelevanceGate(
            RelevanceGateConfig(
                enabled=True,
                min_dense_cosine=0.5,
                dense_score_space="dense_cosine",
                reranker_provider="deterministic",
            )
        )
        source = make_source(
            0.0,
            {
                "rerank_model": "deterministic-lexical-reranker",
                "rerank_provider": "deterministic",
                "dense_score": 0.1,
            },
        )

        with self.assertLogs("rag.relevance_gate", level="WARNING") as logs:
            result = gate.apply([source])

        self.assertEqual(result, [source])
        self.assertEqual(result.relevance_gate_decision.score_space, "lexical_score")
        self.assertIn("unsupported_score_space:lexical_score", "\n".join(logs.output))
        self.assertEqual(
            gate.stats()["skip_reasons"]["unsupported_score_space:lexical_score"],
            1,
        )

    def test_bm25_rrf_and_hash_spaces_never_reject(self):
        cases = [
            (
                RelevanceGateConfig(enabled=True, min_dense_cosine=0.5),
                make_source(0.0, {"retrieval_mode": "bm25"}),
                "bm25",
            ),
            (
                RelevanceGateConfig(enabled=True, min_dense_cosine=0.5),
                make_source(0.0, {"retrieval_mode": "hybrid_rrf"}),
                "rrf",
            ),
            (
                RelevanceGateConfig(
                    enabled=True,
                    min_dense_cosine=0.5,
                    dense_score_space="hash_cosine",
                ),
                make_source(0.0),
                "hash_cosine",
            ),
        ]

        for config, source, expected_space in cases:
            with self.subTest(score_space=expected_space):
                gate = RelevanceGate(config)
                with self.assertLogs("rag.relevance_gate", level="WARNING"):
                    result = gate.apply([source])
                self.assertEqual(result, [source])
                self.assertEqual(result.relevance_gate_decision.action, "skipped")
                self.assertEqual(result.relevance_gate_decision.score_space, expected_space)

    def test_multi_query_uses_original_dense_score_instead_of_rrf_score(self):
        gate = RelevanceGate(
            RelevanceGateConfig(
                enabled=True,
                min_dense_cosine=0.5,
                dense_score_space="dense_cosine",
                query_rewrite_enabled=True,
            )
        )
        source = make_source(
            100.0,
            {"retrieval_mode": "hybrid_rrf", "q0_score": 0.2},
        )

        result = gate.apply([source])

        self.assertEqual(result, [])
        self.assertEqual(result.relevance_gate_decision.score, 0.2)
        self.assertEqual(result.relevance_gate_decision.score_source, 'metadata["q0_score"]')

    def test_hybrid_rerank_preserves_real_dense_score_and_noops_when_it_is_absent(self):
        gate = RelevanceGate(
            RelevanceGateConfig(
                enabled=True,
                min_dense_cosine=0.5,
                dense_score_space="dense_cosine",
                hybrid_enabled=True,
                reranker_provider="cohere",
            )
        )
        pipeline = SimpleNamespace(
            relevance_gate=gate,
            bm25_retriever=object(),
            reranker=HybridScoreOverwritingReranker(),
        )
        orchestrator = RetrievalOrchestrator(pipeline)

        dense_candidate = make_source(
            0.02,
            {
                "retrieval_mode": "hybrid_rrf",
                "rrf_score": 0.02,
                "dense_score": 0.8,
                "dense_rank": 2,
            },
        )
        reranked_dense = orchestrator._rerank_sources("query", [dense_candidate], 1)
        accepted = gate.apply(reranked_dense)

        self.assertEqual(reranked_dense[0].metadata["dense_score"], 0.8)
        self.assertEqual(accepted.relevance_gate_decision.action, "accepted")
        self.assertEqual(accepted.relevance_gate_decision.score, 0.8)

        sparse_only_candidate = make_source(
            0.02,
            {
                "retrieval_mode": "hybrid_rrf",
                "rrf_score": 0.02,
                "sparse_score": 8.0,
                "sparse_rank": 1,
            },
        )
        reranked_sparse_only = orchestrator._rerank_sources("query", [sparse_only_candidate], 1)
        with self.assertLogs("rag.relevance_gate", level="WARNING"):
            skipped = gate.apply(reranked_sparse_only)

        self.assertNotIn("dense_score", reranked_sparse_only[0].metadata)
        self.assertEqual(skipped, reranked_sparse_only)
        self.assertEqual(skipped.relevance_gate_decision.action, "skipped")
        self.assertEqual(skipped.relevance_gate_decision.reason, "usable_score_missing")

        pipeline.relevance_gate = RelevanceGate(RelevanceGateConfig(enabled=False))
        legacy_reranked = orchestrator._rerank_sources("query", [dense_candidate], 1)
        self.assertEqual(legacy_reranked[0].metadata["dense_score"], 0.02)


class RelevanceGatePipelineTests(unittest.TestCase):
    def test_rejected_query_returns_canonical_answer_and_never_calls_chat(self):
        chat = ExplodingChatClient()
        pipeline = build_gate_pipeline(chat)

        response = pipeline.answer("reject this unrelated question")

        self.assertEqual(response.answer, CANONICAL_ABSTENTION_RESPONSE)
        self.assertEqual(response.sources, [])
        self.assertEqual(response.prompt, "")
        self.assertEqual(chat.chat_calls, 0)
        self.assertTrue(answer_is_abstention(response.answer))

    def test_supported_query_is_not_falsely_rejected(self):
        chat = CountingChatClient()
        pipeline = build_gate_pipeline(chat)

        response = pipeline.answer("accept supported policy")

        self.assertEqual(response.answer, "generated answer")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(chat.chat_calls, 1)
        self.assertEqual(pipeline.relevance_gate_stats()["actions"]["accepted"], 1)

    def test_acl_empty_candidates_are_not_reported_as_topic_rejection(self):
        chat = CountingChatClient(answer="acl-safe model response")
        pipeline = build_gate_pipeline(chat)

        response = pipeline.answer(
            "accept supported policy",
            metadata_filter={"acl": ["role:denied"]},
        )

        self.assertEqual(response.answer, "acl-safe model response")
        self.assertEqual(response.sources, [])
        self.assertEqual(chat.chat_calls, 1)
        self.assertEqual(pipeline.relevance_gate_stats()["actions"]["no_candidates"], 1)
        self.assertNotIn("relevance_gate", response.raw_response)

    def test_gate_disabled_preserves_generation_even_below_configured_threshold(self):
        chat = CountingChatClient()
        pipeline = build_gate_pipeline(chat, enabled=False)

        response = pipeline.answer("reject but gate is disabled")

        self.assertEqual(response.answer, "generated answer")
        self.assertEqual(len(response.sources), 1)
        self.assertEqual(chat.chat_calls, 1)
        self.assertEqual(pipeline.relevance_gate_stats()["evaluated"], 0)

    def test_hash_provider_gate_is_noop_and_emits_warning(self):
        provider = HashEmbeddingProvider(dimension=8)
        store = InMemoryVectorStore(dimension=8)
        store.add_records(
            [
                VectorRecord(
                    id="hash-1",
                    content="hash content",
                    embedding=provider.embed_text("hash content"),
                    metadata={"id": "hash-1", "source": "hash.md"},
                )
            ]
        )
        pipeline = RAGPipeline(
            provider,
            store,
            CountingChatClient(),
            relevance_gate_enabled=True,
            relevance_gate_min_dense_cosine=0.99,
        )

        with self.assertLogs("rag.relevance_gate", level="WARNING"):
            sources = pipeline.retrieve("unrelated")

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources.relevance_gate_decision.score_space, "hash_cosine")


class RelevanceGateStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_rejection_emits_canonical_answer_without_stream_chat(self):
        chat = ExplodingChatClient()
        pipeline = build_gate_pipeline(chat)
        app = create_app(ServiceState())
        events = []

        async for raw_event in stream_query_events(
            app,
            QueryRequest(question="reject this unrelated question", top_k=1),
            pipeline=pipeline,
        ):
            events.append(parse_sse(raw_event))

        self.assertEqual([name for name, _ in events], ["sources", "token", "done"])
        self.assertEqual(events[0][1]["sources"], [])
        self.assertEqual(events[1][1]["delta"], CANONICAL_ABSTENTION_RESPONSE)
        self.assertEqual(events[2][1]["answer"], CANONICAL_ABSTENTION_RESPONSE)
        self.assertEqual(chat.stream_calls, 0)


if __name__ == "__main__":
    unittest.main()
