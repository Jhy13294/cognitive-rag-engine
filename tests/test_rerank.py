import json
import os
import unittest
from unittest.mock import patch

from eval.baseline import build_hash_retriever
from eval.golden import load_golden_set
from eval.metrics import evaluate_retriever
from rag import RetrievedSource
from rerank import CohereReranker, DeterministicReranker, RerankConfig


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RerankTests(unittest.TestCase):
    def test_rerank_config_validates_values(self):
        with self.assertRaises(ValueError):
            RerankConfig(model_name="", top_n=1)
        with self.assertRaises(ValueError):
            RerankConfig(model_name="x", top_n=0)
        with self.assertRaises(ValueError):
            RerankConfig(model_name="x", fetch_k=0)

    def test_deterministic_reranker_promotes_lexically_relevant_candidate(self):
        reranker = DeterministicReranker(top_n=2)
        candidates = [
            RetrievedSource(index=1, content="Billing invoices and travel receipts.", score=0.9, metadata={"source": "finance.md"}),
            RetrievedSource(index=2, content="A SEV-1 incident must be declared within fifteen minutes.", score=0.8, metadata={"source": "security.md"}),
        ]

        results = reranker.rerank("How quickly should we declare a serious incident?", candidates, top_n=2)

        self.assertEqual(results[0].index, 1)
        self.assertGreater(results[0].score, results[1].score)

    def test_deterministic_reranker_improves_t04_mrr_targets(self):
        examples = load_golden_set("eval/golden_set.jsonl")
        dense_retrieve, dense_metadata = build_hash_retriever(
            "eval/fixtures/knowledge_base",
            chunk_size=500,
            chunk_overlap=80,
            embedding_dimension=64,
        )
        rerank_retrieve, rerank_metadata = build_hash_retriever(
            "eval/fixtures/knowledge_base",
            chunk_size=500,
            chunk_overlap=80,
            embedding_dimension=64,
            reranker=DeterministicReranker(top_n=10, fetch_k=30),
            fetch_k=30,
        )

        dense_report = evaluate_retriever(dense_retrieve, examples, k_values=[3, 5, 10], metadata=dense_metadata)
        rerank_report = evaluate_retriever(rerank_retrieve, examples, k_values=[3, 5, 10], metadata=rerank_metadata)

        self.assertGreater(rerank_report["metrics"]["3"]["mrr"], dense_report["metrics"]["3"]["mrr"])
        self.assertGreater(
            rerank_report["by_capability"]["long_tail"]["3"]["mrr"],
            dense_report["by_capability"]["long_tail"]["3"]["mrr"],
        )
        self.assertGreaterEqual(rerank_report["metrics"]["5"]["recall"], dense_report["metrics"]["5"]["recall"])
        self.assertLessEqual(
            rerank_report["metrics"]["3"]["negative_false_recall_rate"],
            dense_report["metrics"]["3"]["negative_false_recall_rate"],
        )

    def test_deterministic_reranker_is_reproducible(self):
        examples = load_golden_set("eval/golden_set.jsonl")

        def build_report():
            retrieve, metadata = build_hash_retriever(
                "eval/fixtures/knowledge_base",
                chunk_size=500,
                chunk_overlap=80,
                embedding_dimension=64,
                reranker=DeterministicReranker(top_n=10, fetch_k=30),
                fetch_k=30,
            )
            return evaluate_retriever(retrieve, examples, k_values=[3, 5, 10], metadata=metadata)

        first = build_report()
        second = build_report()

        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["by_capability"], second["by_capability"])
        self.assertEqual(first["cases"], second["cases"])

    def test_deterministic_rerank_report_json_is_byte_stable(self):
        examples = load_golden_set("eval/golden_set.jsonl")

        def build_report_json():
            retrieve, metadata = build_hash_retriever(
                "eval/fixtures/knowledge_base",
                chunk_size=500,
                chunk_overlap=80,
                embedding_dimension=64,
                reranker=DeterministicReranker(top_n=10, fetch_k=30),
                fetch_k=30,
            )
            report = evaluate_retriever(retrieve, examples, k_values=[3, 5, 10], metadata=metadata)
            return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

        self.assertEqual(build_report_json(), build_report_json())

    def test_cohere_reranker_uses_mock_session(self):
        session = FakeSession(
            [
                FakeResponse(
                    payload={
                        "results": [
                            {"index": 1, "relevance_score": 0.95},
                            {"index": 0, "relevance_score": 0.12},
                        ]
                    }
                )
            ]
        )
        reranker = CohereReranker(api_key="test-key", session=session, top_n=2, max_retries=0)
        candidates = [
            RetrievedSource(index=1, content="first", score=0.3, metadata={"source": "a.md"}),
            RetrievedSource(index=2, content="second", score=0.2, metadata={"source": "b.md"}),
        ]

        results = reranker.rerank("query", candidates, top_n=2)

        self.assertEqual([result.index for result in results], [1, 0])
        self.assertEqual(session.calls[0]["kwargs"]["json"]["documents"], ["first", "second"])
        self.assertEqual(session.calls[0]["kwargs"]["json"]["top_n"], 2)

    def test_cohere_reranker_batches_candidate_documents(self):
        session = FakeSession(
            [
                FakeResponse(payload={"results": [{"index": 0, "relevance_score": 0.1}]}),
                FakeResponse(payload={"results": [{"index": 0, "relevance_score": 0.9}]}),
            ]
        )
        reranker = CohereReranker(
            api_key="test-key",
            session=session,
            top_n=2,
            batch_size=1,
            max_retries=0,
        )
        candidates = [
            RetrievedSource(index=1, content="first", score=0.3, metadata={"source": "a.md"}),
            RetrievedSource(index=2, content="second", score=0.2, metadata={"source": "b.md"}),
        ]

        results = reranker.rerank("query", candidates, top_n=2)

        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[0]["kwargs"]["json"]["documents"], ["first"])
        self.assertEqual(session.calls[1]["kwargs"]["json"]["documents"], ["second"])
        self.assertEqual([result.index for result in results], [1, 0])

    def test_cohere_reranker_retries_retryable_status(self):
        session = FakeSession(
            [
                FakeResponse(status_code=429, text="rate limited"),
                FakeResponse(payload={"results": [{"index": 0, "relevance_score": 0.88}]}),
            ]
        )
        reranker = CohereReranker(
            api_key="test-key",
            session=session,
            top_n=1,
            max_retries=1,
            base_delay=0,
            max_delay=0,
        )
        candidates = [RetrievedSource(index=1, content="first", score=0.3, metadata={"source": "a.md"})]

        with patch("rerank.cohere_provider.time.sleep"):
            results = reranker.rerank("query", candidates, top_n=1)

        self.assertEqual(len(session.calls), 2)
        self.assertEqual(results[0].index, 0)

    def test_cohere_reranker_does_not_retry_non_retryable_status(self):
        session = FakeSession([FakeResponse(status_code=400, text="bad request")])
        reranker = CohereReranker(api_key="test-key", session=session, top_n=1, max_retries=3)
        candidates = [RetrievedSource(index=1, content="first", score=0.3, metadata={"source": "a.md"})]

        with self.assertRaisesRegex(Exception, "HTTP 400"):
            reranker.rerank("query", candidates, top_n=1)

        self.assertEqual(len(session.calls), 1)


@unittest.skipUnless(
    os.getenv("RUN_COHERE_RERANK_INTEGRATION") and os.getenv("COHERE_API_KEY"),
    "Set RUN_COHERE_RERANK_INTEGRATION=1 and COHERE_API_KEY to run Cohere rerank integration.",
)
class CohereRerankerIntegrationTests(unittest.TestCase):
    def test_cohere_reranker_smoke(self):
        reranker = CohereReranker(api_key=os.environ["COHERE_API_KEY"], top_n=1)
        candidates = [
            RetrievedSource(index=1, content="A SEV-1 incident must be declared within fifteen minutes.", score=0.2),
            RetrievedSource(index=2, content="Travel receipts are due within seven days.", score=0.1),
        ]

        results = reranker.rerank("How quickly should an incident be declared?", candidates, top_n=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].index, 0)


if __name__ == "__main__":
    unittest.main()
