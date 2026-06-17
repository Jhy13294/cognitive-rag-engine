import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from config import Config
from embeddings.base import EmbeddingConfig, EmbeddingProvider
from query_rewrite import (
    ChatQueryRewriter,
    DeterministicQueryRewriter,
    QueryRewriteConfig,
    normalize_query_variants,
)
from rag import RAGPipeline
from vector_store import InMemoryVectorStore, VectorRecord


class MappingEmbeddingProvider(EmbeddingProvider):
    """Embedding provider with deterministic text-to-vector mapping."""

    def __init__(self):
        super().__init__(EmbeddingConfig(model_name="mapping-test", dimension=2))
        self.vectors = {
            "original question": [1.0, 0.0],
            "variant question": [0.0, 1.0],
        }

    def embed_texts(self, texts):
        """Return mapped vectors or a neutral zero vector."""
        return [list(self.vectors.get(str(text), [0.0, 0.0])) for text in texts]


class FixedQueryRewriter:
    """Query rewriter test double."""

    model_name = "fixed-query-rewriter"

    def __init__(self, variants):
        self.variants = variants
        self.calls = []

    def rewrite(self, question):
        """Return fixed variants."""
        self.calls.append(question)
        return list(self.variants)


class FailingQueryRewriter:
    """Query rewriter that raises a planned error."""

    model_name = "failing-query-rewriter"

    def rewrite(self, question):
        """Raise a planned error."""
        raise RuntimeError("planned rewrite failure")


class FakeChatClient:
    """Chat client test double for query rewriting."""

    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def chat(self, message, system_prompt=None):
        """Return a fixed response or raise a fixed error."""
        if self.error:
            raise self.error
        return self.response


def build_vector_store():
    """Build a small vector store for multi-query tests."""
    store = InMemoryVectorStore(dimension=2)
    store.add_records(
        [
            VectorRecord(
                id="record-original",
                content="Original path content",
                embedding=[1.0, 0.0],
                metadata={"id": "record-original", "source": "original.md", "chunk_index": 0},
            ),
            VectorRecord(
                id="record-variant",
                content="Variant path content",
                embedding=[0.0, 1.0],
                metadata={"id": "record-variant", "source": "variant.md", "chunk_index": 0},
            ),
        ]
    )
    return store


def build_pipeline(query_rewriter=None, enabled=False):
    """Build a RAG pipeline for query rewrite tests."""
    return RAGPipeline(
        embedding_provider=MappingEmbeddingProvider(),
        vector_store=build_vector_store(),
        chat_client=FakeChatClient({"choices": [{"message": {"content": "ok"}}]}),
        top_k=2,
        query_rewriter=query_rewriter,
        query_rewrite_enabled=enabled,
        query_rewrite_num_queries=2,
        query_rewrite_weight_original=1.0,
        query_rewrite_weight_variant=0.7,
    )


class QueryRewriteTests(unittest.TestCase):
    def test_normalize_query_variants_keeps_original_first_and_dedupes(self):
        variants = normalize_query_variants("Original Question", ["variant", " original question ", "variant"], 3)

        self.assertEqual(variants, ["Original Question", "variant"])

    def test_deterministic_rewriter_loads_fixture_and_limits_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_path = Path(temp_dir) / "rewrites.jsonl"
            fixture_path.write_text(
                json.dumps(
                    {
                        "question": "How do I find it?",
                        "rewrites": ["search phrase one", "search phrase two"],
                    }
                ),
                encoding="utf-8",
            )
            rewriter = DeterministicQueryRewriter(
                QueryRewriteConfig(
                    provider="deterministic",
                    num_queries=2,
                    fixture_path=str(fixture_path),
                )
            )

            variants = rewriter.rewrite("How do I find it?")

        self.assertEqual(variants, ["How do I find it?", "search phrase one"])

    def test_chat_rewriter_falls_back_to_original_on_error(self):
        rewriter = ChatQueryRewriter(
            QueryRewriteConfig(provider="chat", num_queries=3),
            chat_client=FakeChatClient(error=RuntimeError("timeout")),
        )

        self.assertEqual(rewriter.rewrite("Original question"), ["Original question"])

    def test_query_rewrite_config_rejects_variant_weight_above_original(self):
        with self.assertRaises(ValueError):
            QueryRewriteConfig(
                provider="deterministic",
                weight_original=0.5,
                weight_variant=0.7,
            )

    def test_config_validate_query_rewrite_rejects_invalid_weights(self):
        original = {
            "QUERY_REWRITE_PROVIDER": Config.QUERY_REWRITE_PROVIDER,
            "QUERY_REWRITE_NUM_QUERIES": Config.QUERY_REWRITE_NUM_QUERIES,
            "QUERY_REWRITE_WEIGHT_ORIGINAL": Config.QUERY_REWRITE_WEIGHT_ORIGINAL,
            "QUERY_REWRITE_WEIGHT_VARIANT": Config.QUERY_REWRITE_WEIGHT_VARIANT,
        }
        try:
            Config.QUERY_REWRITE_PROVIDER = "deterministic"
            Config.QUERY_REWRITE_NUM_QUERIES = 3
            Config.QUERY_REWRITE_WEIGHT_ORIGINAL = 0.5
            Config.QUERY_REWRITE_WEIGHT_VARIANT = 0.7

            with self.assertRaises(ValueError):
                Config.validate_query_rewrite()
        finally:
            for key, value in original.items():
                setattr(Config, key, value)

    def test_multi_query_fusion_keeps_original_weight_and_variant_path(self):
        pipeline = build_pipeline(
            query_rewriter=FixedQueryRewriter(["original question", "variant question"]),
            enabled=True,
        )

        sources = pipeline.retrieve("original question", top_k=2)

        self.assertEqual(pipeline.query_rewrite_weight_original, 1.0)
        self.assertEqual(pipeline.query_rewrite_weight_variant, 0.7)
        self.assertEqual(set(sources[0].metadata["query_rewrite_weights"].keys()), {"q0", "q1"})
        self.assertGreaterEqual(
            sources[0].metadata["query_rewrite_weights"]["q0"],
            sources[0].metadata["query_rewrite_weights"]["q1"],
        )
        self.assertTrue(any(source.metadata.get("q1_rank") == 1 for source in sources))

    def test_rewrite_failure_falls_back_to_single_query_byte_equal(self):
        baseline = build_pipeline()
        failing = build_pipeline(query_rewriter=FailingQueryRewriter(), enabled=True)

        baseline_sources = baseline.retrieve("original question", top_k=2)
        fallback_sources = failing.retrieve("original question", top_k=2)

        self.assertEqual([asdict(source) for source in fallback_sources], [asdict(source) for source in baseline_sources])

    def test_empty_rewrite_falls_back_to_single_query_byte_equal(self):
        baseline = build_pipeline()
        empty = build_pipeline(query_rewriter=FixedQueryRewriter([]), enabled=True)

        baseline_sources = baseline.retrieve("original question", top_k=2)
        fallback_sources = empty.retrieve("original question", top_k=2)

        self.assertEqual([asdict(source) for source in fallback_sources], [asdict(source) for source in baseline_sources])


if __name__ == "__main__":
    unittest.main()
