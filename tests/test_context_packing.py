import unittest
from dataclasses import asdict

from config import Config
from embeddings.base import EmbeddingConfig, EmbeddingProvider
from rag import ContextPacker, RAGPipeline, RetrievedSource
from tokenization import HeuristicTokenCounter
from vector_store import InMemoryVectorStore, VectorRecord


class WordTokenCounter:
    """Deterministic token counter for context packing tests."""

    encoding_name = "word-test"

    def count(self, text: str) -> int:
        """Count whitespace-delimited tokens."""
        return max(1, len((text or "").split()))


class StaticEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for retrieval invariance tests."""

    def __init__(self):
        super().__init__(EmbeddingConfig(model_name="static-test", dimension=2))

    def embed_texts(self, texts):
        """Return the same vector for every text."""
        return [[1.0, 0.0] for _ in list(texts)]


class FakeChatClient:
    """Fake chat client for generation tests."""

    def chat(self, message, system_prompt=None):
        """Return a fixed answer."""
        return {"choices": [{"message": {"content": "ok [1]"}}]}


def make_source(
    index: int,
    content: str,
    source: str = "policy.md",
    start_char: int = 0,
    end_char: int = None,
    record_id: str = None,
    score: float = 1.0,
) -> RetrievedSource:
    """Build a RetrievedSource fixture."""
    metadata = {
        "source": source,
        "chunk_index": index - 1,
        "start_char": start_char,
        "end_char": end_char if end_char is not None else start_char + len(content),
    }
    if record_id:
        metadata["id"] = record_id
    return RetrievedSource(index=index, content=content, score=score, metadata=metadata)


def label(source: RetrievedSource) -> str:
    """Format labels like RAGPipeline for stable tests."""
    return "source={source}; chunk={chunk}; score={score:.4f}".format(
        source=source.metadata.get("source"),
        chunk=source.metadata.get("chunk_index"),
        score=source.score,
    )


def block_for(source: RetrievedSource) -> str:
    """Format a source block like ContextPacker."""
    return f"[{source.index}] {label(source)}\n{source.content}".strip()


class ContextPackerTests(unittest.TestCase):
    def test_legacy_context_still_truncates_mid_block_when_packing_is_off(self):
        source = make_source(1, "Alpha complete. Beta complete.", record_id="a")
        full_block = block_for(source)
        alpha_block = f"[1] {label(source)}\nAlpha complete."
        limit = len(alpha_block) + 3
        pipeline = RAGPipeline(
            embedding_provider=StaticEmbeddingProvider(),
            vector_store=InMemoryVectorStore(dimension=2),
            chat_client=FakeChatClient(),
            max_context_chars=limit,
            context_packing_enabled=False,
        )

        context = pipeline._build_context([source])

        self.assertEqual(context, full_block[:limit].rstrip())
        self.assertTrue(context.endswith("Be"))

    def test_packer_truncates_oversized_block_only_at_sentence_boundary(self):
        source = make_source(
            1,
            "Alpha sentence ends here. Gamma words continue here.",
            record_id="a",
        )
        alpha_block = f"[1] {label(source)}\nAlpha sentence ends here."
        old_word_boundary_limit = len(alpha_block) + len(" Gamma")
        packer = ContextPacker(max_context_chars=old_word_boundary_limit, label_formatter=label)

        packed = packer.pack([source])

        self.assertIn("Alpha sentence ends here.", packed.context)
        self.assertNotIn("Gamma", packed.context)
        self.assertTrue(packed.used_sources[0].content.endswith("."))
        self.assertTrue(packed.used_sources[0].metadata["context_truncated"])
        self.assertEqual(packed.used_sources[0].metadata["context_truncation"], "sentence_boundary")

    def test_packer_skips_oversized_remaining_block_and_renumbers_sources(self):
        first = make_source(1, "Alpha complete.", record_id="first")
        second = make_source(2, "Beta sentence one. Beta sentence two. Beta sentence three.", record_id="second")
        third = make_source(3, "Gamma complete.", record_id="third")
        limit = len(block_for(first)) + 2 + len(block_for(make_source(2, third.content, record_id="third")))
        packer = ContextPacker(max_context_chars=limit, label_formatter=label)

        packed = packer.pack([first, second, third])

        self.assertIn("[1]", packed.context)
        self.assertIn("[2]", packed.context)
        self.assertNotIn("[3]", packed.context)
        self.assertIn("Alpha complete.", packed.context)
        self.assertNotIn("Beta sentence", packed.context)
        self.assertIn("Gamma complete.", packed.context)
        self.assertEqual([source.index for source in packed.used_sources], [1, 2])
        self.assertEqual(packed.used_sources[1].metadata["original_index"], 3)

    def test_exact_dedup_reuses_budget_and_records_dropped_ids(self):
        duplicate_one = make_source(1, "Duplicate content.", record_id="dup-1", start_char=10, end_char=28)
        duplicate_two = make_source(2, "Duplicate content.", record_id="dup-2", start_char=10, end_char=28)
        distinct = make_source(3, "Distinct content.", record_id="distinct", start_char=40, end_char=57)
        renumbered_distinct = make_source(2, distinct.content, record_id="distinct", start_char=40, end_char=57)
        limit = len(block_for(duplicate_one)) + 2 + len(block_for(renumbered_distinct))
        packer = ContextPacker(
            max_context_chars=limit,
            label_formatter=label,
            dedup_enabled=True,
        )

        packed = packer.pack([duplicate_one, duplicate_two, distinct])

        self.assertEqual(packed.context.count("Duplicate content."), 1)
        self.assertIn("Distinct content.", packed.context)
        self.assertEqual(packed.deduped_source_ids, ["dup-2"])
        self.assertEqual(packed.used_sources[0].metadata["deduped_source_ids"], ["dup-2"])
        self.assertEqual(packed.used_sources[0].metadata["dedup_reasons"], ["exact"])

    def test_near_duplicate_dedup_is_disabled_by_default(self):
        first = make_source(1, "Alpha beta gamma delta.", record_id="near-1", start_char=0, end_char=23)
        second = make_source(2, "Alpha beta gamma delta!", record_id="near-2", start_char=40, end_char=63)
        default_packer = ContextPacker(
            max_context_chars=500,
            label_formatter=label,
            dedup_enabled=True,
            near_dup_enabled=False,
        )
        near_dup_packer = ContextPacker(
            max_context_chars=500,
            label_formatter=label,
            dedup_enabled=True,
            near_dup_enabled=True,
            near_dup_threshold=0.8,
        )

        default_packed = default_packer.pack([first, second])
        near_dup_packed = near_dup_packer.pack([first, second])

        self.assertEqual(len(default_packed.used_sources), 2)
        self.assertEqual(len(near_dup_packed.used_sources), 1)
        self.assertEqual(near_dup_packed.deduped_source_ids, ["near-2"])

    def test_token_budget_uses_injected_counter_and_stays_within_limit(self):
        sources = [
            make_source(1, "alpha beta gamma", record_id="a"),
            make_source(2, "delta epsilon zeta eta theta", record_id="b"),
            make_source(3, "iota", record_id="c"),
        ]
        packer = ContextPacker(
            max_context_chars=1000,
            max_context_tokens=18,
            token_counter=WordTokenCounter(),
            label_formatter=label,
        )

        packed = packer.pack(sources)

        self.assertLessEqual(WordTokenCounter().count(packed.context), 18)
        self.assertEqual(packed.budget_unit, "tokens")

    def test_context_packing_does_not_change_retrieve_ranked_list(self):
        vector_records = [
            VectorRecord(id="a", content="Alpha target.", embedding=[1.0, 0.0], metadata={"id": "a"}),
            VectorRecord(id="b", content="Beta target.", embedding=[0.9, 0.1], metadata={"id": "b"}),
        ]
        store_off = InMemoryVectorStore(dimension=2)
        store_on = InMemoryVectorStore(dimension=2)
        store_off.add_records(vector_records)
        store_on.add_records(vector_records)
        common_kwargs = {
            "embedding_provider": StaticEmbeddingProvider(),
            "chat_client": FakeChatClient(),
            "top_k": 2,
        }
        pipeline_off = RAGPipeline(vector_store=store_off, context_packing_enabled=False, **common_kwargs)
        pipeline_on = RAGPipeline(
            vector_store=store_on,
            context_packing_enabled=True,
            context_max_tokens=100,
            token_counter=HeuristicTokenCounter(),
            **common_kwargs,
        )

        off_sources = pipeline_off.retrieve("target", top_k=2)
        on_sources = pipeline_on.retrieve("target", top_k=2)

        self.assertEqual([asdict(source) for source in off_sources], [asdict(source) for source in on_sources])

    def test_config_validates_context_packing_values(self):
        original = {
            "CONTEXT_NEAR_DUP_THRESHOLD": Config.CONTEXT_NEAR_DUP_THRESHOLD,
            "CONTEXT_MAX_TOKENS": Config.CONTEXT_MAX_TOKENS,
            "TOKENIZER_ENCODING": Config.TOKENIZER_ENCODING,
        }
        try:
            Config.CONTEXT_NEAR_DUP_THRESHOLD = 1.1
            with self.assertRaises(ValueError):
                Config.validate_context_packing()

            Config.CONTEXT_NEAR_DUP_THRESHOLD = 0.9
            Config.CONTEXT_MAX_TOKENS = 0
            with self.assertRaises(ValueError):
                Config.validate_context_packing()

            Config.CONTEXT_MAX_TOKENS = 2048
            Config.TOKENIZER_ENCODING = "unknown_encoding"
            with self.assertRaises(ValueError):
                Config.validate_context_packing()
        finally:
            for key, value in original.items():
                setattr(Config, key, value)


if __name__ == "__main__":
    unittest.main()
