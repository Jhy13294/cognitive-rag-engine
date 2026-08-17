import asyncio
import os
import unittest
from unittest.mock import patch

import numpy as np

import eval.bge_embedding as bge
from config import Config
from eval.bge_embedding import BGEEmbeddingError, BGEEmbeddingProvider


class FakeModel:
    """Minimal fastembed TextEmbedding stand-in returning fixed vectors in order."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        for text in texts:
            yield np.array(self.vectors[text], dtype=np.float32)


class BGEEmbeddingProviderTests(unittest.TestCase):
    def setUp(self):
        bge._MODEL_CACHE.clear()

    def tearDown(self):
        bge._MODEL_CACHE.clear()

    def provider(self, vectors, dimension=3):
        model = FakeModel(vectors)
        provider = BGEEmbeddingProvider(model_name="fake-bge", dimension=dimension)
        # Inject the fake model directly; the provider loads lazily on first embed.
        provider._model = model
        return provider, model

    def test_embed_texts_preserves_order_and_returns_lists(self):
        vectors = {"one": [1, 0, 0], "two": [0, 1, 0], "three": [0, 0, 1]}
        provider, _ = self.provider(vectors)
        result = provider.embed_texts(["one", "two", "three"])
        self.assertEqual(result, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertIsInstance(result[0][0], float)

    def test_embed_text_single(self):
        provider, _ = self.provider({"hi": [0.1, 0.2, 0.3]})
        vector = provider.embed_text("hi")
        self.assertEqual(len(vector), 3)
        # fastembed returns float32; compare with tolerance, not bit-exact.
        for got, expected in zip(vector, [0.1, 0.2, 0.3], strict=True):
            self.assertAlmostEqual(got, expected, places=5)

    def test_empty_list_returns_empty(self):
        provider, model = self.provider({})
        self.assertEqual(provider.embed_texts([]), [])
        self.assertEqual(model.calls, [])

    def test_blank_input_is_rejected(self):
        provider, _ = self.provider({})
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            provider.embed_texts(["  "])

    def test_dimension_mismatch_is_rejected(self):
        provider, _ = self.provider({"bad": [0.1, 0.2]}, dimension=3)
        with self.assertRaisesRegex(BGEEmbeddingError, "dimension mismatch"):
            provider.embed_text("bad")

    def test_async_embed_texts_matches_sync(self):
        vectors = {"a": [1, 0, 0], "b": [0, 1, 0]}
        provider, _ = self.provider(vectors)
        result = asyncio.run(provider.async_embed_texts(["a", "b"]))
        self.assertEqual(result, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    def test_determinism_same_input_same_output(self):
        provider, _ = self.provider({"x": [0.5, 0.5, 0.5]})
        self.assertEqual(provider.embed_text("x"), provider.embed_text("x"))

    def test_model_is_process_cached_not_reloaded(self):
        construct_count = {"n": 0}

        class CountingModel(FakeModel):
            def __init__(self):
                super().__init__({"x": [1, 0, 0]})
                construct_count["n"] += 1

        with patch("fastembed.TextEmbedding", side_effect=lambda **kwargs: CountingModel()):
            p1 = BGEEmbeddingProvider(model_name="cache-me", dimension=3)
            p2 = BGEEmbeddingProvider(model_name="cache-me", dimension=3)
            # Loading is lazy; both first-embeds must share one cached model.
            p1.embed_text("x")
            p2.embed_text("x")
        self.assertEqual(construct_count["n"], 1)


class BGEConfigBranchTests(unittest.TestCase):
    """Local bge mode must not require remote embedding key/url; gemini unchanged."""

    def _base_overrides(self, provider):
        return {
            "RUN_RAGAS_EVAL": True,
            "RAGAS_EMBEDDING_PROVIDER": provider,
            "RAGAS_JUDGE_API_KEY": "judge-key",
            "RAGAS_JUDGE_BASE_URL": "https://api.deepseek.com/v1",
            "RAGAS_EMBEDDING_MODEL": "model-id",
            "API_KEY": "deepseek-key",
        }

    def test_bge_mode_does_not_require_embedding_key_or_url(self):
        overrides = self._base_overrides("bge")
        overrides["RAGAS_EMBEDDING_API_KEY"] = None
        overrides["RAGAS_EMBEDDING_BASE_URL"] = None
        with patch.multiple(Config, **overrides):
            self.assertTrue(Config.validate_ragas())

    def test_gemini_mode_still_requires_embedding_key_and_url(self):
        overrides = self._base_overrides("gemini")
        overrides["RAGAS_EMBEDDING_API_KEY"] = None
        overrides["RAGAS_EMBEDDING_BASE_URL"] = "https://x"
        with patch.multiple(Config, **overrides):
            with self.assertRaisesRegex(ValueError, "RAGAS_EMBEDDING_API_KEY"):
                Config.validate_ragas()

    def test_unknown_provider_is_rejected(self):
        overrides = self._base_overrides("openai")
        overrides["RAGAS_EMBEDDING_API_KEY"] = "k"
        overrides["RAGAS_EMBEDDING_BASE_URL"] = "https://x"
        with patch.multiple(Config, **overrides):
            with self.assertRaisesRegex(ValueError, "must be one of"):
                Config.validate_ragas()


@unittest.skipUnless(
    os.getenv("RUN_BGE_REAL_EMBEDDING") == "1",
    "Set RUN_BGE_REAL_EMBEDDING=1 to exercise the real local BGE model.",
)
class BGERealModelTests(unittest.TestCase):
    def test_real_model_emits_normalized_384d_vectors(self):
        provider = BGEEmbeddingProvider(
            model_name="BAAI/bge-small-en-v1.5",
            dimension=384,
            cache_dir=os.getenv("RAGAS_EMBEDDING_CACHE_DIR") or None,
        )
        vector = provider.embed_text("What is retrieval augmented generation?")
        self.assertEqual(len(vector), 384)
        norm = float(np.linalg.norm(vector))
        self.assertAlmostEqual(norm, 1.0, places=3)
        # local inference is deterministic
        self.assertEqual(vector, provider.embed_text("What is retrieval augmented generation?"))


if __name__ == "__main__":
    unittest.main()
