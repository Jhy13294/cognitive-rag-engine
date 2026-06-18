import math
import unittest
from unittest.mock import AsyncMock, patch

from document_loader import Document, load_and_split_document
from embeddings import (
    EmbeddingConfig,
    HashEmbeddingProvider,
    OpenAIEmbeddingError,
    OpenAIEmbeddingProvider,
    cosine_similarity,
    normalize_vector,
)

from tests.test_document_ingestion import FIXTURES_DIR


class EmbeddingUtilityTests(unittest.TestCase):
    def test_embedding_config_validates_required_values(self):
        with self.assertRaises(ValueError):
            EmbeddingConfig(model_name="", dimension=128)

        with self.assertRaises(ValueError):
            EmbeddingConfig(model_name="demo", dimension=0)

        with self.assertRaises(ValueError):
            EmbeddingConfig(model_name="demo", dimension=128, batch_size=0)

    def test_normalize_vector_returns_unit_vector(self):
        vector = normalize_vector([3.0, 4.0])
        norm = math.sqrt(sum(value * value for value in vector))

        self.assertAlmostEqual(norm, 1.0)
        self.assertEqual(vector, [0.6, 0.8])

    def test_cosine_similarity_rejects_dimension_mismatch(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1.0], [1.0, 0.0])

    def test_cosine_similarity_for_identical_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)


class HashEmbeddingProviderTests(unittest.TestCase):
    def test_embed_text_returns_expected_dimension(self):
        provider = HashEmbeddingProvider(dimension=32)
        vector = provider.embed_text("enterprise rag knowledge base")

        self.assertEqual(len(vector), 32)

    def test_embed_text_is_deterministic(self):
        provider = HashEmbeddingProvider(dimension=32)
        first = provider.embed_text("same input")
        second = provider.embed_text("same input")

        self.assertEqual(first, second)

    def test_embed_text_normalizes_non_empty_vectors(self):
        provider = HashEmbeddingProvider(dimension=32, normalize=True)
        vector = provider.embed_text("normalization check")
        norm = math.sqrt(sum(value * value for value in vector))

        self.assertAlmostEqual(norm, 1.0)

    def test_embed_text_can_return_zero_vector_for_empty_text(self):
        provider = HashEmbeddingProvider(dimension=32)
        vector = provider.embed_text("")

        self.assertEqual(vector, [0.0] * 32)

    def test_embed_documents_preserves_metadata_and_adds_embedding_metadata(self):
        chunks = load_and_split_document(
            str(FIXTURES_DIR / "sample.txt"),
            clean=True,
            chunk_size=80,
            chunk_overlap=10,
        )
        provider = HashEmbeddingProvider(dimension=64)
        embedded_documents = provider.embed_documents(chunks)

        self.assertEqual(len(embedded_documents), len(chunks))
        self.assertEqual(len(embedded_documents[0].embedding), 64)
        self.assertEqual(embedded_documents[0].metadata["embedding_model"], "hash-embedding-local")
        self.assertEqual(embedded_documents[0].metadata["embedding_dimension"], 64)
        self.assertTrue(embedded_documents[0].metadata["embedding_normalized"])
        self.assertIn("chunk_index", embedded_documents[0].metadata)

    def test_similar_texts_can_be_compared(self):
        provider = HashEmbeddingProvider(dimension=64)
        first = provider.embed_text("rag document chunk")
        second = provider.embed_text("rag document chunk")

        self.assertAlmostEqual(cosine_similarity(first, second), 1.0)


class FakeEmbeddingResponse:
    def __init__(self, status_code=200, data=None, usage=None, text="{}", headers=None):
        self.status_code = status_code
        self._data = data or []
        self._usage = usage or {"prompt_tokens": len(self._data), "total_tokens": len(self._data)}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return {
            "object": "list",
            "data": self._data,
            "model": "text-embedding-3-small",
            "usage": self._usage,
        }


class FakeEmbeddingAsyncClient:
    """Async httpx client test double for OpenAI embedding requests."""

    created_count = 0
    responses = []
    requests = []

    def __init__(self, timeout=None):
        FakeEmbeddingAsyncClient.created_count += 1
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        FakeEmbeddingAsyncClient.requests.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": self.timeout,
            }
        )
        response = FakeEmbeddingAsyncClient.responses.pop(0)
        if callable(response):
            return response(url=url, headers=headers, json=json)
        return response


def reset_fake_embedding_client(*responses):
    FakeEmbeddingAsyncClient.created_count = 0
    FakeEmbeddingAsyncClient.responses = list(responses)
    FakeEmbeddingAsyncClient.requests = []


class FixedTokenCounter:
    """Fixed token counter for batching tests."""

    encoding_name = "fixed-test"

    def __init__(self, counts):
        self.counts = counts

    def count(self, text):
        return self.counts[text]


def make_embedding_data(count, dimension):
    return [
        {
            "object": "embedding",
            "index": index,
            "embedding": [float(index + 1)] * dimension,
        }
        for index in range(count)
    ]


class OpenAIEmbeddingProviderTests(unittest.TestCase):
    def build_provider(self, **overrides):
        defaults = {
            "api_key": "sk-test",
            "api_url": "https://api.openai.com/v1/embeddings",
            "model_name": "text-embedding-3-small",
            "dimensions": 4,
            "batch_size": 2,
            "max_batch_tokens": 1000,
            "timeout": 5,
            "max_retries": 1,
            "base_delay": 0,
            "max_delay": 0,
        }
        defaults.update(overrides)
        return OpenAIEmbeddingProvider(**defaults)

    @patch("embeddings.openai_provider.httpx.AsyncClient", new=FakeEmbeddingAsyncClient)
    def test_embed_text_sends_dimensions_and_returns_vector(self):
        reset_fake_embedding_client(FakeEmbeddingResponse(data=make_embedding_data(1, 4)))
        provider = self.build_provider(dimensions=4)

        vector = provider.embed_text("hello world")

        self.assertEqual(vector, [1.0, 1.0, 1.0, 1.0])
        payload = FakeEmbeddingAsyncClient.requests[0]["json"]
        self.assertEqual(payload["model"], "text-embedding-3-small")
        self.assertEqual(payload["input"], ["hello world"])
        self.assertEqual(payload["dimensions"], 4)
        self.assertEqual(payload["encoding_format"], "float")

    @patch("embeddings.openai_provider.httpx.AsyncClient", new=FakeEmbeddingAsyncClient)
    def test_embed_documents_batches_requests(self):
        def fake_post(*args, **kwargs):
            batch_size = len(kwargs["json"]["input"])
            return FakeEmbeddingResponse(data=make_embedding_data(batch_size, 4))

        reset_fake_embedding_client(fake_post, fake_post, fake_post)
        provider = self.build_provider(batch_size=2, dimensions=4)
        documents = [
            Document(content=f"document {index}", metadata={"chunk_index": index})
            for index in range(5)
        ]

        embedded_documents = provider.embed_documents(documents)

        self.assertEqual(len(embedded_documents), 5)
        self.assertEqual(len(FakeEmbeddingAsyncClient.requests), 3)
        self.assertEqual(len(FakeEmbeddingAsyncClient.requests[0]["json"]["input"]), 2)
        self.assertEqual(len(FakeEmbeddingAsyncClient.requests[1]["json"]["input"]), 2)
        self.assertEqual(len(FakeEmbeddingAsyncClient.requests[2]["json"]["input"]), 1)
        self.assertEqual(embedded_documents[0].metadata["embedding_provider"], "openai")
        self.assertEqual(embedded_documents[0].metadata["embedding_dimension"], 4)
        self.assertEqual(embedded_documents[0].metadata["chunk_index"], 0)

    def test_build_batches_uses_injected_token_counter(self):
        provider = self.build_provider(
            batch_size=10,
            max_batch_tokens=5,
            token_counter=FixedTokenCounter(
                {
                    "中文片段一": 4,
                    "中文片段二": 4,
                    "english": 1,
                }
            ),
        )

        batches = provider._build_batches(["中文片段一", "中文片段二", "english"])

        self.assertEqual(batches, [["中文片段一"], ["中文片段二", "english"]])

    def test_build_batches_rejects_single_input_over_token_limit(self):
        provider = self.build_provider(
            batch_size=10,
            max_batch_tokens=5,
            token_counter=FixedTokenCounter({"too long": 6}),
        )

        with self.assertRaises(OpenAIEmbeddingError):
            provider._build_batches(["too long"])

    @patch("embeddings.openai_provider.httpx.AsyncClient", new=FakeEmbeddingAsyncClient)
    def test_embed_text_retries_rate_limit(self):
        reset_fake_embedding_client(
            FakeEmbeddingResponse(
                status_code=429,
                text='{"error":{"message":"rate limited"}}',
                headers={"Retry-After": "0"},
            ),
            FakeEmbeddingResponse(data=make_embedding_data(1, 4)),
        )
        provider = self.build_provider(max_retries=1, dimensions=4)

        with patch("embeddings.openai_provider.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            vector = provider.embed_text("retry me")

        self.assertEqual(vector, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(FakeEmbeddingAsyncClient.created_count, 1)
        self.assertEqual(len(FakeEmbeddingAsyncClient.requests), 2)
        mock_sleep.assert_awaited_once()

    @patch("embeddings.openai_provider.httpx.AsyncClient", new=FakeEmbeddingAsyncClient)
    def test_embed_text_raises_non_retryable_error(self):
        reset_fake_embedding_client(
            FakeEmbeddingResponse(
                status_code=401,
                text='{"error":{"message":"invalid api key"}}',
            )
        )
        provider = self.build_provider(max_retries=1, dimensions=4)

        with self.assertRaises(OpenAIEmbeddingError):
            provider.embed_text("auth failure")


if __name__ == "__main__":
    unittest.main()
