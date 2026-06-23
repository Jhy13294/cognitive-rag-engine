import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from eval.gemini_embedding import GeminiEmbeddingError, GeminiEmbeddingProvider


class FakeResponse:
    """Minimal httpx-like response for provider tests."""

    def __init__(self, status_code, payload=None, text="", headers=None):
        self.status_code = status_code
        self.payload = payload or {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self.payload


class FakeAsyncClient:
    """Async context manager that returns queued responses."""

    def __init__(self, responses, calls):
        self.responses = responses
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def post(self, url, headers, json):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)


class GeminiEmbeddingProviderTests(unittest.TestCase):
    def provider(self, **overrides):
        settings = {
            "api_key": "gemini-test-key",
            "base_url": "https://generativelanguage.googleapis.com/v1beta",
            "model_name": "gemini-embedding-001",
            "dimension": 3,
            "batch_size": 2,
            "max_retries": 1,
            "base_delay": 0,
            "max_delay": 0,
        }
        settings.update(overrides)
        return GeminiEmbeddingProvider(**settings)

    def test_single_embedding_uses_native_query_task_contract(self):
        calls = []
        responses = [FakeResponse(200, {"embeddings": [{"values": [0.1, 0.2, 0.3]}]})]
        with patch(
            "eval.gemini_embedding.httpx.AsyncClient",
            return_value=FakeAsyncClient(responses, calls),
        ):
            vector = asyncio.run(self.provider().async_embed_text("What is RAG?"))

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        self.assertEqual(
            calls[0]["url"],
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents",
        )
        self.assertEqual(calls[0]["headers"]["x-goog-api-key"], "gemini-test-key")
        request = calls[0]["json"]["requests"][0]
        self.assertEqual(request["model"], "models/gemini-embedding-001")
        self.assertEqual(request["taskType"], "RETRIEVAL_QUERY")
        self.assertEqual(request["content"], {"parts": [{"text": "What is RAG?"}]})

    def test_multiple_texts_are_sent_in_bounded_batches(self):
        calls = []
        responses = [
            FakeResponse(
                200,
                {"embeddings": [{"values": [1, 0, 0]}, {"values": [0, 1, 0]}]},
            ),
            FakeResponse(200, {"embeddings": [{"values": [0, 0, 1]}]}),
        ]
        with patch(
            "eval.gemini_embedding.httpx.AsyncClient",
            side_effect=lambda **kwargs: FakeAsyncClient(responses, calls),
        ):
            vectors = asyncio.run(self.provider().async_embed_texts(["one", "two", "three"]))

        self.assertEqual(vectors, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        self.assertEqual([len(call["json"]["requests"]) for call in calls], [2, 1])

    def test_rate_limit_retries_with_async_sleep(self):
        calls = []
        responses = [
            FakeResponse(429, text="rate limited", headers={"Retry-After": "0"}),
            FakeResponse(200, {"embeddings": [{"values": [0.1, 0.2, 0.3]}]}),
        ]
        with (
            patch(
                "eval.gemini_embedding.httpx.AsyncClient",
                return_value=FakeAsyncClient(responses, calls),
            ),
            patch("eval.gemini_embedding.asyncio.sleep", new=AsyncMock()) as sleep,
        ):
            vector = asyncio.run(self.provider().async_embed_text("retry me"))

        self.assertEqual(vector, [0.1, 0.2, 0.3])
        self.assertEqual(len(calls), 2)
        sleep.assert_awaited_once_with(0.0)

    def test_dimension_mismatch_is_rejected(self):
        calls = []
        responses = [FakeResponse(200, {"embeddings": [{"values": [0.1, 0.2]}]})]
        with patch(
            "eval.gemini_embedding.httpx.AsyncClient",
            return_value=FakeAsyncClient(responses, calls),
        ):
            with self.assertRaisesRegex(GeminiEmbeddingError, "dimension mismatch"):
                asyncio.run(self.provider().async_embed_text("bad dimension"))

    def test_batch_size_above_native_limit_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "between 1 and 100"):
            self.provider(batch_size=101)


if __name__ == "__main__":
    unittest.main()
