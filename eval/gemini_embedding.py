import asyncio
import random
import re
import time
from typing import Dict, Iterable, List, Optional

import httpx

from api_client import run_async_blocking
from logger import mask_sensitive_info, setup_logger


logger = setup_logger(__name__)


class GeminiEmbeddingError(Exception):
    """Raised when the Gemini embedding endpoint cannot produce valid vectors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class GeminiEmbeddingProvider:
    """Async batched provider for Gemini native batchEmbedContents."""

    RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}
    MAX_BATCH_SIZE = 100

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_name: str = "gemini-embedding-001",
        dimension: int = 3072,
        batch_size: int = 100,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
    ):
        """Initialize and validate the Gemini embedding client settings."""
        if not api_key:
            raise ValueError("Gemini embedding API key is required.")
        if not base_url:
            raise ValueError("Gemini embedding base URL is required.")
        normalized_model = str(model_name).removeprefix("models/").strip()
        if not normalized_model or re.fullmatch(r"[A-Za-z0-9._-]+", normalized_model) is None:
            raise ValueError("Gemini embedding model name contains unsupported characters.")
        if dimension <= 0:
            raise ValueError("Gemini embedding dimension must be greater than 0.")
        if batch_size <= 0 or batch_size > self.MAX_BATCH_SIZE:
            raise ValueError(f"Gemini embedding batch_size must be between 1 and {self.MAX_BATCH_SIZE}.")
        if timeout <= 0:
            raise ValueError("Gemini embedding timeout must be greater than 0.")
        if max_retries < 0:
            raise ValueError("Gemini embedding max_retries cannot be negative.")
        if base_delay < 0 or max_delay < base_delay:
            raise ValueError("Gemini embedding retry delays are invalid.")

        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_name = normalized_model
        self.dimension = dimension
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.request_count = 0
        self.total_inputs = 0

        logger.info(
            "GeminiEmbeddingProvider initialized | model=%s | dimension=%s | batch_size=%s | api_key=%s",
            self.model_name,
            self.dimension,
            self.batch_size,
            mask_sensitive_info(self.api_key),
        )

    def embed_text(self, text: str) -> List[float]:
        """Embed one query-like text from synchronous code."""
        return run_async_blocking(self.async_embed_text(text))

    async def async_embed_text(self, text: str) -> List[float]:
        """Embed one query-like text asynchronously."""
        return (await self.async_embed_texts([text]))[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed query-like texts in bounded Gemini batches from synchronous code."""
        return run_async_blocking(self.async_embed_texts(texts))

    async def async_embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed query-like texts with native Gemini batch requests."""
        text_list = [str(text).strip() for text in texts]
        if not text_list:
            return []
        if any(not text for text in text_list):
            raise ValueError("Gemini embedding inputs must not be empty.")

        vectors = []
        for start in range(0, len(text_list), self.batch_size):
            vectors.extend(await self._embed_batch(text_list[start : start + self.batch_size]))
        return vectors

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Send one batchEmbedContents request and preserve request order."""
        model_path = f"models/{self.model_name}"
        payload = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": text}]},
                    "taskType": "RETRIEVAL_QUERY",
                }
                for text in texts
            ]
        }
        started = time.monotonic()
        response_data = await self._post_with_retries(payload)
        embeddings = response_data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise GeminiEmbeddingError(
                f"Invalid Gemini batch response: expected {len(texts)} embeddings, "
                f"got {len(embeddings) if isinstance(embeddings, list) else 'none'}."
            )

        vectors = []
        for index, embedding in enumerate(embeddings):
            vector = embedding.get("values") if isinstance(embedding, dict) else None
            if not isinstance(vector, list):
                raise GeminiEmbeddingError(f"Gemini embedding {index} is missing values.")
            numeric_vector = [float(value) for value in vector]
            if len(numeric_vector) != self.dimension:
                raise GeminiEmbeddingError(
                    f"Gemini embedding dimension mismatch: expected {self.dimension}, "
                    f"got {len(numeric_vector)}."
                )
            vectors.append(numeric_vector)

        self.request_count += 1
        self.total_inputs += len(texts)
        logger.info(
            "Gemini embeddings created | model=%s | batch_size=%s | elapsed=%.2fs",
            self.model_name,
            len(texts),
            time.monotonic() - started,
        )
        return vectors

    async def _post_with_retries(self, payload: Dict) -> Dict:
        """Call Gemini with bounded exponential backoff and typed failures."""
        endpoint = f"{self.base_url}/models/{self.model_name}:batchEmbedContents"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key,
        }
        last_error = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await client.post(endpoint, headers=headers, json=payload)
                    if response.status_code == 200:
                        return response.json()

                    retryable = response.status_code in self.RETRYABLE_STATUS_CODES
                    last_error = GeminiEmbeddingError(
                        f"Gemini embedding request failed: {response.text[:300]}",
                        status_code=response.status_code,
                        retryable=retryable,
                    )
                    if retryable and attempt < self.max_retries:
                        await self._sleep_before_retry(attempt, response.headers.get("Retry-After"))
                        continue
                    raise last_error
                except (httpx.TimeoutException, httpx.RequestError) as error:
                    last_error = GeminiEmbeddingError(
                        f"Gemini embedding network failure: {error}",
                        retryable=True,
                    )
                    if attempt < self.max_retries:
                        await self._sleep_before_retry(attempt)
                        continue
                    raise last_error from None
        raise last_error

    async def _sleep_before_retry(self, attempt: int, retry_after: Optional[str] = None) -> None:
        """Wait asynchronously before a retry without blocking the event loop."""
        if retry_after:
            delay = float(retry_after)
        else:
            delay = min(self.base_delay * (2**attempt), self.max_delay)
            delay += random.uniform(0, delay * 0.1)
        logger.warning(
            "Retrying Gemini embedding request | attempt=%s | delay=%.2fs",
            attempt + 1,
            delay,
        )
        await asyncio.sleep(delay)
