import random
import time
from typing import Dict, Iterable, List, Optional

import requests

from config import Config
from document_loader import Document
from logger import mask_sensitive_info, setup_logger
from tokenization import HeuristicTokenCounter, TokenCounter, create_token_counter

from .base import EmbeddedDocument, EmbeddingConfig, EmbeddingProvider, Vector

logger = setup_logger(__name__)


class OpenAIEmbeddingError(Exception):
    """Raised when OpenAI embedding generation fails."""

    def __init__(self, message: str, status_code: int = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable

    def __str__(self):
        if self.status_code:
            return f"[HTTP {self.status_code}] {super().__str__()}"
        return super().__str__()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Production embedding provider for OpenAI text-embedding-3 models."""

    RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: Optional[int] = None,
        max_batch_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        user: Optional[str] = None,
        token_counter: Optional[TokenCounter] = None,
        tokenizer_encoding: Optional[str] = None,
    ):
        """Initialize the OpenAI embedding provider."""
        self.api_key = api_key or Config.EMBEDDING_API_KEY
        self.api_url = api_url or Config.EMBEDDING_API_URL
        self.model_name_value = model_name or Config.EMBEDDING_MODEL_NAME
        self.dimensions = dimensions if dimensions is not None else Config.EMBEDDING_DIMENSION
        self.batch_size = batch_size or Config.EMBEDDING_BATCH_SIZE
        self.max_batch_tokens = max_batch_tokens or Config.EMBEDDING_MAX_BATCH_TOKENS
        self.timeout = timeout if timeout is not None else Config.EMBEDDING_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else Config.EMBEDDING_MAX_RETRIES
        self.base_delay = base_delay if base_delay is not None else Config.EMBEDDING_BASE_DELAY
        self.max_delay = max_delay if max_delay is not None else Config.EMBEDDING_MAX_DELAY
        self.user = user if user is not None else Config.EMBEDDING_USER
        self.tokenizer_encoding = tokenizer_encoding or Config.TOKENIZER_ENCODING
        self.token_counter = token_counter or create_token_counter(self.tokenizer_encoding)

        if not self.api_key:
            raise ValueError("OpenAI embedding API key is required")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be greater than 0")

        dimension = self.dimensions or default_openai_embedding_dimension(self.model_name_value)
        super().__init__(
            EmbeddingConfig(
                model_name=self.model_name_value,
                dimension=dimension,
                normalize=False,
                batch_size=self.batch_size,
                metadata={
                    "provider": "openai",
                    "api_url": self.api_url,
                    "dimensions": self.dimensions,
                },
            )
        )

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.total_prompt_tokens = 0
        self.total_tokens = 0
        self.request_count = 0

        logger.info(
            "OpenAIEmbeddingProvider initialized | model=%s | dimensions=%s | batch_size=%s | api_key=%s",
            self.model_name,
            self.dimensions or self.dimension,
            self.batch_size,
            mask_sensitive_info(self.api_key),
        )

    def embed_text(self, text: str) -> Vector:
        """Embed a single text string."""
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> List[Vector]:
        """Embed multiple text strings with batching and retry handling."""
        text_list = [str(text) for text in texts]
        if not text_list:
            return []

        vectors = []
        for batch in self._build_batches(text_list):
            vectors.extend(self._embed_batch(batch))

        return vectors

    def embed_documents(self, documents: List[Document]) -> List[EmbeddedDocument]:
        """Embed Document objects in batches while preserving metadata."""
        document_list = list(documents)
        vectors = self.embed_texts(document.content for document in document_list)
        embedded_documents = []

        for document, vector in zip(document_list, vectors):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "embedding_provider": "openai",
                    "embedding_model": self.model_name,
                    "embedding_dimension": self.dimension,
                    "embedding_dimensions_parameter": self.dimensions,
                    "embedding_normalized": self.config.normalize,
                }
            )
            embedded_documents.append(
                EmbeddedDocument(
                    content=document.content,
                    metadata=metadata,
                    embedding=vector,
                )
            )

        return embedded_documents

    def _embed_batch(self, texts: List[str]) -> List[Vector]:
        """Embed one batch of texts."""
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions
        if self.user:
            payload["user"] = self.user

        start_time = time.time()
        response_data = self._post_with_retries(payload)
        elapsed = time.time() - start_time

        vectors = self._extract_vectors(response_data, expected_count=len(texts))
        usage = response_data.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)

        self.request_count += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_tokens += total_tokens

        logger.info(
            "OpenAI embeddings created | batch_size=%s | model=%s | prompt_tokens=%s | total_tokens=%s | elapsed=%.2fs",
            len(texts),
            response_data.get("model", self.model_name),
            prompt_tokens,
            total_tokens,
            elapsed,
        )
        return vectors

    def _post_with_retries(self, payload: Dict) -> Dict:
        """Post an embedding request with retry handling."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return response.json()

                retryable = response.status_code in self.RETRYABLE_STATUS_CODES
                error = OpenAIEmbeddingError(
                    f"OpenAI embedding request failed: {response.text[:300]}",
                    status_code=response.status_code,
                    retryable=retryable,
                )

                if retryable and attempt < self.max_retries:
                    self._sleep_before_retry(attempt, response)
                    last_error = error
                    continue

                logger.error("OpenAI embedding request failed | error=%s", error)
                raise error

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = OpenAIEmbeddingError(f"OpenAI embedding network error: {e}", retryable=True)
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise last_error

            except requests.exceptions.RequestException as e:
                last_error = OpenAIEmbeddingError(f"OpenAI embedding request error: {e}", retryable=True)
                if attempt < self.max_retries:
                    self._sleep_before_retry(attempt)
                    continue
                raise last_error

        raise last_error

    def _sleep_before_retry(self, attempt: int, response=None) -> None:
        """Sleep before retrying a failed request."""
        retry_after = None
        if response is not None:
            retry_after = response.headers.get("Retry-After")

        if retry_after:
            delay = float(retry_after)
        else:
            delay = min(self.base_delay * (2 ** attempt), self.max_delay)
            delay += random.uniform(0, delay * 0.1)

        logger.warning("Retrying OpenAI embedding request | attempt=%s | delay=%.2fs", attempt + 1, delay)
        time.sleep(delay)

    def _extract_vectors(self, response_data: Dict, expected_count: int) -> List[Vector]:
        """Extract vectors from an OpenAI embeddings response."""
        data = response_data.get("data")
        if not isinstance(data, list):
            raise OpenAIEmbeddingError("Invalid OpenAI embeddings response: missing data list")

        if len(data) != expected_count:
            raise OpenAIEmbeddingError(
                f"Invalid OpenAI embeddings response: expected {expected_count} vectors, got {len(data)}"
            )

        vectors = []
        for item in sorted(data, key=lambda value: value.get("index", 0)):
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise OpenAIEmbeddingError("Invalid OpenAI embeddings response: embedding is not a list")
            self._validate_vector(vector)
            vectors.append(vector)

        return vectors

    def _build_batches(self, texts: List[str]) -> List[List[str]]:
        """Build request batches by item count and token count."""
        batches = []
        current_batch = []
        current_tokens = 0

        for text in texts:
            token_count = estimate_token_count(text, token_counter=self.token_counter)
            if token_count > self.max_batch_tokens:
                raise OpenAIEmbeddingError(
                    "Single embedding input exceeds max_batch_tokens; split the document into smaller chunks."
                )
            if current_batch and (
                len(current_batch) >= self.batch_size
                or current_tokens + token_count > self.max_batch_tokens
            ):
                batches.append(current_batch)
                current_batch = []
                current_tokens = 0

            current_batch.append(text)
            current_tokens += token_count

        if current_batch:
            batches.append(current_batch)

        return batches


def estimate_token_count(text: str, token_counter: Optional[TokenCounter] = None) -> int:
    """Estimate token count through an injectable counter."""
    counter = token_counter or HeuristicTokenCounter()
    return counter.count(text)


def default_openai_embedding_dimension(model_name: str) -> int:
    """Return the native dimension for common OpenAI embedding models."""
    if model_name == "text-embedding-3-large":
        return 3072
    return 1536
