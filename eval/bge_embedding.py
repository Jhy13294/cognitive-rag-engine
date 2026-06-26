import asyncio
import threading
from typing import Iterable, List, Optional

from logger import setup_logger


logger = setup_logger(__name__)


class BGEEmbeddingError(Exception):
    """Raised when the local BGE embedding model cannot produce valid vectors."""


# Process-level cache of loaded fastembed models keyed by (model_name, cache_dir).
# ONNX model load is slow, so every provider instance for the same model reuses
# one TextEmbedding rather than reloading on each ascore call.
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _load_model(model_name: str, cache_dir: Optional[str]):
    """Return a process-cached fastembed TextEmbedding for the given model."""
    cache_key = (model_name, cache_dir)
    model = _MODEL_CACHE.get(cache_key)
    if model is not None:
        return model
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(cache_key)
        if model is not None:
            return model
        try:
            from fastembed import TextEmbedding
        except (ImportError, ModuleNotFoundError) as error:
            raise BGEEmbeddingError(
                "Local BGE embedding requires the optional 'fastembed' dependency. "
                "Install requirements.txt (fastembed pulls onnxruntime, not torch). "
                f"Original error: {error}"
            ) from error
        try:
            model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        except Exception as error:  # fastembed raises ValueError/OSError on bad model
            raise BGEEmbeddingError(
                f"Failed to load local BGE model '{model_name}': {error}"
            ) from error
        _MODEL_CACHE[cache_key] = model
        logger.info(
            "BGEEmbeddingProvider model loaded | model=%s | cache_dir=%s",
            model_name,
            cache_dir or "<default>",
        )
        return model


class BGEEmbeddingProvider:
    """Local BGE provider (fastembed/ONNX) mirroring the project provider interface.

    Exposes the same four methods as GeminiEmbeddingProvider so the live Ragas
    answer-relevance adapter can swap providers without touching metric wiring.
    Inference is local and deterministic, so there is no retry/backoff surface.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        cache_dir: Optional[str] = None,
    ):
        """Initialize and validate the local BGE embedding settings."""
        normalized_model = str(model_name).strip()
        if not normalized_model:
            raise ValueError("BGE embedding model name is required.")
        if dimension <= 0:
            raise ValueError("BGE embedding dimension must be greater than 0.")

        self.model_name = normalized_model
        self.dimension = dimension
        self.cache_dir = cache_dir.strip() if isinstance(cache_dir, str) and cache_dir.strip() else None
        self.request_count = 0
        self.total_inputs = 0
        # Lazy: the ONNX model loads on first embed call, not at construction, so
        # building the judge stays cheap and offline-test-safe when bge is the
        # default provider but no embedding work is performed.
        self._model = None

        logger.info(
            "BGEEmbeddingProvider initialized | model=%s | dimension=%s",
            self.model_name,
            self.dimension,
        )

    def _ensure_model(self):
        """Load and cache the fastembed model on first use (process-level singleton)."""
        if self._model is None:
            self._model = _load_model(self.model_name, self.cache_dir)
        return self._model

    def embed_text(self, text: str) -> List[float]:
        """Embed one query-like text from synchronous code."""
        return self.embed_texts([text])[0]

    async def async_embed_text(self, text: str) -> List[float]:
        """Embed one query-like text asynchronously."""
        return (await self.async_embed_texts([text]))[0]

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed query-like texts locally, preserving input order."""
        text_list = [str(text).strip() for text in texts]
        if not text_list:
            return []
        if any(not text for text in text_list):
            raise ValueError("BGE embedding inputs must not be empty.")

        # fastembed.embed yields one numpy array per input in input order.
        raw_vectors = list(self._ensure_model().embed(text_list))
        if len(raw_vectors) != len(text_list):
            raise BGEEmbeddingError(
                f"BGE embedding count mismatch: expected {len(text_list)}, "
                f"got {len(raw_vectors)}."
            )

        vectors: List[List[float]] = []
        for index, raw in enumerate(raw_vectors):
            numeric_vector = [float(value) for value in raw.tolist()]
            if len(numeric_vector) != self.dimension:
                raise BGEEmbeddingError(
                    f"BGE embedding {index} dimension mismatch: expected "
                    f"{self.dimension}, got {len(numeric_vector)}."
                )
            vectors.append(numeric_vector)

        self.request_count += 1
        self.total_inputs += len(text_list)
        return vectors

    async def async_embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed query-like texts off the event loop via a worker thread."""
        text_list = list(texts)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed_texts, text_list)
