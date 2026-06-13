from typing import Optional

from config import Config

from .base import VectorStore
from .memory_store import InMemoryVectorStore
from .qdrant_store import QdrantVectorStore


def create_vector_store(
    provider_name: Optional[str] = None,
    dimension: Optional[int] = None,
    **overrides,
) -> VectorStore:
    """Create a vector store by provider name."""
    if dimension is None or dimension <= 0:
        raise ValueError("dimension must be greater than 0")

    provider = (provider_name or Config.VECTOR_STORE_PROVIDER).lower()

    if provider == "memory":
        return InMemoryVectorStore(dimension=dimension)

    if provider == "qdrant":
        Config.validate_vector_store(provider)
        return QdrantVectorStore(
            collection_name=_override(overrides, "collection_name", Config.VECTOR_STORE_COLLECTION),
            dimension=dimension,
            host=_override(overrides, "host", Config.VECTOR_STORE_HOST),
            port=_override(overrides, "port", Config.VECTOR_STORE_PORT),
            api_key=_override(overrides, "api_key", Config.VECTOR_STORE_API_KEY),
            url=_override(overrides, "url", Config.VECTOR_STORE_URL),
            distance=_override(overrides, "distance", Config.VECTOR_STORE_DISTANCE_METRIC),
            client=_override(overrides, "client", None),
            recreate=_override(overrides, "recreate", Config.VECTOR_STORE_RECREATE),
            batch_size=_override(overrides, "batch_size", Config.VECTOR_STORE_BATCH_SIZE),
            timeout=_override(overrides, "timeout", Config.VECTOR_STORE_TIMEOUT),
            max_retries=_override(overrides, "max_retries", Config.VECTOR_STORE_MAX_RETRIES),
            base_delay=_override(overrides, "base_delay", Config.VECTOR_STORE_BASE_DELAY),
            max_delay=_override(overrides, "max_delay", Config.VECTOR_STORE_MAX_DELAY),
        )

    raise ValueError(f"Unsupported vector store provider: {provider}")


def _override(overrides: dict, key: str, default):
    """Return an override value when it is explicitly provided."""
    value = overrides.get(key)
    return default if value is None else value
