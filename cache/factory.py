import hashlib
from typing import Iterable, Optional

from config import Config
from embeddings import EmbeddingProvider
from logger import setup_logger
from vector_store.base import VectorRecord

from .decorators import CachingEmbeddingProvider, CachingRAGPipeline
from .redis_store import CacheSettings, RedisCacheStore

logger = setup_logger(__name__)

_DEFAULT_CACHE_STORE: Optional[RedisCacheStore] = None


def cache_settings_from_config() -> CacheSettings:
    """Build cache settings from environment-backed Config values."""
    return CacheSettings(
        enabled=Config.CACHE_ENABLED,
        redis_url=Config.REDIS_URL,
        namespace=Config.CACHE_NAMESPACE,
        embedding_enabled=Config.CACHE_EMBEDDING_ENABLED,
        retrieval_enabled=Config.CACHE_RETRIEVAL_ENABLED,
        answer_enabled=Config.CACHE_ANSWER_ENABLED,
        embedding_ttl=Config.CACHE_EMBEDDING_TTL,
        retrieval_ttl=Config.CACHE_RETRIEVAL_TTL,
        answer_ttl=Config.CACHE_ANSWER_TTL,
        timeout_seconds=Config.CACHE_TIMEOUT,
    )


def get_default_cache_store() -> Optional[RedisCacheStore]:
    """Return the process-wide Redis cache store when caching is enabled."""
    global _DEFAULT_CACHE_STORE
    settings = cache_settings_from_config()
    if not settings.enabled:
        return None
    if _DEFAULT_CACHE_STORE is not None:
        return _DEFAULT_CACHE_STORE

    try:
        _DEFAULT_CACHE_STORE = RedisCacheStore.from_url(settings)
    except ImportError:
        logger.warning("redis package is not installed; cache is disabled fail-open")
        return None
    return _DEFAULT_CACHE_STORE


def reset_default_cache_store() -> None:
    """Reset the process-wide cache store, primarily for tests."""
    global _DEFAULT_CACHE_STORE
    if _DEFAULT_CACHE_STORE is not None:
        _DEFAULT_CACHE_STORE.close()
    _DEFAULT_CACHE_STORE = None


def maybe_wrap_embedding_provider(
    provider: EmbeddingProvider,
    cache_store: Optional[RedisCacheStore] = None,
):
    """Wrap an embedding provider with L1 cache when enabled."""
    selected_store = cache_store or get_default_cache_store()
    if selected_store is None or not selected_store.settings.embedding_enabled:
        return provider
    if isinstance(provider, CachingEmbeddingProvider):
        return provider
    return CachingEmbeddingProvider(provider, selected_store)


def maybe_wrap_pipeline(
    pipeline,
    vector_records: Iterable[VectorRecord],
    cache_store: Optional[RedisCacheStore] = None,
):
    """Wrap a pipeline with L2/L3 cache when enabled."""
    selected_store = cache_store or get_default_cache_store()
    if selected_store is None:
        return pipeline
    if not (selected_store.settings.retrieval_enabled or selected_store.settings.answer_enabled):
        return pipeline
    if isinstance(pipeline, CachingRAGPipeline):
        return pipeline
    return CachingRAGPipeline(
        pipeline=pipeline,
        cache_store=selected_store,
        corpus_identity=build_corpus_identity(pipeline.vector_store, vector_records),
    )


def build_corpus_identity(vector_store, vector_records: Iterable[VectorRecord]) -> str:
    """Return an extra corpus scope to prevent cross-collection key reuse."""
    collection_name = getattr(vector_store, "collection_name", None)
    if collection_name:
        return f"{type(vector_store).__name__}:{collection_name}"

    digest = hashlib.sha256()
    for record in sorted(vector_records, key=lambda item: item.id):
        digest.update(str(record.id).encode("utf-8"))
        digest.update(b"\0")
    return f"{type(vector_store).__name__}:{digest.hexdigest()}"
