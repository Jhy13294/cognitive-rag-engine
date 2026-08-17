from typing import Dict, Iterable, List, Optional

from api_client import run_async_blocking
from config import Config
from embeddings import EmbeddedDocument, EmbeddingProvider
from embeddings.base import Vector
from logger import setup_logger
from rag import RAGResponse, RetrievedSource

from .keys import canonical_metadata_filter, namespaced_key, normalize_question
from .redis_store import RedisCacheStore
from .serialization import (
    decode_vector,
    encode_vector,
    response_from_payload,
    response_to_payload,
    sources_from_payload,
    sources_to_payload,
)

logger = setup_logger(__name__)


class CachingEmbeddingProvider(EmbeddingProvider):
    """EmbeddingProvider decorator that caches single query embeddings."""

    def __init__(self, provider: EmbeddingProvider, cache_store: RedisCacheStore):
        """Initialize the caching decorator."""
        super().__init__(provider.config)
        self.provider = provider
        self.cache_store = cache_store

    def embed_text(self, text: str) -> Vector:
        """Embed a single text string with L1 Redis caching."""
        if not self.cache_store.settings.embedding_enabled:
            return self.provider.embed_text(text)
        return run_async_blocking(self._embed_text_cached(text))

    def embed_texts(self, texts: Iterable[str]) -> List[Vector]:
        """Delegate bulk embedding without caching corpus/document batches."""
        return self.provider.embed_texts(texts)

    def embed_documents(self, documents: Iterable) -> List[EmbeddedDocument]:
        """Delegate document embedding without caching corpus vectors."""
        return self.provider.embed_documents(documents)

    async def _embed_text_cached(self, text: str) -> Vector:
        key = self._cache_key(text)
        cached = await self.cache_store.get_bytes("l1", key)
        if cached is not None:
            try:
                return decode_vector(cached)
            except Exception as e:
                logger.warning("Invalid cached embedding vector; recomputing | error=%s", e)

        vector = self.provider.embed_text(text)
        await self.cache_store.set_bytes(
            "l1",
            key,
            encode_vector(vector),
            self.cache_store.settings.embedding_ttl,
        )
        return list(vector)

    def _cache_key(self, text: str) -> str:
        payload = {
            "model": self.model_name,
            "dimension": self.dimension,
            "normalize": bool(self.config.normalize),
            "question": normalize_question(text),
        }
        return namespaced_key(self.cache_store.settings.namespace, "l1-embedding", payload)


class CachingRAGPipeline:
    """RAGPipeline decorator that adds L2 retrieval and L3 answer caching."""

    def __init__(self, pipeline, cache_store: RedisCacheStore, corpus_identity: str):
        """Wrap an existing pipeline without editing its core implementation."""
        self._pipeline = pipeline
        self._cache_store = cache_store
        self._corpus_identity = corpus_identity
        self._source_retrieve = pipeline.retrieve
        pipeline.retrieve = self.retrieve

    def __getattr__(self, name: str):
        return getattr(self._pipeline, name)

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedSource]:
        """Retrieve sources with L2 Redis caching."""
        if not self._cache_store.settings.retrieval_enabled:
            return self._source_retrieve(question, top_k=top_k, metadata_filter=metadata_filter)
        return run_async_blocking(self._retrieve_cached(question, top_k, metadata_filter))

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> RAGResponse:
        """Generate an answer with L3 Redis caching."""
        if not self._cache_store.settings.answer_enabled:
            return self._pipeline.answer(question, top_k=top_k, metadata_filter=metadata_filter)
        return run_async_blocking(self._answer_cached(question, top_k, metadata_filter))

    def get_cached_answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> Optional[RAGResponse]:
        """Return a cached L3 response when present, without generating."""
        if not self._cache_store.settings.answer_enabled:
            return None
        return run_async_blocking(self._get_cached_answer(question, top_k, metadata_filter))

    def store_answer(
        self,
        response: RAGResponse,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> None:
        """Store a streamed answer response into L3 cache."""
        if not self._cache_store.settings.answer_enabled:
            return
        run_async_blocking(self._store_answer(response, top_k, metadata_filter))

    def cache_stats(self) -> Dict:
        """Expose cache metrics."""
        return self._cache_store.stats()

    async def _retrieve_cached(
        self,
        question: str,
        top_k: Optional[int],
        metadata_filter: Optional[Dict],
    ) -> List[RetrievedSource]:
        version = await self._cache_store.get_corpus_version()
        if version is None:
            return self._source_retrieve(question, top_k=top_k, metadata_filter=metadata_filter)

        key = self._retrieval_key(question, top_k, metadata_filter, version)
        cached_payload = await self._cache_store.get_json("l2", key)
        if cached_payload is not None:
            return sources_from_payload(cached_payload)

        sources = self._source_retrieve(question, top_k=top_k, metadata_filter=metadata_filter)
        await self._cache_store.set_json(
            "l2",
            key,
            sources_to_payload(sources),
            self._cache_store.settings.retrieval_ttl,
        )
        return list(sources)

    async def _answer_cached(
        self,
        question: str,
        top_k: Optional[int],
        metadata_filter: Optional[Dict],
    ) -> RAGResponse:
        cached_response = await self._get_cached_answer(question, top_k, metadata_filter)
        if cached_response is not None:
            return cached_response

        response = self._pipeline.answer(question, top_k=top_k, metadata_filter=metadata_filter)
        await self._store_answer(response, top_k, metadata_filter)
        return response

    async def _get_cached_answer(
        self,
        question: str,
        top_k: Optional[int],
        metadata_filter: Optional[Dict],
    ) -> Optional[RAGResponse]:
        version = await self._cache_store.get_corpus_version()
        if version is None:
            return None
        key = self._answer_key(question, top_k, metadata_filter, version)
        cached_payload = await self._cache_store.get_json("l3", key)
        if cached_payload is None:
            return None
        return response_from_payload(cached_payload)

    async def _store_answer(
        self,
        response: RAGResponse,
        top_k: Optional[int],
        metadata_filter: Optional[Dict],
    ) -> None:
        version = await self._cache_store.get_corpus_version()
        if version is None:
            return
        key = self._answer_key(response.question, top_k, metadata_filter, version)
        await self._cache_store.set_json(
            "l3",
            key,
            response_to_payload(response),
            self._cache_store.settings.answer_ttl,
        )

    def _retrieval_key(
        self,
        question: str,
        top_k: Optional[int],
        metadata_filter: Optional[Dict],
        corpus_version: int,
    ) -> str:
        payload = self._base_key_payload(question, top_k, metadata_filter, corpus_version)
        payload["scope"] = "retrieval"
        return namespaced_key(self._cache_store.settings.namespace, "l2-retrieval", payload)

    def _answer_key(
        self,
        question: str,
        top_k: Optional[int],
        metadata_filter: Optional[Dict],
        corpus_version: int,
    ) -> str:
        payload = self._base_key_payload(question, top_k, metadata_filter, corpus_version)
        payload.update(
            {
                "scope": "answer",
                "system_prompt_hash": namespaced_key(
                    "prompt", "system", {"text": self._pipeline.system_prompt}
                ),
                "chat_model": self._chat_model_name(),
            }
        )
        return namespaced_key(self._cache_store.settings.namespace, "l3-answer", payload)

    def _base_key_payload(
        self,
        question: str,
        top_k: Optional[int],
        metadata_filter: Optional[Dict],
        corpus_version: int,
    ) -> Dict:
        requested_top_k = top_k or self._pipeline.top_k
        return {
            "corpus_identity": self._corpus_identity,
            "corpus_version": corpus_version,
            "top_k": requested_top_k,
            "metadata_filter": canonical_metadata_filter(metadata_filter),
            "question": normalize_question(question),
            "pipeline_config": self._pipeline_config(),
        }

    def _pipeline_config(self) -> Dict:
        embedding_provider = self._pipeline.embedding_provider
        reranker = self._pipeline.reranker
        bm25_retriever = self._pipeline.bm25_retriever
        rrf = self._pipeline.rrf
        query_rewriter = self._pipeline.query_rewriter
        return {
            "embedding_model": embedding_provider.model_name,
            "embedding_dimension": embedding_provider.dimension,
            "embedding_normalize": bool(embedding_provider.config.normalize),
            "vector_store_type": type(self._pipeline.vector_store).__name__,
            "vector_store_collection": getattr(
                self._pipeline.vector_store, "collection_name", None
            ),
            "reranker_model": reranker.model_name if reranker else None,
            "rerank_fetch_k": reranker.fetch_k if reranker else None,
            "rerank_top_n": reranker.top_n if reranker else None,
            "fetch_k": self._pipeline.fetch_k,
            "hybrid_enabled": bm25_retriever is not None,
            "bm25_k1": getattr(bm25_retriever, "k1", None),
            "bm25_b": getattr(bm25_retriever, "b", None),
            "rrf_k": rrf.config.k if rrf else None,
            "rrf_weights": dict(rrf.config.weights) if rrf else None,
            "parent_child_enabled": self._pipeline.parent_store is not None,
            "expand_parent_context": self._pipeline.expand_parent_context,
            "max_context_chars": self._pipeline.max_context_chars,
            "context_packing_enabled": self._pipeline.context_packing_enabled,
            "context_dedup_enabled": self._pipeline.context_dedup_enabled,
            "context_near_dup_enabled": self._pipeline.context_near_dup_enabled,
            "context_near_dup_threshold": self._pipeline.context_near_dup_threshold,
            "context_max_tokens": self._pipeline.context_max_tokens,
            "tokenizer_encoding": self._pipeline.tokenizer_encoding,
            "query_rewrite_enabled": self._pipeline.query_rewrite_enabled,
            "query_rewriter_model": query_rewriter.model_name if query_rewriter else None,
            "query_rewrite_num_queries": self._pipeline.query_rewrite_num_queries,
            "query_rewrite_weight_original": self._pipeline.query_rewrite_weight_original,
            "query_rewrite_weight_variant": self._pipeline.query_rewrite_weight_variant,
        }

    def _chat_model_name(self) -> Dict:
        return {
            "client_type": type(self._pipeline.chat_client).__name__,
            "model": Config.MODEL_NAME,
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
        }
