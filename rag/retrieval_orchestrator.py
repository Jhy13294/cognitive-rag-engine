"""Retrieval orchestration for the RAG pipeline.

Owns the candidate-producing half of the retrieval funnel: single-query
dense retrieval, hybrid dense + BM25 retrieval with RRF fusion, multi-query
rewriting with cross-query fusion, and rerank with fallback to the fused
order. It returns reranked, top-k-truncated candidates; parent expansion and
other post-selection steps belong to the source finalizer.
"""

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from hybrid import RankedRecord, ReciprocalRankFusion, RRFConfig
from logger import setup_logger
from query_rewrite import normalize_query_variants
from vector_store import SearchResult

from .models import RetrievedSource

if TYPE_CHECKING:
    from .pipeline import RAGPipeline

logger = setup_logger(__name__)


class RetrievalOrchestrator:
    """Produce reranked retrieval candidates for a question.

    The orchestrator reads retrieval configuration and collaborators from the
    owning pipeline at call time, so runtime reconfiguration and cache
    wrappers keep working exactly as they do against the pipeline itself.
    """

    def __init__(self, pipeline: "RAGPipeline"):
        self._pipeline = pipeline

    def retrieve(
        self,
        question: str,
        requested_top_k: int,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedSource]:
        """Return reranked candidates, trying multi-query retrieval first."""
        if self._pipeline.query_rewrite_enabled:
            multi_query_sources = self._try_multi_query_retrieve(
                question=question,
                requested_top_k=requested_top_k,
                metadata_filter=metadata_filter,
            )
            if multi_query_sources is not None:
                return multi_query_sources

        return self._retrieve_single_query(
            question=question,
            requested_top_k=requested_top_k,
            metadata_filter=metadata_filter,
        )

    def _retrieve_single_query(
        self,
        question: str,
        requested_top_k: int,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedSource]:
        """Run the single-query retrieval path."""
        sources = self._retrieve_single_query_candidates(
            question=question,
            requested_top_k=requested_top_k,
            metadata_filter=metadata_filter,
        )
        fallback_label = "hybrid" if self._pipeline.bm25_retriever is not None else "dense"
        return self._rerank_or_truncate(
            question=question,
            sources=sources,
            requested_top_k=requested_top_k,
            fallback_label=fallback_label,
        )

    def _retrieve_single_query_candidates(
        self,
        question: str,
        requested_top_k: int,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedSource]:
        """Return dense or hybrid candidates before rerank and parent expansion."""
        pipeline = self._pipeline
        query_embedding = pipeline.embedding_provider.embed_text(question)

        if pipeline.bm25_retriever is not None:
            return self._hybrid_retrieve(
                question=question,
                query_embedding=query_embedding,
                requested_top_k=requested_top_k,
                metadata_filter=metadata_filter,
            )

        search_results = pipeline.vector_store.similarity_search(
            query_embedding,
            top_k=self._search_top_k(requested_top_k),
            metadata_filter=metadata_filter,
        )
        return self._to_sources(search_results)

    def _rerank_or_truncate(
        self,
        question: str,
        sources: List[RetrievedSource],
        requested_top_k: int,
        fallback_label: str,
    ) -> List[RetrievedSource]:
        """Apply rerank when configured, falling back to the incoming order."""
        if self._pipeline.reranker is None:
            return sources[:requested_top_k]

        try:
            return self._rerank_sources(question, sources, requested_top_k)
        except Exception as e:
            logger.warning(
                "Rerank failed; falling back to %s order | error=%s | requested_top_k=%s",
                fallback_label,
                e,
                requested_top_k,
            )
            return sources[:requested_top_k]

    def _try_multi_query_retrieve(
        self,
        question: str,
        requested_top_k: int,
        metadata_filter: Optional[Dict] = None,
    ) -> Optional[List[RetrievedSource]]:
        """Run multi-query retrieval, returning None when it should be bypassed."""
        pipeline = self._pipeline
        if pipeline.query_rewriter is None:
            return None

        try:
            raw_variants = pipeline.query_rewriter.rewrite(question)
        except Exception as e:
            logger.warning("Query rewrite failed; falling back to single query | error=%s", e)
            return None

        variants = normalize_query_variants(
            question, raw_variants, pipeline.query_rewrite_num_queries
        )
        if not variants:
            logger.warning(
                "Query rewrite returned no variants; falling back to single query | question=%s",
                question,
            )
            return None
        if len(variants) <= 1:
            return None

        logger.info("Multi-query retrieval enabled | original=%s | variants=%s", question, variants)
        candidate_count = self._multi_query_candidate_count(requested_top_k, len(variants))
        ranked_lists = {}

        for path_name, variant in self._query_variant_paths(variants):
            candidates = self._retrieve_single_query_candidates(
                question=variant,
                requested_top_k=candidate_count,
                metadata_filter=metadata_filter,
            )
            ranked_lists[path_name] = retrieved_sources_to_ranked_records(candidates)

        weights = self._query_variant_weights(len(variants))
        fusion = ReciprocalRankFusion(
            RRFConfig(
                k=pipeline.rrf.config.k,
                weights=weights,
            )
        )
        fusion_top_k = candidate_count if pipeline.reranker is not None else requested_top_k
        fused_records = fusion.fuse(ranked_lists, top_k=fusion_top_k)
        fused_sources = self._ranked_records_to_sources(fused_records)

        for source in fused_sources:
            source.metadata["query_rewrite_enabled"] = True
            source.metadata["query_rewrite_variants"] = list(variants)
            source.metadata["query_rewrite_weights"] = dict(weights)

        return self._rerank_or_truncate(
            question=question,
            sources=fused_sources,
            requested_top_k=requested_top_k,
            fallback_label="multi-query fused",
        )

    def _multi_query_candidate_count(self, requested_top_k: int, variant_count: int) -> int:
        """Return per-variant candidate count for multi-query retrieval."""
        base_count = self._search_top_k(requested_top_k)
        if self._pipeline.reranker is None:
            return max(base_count, requested_top_k)
        return max(base_count, requested_top_k * variant_count)

    def _query_variant_paths(self, variants: List[str]) -> List[Tuple[str, str]]:
        """Return explicit path names for query variants."""
        return [(f"q{index}", variant) for index, variant in enumerate(variants)]

    def _query_variant_weights(self, variant_count: int) -> Dict[str, float]:
        """Return explicit RRF weights for every query variant path."""
        weights = {"q0": self._pipeline.query_rewrite_weight_original}
        for index in range(1, variant_count):
            weights[f"q{index}"] = self._pipeline.query_rewrite_weight_variant
        return weights

    def _search_top_k(self, requested_top_k: int) -> int:
        """Return dense candidate count for retrieval."""
        pipeline = self._pipeline
        if pipeline.reranker is None and pipeline.bm25_retriever is None:
            return requested_top_k
        default_fetch_k = (
            pipeline.reranker.fetch_k if pipeline.reranker is not None else requested_top_k
        )
        return max(pipeline.fetch_k or default_fetch_k, requested_top_k)

    def _hybrid_retrieve(
        self,
        question: str,
        query_embedding: List[float],
        requested_top_k: int,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedSource]:
        """Retrieve with dense and BM25 paths, then fuse by RRF."""
        pipeline = self._pipeline
        candidate_count = self._search_top_k(requested_top_k)
        dense_results = pipeline.vector_store.similarity_search(
            query_embedding,
            top_k=candidate_count,
            metadata_filter=metadata_filter,
        )
        sparse_results = pipeline.bm25_retriever.retrieve(
            question,
            top_k=candidate_count,
            metadata_filter=metadata_filter,
        )
        fused_records = pipeline.rrf.fuse(
            {
                "dense": dense_search_results_to_ranked_records(dense_results),
                "sparse": sparse_results,
            },
            top_k=candidate_count,
        )
        return self._ranked_records_to_sources(fused_records)

    def _rerank_sources(
        self,
        question: str,
        sources: List[RetrievedSource],
        requested_top_k: int,
    ) -> List[RetrievedSource]:
        """Rerank dense sources and preserve dense retrieval observability."""
        reranker = self._pipeline.reranker
        reranked = reranker.rerank(question, sources, top_n=requested_top_k)
        if not reranked:
            return sources[:requested_top_k]

        reranked_sources = []
        for new_index, result in enumerate(reranked[:requested_top_k], start=1):
            if result.index < 0 or result.index >= len(sources):
                continue

            dense_source = sources[result.index]
            metadata = dict(dense_source.metadata)
            metadata.update(result.metadata)
            metadata.setdefault("dense_score", dense_source.score)
            metadata.setdefault("dense_rank", dense_source.index)
            metadata["rerank_model"] = reranker.model_name

            reranked_sources.append(
                RetrievedSource(
                    index=new_index,
                    content=result.content or dense_source.content,
                    score=float(result.score),
                    metadata=metadata,
                )
            )

        if not reranked_sources:
            return sources[:requested_top_k]
        return reranked_sources

    def _to_sources(self, search_results: List[SearchResult]) -> List[RetrievedSource]:
        """Convert vector search results to RAG sources."""
        sources = []
        for index, result in enumerate(search_results, start=1):
            sources.append(
                RetrievedSource(
                    index=index,
                    content=result.content,
                    score=result.score,
                    metadata=dict(result.metadata),
                )
            )
        return sources

    def _ranked_records_to_sources(
        self, ranked_records: List[RankedRecord]
    ) -> List[RetrievedSource]:
        """Convert fused ranked records to RAG sources."""
        sources = []
        for index, record in enumerate(ranked_records, start=1):
            sources.append(
                RetrievedSource(
                    index=index,
                    content=record.content,
                    score=record.score,
                    metadata=dict(record.metadata),
                )
            )
        return sources


def dense_search_results_to_ranked_records(
    search_results: List[SearchResult],
) -> List[RankedRecord]:
    """Convert dense vector search results to stable ranked records."""
    ranked_records = []
    for result in search_results:
        metadata = dict(result.metadata)
        metadata["id"] = result.record.id
        metadata["retrieval_mode"] = "dense"
        metadata["dense_score"] = result.score
        ranked_records.append(
            RankedRecord(
                id=result.record.id,
                score=result.score,
                content=result.content,
                metadata=metadata,
                record=result.record,
            )
        )
    return ranked_records


def retrieved_sources_to_ranked_records(sources: List[RetrievedSource]) -> List[RankedRecord]:
    """Convert retrieved sources to stable ranked records for cross-query RRF."""
    ranked_records = []
    for source in sources:
        metadata = dict(source.metadata)
        record_id = str(metadata.get("id") or metadata.get("child_id") or "")
        if not record_id:
            continue
        ranked_records.append(
            RankedRecord(
                id=record_id,
                score=source.score,
                content=source.content,
                metadata=metadata,
            )
        )
    return ranked_records
