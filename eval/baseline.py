from typing import Callable, Dict, List, Optional, Tuple

from cache import maybe_wrap_embedding_provider, maybe_wrap_pipeline
from document_loader import load_and_split_documents, load_and_split_documents_hierarchical
from embeddings import EmbeddingProvider, HashEmbeddingProvider
from hybrid import BM25Retriever, RankedRecord, ReciprocalRankFusion, RRFConfig
from logger import setup_logger
from parent_store import InMemoryParentStore
from query_rewrite import create_query_rewriter, normalize_query_variants
from rag import RAGPipeline, RelevanceGate, RelevanceGateConfig, RetrievedSource
from rerank import Reranker
from vector_store import InMemoryVectorStore
from vector_store.base import embedded_document_to_record

logger = setup_logger(__name__)


def build_hash_retriever(
    knowledge_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    embedding_dimension: int = 64,
    reranker: Reranker = None,
    fetch_k: int = None,
    retrieval_mode: str = "dense",
    rrf_k: int = 60,
    dense_weight: float = 0.2,
    sparse_weight: float = 1.0,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    parent_child_enabled: bool = False,
    expand_parent_context: bool = False,
    parent_chunk_size: int = 1600,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 80,
    query_rewrite_enabled: bool = False,
    query_rewrite_provider: str = "deterministic",
    query_rewrite_fixture_path: str = "eval/fixtures/query_rewrites.jsonl",
    query_rewrite_num_queries: int = 3,
    query_rewrite_temperature: float = 0.1,
    query_rewrite_cache_enabled: bool = True,
    query_rewrite_weight_original: float = 1.0,
    query_rewrite_weight_variant: float = 0.7,
    cache_enabled: bool = False,
    embedding_provider: Optional[EmbeddingProvider] = None,
    relevance_gate_enabled: bool = False,
    relevance_gate_min_dense_cosine: Optional[float] = None,
    relevance_gate_min_cohere_rerank_score: Optional[float] = None,
    relevance_gate_dense_score_space: Optional[str] = None,
) -> Tuple[Callable[[str, int], List[RetrievedSource]], Dict]:
    """Build a deterministic offline retriever for evaluation baselines.

    Documents are embedded and inserted once before evaluation starts. The
    returned callable only performs query embedding and vector retrieval.
    """
    parent_store = None
    parent_count = 0
    if parent_child_enabled:
        hierarchical = load_and_split_documents_hierarchical(
            knowledge_path,
            recursive=True,
            clean=True,
            parent_chunk_size=parent_chunk_size,
            parent_chunk_overlap=parent_chunk_overlap,
            child_chunk_size=child_chunk_size,
            child_chunk_overlap=child_chunk_overlap,
        )
        chunks = hierarchical.children
        parent_count = len(hierarchical.parents)
        if expand_parent_context:
            parent_store = InMemoryParentStore()
            parent_store.add_parents(hierarchical.parents)
    else:
        chunks = load_and_split_documents(
            knowledge_path,
            recursive=True,
            clean=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    if not chunks:
        raise ValueError(f"No supported documents loaded from knowledge path: {knowledge_path}")

    embedding_provider = embedding_provider or HashEmbeddingProvider(dimension=embedding_dimension)
    if cache_enabled:
        embedding_provider = maybe_wrap_embedding_provider(embedding_provider)
    embedded_documents = embedding_provider.embed_documents(chunks)
    vector_records = [embedded_document_to_record(document) for document in embedded_documents]
    vector_store = InMemoryVectorStore(dimension=embedding_provider.dimension)
    vector_store.add_records(vector_records)
    bm25_retriever = None
    rrf = None
    normalized_mode = retrieval_mode.lower()

    if normalized_mode not in {"dense", "bm25", "hybrid"}:
        raise ValueError("retrieval_mode must be one of: dense, bm25, hybrid")

    if normalized_mode in {"bm25", "hybrid"}:
        bm25_retriever = BM25Retriever(vector_records, k1=bm25_k1, b=bm25_b)

    if normalized_mode == "hybrid":
        rrf = ReciprocalRankFusion(
            RRFConfig(
                k=rrf_k,
                weights={"dense": dense_weight, "sparse": sparse_weight},
            )
        )

    query_rewriter = create_query_rewriter(
        provider_name=query_rewrite_provider,
        enabled=query_rewrite_enabled,
        num_queries=query_rewrite_num_queries,
        temperature=query_rewrite_temperature,
        cache_enabled=query_rewrite_cache_enabled,
        weight_original=query_rewrite_weight_original,
        weight_variant=query_rewrite_weight_variant,
        fixture_path=query_rewrite_fixture_path,
    )

    pipeline = RAGPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_client=NoopChatClient(),
        reranker=reranker,
        fetch_k=fetch_k,
        bm25_retriever=bm25_retriever if normalized_mode == "hybrid" else None,
        rrf=rrf,
        parent_store=parent_store,
        expand_parent_context=expand_parent_context,
        query_rewriter=query_rewriter,
        query_rewrite_enabled=query_rewrite_enabled,
        query_rewrite_num_queries=query_rewrite_num_queries,
        query_rewrite_weight_original=query_rewrite_weight_original,
        query_rewrite_weight_variant=query_rewrite_weight_variant,
        relevance_gate_enabled=relevance_gate_enabled,
        relevance_gate_min_dense_cosine=relevance_gate_min_dense_cosine,
        relevance_gate_min_cohere_rerank_score=relevance_gate_min_cohere_rerank_score,
        relevance_gate_dense_score_space=relevance_gate_dense_score_space,
    )
    if cache_enabled:
        pipeline = maybe_wrap_pipeline(pipeline, vector_records)

    bm25_gate = RelevanceGate(
        RelevanceGateConfig(
            enabled=relevance_gate_enabled,
            min_dense_cosine=relevance_gate_min_dense_cosine,
            min_cohere_rerank_score=relevance_gate_min_cohere_rerank_score,
            dense_score_space="bm25",
            query_rewrite_enabled=query_rewrite_enabled,
        )
    )

    def retrieve(question: str, top_k: int) -> List[RetrievedSource]:
        if normalized_mode == "bm25":
            if query_rewriter is not None:
                multi_query_sources = retrieve_bm25_multi_query(
                    question=question,
                    top_k=top_k,
                    bm25_retriever=bm25_retriever,
                    query_rewriter=query_rewriter,
                    num_queries=query_rewrite_num_queries,
                    rrf_k=rrf_k,
                    weight_original=query_rewrite_weight_original,
                    weight_variant=query_rewrite_weight_variant,
                )
                if multi_query_sources is not None:
                    return bm25_gate.apply(multi_query_sources)
            sources = ranked_records_to_sources(bm25_retriever.retrieve(question, top_k=top_k))
            return bm25_gate.apply(sources)
        return pipeline.retrieve(question, top_k=top_k)

    retrieve.relevance_gate_stats = (
        bm25_gate.stats if normalized_mode == "bm25" else pipeline.relevance_gate_stats
    )

    metadata = {
        "embedding_provider": embedding_provider.model_name,
        "embedding_dimension": embedding_provider.dimension,
        "vector_store": "memory",
        "retrieval_mode": normalized_mode,
        "hybrid_enabled": normalized_mode == "hybrid",
        "rrf_k": rrf_k if normalized_mode == "hybrid" else None,
        "hybrid_dense_weight": dense_weight if normalized_mode == "hybrid" else None,
        "hybrid_sparse_weight": sparse_weight if normalized_mode == "hybrid" else None,
        "bm25_k1": bm25_k1 if normalized_mode in {"bm25", "hybrid"} else None,
        "bm25_b": bm25_b if normalized_mode in {"bm25", "hybrid"} else None,
        "reranker": reranker.model_name if reranker else None,
        "rerank_fetch_k": fetch_k or (reranker.fetch_k if reranker else None),
        "rerank_top_n": reranker.top_n if reranker else None,
        "knowledge_path": knowledge_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunks),
        "record_count": len(vector_records),
        "cache_enabled": cache_enabled,
    }
    if relevance_gate_enabled:
        metadata.update(
            {
                "relevance_gate_enabled": True,
                "relevance_gate_min_dense_cosine": relevance_gate_min_dense_cosine,
                "relevance_gate_min_cohere_rerank_score": (relevance_gate_min_cohere_rerank_score),
                "relevance_gate_dense_score_space": (
                    pipeline.relevance_gate.config.dense_score_space
                ),
            }
        )
    if query_rewrite_enabled:
        metadata.update(
            {
                "query_rewrite_enabled": True,
                "query_rewrite_provider": query_rewrite_provider,
                "query_rewrite_fixture_path": query_rewrite_fixture_path,
                "query_rewrite_num_queries": query_rewrite_num_queries,
                "query_rewrite_temperature": query_rewrite_temperature,
                "query_rewrite_cache_enabled": query_rewrite_cache_enabled,
                "query_rewrite_weight_original": query_rewrite_weight_original,
                "query_rewrite_weight_variant": query_rewrite_weight_variant,
            }
        )
    if parent_child_enabled:
        metadata.update(
            {
                "parent_child_enabled": True,
                "expand_parent_context": expand_parent_context,
                "parent_chunk_size": parent_chunk_size,
                "parent_chunk_overlap": parent_chunk_overlap,
                "child_chunk_size": child_chunk_size,
                "child_chunk_overlap": child_chunk_overlap,
                "parent_count": parent_count,
            }
        )
    return retrieve, metadata


def retrieve_bm25_multi_query(
    question: str,
    top_k: int,
    bm25_retriever: BM25Retriever,
    query_rewriter,
    num_queries: int,
    rrf_k: int,
    weight_original: float,
    weight_variant: float,
) -> Optional[List[RetrievedSource]]:
    """Run multi-query retrieval for a BM25-only evaluation path."""
    try:
        variants = normalize_query_variants(question, query_rewriter.rewrite(question), num_queries)
    except Exception as e:
        logger.warning("BM25 query rewrite failed; falling back to single query | error=%s", e)
        return None

    if len(variants) <= 1:
        return None

    ranked_lists = {}
    for index, variant in enumerate(variants):
        ranked_lists[f"q{index}"] = bm25_retriever.retrieve(variant, top_k=top_k)

    weights = {"q0": weight_original}
    for index in range(1, len(variants)):
        weights[f"q{index}"] = weight_variant

    fused_records = ReciprocalRankFusion(RRFConfig(k=rrf_k, weights=weights)).fuse(
        ranked_lists, top_k=top_k
    )
    sources = ranked_records_to_sources(fused_records)
    for source in sources:
        source.metadata["query_rewrite_enabled"] = True
        source.metadata["query_rewrite_variants"] = list(variants)
        source.metadata["query_rewrite_weights"] = dict(weights)
    return sources


def ranked_records_to_sources(records: List[RankedRecord]) -> List[RetrievedSource]:
    """Convert BM25 ranked records to RAG sources."""
    sources = []
    for index, record in enumerate(records, start=1):
        sources.append(
            RetrievedSource(
                index=index,
                content=record.content,
                score=record.score,
                metadata=dict(record.metadata),
            )
        )
    return sources


class NoopChatClient:
    """Chat client placeholder that makes accidental generation obvious."""

    def chat(self, message: str, system_prompt: str = None) -> Dict:
        """Raise if evaluation accidentally calls generation."""
        raise RuntimeError("Retrieval evaluation must not call chat generation")
