from typing import Callable, Dict, List, Tuple

from document_loader import load_and_split_documents
from embeddings import HashEmbeddingProvider
from hybrid import BM25Retriever, RRFConfig, RankedRecord, ReciprocalRankFusion
from rag import RAGPipeline, RetrievedSource
from rerank import Reranker
from vector_store import InMemoryVectorStore
from vector_store.base import embedded_document_to_record


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
) -> Tuple[Callable[[str, int], List[RetrievedSource]], Dict]:
    """Build a deterministic offline retriever for evaluation baselines.

    Documents are embedded and inserted once before evaluation starts. The
    returned callable only performs query embedding and vector retrieval.
    """
    chunks = load_and_split_documents(
        knowledge_path,
        recursive=True,
        clean=True,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"No supported documents loaded from knowledge path: {knowledge_path}")

    embedding_provider = HashEmbeddingProvider(dimension=embedding_dimension)
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

    pipeline = RAGPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_client=NoopChatClient(),
        reranker=reranker,
        fetch_k=fetch_k,
        bm25_retriever=bm25_retriever if normalized_mode == "hybrid" else None,
        rrf=rrf,
    )

    def retrieve(question: str, top_k: int) -> List[RetrievedSource]:
        if normalized_mode == "bm25":
            return ranked_records_to_sources(bm25_retriever.retrieve(question, top_k=top_k))
        return pipeline.retrieve(question, top_k=top_k)

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
    }
    return retrieve, metadata


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
