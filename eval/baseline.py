from typing import Callable, Dict, List, Tuple

from document_loader import load_and_split_documents
from embeddings import HashEmbeddingProvider
from rag import RAGPipeline, RetrievedSource
from rerank import Reranker
from vector_store import InMemoryVectorStore


def build_hash_retriever(
    knowledge_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    embedding_dimension: int = 64,
    reranker: Reranker = None,
    fetch_k: int = None,
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
    vector_store = InMemoryVectorStore(dimension=embedding_provider.dimension)
    vector_store.add_documents(embedded_documents)
    pipeline = RAGPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_client=NoopChatClient(),
        reranker=reranker,
        fetch_k=fetch_k,
    )

    def retrieve(question: str, top_k: int) -> List[RetrievedSource]:
        return pipeline.retrieve(question, top_k=top_k)

    metadata = {
        "embedding_provider": embedding_provider.model_name,
        "embedding_dimension": embedding_provider.dimension,
        "vector_store": "memory",
        "reranker": reranker.model_name if reranker else None,
        "rerank_fetch_k": fetch_k or (reranker.fetch_k if reranker else None),
        "rerank_top_n": reranker.top_n if reranker else None,
        "knowledge_path": knowledge_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunks),
    }
    return retrieve, metadata


class NoopChatClient:
    """Chat client placeholder that makes accidental generation obvious."""

    def chat(self, message: str, system_prompt: str = None) -> Dict:
        """Raise if evaluation accidentally calls generation."""
        raise RuntimeError("Retrieval evaluation must not call chat generation")
