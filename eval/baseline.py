from typing import Callable, Dict, List, Tuple

from document_loader import load_and_split_documents
from embeddings import HashEmbeddingProvider
from rag import RetrievedSource
from vector_store import InMemoryVectorStore


def build_hash_retriever(
    knowledge_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 80,
    embedding_dimension: int = 64,
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

    def retrieve(question: str, top_k: int) -> List[RetrievedSource]:
        query_embedding = embedding_provider.embed_text(question)
        results = vector_store.similarity_search(query_embedding, top_k=top_k)
        sources = []
        for index, result in enumerate(results, start=1):
            sources.append(
                RetrievedSource(
                    index=index,
                    content=result.content,
                    score=result.score,
                    metadata=dict(result.metadata),
                )
            )
        return sources

    metadata = {
        "embedding_provider": embedding_provider.model_name,
        "embedding_dimension": embedding_provider.dimension,
        "vector_store": "memory",
        "knowledge_path": knowledge_path,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunk_count": len(chunks),
    }
    return retrieve, metadata

