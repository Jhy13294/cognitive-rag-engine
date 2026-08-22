from typing import Iterable, List, Optional

from document_loader import Document
from embeddings import EmbeddedDocument, EmbeddingConfig, EmbeddingProvider

from .bge_embedding import BGEEmbeddingProvider

BGE_RETRIEVAL_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGERetrievalEmbeddingProvider(EmbeddingProvider):
    """Eval-only BGE adapter that keeps query and passage encoding distinct."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        dimension: int = 384,
        cache_dir: Optional[str] = None,
        provider=None,
    ) -> None:
        super().__init__(
            EmbeddingConfig(
                model_name=model_name,
                dimension=dimension,
                normalize=True,
                metadata={
                    "provider": "bge_eval",
                    "query_prefix": BGE_RETRIEVAL_QUERY_PREFIX,
                },
            )
        )
        self.provider = provider or BGEEmbeddingProvider(
            model_name=model_name,
            dimension=dimension,
            cache_dir=cache_dir,
        )

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        """Embed retrieval queries with the BGE retrieval instruction."""
        queries = [f"{BGE_RETRIEVAL_QUERY_PREFIX}{str(text).strip()}" for text in texts]
        return self.provider.embed_texts(queries)

    def embed_documents(self, documents: Iterable[Document]) -> List[EmbeddedDocument]:
        """Embed passages without applying the query instruction."""
        document_list = list(documents)
        vectors = self.provider.embed_texts(document.content for document in document_list)
        embedded_documents = []
        for document, vector in zip(document_list, vectors, strict=True):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "embedding_model": self.model_name,
                    "embedding_dimension": self.dimension,
                    "embedding_normalized": True,
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
