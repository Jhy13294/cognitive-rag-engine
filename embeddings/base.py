import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List

from document_loader import Document


Vector = List[float]


@dataclass
class EmbeddingConfig:
    """Configuration shared by embedding providers."""

    model_name: str
    dimension: int
    normalize: bool = True
    batch_size: int = 32
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate embedding configuration."""
        if not self.model_name:
            raise ValueError("model_name is required")
        if self.dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")


@dataclass
class EmbeddedDocument:
    """Document content paired with an embedding vector."""

    content: str
    metadata: Dict
    embedding: Vector


class EmbeddingProvider(ABC):
    """Base interface for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize an embedding provider."""
        self.config = config

    @property
    def model_name(self) -> str:
        """Return the provider model name."""
        return self.config.model_name

    @property
    def dimension(self) -> int:
        """Return embedding vector dimension."""
        return self.config.dimension

    def embed_text(self, text: str) -> Vector:
        """Embed a single text string."""
        results = self.embed_texts([text])
        return results[0]

    @abstractmethod
    def embed_texts(self, texts: Iterable[str]) -> List[Vector]:
        """Embed multiple text strings."""
        pass

    def embed_documents(self, documents: Iterable[Document]) -> List[EmbeddedDocument]:
        """Embed Document objects while preserving metadata."""
        document_list = list(documents)
        vectors = self.embed_texts(document.content for document in document_list)
        embedded_documents = []

        for document, vector in zip(document_list, vectors, strict=True):
            metadata = dict(document.metadata)
            metadata.update(
                {
                    "embedding_model": self.model_name,
                    "embedding_dimension": self.dimension,
                    "embedding_normalized": self.config.normalize,
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

    def _validate_vector(self, vector: Vector) -> None:
        """Validate vector dimension."""
        if len(vector) != self.dimension:
            raise ValueError(f"Expected vector dimension {self.dimension}, got {len(vector)}")


def normalize_vector(vector: Vector) -> Vector:
    """Return an L2-normalized copy of a vector."""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return list(vector)
    return [value / norm for value in vector]


def cosine_similarity(left: Vector, right: Vector) -> float:
    """Calculate cosine similarity between two vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(
        left_value * right_value for left_value, right_value in zip(left, right, strict=True)
    )
    return dot_product / (left_norm * right_norm)
