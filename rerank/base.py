from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from rag.pipeline import RetrievedSource


class RerankerError(RuntimeError):
    """Raised when a reranker operation fails."""


@dataclass
class RerankConfig:
    """Configuration shared by reranker providers."""

    model_name: str
    top_n: int = 5
    fetch_k: int = 30
    batch_size: int = 32
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate reranker configuration."""
        if not self.model_name:
            raise ValueError("model_name is required")
        if self.top_n <= 0:
            raise ValueError("top_n must be greater than 0")
        if self.fetch_k <= 0:
            raise ValueError("fetch_k must be greater than 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")


@dataclass
class RerankResult:
    """Reranked candidate with a score and original candidate index."""

    index: int
    score: float
    content: str
    metadata: Dict = field(default_factory=dict)


class Reranker(ABC):
    """Base interface for candidate rerankers."""

    def __init__(self, config: RerankConfig):
        """Initialize a reranker."""
        self.config = config

    @property
    def model_name(self) -> str:
        """Return the reranker model name."""
        return self.config.model_name

    @property
    def top_n(self) -> int:
        """Return the default number of reranked candidates to keep."""
        return self.config.top_n

    @property
    def fetch_k(self) -> int:
        """Return the default dense candidate pool size."""
        return self.config.fetch_k

    @property
    def batch_size(self) -> int:
        """Return the maximum candidates sent in one provider request."""
        return self.config.batch_size

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence["RetrievedSource"],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank candidate sources for a query."""
        pass
