from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List


class QueryRewriterError(Exception):
    """Base exception for query rewriting errors."""


@dataclass
class QueryRewriteConfig:
    """Configuration shared by query rewriters."""

    provider: str
    num_queries: int = 3
    temperature: float = 0.1
    cache_enabled: bool = True
    weight_original: float = 1.0
    weight_variant: float = 0.7
    fixture_path: str = "eval/fixtures/query_rewrites.jsonl"

    def __post_init__(self):
        """Validate query rewrite configuration."""
        if self.provider not in {"deterministic", "chat"}:
            raise ValueError("QUERY_REWRITE_PROVIDER must be one of: deterministic, chat.")
        if self.num_queries < 1:
            raise ValueError("QUERY_REWRITE_NUM_QUERIES must be greater than or equal to 1.")
        if self.temperature < 0:
            raise ValueError("QUERY_REWRITE_TEMPERATURE must be non-negative.")
        if self.weight_original < 0:
            raise ValueError("QUERY_REWRITE_WEIGHT_ORIGINAL must be non-negative.")
        if self.weight_variant < 0:
            raise ValueError("QUERY_REWRITE_WEIGHT_VARIANT must be non-negative.")
        if self.weight_original < self.weight_variant:
            raise ValueError(
                "QUERY_REWRITE_WEIGHT_ORIGINAL must be greater than or equal to QUERY_REWRITE_WEIGHT_VARIANT."
            )
        if self.weight_original + self.weight_variant <= 0:
            raise ValueError("At least one query rewrite weight must be greater than 0.")


class QueryRewriter(ABC):
    """Base interface for query rewriting providers."""

    def __init__(self, config: QueryRewriteConfig):
        """Initialize a query rewriter."""
        self.config = config

    @property
    def model_name(self) -> str:
        """Return a provider-visible model name."""
        return self.config.provider

    @abstractmethod
    def rewrite(self, question: str) -> List[str]:
        """Return ordered query variants with the original question first."""
        pass


def normalize_query_variants(question: str, variants: Iterable[str], limit: int) -> List[str]:
    """Return de-duplicated variants with the original question at index zero."""
    if limit < 1:
        raise ValueError("limit must be greater than or equal to 1")

    original = str(question).strip()
    ordered = [original] if original else []
    seen = {_dedup_key(original)} if original else set()

    for variant in variants:
        text = str(variant).strip()
        if not text:
            continue
        key = _dedup_key(text)
        if key in seen:
            continue
        ordered.append(text)
        seen.add(key)
        if len(ordered) >= limit:
            break

    return ordered


def _dedup_key(text: str) -> str:
    """Return a stable key for query de-duplication."""
    return " ".join(text.lower().split())
