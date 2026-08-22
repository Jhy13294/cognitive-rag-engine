"""Deterministic retrieval relevance gate.

The gate only makes a rejection decision when the active score space is
explicitly supported and has its own configured threshold. Rank-derived RRF
scores, BM25 scores, lexical reranker scores, hash embeddings, and unknown
spaces are observable no-ops rather than implicit evidence of irrelevance.
"""

import math
import threading
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from logger import setup_logger

from .models import RetrievedSource

logger = setup_logger(__name__)


@dataclass(frozen=True)
class RelevanceGateConfig:
    """Runtime configuration whose thresholds each belong to one score space."""

    enabled: bool = False
    min_dense_cosine: Optional[float] = None
    min_cohere_rerank_score: Optional[float] = None
    dense_score_space: str = "unknown"
    hybrid_enabled: bool = False
    reranker_provider: Optional[str] = None
    query_rewrite_enabled: bool = False

    def __post_init__(self) -> None:
        """Reject invalid thresholds without inventing product defaults."""
        if self.min_dense_cosine is not None and not -1.0 <= self.min_dense_cosine <= 1.0:
            raise ValueError("min_dense_cosine must be between -1 and 1")
        if (
            self.min_cohere_rerank_score is not None
            and not 0.0 <= self.min_cohere_rerank_score <= 1.0
        ):
            raise ValueError("min_cohere_rerank_score must be between 0 and 1")


@dataclass(frozen=True)
class RelevanceGateDecision:
    """One gate outcome carried with retrieval results and cache payloads."""

    action: str
    reason: str
    score_space: Optional[str] = None
    score: Optional[float] = None
    threshold: Optional[float] = None
    score_source: Optional[str] = None

    @property
    def rejected(self) -> bool:
        """Return whether generation must be short-circuited."""
        return self.action == "rejected"

    def to_dict(self) -> Dict:
        """Serialize the decision for Redis and response diagnostics."""
        return {
            "action": self.action,
            "reason": self.reason,
            "score_space": self.score_space,
            "score": self.score,
            "threshold": self.threshold,
            "score_source": self.score_source,
        }

    @classmethod
    def from_dict(cls, payload: Dict) -> "RelevanceGateDecision":
        """Deserialize a previously cached decision."""
        score = payload.get("score")
        threshold = payload.get("threshold")
        return cls(
            action=str(payload.get("action") or "skipped"),
            reason=str(payload.get("reason") or "cached_decision_missing_reason"),
            score_space=_optional_string(payload.get("score_space")),
            score=float(score) if score is not None else None,
            threshold=float(threshold) if threshold is not None else None,
            score_source=_optional_string(payload.get("score_source")),
        )


class RelevanceGatedSources(list):
    """List-compatible sources carrying the decision that produced them."""

    def __init__(
        self,
        sources: Iterable[RetrievedSource],
        decision: RelevanceGateDecision,
    ) -> None:
        super().__init__(sources)
        self.relevance_gate_decision = decision


class RelevanceGate:
    """Reject low-relevance retrieval only in calibrated score spaces."""

    def __init__(self, config: RelevanceGateConfig):
        self.config = config
        self._lock = threading.Lock()
        self._actions: Dict[str, int] = {}
        self._skip_reasons: Dict[str, int] = {}
        self._score_spaces: Dict[str, int] = {}

    def apply(self, sources: List[RetrievedSource]) -> List[RetrievedSource]:
        """Return either the original sources or an empty, provenance-bearing list."""
        if not self.config.enabled:
            return sources

        if not sources:
            decision = RelevanceGateDecision(
                action="no_candidates",
                reason="retrieval_returned_no_candidates",
            )
            self._record(decision)
            return RelevanceGatedSources([], decision)

        score, score_space, score_source = self._resolve_score(sources[0])
        threshold = self._threshold_for(score_space)

        if score_space not in {"dense_cosine", "cohere_rerank_score"}:
            decision = RelevanceGateDecision(
                action="skipped",
                reason=f"unsupported_score_space:{score_space}",
                score_space=score_space,
                score=score,
                score_source=score_source,
            )
            self._record(decision)
            logger.warning(
                "Relevance gate skipped | reason=%s | score_space=%s | score_source=%s",
                decision.reason,
                score_space,
                score_source,
            )
            return RelevanceGatedSources(sources, decision)

        if score is None or not math.isfinite(score):
            decision = RelevanceGateDecision(
                action="skipped",
                reason="usable_score_missing",
                score_space=score_space,
                score=score,
                threshold=threshold,
                score_source=score_source,
            )
            self._record(decision)
            logger.warning(
                "Relevance gate skipped | reason=%s | score_space=%s | score_source=%s",
                decision.reason,
                score_space,
                score_source,
            )
            return RelevanceGatedSources(sources, decision)

        if threshold is None:
            decision = RelevanceGateDecision(
                action="skipped",
                reason="threshold_not_configured",
                score_space=score_space,
                score=score,
                score_source=score_source,
            )
            self._record(decision)
            logger.warning(
                "Relevance gate skipped | reason=%s | score_space=%s | score_source=%s",
                decision.reason,
                score_space,
                score_source,
            )
            return RelevanceGatedSources(sources, decision)

        rejected = score < threshold
        decision = RelevanceGateDecision(
            action="rejected" if rejected else "accepted",
            reason="score_below_threshold" if rejected else "score_meets_threshold",
            score_space=score_space,
            score=score,
            threshold=threshold,
            score_source=score_source,
        )
        self._record(decision)
        logger.info(
            "Relevance gate %s | score_space=%s | score=%.6f | threshold=%.6f | score_source=%s",
            decision.action,
            score_space,
            score,
            threshold,
            score_source,
        )
        return RelevanceGatedSources([] if rejected else sources, decision)

    def stats(self) -> Dict:
        """Return thread-safe counters proving rejection and no-op behavior."""
        with self._lock:
            actions = dict(sorted(self._actions.items()))
            skip_reasons = dict(sorted(self._skip_reasons.items()))
            score_spaces = dict(sorted(self._score_spaces.items()))
        return {
            "enabled": self.config.enabled,
            "actions": actions,
            "skip_reasons": skip_reasons,
            "score_spaces": score_spaces,
            "evaluated": sum(actions.values()),
        }

    def _resolve_score(self, source: RetrievedSource) -> Tuple[Optional[float], str, str]:
        """Resolve the physical score without ever treating RRF as relevance."""
        metadata = dict(source.metadata or {})
        retrieval_mode = str(metadata.get("retrieval_mode") or "").lower()

        if retrieval_mode == "bm25":
            return _optional_float(source.score), "bm25", "source.score"

        if self.config.hybrid_enabled:
            return (
                _optional_float(metadata.get("dense_score")),
                self.config.dense_score_space,
                'metadata["dense_score"]',
            )

        rerank_applied = bool(metadata.get("rerank_model"))
        if rerank_applied:
            provider = str(
                metadata.get("rerank_provider") or self.config.reranker_provider or "unknown"
            ).lower()
            if provider == "cohere":
                return _optional_float(source.score), "cohere_rerank_score", "source.score"
            if provider == "deterministic":
                return _optional_float(source.score), "lexical_score", "source.score"
            return _optional_float(source.score), f"unknown_reranker:{provider}", "source.score"

        if retrieval_mode == "hybrid_rrf":
            if self.config.query_rewrite_enabled:
                return (
                    _optional_float(metadata.get("q0_score")),
                    self.config.dense_score_space,
                    'metadata["q0_score"]',
                )
            return _optional_float(source.score), "rrf", "source.score"

        return _optional_float(source.score), self.config.dense_score_space, "source.score"

    def _threshold_for(self, score_space: str) -> Optional[float]:
        if score_space == "dense_cosine":
            return self.config.min_dense_cosine
        if score_space == "cohere_rerank_score":
            return self.config.min_cohere_rerank_score
        return None

    def _record(self, decision: RelevanceGateDecision) -> None:
        with self._lock:
            self._actions[decision.action] = self._actions.get(decision.action, 0) + 1
            if decision.action == "skipped":
                self._skip_reasons[decision.reason] = self._skip_reasons.get(decision.reason, 0) + 1
            if decision.score_space:
                self._score_spaces[decision.score_space] = (
                    self._score_spaces.get(decision.score_space, 0) + 1
                )


def relevance_gate_decision_for(
    sources: List[RetrievedSource],
) -> Optional[RelevanceGateDecision]:
    """Read a decision from a list returned by the gate or L2 cache."""
    decision = getattr(sources, "relevance_gate_decision", None)
    return decision if isinstance(decision, RelevanceGateDecision) else None


def infer_dense_score_space(embedding_provider, vector_store) -> str:
    """Identify semantic cosine only when both provider and store contracts support it."""
    model_name = str(getattr(embedding_provider, "model_name", "") or "").lower()
    if "hash-embedding" in model_name or model_name.startswith("hash"):
        return "hash_cosine"

    distance = getattr(vector_store, "distance", None)
    if distance is not None:
        normalized_distance = str(distance).lower()
        if normalized_distance != "cosine":
            return f"dense_{normalized_distance}"
        return "dense_cosine"

    if any(base.__name__ == "InMemoryVectorStore" for base in type(vector_store).__mro__):
        return "dense_cosine"
    return "unknown"


def _optional_float(value) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _optional_string(value) -> Optional[str]:
    return str(value) if value is not None else None
