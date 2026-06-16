from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Sequence

from .models import RankedRecord


@dataclass
class RRFConfig:
    """Reciprocal Rank Fusion configuration."""

    k: int = 60
    weights: Dict[str, float] = field(default_factory=lambda: {"dense": 0.2, "sparse": 1.0})

    def __post_init__(self):
        """Validate RRF configuration."""
        if self.k <= 0:
            raise ValueError("RRF k must be greater than 0")
        if not self.weights:
            raise ValueError("At least one RRF weight is required")
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("RRF weights must be non-negative")
        if sum(self.weights.values()) <= 0:
            raise ValueError("At least one RRF weight must be greater than 0")


class ReciprocalRankFusion:
    """Fuse multiple ranked lists using rank positions only."""

    def __init__(self, config: RRFConfig = None):
        """Initialize the fusion strategy."""
        self.config = config or RRFConfig()

    def fuse(
        self,
        ranked_lists: Mapping[str, Sequence[RankedRecord]],
        top_k: int = None,
    ) -> List[RankedRecord]:
        """Fuse ranked lists into one deterministic ranking."""
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        scores: Dict[str, float] = {}
        representatives: Dict[str, RankedRecord] = {}
        path_metadata: Dict[str, Dict] = {}

        for path_name in sorted(ranked_lists):
            weight = self.config.weights.get(path_name, 0.0)
            if weight <= 0:
                continue

            for rank, record in enumerate(ranked_lists[path_name], start=1):
                scores[record.id] = scores.get(record.id, 0.0) + weight / (self.config.k + rank)
                representatives.setdefault(record.id, record)
                item_metadata = path_metadata.setdefault(record.id, {})
                item_metadata[f"{path_name}_rank"] = rank
                item_metadata[f"{path_name}_score"] = record.score

        fused_records = []
        for record_id, score in scores.items():
            representative = representatives[record_id]
            metadata = dict(representative.metadata)
            metadata.update(path_metadata.get(record_id, {}))
            metadata["id"] = record_id
            metadata["retrieval_mode"] = "hybrid_rrf"
            metadata["rrf_score"] = score
            fused_records.append(
                RankedRecord(
                    id=record_id,
                    score=score,
                    content=representative.content,
                    metadata=metadata,
                    record=representative.record,
                )
            )

        fused_records.sort(key=lambda record: (-record.score, record.id))
        return fused_records[:top_k] if top_k is not None else fused_records
