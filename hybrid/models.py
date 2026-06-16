from dataclasses import dataclass, field
from typing import Dict, Optional

from vector_store import VectorRecord


@dataclass
class RankedRecord:
    """A retriever result keyed by the stable VectorRecord id."""

    id: str
    score: float
    content: str
    metadata: Dict = field(default_factory=dict)
    record: Optional[VectorRecord] = None
