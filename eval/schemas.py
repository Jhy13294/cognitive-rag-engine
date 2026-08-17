from dataclasses import dataclass
from typing import Dict, List, Optional

ALLOWED_CAPABILITIES = {"exact_name", "paraphrase", "long_tail", "negative"}


@dataclass(frozen=True)
class RelevantItem:
    """A manually labeled relevant source for one golden query."""

    source: str
    chunk_index: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict) -> "RelevantItem":
        """Create a RelevantItem from a JSON object."""
        if not isinstance(data, dict):
            raise ValueError("Each relevant item must be an object")
        source = data.get("source")
        if not source:
            raise ValueError("Each relevant item must include source")

        chunk_index = data.get("chunk_index")
        if chunk_index is not None:
            chunk_index = int(chunk_index)

        return cls(source=str(source), chunk_index=chunk_index)

    def to_dict(self) -> Dict:
        """Serialize the relevant item."""
        data = {"source": self.source}
        if self.chunk_index is not None:
            data["chunk_index"] = self.chunk_index
        return data


@dataclass(frozen=True)
class GoldenExample:
    """One deterministic retrieval-evaluation example."""

    qid: str
    question: str
    relevant: List[RelevantItem]
    capability: str
    note: str
    ground_truth: str = ""

    @classmethod
    def from_dict(cls, data: Dict) -> "GoldenExample":
        """Create a validated GoldenExample from a JSON object."""
        if not isinstance(data, dict):
            raise ValueError("Golden example must be an object")

        qid = str(data.get("qid", "")).strip()
        question = str(data.get("question", "")).strip()
        ground_truth = str(data.get("ground_truth", "")).strip()
        capability = str(data.get("capability", "")).strip()
        note = str(data.get("note", "")).strip()
        relevant_raw = data.get("relevant")

        if not qid:
            raise ValueError("Golden example qid is required")
        if not question:
            raise ValueError(f"Golden example {qid} question is required")
        if not ground_truth:
            raise ValueError(f"Golden example {qid} ground_truth is required")
        if capability not in ALLOWED_CAPABILITIES:
            raise ValueError(f"Golden example {qid} has unsupported capability: {capability}")
        if not isinstance(relevant_raw, list):
            raise ValueError(f"Golden example {qid} must use relevant as a list")
        if not note:
            raise ValueError(f"Golden example {qid} note is required")

        relevant = [RelevantItem.from_dict(item) for item in relevant_raw]
        if capability == "negative" and relevant:
            raise ValueError(f"Negative golden example {qid} must have an empty relevant list")
        if capability != "negative" and not relevant:
            raise ValueError(f"Positive golden example {qid} must have at least one relevant item")

        return cls(
            qid=qid,
            question=question,
            relevant=relevant,
            ground_truth=ground_truth,
            capability=capability,
            note=note,
        )

    def to_dict(self) -> Dict:
        """Serialize the golden example."""
        return {
            "qid": self.qid,
            "question": self.question,
            "relevant": [item.to_dict() for item in self.relevant],
            "ground_truth": self.ground_truth,
            "capability": self.capability,
            "note": self.note,
        }
