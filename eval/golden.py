import json
from pathlib import Path
from typing import List

from .schemas import GoldenExample


def load_golden_set(path: str) -> List[GoldenExample]:
    """Load and validate a JSONL golden set.

    The schema intentionally uses relevant as a list so recall@k can measure
    how many manually labeled relevant sources were retrieved for each query.
    """
    golden_path = Path(path)
    if not golden_path.exists():
        raise FileNotFoundError(f"Golden set not found: {golden_path}")

    examples = []
    seen_qids = set()
    with golden_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            try:
                raw_example = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {golden_path}:{line_number}: {e}") from e

            example = GoldenExample.from_dict(raw_example)
            if example.qid in seen_qids:
                raise ValueError(f"Duplicate golden qid: {example.qid}")
            seen_qids.add(example.qid)
            examples.append(example)

    if not examples:
        raise ValueError(f"Golden set is empty: {golden_path}")

    return examples

