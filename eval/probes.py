import json
from pathlib import Path
from typing import List

from .schemas import GoldenExample

NEGATIVE_PROBE_CATEGORIES = {
    "out_of_domain",
    "near_domain_out_of_scope",
    "explicitly_excluded",
    "hallucinated_entity",
}


def load_negative_probes(path: str) -> List[GoldenExample]:
    """Load unlabeled negative probes without changing the golden-set schema."""
    probe_path = Path(path)
    examples = []
    seen_ids = set()
    seen_questions = set()

    with probe_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid negative probe JSON at {probe_path}:{line_number}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(f"Negative probe at {probe_path}:{line_number} must be an object")
            if "relevant" in payload:
                raise ValueError(
                    f"Negative probe at {probe_path}:{line_number} must not contain relevance labels"
                )

            probe_id = str(payload.get("probe_id") or "").strip()
            question = str(payload.get("question") or "").strip()
            category = str(payload.get("category") or "").strip()
            note = str(payload.get("note") or "").strip()
            if not probe_id or not question or not note:
                raise ValueError(
                    f"Negative probe at {probe_path}:{line_number} requires probe_id, question, and note"
                )
            if category not in NEGATIVE_PROBE_CATEGORIES:
                raise ValueError(f"Negative probe {probe_id} has unsupported category: {category}")
            if probe_id in seen_ids:
                raise ValueError(f"Duplicate negative probe id: {probe_id}")
            normalized_question = " ".join(question.lower().split())
            if normalized_question in seen_questions:
                raise ValueError(f"Duplicate negative probe question: {question}")

            seen_ids.add(probe_id)
            seen_questions.add(normalized_question)
            examples.append(
                GoldenExample(
                    qid=probe_id,
                    question=question,
                    relevant=[],
                    capability=f"negative_probe_{category}",
                    note=note,
                    ground_truth="The assistant should abstain.",
                )
            )

    if not examples:
        raise ValueError(f"Negative probe file is empty: {probe_path}")
    return examples
