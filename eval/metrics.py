from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .schemas import GoldenExample, RelevantItem

RetrieveFn = Callable[[str, int], Sequence[Any]]

PARENT_EXPANSION_EVAL_ERROR = (
    "Refusing to score parent-expanded retrieval results. "
    "Parent expansion is a generation-side content transform that can collapse sibling child chunks "
    "and change the measured top-k window. Evaluate the child ranked list instead."
)


def evaluate_retriever(
    retrieve: RetrieveFn,
    examples: Sequence[GoldenExample],
    k_values: Sequence[int] = (3, 5, 10),
    match_scope: str = "source",
    metadata: Optional[Dict] = None,
) -> Dict:
    """Evaluate a retrieval callable against a golden set.

    Metric definitions:
    - hit_rate@k: fraction of positive queries where top-k contains at least
      one relevant item.
    - MRR@k: mean reciprocal rank of the first relevant item in top-k; zero
      when no relevant item is retrieved.
    - recall@k: macro average of retrieved relevant items divided by all
      relevant items for each positive query.
    - negative_empty_rate@k: fraction of negative queries with no retrieved
      results.
    - negative_false_recall_rate@k: fraction of negative queries with at least
      one retrieved result.

    The evaluator accepts retrieve(question, top_k) instead of a pipeline so
    rerankers, hybrid retrievers, and query rewrite variants can reuse the
    same measurement code without coupling to generation.
    """
    if not examples:
        raise ValueError("examples must not be empty")
    if match_scope not in {"source", "chunk"}:
        raise ValueError("match_scope must be either source or chunk")

    normalized_k_values = sorted({int(k) for k in k_values})
    if any(k <= 0 for k in normalized_k_values):
        raise ValueError("all k values must be greater than 0")

    assert_metric_input_is_child_ranked_list(metadata=metadata)

    max_k = max(normalized_k_values)
    cases = []
    for example in examples:
        retrieved = list(retrieve(example.question, max_k))
        assert_metric_input_is_child_ranked_list(retrieved=retrieved)
        case = {
            "qid": example.qid,
            "question": example.question,
            "capability": example.capability,
            "note": example.note,
            "relevant": [item.to_dict() for item in example.relevant],
            "per_k": {},
        }

        for k in normalized_k_values:
            case["per_k"][str(k)] = evaluate_case_at_k(
                example=example,
                retrieved=retrieved[:k],
                k=k,
                match_scope=match_scope,
            )

        cases.append(case)

    report = {
        "metadata": metadata or {},
        "k_values": normalized_k_values,
        "match_scope": match_scope,
        "metrics": {str(k): summarize_cases(cases, str(k)) for k in normalized_k_values},
        "by_capability": summarize_by_capability(cases, normalized_k_values),
        "cases": cases,
        "low_recall_cases_by_k": {
            str(k): low_recall_cases(cases, str(k)) for k in normalized_k_values
        },
        "low_recall_cases": low_recall_cases(cases, str(max_k)),
    }
    return report


def assert_metric_input_is_child_ranked_list(
    metadata: Optional[Dict] = None,
    retrieved: Optional[Sequence[Any]] = None,
) -> None:
    """Reject generation-side parent-expanded results in retrieval metrics."""
    if metadata and metadata.get("expand_parent_context") is True:
        raise ValueError(PARENT_EXPANSION_EVAL_ERROR)

    if retrieved is None:
        return

    for item in retrieved:
        item_metadata = dict(getattr(item, "metadata", {}) or {})
        if item_metadata.get("parent_expanded") is True or "collapsed_child_count" in item_metadata:
            raise ValueError(PARENT_EXPANSION_EVAL_ERROR)


def evaluate_case_at_k(
    example: GoldenExample,
    retrieved: Sequence[Any],
    k: int,
    match_scope: str = "source",
) -> Dict:
    """Evaluate one query at one k value."""
    retrieved_items = [retrieved_source_to_dict(item) for item in retrieved]
    relevant_keys = relevant_keys_for(example.relevant, match_scope=match_scope)
    retrieved_keys = [retrieved_key_for(item, match_scope=match_scope) for item in retrieved_items]

    if not relevant_keys:
        is_empty = len(retrieved_items) == 0
        return {
            "k": k,
            "is_negative": True,
            "hit": False,
            "hit_count": 0,
            "relevant_count": 0,
            "first_hit_rank": None,
            "reciprocal_rank": 0.0,
            "recall": None,
            "negative_empty": is_empty,
            "negative_false_recall": not is_empty,
            "retrieved": retrieved_items,
            "miss_reason": "Negative query returned no results."
            if is_empty
            else "Negative query returned non-empty retrieval results.",
        }

    first_hit_rank = None
    matched_keys = set()
    for rank, key in enumerate(retrieved_keys, start=1):
        matched_key = first_matching_key(key, relevant_keys)
        if matched_key is None:
            continue
        matched_keys.add(matched_key)
        if first_hit_rank is None:
            first_hit_rank = rank

    hit_count = len(matched_keys)
    relevant_count = len(relevant_keys)
    recall = hit_count / relevant_count if relevant_count else 0.0
    hit = first_hit_rank is not None
    reciprocal_rank = 1.0 / first_hit_rank if first_hit_rank else 0.0

    return {
        "k": k,
        "is_negative": False,
        "hit": hit,
        "hit_count": hit_count,
        "relevant_count": relevant_count,
        "first_hit_rank": first_hit_rank,
        "reciprocal_rank": reciprocal_rank,
        "recall": recall,
        "negative_empty": False,
        "negative_false_recall": False,
        "retrieved": retrieved_items,
        "miss_reason": miss_reason(hit=hit, hit_count=hit_count, relevant_count=relevant_count),
    }


def summarize_cases(cases: Sequence[Dict], k_key: str) -> Dict:
    """Summarize positive and negative cases at one k value."""
    positive = [case["per_k"][k_key] for case in cases if not case["per_k"][k_key]["is_negative"]]
    negative = [case["per_k"][k_key] for case in cases if case["per_k"][k_key]["is_negative"]]

    positive_count = len(positive)
    negative_count = len(negative)
    hit_rate = _mean(1.0 if item["hit"] else 0.0 for item in positive)
    mrr = _mean(item["reciprocal_rank"] for item in positive)
    recall = _mean(item["recall"] for item in positive)
    negative_empty_rate = _mean(1.0 if item["negative_empty"] else 0.0 for item in negative)
    negative_false_recall_rate = _mean(
        1.0 if item["negative_false_recall"] else 0.0 for item in negative
    )

    return {
        "query_count": len(cases),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "recall": recall,
        "negative_empty_rate": negative_empty_rate,
        "negative_false_recall_rate": negative_false_recall_rate,
    }


def summarize_by_capability(cases: Sequence[Dict], k_values: Sequence[int]) -> Dict:
    """Return metric summaries grouped by capability label."""
    capabilities = sorted({case["capability"] for case in cases})
    grouped = {}
    for capability in capabilities:
        capability_cases = [case for case in cases if case["capability"] == capability]
        grouped[capability] = {str(k): summarize_cases(capability_cases, str(k)) for k in k_values}
    return grouped


def low_recall_cases(cases: Sequence[Dict], k_key: str) -> List[Dict]:
    """Return cases that need inspection at the selected k value."""
    low_cases = []
    for case in cases:
        metrics = case["per_k"][k_key]
        is_low_positive = not metrics["is_negative"] and metrics["recall"] < 1.0
        is_false_negative = metrics["is_negative"] and metrics["negative_false_recall"]
        if is_low_positive or is_false_negative:
            low_cases.append(
                {
                    "qid": case["qid"],
                    "question": case["question"],
                    "capability": case["capability"],
                    "note": case["note"],
                    "relevant": case["relevant"],
                    "k": int(k_key),
                    "first_hit_rank": metrics["first_hit_rank"],
                    "miss_reason": metrics["miss_reason"],
                    "retrieved": metrics["retrieved"],
                }
            )
    return low_cases


def relevant_keys_for(
    items: Sequence[RelevantItem], match_scope: str
) -> Set[Tuple[str, Optional[int]]]:
    """Build match keys from manually labeled relevant items."""
    return {
        make_match_key(item.source, item.chunk_index, match_scope=match_scope) for item in items
    }


def retrieved_key_for(item: Dict, match_scope: str) -> Tuple[str, Optional[int]]:
    """Build a match key from one retrieved item."""
    return make_match_key(
        source=str(item.get("source", "")),
        chunk_index=item.get("chunk_index"),
        match_scope=match_scope,
    )


def make_match_key(
    source: str, chunk_index: Optional[int], match_scope: str
) -> Tuple[str, Optional[int]]:
    """Create a source-level or chunk-level match key."""
    normalized_source = normalize_source(source)
    if match_scope == "source" or chunk_index is None:
        return normalized_source, None
    return normalized_source, int(chunk_index)


def first_matching_key(
    retrieved_key: Tuple[str, Optional[int]],
    relevant_keys: Set[Tuple[str, Optional[int]]],
) -> Optional[Tuple[str, Optional[int]]]:
    """Return the relevant key matched by a retrieved item, if any."""
    if retrieved_key in relevant_keys:
        return retrieved_key

    source_key = (retrieved_key[0], None)
    if source_key in relevant_keys:
        return source_key

    return None


def retrieved_source_to_dict(source: Any) -> Dict:
    """Convert a RetrievedSource-like object into report-safe data."""
    metadata = dict(getattr(source, "metadata", {}) or {})
    return {
        "id": str(metadata.get("id", "")),
        "source": normalize_source(str(metadata.get("source", ""))),
        "chunk_index": metadata.get("chunk_index"),
        "score": float(getattr(source, "score", 0.0)),
        "content_preview": str(getattr(source, "content", ""))[:180],
    }


def normalize_source(source: str) -> str:
    """Normalize source paths for stable report matching across OS paths."""
    raw_source = str(source).replace("\\", "/")
    if not raw_source:
        return ""

    try:
        path = Path(source)
        if path.is_absolute():
            return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return raw_source

    return raw_source[2:] if raw_source.startswith("./") else raw_source


def miss_reason(hit: bool, hit_count: int, relevant_count: int) -> str:
    """Explain why a case is shown in low-recall reports."""
    if not hit:
        return "No relevant source appeared in top-k."
    if hit_count < relevant_count:
        return f"Only {hit_count}/{relevant_count} relevant items were retrieved."
    return "All relevant items were retrieved."


def _mean(values: Iterable[float]) -> float:
    """Return a rounded arithmetic mean."""
    value_list = list(values)
    if not value_list:
        return 0.0
    return round(sum(value_list) / len(value_list), 6)
