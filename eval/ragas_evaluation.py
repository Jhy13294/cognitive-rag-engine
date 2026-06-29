import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .schemas import GoldenExample


GATED_METRICS: Tuple[str, ...] = (
    "faithfulness",
    "answer_relevance",
)
REPORTED_METRICS: Tuple[str, ...] = (
    "context_precision",
    "context_recall",
)
RAGAS_METRICS: Tuple[str, ...] = GATED_METRICS + REPORTED_METRICS
FIXTURE_SCHEMA_VERSION = "ragas-verdicts-v5"
REPORT_SCHEMA_VERSION = "ragas-report-v1"


def file_sha256(path: str) -> str:
    """Return a stable SHA256 fingerprint for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(parts: Iterable[str]) -> str:
    """Hash ordered text parts without storing evaluation content in fixtures."""
    digest = hashlib.sha256()
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def load_verdict_fixture(path: str) -> Dict:
    """Load and validate a deterministic Ragas verdict fixture."""
    fixture_path = Path(path)
    if not fixture_path.exists():
        raise FileNotFoundError(f"Ragas verdict fixture not found: {fixture_path}")

    records = []
    with fixture_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid Ragas fixture JSONL at {fixture_path}:{line_number}: {error}"
                ) from error

    if not records or records[0].get("record_type") != "metadata":
        raise ValueError("Ragas fixture must start with a metadata record.")

    metadata = dict(records[0])
    cases = [dict(record) for record in records[1:]]
    validate_fixture(metadata, cases)
    return {"metadata": metadata, "cases": cases}


def validate_fixture(metadata: Dict, cases: Sequence[Dict]) -> None:
    """Validate fixture provenance, score bounds, and case uniqueness."""
    if metadata.get("schema_version") != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported Ragas fixture schema: {metadata.get('schema_version')}"
        )
    required_metadata = (
        "judge_model_id",
        "ragas_version",
        "temperature",
        "repetitions",
        "pipeline_profile",
        "golden_version",
        "recorded_at",
        "recording_mode",
        "generation_prompt_version",
        "statement_prompt_version",
        "faithfulness_context_format",
        "gated_metrics",
        "reported_only_metrics",
    )
    missing = [
        name
        for name in required_metadata
        if name not in metadata or metadata.get(name) is None or metadata.get(name) == ""
    ]
    if missing:
        raise ValueError("Ragas fixture metadata is missing: " + ", ".join(missing))

    repetitions = int(metadata["repetitions"])
    if repetitions < 2:
        raise ValueError("Ragas fixture repetitions must be at least 2.")
    if metadata["recording_mode"] != "live_gated":
        raise ValueError(
            "Ragas replay accepts only live_gated recordings; synthetic/bootstrap scores cannot gate CI."
        )

    seen_qids = set()
    for case in cases:
        if case.get("record_type") != "verdict":
            raise ValueError("Every non-metadata fixture record must be a verdict.")
        qid = str(case.get("qid", "")).strip()
        capability = str(case.get("capability", "")).strip()
        if not qid:
            raise ValueError("Ragas fixture verdict qid is required.")
        if qid in seen_qids:
            raise ValueError(f"Duplicate Ragas fixture qid: {qid}")
        seen_qids.add(qid)

        if capability == "negative":
            negative_runs = case.get("negative_runs")
            if not isinstance(negative_runs, list) or len(negative_runs) != repetitions:
                raise ValueError(
                    f"Negative verdict {qid} must contain {repetitions} negative_runs."
                )
            for run in negative_runs:
                if not isinstance(run, dict):
                    raise ValueError(f"Negative verdict {qid} contains an invalid run.")
                if not isinstance(run.get("abstained"), bool):
                    raise ValueError(f"Negative verdict {qid} abstained must be boolean.")
                if not isinstance(run.get("fabricated"), bool):
                    raise ValueError(f"Negative verdict {qid} fabricated must be boolean.")
            continue

        runs = case.get("runs")
        if not isinstance(runs, list) or len(runs) != repetitions:
            raise ValueError(f"Positive verdict {qid} must contain {repetitions} runs.")
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError(f"Positive verdict {qid} contains an invalid run.")
            for metric in RAGAS_METRICS:
                if metric not in run:
                    raise ValueError(f"Positive verdict {qid} is missing metric {metric}.")
                validate_score(run[metric], f"{qid}.{metric}")


def validate_score(value, label: str) -> float:
    """Return a finite score in the closed interval [0, 1]."""
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Ragas score {label} must be numeric.") from error
    if not math.isfinite(score) or score < 0 or score > 1:
        raise ValueError(f"Ragas score {label} must be between 0 and 1.")
    return score


def validate_fixture_against_golden(
    fixture: Dict,
    examples: Sequence[GoldenExample],
    golden_version: str,
) -> None:
    """Reject stale or incomplete fixtures before calculating a quality gate."""
    metadata = fixture["metadata"]
    if metadata.get("golden_version") != golden_version:
        raise ValueError(
            "Ragas fixture golden_version does not match the current golden set; "
            "run a gated live refresh."
        )

    expected = {example.qid: example.capability for example in examples}
    actual = {case["qid"]: case["capability"] for case in fixture["cases"]}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        mismatched = sorted(
            qid for qid in set(expected) & set(actual) if expected[qid] != actual[qid]
        )
        raise ValueError(
            "Ragas fixture cases do not match the golden set | "
            f"missing={missing} | extra={extra} | capability_mismatch={mismatched}"
        )


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    """Summarize repeated judge scores with conventional and robust spread."""
    if not values:
        return {
            "mean": 0.0,
            "median": 0.0,
            "stddev": 0.0,
            "mad": 0.0,
            "min": 0.0,
            "max": 0.0,
            "n": 0,
        }
    numeric = [float(value) for value in values]
    median = statistics.median(numeric)
    return {
        "mean": statistics.fmean(numeric),
        "median": median,
        "stddev": statistics.pstdev(numeric) if len(numeric) > 1 else 0.0,
        "mad": statistics.median(abs(value - median) for value in numeric),
        "min": min(numeric),
        "max": max(numeric),
        "n": len(numeric),
    }


def build_replay_report(
    fixture: Dict,
    examples: Sequence[GoldenExample],
    golden_path: str,
    thresholds: Dict[str, float],
    margins: Dict[str, float],
    sigma_multiplier: float,
    negative_abstention_threshold: float,
    expected_judge_model: str,
    installed_ragas_version: Optional[str],
    expected_generation_prompt_version: str,
    expected_statement_prompt_version: str,
    expected_faithfulness_context_format: str,
) -> Dict:
    """Build a deterministic replay report and evaluate all quality gates."""
    golden_version = file_sha256(golden_path)
    validate_fixture_against_golden(fixture, examples, golden_version)

    fixture_metadata = fixture["metadata"]
    case_summaries = []
    values_by_metric = {metric: [] for metric in RAGAS_METRICS}
    values_by_capability: Dict[str, Dict[str, List[float]]] = {}
    max_case_stddev = {metric: {"qid": None, "value": 0.0} for metric in RAGAS_METRICS}
    max_case_mad = {metric: {"qid": None, "value": 0.0} for metric in RAGAS_METRICS}
    failures = []
    negative_abstentions = []
    negative_fabrications = []

    if fixture_metadata["judge_model_id"] != expected_judge_model:
        failures.append(
            {
                "kind": "baseline_invalidated",
                "field": "judge_model_id",
                "actual": fixture_metadata["judge_model_id"],
                "expected": expected_judge_model,
            }
        )
    if installed_ragas_version and fixture_metadata["ragas_version"] != installed_ragas_version:
        failures.append(
            {
                "kind": "baseline_invalidated",
                "field": "ragas_version",
                "actual": fixture_metadata["ragas_version"],
                "expected": installed_ragas_version,
            }
        )
    for field, expected in (
        ("generation_prompt_version", expected_generation_prompt_version),
        ("statement_prompt_version", expected_statement_prompt_version),
        ("faithfulness_context_format", expected_faithfulness_context_format),
    ):
        if fixture_metadata.get(field) != expected:
            failures.append(
                {
                    "kind": "baseline_invalidated",
                    "field": field,
                    "actual": fixture_metadata.get(field),
                    "expected": expected,
                }
            )
    for field, expected in (
        ("gated_metrics", GATED_METRICS),
        ("reported_only_metrics", REPORTED_METRICS),
    ):
        actual = tuple(fixture_metadata.get(field) or ())
        if actual != expected:
            failures.append(
                {
                    "kind": "baseline_invalidated",
                    "field": field,
                    "actual": list(actual),
                    "expected": list(expected),
                }
            )

    for case in fixture["cases"]:
        qid = case["qid"]
        capability = case["capability"]
        summary = {
            "qid": qid,
            "capability": capability,
            "answer_sha256": case.get("answer_sha256"),
            "contexts_sha256": case.get("contexts_sha256"),
            "faithfulness_contexts_sha256": case.get(
                "faithfulness_contexts_sha256"
            ),
        }
        if capability == "negative":
            abstained = [bool(run["abstained"]) for run in case["negative_runs"]]
            fabricated = [bool(run["fabricated"]) for run in case["negative_runs"]]
            abstention_rate = statistics.fmean(int(value) for value in abstained)
            fabrication_rate = statistics.fmean(int(value) for value in fabricated)
            negative_abstentions.extend(abstained)
            negative_fabrications.extend(fabricated)
            summary["negative"] = {
                "abstention_rate": abstention_rate,
                "fabrication_rate": fabrication_rate,
                "n": len(abstained),
            }
            if abstention_rate < negative_abstention_threshold:
                failures.append(
                    {
                        "kind": "negative_abstention",
                        "qid": qid,
                        "metric": "negative_abstention",
                        "score": abstention_rate,
                        "threshold": negative_abstention_threshold,
                    }
                )
            if fabrication_rate > 1.0 - negative_abstention_threshold:
                failures.append(
                    {
                        "kind": "negative_fabrication",
                        "qid": qid,
                        "metric": "negative_fabrication",
                        "score": fabrication_rate,
                        "threshold": 1.0 - negative_abstention_threshold,
                    }
                )
            case_summaries.append(summary)
            continue

        metric_summaries = {}
        capability_values = values_by_capability.setdefault(
            capability,
            {metric: [] for metric in RAGAS_METRICS},
        )
        for metric in RAGAS_METRICS:
            values = [validate_score(run[metric], f"{qid}.{metric}") for run in case["runs"]]
            metric_summary = summarize_values(values)
            metric_summaries[metric] = metric_summary
            values_by_metric[metric].extend(values)
            capability_values[metric].extend(values)
            if metric_summary["stddev"] > max_case_stddev[metric]["value"]:
                max_case_stddev[metric] = {
                    "qid": qid,
                    "value": metric_summary["stddev"],
                }
            if metric_summary["mad"] > max_case_mad[metric]["value"]:
                max_case_mad[metric] = {"qid": qid, "value": metric_summary["mad"]}
            if (
                metric in GATED_METRICS
                and metric_summary["median"] < thresholds[metric]
            ):
                failures.append(
                    {
                        "kind": "quality_threshold",
                        "qid": qid,
                        "metric": metric,
                        "score": metric_summary["median"],
                        "threshold": thresholds[metric],
                    }
                )
        summary["metrics"] = metric_summaries
        case_summaries.append(summary)

    metric_summaries = {}
    for metric in RAGAS_METRICS:
        metric_summary = summarize_values(values_by_metric[metric])
        required_margin = sigma_multiplier * max_case_mad[metric]["value"]
        metric_summary.update(
            {
                "threshold": thresholds[metric],
                "configured_margin": margins[metric],
                "sigma_multiplier": sigma_multiplier,
                "max_case_stddev": max_case_stddev[metric]["value"],
                "max_case_stddev_qid": max_case_stddev[metric]["qid"],
                "max_case_mad": max_case_mad[metric]["value"],
                "max_case_mad_qid": max_case_mad[metric]["qid"],
                "required_margin": required_margin,
                "margin_sufficient": margins[metric] + 1e-12 >= required_margin,
            }
        )
        if metric in GATED_METRICS and not metric_summary["margin_sufficient"]:
            failures.append(
                {
                    "kind": "insufficient_margin",
                    "qid": max_case_mad[metric]["qid"],
                    "metric": metric,
                    "score": margins[metric],
                    "threshold": required_margin,
                }
            )
        metric_summaries[metric] = metric_summary

    by_capability = {
        capability: {
            metric: summarize_values(metric_values)
            for metric, metric_values in capability_metrics.items()
        }
        for capability, capability_metrics in sorted(values_by_capability.items())
    }
    negative_summary = {
        "query_count": sum(1 for case in fixture["cases"] if case["capability"] == "negative"),
        "run_count": len(negative_abstentions),
        "abstention_rate": (
            statistics.fmean(int(value) for value in negative_abstentions)
            if negative_abstentions
            else 0.0
        ),
        "fabrication_rate": (
            statistics.fmean(int(value) for value in negative_fabrications)
            if negative_fabrications
            else 0.0
        ),
        "threshold": negative_abstention_threshold,
        "included_in_positive_means": False,
    }

    positive_count = sum(1 for example in examples if example.capability != "negative")
    metadata = {
        key: value for key, value in fixture_metadata.items() if key != "record_type"
    }
    metadata.update(
        {
            "mode": "deterministic_fixture_replay",
            "fixture_schema_version": metadata.pop("schema_version"),
            "fixture_sha256": content_sha256(
                json.dumps(record, ensure_ascii=False, sort_keys=True)
                for record in [fixture_metadata, *fixture["cases"]]
            ),
            "golden_path": golden_path,
            "golden_version": golden_version,
            "sample_count": len(examples),
            "positive_sample_count": positive_count,
            "negative_sample_count": len(examples) - positive_count,
            "aggregation_method": "median",
            "spread_method": "mad",
            "judge_model_change_invalidates_baseline": True,
            "ragas_version_change_invalidates_baseline": True,
            "aggregation_method_change_invalidates_baseline": True,
            "spread_method_change_invalidates_baseline": True,
            "repetitions_change_invalidates_baseline": True,
            "generation_prompt_change_invalidates_baseline": True,
            "statement_prompt_change_invalidates_baseline": True,
            "faithfulness_context_format_change_invalidates_baseline": True,
            "gating_scope_change_invalidates_baseline": True,
            "temperature_zero_is_deterministic": False,
            "data_egress": "none; replay reads hashes and recorded verdicts only",
            "interpretation": (
                "LLM-judge estimates on a small hand-labeled set; not production truth or a deterministic live score."
            ),
        }
    )
    return {
        "report_type": "ragas_generation_quality",
        "schema_version": REPORT_SCHEMA_VERSION,
        "metadata": metadata,
        "metrics": metric_summaries,
        "by_capability": by_capability,
        "negative": negative_summary,
        "cases": case_summaries,
        "gate": {
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
        },
    }


def write_verdict_fixture(metadata: Dict, cases: Sequence[Dict], path: str) -> Path:
    """Atomically write metadata followed by deterministic JSONL verdict records."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    records = [
        {
            "record_type": "metadata",
            "schema_version": FIXTURE_SCHEMA_VERSION,
            **metadata,
        },
        *cases,
    ]
    with temporary_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            stream.write("\n")
    temporary_path.replace(output_path)
    return output_path


def write_ragas_report(report: Dict, report_dir: str, timestamp: Optional[str] = None) -> Path:
    """Write one stable JSON report under the isolated Ragas report directory."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{report_timestamp}.json"
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return output_path


def build_comparison_report(before: Dict, after: Dict, before_label: str, after_label: str) -> Dict:
    """Compare two replay reports while preserving judge spread in every delta."""
    comparisons = {}
    for metric in RAGAS_METRICS:
        before_metric = before["metrics"][metric]
        after_metric = after["metrics"][metric]
        delta = after_metric["mean"] - before_metric["mean"]
        noise_band = max(before_metric["stddev"], after_metric["stddev"])
        comparisons[metric] = {
            "before_mean": before_metric["mean"],
            "after_mean": after_metric["mean"],
            "delta": delta,
            "before_stddev": before_metric["stddev"],
            "after_stddev": after_metric["stddev"],
            "noise_band": noise_band,
            "direction_is_resolved": abs(delta) > noise_band,
        }
    return {
        "report_type": "ragas_generation_quality_comparison",
        "schema_version": REPORT_SCHEMA_VERSION,
        "metadata": {
            "before": before_label,
            "after": after_label,
            "interpretation": (
                "A single delta inside the reported judge spread is unresolved and must not be called an improvement."
            ),
        },
        "metrics": comparisons,
        "negative": {
            "before_abstention_rate": before["negative"]["abstention_rate"],
            "after_abstention_rate": after["negative"]["abstention_rate"],
        },
    }
