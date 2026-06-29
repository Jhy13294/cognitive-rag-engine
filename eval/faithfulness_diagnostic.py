"""Paid Faithfulness chain diagnostics for live Ragas investigations."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config import Config

from .golden import load_golden_set
from .ragas_live import (
    LIVE_PROFILES,
    LiveRagasJudge,
    RagasDependencyError,
    build_live_pipeline,
    clamp_live_score,
    collect_live_samples,
    installed_ragas_version,
)
from .schemas import GoldenExample


DEFAULT_GOLDEN_SET = "eval/golden_set.jsonl"
DEFAULT_KNOWLEDGE_PATH = "eval/fixtures/knowledge_base"
TARGET_QIDS: Tuple[str, ...] = ("q013",)
CONTROL_QIDS: Tuple[str, ...] = ("q001", "q016")
DIAGNOSTIC_SCHEMA_VERSION = "faithfulness-diagnostic-v1"


@dataclass(frozen=True)
class FaithfulnessStatementVerdict:
    """One NLI verdict emitted for a generated statement."""

    statement: str
    reason: str
    verdict: int


@dataclass(frozen=True)
class FaithfulnessRunTrace:
    """One repeated Faithfulness run with statement and verdict internals."""

    run_index: int
    value: float
    statements: List[str]
    verdicts: List[FaithfulnessStatementVerdict]


async def trace_faithfulness_once(
    metric,
    user_input: str,
    response: str,
    retrieved_contexts: Sequence[str],
    run_index: int,
) -> FaithfulnessRunTrace:
    """Run Ragas Faithfulness once and expose statement/NLI internals."""
    statements = await metric._create_statements(user_input, response)
    context = "\n".join(retrieved_contexts)
    verdict_output = await metric._create_verdicts(statements, context)
    value = metric._compute_score(verdict_output)
    return FaithfulnessRunTrace(
        run_index=run_index,
        value=clamp_live_score(value, f"faithfulness_diagnostic.run_{run_index}"),
        statements=[str(statement) for statement in statements],
        verdicts=[
            FaithfulnessStatementVerdict(
                statement=str(verdict.statement),
                reason=str(verdict.reason),
                verdict=int(verdict.verdict),
            )
            for verdict in verdict_output.statements
        ],
    )


async def trace_faithfulness_repetitions(
    metric,
    user_input: str,
    response: str,
    retrieved_contexts: Sequence[str],
    repetitions: int,
) -> List[FaithfulnessRunTrace]:
    """Run repeated Faithfulness traces over one fixed answer/context input."""
    return [
        await trace_faithfulness_once(
            metric,
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
            run_index=run_index,
        )
        for run_index in range(1, repetitions + 1)
    ]


def summarize_scores(values: Sequence[float]) -> Dict[str, float]:
    """Summarize one short diagnostic score vector."""
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2
    return {
        "mean": math.fsum(ordered) / len(ordered),
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
    }


def select_examples(
    examples: Sequence[GoldenExample],
    qids: Iterable[str],
) -> List[GoldenExample]:
    """Return golden examples in the requested qid order."""
    by_qid = {example.qid: example for example in examples}
    missing = [qid for qid in qids if qid not in by_qid]
    if missing:
        raise ValueError("Golden set is missing diagnostic qids: " + ", ".join(missing))
    return [by_qid[qid] for qid in qids]


async def build_faithfulness_diagnostic_report(
    *,
    golden_path: str,
    knowledge_path: str,
    profile: str,
    repetitions: int,
    include_controls: bool,
    qids_override: Optional[Sequence[str]] = None,
) -> Dict:
    """Collect a paid, content-bearing Faithfulness diagnostic report."""
    if repetitions < 2:
        raise ValueError("Faithfulness diagnostics require at least two repetitions.")

    examples = load_golden_set(golden_path)
    if qids_override:
        qids = list(qids_override)
    else:
        qids = list(TARGET_QIDS)
        if include_controls:
            qids.extend(CONTROL_QIDS)
    selected_examples = select_examples(examples, qids)
    pipeline = build_live_pipeline(profile, knowledge_path)
    samples = await collect_live_samples(pipeline, selected_examples, Config.RAGAS_TOP_K)
    sample_by_qid = {sample.qid: sample for sample in samples}
    judge = LiveRagasJudge()
    metric = judge.metrics["faithfulness"]

    cases = []
    for qid in qids:
        sample = sample_by_qid[qid]
        runs = await trace_faithfulness_repetitions(
            metric,
            user_input=sample.question,
            response=sample.answer,
            retrieved_contexts=sample.faithfulness_contexts,
            repetitions=repetitions,
        )
        cases.append(
            {
                "qid": qid,
                "capability": sample.capability,
                "role": classify_qid_role(qid),
                "question": sample.question,
                "answer": sample.answer,
                "ground_truth": sample.ground_truth,
                "contexts": sample.contexts,
                "faithfulness_contexts": sample.faithfulness_contexts,
                "faithfulness": summarize_scores([run.value for run in runs]),
                "runs": [
                    {
                        **asdict(run),
                        "verdicts": [asdict(verdict) for verdict in run.verdicts],
                    }
                    for run in runs
                ],
            }
        )

    return {
        "report_type": "faithfulness_diagnostic",
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "metadata": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
            "golden_path": golden_path,
            "knowledge_path": knowledge_path,
            "repetitions": repetitions,
            "target_qids": list(TARGET_QIDS),
            "control_qids": list(CONTROL_QIDS) if include_controls else [],
            "judge_model_id": Config.RAGAS_JUDGE_MODEL,
            "judge_temperature": Config.RAGAS_JUDGE_TEMPERATURE,
            "generator_model_id": Config.MODEL_NAME,
            "generator_temperature": Config.RAGAS_LIVE_GENERATION_TEMPERATURE,
            "ragas_version": installed_ragas_version(),
            "raw_evaluation_text_stored": True,
            "data_egress": (
                "questions, generated answers, and retrieved faithfulness contexts "
                "are sent to the configured judge"
            ),
        },
        "cases": cases,
    }


def classify_qid_role(qid: str) -> str:
    """Classify a qid's diagnostic purpose for report readers."""
    if qid in TARGET_QIDS:
        return "target_flipper"
    if qid in CONTROL_QIDS:
        return "stable_control"
    return "additional"


def write_diagnostic_report(report: Dict, report_dir: str, timestamp: Optional[str]) -> Path:
    """Write a raw-text diagnostic report."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"faithfulness-diagnostic-{report_timestamp}.json"
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the paid Faithfulness diagnostic CLI."""
    parser = argparse.ArgumentParser(
        description="Dump Ragas Faithfulness statement and NLI internals."
    )
    parser.add_argument("--golden-set", default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--knowledge-path", default=DEFAULT_KNOWLEDGE_PATH)
    parser.add_argument("--profile", choices=LIVE_PROFILES, default="baseline")
    parser.add_argument("--repetitions", type=int, default=Config.RAGAS_LIVE_REPETITIONS)
    parser.add_argument("--report-dir", default=Config.RAGAS_REPORT_DIR)
    parser.add_argument("--timestamp", default=None)
    parser.add_argument(
        "--qids",
        default=None,
        help="Comma-separated qids for a smaller resumable diagnostic run.",
    )
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip stable-control qids.",
    )
    return parser


def parse_qids(raw_qids: Optional[str]) -> Optional[List[str]]:
    """Parse an optional comma-separated qid override."""
    if raw_qids is None:
        return None
    qids = [item.strip() for item in raw_qids.split(",") if item.strip()]
    if not qids:
        raise ValueError("--qids must contain at least one qid when provided.")
    return qids


def main(argv: Optional[List[str]] = None) -> int:
    """Run the gated paid diagnostic only after explicit live Ragas opt-in."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if not Config.RAGAS_ENABLED:
            raise ValueError("RAGAS_ENABLED=true is required.")
        if not Config.RUN_RAGAS_EVAL:
            raise ValueError("RUN_RAGAS_EVAL=true is required for paid diagnostics.")
        Config.validate_ragas()
        report = asyncio.run(
            build_faithfulness_diagnostic_report(
                golden_path=args.golden_set,
                knowledge_path=args.knowledge_path,
                profile=args.profile,
                repetitions=args.repetitions,
                include_controls=not args.skip_controls,
                qids_override=parse_qids(args.qids),
            )
        )
        output_path = write_diagnostic_report(report, args.report_dir, args.timestamp)
        print(f"Faithfulness diagnostic written: {output_path}")
        return 0
    except (FileNotFoundError, ValueError, RagasDependencyError) as error:
        print(f"Faithfulness diagnostic error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
