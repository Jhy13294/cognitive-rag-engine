import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from config import Config
from rag.pipeline import RAG_SYSTEM_PROMPT_VERSION

from .faithfulness_prompt import STATEMENT_PROMPT_VERSION
from .golden import load_golden_set
from .ragas_evaluation import (
    FIXTURE_SCHEMA_VERSION,
    build_comparison_report,
    build_replay_report,
    load_verdict_fixture,
    write_ragas_report,
    write_verdict_fixture,
)
from .ragas_live import (
    FAITHFULNESS_CONTEXT_FORMAT,
    LIVE_PROFILES,
    LiveRagasJudge,
    RagasDependencyError,
    build_live_fixture_metadata,
    build_live_pipeline,
    collect_live_samples,
    installed_ragas_version,
    record_live_verdicts,
)

DEFAULT_GOLDEN_SET = "eval/golden_set.jsonl"
DEFAULT_KNOWLEDGE_PATH = "eval/fixtures/knowledge_base"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the isolated Ragas evaluation CLI."""
    parser = argparse.ArgumentParser(
        description="Run deterministic Ragas fixture replay or gated live evaluation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser(
        "replay", help="Replay recorded verdicts without network access."
    )
    add_common_arguments(replay)
    replay.add_argument("--fixture", default=Config.RAGAS_FIXTURE_PATH)

    live = subparsers.add_parser("live", help="Run the gated external Ragas judge.")
    add_common_arguments(live)
    live.add_argument("--knowledge-path", default=DEFAULT_KNOWLEDGE_PATH)
    live.add_argument("--profile", choices=LIVE_PROFILES, default="baseline")
    live.add_argument("--repetitions", type=int, default=Config.RAGAS_LIVE_REPETITIONS)
    live.add_argument("--fixture", default=Config.RAGAS_FIXTURE_PATH)
    live.add_argument(
        "--candidate-fixture",
        default=None,
        help="Always write a non-gating candidate fixture for calibration or review.",
    )
    live.add_argument(
        "--refresh-fixture",
        action="store_true",
        help="Replace the replay fixture only when the live quality gate passes.",
    )

    compare = subparsers.add_parser(
        "compare",
        help="Compare two recorded live profiles with judge spread attached.",
    )
    add_common_arguments(compare)
    compare.add_argument("--before", required=True, help="Before-profile fixture path.")
    compare.add_argument("--after", required=True, help="After-profile fixture path.")
    compare.add_argument("--before-label", default="before")
    compare.add_argument("--after-label", default="after")
    return parser


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add report and golden-set arguments shared by all commands."""
    parser.add_argument("--golden-set", default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--report-dir", default=Config.RAGAS_REPORT_DIR)
    parser.add_argument(
        "--timestamp", default=None, help="Stable report filename timestamp for automation."
    )
    parser.add_argument("--no-write-report", action="store_true")
    parser.add_argument("--quiet", action="store_true")


def thresholds_from_config() -> Dict[str, float]:
    """Return per-case quality thresholds keyed by stable metric name."""
    return {
        "faithfulness": Config.RAGAS_FAITHFULNESS_THRESHOLD,
        "answer_relevance": Config.RAGAS_ANSWER_RELEVANCE_THRESHOLD,
        "context_precision": Config.RAGAS_CONTEXT_PRECISION_THRESHOLD,
        "context_recall": Config.RAGAS_CONTEXT_RECALL_THRESHOLD,
    }


def margins_from_config() -> Dict[str, float]:
    """Return configured noise margins keyed by stable metric name."""
    return {
        "faithfulness": Config.RAGAS_FAITHFULNESS_MARGIN,
        "answer_relevance": Config.RAGAS_ANSWER_RELEVANCE_MARGIN,
        "context_precision": Config.RAGAS_CONTEXT_PRECISION_MARGIN,
        "context_recall": Config.RAGAS_CONTEXT_RECALL_MARGIN,
    }


def replay_report(fixture: Dict, golden_path: str) -> Dict:
    """Build one replay report using the configured gates."""
    examples = load_golden_set(golden_path)
    return build_replay_report(
        fixture=fixture,
        examples=examples,
        golden_path=golden_path,
        thresholds=thresholds_from_config(),
        margins=margins_from_config(),
        sigma_multiplier=Config.RAGAS_SIGMA_MULTIPLIER,
        negative_abstention_threshold=Config.RAGAS_NEGATIVE_ABSTENTION_THRESHOLD,
        expected_judge_model=Config.RAGAS_JUDGE_MODEL,
        installed_ragas_version=installed_ragas_version(),
        expected_generation_prompt_version=RAG_SYSTEM_PROMPT_VERSION,
        expected_statement_prompt_version=STATEMENT_PROMPT_VERSION,
        expected_faithfulness_context_format=FAITHFULNESS_CONTEXT_FORMAT,
    )


def run_replay(args) -> int:
    """Run the deterministic no-network fixture gate."""
    fixture = load_verdict_fixture(args.fixture)
    report = replay_report(fixture, args.golden_set)
    emit_report(report, args)
    return emit_gate_result(report)


def run_live(args) -> int:
    """Generate fixed answers, repeat the external judge, and optionally refresh fixture."""
    if not Config.RUN_RAGAS_EVAL:
        print(
            "Live Ragas evaluation is disabled. Set RUN_RAGAS_EVAL=true explicitly; "
            "this sends question, answer, context text, and ground truth to external endpoints.",
            file=sys.stderr,
        )
        return 2
    if args.repetitions < 2:
        raise ValueError("--repetitions must be at least 2.")

    examples = load_golden_set(args.golden_set)

    async def capture():
        pipeline = build_live_pipeline(args.profile, args.knowledge_path)
        samples = await collect_live_samples(pipeline, examples, Config.RAGAS_TOP_K)
        judge = LiveRagasJudge()
        cases = await record_live_verdicts(samples, judge, args.repetitions)
        return cases

    cases = asyncio.run(capture())
    metadata = build_live_fixture_metadata(
        golden_path=args.golden_set,
        profile=args.profile,
        repetitions=args.repetitions,
    )
    fixture = {
        "metadata": {
            "record_type": "metadata",
            "schema_version": FIXTURE_SCHEMA_VERSION,
            **metadata,
        },
        "cases": cases,
    }
    report = replay_report(fixture, args.golden_set)
    emit_report(report, args)
    exit_code = emit_gate_result(report)
    if args.candidate_fixture:
        if Path(args.candidate_fixture).resolve() == Path(args.fixture).resolve():
            raise ValueError("--candidate-fixture must not overwrite the configured gate fixture.")
        candidate_path = write_verdict_fixture(metadata, cases, args.candidate_fixture)
        print(f"Candidate fixture written: {candidate_path}")
    if args.refresh_fixture:
        if exit_code != 0:
            print("Fixture not refreshed because the live quality gate failed.", file=sys.stderr)
        else:
            fixture_path = write_verdict_fixture(metadata, cases, args.fixture)
            print(f"Fixture refreshed: {fixture_path}")
    return exit_code


def run_compare(args) -> int:
    """Compare two compatible live recordings without treating deltas as deterministic."""
    before_fixture = load_verdict_fixture(args.before)
    after_fixture = load_verdict_fixture(args.after)
    before_metadata = before_fixture["metadata"]
    after_metadata = after_fixture["metadata"]
    for field in (
        "judge_model_id",
        "ragas_version",
        "temperature",
        "repetitions",
        "statement_prompt_version",
        "faithfulness_context_format",
        "gated_metrics",
        "reported_only_metrics",
    ):
        if before_metadata[field] != after_metadata[field]:
            raise ValueError(
                f"Cannot compare fixtures with different {field}: "
                f"{before_metadata[field]} != {after_metadata[field]}"
            )
    before_report = replay_report(before_fixture, args.golden_set)
    after_report = replay_report(after_fixture, args.golden_set)
    report = build_comparison_report(
        before_report,
        after_report,
        before_label=args.before_label,
        after_label=args.after_label,
    )
    if not args.no_write_report:
        path = write_ragas_report(report, args.report_dir, timestamp=args.timestamp)
        print(f"Report written: {path}")
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def emit_report(report: Dict, args) -> None:
    """Write and optionally print an isolated Ragas report."""
    if not args.no_write_report:
        path = write_ragas_report(report, args.report_dir, timestamp=args.timestamp)
        print(f"Report written: {path}")
    if not args.quiet:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


def emit_gate_result(report: Dict) -> int:
    """Print actionable gate failures and return a process exit code."""
    gate = report["gate"]
    if gate["passed"]:
        print("Ragas replay gate passed.")
        return 0
    for failure in gate["failures"]:
        print(format_failure(failure), file=sys.stderr)
    return 1


def format_failure(failure: Dict) -> str:
    """Format one gate failure without leaking raw evaluation content."""
    kind = failure.get("kind", "unknown")
    if kind == "baseline_invalidated":
        return (
            "Ragas baseline invalidated | "
            f"field={failure['field']} | actual={failure['actual']} | expected={failure['expected']}"
        )
    return (
        "Ragas gate failed | "
        f"kind={kind} | qid={failure.get('qid')} | metric={failure.get('metric')} | "
        f"score={float(failure.get('score', 0.0)):.6f} | "
        f"threshold={float(failure.get('threshold', 0.0)):.6f}"
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Run the isolated Ragas CLI without touching deterministic retrieval reports."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        if not Config.RAGAS_ENABLED:
            raise ValueError("RAGAS_ENABLED=true is required to run the Ragas command.")
        Config.validate_ragas()
        if args.command == "replay":
            return run_replay(args)
        if args.command == "live":
            return run_live(args)
        if args.command == "compare":
            return run_compare(args)
        parser.error(f"Unsupported command: {args.command}")
    except (FileNotFoundError, ValueError, RagasDependencyError) as error:
        print(f"Ragas evaluation error: {error}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
