import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

BASELINES = {
    "dense": {
        "args": [],
        "expected_mrr_at_3": "0.677083",
    },
    "hybrid": {
        "args": ["--retrieval-mode", "hybrid"],
        "expected_mrr_at_3": "1.000000",
    },
    "parent_child": {
        "args": ["--parent-child"],
        # Exact source-slice chunk semantics replaced the former fallback slices.
        "expected_mrr_at_3": "0.614583",
    },
    "rerank": {
        "args": ["--rerank-provider", "deterministic"],
        "expected_mrr_at_3": "1.000000",
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the retrieval baseline gate CLI."""
    parser = argparse.ArgumentParser(
        description="Run deterministic retrieval baselines and assert exact MRR@3 values."
    )
    parser.add_argument("--report-dir", default="ci-artifacts/retrieval-baselines")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--golden-set", default="eval/golden_set.jsonl")
    parser.add_argument("--knowledge-path", default="eval/fixtures/knowledge_base")
    return parser


def main(argv=None) -> int:
    """Run all configured retrieval baseline profiles."""
    args = build_arg_parser().parse_args(argv)
    root = Path(args.report_dir)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for profile, spec in BASELINES.items():
        report = run_eval_profile(
            profile=profile,
            profile_args=list(spec["args"]),
            report_root=root,
            python_executable=args.python,
            golden_set=args.golden_set,
            knowledge_path=args.knowledge_path,
        )
        summaries.append(assert_profile_mrr(profile, report, spec["expected_mrr_at_3"]))

    summary_path = root / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def run_eval_profile(
    profile: str,
    profile_args: List[str],
    report_root: Path,
    python_executable: str,
    golden_set: str,
    knowledge_path: str,
) -> Dict:
    """Run eval.run for one profile and return its JSON report."""
    profile_dir = report_root / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    for path in profile_dir.glob("*.json"):
        path.unlink()
    command = [
        python_executable,
        "-m",
        "eval.run",
        "--no-cache",
        "--golden-set",
        golden_set,
        "--knowledge-path",
        knowledge_path,
        "--k",
        "3",
        "--quiet",
        "--report-dir",
        str(profile_dir),
        *profile_args,
    ]
    subprocess.run(command, check=True)
    json_reports = sorted(profile_dir.glob("*.json"))
    if len(json_reports) != 1:
        raise RuntimeError(
            f"Expected exactly one JSON report for {profile}, found {len(json_reports)}."
        )
    return json.loads(json_reports[0].read_text(encoding="utf-8"))


def assert_profile_mrr(profile: str, report: Dict, expected_mrr_at_3: str) -> Dict:
    """Assert exact six-decimal MRR@3 for one report."""
    actual = format_metric(report["metrics"]["3"]["mrr"])
    if actual != expected_mrr_at_3:
        raise AssertionError(
            f"{profile} MRR@3 drifted: actual={actual} expected={expected_mrr_at_3}"
        )
    return {
        "profile": profile,
        "mrr_at_3": actual,
        "expected_mrr_at_3": expected_mrr_at_3,
    }


def format_metric(value) -> str:
    """Format a metric exactly as the frozen baseline contract expects."""
    return f"{float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
