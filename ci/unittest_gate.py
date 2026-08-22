import argparse
import json
import os
import sys
import unittest
from collections import Counter
from pathlib import Path
from typing import Dict

EXPECTED_RAN = 339
EXPECTED_FAILURES = 0
EXPECTED_ERRORS = 0
EXPECTED_SKIPPED = 19
EXPECTED_SKIP_BUCKETS: Dict[str, int] = {
    "REDIS_URL is not set; live Redis cross-loop test is gated": 1,
    "REDIS_URL is not set; live Redis smoke is gated": 1,
    "Set QDRANT_URL and install qdrant-client to run Qdrant integration tests.": 3,
    "Set RUN_BGE_REAL_EMBEDDING=1 to exercise the real local BGE model.": 1,
    "Set RUN_COHERE_RERANK_INTEGRATION=1 and COHERE_API_KEY to run Cohere rerank integration.": 1,
    "Set RUN_E2E_INTEGRATION=1 plus QDRANT_URL, REDIS_URL, and METADATA_DB_URL or MYSQL_HOST/USER/DATABASE.": 6,
    "Set RUN_MYSQL_ACL_INTEGRATION=1 and METADATA_DB_URL or MYSQL_HOST/USER/DATABASE to run real-MySQL ACL tests.": 5,
    "Set RUN_RAPIDOCR_TEST=1 to run the gated RapidOCR integration test": 1,
}
OFFLINE_ENV_VARS = (
    "RUN_BGE_REAL_EMBEDDING",
    "RUN_COHERE_RERANK_INTEGRATION",
    "RUN_E2E_INTEGRATION",
    "RUN_MYSQL_ACL_INTEGRATION",
    "RUN_RAPIDOCR_TEST",
    "COHERE_API_KEY",
    "QDRANT_URL",
    "REDIS_URL",
    "METADATA_DB_URL",
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the full unittest gate CLI."""
    parser = argparse.ArgumentParser(
        description="Run the full offline unittest suite with pinned skip accounting."
    )
    parser.add_argument("--start-directory", default="tests")
    parser.add_argument("--pattern", default="test*.py")
    parser.add_argument("--output", default="ci-artifacts/unittest-summary.json")
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail when ran/failure/error/skip counts or skip buckets drift.",
    )
    parser.add_argument(
        "--offline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Blank gated integration environment variables before discovery.",
    )
    parser.add_argument("-v", "--verbosity", type=int, default=2)
    return parser


def main(argv=None) -> int:
    """Run unittest discovery and enforce deterministic CI accounting."""
    args = build_arg_parser().parse_args(argv)
    blanked_env = clear_offline_environment() if args.offline else []
    if args.offline:
        apply_offline_config_defaults()
    suite = unittest.defaultTestLoader.discover(
        start_dir=args.start_directory,
        pattern=args.pattern,
    )
    result = unittest.TextTestRunner(verbosity=args.verbosity).run(suite)
    skip_buckets = Counter(reason for _, reason in result.skipped)
    summary = {
        "ran": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "skip_buckets": dict(sorted(skip_buckets.items())),
        "offline_environment": {
            "enabled": args.offline,
            "blanked": blanked_env,
        },
        "expected": {
            "ran": EXPECTED_RAN,
            "failures": EXPECTED_FAILURES,
            "errors": EXPECTED_ERRORS,
            "skipped": EXPECTED_SKIPPED,
            "skip_buckets": dict(sorted(EXPECTED_SKIP_BUCKETS.items())),
        },
    }
    write_summary(summary, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))

    if not result.wasSuccessful():
        return 1
    if args.strict:
        mismatches = accounting_mismatches(summary)
        if mismatches:
            for mismatch in mismatches:
                print(mismatch, file=sys.stderr)
            return 1
    return 0


def clear_offline_environment() -> list:
    """Blank gated service variables so push CI remains a secretless offline gate."""
    blanked = []
    for name in OFFLINE_ENV_VARS:
        if os.environ.get(name):
            blanked.append(name)
        # Keep the key present so later dotenv loading cannot rehydrate a local
        # service URL and accidentally turn an offline CI run into a live test.
        os.environ[name] = ""
    return blanked


def apply_offline_config_defaults() -> None:
    """Keep pure unit tests configurable while live Redis tests stay skipped."""
    from config import Config

    Config.REDIS_URL = "redis://127.0.0.1:1/0"


def write_summary(summary: Dict, output_path: str) -> None:
    """Write the CI summary with stable formatting for inspection."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def accounting_mismatches(summary: Dict) -> list:
    """Return human-readable accounting mismatches."""
    expected = summary["expected"]
    mismatches = []
    for field in ("ran", "failures", "errors", "skipped"):
        if summary[field] != expected[field]:
            mismatches.append(
                f"unittest {field} drifted: actual={summary[field]} expected={expected[field]}"
            )
    if summary["skip_buckets"] != expected["skip_buckets"]:
        mismatches.append(
            "unittest skip buckets drifted: "
            f"actual={summary['skip_buckets']} expected={expected['skip_buckets']}"
        )
    return mismatches


if __name__ == "__main__":
    raise SystemExit(main())
