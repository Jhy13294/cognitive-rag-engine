import argparse
import hashlib
import subprocess
from pathlib import Path
from typing import List

from .baseline import build_hash_retriever
from .golden import load_golden_set
from .metrics import evaluate_retriever
from .reporting import render_markdown_report, write_reports
from rerank import create_reranker


DEFAULT_GOLDEN_SET = "eval/golden_set.jsonl"
DEFAULT_KNOWLEDGE_PATH = "eval/fixtures/knowledge_base"
DEFAULT_REPORT_DIR = "eval/reports"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the retrieval-evaluation CLI parser."""
    parser = argparse.ArgumentParser(description="Run deterministic retrieval evaluation.")
    parser.add_argument("--golden-set", default=DEFAULT_GOLDEN_SET, help="JSONL golden set path.")
    parser.add_argument("--knowledge-path", default=DEFAULT_KNOWLEDGE_PATH, help="Knowledge fixture path.")
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR, help="Directory for report artifacts.")
    parser.add_argument("--k", nargs="+", type=int, default=[3, 5, 10], help="K values to report.")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size for baseline indexing.")
    parser.add_argument("--chunk-overlap", type=int, default=80, help="Chunk overlap for baseline indexing.")
    parser.add_argument("--embedding-dimension", type=int, default=64, help="Hash embedding dimension.")
    parser.add_argument("--match-scope", choices=["source", "chunk"], default="source", help="Hit matching scope.")
    parser.add_argument(
        "--rerank-provider",
        choices=["none", "deterministic", "cohere"],
        default="none",
        help="Optional reranker provider for retrieval evaluation.",
    )
    parser.add_argument("--rerank-fetch-k", type=int, default=30, help="Dense candidate count before rerank.")
    parser.add_argument("--rerank-top-n", type=int, default=5, help="Default number of reranked candidates.")
    parser.add_argument("--fail-under-hit-rate", type=float, default=None, help="Fail if any hit_rate@k is below this value.")
    parser.add_argument("--no-write-report", action="store_true", help="Print only; do not write report files.")
    parser.add_argument("--quiet", action="store_true", help="Do not print the Markdown report.")
    return parser


def main(argv: List[str] = None) -> int:
    """Run the deterministic hash baseline evaluation."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    examples = load_golden_set(args.golden_set)
    reranker = create_reranker(
        provider_name=args.rerank_provider,
        enabled=args.rerank_provider != "none",
        fetch_k=args.rerank_fetch_k,
        top_n=args.rerank_top_n,
    )
    retrieve, baseline_metadata = build_hash_retriever(
        knowledge_path=args.knowledge_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_dimension=args.embedding_dimension,
        reranker=reranker,
        fetch_k=args.rerank_fetch_k if reranker else None,
    )

    metadata = {
        **baseline_metadata,
        "golden_path": args.golden_set,
        "golden_count": len(examples),
        "golden_version": file_sha256(args.golden_set),
        "git_sha": git_sha(),
        "k_values": sorted(set(args.k)),
        "match_scope": args.match_scope,
        "evaluator_contract": "retrieve(question, top_k)",
    }
    report = evaluate_retriever(
        retrieve=retrieve,
        examples=examples,
        k_values=args.k,
        match_scope=args.match_scope,
        metadata=metadata,
    )

    if not args.no_write_report:
        json_path, markdown_path = write_reports(report, args.report_dir)
        print(f"Report written: {json_path}")
        print(f"Report written: {markdown_path}")

    if not args.quiet:
        print(render_markdown_report(report))

    if args.fail_under_hit_rate is not None:
        failed = [
            k for k, values in report["metrics"].items()
            if values["hit_rate"] < args.fail_under_hit_rate
        ]
        if failed:
            print(f"Evaluation failed: hit_rate below threshold for k={', '.join(failed)}")
            return 1

    return 0


def file_sha256(path: str) -> str:
    """Return a SHA256 fingerprint for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha() -> str:
    """Return the current short git SHA, or unknown outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
