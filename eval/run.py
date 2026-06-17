import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from config import Config
from rerank import create_reranker
from .baseline import build_hash_retriever
from .golden import load_golden_set
from .metrics import evaluate_retriever
from .reporting import render_markdown_report, write_reports


DEFAULT_GOLDEN_SET = "eval/golden_set.jsonl"
DEFAULT_KNOWLEDGE_PATH = "eval/fixtures/knowledge_base"
DEFAULT_REPORT_DIR = "eval/reports"
PARENT_EXPANSION_SCORING_ERROR = (
    "Refusing to score --expand-parents output. Parent expansion is diagnostic/generation-side only; "
    "it collapses sibling child chunks and can inflate top-k retrieval metrics. "
    "Run --parent-child without --expand-parents for metrics of record."
)


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
        "--retrieval-mode",
        choices=["dense", "bm25", "hybrid"],
        default="dense",
        help="Retrieval mode for a single report.",
    )
    parser.add_argument(
        "--compare-hybrid",
        action="store_true",
        help="Run dense-only, bm25-only, and fused reports with reranker disabled.",
    )
    parser.add_argument("--hybrid-fetch-k", type=int, default=30, help="Candidate count per path before RRF fusion.")
    parser.add_argument("--rrf-k", type=int, default=Config.RRF_K, help="RRF rank constant.")
    parser.add_argument(
        "--hybrid-dense-weight",
        type=float,
        default=Config.HYBRID_DENSE_WEIGHT,
        help="Dense path weight for RRF.",
    )
    parser.add_argument(
        "--hybrid-sparse-weight",
        type=float,
        default=Config.HYBRID_SPARSE_WEIGHT,
        help="Sparse BM25 path weight for RRF.",
    )
    parser.add_argument("--bm25-k1", type=float, default=Config.BM25_K1, help="BM25 k1 parameter.")
    parser.add_argument("--bm25-b", type=float, default=Config.BM25_B, help="BM25 b parameter.")
    parser.add_argument("--parent-child", action="store_true", help="Use parent-child chunking.")
    parser.add_argument(
        "--expand-parents",
        action="store_true",
        help="Deprecated diagnostic flag; scored retrieval evaluation refuses parent-expanded output.",
    )
    parser.add_argument(
        "--no-parent-expand",
        action="store_true",
        help="Compatibility flag; parent expansion is disabled by default for retrieval metrics.",
    )
    parser.add_argument("--parent-chunk-size", type=int, default=Config.PARENT_CHUNK_SIZE, help="Parent chunk size.")
    parser.add_argument(
        "--parent-chunk-overlap",
        type=int,
        default=Config.PARENT_CHUNK_OVERLAP,
        help="Parent chunk overlap.",
    )
    parser.add_argument("--child-chunk-size", type=int, default=Config.CHILD_CHUNK_SIZE, help="Child chunk size.")
    parser.add_argument(
        "--child-chunk-overlap",
        type=int,
        default=Config.CHILD_CHUNK_OVERLAP,
        help="Child chunk overlap.",
    )
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

    if expanded_parent_scoring_requested(args):
        print(PARENT_EXPANSION_SCORING_ERROR, file=sys.stderr)
        return 2

    examples = load_golden_set(args.golden_set)
    if args.compare_hybrid:
        report = build_hybrid_comparison_report(args, examples)
        if not args.no_write_report:
            json_path, markdown_path = write_reports(report, args.report_dir)
            print(f"Report written: {json_path}")
            print(f"Report written: {markdown_path}")

        if not args.quiet:
            print(render_markdown_report(report))
        return 0

    reranker = create_reranker(
        provider_name=args.rerank_provider,
        enabled=args.rerank_provider != "none",
        fetch_k=args.rerank_fetch_k,
        top_n=args.rerank_top_n,
    )
    fetch_k = single_report_fetch_k(args, reranker is not None)
    retrieve, baseline_metadata = build_hash_retriever(
        knowledge_path=args.knowledge_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_dimension=args.embedding_dimension,
        reranker=reranker,
        fetch_k=fetch_k,
        retrieval_mode=args.retrieval_mode,
        rrf_k=args.rrf_k,
        dense_weight=args.hybrid_dense_weight,
        sparse_weight=args.hybrid_sparse_weight,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
        parent_child_enabled=args.parent_child,
        expand_parent_context=args.expand_parents and not args.no_parent_expand,
        parent_chunk_size=args.parent_chunk_size,
        parent_chunk_overlap=args.parent_chunk_overlap,
        child_chunk_size=args.child_chunk_size,
        child_chunk_overlap=args.child_chunk_overlap,
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


def expanded_parent_scoring_requested(args) -> bool:
    """Return True when CLI args would score parent-expanded sources."""
    return bool(args.parent_child and args.expand_parents and not args.no_parent_expand)


def build_hybrid_comparison_report(args, examples) -> Dict:
    """Build dense-only, bm25-only, and fused reports with reranker disabled."""
    modes = [
        ("dense-only", "dense"),
        ("bm25-only", "bm25"),
        ("fused", "hybrid"),
    ]
    reports = {}
    common_metadata = base_metadata(args, examples)

    for label, retrieval_mode in modes:
        retrieve, baseline_metadata = build_hash_retriever(
            knowledge_path=args.knowledge_path,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_dimension=args.embedding_dimension,
            reranker=None,
            fetch_k=args.hybrid_fetch_k if retrieval_mode == "hybrid" else None,
            retrieval_mode=retrieval_mode,
            rrf_k=args.rrf_k,
            dense_weight=args.hybrid_dense_weight,
            sparse_weight=args.hybrid_sparse_weight,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            parent_child_enabled=args.parent_child,
            expand_parent_context=args.expand_parents and not args.no_parent_expand,
            parent_chunk_size=args.parent_chunk_size,
            parent_chunk_overlap=args.parent_chunk_overlap,
            child_chunk_size=args.child_chunk_size,
            child_chunk_overlap=args.child_chunk_overlap,
        )
        reports[label] = evaluate_retriever(
            retrieve=retrieve,
            examples=examples,
            k_values=args.k,
            match_scope=args.match_scope,
            metadata={
                **baseline_metadata,
                **common_metadata,
                "comparison_label": label,
                "reranker": None,
            },
        )

    metadata = {
        **common_metadata,
        "comparison_modes": [label for label, _ in modes],
        "reranker": None,
        "rrf_k": args.rrf_k,
        "hybrid_dense_weight": args.hybrid_dense_weight,
        "hybrid_sparse_weight": args.hybrid_sparse_weight,
        "bm25_k1": args.bm25_k1,
        "bm25_b": args.bm25_b,
        "hybrid_fetch_k": args.hybrid_fetch_k,
        "evaluator_contract": "retrieve(question, top_k)",
    }
    if args.parent_child:
        metadata.update(
            {
                "parent_child_enabled": True,
                "expand_parent_context": args.expand_parents and not args.no_parent_expand,
                "parent_chunk_size": args.parent_chunk_size,
                "parent_chunk_overlap": args.parent_chunk_overlap,
                "child_chunk_size": args.child_chunk_size,
                "child_chunk_overlap": args.child_chunk_overlap,
            }
        )

    return {
        "report_type": "hybrid_comparison",
        "metadata": metadata,
        "k_values": sorted(set(args.k)),
        "reports": reports,
        "comparison": summarize_comparison(reports),
    }


def summarize_comparison(reports: Dict[str, Dict]) -> Dict:
    """Extract the T06 comparison slice from full reports."""
    comparison = {}
    for label, report in reports.items():
        comparison[label] = {
            "metrics": report.get("metrics", {}),
            "exact_name": report.get("by_capability", {}).get("exact_name", {}),
            "long_tail": report.get("by_capability", {}).get("long_tail", {}),
        }
    return comparison


def base_metadata(args, examples) -> Dict:
    """Return metadata shared by single and comparison reports."""
    return {
        "golden_path": args.golden_set,
        "golden_count": len(examples),
        "golden_version": file_sha256(args.golden_set),
        "git_sha": git_sha(),
        "k_values": sorted(set(args.k)),
        "match_scope": args.match_scope,
        "evaluator_contract": "retrieve(question, top_k)",
    }


def single_report_fetch_k(args, has_reranker: bool) -> Optional[int]:
    """Return candidate count for single-report retrieval."""
    if args.retrieval_mode == "hybrid" and has_reranker:
        return max(args.hybrid_fetch_k, args.rerank_fetch_k)
    if args.retrieval_mode == "hybrid":
        return args.hybrid_fetch_k
    if has_reranker:
        return args.rerank_fetch_k
    return None


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
