import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple


def write_reports(report: Dict, report_dir: str, timestamp: str = None) -> Tuple[Path, Path]:
    """Write machine-readable JSON and human-readable Markdown reports."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = output_dir / f"{report_timestamp}.json"
    markdown_path = output_dir / f"{report_timestamp}.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, sort_keys=True)

    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown_report(report: Dict) -> str:
    """Render a concise Markdown retrieval-evaluation report."""
    metadata = report.get("metadata", {})
    lines = [
        "# Retrieval Evaluation Report",
        "",
        "## Metadata",
        "",
        f"- embedding_provider: `{metadata.get('embedding_provider', 'unknown')}`",
        f"- embedding_dimension: `{metadata.get('embedding_dimension', 'unknown')}`",
        f"- vector_store: `{metadata.get('vector_store', 'unknown')}`",
        f"- reranker: `{metadata.get('reranker', None)}`",
        f"- rerank_fetch_k: `{metadata.get('rerank_fetch_k', None)}`",
        f"- rerank_top_n: `{metadata.get('rerank_top_n', None)}`",
        f"- golden_count: `{metadata.get('golden_count', 'unknown')}`",
        f"- golden_version: `{metadata.get('golden_version', 'unknown')}`",
        f"- git_sha: `{metadata.get('git_sha', 'unknown')}`",
        f"- evaluator_contract: `{metadata.get('evaluator_contract', 'retrieve(question, top_k)')}`",
        "",
        "## Overall Metrics",
        "",
        "| k | hit_rate | MRR | recall | negative_empty_rate | negative_false_recall_rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for k, values in report.get("metrics", {}).items():
        lines.append(
            "| {k} | {hit_rate:.6f} | {mrr:.6f} | {recall:.6f} | {neg_empty:.6f} | {neg_false:.6f} |".format(
                k=k,
                hit_rate=values.get("hit_rate", 0.0),
                mrr=values.get("mrr", 0.0),
                recall=values.get("recall", 0.0),
                neg_empty=values.get("negative_empty_rate", 0.0),
                neg_false=values.get("negative_false_recall_rate", 0.0),
            )
        )

    lines.extend(["", "## Capability Metrics", ""])
    for capability, per_k in report.get("by_capability", {}).items():
        lines.extend([
            f"### {capability}",
            "",
            "| k | query_count | hit_rate | MRR | recall | negative_empty_rate | negative_false_recall_rate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for k, values in per_k.items():
            lines.append(
                "| {k} | {query_count} | {hit_rate:.6f} | {mrr:.6f} | {recall:.6f} | {neg_empty:.6f} | {neg_false:.6f} |".format(
                    k=k,
                    query_count=values.get("query_count", 0),
                    hit_rate=values.get("hit_rate", 0.0),
                    mrr=values.get("mrr", 0.0),
                    recall=values.get("recall", 0.0),
                    neg_empty=values.get("negative_empty_rate", 0.0),
                    neg_false=values.get("negative_false_recall_rate", 0.0),
                )
            )
        lines.append("")

    lines.extend(["## Low Recall Cases", ""])
    low_cases_by_k = report.get("low_recall_cases_by_k")
    if not low_cases_by_k:
        low_cases_by_k = {"max": report.get("low_recall_cases", [])}

    for k, low_cases in low_cases_by_k.items():
        lines.extend([f"### k={k}", ""])
        if not low_cases:
            lines.extend(["No low-recall cases.", ""])
            continue

        for case in low_cases:
            lines.extend([
                f"#### {case['qid']} ({case['capability']})",
                "",
                f"- question: {case['question']}",
                f"- relevant: `{json.dumps(case['relevant'], ensure_ascii=False)}`",
                f"- first_hit_rank: `{case['first_hit_rank'] if case['first_hit_rank'] is not None else 'miss'}`",
                f"- miss_reason: {case['miss_reason']}",
                "- retrieved:",
            ])
            for item in case.get("retrieved", []):
                lines.append(
                    "  - source={source}; chunk={chunk}; score={score:.6f}; preview={preview}".format(
                        source=item.get("source", ""),
                        chunk=item.get("chunk_index", "unknown"),
                        score=item.get("score", 0.0),
                        preview=str(item.get("content_preview", "")).replace("\n", " "),
                    )
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
