import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from eval.baseline import build_hash_retriever
from eval.golden import load_golden_set
from eval.metrics import evaluate_retriever
from eval.run import main as eval_main
from eval.schemas import GoldenExample, RelevantItem


def make_source(source: str, chunk_index: int = 0, score: float = 1.0, metadata=None):
    source_metadata = {"source": source, "chunk_index": chunk_index}
    if metadata:
        source_metadata.update(metadata)
    return SimpleNamespace(
        content=f"content from {source}",
        score=score,
        metadata=source_metadata,
    )


class RetrievalEvaluationTests(unittest.TestCase):
    def test_load_golden_set_requires_relevant_list(self):
        examples = load_golden_set("eval/golden_set.jsonl")

        self.assertEqual(len(examples), 20)
        self.assertEqual(examples[0].capability, "exact_name")
        self.assertIsInstance(examples[0].relevant, list)
        self.assertEqual(examples[-1].capability, "negative")
        self.assertEqual(examples[-1].relevant, [])

    def test_metrics_cover_hit_mrr_recall_and_negative_rates(self):
        examples = [
            GoldenExample(
                qid="q1",
                question="first",
                relevant=[RelevantItem(source="a.md"), RelevantItem(source="b.md")],
                capability="exact_name",
                note="manual",
            ),
            GoldenExample(
                qid="q2",
                question="second",
                relevant=[RelevantItem(source="d.md")],
                capability="paraphrase",
                note="manual",
            ),
            GoldenExample(
                qid="q3",
                question="negative",
                relevant=[],
                capability="negative",
                note="manual",
            ),
        ]

        def retrieve(question: str, top_k: int):
            if question == "first":
                return [make_source("a.md"), make_source("c.md"), make_source("b.md")][:top_k]
            if question == "second":
                return [make_source("c.md"), make_source("d.md")][:top_k]
            return [make_source("noise.md")][:top_k]

        report = evaluate_retriever(retrieve, examples, k_values=[1, 3])

        self.assertEqual(report["metrics"]["1"]["positive_count"], 2)
        self.assertEqual(report["metrics"]["1"]["negative_count"], 1)
        self.assertEqual(report["metrics"]["1"]["hit_rate"], 0.5)
        self.assertEqual(report["metrics"]["1"]["mrr"], 0.5)
        self.assertEqual(report["metrics"]["1"]["recall"], 0.25)
        self.assertEqual(report["metrics"]["1"]["negative_false_recall_rate"], 1.0)

        self.assertEqual(report["metrics"]["3"]["hit_rate"], 1.0)
        self.assertEqual(report["metrics"]["3"]["mrr"], 0.75)
        self.assertEqual(report["metrics"]["3"]["recall"], 1.0)

    def test_evaluator_calls_retrieve_callable_once_with_max_k(self):
        examples = [
            GoldenExample(
                qid="q1",
                question="query",
                relevant=[RelevantItem(source="a.md")],
                capability="exact_name",
                note="manual",
            )
        ]
        calls = []

        def retrieve(question: str, top_k: int):
            calls.append((question, top_k))
            return [make_source("a.md")]

        evaluate_retriever(retrieve, examples, k_values=[3, 5, 10])

        self.assertEqual(calls, [("query", 10)])

    def test_evaluator_rejects_parent_expanded_metadata(self):
        examples = [
            GoldenExample(
                qid="q1",
                question="query",
                relevant=[RelevantItem(source="a.md")],
                capability="exact_name",
                note="manual",
            )
        ]
        calls = []

        def retrieve(question: str, top_k: int):
            calls.append((question, top_k))
            return [make_source("a.md")]

        with self.assertRaisesRegex(ValueError, "parent-expanded retrieval results"):
            evaluate_retriever(
                retrieve,
                examples,
                k_values=[3],
                metadata={"expand_parent_context": True},
            )

        self.assertEqual(calls, [])

    def test_evaluator_rejects_parent_expanded_sources(self):
        examples = [
            GoldenExample(
                qid="q1",
                question="query",
                relevant=[RelevantItem(source="a.md")],
                capability="exact_name",
                note="manual",
            )
        ]

        def retrieve(question: str, top_k: int):
            return [
                make_source(
                    "a.md",
                    metadata={
                        "parent_expanded": True,
                        "collapsed_child_count": 2,
                    },
                )
            ]

        with self.assertRaisesRegex(ValueError, "parent-expanded retrieval results"):
            evaluate_retriever(retrieve, examples, k_values=[3])

    def test_same_evaluator_detects_worse_retriever(self):
        examples = [
            GoldenExample(
                qid="q1",
                question="query one",
                relevant=[RelevantItem(source="a.md")],
                capability="exact_name",
                note="manual",
            ),
            GoldenExample(
                qid="q2",
                question="query two",
                relevant=[RelevantItem(source="b.md")],
                capability="paraphrase",
                note="manual",
            ),
        ]

        def good_retrieve(question: str, top_k: int):
            return [make_source("a.md" if question == "query one" else "b.md")]

        def bad_retrieve(question: str, top_k: int):
            return [make_source("noise.md")]

        good_report = evaluate_retriever(good_retrieve, examples, k_values=[1])
        bad_report = evaluate_retriever(bad_retrieve, examples, k_values=[1])

        self.assertGreater(
            good_report["metrics"]["1"]["hit_rate"], bad_report["metrics"]["1"]["hit_rate"]
        )
        self.assertGreater(good_report["metrics"]["1"]["mrr"], bad_report["metrics"]["1"]["mrr"])

    def test_hash_baseline_is_deterministic(self):
        examples = load_golden_set("eval/golden_set.jsonl")
        retrieve_one, metadata_one = build_hash_retriever(
            "eval/fixtures/knowledge_base",
            chunk_size=500,
            chunk_overlap=80,
            embedding_dimension=64,
        )
        retrieve_two, metadata_two = build_hash_retriever(
            "eval/fixtures/knowledge_base",
            chunk_size=500,
            chunk_overlap=80,
            embedding_dimension=64,
        )

        report_one = evaluate_retriever(
            retrieve_one, examples, k_values=[3, 5, 10], metadata=metadata_one
        )
        report_two = evaluate_retriever(
            retrieve_two, examples, k_values=[3, 5, 10], metadata=metadata_two
        )

        self.assertEqual(report_one["metrics"], report_two["metrics"])
        self.assertEqual(report_one["by_capability"], report_two["by_capability"])
        self.assertEqual(report_one["low_recall_cases"], report_two["low_recall_cases"])

    def test_eval_run_writes_json_and_markdown_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code = eval_main(
                [
                    "--golden-set",
                    "eval/golden_set.jsonl",
                    "--knowledge-path",
                    "eval/fixtures/knowledge_base",
                    "--report-dir",
                    temp_dir,
                    "--quiet",
                ]
            )

            report_files = list(Path(temp_dir).glob("*"))

        self.assertEqual(exit_code, 0)
        self.assertTrue(any(path.suffix == ".json" for path in report_files))
        self.assertTrue(any(path.suffix == ".md" for path in report_files))

    def test_eval_run_rejects_expand_parents_for_scored_metrics(self):
        exit_code = eval_main(
            [
                "--golden-set",
                "eval/golden_set.jsonl",
                "--knowledge-path",
                "eval/fixtures/knowledge_base",
                "--parent-child",
                "--expand-parents",
                "--no-write-report",
                "--quiet",
            ]
        )

        self.assertEqual(exit_code, 2)

    def test_eval_run_writes_hybrid_comparison_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            exit_code = eval_main(
                [
                    "--golden-set",
                    "eval/golden_set.jsonl",
                    "--knowledge-path",
                    "eval/fixtures/knowledge_base",
                    "--report-dir",
                    temp_dir,
                    "--compare-hybrid",
                    "--quiet",
                ]
            )

            json_path = next(Path(temp_dir).glob("*.json"))
            report = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["report_type"], "hybrid_comparison")
        self.assertEqual(sorted(report["reports"].keys()), ["bm25-only", "dense-only", "fused"])
        self.assertIsNone(report["metadata"]["reranker"])
        self.assertIn("exact_name", report["comparison"]["fused"])

    def test_hybrid_comparison_report_is_reproducible(self):
        def run_report_json():
            with tempfile.TemporaryDirectory() as temp_dir:
                eval_main(
                    [
                        "--golden-set",
                        "eval/golden_set.jsonl",
                        "--knowledge-path",
                        "eval/fixtures/knowledge_base",
                        "--report-dir",
                        temp_dir,
                        "--compare-hybrid",
                        "--quiet",
                    ]
                )
                json_path = next(Path(temp_dir).glob("*.json"))
                return json_path.read_text(encoding="utf-8")

        self.assertEqual(run_report_json(), run_report_json())

    def test_multi_query_report_is_reproducible(self):
        def run_report_json():
            with tempfile.TemporaryDirectory() as temp_dir:
                eval_main(
                    [
                        "--golden-set",
                        "eval/golden_set.jsonl",
                        "--knowledge-path",
                        "eval/fixtures/knowledge_base",
                        "--report-dir",
                        temp_dir,
                        "--multi-query",
                        "--quiet",
                    ]
                )
                json_path = next(Path(temp_dir).glob("*.json"))
                return json_path.read_text(encoding="utf-8")

        self.assertEqual(run_report_json(), run_report_json())

    def test_multi_query_capability_guardrails(self):
        examples = load_golden_set("eval/golden_set.jsonl")
        single_retrieve, single_metadata = build_hash_retriever(
            "eval/fixtures/knowledge_base",
            chunk_size=500,
            chunk_overlap=80,
            embedding_dimension=64,
        )
        multi_retrieve, multi_metadata = build_hash_retriever(
            "eval/fixtures/knowledge_base",
            chunk_size=500,
            chunk_overlap=80,
            embedding_dimension=64,
            query_rewrite_enabled=True,
            query_rewrite_provider="deterministic",
            query_rewrite_fixture_path="eval/fixtures/query_rewrites.jsonl",
            query_rewrite_num_queries=3,
            query_rewrite_weight_original=1.0,
            query_rewrite_weight_variant=0.7,
        )

        single_report = evaluate_retriever(
            single_retrieve, examples, k_values=[3], metadata=single_metadata
        )
        multi_report = evaluate_retriever(
            multi_retrieve, examples, k_values=[3], metadata=multi_metadata
        )

        single_capability = single_report["by_capability"]
        multi_capability = multi_report["by_capability"]
        self.assertGreaterEqual(
            multi_capability["long_tail"]["3"]["recall"],
            single_capability["long_tail"]["3"]["recall"],
        )
        self.assertGreaterEqual(
            multi_capability["paraphrase"]["3"]["recall"],
            single_capability["paraphrase"]["3"]["recall"],
        )
        self.assertGreaterEqual(
            multi_capability["exact_name"]["3"]["recall"],
            single_capability["exact_name"]["3"]["recall"],
        )
        self.assertLessEqual(
            multi_capability["negative"]["3"]["negative_false_recall_rate"],
            single_capability["negative"]["3"]["negative_false_recall_rate"],
        )

    def test_eval_run_report_contains_relevant_list_schema(self):
        examples = load_golden_set("eval/golden_set.jsonl")
        report = evaluate_retriever(
            lambda question, top_k: [
                make_source("eval/fixtures/knowledge_base/employee-handbook.md")
            ],
            examples[:1],
            k_values=[1],
        )

        serialized = json.dumps(report, ensure_ascii=False)

        self.assertIn('"relevant"', serialized)
        self.assertIn("eval/fixtures/knowledge_base/employee-handbook.md", serialized)

    def test_report_keeps_low_recall_cases_per_k(self):
        examples = [
            GoldenExample(
                qid="q1",
                question="query",
                relevant=[RelevantItem(source="a.md")],
                capability="exact_name",
                note="manual",
            )
        ]

        report = evaluate_retriever(
            lambda question, top_k: [make_source("noise.md"), make_source("a.md")],
            examples,
            k_values=[1, 2],
        )

        self.assertEqual(len(report["low_recall_cases_by_k"]["1"]), 1)
        self.assertEqual(report["low_recall_cases_by_k"]["2"], [])


if __name__ == "__main__":
    unittest.main()
