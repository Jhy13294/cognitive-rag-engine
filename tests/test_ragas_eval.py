import asyncio
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from api_client import APIClient
from config import Config
from eval.answer_relevance_diagnostic import (
    classify_qid_role,
    cosine_similarity,
    parse_qids,
    trace_answer_relevance_once,
    trace_answer_relevance_repetitions,
    trace_semantic_baselines,
)
from eval.faithfulness_diagnostic import trace_faithfulness_once
from eval.faithfulness_prompt import (
    STATEMENT_PROMPT_VERSION,
    build_faithfulness_metric,
    build_statement_generator_prompt,
    preserve_clause_scoped_qualifier,
    preserve_question_dependent_rationale,
)
from eval.golden import load_golden_set
from eval.ragas_evaluation import (
    GATED_METRICS,
    REPORTED_METRICS,
    build_replay_report,
    file_sha256,
    load_verdict_fixture,
    summarize_values,
    validate_score,
    write_ragas_report,
    write_verdict_fixture,
)
from eval.ragas_live import (
    FAITHFULNESS_CONTEXT_FORMAT,
    LIVE_SCORE_CLAMP_TOLERANCE,
    LiveRagasJudge,
    LiveSample,
    answer_is_abstention,
    build_live_fixture_metadata,
    clamp_live_score,
    collect_live_samples,
    installed_ragas_version,
    normalize_embedding_l2,
    record_live_verdicts,
)
from eval.ragas_run import main as ragas_main
from eval.schemas import GoldenExample, RelevantItem
from rag.pipeline import RAG_SYSTEM_PROMPT_VERSION


METRICS = (
    "faithfulness",
    "answer_relevance",
    "context_precision",
    "context_recall",
)


class StaticJudge:
    """Return fixed scores without importing or calling a live judge."""

    def __init__(self, score=0.9):
        self.score_value = score
        self.calls = []

    async def score(self, sample):
        self.calls.append(sample.qid)
        return {metric: self.score_value for metric in METRICS}


class RagasEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.thresholds = {metric: 0.7 for metric in METRICS}
        self.margins = {metric: 0.1 for metric in METRICS}

    def test_golden_set_requires_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "golden.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "qid": "q1",
                        "question": "Question?",
                        "relevant": [{"source": "a.md"}],
                        "capability": "exact_name",
                        "note": "Manual reference.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "ground_truth"):
                load_golden_set(str(path))

    def test_replay_is_deterministic_and_excludes_negative_from_four_means(self):
        with self.fixture_workspace() as workspace:
            fixture = load_verdict_fixture(str(workspace.fixture_path))
            first = self.build_report(fixture, workspace)
            second = self.build_report(fixture, workspace)

            self.assertEqual(first, second)
            self.assertTrue(first["gate"]["passed"])
            self.assertEqual(first["metrics"]["faithfulness"]["n"], 3)
            self.assertEqual(first["negative"]["run_count"], 3)
            self.assertFalse(first["negative"]["included_in_positive_means"])

    def test_report_writer_is_byte_reproducible(self):
        with self.fixture_workspace() as workspace:
            report = self.build_report(
                load_verdict_fixture(str(workspace.fixture_path)),
                workspace,
            )
            first = write_ragas_report(report, str(workspace.root / "one"), timestamp="fixed")
            second = write_ragas_report(report, str(workspace.root / "two"), timestamp="fixed")

            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_bad_score_fails_and_names_case_metric_and_threshold(self):
        with self.fixture_workspace(positive_score=0.2) as workspace:
            fixture = load_verdict_fixture(str(workspace.fixture_path))
            report = self.build_report(fixture, workspace)
            failures = report["gate"]["failures"]

            self.assertFalse(report["gate"]["passed"])
            failure = next(
                item
                for item in failures
                if item.get("qid") == "q001" and item.get("metric") == "faithfulness"
            )
            self.assertAlmostEqual(failure["score"], 0.2)
            self.assertAlmostEqual(failure["threshold"], 0.7)

    def test_cli_returns_nonzero_and_prints_degraded_case(self):
        with self.fixture_workspace(positive_score=0.2) as workspace:
            stderr = io.StringIO()
            with self.configured_replay(), contextlib.redirect_stderr(stderr):
                exit_code = ragas_main(
                    [
                        "replay",
                        "--golden-set",
                        str(workspace.golden_path),
                        "--fixture",
                        str(workspace.fixture_path),
                        "--no-write-report",
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("qid=q001", stderr.getvalue())
            self.assertIn("metric=faithfulness", stderr.getvalue())
            self.assertIn("threshold=0.700000", stderr.getvalue())

    def test_clean_cli_replay_passes_without_api_keys(self):
        with self.fixture_workspace() as workspace:
            with self.configured_replay(), patch.object(Config, "RAGAS_JUDGE_API_KEY", None):
                exit_code = ragas_main(
                    [
                        "replay",
                        "--golden-set",
                        str(workspace.golden_path),
                        "--fixture",
                        str(workspace.fixture_path),
                        "--no-write-report",
                        "--quiet",
                    ]
                )

            self.assertEqual(exit_code, 0)

    def test_margin_must_cover_observed_judge_mad(self):
        with self.fixture_workspace(positive_runs=[0.5, 0.9, 0.7]) as workspace:
            fixture = load_verdict_fixture(str(workspace.fixture_path))
            report = build_replay_report(
                fixture=fixture,
                examples=workspace.examples,
                golden_path=str(workspace.golden_path),
                thresholds={metric: 0.4 for metric in METRICS},
                margins={metric: 0.01 for metric in METRICS},
                sigma_multiplier=1.0,
                negative_abstention_threshold=1.0,
                expected_judge_model="judge-model-pinned",
                installed_ragas_version=installed_ragas_version(),
                expected_generation_prompt_version=RAG_SYSTEM_PROMPT_VERSION,
                expected_statement_prompt_version=STATEMENT_PROMPT_VERSION,
                expected_faithfulness_context_format=FAITHFULNESS_CONTEXT_FORMAT,
            )

            self.assertTrue(
                any(failure["kind"] == "insufficient_margin" for failure in report["gate"]["failures"])
            )

    def test_median_gate_ignores_minority_zero_outliers(self):
        with self.fixture_workspace(positive_runs=[1.0, 1.0, 1.0, 0.0, 0.0]) as workspace:
            report = self.build_report(
                load_verdict_fixture(str(workspace.fixture_path)),
                workspace,
            )

        case_metric = report["cases"][0]["metrics"]["faithfulness"]
        self.assertAlmostEqual(case_metric["mean"], 0.6)
        self.assertAlmostEqual(case_metric["median"], 1.0)
        self.assertTrue(report["gate"]["passed"])

    def test_median_gate_rejects_majority_zero_verdicts(self):
        with self.fixture_workspace(positive_runs=[1.0, 0.0, 0.0, 0.0, 0.0]) as workspace:
            report = self.build_report(
                load_verdict_fixture(str(workspace.fixture_path)),
                workspace,
            )

        failure = next(
            item
            for item in report["gate"]["failures"]
            if item.get("qid") == "q001" and item.get("metric") == "faithfulness"
        )
        self.assertAlmostEqual(failure["score"], 0.0)
        self.assertAlmostEqual(failure["threshold"], 0.7)

    def test_mad_ignores_single_outlier_and_is_used_for_margin(self):
        summary = summarize_values([1.0, 1.0, 1.0, 1.0, 0.0])
        self.assertAlmostEqual(summary["mad"], 0.0)
        self.assertGreater(summary["stddev"], 0.0)

        with self.fixture_workspace(positive_runs=[1.0, 1.0, 1.0, 1.0, 0.0]) as workspace:
            report = self.build_report(
                load_verdict_fixture(str(workspace.fixture_path)),
                workspace,
            )

        metric = report["metrics"]["faithfulness"]
        self.assertAlmostEqual(metric["max_case_mad"], 0.0)
        self.assertGreater(metric["max_case_stddev"], 0.0)
        self.assertTrue(metric["margin_sufficient"])

    def test_report_metadata_records_robust_aggregation_provenance(self):
        with self.fixture_workspace() as workspace:
            report = self.build_report(
                load_verdict_fixture(str(workspace.fixture_path)),
                workspace,
            )

        metadata = report["metadata"]
        self.assertEqual(metadata["aggregation_method"], "median")
        self.assertEqual(metadata["spread_method"], "mad")
        self.assertTrue(metadata["aggregation_method_change_invalidates_baseline"])
        self.assertTrue(metadata["spread_method_change_invalidates_baseline"])
        self.assertEqual(metadata["gated_metrics"], list(GATED_METRICS))
        self.assertEqual(metadata["reported_only_metrics"], list(REPORTED_METRICS))
        self.assertTrue(metadata["gating_scope_change_invalidates_baseline"])

    def test_reported_context_metrics_do_not_fail_quality_threshold(self):
        with self.fixture_workspace() as workspace:
            fixture = load_verdict_fixture(str(workspace.fixture_path))
            for run in fixture["cases"][0]["runs"]:
                run["context_precision"] = 0.33
                run["context_recall"] = 0.25
            report = self.build_report(fixture, workspace)

        self.assertTrue(report["gate"]["passed"])
        self.assertAlmostEqual(report["metrics"]["context_precision"]["median"], 0.33)
        self.assertAlmostEqual(
            report["by_capability"]["exact_name"]["context_recall"]["median"],
            0.25,
        )
        self.assertFalse(
            any(
                failure.get("metric") in REPORTED_METRICS
                and failure.get("kind") == "quality_threshold"
                for failure in report["gate"]["failures"]
            )
        )

    def test_generation_sensitive_metrics_remain_gated(self):
        with self.fixture_workspace() as workspace:
            fixture = load_verdict_fixture(str(workspace.fixture_path))
            for run in fixture["cases"][0]["runs"]:
                run["faithfulness"] = 0.2
                run["answer_relevance"] = 0.3
            report = self.build_report(fixture, workspace)

        failures = {
            failure.get("metric")
            for failure in report["gate"]["failures"]
            if failure.get("kind") == "quality_threshold"
        }
        self.assertEqual(failures, set(GATED_METRICS))

    def test_reported_context_metric_mad_does_not_fail_margin_gate(self):
        with self.fixture_workspace(positive_runs=[0.9] * 5) as workspace:
            fixture = load_verdict_fixture(str(workspace.fixture_path))
            context_precision_runs = [0.0, 0.0, 0.5, 1.0, 1.0]
            for run, score in zip(
                fixture["cases"][0]["runs"],
                context_precision_runs,
            ):
                run["context_precision"] = score
            report = self.build_report(fixture, workspace)

        metric = report["metrics"]["context_precision"]
        self.assertGreater(metric["max_case_mad"], self.margins["context_precision"])
        self.assertFalse(metric["margin_sufficient"])
        self.assertFalse(
            any(
                failure.get("metric") == "context_precision"
                and failure.get("kind") == "insufficient_margin"
                for failure in report["gate"]["failures"]
            )
        )

    def test_judge_model_change_invalidates_fixture(self):
        with self.fixture_workspace() as workspace:
            report = build_replay_report(
                fixture=load_verdict_fixture(str(workspace.fixture_path)),
                examples=workspace.examples,
                golden_path=str(workspace.golden_path),
                thresholds=self.thresholds,
                margins=self.margins,
                sigma_multiplier=1.0,
                negative_abstention_threshold=1.0,
                expected_judge_model="different-model",
                installed_ragas_version=installed_ragas_version(),
                expected_generation_prompt_version=RAG_SYSTEM_PROMPT_VERSION,
                expected_statement_prompt_version=STATEMENT_PROMPT_VERSION,
                expected_faithfulness_context_format=FAITHFULNESS_CONTEXT_FORMAT,
            )

            self.assertTrue(
                any(
                    failure["kind"] == "baseline_invalidated"
                    and failure["field"] == "judge_model_id"
                    for failure in report["gate"]["failures"]
                )
            )

    def test_collect_live_samples_uses_existing_answer_path_once(self):
        examples = self.make_examples()
        pipeline = Mock()
        pipeline.answer.side_effect = [
            SimpleNamespace(
                answer="Supported answer [1].",
                sources=[
                    SimpleNamespace(
                        index=1,
                        content="Context one",
                        metadata={"source": r"C:\private\knowledge\policy.md"},
                    )
                ],
            ),
            SimpleNamespace(
                answer="The answer is not available in the knowledge base.",
                sources=[],
            ),
        ]

        samples = asyncio.run(collect_live_samples(pipeline, examples, top_k=5))

        self.assertEqual(pipeline.answer.call_count, 2)
        self.assertEqual(samples[0].contexts, ["Context one"])
        self.assertEqual(
            samples[0].faithfulness_contexts,
            ["[1] policy.md: Context one"],
        )
        self.assertNotIn("C:\\private", samples[0].faithfulness_contexts[0])
        self.assertEqual(samples[0].ground_truth, "Reference answer.")

    def test_faithfulness_context_labels_follow_generation_citation_order(self):
        example = self.make_examples()[0]
        pipeline = Mock()
        pipeline.answer.return_value = SimpleNamespace(
            answer="The definition is in handbook.md [2].",
            sources=[
                SimpleNamespace(
                    index=1,
                    content="First context.",
                    metadata={"source": "overview.md"},
                ),
                SimpleNamespace(
                    index=2,
                    content="Definition context.",
                    metadata={"source": "handbook.md"},
                ),
            ],
        )

        sample = asyncio.run(collect_live_samples(pipeline, [example], top_k=5))[0]

        self.assertEqual(
            sample.faithfulness_contexts,
            [
                "[1] overview.md: First context.",
                "[2] handbook.md: Definition context.",
            ],
        )

    def test_only_faithfulness_receives_source_labeled_contexts(self):
        metric_result = SimpleNamespace(value=1.0)
        judge = object.__new__(LiveRagasJudge)
        judge.metrics = {
            metric: SimpleNamespace(ascore=AsyncMock(return_value=metric_result))
            for metric in METRICS
        }
        sample = LiveSample(
            qid="q001",
            capability="exact_name",
            question="Question?",
            answer="Supported answer [1].",
            contexts=["Context."],
            faithfulness_contexts=["[1] policy.md: Context."],
            ground_truth="Reference answer.",
        )

        asyncio.run(judge.score(sample))

        judge.metrics["faithfulness"].ascore.assert_awaited_once_with(
            user_input=sample.question,
            response=sample.answer,
            retrieved_contexts=sample.faithfulness_contexts,
        )
        judge.metrics["context_precision"].ascore.assert_awaited_once_with(
            user_input=sample.question,
            reference=sample.ground_truth,
            retrieved_contexts=sample.contexts,
        )
        judge.metrics["context_recall"].ascore.assert_awaited_once_with(
            user_input=sample.question,
            reference=sample.ground_truth,
            retrieved_contexts=sample.contexts,
        )

    def test_live_score_clamp_absorbs_only_floating_point_epsilon(self):
        self.assertEqual(clamp_live_score(1.0000001, "upper"), 1.0)
        self.assertEqual(clamp_live_score(-0.0000001, "lower"), 0.0)
        self.assertEqual(
            LIVE_SCORE_CLAMP_TOLERANCE,
            1e-6,
        )

        for invalid in (1.5, -0.3, float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    clamp_live_score(invalid, "invalid")

        with self.assertRaises(ValueError):
            validate_score(1.0000001, "strict-replay")

    def test_live_judge_clamps_fake_answer_relevance_epsilon(self):
        judge = object.__new__(LiveRagasJudge)
        judge.metrics = {
            metric: SimpleNamespace(
                ascore=AsyncMock(
                    return_value=SimpleNamespace(
                        value=1.0000001 if metric == "answer_relevance" else 0.9,
                    )
                )
            )
            for metric in METRICS
        }
        sample = LiveSample(
            qid="q005",
            capability="exact_name",
            question="Question?",
            answer="Supported answer [1].",
            contexts=["Context."],
            faithfulness_contexts=["[1] policy.md: Context."],
            ground_truth="Reference answer.",
        )

        scores = asyncio.run(judge.score(sample))

        self.assertEqual(scores["answer_relevance"], 1.0)
        self.assertEqual(scores["faithfulness"], 0.9)

    def test_faithfulness_diagnostic_traces_statement_verdicts(self):
        class FakeFaithfulnessMetric:
            async def _create_statements(self, question, response):
                return ["The answer is supported.", "The answer adds an unsupported detail."]

            async def _create_verdicts(self, statements, context):
                return SimpleNamespace(
                    statements=[
                        SimpleNamespace(
                            statement=statements[0],
                            reason="The context states this directly.",
                            verdict=1,
                        ),
                        SimpleNamespace(
                            statement=statements[1],
                            reason="The context does not mention the extra detail.",
                            verdict=0,
                        ),
                    ]
                )

            def _compute_score(self, verdicts):
                return sum(int(item.verdict) for item in verdicts.statements) / len(verdicts.statements)

        trace = asyncio.run(
            trace_faithfulness_once(
                FakeFaithfulnessMetric(),
                user_input="Question?",
                response="Answer.",
                retrieved_contexts=["Context."],
                run_index=1,
            )
        )

        self.assertEqual(trace.value, 0.5)
        self.assertEqual(
            trace.statements,
            ["The answer is supported.", "The answer adds an unsupported detail."],
        )
        self.assertEqual(trace.verdicts[0].verdict, 1)
        self.assertEqual(trace.verdicts[1].verdict, 0)
        self.assertIn("extra detail", trace.verdicts[1].reason)

    def test_answer_relevance_diagnostic_traces_reverse_question_cosines(self):
        class FakePrompt:
            def to_string(self, input_data):
                return f"prompt: {input_data.response}"

        class FakeLLM:
            def __init__(self):
                self.outputs = [
                    SimpleNamespace(question="same question", noncommittal=0),
                    SimpleNamespace(question="orthogonal question", noncommittal=0),
                    SimpleNamespace(question="partial question", noncommittal=0),
                ]

            async def agenerate(self, *args, **kwargs):
                return self.outputs.pop(0)

        class FakeEmbeddings:
            async def aembed_text(self, text):
                self.seen_original = text
                return [1.0, 0.0]

            async def aembed_texts(self, texts):
                self.seen_generated = list(texts)
                return [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]]

        embeddings = FakeEmbeddings()
        metric = SimpleNamespace(
            strictness=3,
            prompt=FakePrompt(),
            llm=FakeLLM(),
            embeddings=embeddings,
        )

        trace = asyncio.run(
            trace_answer_relevance_once(
                metric,
                user_input="original question",
                response="diagnostic answer",
                run_index=1,
            )
        )

        self.assertEqual(embeddings.seen_original, "original question")
        self.assertEqual(
            embeddings.seen_generated,
            ["same question", "orthogonal question", "partial question"],
        )
        self.assertFalse(trace.all_noncommittal)
        self.assertAlmostEqual(trace.reverse_questions[0].cosine_to_original, 1.0)
        self.assertAlmostEqual(trace.reverse_questions[1].cosine_to_original, 0.0)
        self.assertAlmostEqual(trace.reverse_questions[2].cosine_to_original, 0.6)
        self.assertAlmostEqual(trace.value, 0.5333333333333333)

    def test_answer_relevance_diagnostic_all_noncommittal_zeroes_value(self):
        class FakePrompt:
            def to_string(self, input_data):
                return input_data.response

        class FakeLLM:
            async def agenerate(self, *args, **kwargs):
                return SimpleNamespace(question="generated question", noncommittal=1)

        class FakeEmbeddings:
            async def aembed_text(self, text):
                return [1.0, 0.0]

            async def aembed_texts(self, texts):
                return [[1.0, 0.0] for _ in texts]

        metric = SimpleNamespace(
            strictness=3,
            prompt=FakePrompt(),
            llm=FakeLLM(),
            embeddings=FakeEmbeddings(),
        )

        trace = asyncio.run(
            trace_answer_relevance_once(
                metric,
                user_input="original question",
                response="noncommittal answer",
                run_index=1,
            )
        )

        self.assertTrue(trace.all_noncommittal)
        self.assertEqual(trace.value, 0.0)

    def test_answer_relevance_repetition_trace_batches_embeddings(self):
        class FakePrompt:
            def to_string(self, input_data):
                return input_data.response

        class FakeLLM:
            def __init__(self):
                self.count = 0

            async def agenerate(self, *args, **kwargs):
                self.count += 1
                return SimpleNamespace(
                    question=f"generated {self.count}",
                    noncommittal=0,
                )

        class FakeEmbeddings:
            def __init__(self):
                self.single_calls = 0
                self.batch_calls = 0

            async def aembed_text(self, text):
                self.single_calls += 1
                return [1.0, 0.0]

            async def aembed_texts(self, texts):
                self.batch_calls += 1
                self.batch_texts = list(texts)
                return [[1.0, 0.0] for _ in texts]

        embeddings = FakeEmbeddings()
        metric = SimpleNamespace(
            strictness=3,
            prompt=FakePrompt(),
            llm=FakeLLM(),
            embeddings=embeddings,
        )

        traces = asyncio.run(
            trace_answer_relevance_repetitions(
                metric,
                user_input="original",
                response="answer",
                repetitions=2,
            )
        )

        self.assertEqual(len(traces), 2)
        self.assertEqual(embeddings.single_calls, 1)
        self.assertEqual(embeddings.batch_calls, 1)
        self.assertEqual(
            embeddings.batch_texts,
            [
                "generated 1",
                "generated 2",
                "generated 3",
                "generated 4",
                "generated 5",
                "generated 6",
            ],
        )
        self.assertEqual(parse_qids("q010, q015"), ["q010", "q015"])

    def test_answer_relevance_semantic_baselines_use_embedding_cosine(self):
        class FakeEmbeddings:
            async def aembed_text(self, text):
                return [1.0, 0.0]

            async def aembed_texts(self, texts):
                return [[1.0, 0.0], [0.0, 1.0]]

        example = GoldenExample(
            qid="q015",
            question="Which service routes telemetry?",
            relevant=[RelevantItem(source="product-glossary.md")],
            capability="long_tail",
            note="Manual reference.",
            ground_truth="Nimbus Gateway routes telemetry.",
        )

        traces = asyncio.run(
            trace_semantic_baselines(
                FakeEmbeddings(),
                example,
                {
                    "equivalent": ["What service routes telemetry?"],
                    "off_topic": ["Who approves invoices?"],
                },
            )
        )

        self.assertEqual([trace.kind for trace in traces], ["equivalent", "off_topic"])
        self.assertAlmostEqual(traces[0].cosine_to_original, 1.0)
        self.assertAlmostEqual(traces[1].cosine_to_original, 0.0)
        self.assertEqual(classify_qid_role("q010"), "target_stable_low")
        self.assertEqual(classify_qid_role("q014"), "high_score_control")
        self.assertAlmostEqual(cosine_similarity([3.0, 4.0], [3.0, 4.0]), 1.0)

    def test_embedding_adapter_normalizes_sync_and_async_vectors(self):
        with (
            patch.object(Config, "RAGAS_JUDGE_API_KEY", "judge-test-key"),
            patch.object(Config, "RAGAS_EMBEDDING_API_KEY", "embedding-test-key"),
            patch.object(Config, "RAGAS_JUDGE_BASE_URL", "https://judge.invalid/v1"),
            patch.object(Config, "RAGAS_EMBEDDING_BASE_URL", "https://embedding.invalid/v1beta"),
        ):
            judge = LiveRagasJudge()

        adapter = judge.metrics["answer_relevance"].embeddings
        provider = Mock()
        provider.embed_text.return_value = [3.0, 4.0]
        provider.embed_texts.return_value = [[3.0, 4.0], [0.0, 5.0]]
        provider.async_embed_text = AsyncMock(return_value=[3.0, 4.0])
        provider.async_embed_texts = AsyncMock(
            return_value=[[3.0, 4.0], [0.0, 5.0]],
        )
        adapter.provider = provider

        vectors = [
            adapter.embed_text("one"),
            *adapter.embed_texts(["one", "two"]),
            asyncio.run(adapter.aembed_text("one")),
            *asyncio.run(adapter.aembed_texts(["one", "two"])),
        ]

        for vector in vectors:
            norm = sum(value * value for value in vector) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=12)
        self.assertEqual(normalize_embedding_l2([3.0, 4.0]), [0.6, 0.8])

    def test_live_record_repeats_fixed_positive_input_and_separates_negative(self):
        samples = [
            LiveSample(
                qid="q001",
                capability="exact_name",
                question="Question?",
                answer="Supported answer.",
                contexts=["Context."],
                faithfulness_contexts=["[1] source.md: Context."],
                ground_truth="Reference answer.",
            ),
            LiveSample(
                qid="q002",
                capability="negative",
                question="Unknown?",
                answer="The answer is not available in the knowledge base.",
                contexts=[],
                faithfulness_contexts=[],
                ground_truth="The assistant should abstain.",
            ),
        ]
        judge = StaticJudge()

        cases = asyncio.run(record_live_verdicts(samples, judge, repetitions=3))

        self.assertEqual(judge.calls, ["q001", "q001", "q001"])
        self.assertEqual(len(cases[0]["runs"]), 3)
        self.assertIn("faithfulness_contexts_sha256", cases[0])
        self.assertNotIn("runs", cases[1])
        self.assertTrue(all(run["abstained"] for run in cases[1]["negative_runs"]))

    def test_abstention_classifier_is_conservative(self):
        self.assertTrue(answer_is_abstention("The answer is not available in the knowledge base."))
        self.assertFalse(
            answer_is_abstention(
                "The finance policy does not provide a payroll tax table for the current quarter."
            )
        )
        self.assertFalse(answer_is_abstention("The cafeteria serves noodles today."))
        self.assertFalse(
            answer_is_abstention(
                "The knowledge base mentions leave, so employees may take five pet vacation days."
            )
        )

    def test_live_ragas_components_construct_without_network_calls(self):
        with (
            patch.object(Config, "RAGAS_JUDGE_API_KEY", "judge-test-key"),
            patch.object(Config, "RAGAS_EMBEDDING_API_KEY", "embedding-test-key"),
            patch.object(Config, "RAGAS_JUDGE_BASE_URL", "https://judge.invalid/v1"),
            patch.object(Config, "RAGAS_EMBEDDING_BASE_URL", "https://embedding.invalid/v1beta"),
        ):
            judge = LiveRagasJudge()

        self.assertEqual(set(judge.metrics), set(METRICS))

    def test_joint_requirement_prompt_is_wired_into_faithfulness_metric(self):
        with (
            patch.object(Config, "RAGAS_JUDGE_API_KEY", "judge-test-key"),
            patch.object(Config, "RAGAS_EMBEDDING_API_KEY", "embedding-test-key"),
            patch.object(Config, "RAGAS_JUDGE_BASE_URL", "https://judge.invalid/v1"),
            patch.object(Config, "RAGAS_EMBEDDING_BASE_URL", "https://embedding.invalid/v1beta"),
        ):
            judge = LiveRagasJudge()

        prompt = judge.metrics["faithfulness"].statement_generator_prompt
        self.assertEqual(prompt.prompt_version, STATEMENT_PROMPT_VERSION)
        self.assertEqual(type(prompt).__name__, "SemanticDependencyStatementGeneratorPrompt")

    def test_statement_prompt_preserves_joint_requirements_and_indispensable_rationales(self):
        prompt = build_statement_generator_prompt()
        rendered = prompt.to_string(
            prompt.input_model(
                question="What is the workaround?",
                answer="Call the hotline and ask the manager.",
            )
        )

        self.assertEqual(len(prompt.examples), 4)
        self.assertIn("Albert Einstein", rendered)
        self.assertIn("joint requirement", rendered)
        self.assertIn("call the support hotline", rendered)
        self.assertIn("indispensable rationale", rendered)
        self.assertIn("Source [4] is relevant because", rendered)
        self.assertIn("Vendor records must be encrypted", rendered)
        self.assertIn("Do not move it to an earlier conjunct", rendered)
        conjunctive_output = prompt.examples[1][1]
        self.assertEqual(len(conjunctive_output.statements), 1)
        self.assertIn(" and ", conjunctive_output.statements[0])
        rationale_output = prompt.examples[2][1]
        self.assertEqual(len(rationale_output.statements), 1)
        self.assertIn("relevant because", rationale_output.statements[0])
        self.assertIn("notify security", rationale_output.statements[0])
        scope_output = prompt.examples[3][1]
        self.assertEqual(len(scope_output.statements), 1)
        self.assertIn("before the vendor release", scope_output.statements[0])

    def test_question_dependent_rationale_guard_is_narrow(self):
        answer = (
            "Source [4] is relevant because it directly addresses token recovery: "
            "the employee should notify security."
        )
        extracted = [
            "Source [4] is relevant because it directly addresses token recovery.",
            "The employee should notify security.",
        ]

        self.assertEqual(
            preserve_question_dependent_rationale(answer, extracted),
            [answer],
        )
        self.assertEqual(
            preserve_question_dependent_rationale(
                "Source [4] is relevant. The employee should notify security.",
                extracted,
            ),
            extracted,
        )

    def test_scope_guard_preserves_the_original_clause_attachment(self):
        answer = (
            "Customer PII must be redacted, and the export owner must record the "
            "approval ticket before any transfer begins. [1]"
        )
        extracted = [
            "Customer PII must be redacted before any transfer begins.",
            "The export owner must record the approval ticket before any transfer begins.",
        ]

        self.assertEqual(
            preserve_clause_scoped_qualifier(answer, extracted),
            [answer],
        )

    def test_scope_guard_keeps_independent_clauses_when_no_scope_moves(self):
        answer = (
            "Customer PII must be redacted, and the export owner must record the "
            "approval ticket before any transfer begins. [1]"
        )
        extracted = [
            "Customer PII must be redacted.",
            "The export owner must record the approval ticket before any transfer begins.",
        ]

        self.assertEqual(
            preserve_clause_scoped_qualifier(answer, extracted),
            extracted,
        )

    def test_rationale_guard_does_not_force_an_unfaithful_nli_verdict(self):
        from ragas.llms.base import InstructorBaseRagasLLM
        from ragas.metrics.collections.faithfulness.util import (
            NLIStatementOutput,
            StatementFaithfulnessAnswer,
            StatementGeneratorOutput,
        )

        answer = (
            "Source [4] is relevant because it states that employees receive "
            "ten pet vacation days."
        )

        class UnfaithfulClaimLLM(InstructorBaseRagasLLM):
            def generate(self, prompt, response_model):
                raise AssertionError("Only the async path should be used.")

            async def agenerate(self, prompt, output_model):
                if output_model is StatementGeneratorOutput:
                    return output_model(
                        statements=[
                            "Source [4] is relevant.",
                            "Employees receive ten pet vacation days.",
                        ]
                    )
                if output_model is NLIStatementOutput:
                    return output_model(
                        statements=[
                            StatementFaithfulnessAnswer(
                                statement=answer,
                                reason="The context contains no pet vacation policy.",
                                verdict=0,
                            )
                        ]
                    )
                raise AssertionError(f"Unexpected output model: {output_model}")

        metric = build_faithfulness_metric(UnfaithfulClaimLLM())
        result = asyncio.run(
            metric.ascore(
                user_input="Which source defines pet vacation?",
                response=answer,
                retrieved_contexts=["[4] handbook.md: No pet policy is defined."],
            )
        )

        self.assertEqual(result.value, 0.0)

    def test_statement_creation_sends_custom_prompt_to_fake_llm(self):
        from ragas.llms.base import InstructorBaseRagasLLM
        from ragas.metrics.collections import Faithfulness

        class PromptCapturingLLM(InstructorBaseRagasLLM):
            def __init__(self):
                self.prompt = None

            def generate(self, prompt, response_model):
                self.prompt = prompt
                return response_model(
                    statements=["Call the hotline and ask the manager."],
                )

            async def agenerate(self, prompt, output_model):
                self.prompt = prompt
                return output_model(
                    statements=["Call the hotline and ask the manager."],
                )

        fake_llm = PromptCapturingLLM()
        metric = Faithfulness(llm=fake_llm)
        metric.statement_generator_prompt = build_statement_generator_prompt()

        statements = asyncio.run(
            metric._create_statements(
                "What is the workaround?",
                "Call the hotline and ask the manager.",
            )
        )

        self.assertEqual(statements, ["Call the hotline and ask the manager."])
        self.assertIn("joint requirement", fake_llm.prompt)
        self.assertIn("call the support hotline", fake_llm.prompt)

    def test_api_client_temperature_override_preserves_product_default(self):
        capture_client = APIClient(
            "test-key",
            "https://chat.invalid/v1/chat/completions",
            temperature=0,
        )
        product_client = APIClient(
            "test-key",
            "https://chat.invalid/v1/chat/completions",
        )

        capture_payload = capture_client._build_payload("Question?", None, False)
        product_payload = product_client._build_payload("Question?", None, False)

        self.assertEqual(capture_payload["temperature"], 0)
        self.assertEqual(product_payload["temperature"], Config.TEMPERATURE)

    def test_live_fixture_metadata_distinguishes_judge_and_generator_temperatures(self):
        with (
            self.fixture_workspace() as workspace,
            patch.object(Config, "RAGAS_JUDGE_TEMPERATURE", 0.1),
            patch.object(Config, "RAGAS_LIVE_GENERATION_TEMPERATURE", 0.0),
        ):
            metadata = build_live_fixture_metadata(
                str(workspace.golden_path),
                profile="baseline",
                repetitions=3,
            )

        self.assertEqual(metadata["temperature"], 0.1)
        self.assertEqual(metadata["judge_temperature"], 0.1)
        self.assertEqual(metadata["generator_temperature"], 0.0)
        self.assertEqual(metadata["generation_prompt_version"], RAG_SYSTEM_PROMPT_VERSION)
        self.assertTrue(metadata["generation_prompt_change_invalidates_baseline"])
        self.assertEqual(metadata["statement_prompt_version"], STATEMENT_PROMPT_VERSION)
        self.assertTrue(metadata["statement_prompt_change_invalidates_baseline"])
        self.assertEqual(
            metadata["faithfulness_context_format"],
            FAITHFULNESS_CONTEXT_FORMAT,
        )
        self.assertTrue(
            metadata["faithfulness_context_format_change_invalidates_baseline"]
        )
        self.assertEqual(metadata["gated_metrics"], list(GATED_METRICS))
        self.assertEqual(metadata["reported_only_metrics"], list(REPORTED_METRICS))
        self.assertTrue(metadata["gating_scope_change_invalidates_baseline"])
        self.assertEqual(
            metadata["live_score_clamp_tolerance"],
            LIVE_SCORE_CLAMP_TOLERANCE,
        )

    def test_validate_ragas_rejects_negative_live_generation_temperature(self):
        with (
            patch.object(Config, "RUN_RAGAS_EVAL", False),
            patch.object(Config, "RAGAS_LIVE_GENERATION_TEMPERATURE", -0.1),
        ):
            with self.assertRaisesRegex(
                ValueError,
                "RAGAS_LIVE_GENERATION_TEMPERATURE must be non-negative",
            ):
                Config.validate_ragas()

    def test_validate_ragas_requires_live_secrets_only_for_live_mode(self):
        with (
            patch.object(Config, "RAGAS_ENABLED", True),
            patch.object(Config, "RUN_RAGAS_EVAL", False),
            patch.object(Config, "RAGAS_JUDGE_API_KEY", None),
            patch.object(Config, "RAGAS_EMBEDDING_API_KEY", None),
        ):
            self.assertTrue(Config.validate_ragas())

        with (
            patch.object(Config, "RAGAS_ENABLED", True),
            patch.object(Config, "RUN_RAGAS_EVAL", True),
            patch.object(Config, "RAGAS_JUDGE_API_KEY", None),
            patch.object(Config, "RAGAS_EMBEDDING_API_KEY", None),
        ):
            with self.assertRaisesRegex(ValueError, "RAGAS_JUDGE_API_KEY"):
                Config.validate_ragas()

    def test_fixture_contains_hashes_not_raw_evaluation_text(self):
        with self.fixture_workspace() as workspace:
            text = workspace.fixture_path.read_text(encoding="utf-8")

            self.assertNotIn("Reference answer", text)
            self.assertNotIn("Question?", text)
            self.assertIn("answer_sha256", text)
            self.assertIn("contexts_sha256", text)

    def test_replay_rejects_synthetic_or_bootstrap_recordings(self):
        with self.fixture_workspace() as workspace:
            records = [
                json.loads(line)
                for line in workspace.fixture_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["recording_mode"] = "synthetic_bootstrap"
            workspace.fixture_path.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "only live_gated"):
                load_verdict_fixture(str(workspace.fixture_path))

    def test_previous_fixture_schema_is_invalidated_by_protocol_change(self):
        with self.fixture_workspace() as workspace:
            records = [
                json.loads(line)
                for line in workspace.fixture_path.read_text(encoding="utf-8").splitlines()
            ]
            records[0]["schema_version"] = "ragas-verdicts-v3"
            workspace.fixture_path.write_text(
                "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Unsupported Ragas fixture schema"):
                load_verdict_fixture(str(workspace.fixture_path))

    def test_verdict_input_protocol_drift_invalidates_replay_baseline(self):
        with self.fixture_workspace() as workspace:
            fixture = load_verdict_fixture(str(workspace.fixture_path))
            generation_report = self.build_report(
                fixture,
                workspace,
                expected_generation_prompt_version="different-generation-prompt",
            )
            prompt_report = self.build_report(
                fixture,
                workspace,
                expected_statement_prompt_version="different-statement-prompt",
            )
            context_report = self.build_report(
                fixture,
                workspace,
                expected_faithfulness_context_format="different-context-format",
            )

        generation_failure = next(
            failure
            for failure in generation_report["gate"]["failures"]
            if failure.get("field") == "generation_prompt_version"
        )
        prompt_failure = next(
            failure
            for failure in prompt_report["gate"]["failures"]
            if failure.get("field") == "statement_prompt_version"
        )
        context_failure = next(
            failure
            for failure in context_report["gate"]["failures"]
            if failure.get("field") == "faithfulness_context_format"
        )
        self.assertEqual(generation_failure["kind"], "baseline_invalidated")
        self.assertEqual(prompt_failure["kind"], "baseline_invalidated")
        self.assertEqual(context_failure["kind"], "baseline_invalidated")

    def test_gating_scope_drift_invalidates_replay_baseline(self):
        with self.fixture_workspace() as workspace:
            fixture = load_verdict_fixture(str(workspace.fixture_path))
            fixture["metadata"]["gated_metrics"] = [
                "faithfulness",
                "answer_relevance",
                "context_precision",
            ]
            report = self.build_report(fixture, workspace)

        failure = next(
            failure
            for failure in report["gate"]["failures"]
            if failure.get("field") == "gated_metrics"
        )
        self.assertEqual(failure["kind"], "baseline_invalidated")
        self.assertEqual(failure["expected"], list(GATED_METRICS))

    def test_live_cli_requires_second_explicit_gate_before_any_network_path(self):
        with (
            patch.object(Config, "RAGAS_ENABLED", True),
            patch.object(Config, "RUN_RAGAS_EVAL", False),
            patch("eval.ragas_run.build_live_pipeline") as build_pipeline,
        ):
            exit_code = ragas_main(["live", "--no-write-report", "--quiet"])

        self.assertEqual(exit_code, 2)
        build_pipeline.assert_not_called()

    def build_report(
        self,
        fixture,
        workspace,
        expected_generation_prompt_version=RAG_SYSTEM_PROMPT_VERSION,
        expected_statement_prompt_version=STATEMENT_PROMPT_VERSION,
        expected_faithfulness_context_format=FAITHFULNESS_CONTEXT_FORMAT,
    ):
        return build_replay_report(
            fixture=fixture,
            examples=workspace.examples,
            golden_path=str(workspace.golden_path),
            thresholds=self.thresholds,
            margins=self.margins,
            sigma_multiplier=1.0,
            negative_abstention_threshold=1.0,
            expected_judge_model="judge-model-pinned",
            installed_ragas_version=installed_ragas_version(),
            expected_generation_prompt_version=expected_generation_prompt_version,
            expected_statement_prompt_version=expected_statement_prompt_version,
            expected_faithfulness_context_format=expected_faithfulness_context_format,
        )

    @contextlib.contextmanager
    def configured_replay(self):
        with (
            patch.object(Config, "RAGAS_ENABLED", True),
            patch.object(Config, "RUN_RAGAS_EVAL", False),
            patch.object(Config, "RAGAS_JUDGE_MODEL", "judge-model-pinned"),
            patch.object(Config, "RAGAS_FAITHFULNESS_THRESHOLD", 0.7),
            patch.object(Config, "RAGAS_ANSWER_RELEVANCE_THRESHOLD", 0.7),
            patch.object(Config, "RAGAS_CONTEXT_PRECISION_THRESHOLD", 0.7),
            patch.object(Config, "RAGAS_CONTEXT_RECALL_THRESHOLD", 0.7),
            patch.object(Config, "RAGAS_NEGATIVE_ABSTENTION_THRESHOLD", 1.0),
            patch.object(Config, "RAGAS_FAITHFULNESS_MARGIN", 0.1),
            patch.object(Config, "RAGAS_ANSWER_RELEVANCE_MARGIN", 0.1),
            patch.object(Config, "RAGAS_CONTEXT_PRECISION_MARGIN", 0.1),
            patch.object(Config, "RAGAS_CONTEXT_RECALL_MARGIN", 0.1),
            patch.object(Config, "RAGAS_SIGMA_MULTIPLIER", 1.0),
        ):
            yield

    @contextlib.contextmanager
    def fixture_workspace(self, positive_score=0.9, positive_runs=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            golden_path = root / "golden.jsonl"
            examples = self.make_examples()
            golden_path.write_text(
                "".join(
                    json.dumps(example.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
                    for example in examples
                ),
                encoding="utf-8",
            )
            run_values = positive_runs or [positive_score, positive_score, positive_score]
            cases = [
                {
                    "record_type": "verdict",
                    "qid": "q001",
                    "capability": "exact_name",
                    "answer_sha256": "a" * 64,
                    "contexts_sha256": "b" * 64,
                    "runs": [
                        {metric: value for metric in METRICS}
                        for value in run_values
                    ],
                },
                {
                    "record_type": "verdict",
                    "qid": "q002",
                    "capability": "negative",
                    "answer_sha256": "c" * 64,
                    "contexts_sha256": "d" * 64,
                    "negative_runs": [
                        {"abstained": True, "fabricated": False}
                        for _ in run_values
                    ],
                },
            ]
            metadata = {
                "judge_model_id": "judge-model-pinned",
                "ragas_version": installed_ragas_version(),
                "temperature": 0.0,
                "repetitions": len(run_values),
                "pipeline_profile": "baseline",
                "golden_version": file_sha256(str(golden_path)),
                "recorded_at": "2026-06-21T00:00:00+00:00",
                "recording_mode": "live_gated",
                "generation_prompt_version": RAG_SYSTEM_PROMPT_VERSION,
                "statement_prompt_version": STATEMENT_PROMPT_VERSION,
                "faithfulness_context_format": FAITHFULNESS_CONTEXT_FORMAT,
                "gated_metrics": list(GATED_METRICS),
                "reported_only_metrics": list(REPORTED_METRICS),
            }
            fixture_path = root / "fixture.jsonl"
            write_verdict_fixture(metadata, cases, str(fixture_path))
            yield SimpleNamespace(
                root=root,
                golden_path=golden_path,
                fixture_path=fixture_path,
                examples=examples,
            )

    @staticmethod
    def make_examples():
        return [
            GoldenExample(
                qid="q001",
                question="Question?",
                relevant=[RelevantItem(source="a.md")],
                capability="exact_name",
                note="Manual reference.",
                ground_truth="Reference answer.",
            ),
            GoldenExample(
                qid="q002",
                question="Unknown?",
                relevant=[],
                capability="negative",
                note="Manual negative reference.",
                ground_truth="The assistant should abstain.",
            ),
        ]


if __name__ == "__main__":
    unittest.main()
