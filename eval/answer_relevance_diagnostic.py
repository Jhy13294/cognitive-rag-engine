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
TARGET_QIDS: Tuple[str, ...] = ("q010", "q015")
CONTROL_QIDS: Tuple[str, ...] = ("q011", "q014")
NONCOMMITTAL_PROBE_QIDS: Tuple[str, ...] = ("q005",)
DIAGNOSTIC_SCHEMA_VERSION = "answer-relevance-diagnostic-v1"
CALIBRATION_VERSION = "answer-relevance-calibration-v1"


SEMANTIC_BASELINES: Dict[str, Dict[str, List[str]]] = {
    "q010": {
        "equivalent": [
            "Which internal tool lets account teams review renewal risk?",
            "What tool do account teams use to monitor renewal risk?",
        ],
        "off_topic": [
            "Which policy defines distance-based mileage reimbursement for approved travel?",
        ],
    },
    "q015": {
        "equivalent": [
            "What service sends IoT device telemetry to regional ingestion clusters?",
            "Which service forwards connected device telemetry into regional ingestion clusters?",
        ],
        "off_topic": [
            "Who must approve an invoice above five thousand dollars?",
        ],
    },
    "q011": {
        "equivalent": [
            "What must be done before customer personal data is sent to an external vendor?",
        ],
        "off_topic": [
            "Which service routes IoT device telemetry into ingestion clusters?",
        ],
    },
    "q014": {
        "equivalent": [
            "May travelers get distance-based mileage reimbursement without fuel receipts?",
        ],
        "off_topic": [
            "Where is the workaround for a delayed building badge?",
        ],
    },
}


@dataclass(frozen=True)
class ReverseQuestionTrace:
    """One reverse-question emitted by the AnswerRelevancy prompt."""

    question: str
    noncommittal: bool
    cosine_to_original: float


@dataclass(frozen=True)
class AnswerRelevanceRunTrace:
    """One repeated AnswerRelevancy run with all intermediate evidence."""

    run_index: int
    all_noncommittal: bool
    value: float
    reverse_questions: List[ReverseQuestionTrace]


@dataclass(frozen=True)
class SemanticBaselineTrace:
    """One direct cosine check against a manually labeled query variant."""

    kind: str
    text: str
    cosine_to_original: float


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two finite non-zero vectors."""
    if len(left) != len(right):
        raise ValueError("Cosine inputs must have the same dimension.")
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    left_norm = math.sqrt(math.fsum(value * value for value in left_values))
    right_norm = math.sqrt(math.fsum(value * value for value in right_values))
    if left_norm <= 0 or right_norm <= 0:
        raise ValueError("Cosine inputs must have non-zero norms.")
    return math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_values, right_values)
    ) / (left_norm * right_norm)


async def trace_answer_relevance_once(
    metric,
    user_input: str,
    response: str,
    run_index: int,
) -> AnswerRelevanceRunTrace:
    """Run Ragas AnswerRelevancy once and expose reverse-question internals."""
    from ragas.metrics.collections.answer_relevancy.util import (
        AnswerRelevanceInput,
        AnswerRelevanceOutput,
    )

    generated_questions = []
    noncommittal_flags = []
    for _ in range(metric.strictness):
        prompt_string = metric.prompt.to_string(AnswerRelevanceInput(response=response))
        result = await metric.llm.agenerate(prompt_string, AnswerRelevanceOutput)
        if result.question:
            generated_questions.append(str(result.question))
            noncommittal_flags.append(bool(result.noncommittal))

    if not generated_questions:
        return AnswerRelevanceRunTrace(
            run_index=run_index,
            all_noncommittal=True,
            value=0.0,
            reverse_questions=[],
        )

    question_vector = await metric.embeddings.aembed_text(user_input)
    generated_vectors = await metric.embeddings.aembed_texts(generated_questions)
    traces = [
        ReverseQuestionTrace(
            question=question,
            noncommittal=noncommittal,
            cosine_to_original=cosine_similarity(vector, question_vector),
        )
        for question, noncommittal, vector in zip(
            generated_questions,
            noncommittal_flags,
            generated_vectors,
        )
    ]
    all_noncommittal = all(noncommittal_flags)
    mean_cosine = math.fsum(trace.cosine_to_original for trace in traces) / len(traces)
    value = mean_cosine * int(not all_noncommittal)
    return AnswerRelevanceRunTrace(
        run_index=run_index,
        all_noncommittal=all_noncommittal,
        value=clamp_live_score(value, f"answer_relevance_diagnostic.run_{run_index}"),
        reverse_questions=traces,
    )


async def trace_answer_relevance_repetitions(
    metric,
    user_input: str,
    response: str,
    repetitions: int,
) -> List[AnswerRelevanceRunTrace]:
    """Run repeated reverse-question generations with one batched embedding pass."""
    from ragas.metrics.collections.answer_relevancy.util import (
        AnswerRelevanceInput,
        AnswerRelevanceOutput,
    )

    generated_by_run = []
    for run_index in range(1, repetitions + 1):
        generated_questions = []
        noncommittal_flags = []
        for _ in range(metric.strictness):
            prompt_string = metric.prompt.to_string(AnswerRelevanceInput(response=response))
            result = await metric.llm.agenerate(prompt_string, AnswerRelevanceOutput)
            if result.question:
                generated_questions.append(str(result.question))
                noncommittal_flags.append(bool(result.noncommittal))
        generated_by_run.append((run_index, generated_questions, noncommittal_flags))

    all_questions = [
        question
        for _, generated_questions, _ in generated_by_run
        for question in generated_questions
    ]
    question_vector = await metric.embeddings.aembed_text(user_input)
    generated_vectors = (
        await metric.embeddings.aembed_texts(all_questions)
        if all_questions
        else []
    )

    traces = []
    vector_index = 0
    for run_index, generated_questions, noncommittal_flags in generated_by_run:
        if not generated_questions:
            traces.append(
                AnswerRelevanceRunTrace(
                    run_index=run_index,
                    all_noncommittal=True,
                    value=0.0,
                    reverse_questions=[],
                )
            )
            continue

        run_traces = []
        for question, noncommittal in zip(generated_questions, noncommittal_flags):
            vector = generated_vectors[vector_index]
            vector_index += 1
            run_traces.append(
                ReverseQuestionTrace(
                    question=question,
                    noncommittal=noncommittal,
                    cosine_to_original=cosine_similarity(vector, question_vector),
                )
            )

        all_noncommittal = all(noncommittal_flags)
        mean_cosine = math.fsum(
            trace.cosine_to_original for trace in run_traces
        ) / len(run_traces)
        value = mean_cosine * int(not all_noncommittal)
        traces.append(
            AnswerRelevanceRunTrace(
                run_index=run_index,
                all_noncommittal=all_noncommittal,
                value=clamp_live_score(
                    value,
                    f"answer_relevance_diagnostic.run_{run_index}",
                ),
                reverse_questions=run_traces,
            )
        )

    return traces


async def trace_semantic_baselines(
    embeddings,
    example: GoldenExample,
    variants: Dict[str, List[str]],
) -> List[SemanticBaselineTrace]:
    """Measure direct Gemini cosine for equivalent and off-topic query variants."""
    original_vector = await embeddings.aembed_text(example.question)
    items = [
        (kind, text)
        for kind, texts in variants.items()
        for text in texts
    ]
    if not items:
        return []
    vectors = await embeddings.aembed_texts([text for _, text in items])
    return [
        SemanticBaselineTrace(
            kind=kind,
            text=text,
            cosine_to_original=cosine_similarity(vector, original_vector),
        )
        for (kind, text), vector in zip(items, vectors)
    ]


def summarize_scores(values: Sequence[float]) -> Dict[str, float]:
    """Summarize one short diagnostic score vector without pulling gate code in."""
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


async def build_answer_relevance_diagnostic_report(
    *,
    golden_path: str,
    knowledge_path: str,
    profile: str,
    repetitions: int,
    include_noncommittal_probe: bool,
    qids_override: Optional[Sequence[str]] = None,
) -> Dict:
    """Collect a paid, content-bearing AnswerRelevancy diagnostic report."""
    if repetitions < 2:
        raise ValueError("Answer relevance diagnostics require at least two repetitions.")

    examples = load_golden_set(golden_path)
    if qids_override:
        qids = list(qids_override)
    else:
        qids = list(TARGET_QIDS + CONTROL_QIDS)
        if include_noncommittal_probe:
            qids.extend(NONCOMMITTAL_PROBE_QIDS)
    selected_examples = select_examples(examples, qids)
    pipeline = build_live_pipeline(profile, knowledge_path)
    samples = await collect_live_samples(pipeline, selected_examples, Config.RAGAS_TOP_K)
    sample_by_qid = {sample.qid: sample for sample in samples}
    example_by_qid = {example.qid: example for example in selected_examples}
    judge = LiveRagasJudge()
    metric = judge.metrics["answer_relevance"]

    cases = []
    for qid in qids:
        sample = sample_by_qid[qid]
        example = example_by_qid[qid]
        runs = await trace_answer_relevance_repetitions(
            metric,
            user_input=sample.question,
            response=sample.answer,
            repetitions=repetitions,
        )
        semantic_baselines = await trace_semantic_baselines(
            metric.embeddings,
            example,
            SEMANTIC_BASELINES.get(qid, {}),
        )
        cases.append(
            {
                "qid": qid,
                "capability": sample.capability,
                "role": classify_qid_role(qid),
                "question": sample.question,
                "answer": sample.answer,
                "ground_truth": sample.ground_truth,
                "answer_relevance": summarize_scores([run.value for run in runs]),
                "runs": [
                    {
                        **asdict(run),
                        "reverse_questions": [
                            asdict(trace) for trace in run.reverse_questions
                        ],
                    }
                    for run in runs
                ],
                "semantic_baselines": [
                    asdict(trace) for trace in semantic_baselines
                ],
            }
        )

    return {
        "report_type": "answer_relevance_diagnostic",
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "metadata": {
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
            "golden_path": golden_path,
            "knowledge_path": knowledge_path,
            "repetitions": repetitions,
            "target_qids": list(TARGET_QIDS),
            "control_qids": list(CONTROL_QIDS),
            "noncommittal_probe_qids": (
                list(NONCOMMITTAL_PROBE_QIDS)
                if include_noncommittal_probe
                else []
            ),
            "strictness": metric.strictness,
            "judge_model_id": Config.RAGAS_JUDGE_MODEL,
            "judge_temperature": Config.RAGAS_JUDGE_TEMPERATURE,
            "generator_model_id": Config.MODEL_NAME,
            "generator_temperature": Config.RAGAS_LIVE_GENERATION_TEMPERATURE,
            "ragas_version": installed_ragas_version(),
            "answer_relevance_prompt_version": "ragas-native",
            "answer_embedding_provider": Config.RAGAS_EMBEDDING_PROVIDER,
            "answer_embedding_model_id": Config.RAGAS_EMBEDDING_MODEL,
            "answer_embedding_dimension": Config.RAGAS_EMBEDDING_DIMENSION,
            "answer_embedding_l2_normalized": True,
            "answer_relevance_threshold": Config.RAGAS_ANSWER_RELEVANCE_THRESHOLD,
            "calibration_version": CALIBRATION_VERSION,
            "raw_evaluation_text_stored": True,
            "data_egress": (
                "questions and generated answers are sent to the configured judge; "
                "questions and reverse-questions are sent to the configured embedding endpoint"
            ),
        },
        "cases": cases,
    }


def classify_qid_role(qid: str) -> str:
    """Classify a qid's diagnostic purpose for report readers."""
    if qid in TARGET_QIDS:
        return "target_stable_low"
    if qid in CONTROL_QIDS:
        return "high_score_control"
    if qid in NONCOMMITTAL_PROBE_QIDS:
        return "noncommittal_probe"
    return "additional"


def write_diagnostic_report(report: Dict, report_dir: str, timestamp: Optional[str]) -> Path:
    """Write a diagnostic report with raw text intentionally preserved."""
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"answer-relevance-diagnostic-{report_timestamp}.json"
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the paid AnswerRelevancy diagnostic CLI."""
    parser = argparse.ArgumentParser(
        description="Dump Ragas AnswerRelevancy reverse-question and cosine internals."
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
        "--skip-noncommittal-probe",
        action="store_true",
        help="Skip the q005 noncommittal false-positive probe.",
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
            build_answer_relevance_diagnostic_report(
                golden_path=args.golden_set,
                knowledge_path=args.knowledge_path,
                profile=args.profile,
                repetitions=args.repetitions,
                include_noncommittal_probe=not args.skip_noncommittal_probe,
                qids_override=parse_qids(args.qids),
            )
        )
        output_path = write_diagnostic_report(report, args.report_dir, args.timestamp)
        print(f"Answer relevance diagnostic written: {output_path}")
        return 0
    except (FileNotFoundError, ValueError, RagasDependencyError) as error:
        print(f"Answer relevance diagnostic error: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
