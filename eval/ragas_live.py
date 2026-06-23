import asyncio
import importlib.metadata
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PureWindowsPath
from typing import Dict, List, Sequence

from api_client import APIClient
from config import Config
from rag.pipeline import CANONICAL_ABSTENTION_RESPONSE
from rag_cli import build_rag_pipeline_from_path

from .faithfulness_prompt import (
    STATEMENT_PROMPT_VERSION,
    build_statement_generator_prompt,
)
from .gemini_embedding import GeminiEmbeddingProvider
from .ragas_evaluation import (
    GATED_METRICS,
    RAGAS_METRICS,
    REPORTED_METRICS,
    content_sha256,
    file_sha256,
)
from .schemas import GoldenExample


LIVE_PROFILES = (
    "baseline",
    "rerank",
    "hybrid",
    "parent_child",
    "context_packing",
    "multi_query",
)
ABSTENTION_MARKERS = (
    CANONICAL_ABSTENTION_RESPONSE.lower().rstrip("."),
    "not provided in the knowledge base",
    "knowledge base does not contain",
    "knowledge base does not provide",
    "insufficient context",
    "cannot answer from the provided context",
)
FAITHFULNESS_CONTEXT_FORMAT = "citation-index-filename-content-v1"
LIVE_SCORE_CLAMP_TOLERANCE = 1e-6


class RagasDependencyError(RuntimeError):
    """Raised when the optional live Ragas dependency stack is incompatible."""


@dataclass(frozen=True)
class LiveSample:
    """One fixed answer/context input evaluated repeatedly by the live judge."""

    qid: str
    capability: str
    question: str
    answer: str
    contexts: List[str]
    faithfulness_contexts: List[str]
    ground_truth: str


class LiveRagasJudge:
    """Ragas four-metric judge backed by gated OpenAI-compatible endpoints."""

    def __init__(self):
        """Create Ragas metrics lazily so offline replay never imports Ragas."""
        os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings.base import BaseRagasEmbedding
            from ragas.llms import llm_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall,
                Faithfulness,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise RagasDependencyError(
                "Live Ragas imports failed. Install requirements.txt with the pinned "
                "LangChain <1.0 compatibility range required by ragas==0.4.3. "
                f"Original error: {error}"
            ) from error

        judge_client = AsyncOpenAI(
            api_key=Config.RAGAS_JUDGE_API_KEY,
            base_url=Config.RAGAS_JUDGE_BASE_URL,
            timeout=Config.RAGAS_JUDGE_TIMEOUT,
            max_retries=Config.RAGAS_JUDGE_MAX_RETRIES,
        )
        judge_llm = llm_factory(
            Config.RAGAS_JUDGE_MODEL,
            provider="openai",
            client=judge_client,
            temperature=Config.RAGAS_JUDGE_TEMPERATURE,
            max_tokens=Config.RAGAS_JUDGE_MAX_TOKENS,
        )
        embedding_provider = GeminiEmbeddingProvider(
            api_key=Config.RAGAS_EMBEDDING_API_KEY,
            base_url=Config.RAGAS_EMBEDDING_BASE_URL,
            model_name=Config.RAGAS_EMBEDDING_MODEL,
            dimension=Config.RAGAS_EMBEDDING_DIMENSION,
            batch_size=Config.RAGAS_EMBEDDING_BATCH_SIZE,
            timeout=Config.RAGAS_EMBEDDING_TIMEOUT,
            max_retries=Config.RAGAS_EMBEDDING_MAX_RETRIES,
            base_delay=Config.RAGAS_EMBEDDING_BASE_DELAY,
            max_delay=Config.RAGAS_EMBEDDING_MAX_DELAY,
        )

        class ProjectEmbeddingAdapter(BaseRagasEmbedding):
            """Expose the project embedding provider through the Ragas interface."""

            def __init__(self, provider):
                super().__init__()
                self.provider = provider

            def embed_text(self, text: str, **kwargs) -> List[float]:
                return normalize_embedding_l2(self.provider.embed_text(text))

            async def aembed_text(self, text: str, **kwargs) -> List[float]:
                vector = await self.provider.async_embed_text(text)
                return normalize_embedding_l2(vector)

            def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
                return [
                    normalize_embedding_l2(vector)
                    for vector in self.provider.embed_texts(texts)
                ]

            async def aembed_texts(
                self,
                texts: List[str],
                **kwargs,
            ) -> List[List[float]]:
                vectors = await self.provider.async_embed_texts(texts)
                return [normalize_embedding_l2(vector) for vector in vectors]

        embeddings = ProjectEmbeddingAdapter(embedding_provider)
        faithfulness = Faithfulness(llm=judge_llm)
        faithfulness.statement_generator_prompt = build_statement_generator_prompt()
        self.metrics = {
            "faithfulness": faithfulness,
            "answer_relevance": AnswerRelevancy(
                llm=judge_llm,
                embeddings=embeddings,
            ),
            "context_precision": ContextPrecision(llm=judge_llm),
            "context_recall": ContextRecall(llm=judge_llm),
        }

    async def score(self, sample: LiveSample) -> Dict[str, float]:
        """Evaluate all four dimensions for one fixed positive sample."""
        faithfulness = await self.metrics["faithfulness"].ascore(
            user_input=sample.question,
            response=sample.answer,
            retrieved_contexts=sample.faithfulness_contexts,
        )
        answer_relevance = await self.metrics["answer_relevance"].ascore(
            user_input=sample.question,
            response=sample.answer,
        )
        context_precision = await self.metrics["context_precision"].ascore(
            user_input=sample.question,
            reference=sample.ground_truth,
            retrieved_contexts=sample.contexts,
        )
        context_recall = await self.metrics["context_recall"].ascore(
            user_input=sample.question,
            reference=sample.ground_truth,
            retrieved_contexts=sample.contexts,
        )
        raw_scores = {
            "faithfulness": faithfulness.value,
            "answer_relevance": answer_relevance.value,
            "context_precision": context_precision.value,
            "context_recall": context_recall.value,
        }
        scores = {}
        for metric, raw_score in raw_scores.items():
            scores[metric] = clamp_live_score(
                raw_score,
                f"{sample.qid}.{metric}",
            )
        return scores


def clamp_live_score(
    raw_score,
    label: str,
    tolerance: float = LIVE_SCORE_CLAMP_TOLERANCE,
) -> float:
    """Clamp finite live-score epsilon while rejecting substantive invalid values."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("Live score clamp tolerance must be finite and non-negative.")
    try:
        score = float(raw_score)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Ragas live score {label} must be numeric.") from error
    if not math.isfinite(score):
        raise ValueError(f"Ragas live score {label} must be finite.")
    if score < -tolerance or score > 1.0 + tolerance:
        raise ValueError(
            f"Ragas live score {label} must be between 0 and 1 within "
            f"tolerance {tolerance}."
        )
    return min(1.0, max(0.0, score))


def normalize_embedding_l2(vector: Sequence[float]) -> List[float]:
    """Return a finite unit-length embedding for cosine-based live metrics."""
    numeric = [float(value) for value in vector]
    norm = math.sqrt(math.fsum(value * value for value in numeric))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Ragas embedding must have a finite, non-zero L2 norm.")
    return [value / norm for value in numeric]


def installed_ragas_version() -> str:
    """Return the installed Ragas distribution version without importing it."""
    try:
        return importlib.metadata.version("ragas")
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def build_live_pipeline(profile: str, knowledge_path: str):
    """Build one existing RAGPipeline profile without introducing a second path."""
    if profile not in LIVE_PROFILES:
        raise ValueError(f"Unsupported Ragas live profile: {profile}")
    if Config.CACHE_ENABLED:
        raise ValueError("Live Ragas capture requires CACHE_ENABLED=false so answers are freshly generated.")
    if Config.ACL_ENABLED:
        raise ValueError("Live fixture evaluation requires ACL_ENABLED=false for the public test corpus.")

    profile_options = {
        "baseline": {},
        "rerank": {
            "rerank_provider_name": "deterministic",
            "rerank_fetch_k": 30,
        },
        "hybrid": {
            "hybrid_enabled": True,
            "hybrid_fetch_k": 30,
        },
        "parent_child": {
            "parent_child_enabled": True,
        },
        "context_packing": {
            "parent_child_enabled": True,
            "context_packing_enabled": True,
            "context_dedup_enabled": True,
        },
        "multi_query": {
            "parent_child_enabled": True,
            "context_packing_enabled": True,
            "context_dedup_enabled": True,
            "query_rewrite_enabled": True,
            "query_rewrite_provider_name": "deterministic",
        },
    }
    chat_client = APIClient(
        Config.API_KEY,
        Config.API_URL,
        temperature=Config.RAGAS_LIVE_GENERATION_TEMPERATURE,
    )
    return build_rag_pipeline_from_path(
        knowledge_path,
        chat_client=chat_client,
        chunk_size=500,
        chunk_overlap=80,
        embedding_provider_name="hash",
        embedding_dimension=64,
        vector_store_name="memory",
        top_k=Config.RAGAS_TOP_K,
        **profile_options[profile],
    )


async def collect_live_samples(
    pipeline,
    examples: Sequence[GoldenExample],
    top_k: int,
) -> List[LiveSample]:
    """Call the existing pipeline answer path once for every golden example."""
    samples = []
    for example in examples:
        response = await asyncio.to_thread(
            pipeline.answer,
            example.question,
            top_k,
        )
        content_contexts = [source.content for source in response.sources]
        faithfulness_contexts = [
            format_faithfulness_context(source)
            for source in response.sources
        ]
        samples.append(
            LiveSample(
                qid=example.qid,
                capability=example.capability,
                question=example.question,
                answer=response.answer,
                contexts=content_contexts,
                faithfulness_contexts=faithfulness_contexts,
                ground_truth=example.ground_truth,
            )
        )
    return samples


def format_faithfulness_context(source) -> str:
    """Label one context with the same citation index used during generation."""
    metadata = getattr(source, "metadata", {}) or {}
    raw_source = metadata.get("source")
    if raw_source:
        source_label = PureWindowsPath(str(raw_source)).name
    else:
        source_label = str(
            metadata.get("id")
            or metadata.get("child_id")
            or metadata.get("parent_id")
            or f"source-{source.index}"
        )
    return f"[{source.index}] {source_label}: {source.content}"


async def record_live_verdicts(
    samples: Sequence[LiveSample],
    judge,
    repetitions: int,
) -> List[Dict]:
    """Repeat the judge on fixed inputs and return content-free fixture records."""
    if repetitions < 2:
        raise ValueError("Live Ragas capture requires at least two repetitions.")

    cases = []
    for sample in samples:
        base_record = {
            "record_type": "verdict",
            "qid": sample.qid,
            "capability": sample.capability,
            "answer_sha256": content_sha256([sample.answer]),
            "contexts_sha256": content_sha256(sample.contexts),
            "faithfulness_contexts_sha256": content_sha256(
                sample.faithfulness_contexts
            ),
        }
        if sample.capability == "negative":
            abstained = answer_is_abstention(sample.answer)
            base_record["negative_runs"] = [
                {"abstained": abstained, "fabricated": not abstained}
                for _ in range(repetitions)
            ]
        else:
            base_record["runs"] = [
                await judge.score(sample)
                for _ in range(repetitions)
            ]
        cases.append(base_record)
    return cases


def answer_is_abstention(answer: str) -> bool:
    """Classify the pipeline's contractually expected abstention phrasing."""
    normalized = " ".join(str(answer).lower().split())
    return any(marker in normalized for marker in ABSTENTION_MARKERS)


def build_live_fixture_metadata(
    golden_path: str,
    profile: str,
    repetitions: int,
) -> Dict:
    """Build provenance metadata for a live verdict recording."""
    return {
        "judge_model_id": Config.RAGAS_JUDGE_MODEL,
        "ragas_version": installed_ragas_version(),
        # Retained for fixture-schema and comparison compatibility; this is the judge temperature.
        "temperature": Config.RAGAS_JUDGE_TEMPERATURE,
        "judge_temperature": Config.RAGAS_JUDGE_TEMPERATURE,
        "generator_temperature": Config.RAGAS_LIVE_GENERATION_TEMPERATURE,
        "repetitions": repetitions,
        "pipeline_profile": profile,
        "golden_version": file_sha256(golden_path),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "recording_mode": "live_gated",
        "generator_model_id": Config.MODEL_NAME,
        "statement_prompt_version": STATEMENT_PROMPT_VERSION,
        "statement_prompt_change_invalidates_baseline": True,
        "faithfulness_context_format": FAITHFULNESS_CONTEXT_FORMAT,
        "faithfulness_context_format_change_invalidates_baseline": True,
        "gated_metrics": list(GATED_METRICS),
        "reported_only_metrics": list(REPORTED_METRICS),
        "gating_scope_change_invalidates_baseline": True,
        "live_score_clamp_tolerance": LIVE_SCORE_CLAMP_TOLERANCE,
        "judge_provider": "deepseek_openai_compatible",
        "answer_embedding_provider": Config.RAGAS_EMBEDDING_PROVIDER,
        "answer_embedding_model_id": Config.RAGAS_EMBEDDING_MODEL,
        "answer_embedding_dimension": Config.RAGAS_EMBEDDING_DIMENSION,
        "negative_evaluator": "deterministic_abstention_contract_v1",
        "data_egress": (
            "positive question, fixed generated answer, retrieved context text, and ground truth are sent "
            "to the configured external judge; answer relevance also sends text to the embedding endpoint"
        ),
        "raw_evaluation_text_stored": False,
        "metrics": list(RAGAS_METRICS),
    }
