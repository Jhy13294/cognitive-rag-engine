import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from logger import setup_logger
from tokenization import TokenCounter, create_token_counter

from .audit import StructuredAuditEmitter
from .metrics import MetricsRegistry

logger = setup_logger(__name__)


@dataclass
class QueryObservation:
    """Hold bounded per-request observation state outside the RAG pipeline."""

    request_id: str
    route: str
    principal: Optional[str]
    question: str
    top_k: int
    retrieval_mode: str
    started_at: float
    pipeline: Any = None
    cache_before: Optional[Dict] = None
    finalized: bool = False


class ObservabilityManager:
    """Coordinate fail-open audit, metrics, token accounting, and alert hooks."""

    def __init__(
        self,
        audit_emitter: Optional[StructuredAuditEmitter] = None,
        metrics_registry: Optional[MetricsRegistry] = None,
        cache_stats_provider: Optional[Callable[[], Dict]] = None,
        token_counter: Optional[TokenCounter] = None,
        alert_hooks: Optional[Iterable[Callable[[str, Dict[str, Any]], None]]] = None,
    ):
        """Initialize independently optional observability components."""
        self.audit_emitter = audit_emitter
        self.metrics_registry = metrics_registry
        self.cache_stats_provider = cache_stats_provider
        self.token_counter = token_counter or create_token_counter()
        self.alert_hooks = list(alert_hooks or [])
        self._usage_lock = threading.Lock()
        self._usage_watermarks: Dict[Tuple[str, int], Tuple[int, int]] = {}

    def start_query(
        self,
        *,
        request_id: str,
        route: str,
        principal: Optional[str],
        question: str,
        top_k: int,
        retrieval_mode: str,
    ) -> QueryObservation:
        """Create per-request observation state without touching retrieval behavior."""
        return QueryObservation(
            request_id=request_id,
            route=route,
            principal=principal,
            question=question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            started_at=time.perf_counter(),
        )

    def bind_pipeline(self, observation: QueryObservation, pipeline: Any) -> None:
        """Capture pre-query cache and embedding counters from an existing pipeline."""
        observation.pipeline = pipeline
        observation.cache_before = self._safe_cache_stats(pipeline)
        self._register_usage_counter("embedding", self._embedding_usage_counter(pipeline))
        self._register_usage_counter("chat", self._chat_usage_counter(pipeline))

    def complete_query(
        self,
        observation: QueryObservation,
        *,
        sources: Optional[List[Any]] = None,
        status_code: int = 200,
        raw_response: Optional[Dict] = None,
        prompt: str = "",
        answer: str = "",
    ) -> None:
        """Finalize one query exactly once while containing all observer failures."""
        if observation.finalized:
            return
        observation.finalized = True

        source_list = list(sources or [])
        latency_seconds = max(0.0, time.perf_counter() - observation.started_at)
        outcome = self._outcome(status_code, source_list, observation.route)
        cache_after = self._safe_cache_stats(observation.pipeline)
        try:
            cache_outcome = self._cache_outcome(observation.cache_before, cache_after)
        except Exception as error:
            cache_outcome = "error"
            logger.warning(
                "Cache outcome calculation failed open | alert_code=cache_outcome_failed | error_type=%s",
                type(error).__name__,
            )

        self._safe_metric(
            "observe_request",
            observation.route,
            outcome,
            latency_seconds,
            request_id=observation.request_id,
        )
        if status_code < 400:
            self._safe_metric(
                "observe_scores",
                observation.route,
                observation.retrieval_mode,
                self._source_scores(source_list),
                request_id=observation.request_id,
            )
            if not source_list:
                self._safe_metric(
                    "increment_empty_retrieval",
                    observation.route,
                    observation.retrieval_mode,
                    request_id=observation.request_id,
                )
                self._emit_alert("empty_retrieval", observation, status_code)
            try:
                self._record_token_usage(
                    observation,
                    cache_outcome=cache_outcome,
                    raw_response=raw_response or {},
                    prompt=prompt,
                    answer=answer,
                )
            except Exception as error:
                logger.warning(
                    "Token metrics failed open | alert_code=token_metrics_failed | request_id=%s | error_type=%s",
                    observation.request_id,
                    type(error).__name__,
                )
        elif status_code == 403:
            self._emit_alert("acl_denied", observation, status_code)
        elif status_code >= 500:
            self._emit_alert("service_error", observation, status_code)

        if self.audit_emitter is not None:
            try:
                self.audit_emitter.emit_query(
                    request_id=observation.request_id,
                    route=observation.route,
                    principal=observation.principal,
                    question=observation.question,
                    top_k=observation.top_k,
                    retrieval_mode=observation.retrieval_mode,
                    sources=source_list,
                    latency_ms=latency_seconds * 1000.0,
                    cache_outcome=cache_outcome,
                    status_code=status_code,
                    outcome=outcome,
                )
            except Exception as error:
                logger.warning(
                    "Audit manager failed open | alert_code=audit_manager_failed | request_id=%s | error_type=%s",
                    observation.request_id,
                    type(error).__name__,
                )

    def render_metrics(self) -> str:
        """Render metrics fail-open, including true cache store counters."""
        if self.metrics_registry is None:
            return "# metrics disabled\n"
        try:
            cache_stats = self._safe_cache_stats(None)
            return self.metrics_registry.render(cache_stats=cache_stats)
        except Exception as error:
            logger.warning(
                "Metrics render failed open | alert_code=metrics_render_failed | error_type=%s",
                type(error).__name__,
            )
            return "# metrics unavailable\n"

    def close(self) -> None:
        """Close owned audit resources without affecting service shutdown."""
        if self.audit_emitter is None:
            return
        try:
            self.audit_emitter.close()
        except Exception as error:
            logger.warning(
                "Audit shutdown failed open | alert_code=audit_close_failed | error_type=%s",
                type(error).__name__,
            )

    def _record_token_usage(
        self,
        observation: QueryObservation,
        *,
        cache_outcome: str,
        raw_response: Dict,
        prompt: str,
        answer: str,
    ) -> None:
        """Record provider-reported usage or explicitly labeled local estimates."""
        system_prompt = str(getattr(observation.pipeline, "system_prompt", "") or "")
        estimated_chat_text = f"{system_prompt}\n{prompt}\n{answer}"
        chat_requests, reported_chat, chat_counter_available = self._claim_usage_delta(
            "chat",
            self._chat_usage_counter(observation.pipeline),
        )
        if chat_counter_available:
            if chat_requests > 0 and reported_chat > 0:
                self._safe_metric(
                    "add_token_usage",
                    "chat",
                    "reported",
                    reported_chat,
                    request_id=observation.request_id,
                )
            elif chat_requests > 0:
                self._safe_metric(
                    "add_token_usage",
                    "chat",
                    "estimated",
                    self._safe_count_tokens(estimated_chat_text),
                    request_id=observation.request_id,
                )
        elif cache_outcome != "hit_l3":
            usage = raw_response.get("usage") if isinstance(raw_response, dict) else None
            reported_chat = self._usage_total(usage)
            if reported_chat > 0:
                self._safe_metric(
                    "add_token_usage",
                    "chat",
                    "reported",
                    reported_chat,
                    request_id=observation.request_id,
                )
            else:
                self._safe_metric(
                    "add_token_usage",
                    "chat",
                    "estimated",
                    self._safe_count_tokens(estimated_chat_text),
                    request_id=observation.request_id,
                )

        embedding_requests, reported_embedding, embedding_counter_available = (
            self._claim_usage_delta(
                "embedding",
                self._embedding_usage_counter(observation.pipeline),
            )
        )
        if embedding_counter_available:
            if embedding_requests > 0 and reported_embedding > 0:
                self._safe_metric(
                    "add_token_usage",
                    "embedding",
                    "reported",
                    reported_embedding,
                    request_id=observation.request_id,
                )
            elif embedding_requests > 0:
                self._safe_metric(
                    "add_token_usage",
                    "embedding",
                    "estimated",
                    self._safe_count_tokens(observation.question),
                    request_id=observation.request_id,
                )
        elif cache_outcome not in {"hit_l1", "hit_l2", "hit_l3"}:
            self._safe_metric(
                "add_token_usage",
                "embedding",
                "estimated",
                self._safe_count_tokens(observation.question),
                request_id=observation.request_id,
            )

    def _safe_metric(self, method_name: str, *args: Any, request_id: Optional[str] = None) -> None:
        """Invoke one metric operation without allowing failures into the request path."""
        if self.metrics_registry is None:
            return
        try:
            method = getattr(self.metrics_registry, method_name)
            method(*args)
        except Exception as error:
            logger.warning(
                "Metric emit failed open | alert_code=metrics_emit_failed | metric=%s | request_id=%s | error_type=%s",
                method_name,
                request_id or "unknown",
                type(error).__name__,
            )

    def _safe_cache_stats(self, pipeline: Any) -> Optional[Dict]:
        """Read existing cache counters without changing cache behavior."""
        try:
            if pipeline is not None:
                cache_stats = getattr(pipeline, "cache_stats", None)
                if cache_stats is not None:
                    return dict(cache_stats())
            if self.cache_stats_provider is not None:
                return dict(self.cache_stats_provider())
        except Exception as error:
            logger.warning(
                "Cache stats unavailable for observability | alert_code=cache_stats_failed | error_type=%s",
                type(error).__name__,
            )
        return None

    def _emit_alert(self, code: str, observation: QueryObservation, status_code: int) -> None:
        """Emit a stable alert code and call optional hooks fail-open."""
        payload = {
            "request_id": observation.request_id,
            "route": observation.route,
            "status_code": int(status_code),
        }
        logger.warning(
            "Observability alert | alert_code=%s | request_id=%s | route=%s | status_code=%s",
            code,
            observation.request_id,
            observation.route,
            status_code,
        )
        for hook in self.alert_hooks:
            try:
                hook(code, dict(payload))
            except Exception as error:
                logger.warning(
                    "Alert hook failed open | alert_code=alert_hook_failed | hook_event=%s | error_type=%s",
                    code,
                    type(error).__name__,
                )

    @staticmethod
    def _embedding_usage_counter(pipeline: Any) -> Optional[Tuple[int, int, int]]:
        """Read cumulative OpenAI embedding requests and tokens through decorators."""
        current = pipeline
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if hasattr(current, "embedding_provider"):
                current = current.embedding_provider
                continue
            total = getattr(current, "total_prompt_tokens", None)
            requests = getattr(current, "request_count", None)
            if total is not None and requests is not None:
                return id(current), int(requests), int(total)
            if hasattr(current, "provider"):
                current = current.provider
                continue
            if hasattr(current, "_pipeline"):
                current = current._pipeline
                continue
            break
        return None

    @staticmethod
    def _chat_usage_counter(pipeline: Any) -> Optional[Tuple[int, int, int]]:
        """Read cumulative chat requests and tokens from the existing client."""
        current = pipeline
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chat_client = getattr(current, "chat_client", None)
            if chat_client is not None:
                stats_getter = getattr(chat_client, "usage_stats", None)
                if stats_getter is not None:
                    stats = stats_getter()
                    return (
                        id(chat_client),
                        int(stats.get("request_count", 0)),
                        int(stats.get("total_tokens", 0)),
                    )
                requests = getattr(chat_client, "request_count", None)
                total = getattr(chat_client, "total_tokens", None)
                if requests is not None and total is not None:
                    return id(chat_client), int(requests), int(total)
                return None
            if hasattr(current, "_pipeline"):
                current = current._pipeline
                continue
            break
        return None

    def _register_usage_counter(
        self,
        kind: str,
        counter: Optional[Tuple[int, int, int]],
    ) -> None:
        """Register the current provider watermark before a request executes."""
        if counter is None:
            return
        provider_id, requests, tokens = counter
        with self._usage_lock:
            self._usage_watermarks.setdefault((kind, provider_id), (requests, tokens))

    def _claim_usage_delta(
        self,
        kind: str,
        counter: Optional[Tuple[int, int, int]],
    ) -> Tuple[int, int, bool]:
        """Atomically claim cumulative provider deltas once across concurrent requests."""
        if counter is None:
            return 0, 0, False
        provider_id, requests, tokens = counter
        key = (kind, provider_id)
        with self._usage_lock:
            previous_requests, previous_tokens = self._usage_watermarks.get(key, (requests, tokens))
            request_delta = max(0, requests - previous_requests)
            token_delta = max(0, tokens - previous_tokens)
            self._usage_watermarks[key] = (requests, tokens)
        return request_delta, token_delta, True

    def _safe_count_tokens(self, text: str) -> int:
        """Count tokens with a deterministic fallback on counter failure."""
        try:
            return max(0, int(self.token_counter.count(text)))
        except Exception as error:
            logger.warning(
                "Token estimation failed open | alert_code=token_estimate_failed | error_type=%s",
                type(error).__name__,
            )
            return 0

    @staticmethod
    def _source_scores(sources: List[Any]) -> List[float]:
        """Return finite source scores while isolating malformed observability data."""
        scores = []
        for source in sources:
            try:
                score = float(source.score)
                if not math.isfinite(score):
                    raise ValueError("Source score must be finite.")
                scores.append(score)
            except (AttributeError, TypeError, ValueError):
                logger.warning(
                    "Source score skipped | alert_code=source_score_invalid | source_type=%s",
                    type(source).__name__,
                )
        return scores

    @staticmethod
    def _usage_total(usage: Any) -> int:
        """Extract total token usage from common provider response shapes."""
        if not isinstance(usage, dict):
            return 0
        total = int(usage.get("total_tokens", 0) or 0)
        if total > 0:
            return total
        return int(usage.get("prompt_tokens", 0) or 0) + int(usage.get("completion_tokens", 0) or 0)

    @staticmethod
    def _cache_outcome(before: Optional[Dict], after: Optional[Dict]) -> str:
        """Describe per-request cache behavior from existing cumulative counters."""
        if before is None or after is None:
            return "disabled"
        before_errors = int(before.get("errors", 0) or 0)
        after_errors = int(after.get("errors", 0) or 0)
        if after_errors > before_errors:
            return "error"

        for layer in ("l3", "l2", "l1"):
            before_hits = int((before.get("hits") or {}).get(layer, 0) or 0)
            after_hits = int((after.get("hits") or {}).get(layer, 0) or 0)
            if after_hits > before_hits:
                return f"hit_{layer}"
        for layer in ("l3", "l2", "l1"):
            before_misses = int((before.get("misses") or {}).get(layer, 0) or 0)
            after_misses = int((after.get("misses") or {}).get(layer, 0) or 0)
            if after_misses > before_misses:
                return "miss"
        return "unchanged"

    @staticmethod
    def _outcome(status_code: int, sources: List[Any], route: str) -> str:
        """Map a response into a bounded metrics outcome label."""
        if status_code == 403:
            return "acl_denied"
        if status_code >= 500:
            return "stream_error" if route == "query_stream" else "server_error"
        if status_code >= 400:
            return "client_error"
        if not sources:
            return "empty"
        return "success"
