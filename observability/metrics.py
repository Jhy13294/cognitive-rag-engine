import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


REQUEST_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
RETRIEVAL_SCORE_BUCKETS = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 0.9, 1.0)

METRIC_LABEL_NAMES = {
    "request_duration_seconds": ("route", "outcome"),
    "token_usage_total": ("kind", "source"),
    "retrieval_score": ("route", "retrieval_mode"),
    "empty_retrieval_total": ("route", "retrieval_mode"),
    "cache_operations_total": ("cache_layer", "outcome"),
}

ALLOWED_ROUTES = {"query", "query_stream"}
ALLOWED_OUTCOMES = {"success", "empty", "acl_denied", "client_error", "server_error", "stream_error"}
ALLOWED_RETRIEVAL_MODES = {"dense", "hybrid", "multi_query"}
ALLOWED_TOKEN_KINDS = {"embedding", "chat"}
ALLOWED_TOKEN_SOURCES = {"reported", "estimated"}


@dataclass
class HistogramState:
    """Store cumulative bucket counts for one bounded label set."""

    buckets: Tuple[float, ...]
    bucket_counts: List[int] = field(init=False)
    count: int = 0
    total: float = 0.0

    def __post_init__(self) -> None:
        """Initialize one cumulative counter per finite bucket."""
        self.bucket_counts = [0 for _ in self.buckets]

    def observe(self, value: float) -> None:
        """Record one value into cumulative histogram buckets."""
        numeric = float(value)
        self.count += 1
        self.total += numeric
        for index, boundary in enumerate(self.buckets):
            if numeric <= boundary:
                self.bucket_counts[index] += 1


class MetricsRegistry:
    """Thread-safe bounded-label metrics registry with Prometheus rendering."""

    def __init__(self, namespace: str = "rag"):
        """Initialize an empty in-process registry."""
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", namespace or "") is None:
            raise ValueError("Metrics namespace must contain only letters, numbers, and underscores.")
        self.namespace = namespace
        self._lock = threading.Lock()
        self._request_duration: Dict[Tuple[str, str], HistogramState] = {}
        self._retrieval_scores: Dict[Tuple[str, str], HistogramState] = {}
        self._token_usage: Dict[Tuple[str, str], float] = {}
        self._empty_retrieval: Dict[Tuple[str, str], int] = {}

    def observe_request(self, route: str, outcome: str, duration_seconds: float) -> None:
        """Observe request latency for a bounded route and outcome."""
        self._validate(route, ALLOWED_ROUTES, "route")
        self._validate(outcome, ALLOWED_OUTCOMES, "outcome")
        with self._lock:
            state = self._request_duration.setdefault(
                (route, outcome),
                HistogramState(REQUEST_DURATION_BUCKETS),
            )
            state.observe(max(0.0, duration_seconds))

    def add_token_usage(self, kind: str, source: str, tokens: int) -> None:
        """Add reported or estimated token usage with bounded labels."""
        self._validate(kind, ALLOWED_TOKEN_KINDS, "kind")
        self._validate(source, ALLOWED_TOKEN_SOURCES, "source")
        if tokens < 0:
            raise ValueError("Token usage cannot be negative.")
        with self._lock:
            key = (kind, source)
            self._token_usage[key] = self._token_usage.get(key, 0.0) + int(tokens)

    def observe_scores(self, route: str, retrieval_mode: str, scores: Iterable[float]) -> None:
        """Observe top-k scores in fixed histogram buckets."""
        self._validate(route, ALLOWED_ROUTES, "route")
        self._validate(retrieval_mode, ALLOWED_RETRIEVAL_MODES, "retrieval_mode")
        with self._lock:
            state = self._retrieval_scores.setdefault(
                (route, retrieval_mode),
                HistogramState(RETRIEVAL_SCORE_BUCKETS),
            )
            for score in scores:
                state.observe(float(score))

    def increment_empty_retrieval(self, route: str, retrieval_mode: str) -> None:
        """Increment the empty retrieval counter."""
        self._validate(route, ALLOWED_ROUTES, "route")
        self._validate(retrieval_mode, ALLOWED_RETRIEVAL_MODES, "retrieval_mode")
        with self._lock:
            key = (route, retrieval_mode)
            self._empty_retrieval[key] = self._empty_retrieval.get(key, 0) + 1

    def render(self, cache_stats: Optional[Mapping[str, Any]] = None) -> str:
        """Render all core metrics in Prometheus text exposition format."""
        with self._lock:
            request_duration = {
                key: self._copy_histogram(state)
                for key, state in self._request_duration.items()
            }
            retrieval_scores = {
                key: self._copy_histogram(state)
                for key, state in self._retrieval_scores.items()
            }
            token_usage = dict(self._token_usage)
            empty_retrieval = dict(self._empty_retrieval)

        lines = []
        self._render_histogram(
            lines,
            "request_duration_seconds",
            "RAG query request latency in seconds.",
            METRIC_LABEL_NAMES["request_duration_seconds"],
            request_duration,
        )
        self._render_counter(
            lines,
            "token_usage_total",
            "Reported or locally estimated token usage.",
            METRIC_LABEL_NAMES["token_usage_total"],
            {
                (kind, source): token_usage.get((kind, source), 0.0)
                for kind in sorted(ALLOWED_TOKEN_KINDS)
                for source in sorted(ALLOWED_TOKEN_SOURCES)
            },
        )
        self._render_histogram(
            lines,
            "retrieval_score",
            "Distribution of scores returned in top-k sources.",
            METRIC_LABEL_NAMES["retrieval_score"],
            retrieval_scores,
        )
        self._render_counter(
            lines,
            "empty_retrieval_total",
            "Queries that returned no sources.",
            METRIC_LABEL_NAMES["empty_retrieval_total"],
            empty_retrieval,
        )
        self._render_counter(
            lines,
            "cache_operations_total",
            "Cache hits and misses from the underlying L1, L2, and L3 store.",
            METRIC_LABEL_NAMES["cache_operations_total"],
            self._cache_counters(cache_stats),
        )
        return "\n".join(lines) + "\n"

    def label_names(self) -> Dict[str, Tuple[str, ...]]:
        """Return a copy of the bounded metric label schema."""
        return dict(METRIC_LABEL_NAMES)

    def _render_histogram(
        self,
        lines: List[str],
        name: str,
        help_text: str,
        label_names: Tuple[str, ...],
        states: Mapping[Tuple[str, ...], HistogramState],
    ) -> None:
        """Append one histogram family in Prometheus format."""
        metric_name = f"{self.namespace}_{name}"
        lines.extend((f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} histogram"))
        for labels, state in sorted(states.items()):
            base_labels = dict(zip(label_names, labels))
            for boundary, count in zip(state.buckets, state.bucket_counts):
                bucket_labels = dict(base_labels)
                bucket_labels["le"] = self._format_number(boundary)
                lines.append(f"{metric_name}_bucket{self._format_labels(bucket_labels)} {count}")
            infinite_labels = dict(base_labels)
            infinite_labels["le"] = "+Inf"
            lines.append(f"{metric_name}_bucket{self._format_labels(infinite_labels)} {state.count}")
            lines.append(f"{metric_name}_sum{self._format_labels(base_labels)} {self._format_number(state.total)}")
            lines.append(f"{metric_name}_count{self._format_labels(base_labels)} {state.count}")

    def _render_counter(
        self,
        lines: List[str],
        name: str,
        help_text: str,
        label_names: Tuple[str, ...],
        values: Mapping[Tuple[str, ...], float],
    ) -> None:
        """Append one counter family in Prometheus format."""
        metric_name = f"{self.namespace}_{name}"
        lines.extend((f"# HELP {metric_name} {help_text}", f"# TYPE {metric_name} counter"))
        for labels, value in sorted(values.items()):
            label_values = dict(zip(label_names, labels))
            lines.append(f"{metric_name}{self._format_labels(label_values)} {self._format_number(value)}")

    def _cache_counters(
        self,
        cache_stats: Optional[Mapping[str, Any]],
    ) -> Dict[Tuple[str, str], float]:
        """Convert existing cache store counters into bounded metric labels."""
        stats = cache_stats or {}
        hits = stats.get("hits", {}) if isinstance(stats, Mapping) else {}
        misses = stats.get("misses", {}) if isinstance(stats, Mapping) else {}
        values = {}
        for layer in ("L1", "L2", "L3"):
            raw_layer = layer.lower()
            values[(layer, "hit")] = float(hits.get(raw_layer, 0) or 0)
            values[(layer, "miss")] = float(misses.get(raw_layer, 0) or 0)
        return values

    @staticmethod
    def _copy_histogram(state: HistogramState) -> HistogramState:
        """Copy mutable histogram state for lock-free rendering."""
        copied = HistogramState(state.buckets)
        copied.bucket_counts = list(state.bucket_counts)
        copied.count = state.count
        copied.total = state.total
        return copied

    @staticmethod
    def _format_labels(labels: Mapping[str, str]) -> str:
        """Render deterministic Prometheus labels."""
        if not labels:
            return ""
        values = []
        for key, value in labels.items():
            escaped = str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
            values.append(f'{key}="{escaped}"')
        return "{" + ",".join(values) + "}"

    @staticmethod
    def _format_number(value: float) -> str:
        """Render finite metric values without locale-specific formatting."""
        numeric = float(value)
        if math.isinf(numeric):
            return "+Inf" if numeric > 0 else "-Inf"
        if numeric.is_integer():
            return str(int(numeric))
        return format(numeric, ".12g")

    @staticmethod
    def _validate(value: str, allowed: set[str], label_name: str) -> None:
        """Reject unbounded or unknown label values before series creation."""
        if value not in allowed:
            raise ValueError(f"Unsupported {label_name} label: {value}")
