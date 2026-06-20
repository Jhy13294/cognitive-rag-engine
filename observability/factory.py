from typing import Any, Callable, Iterable, Optional

from tokenization import create_token_counter

from .audit import AuditSink, JSONLineAuditSink, StructuredAuditEmitter
from .manager import ObservabilityManager
from .metrics import MetricsRegistry


def create_observability_manager(
    config: Any,
    *,
    cache_store: Any = None,
    audit_sink: Optional[AuditSink] = None,
    metrics_registry: Optional[MetricsRegistry] = None,
    alert_hooks: Optional[Iterable[Callable]] = None,
) -> ObservabilityManager:
    """Create configured audit and metrics components for the service layer."""
    audit_emitter = None
    if config.AUDIT_ENABLED:
        selected_sink = audit_sink or JSONLineAuditSink(
            path=config.AUDIT_LOG_PATH,
            max_bytes=config.AUDIT_LOG_MAX_BYTES,
            backup_count=config.AUDIT_LOG_BACKUP_COUNT,
            queue_size=config.AUDIT_QUEUE_SIZE,
        )
        audit_emitter = StructuredAuditEmitter(
            sink=selected_sink,
            hash_salt=config.AUDIT_HASH_SALT,
            principal_mode=config.AUDIT_PRINCIPAL_MODE,
            log_query_text=config.AUDIT_LOG_QUERY_TEXT,
            query_text_max_length=config.AUDIT_QUERY_TEXT_MAX_LENGTH,
        )

    selected_metrics = metrics_registry
    if config.METRICS_ENABLED and selected_metrics is None:
        selected_metrics = MetricsRegistry(namespace=config.METRICS_NAMESPACE)

    cache_stats_provider = None
    if cache_store is not None and hasattr(cache_store, "stats"):
        cache_stats_provider = cache_store.stats

    return ObservabilityManager(
        audit_emitter=audit_emitter,
        metrics_registry=selected_metrics,
        cache_stats_provider=cache_stats_provider,
        token_counter=create_token_counter(config.TOKENIZER_ENCODING),
        alert_hooks=alert_hooks,
    )
