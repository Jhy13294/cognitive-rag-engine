import hashlib
import hmac
import json
import logging
import os
import queue
import re
import threading
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Iterable, Optional, Protocol

from logger import mask_sensitive_info, setup_logger

logger = setup_logger(__name__)


class AuditSink(Protocol):
    """Persist one sanitized structured audit record."""

    def emit(self, record: Dict[str, Any]) -> None:
        """Persist a single audit record."""


class _RaisingRotatingFileHandler(RotatingFileHandler):
    """Raise handler failures so the audit worker can emit a stable warning."""

    def handleError(self, record: logging.LogRecord) -> None:
        """Re-raise the active logging exception for worker-level handling."""
        raise


class JSONLineAuditSink:
    """Write JSON-line audit records through a bounded background queue."""

    def __init__(
        self,
        path: str,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
        queue_size: int = 10000,
    ):
        """Initialize a non-blocking rotating JSON-line sink."""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)

        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._sentinel = object()
        self._handler = _RaisingRotatingFileHandler(
            filename=path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._closed = False
        self._worker = threading.Thread(
            target=self._run_worker,
            name="audit-jsonl-writer",
            daemon=True,
        )
        self._worker.start()

    def emit(self, record: Dict[str, Any]) -> None:
        """Queue one JSON-line record without blocking the request thread."""
        if self._closed:
            raise RuntimeError("Audit sink is closed.")
        payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        self._queue.put_nowait((str(record.get("request_id", "unknown")), payload))

    def close(self) -> None:
        """Flush queued records and close the rotating file handler."""
        if self._closed:
            return
        self._closed = True
        self._queue.put(self._sentinel)
        self._worker.join(timeout=5.0)
        if self._worker.is_alive():
            logger.warning(
                "Audit worker did not stop cleanly | alert_code=audit_worker_shutdown_timeout"
            )
            return
        self._handler.close()

    def _run_worker(self) -> None:
        """Drain queued JSON records and contain filesystem write failures."""
        while True:
            item = self._queue.get()
            try:
                if item is self._sentinel:
                    return
                request_id, payload = item
                log_record = logging.LogRecord(
                    name="rag.audit",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg=payload,
                    args=(),
                    exc_info=None,
                )
                try:
                    self._handler.handle(log_record)
                except Exception as error:
                    logger.warning(
                        "Audit background write failed | alert_code=audit_sink_write_failed | request_id=%s | error_type=%s",
                        request_id,
                        type(error).__name__,
                    )
            finally:
                self._queue.task_done()


class StructuredAuditEmitter:
    """Build sanitized query audit records and emit them fail-open."""

    def __init__(
        self,
        sink: AuditSink,
        hash_salt: str,
        principal_mode: str = "hash",
        log_query_text: bool = False,
        query_text_max_length: int = 128,
    ):
        """Initialize audit redaction and persistence settings."""
        self.sink = sink
        self.hash_salt = hash_salt.encode("utf-8")
        self.principal_mode = principal_mode
        self.log_query_text = log_query_text
        self.query_text_max_length = query_text_max_length

    def emit_query(
        self,
        *,
        request_id: str,
        route: str,
        principal: Optional[str],
        question: str,
        top_k: int,
        retrieval_mode: str,
        sources: Iterable[Any],
        latency_ms: float,
        cache_outcome: str,
        status_code: int,
        outcome: str,
    ) -> bool:
        """Emit exactly one sanitized query record, returning false on sink failure."""
        try:
            record = self.build_query_record(
                request_id=request_id,
                route=route,
                principal=principal,
                question=question,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                sources=sources,
                latency_ms=latency_ms,
                cache_outcome=cache_outcome,
                status_code=status_code,
                outcome=outcome,
            )
            self.sink.emit(record)
            return True
        except Exception as error:
            logger.warning(
                "Audit record dropped fail-open | alert_code=audit_emit_failed | request_id=%s | outcome=%s | error_type=%s",
                request_id,
                outcome,
                type(error).__name__,
            )
            return False

    def build_query_record(
        self,
        *,
        request_id: str,
        route: str,
        principal: Optional[str],
        question: str,
        top_k: int,
        retrieval_mode: str,
        sources: Iterable[Any],
        latency_ms: float,
        cache_outcome: str,
        status_code: int,
        outcome: str,
    ) -> Dict[str, Any]:
        """Build a sanitized query audit record without source content or ACL data."""
        source_ids = [self._source_id(source) for source in sources]
        record = {
            "request_id": request_id,
            "ts": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "route": route,
            "principal_id": self._principal_id(principal),
            "query_hash": self._digest("query", question),
            "top_k": int(top_k),
            "retrieval_mode": retrieval_mode,
            "n_sources": len(source_ids),
            "source_ids": source_ids,
            "latency_ms": round(max(0.0, float(latency_ms)), 3),
            "cache_outcome": cache_outcome,
            "status_code": int(status_code),
            "outcome": outcome,
        }
        if self.log_query_text:
            record["query_text"] = question[: self.query_text_max_length]
        return record

    def close(self) -> None:
        """Close the underlying sink when it owns background resources."""
        close = getattr(self.sink, "close", None)
        if close is not None:
            close()

    def _principal_id(self, principal: Optional[str]) -> str:
        """Return the configured audit representation for a principal."""
        if principal is None or not str(principal).strip():
            return "anonymous"
        normalized = str(principal).strip()
        if self.principal_mode == "raw":
            return normalized
        if self.principal_mode == "mask":
            return mask_sensitive_info(normalized, visible_length=2)
        return self._digest("principal", normalized, prefix="p_")

    def _source_id(self, source: Any) -> str:
        """Return a stable source identifier without logging source paths or content."""
        metadata = dict(getattr(source, "metadata", {}) or {})
        for key in ("id", "record_id", "child_id", "parent_id"):
            value = metadata.get(key)
            if value:
                identifier = str(value)
                if re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", identifier):
                    return identifier
                return self._digest("source_id", identifier, prefix="s_")

        descriptor_parts = [
            str(metadata.get(key, ""))
            for key in ("source", "start_char", "end_char", "chunk_index")
        ]
        descriptor_parts.extend(
            (
                str(getattr(source, "index", "")),
                str(getattr(source, "content", "")),
            )
        )
        descriptor = "|".join(descriptor_parts)
        return self._digest("source", descriptor, prefix="s_")

    def _digest(self, domain: str, value: str, prefix: str = "q_") -> str:
        """Return a domain-separated keyed digest for sensitive audit values."""
        message = f"{domain}\0{value}".encode("utf-8")
        digest = hmac.new(self.hash_salt, message, hashlib.sha256).hexdigest()
        return f"{prefix}{digest}"
