import asyncio
import inspect
import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from logger import setup_logger

logger = setup_logger(__name__, level=logging.INFO)


@dataclass
class CacheSettings:
    """Runtime settings for Redis-backed caching."""

    enabled: bool = False
    redis_url: str = "redis://localhost:6379/0"
    namespace: str = "rag-cache"
    embedding_enabled: bool = True
    retrieval_enabled: bool = True
    answer_enabled: bool = True
    embedding_ttl: int = 604800
    retrieval_ttl: int = 900
    answer_ttl: int = 300
    timeout_seconds: float = 0.25


@dataclass
class CacheMetrics:
    """In-process cache metrics exposed by the service layer."""

    hits: Dict[str, int] = field(default_factory=lambda: {"l1": 0, "l2": 0, "l3": 0})
    misses: Dict[str, int] = field(default_factory=lambda: {"l1": 0, "l2": 0, "l3": 0})
    writes: Dict[str, int] = field(default_factory=lambda: {"l1": 0, "l2": 0, "l3": 0})
    errors: int = 0
    corpus_version_reads: int = 0
    corpus_version_bumps: int = 0

    def snapshot(self) -> Dict[str, Any]:
        """Return a copy safe for HTTP responses."""
        return {
            "hits": dict(self.hits),
            "misses": dict(self.misses),
            "writes": dict(self.writes),
            "errors": self.errors,
            "corpus_version_reads": self.corpus_version_reads,
            "corpus_version_bumps": self.corpus_version_bumps,
        }


class RedisCacheStore:
    """Fail-open Redis cache store with async I/O and typed metrics."""

    def __init__(
        self,
        client: Any,
        settings: CacheSettings,
        client_factory: Optional[Callable[[], Any]] = None,
        use_background_loop: bool = False,
    ):
        """Initialize the cache store."""
        self.client = client
        self.settings = settings
        self.metrics = CacheMetrics()
        self._client_factory = client_factory
        self._use_background_loop = use_background_loop
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        self._loop_thread_id: Optional[int] = None
        self._closed = False
        self._metrics_lock = threading.Lock()

        if self._use_background_loop:
            self._start_background_loop()

    @classmethod
    def from_url(cls, settings: CacheSettings) -> "RedisCacheStore":
        """Create a store whose redis.asyncio client is pinned to a private loop."""
        from redis import asyncio as redis_asyncio

        return cls(
            client=None,
            settings=settings,
            client_factory=lambda: redis_asyncio.from_url(
                settings.redis_url, decode_responses=False
            ),
            use_background_loop=True,
        )

    async def get_json(self, layer: str, key: str) -> Optional[Any]:
        """Read JSON data, returning None on miss or Redis failure."""
        return await self._run_on_store_loop(
            lambda: self._get_json_impl(layer, key),
            default=None,
            operation="get_json",
        )

    async def _get_json_impl(self, layer: str, key: str) -> Optional[Any]:
        """Read JSON data on the client-owned event loop."""
        raw_value = await self._safe_call("get", key)
        if raw_value is None:
            self.record_miss(layer)
            return None

        try:
            if isinstance(raw_value, bytes):
                raw_value = raw_value.decode("utf-8")
            value = json.loads(raw_value)
        except Exception as e:
            logger.warning(
                "Invalid cached JSON payload; treating as miss | layer=%s | error=%s", layer, e
            )
            self.record_miss(layer)
            return None

        self.record_hit(layer)
        return value

    async def set_json(self, layer: str, key: str, value: Any, ttl: int) -> None:
        """Write JSON data, ignoring Redis failures."""
        await self._run_on_store_loop(
            lambda: self._set_json_impl(layer, key, value, ttl),
            default=None,
            operation="set_json",
        )

    async def _set_json_impl(self, layer: str, key: str, value: Any, ttl: int) -> None:
        """Write JSON data on the client-owned event loop."""
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        await self._safe_set(layer, key, payload, ttl)

    async def get_bytes(self, layer: str, key: str) -> Optional[bytes]:
        """Read raw bytes, returning None on miss or Redis failure."""
        return await self._run_on_store_loop(
            lambda: self._get_bytes_impl(layer, key),
            default=None,
            operation="get_bytes",
        )

    async def _get_bytes_impl(self, layer: str, key: str) -> Optional[bytes]:
        """Read raw bytes on the client-owned event loop."""
        raw_value = await self._safe_call("get", key)
        if raw_value is None:
            self.record_miss(layer)
            return None
        if isinstance(raw_value, str):
            raw_value = raw_value.encode("latin1")
        self.record_hit(layer)
        return bytes(raw_value)

    async def set_bytes(self, layer: str, key: str, value: bytes, ttl: int) -> None:
        """Write raw bytes, ignoring Redis failures."""
        await self._run_on_store_loop(
            lambda: self._set_bytes_impl(layer, key, value, ttl),
            default=None,
            operation="set_bytes",
        )

    async def _set_bytes_impl(self, layer: str, key: str, value: bytes, ttl: int) -> None:
        """Write raw bytes on the client-owned event loop."""
        await self._safe_set(layer, key, value, ttl)

    async def get_corpus_version(self) -> Optional[int]:
        """Return the persistent Redis corpus version, or None when unavailable."""
        return await self._run_on_store_loop(
            self._get_corpus_version_impl,
            default=None,
            operation="get_corpus_version",
        )

    async def _get_corpus_version_impl(self) -> Optional[int]:
        """Return the corpus version on the client-owned event loop."""
        version_key = self._corpus_version_key()
        raw_value = await self._safe_call("get", version_key)
        if raw_value is None:
            initialized = await self._safe_call("setnx", version_key, 0)
            if initialized is None:
                return None
            raw_value = await self._safe_call("get", version_key)
            if raw_value is None:
                return None

        try:
            if isinstance(raw_value, bytes):
                raw_value = raw_value.decode("utf-8")
            version = int(raw_value)
        except (TypeError, ValueError):
            logger.warning("Invalid Redis corpus version; disabling L2/L3 for this request")
            return None

        self._increment_metric("corpus_version_reads")
        return version

    async def bump_corpus_version(self) -> Optional[int]:
        """Atomically increment the persistent corpus version."""
        return await self._run_on_store_loop(
            self._bump_corpus_version_impl,
            default=None,
            operation="bump_corpus_version",
        )

    async def _bump_corpus_version_impl(self) -> Optional[int]:
        """Atomically increment the corpus version on the client-owned event loop."""
        version = await self._safe_call("incr", self._corpus_version_key())
        if version is None:
            return None
        self._increment_metric("corpus_version_bumps")
        return int(version)

    async def ping(self, timeout_seconds: Optional[float] = None) -> bool:
        """Return whether Redis responds through the store-owned event loop."""
        return bool(
            await self._run_on_store_loop(
                lambda: self._ping_impl(timeout_seconds),
                default=False,
                operation="ping",
            )
        )

    async def _ping_impl(self, timeout_seconds: Optional[float] = None) -> bool:
        """Ping Redis on the client-owned event loop."""
        timeout = timeout_seconds if timeout_seconds is not None else self.settings.timeout_seconds
        try:
            client = await self._ensure_client()
            result = client.ping()
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=timeout)
            return bool(result)
        except Exception as e:
            self._increment_metric("errors")
            logger.warning("Redis cache ping failed | error=%s", e)
            return False

    def stats(self) -> Dict[str, Any]:
        """Return cache observability counters."""
        with self._metrics_lock:
            payload = self.metrics.snapshot()
        payload.update(
            {
                "enabled": self.settings.enabled,
                "namespace": self.settings.namespace,
                "embedding_enabled": self.settings.embedding_enabled,
                "retrieval_enabled": self.settings.retrieval_enabled,
                "answer_enabled": self.settings.answer_enabled,
                "background_loop_enabled": self._use_background_loop,
                "background_loop_thread_id": self._loop_thread_id,
                "background_loop_alive": bool(self._loop_thread and self._loop_thread.is_alive()),
                "client_type": type(self.client).__name__ if self.client is not None else None,
            }
        )
        return payload

    def record_hit(self, layer: str) -> None:
        """Increment a layer hit counter."""
        with self._metrics_lock:
            self.metrics.hits[layer] = self.metrics.hits.get(layer, 0) + 1

    def record_miss(self, layer: str) -> None:
        """Increment a layer miss counter."""
        with self._metrics_lock:
            self.metrics.misses[layer] = self.metrics.misses.get(layer, 0) + 1

    async def aclose(self) -> None:
        """Close the Redis client and stop the private event loop if present."""
        if self._closed:
            return
        self._closed = True

        if self._use_background_loop:
            if self._loop and self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self._close_client(), self._loop)
                await asyncio.wrap_future(future)
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread and self._loop_thread.is_alive():
                await asyncio.to_thread(self._loop_thread.join, 2.0)
            return

        await self._close_client()

    def close(self) -> None:
        """Synchronous close helper for CLI shutdown and tests."""
        if self._closed:
            return
        self._closed = True

        if self._use_background_loop:
            if self._loop and self._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(self._close_client(), self._loop)
                future.result(timeout=max(self.settings.timeout_seconds, 1.0))
                self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2.0)
            return

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._close_client())
        else:
            logger.warning("RedisCacheStore.close called from a running loop; use aclose instead")

    async def _safe_set(self, layer: str, key: str, value: Any, ttl: int) -> None:
        ttl_value = int(ttl) if ttl and ttl > 0 else None
        kwargs = {"ex": ttl_value} if ttl_value else {}
        result = await self._safe_call("set", key, value, **kwargs)
        if result is not None:
            with self._metrics_lock:
                self.metrics.writes[layer] = self.metrics.writes.get(layer, 0) + 1

    async def _safe_call(self, method_name: str, *args, **kwargs) -> Any:
        try:
            client = await self._ensure_client()
            method = getattr(client, method_name)
            result = method(*args, **kwargs)
            if not inspect.isawaitable(result):
                return result
            return await asyncio.wait_for(result, timeout=self.settings.timeout_seconds)
        except Exception as e:
            self._increment_metric("errors")
            logger.warning(
                "Redis cache unavailable; fail-open as cache miss | operation=%s | error=%s",
                method_name,
                e,
            )
            return None

    async def _run_on_store_loop(
        self, coroutine_factory: Callable[[], Any], default: Any, operation: str
    ) -> Any:
        if not self._use_background_loop:
            return await coroutine_factory()
        if self._closed or self._loop is None or not self._loop.is_running():
            self._increment_metric("errors")
            logger.warning(
                "Redis cache loop is unavailable; fail-open as cache miss | operation=%s", operation
            )
            return default

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is self._loop:
            return await coroutine_factory()

        try:
            future = asyncio.run_coroutine_threadsafe(coroutine_factory(), self._loop)
            return await asyncio.wrap_future(future)
        except Exception as e:
            self._increment_metric("errors")
            logger.warning(
                "Redis cache background loop call failed; fail-open as cache miss | operation=%s | error=%s",
                operation,
                e,
            )
            return default

    async def _ensure_client(self) -> Any:
        if self.client is None:
            if self._client_factory is None:
                raise RuntimeError("Redis client is not configured")
            self.client = self._client_factory()
        return self.client

    async def _close_client(self) -> None:
        if self.client is None:
            return
        for method_name in ("aclose", "close"):
            method = getattr(self.client, method_name, None)
            if method is None:
                continue
            result = method()
            if inspect.isawaitable(result):
                await result
            return

    def _increment_metric(self, metric_name: str) -> None:
        with self._metrics_lock:
            setattr(self.metrics, metric_name, getattr(self.metrics, metric_name) + 1)

    def _start_background_loop(self) -> None:
        ready = threading.Event()

        def run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
            self._loop_thread_id = threading.get_ident()
            ready.set()
            loop.run_forever()
            loop.close()

        self._loop_thread = threading.Thread(target=run_loop, name="redis-cache-loop", daemon=True)
        self._loop_thread.start()
        if not ready.wait(timeout=max(self.settings.timeout_seconds, 1.0)):
            raise RuntimeError("Redis cache background loop did not start")

    def _corpus_version_key(self) -> str:
        return f"{self.settings.namespace}:corpus:version"
