import asyncio
import random
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, TypeVar

from embeddings.base import Vector
from logger import setup_logger

from .base import SearchResult, VectorRecord, VectorStore

logger = setup_logger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class _FallbackVectorParams:
    size: int
    distance: Any


@dataclass
class _FallbackPointStruct:
    id: Any
    vector: Vector
    payload: Dict


@dataclass
class _FallbackMatchValue:
    value: Any


@dataclass
class _FallbackMatchAny:
    any: List[Any]


@dataclass
class _FallbackFieldCondition:
    key: str
    match: Any


@dataclass
class _FallbackFilter:
    must: List[Any]


@dataclass
class _FallbackPointIdsList:
    points: List[Any]


class _FallbackPayloadSchemaType:
    KEYWORD = "keyword"
    INTEGER = "integer"


class _FallbackDistance:
    COSINE = "Cosine"
    DOT = "Dot"
    EUCLID = "Euclid"


class _FallbackQdrantModels:
    PayloadSchemaType = _FallbackPayloadSchemaType
    Distance = _FallbackDistance
    VectorParams = _FallbackVectorParams
    PointStruct = _FallbackPointStruct
    MatchValue = _FallbackMatchValue
    MatchAny = _FallbackMatchAny
    FieldCondition = _FallbackFieldCondition
    Filter = _FallbackFilter
    PointIdsList = _FallbackPointIdsList


try:
    from qdrant_client.http import models as qmodels

    _QDRANT_MODELS_AVAILABLE = True
except ImportError:
    qmodels = _FallbackQdrantModels()
    _QDRANT_MODELS_AVAILABLE = False


INDEXED_PAYLOAD_FIELDS = {
    "source": qmodels.PayloadSchemaType.KEYWORD,
    "file_type": qmodels.PayloadSchemaType.KEYWORD,
    "acl": qmodels.PayloadSchemaType.KEYWORD,
    "chunk_index": qmodels.PayloadSchemaType.INTEGER,
}

_DISTANCE_MAP = {
    "cosine": qmodels.Distance.COSINE,
    "dot": qmodels.Distance.DOT,
    "euclid": qmodels.Distance.EUCLID,
    "euclidean": qmodels.Distance.EUCLID,
}


class QdrantVectorStoreError(RuntimeError):
    """Raised when a Qdrant vector-store operation fails."""


def is_qdrant_client_available() -> bool:
    """Return whether the Qdrant SDK is installed."""
    if not _QDRANT_MODELS_AVAILABLE:
        return False

    try:
        import qdrant_client  # noqa: F401

        return True
    except ImportError:
        return False


class QdrantVectorStore(VectorStore):
    """Production vector store backed by Qdrant."""

    def __init__(
        self,
        collection_name: str,
        dimension: int,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        url: Optional[str] = None,
        distance: str = "cosine",
        client: Optional[Any] = None,
        recreate: bool = False,
        batch_size: int = 64,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
    ):
        """Initialize the Qdrant-backed vector store and ensure its collection."""
        if not collection_name:
            raise ValueError("collection_name is required")
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if max_retries < 0:
            raise ValueError("max_retries must be zero or greater")

        normalized_distance = distance.lower()
        if normalized_distance not in _DISTANCE_MAP:
            supported = ", ".join(sorted(_DISTANCE_MAP))
            raise ValueError(f"Unsupported distance metric: {distance}. Supported: {supported}")

        self.collection_name = collection_name
        self.dimension = dimension
        self.distance = normalized_distance
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self._client = client or self._build_client(
            host=host,
            port=port,
            api_key=api_key,
            url=url,
            timeout=timeout,
        )

        self._ensure_collection(recreate=recreate)

    def add_records(self, records: Iterable[VectorRecord]) -> List[str]:
        """Batch-upsert vector records and return their stable business ids."""
        record_list = list(records)
        if not record_list:
            return []

        points = []
        ids = []
        for record in record_list:
            self._validate_record(record)
            point_id = self._to_point_id(record.id)
            payload = dict(record.metadata)
            payload["content"] = record.content
            payload["record_id"] = record.id
            payload["id"] = record.id
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=list(record.embedding),
                    payload=payload,
                )
            )
            ids.append(record.id)

        start_time = time.monotonic()
        batch_count = 0
        for batch in self._iter_batches(points, self.batch_size):
            batch_count += 1
            self._call_with_retries(
                "qdrant_upsert",
                self._client.upsert,
                collection_name=self.collection_name,
                points=batch,
                wait=True,
            )

        elapsed = time.monotonic() - start_time
        logger.info(
            "Qdrant records upserted | collection=%s | count=%s | batches=%s | elapsed=%.3fs",
            self.collection_name,
            len(ids),
            batch_count,
            elapsed,
        )
        return ids

    async def async_add_records(self, records: Iterable[VectorRecord]) -> List[str]:
        """Run add_records from async code without blocking the event loop."""
        return await asyncio.to_thread(self.add_records, list(records))

    def similarity_search(
        self,
        query_embedding: Vector,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Search by vector similarity with optional metadata pre-filtering."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self._validate_vector(query_embedding)
        query_filter = self._build_filter(metadata_filter)

        if hasattr(self._client, "search"):
            hits = self._call_with_retries(
                "qdrant_search",
                self._client.search,
                collection_name=self.collection_name,
                query_vector=list(query_embedding),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
        else:
            query_result = self._call_with_retries(
                "qdrant_query_points",
                self._client.query_points,
                collection_name=self.collection_name,
                query=list(query_embedding),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            hits = getattr(query_result, "points", query_result)

        return [self._hit_to_result(hit) for hit in hits]

    async def async_similarity_search(
        self,
        query_embedding: Vector,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Run similarity_search from async code without blocking the event loop."""
        return await asyncio.to_thread(
            self.similarity_search,
            query_embedding,
            top_k,
            metadata_filter,
        )

    def get_record(self, record_id: str) -> Optional[VectorRecord]:
        """Return a record by its stable business id."""
        points = self._call_with_retries(
            "qdrant_retrieve",
            self._client.retrieve,
            collection_name=self.collection_name,
            ids=[self._to_point_id(record_id)],
            with_payload=True,
            with_vectors=True,
        )
        if not points:
            return None
        return self._point_to_record(points[0])

    async def async_get_record(self, record_id: str) -> Optional[VectorRecord]:
        """Run get_record from async code without blocking the event loop."""
        return await asyncio.to_thread(self.get_record, record_id)

    def list_records(self, limit: Optional[int] = None) -> List[VectorRecord]:
        """Scroll all stored records for deterministic local index reconstruction."""
        if limit is not None and limit < 0:
            raise ValueError("limit must be non-negative")
        if not hasattr(self._client, "scroll"):
            raise QdrantVectorStoreError("Qdrant client does not support scroll; cannot list records.")

        records: List[VectorRecord] = []
        offset = None
        while True:
            remaining = None if limit is None else limit - len(records)
            if remaining is not None and remaining <= 0:
                break

            batch_limit = self.batch_size if remaining is None else min(self.batch_size, remaining)
            scroll_result = self._call_with_retries(
                "qdrant_scroll",
                self._client.scroll,
                collection_name=self.collection_name,
                limit=batch_limit,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            points, next_offset = self._normalize_scroll_result(scroll_result)
            records.extend(self._point_to_record(point) for point in points)

            if not next_offset:
                break
            offset = next_offset

        return records

    async def async_list_records(self, limit: Optional[int] = None) -> List[VectorRecord]:
        """Run list_records from async code without blocking the event loop."""
        return await asyncio.to_thread(self.list_records, limit)

    def delete(self, record_id: str) -> bool:
        """Delete a record by its stable business id."""
        if self.get_record(record_id) is None:
            return False

        self._call_with_retries(
            "qdrant_delete",
            self._client.delete,
            collection_name=self.collection_name,
            points_selector=qmodels.PointIdsList(points=[self._to_point_id(record_id)]),
            wait=True,
        )
        return True

    async def async_delete(self, record_id: str) -> bool:
        """Run delete from async code without blocking the event loop."""
        return await asyncio.to_thread(self.delete, record_id)

    def clear(self) -> None:
        """Delete all records by recreating the collection."""
        self._call_with_retries(
            "qdrant_delete_collection",
            self._client.delete_collection,
            collection_name=self.collection_name,
        )
        self._ensure_collection(recreate=False)

    async def async_clear(self) -> None:
        """Run clear from async code without blocking the event loop."""
        await asyncio.to_thread(self.clear)

    def count(self) -> int:
        """Return the number of stored points."""
        result = self._call_with_retries(
            "qdrant_count",
            self._client.count,
            collection_name=self.collection_name,
            exact=True,
        )
        if isinstance(result, dict):
            return int(result.get("count", 0))
        return int(getattr(result, "count", 0))

    async def async_count(self) -> int:
        """Run count from async code without blocking the event loop."""
        return await asyncio.to_thread(self.count)

    def _build_client(
        self,
        host: str,
        port: int,
        api_key: Optional[str],
        url: Optional[str],
        timeout: float,
    ) -> Any:
        """Create a Qdrant SDK client."""
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise QdrantVectorStoreError(
                "qdrant-client is required for QdrantVectorStore. "
                "Install it with `pip install qdrant-client`."
            ) from e

        client_kwargs = {"api_key": api_key, "timeout": timeout}
        client_kwargs = {key: value for key, value in client_kwargs.items() if value is not None}

        if url:
            return QdrantClient(url=url, **client_kwargs)
        return QdrantClient(host=host, port=port, **client_kwargs)

    def _ensure_collection(self, recreate: bool) -> None:
        """Create or validate the Qdrant collection."""
        exists = bool(
            self._call_with_retries(
                "qdrant_collection_exists",
                self._client.collection_exists,
                collection_name=self.collection_name,
            )
        )

        if exists and recreate:
            self._call_with_retries(
                "qdrant_delete_collection",
                self._client.delete_collection,
                collection_name=self.collection_name,
            )
            exists = False

        if exists:
            self._validate_existing_collection()
            self._ensure_payload_indexes()
            return

        self._call_with_retries(
            "qdrant_create_collection",
            self._client.create_collection,
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(
                size=self.dimension,
                distance=_DISTANCE_MAP[self.distance],
            ),
        )
        self._ensure_payload_indexes()
        logger.info(
            "Qdrant collection ready | collection=%s | dimension=%s | distance=%s",
            self.collection_name,
            self.dimension,
            self.distance,
        )

    def _validate_existing_collection(self) -> None:
        """Validate dimension when the SDK can expose collection metadata."""
        if not hasattr(self._client, "get_collection"):
            return

        collection_info = self._call_with_retries(
            "qdrant_get_collection",
            self._client.get_collection,
            collection_name=self.collection_name,
        )
        vector_size = self._extract_vector_size(collection_info)
        if vector_size is not None and vector_size != self.dimension:
            raise ValueError(
                f"Qdrant collection dimension mismatch: expected {self.dimension}, got {vector_size}"
            )

    def _ensure_payload_indexes(self) -> None:
        """Ensure payload indexes used by metadata filters exist."""
        for field_name, field_schema in INDEXED_PAYLOAD_FIELDS.items():
            self._call_with_retries(
                "qdrant_create_payload_index",
                self._client.create_payload_index,
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=field_schema,
            )

    def _build_filter(self, metadata_filter: Optional[Dict]) -> Optional[Any]:
        """Translate exact-match and list-membership filters into Qdrant filters."""
        if not metadata_filter:
            return None

        conditions = []
        for key, value in metadata_filter.items():
            if isinstance(value, (list, tuple, set)):
                match = qmodels.MatchAny(any=list(value))
            else:
                match = qmodels.MatchValue(value=value)

            conditions.append(qmodels.FieldCondition(key=key, match=match))

        if not conditions:
            return None
        return qmodels.Filter(must=conditions)

    def _hit_to_result(self, hit: Any) -> SearchResult:
        """Convert a Qdrant search hit into a SearchResult."""
        score = self._normalize_score(float(getattr(hit, "score", 0.0)))
        record = self._payload_to_record(
            point_id=getattr(hit, "id", ""),
            payload=getattr(hit, "payload", {}),
            embedding=self._extract_vector(hit),
        )
        return SearchResult(record=record, score=score)

    def _point_to_record(self, point: Any) -> VectorRecord:
        """Convert a retrieved Qdrant point into a VectorRecord."""
        return self._payload_to_record(
            point_id=getattr(point, "id", ""),
            payload=getattr(point, "payload", {}),
            embedding=self._extract_vector(point),
        )

    def _normalize_scroll_result(self, scroll_result: Any) -> tuple:
        """Return points and next offset from multiple Qdrant SDK shapes."""
        if isinstance(scroll_result, tuple):
            points = scroll_result[0] if len(scroll_result) > 0 else []
            next_offset = scroll_result[1] if len(scroll_result) > 1 else None
            return list(points or []), next_offset

        points = getattr(scroll_result, "points", scroll_result)
        next_offset = getattr(scroll_result, "next_page_offset", None)
        return list(points or []), next_offset

    def _payload_to_record(self, point_id: Any, payload: Optional[Dict], embedding: Optional[Vector] = None) -> VectorRecord:
        """Restore a VectorRecord from Qdrant payload fields."""
        metadata = dict(payload or {})
        content = metadata.pop("content", "")
        record_id = str(metadata.pop("record_id", metadata.get("id", point_id)))
        metadata["id"] = record_id

        return VectorRecord(
            id=record_id,
            content=content,
            embedding=list(embedding or []),
            metadata=metadata,
        )

    def _to_point_id(self, record_id: str) -> str:
        """Map a stable business id to a deterministic Qdrant-compatible UUID."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, str(record_id)))

    def _validate_record(self, record: VectorRecord) -> None:
        """Validate a record before insertion."""
        if not record.id:
            raise ValueError("record id is required")
        self._validate_vector(record.embedding)

    def _validate_vector(self, vector: Vector) -> None:
        """Validate vector dimension."""
        if len(vector) != self.dimension:
            raise ValueError(f"Expected vector dimension {self.dimension}, got {len(vector)}")

    def _extract_vector(self, item: Any) -> Vector:
        """Extract a dense vector from SDK point objects."""
        vector = getattr(item, "vector", None)
        if vector is None:
            vector = getattr(item, "vectors", None)
        if isinstance(vector, dict):
            vector = next(iter(vector.values()), [])
        return list(vector or [])

    def _extract_vector_size(self, collection_info: Any) -> Optional[int]:
        """Extract single-vector collection size from Qdrant collection metadata."""
        config = self._get_field(collection_info, "config")
        params = self._get_field(config, "params")
        vectors = self._get_field(params, "vectors")

        if isinstance(vectors, dict):
            first_vector = next(iter(vectors.values()), None)
            return self._get_field(first_vector, "size")

        return self._get_field(vectors, "size")

    def _get_field(self, value: Any, key: str) -> Any:
        """Read a field from either an object or dictionary."""
        if value is None:
            return None
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    def _normalize_score(self, raw_score: float) -> float:
        """Keep the VectorStore contract that higher scores are more relevant."""
        if self.distance in {"euclid", "euclidean"}:
            return -raw_score
        return raw_score

    def _call_with_retries(self, operation_name: str, operation: Callable[..., T], *args, **kwargs) -> T:
        """Run a Qdrant operation with exponential backoff retries."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            start_time = time.monotonic()
            try:
                result = operation(*args, **kwargs)
                elapsed = time.monotonic() - start_time
                logger.debug(
                    "Qdrant operation succeeded | operation=%s | attempt=%s | elapsed=%.3fs",
                    operation_name,
                    attempt + 1,
                    elapsed,
                )
                return result
            except Exception as e:
                last_error = e
                elapsed = time.monotonic() - start_time
                if attempt >= self.max_retries or not self._is_retryable_error(e):
                    logger.error(
                        "Qdrant operation failed | operation=%s | attempt=%s | elapsed=%.3fs | error=%s",
                        operation_name,
                        attempt + 1,
                        elapsed,
                        e,
                    )
                    raise QdrantVectorStoreError(
                        f"Qdrant operation failed: {operation_name}: {e}"
                    ) from e

                delay = self._retry_delay(attempt)
                logger.warning(
                    "Retrying Qdrant operation | operation=%s | attempt=%s | delay=%.2fs | error=%s",
                    operation_name,
                    attempt + 1,
                    delay,
                    e,
                )
                time.sleep(delay)

        raise QdrantVectorStoreError(f"Qdrant operation failed: {operation_name}: {last_error}")

    def _is_retryable_error(self, error: Exception) -> bool:
        """Return whether an operation error should be retried."""
        status_code = self._extract_status_code(error)
        if status_code is not None:
            return status_code in RETRYABLE_STATUS_CODES

        retryable_names = (
            "timeout",
            "timedout",
            "connection",
            "connecterror",
            "readerror",
            "network",
            "ratelimit",
            "too_many_requests",
            "temporarilyunavailable",
        )
        error_name = type(error).__name__.lower()
        return isinstance(error, (TimeoutError, ConnectionError, OSError)) or any(
            name in error_name for name in retryable_names
        )

    def _extract_status_code(self, error: Exception) -> Optional[int]:
        """Extract an HTTP status code from SDK or transport exceptions."""
        for attr_name in ("status_code", "status"):
            value = getattr(error, attr_name, None)
            if isinstance(value, int):
                return value

        response = getattr(error, "response", None)
        if response is not None:
            value = getattr(response, "status_code", None)
            if isinstance(value, int):
                return value

        return None

    def _retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with small jitter."""
        delay = min(self.max_delay, self.base_delay * (2**attempt))
        jitter = random.uniform(0.0, min(0.25, delay * 0.1))
        return delay + jitter

    def _iter_batches(self, items: Sequence[Any], batch_size: int) -> Iterable[List[Any]]:
        """Yield fixed-size batches."""
        for start in range(0, len(items), batch_size):
            yield list(items[start : start + batch_size])
