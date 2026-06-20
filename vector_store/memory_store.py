from typing import Dict, Iterable, List, Optional

from access import metadata_matches
from embeddings import cosine_similarity
from embeddings.base import Vector
from logger import setup_logger

from .base import SearchResult, VectorRecord, VectorStore

logger = setup_logger(__name__)


class InMemoryVectorStore(VectorStore):
    """In-memory vector store for local development and tests."""

    def __init__(self, dimension: Optional[int] = None):
        """Initialize the in-memory vector store.

        Args:
            dimension: Optional fixed vector dimension.
        """
        self.dimension = dimension
        self._records: Dict[str, VectorRecord] = {}

    def add_records(self, records: Iterable[VectorRecord]) -> List[str]:
        """Add vector records and return their ids."""
        ids = []

        for record in records:
            self._validate_record(record)
            self._records[record.id] = record
            ids.append(record.id)

        logger.info("Vector records added | count=%s | total=%s", len(ids), self.count())
        return ids

    def similarity_search(
        self,
        query_embedding: Vector,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Search records by cosine similarity."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        self._validate_vector(query_embedding)

        results = []
        for record in self._records.values():
            if metadata_filter and not self._metadata_matches(record.metadata, metadata_filter):
                continue

            score = cosine_similarity(query_embedding, record.embedding)
            results.append(SearchResult(record=record, score=score))

        results.sort(key=lambda result: result.score, reverse=True)
        return results[:top_k]

    def get_record(self, record_id: str) -> Optional[VectorRecord]:
        """Return a record by id."""
        return self._records.get(record_id)

    def list_records(self, limit: Optional[int] = None) -> List[VectorRecord]:
        """Return stored records in insertion order."""
        records = list(self._records.values())
        if limit is None:
            return records
        if limit < 0:
            raise ValueError("limit must be non-negative")
        return records[:limit]

    def delete(self, record_id: str) -> bool:
        """Delete a record by id."""
        if record_id not in self._records:
            return False

        del self._records[record_id]
        return True

    def clear(self) -> None:
        """Delete all records."""
        self._records.clear()

    def count(self) -> int:
        """Return the number of stored records."""
        return len(self._records)

    def _validate_record(self, record: VectorRecord) -> None:
        """Validate a record before insertion."""
        if not record.id:
            raise ValueError("record id is required")
        self._validate_vector(record.embedding)

    def _validate_vector(self, vector: Vector) -> None:
        """Validate vector dimension."""
        if self.dimension is None:
            self.dimension = len(vector)

        if len(vector) != self.dimension:
            raise ValueError(f"Expected vector dimension {self.dimension}, got {len(vector)}")

    def _metadata_matches(self, metadata: Dict, metadata_filter: Dict) -> bool:
        """Return whether metadata matches exact scalar and list-intersection filters."""
        return metadata_matches(metadata, metadata_filter)
