from typing import Dict, Iterable, List, Optional

from document_loader import Document
from logger import setup_logger

from .base import ParentRecord, ParentStore, document_to_parent_record

logger = setup_logger(__name__)


class InMemoryParentStore(ParentStore):
    """Deterministic in-memory key-value store for parent chunks."""

    def __init__(self):
        """Initialize an empty parent store."""
        self._records: Dict[str, ParentRecord] = {}

    def add_parents(self, parents: Iterable[Document]) -> List[str]:
        """Add parent documents and return their ids."""
        ids = []
        for parent in parents:
            record = document_to_parent_record(parent)
            self._records[record.id] = record
            ids.append(record.id)

        logger.info("Parent records added | count=%s | total=%s", len(ids), self.count())
        return ids

    def get_parent(self, parent_id: str) -> Optional[ParentRecord]:
        """Return a parent record by id."""
        return self._records.get(str(parent_id))

    def count(self) -> int:
        """Return the number of parent records."""
        return len(self._records)

    def clear(self) -> None:
        """Delete all parent records."""
        self._records.clear()
