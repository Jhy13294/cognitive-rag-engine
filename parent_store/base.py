from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from document_loader import Document


@dataclass
class ParentRecord:
    """Stored parent chunk used for generation context expansion."""

    id: str
    content: str
    metadata: Dict = field(default_factory=dict)


class ParentStore(ABC):
    """Key-value store interface for parent chunks."""

    @abstractmethod
    def add_parents(self, parents: Iterable[Document]) -> List[str]:
        """Add parent documents and return their ids."""
        pass

    @abstractmethod
    def get_parent(self, parent_id: str) -> Optional[ParentRecord]:
        """Return a parent record by id."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of parent records."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all parent records."""
        pass


def document_to_parent_record(document: Document) -> ParentRecord:
    """Convert a parent Document into a ParentRecord."""
    metadata = dict(document.metadata)
    parent_id = metadata.get("parent_id")
    if not parent_id:
        raise ValueError("parent document metadata must include parent_id")

    metadata["parent_id"] = str(parent_id)
    return ParentRecord(
        id=str(parent_id),
        content=document.content,
        metadata=metadata,
    )
