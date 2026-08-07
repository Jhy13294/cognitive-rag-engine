import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from embeddings import EmbeddedDocument
from embeddings.base import Vector


@dataclass
class VectorRecord:
    """Stored vector record with content and metadata."""

    id: str
    content: str
    embedding: Vector
    metadata: Dict = field(default_factory=dict)


@dataclass
class SearchResult:
    """Similarity search result."""

    record: VectorRecord
    score: float

    @property
    def content(self) -> str:
        """Return matched content."""
        return self.record.content

    @property
    def metadata(self) -> Dict:
        """Return matched metadata."""
        return self.record.metadata


class VectorStore(ABC):
    """Base interface for vector stores."""

    @abstractmethod
    def add_records(self, records: Iterable[VectorRecord]) -> List[str]:
        """Add vector records and return their ids."""
        pass

    def add_documents(self, documents: Iterable[EmbeddedDocument]) -> List[str]:
        """Convert embedded documents to records and add them."""
        records = [embedded_document_to_record(document) for document in documents]
        return self.add_records(records)

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: Vector,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Search for records similar to the query embedding."""
        pass

    @abstractmethod
    def get_record(self, record_id: str) -> Optional[VectorRecord]:
        """Return a record by id."""
        pass

    @abstractmethod
    def list_records(self, limit: Optional[int] = None) -> List[VectorRecord]:
        """Return stored records for deterministic local index reconstruction."""
        pass

    @abstractmethod
    def delete(self, record_id: str) -> bool:
        """Delete a record by id."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Delete all records."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of stored records."""
        pass


def embedded_document_to_record(document: EmbeddedDocument) -> VectorRecord:
    """Convert an EmbeddedDocument into a VectorRecord."""
    metadata = dict(document.metadata)
    record_id = metadata.get("id") or metadata.get("document_id")

    if not record_id:
        record_id = build_record_id(document.content, metadata)

    metadata["id"] = record_id

    return VectorRecord(
        id=str(record_id),
        content=document.content,
        embedding=list(document.embedding),
        metadata=metadata,
    )


def build_record_id(content: str, metadata: Dict) -> str:
    """Build a deterministic record id from content and stable metadata."""
    stable_metadata = {
        key: metadata.get(key)
        for key in [
            "source",
            "chunk_index",
            "start_char",
            "end_char",
            "embedding_model",
            "embedding_dimension",
        ]
        if key in metadata
    }
    if "source" in stable_metadata and stable_metadata["source"] is not None:
        stable_metadata["source"] = str(stable_metadata["source"]).replace("\\", "/")
    payload = json.dumps(
        {
            "content": content,
            "metadata": stable_metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
