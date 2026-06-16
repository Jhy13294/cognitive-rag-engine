from typing import Dict, Iterable, List, Optional, Set

from lexical import bm25_score, build_bm25_index, tokenize
from vector_store import VectorRecord

from .models import RankedRecord


class BM25Retriever:
    """Sparse retriever over the same VectorRecord id space as dense retrieval."""

    def __init__(
        self,
        records: Iterable[VectorRecord],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        """Build a BM25 index from vector records."""
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if b < 0 or b > 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self.records_by_id: Dict[str, VectorRecord] = {}
        tokenized_documents = {}

        for record in records:
            if not record.id:
                raise ValueError("VectorRecord.id is required for BM25 retrieval")
            if record.id in self.records_by_id:
                raise ValueError(f"Duplicate VectorRecord.id in BM25 corpus: {record.id}")

            self.records_by_id[record.id] = record
            tokenized_documents[record.id] = tokenize(record.content)

        self.index = build_bm25_index(tokenized_documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RankedRecord]:
        """Return BM25-ranked records for a query."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        candidate_ids = self._candidate_ids(query_tokens)
        results = []
        for record_id in candidate_ids:
            record = self.records_by_id[record_id]
            if metadata_filter and not metadata_matches(record.metadata, metadata_filter):
                continue

            score = bm25_score(
                query_tokens=query_tokens,
                index=self.index,
                document_id=record_id,
                k1=self.k1,
                b=self.b,
            )
            if score <= 0:
                continue

            metadata = dict(record.metadata)
            metadata["id"] = record.id
            metadata["retrieval_mode"] = "bm25"
            metadata["bm25_score"] = score
            results.append(
                RankedRecord(
                    id=record.id,
                    score=score,
                    content=record.content,
                    metadata=metadata,
                    record=record,
                )
            )

        results.sort(key=lambda result: (-result.score, result.id))
        return results[:top_k]

    def _candidate_ids(self, query_tokens: List[str]) -> Set[str]:
        """Return records containing at least one query token."""
        candidate_ids: Set[str] = set()
        for token in set(query_tokens):
            candidate_ids.update(self.index.postings.get(token, set()))
        return candidate_ids


def metadata_matches(metadata: Dict, metadata_filter: Dict) -> bool:
    """Return whether metadata matches an exact-match filter."""
    return all(metadata.get(key) == value for key, value in metadata_filter.items())
