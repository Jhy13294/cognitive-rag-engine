from collections import Counter
from typing import TYPE_CHECKING, List, Optional, Sequence

from lexical import build_idf, lexical_score, tokenize

from .base import RerankConfig, RerankResult, Reranker

if TYPE_CHECKING:
    from rag import RetrievedSource


class DeterministicReranker(Reranker):
    """Deterministic lexical reranker for offline CI gates.

    The scorer reads query and candidate text only. It uses token overlap,
    candidate-local IDF, and a small exact-phrase bonus to produce stable
    relevance scores without network calls or model downloads.
    """

    def __init__(
        self,
        model_name: str = "deterministic-lexical-reranker",
        top_n: int = 5,
        fetch_k: int = 30,
        batch_size: int = 32,
    ):
        """Initialize the deterministic lexical reranker."""
        super().__init__(
            RerankConfig(
                model_name=model_name,
                top_n=top_n,
                fetch_k=fetch_k,
                batch_size=batch_size,
                metadata={"provider": "deterministic"},
            )
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence["RetrievedSource"],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank candidates by deterministic lexical relevance."""
        limit = min(top_n or self.top_n, len(candidates))
        if limit <= 0:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return self._dense_order(candidates, limit)

        candidate_tokens = [tokenize(candidate.content) for candidate in candidates]
        idf = build_idf(candidate_tokens)
        query_counts = Counter(query_tokens)
        query_set = set(query_tokens)

        scored_results = []
        for index, candidate in enumerate(candidates):
            tokens = candidate_tokens[index]
            score = lexical_score(
                query_counts=query_counts,
                query_set=query_set,
                candidate_tokens=tokens,
                idf=idf,
                dense_rank=index + 1,
            )
            scored_results.append(
                RerankResult(
                    index=index,
                    score=score,
                    content=candidate.content,
                    metadata={
                        "rerank_provider": "deterministic",
                        "dense_score": candidate.score,
                        "dense_rank": index + 1,
                    },
                )
            )

        scored_results.sort(key=lambda result: (-result.score, result.index))
        return scored_results[:limit]

    def _dense_order(self, candidates: Sequence["RetrievedSource"], limit: int) -> List[RerankResult]:
        """Return candidates in original dense order when the query has no tokens."""
        return [
            RerankResult(
                index=index,
                score=float(candidate.score),
                content=candidate.content,
                metadata={
                    "rerank_provider": "deterministic",
                    "dense_score": candidate.score,
                    "dense_rank": index + 1,
                },
            )
            for index, candidate in enumerate(candidates[:limit])
        ]
