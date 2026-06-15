import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Set

from .base import RerankConfig, RerankResult, Reranker

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


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


def tokenize(text: str) -> List[str]:
    """Tokenize and lightly normalize English words, numbers, and CJK chars."""
    tokens = []
    for raw_token in TOKEN_PATTERN.findall(text.lower()):
        token = normalize_token(raw_token)
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def normalize_token(token: str) -> str:
    """Apply deterministic light stemming."""
    if len(token) <= 3:
        return token

    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            token = token[: -len(suffix)]
            break

    if token.endswith("e") and len(token) > 4:
        token = token[:-1]

    return token


def build_idf(candidate_tokens: Sequence[List[str]]) -> Dict[str, float]:
    """Build candidate-local IDF values."""
    document_count = max(len(candidate_tokens), 1)
    document_frequency = Counter()
    for tokens in candidate_tokens:
        document_frequency.update(set(tokens))

    return {
        token: math.log((document_count + 1) / (frequency + 1)) + 1.0
        for token, frequency in document_frequency.items()
    }


def lexical_score(
    query_counts: Counter,
    query_set: Set[str],
    candidate_tokens: List[str],
    idf: Dict[str, float],
    dense_rank: int,
) -> float:
    """Score one candidate with deterministic lexical features."""
    if not candidate_tokens:
        return 0.0

    candidate_counts = Counter(candidate_tokens)
    candidate_set = set(candidate_tokens)
    overlap = query_set & candidate_set

    weighted_overlap = 0.0
    for token in overlap:
        tf = 1.0 + math.log(candidate_counts[token])
        weighted_overlap += query_counts[token] * tf * idf.get(token, 1.0)

    coverage = len(overlap) / max(len(query_set), 1)
    phrase_bonus = contiguous_overlap_bonus(list(query_counts.keys()), candidate_tokens)
    dense_tiebreaker = 1.0 / (1000.0 + dense_rank)
    return weighted_overlap + (2.0 * coverage) + phrase_bonus + dense_tiebreaker


def contiguous_overlap_bonus(query_tokens: List[str], candidate_tokens: List[str]) -> float:
    """Return a small bonus for adjacent query terms appearing in candidate text."""
    if len(query_tokens) < 2 or len(candidate_tokens) < 2:
        return 0.0

    query_bigrams = set(zip(query_tokens, query_tokens[1:]))
    candidate_bigrams = set(zip(candidate_tokens, candidate_tokens[1:]))
    return 0.25 * len(query_bigrams & candidate_bigrams)

