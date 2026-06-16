import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Counter as CounterType, Dict, Iterable, List, Mapping, Sequence, Set


@dataclass
class BM25Index:
    """Token-level BM25 index primitives keyed by stable document id."""

    document_tokens: Dict[str, List[str]]
    term_frequencies: Dict[str, CounterType[str]]
    document_frequencies: Dict[str, int]
    postings: Dict[str, Set[str]]
    document_lengths: Dict[str, int]
    average_document_length: float
    document_count: int

    @property
    def ids(self) -> List[str]:
        """Return indexed document ids in deterministic order."""
        return sorted(self.document_tokens)


def build_bm25_index(tokenized_documents: Mapping[str, Sequence[str]]) -> BM25Index:
    """Build BM25 statistics from tokenized documents keyed by record id."""
    document_tokens = {doc_id: list(tokens) for doc_id, tokens in tokenized_documents.items()}
    term_frequencies = {doc_id: Counter(tokens) for doc_id, tokens in document_tokens.items()}
    document_frequencies: Dict[str, int] = defaultdict(int)
    postings: Dict[str, Set[str]] = defaultdict(set)
    document_lengths = {doc_id: len(tokens) for doc_id, tokens in document_tokens.items()}

    for doc_id, tokens in document_tokens.items():
        for token in set(tokens):
            document_frequencies[token] += 1
            postings[token].add(doc_id)

    document_count = len(document_tokens)
    average_document_length = (
        sum(document_lengths.values()) / document_count if document_count else 0.0
    )

    return BM25Index(
        document_tokens=document_tokens,
        term_frequencies=term_frequencies,
        document_frequencies=dict(document_frequencies),
        postings={token: set(ids) for token, ids in postings.items()},
        document_lengths=document_lengths,
        average_document_length=average_document_length,
        document_count=document_count,
    )


def build_idf(candidate_tokens: Sequence[List[str]]) -> Dict[str, float]:
    """Build candidate-local IDF values for deterministic lexical scoring."""
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


def bm25_score(
    query_tokens: Iterable[str],
    index: BM25Index,
    document_id: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    """Compute BM25 score for one indexed document."""
    if index.document_count == 0 or document_id not in index.term_frequencies:
        return 0.0
    if k1 <= 0:
        raise ValueError("k1 must be greater than 0")
    if b < 0 or b > 1:
        raise ValueError("b must be between 0 and 1")

    score = 0.0
    term_frequency = index.term_frequencies[document_id]
    document_length = index.document_lengths.get(document_id, 0)
    average_length = index.average_document_length or 1.0

    for token in Counter(query_tokens):
        frequency = term_frequency.get(token, 0)
        if frequency <= 0:
            continue
        document_frequency = index.document_frequencies.get(token, 0)
        idf = math.log(1.0 + (index.document_count - document_frequency + 0.5) / (document_frequency + 0.5))
        denominator = frequency + k1 * (1.0 - b + b * (document_length / average_length))
        score += idf * ((frequency * (k1 + 1.0)) / denominator)

    return score
