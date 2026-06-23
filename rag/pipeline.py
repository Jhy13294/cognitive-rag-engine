from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Tuple

from embeddings import EmbeddingProvider
from hybrid import BM25Retriever, RRFConfig, RankedRecord, ReciprocalRankFusion
from logger import setup_logger
from parent_store import ParentStore
from query_rewrite import QueryRewriter, normalize_query_variants
from rerank import Reranker
from tokenization import TokenCounter, create_token_counter
from vector_store import SearchResult, VectorStore

from .context_packing import ContextPacker

logger = setup_logger(__name__)


class ChatClient(Protocol):
    """Protocol for chat-completion clients used by the RAG pipeline."""

    def chat(self, message: str, system_prompt: Optional[str] = None) -> Dict:
        """Send a chat request and return a provider response."""
        ...


@dataclass
class RetrievedSource:
    """Source chunk used to answer a query."""

    index: int
    content: str
    score: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Answer returned by the RAG pipeline."""

    question: str
    answer: str
    sources: List[RetrievedSource]
    prompt: str
    raw_response: Dict


CANONICAL_ABSTENTION_RESPONSE = "The answer is not available in the knowledge base."


class RAGPipeline:
    """Minimal retrieval-augmented generation pipeline."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an enterprise knowledge-base assistant. "
        "Answer only with claims directly supported by the provided context. "
        "Do not infer a policy, procedure, value, or current fact from merely related context. "
        "If the context does not directly support an answer, begin with exactly this sentence: "
        f'"{CANONICAL_ABSTENTION_RESPONSE}" '
        "You may briefly explain what information is missing, but do not supply an unsupported answer. "
        "Cite sources with bracketed numbers like [1], [2]."
    )

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chat_client: ChatClient,
        top_k: int = 5,
        max_context_chars: int = 4000,
        system_prompt: Optional[str] = None,
        reranker: Optional[Reranker] = None,
        fetch_k: Optional[int] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        rrf: Optional[ReciprocalRankFusion] = None,
        parent_store: Optional[ParentStore] = None,
        expand_parent_context: bool = True,
        context_packing_enabled: bool = False,
        context_dedup_enabled: bool = False,
        context_near_dup_enabled: bool = False,
        context_near_dup_threshold: float = 0.9,
        context_max_tokens: Optional[int] = None,
        tokenizer_encoding: str = "cl100k_base",
        token_counter: Optional[TokenCounter] = None,
        query_rewriter: Optional[QueryRewriter] = None,
        query_rewrite_enabled: bool = False,
        query_rewrite_num_queries: int = 3,
        query_rewrite_weight_original: float = 1.0,
        query_rewrite_weight_variant: float = 0.7,
    ):
        """Initialize the RAG pipeline."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than 0")
        if fetch_k is not None and fetch_k <= 0:
            raise ValueError("fetch_k must be greater than 0")
        if query_rewrite_num_queries < 1:
            raise ValueError("query_rewrite_num_queries must be greater than or equal to 1")
        if query_rewrite_enabled and query_rewriter is None:
            raise ValueError("query_rewriter is required when query_rewrite_enabled is true")
        if query_rewrite_weight_original < 0 or query_rewrite_weight_variant < 0:
            raise ValueError("query rewrite weights must be non-negative")
        if query_rewrite_weight_original < query_rewrite_weight_variant:
            raise ValueError("query_rewrite_weight_original must be greater than or equal to query_rewrite_weight_variant")

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chat_client = chat_client
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.bm25_retriever = bm25_retriever
        self.rrf = rrf or ReciprocalRankFusion()
        self.parent_store = parent_store
        self.expand_parent_context = expand_parent_context
        self.context_packing_enabled = context_packing_enabled
        self.context_dedup_enabled = context_dedup_enabled
        self.context_near_dup_enabled = context_near_dup_enabled
        self.context_near_dup_threshold = context_near_dup_threshold
        self.context_max_tokens = context_max_tokens
        self.tokenizer_encoding = tokenizer_encoding
        self.token_counter = token_counter
        self.context_packer = None
        self.query_rewriter = query_rewriter
        self.query_rewrite_enabled = query_rewrite_enabled
        self.query_rewrite_num_queries = query_rewrite_num_queries
        self.query_rewrite_weight_original = query_rewrite_weight_original
        self.query_rewrite_weight_variant = query_rewrite_weight_variant

        if self.context_packing_enabled:
            self.token_counter = token_counter or create_token_counter(tokenizer_encoding)
            self.context_packer = ContextPacker(
                max_context_chars=self.max_context_chars,
                max_context_tokens=self.context_max_tokens,
                token_counter=self.token_counter if self.context_max_tokens is not None else None,
                label_formatter=self._format_source_label,
                dedup_enabled=self.context_dedup_enabled,
                near_dup_enabled=self.context_near_dup_enabled,
                near_dup_threshold=self.context_near_dup_threshold,
            )

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedSource]:
        """Retrieve source chunks for a question."""
        requested_top_k = top_k or self.top_k
        if self.query_rewrite_enabled:
            multi_query_sources = self._try_multi_query_retrieve(
                question=question,
                requested_top_k=requested_top_k,
                metadata_filter=metadata_filter,
            )
            if multi_query_sources is not None:
                return multi_query_sources

        return self._retrieve_single_query(
            question=question,
            requested_top_k=requested_top_k,
            metadata_filter=metadata_filter,
        )

    def _retrieve_single_query(
        self,
        question: str,
        requested_top_k: int,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedSource]:
        """Run the pre-T09 single-query retrieval path."""
        sources = self._retrieve_single_query_candidates(
            question=question,
            requested_top_k=requested_top_k,
            metadata_filter=metadata_filter,
        )
        fallback_label = "hybrid" if self.bm25_retriever is not None else "dense"
        return self._rerank_or_finalize(
            question=question,
            sources=sources,
            requested_top_k=requested_top_k,
            fallback_label=fallback_label,
        )

    def _retrieve_single_query_candidates(
        self,
        question: str,
        requested_top_k: int,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedSource]:
        """Return dense or hybrid candidates before rerank and parent expansion."""
        query_embedding = self.embedding_provider.embed_text(question)

        if self.bm25_retriever is not None:
            return self._hybrid_retrieve(
                question=question,
                query_embedding=query_embedding,
                requested_top_k=requested_top_k,
                metadata_filter=metadata_filter,
            )

        search_results = self.vector_store.similarity_search(
            query_embedding,
            top_k=self._search_top_k(requested_top_k),
            metadata_filter=metadata_filter,
        )
        return self._to_sources(search_results)

    def _rerank_or_finalize(
        self,
        question: str,
        sources: List[RetrievedSource],
        requested_top_k: int,
        fallback_label: str,
    ) -> List[RetrievedSource]:
        """Apply rerank when configured, then parent expansion."""
        if self.reranker is None:
            return self._finalize_sources(sources[:requested_top_k])

        try:
            return self._finalize_sources(self._rerank_sources(question, sources, requested_top_k))
        except Exception as e:
            logger.warning(
                "Rerank failed; falling back to %s order | error=%s | requested_top_k=%s",
                fallback_label,
                e,
                requested_top_k,
            )
            return self._finalize_sources(sources[:requested_top_k])

    def _try_multi_query_retrieve(
        self,
        question: str,
        requested_top_k: int,
        metadata_filter: Optional[Dict] = None,
    ) -> Optional[List[RetrievedSource]]:
        """Run T09 multi-query retrieval, returning None when it should be bypassed."""
        if self.query_rewriter is None:
            return None

        try:
            raw_variants = self.query_rewriter.rewrite(question)
        except Exception as e:
            logger.warning("Query rewrite failed; falling back to single query | error=%s", e)
            return None

        variants = normalize_query_variants(question, raw_variants, self.query_rewrite_num_queries)
        if not variants:
            logger.warning("Query rewrite returned no variants; falling back to single query | question=%s", question)
            return None
        if len(variants) <= 1:
            return None

        logger.info("Multi-query retrieval enabled | original=%s | variants=%s", question, variants)
        candidate_count = self._multi_query_candidate_count(requested_top_k, len(variants))
        ranked_lists = {}

        for path_name, variant in self._query_variant_paths(variants):
            candidates = self._retrieve_single_query_candidates(
                question=variant,
                requested_top_k=candidate_count,
                metadata_filter=metadata_filter,
            )
            ranked_lists[path_name] = retrieved_sources_to_ranked_records(candidates)

        weights = self._query_variant_weights(len(variants))
        fusion = ReciprocalRankFusion(
            RRFConfig(
                k=self.rrf.config.k,
                weights=weights,
            )
        )
        fusion_top_k = candidate_count if self.reranker is not None else requested_top_k
        fused_records = fusion.fuse(ranked_lists, top_k=fusion_top_k)
        fused_sources = self._ranked_records_to_sources(fused_records)

        for source in fused_sources:
            source.metadata["query_rewrite_enabled"] = True
            source.metadata["query_rewrite_variants"] = list(variants)
            source.metadata["query_rewrite_weights"] = dict(weights)

        return self._rerank_or_finalize(
            question=question,
            sources=fused_sources,
            requested_top_k=requested_top_k,
            fallback_label="multi-query fused",
        )

    def _multi_query_candidate_count(self, requested_top_k: int, variant_count: int) -> int:
        """Return per-variant candidate count for multi-query retrieval."""
        base_count = self._search_top_k(requested_top_k)
        if self.reranker is None:
            return max(base_count, requested_top_k)
        return max(base_count, requested_top_k * variant_count)

    def _query_variant_paths(self, variants: List[str]) -> List[Tuple[str, str]]:
        """Return explicit path names for query variants."""
        return [(f"q{index}", variant) for index, variant in enumerate(variants)]

    def _query_variant_weights(self, variant_count: int) -> Dict[str, float]:
        """Return explicit RRF weights for every query variant path."""
        weights = {"q0": self.query_rewrite_weight_original}
        for index in range(1, variant_count):
            weights[f"q{index}"] = self.query_rewrite_weight_variant
        return weights

    def build_prompt(self, question: str, sources: List[RetrievedSource]) -> str:
        """Build the user prompt sent to the chat client."""
        prompt, _ = self._build_prompt_and_sources(question, sources)
        return prompt

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> RAGResponse:
        """Retrieve context and generate an answer."""
        sources = self.retrieve(question, top_k=top_k, metadata_filter=metadata_filter)
        prompt, used_sources = self._build_prompt_and_sources(question, sources)
        raw_response = self.chat_client.chat(prompt, system_prompt=self.system_prompt)
        answer = extract_chat_content(raw_response)

        logger.info("RAG answer generated | sources=%s | answer_chars=%s", len(used_sources), len(answer))
        return RAGResponse(
            question=question,
            answer=answer,
            sources=used_sources,
            prompt=prompt,
            raw_response=raw_response,
        )

    def _build_prompt_and_sources(
        self,
        question: str,
        sources: List[RetrievedSource],
    ) -> Tuple[str, List[RetrievedSource]]:
        """Build a prompt and return the sources actually used in context."""
        if self.context_packing_enabled:
            packed_context = self.context_packer.pack(sources)
            context = packed_context.context
            used_sources = packed_context.used_sources
        else:
            context = self._build_context(sources)
            used_sources = sources

        prompt = (
            "Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer with source citations."
        )
        return prompt, used_sources

    def _to_sources(self, search_results: List[SearchResult]) -> List[RetrievedSource]:
        """Convert vector search results to RAG sources."""
        sources = []
        for index, result in enumerate(search_results, start=1):
            sources.append(
                RetrievedSource(
                    index=index,
                    content=result.content,
                    score=result.score,
                    metadata=dict(result.metadata),
                )
            )
        return sources

    def _search_top_k(self, requested_top_k: int) -> int:
        """Return dense candidate count for retrieval."""
        if self.reranker is None and self.bm25_retriever is None:
            return requested_top_k
        default_fetch_k = self.reranker.fetch_k if self.reranker is not None else requested_top_k
        return max(self.fetch_k or default_fetch_k, requested_top_k)

    def _hybrid_retrieve(
        self,
        question: str,
        query_embedding: List[float],
        requested_top_k: int,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedSource]:
        """Retrieve with dense and BM25 paths, then fuse by RRF."""
        candidate_count = self._search_top_k(requested_top_k)
        dense_results = self.vector_store.similarity_search(
            query_embedding,
            top_k=candidate_count,
            metadata_filter=metadata_filter,
        )
        sparse_results = self.bm25_retriever.retrieve(
            question,
            top_k=candidate_count,
            metadata_filter=metadata_filter,
        )
        fused_records = self.rrf.fuse(
            {
                "dense": dense_search_results_to_ranked_records(dense_results),
                "sparse": sparse_results,
            },
            top_k=candidate_count,
        )
        return self._ranked_records_to_sources(fused_records)

    def _rerank_sources(
        self,
        question: str,
        sources: List[RetrievedSource],
        requested_top_k: int,
    ) -> List[RetrievedSource]:
        """Rerank dense sources and preserve dense retrieval observability."""
        reranked = self.reranker.rerank(question, sources, top_n=requested_top_k)
        if not reranked:
            return sources[:requested_top_k]

        reranked_sources = []
        for new_index, result in enumerate(reranked[:requested_top_k], start=1):
            if result.index < 0 or result.index >= len(sources):
                continue

            dense_source = sources[result.index]
            metadata = dict(dense_source.metadata)
            metadata.update(result.metadata)
            metadata.setdefault("dense_score", dense_source.score)
            metadata.setdefault("dense_rank", dense_source.index)
            metadata["rerank_model"] = self.reranker.model_name

            reranked_sources.append(
                RetrievedSource(
                    index=new_index,
                    content=result.content or dense_source.content,
                    score=float(result.score),
                    metadata=metadata,
                )
            )

        if not reranked_sources:
            return sources[:requested_top_k]
        return reranked_sources

    def _ranked_records_to_sources(self, ranked_records: List[RankedRecord]) -> List[RetrievedSource]:
        """Convert fused ranked records to RAG sources."""
        sources = []
        for index, record in enumerate(ranked_records, start=1):
            sources.append(
                RetrievedSource(
                    index=index,
                    content=record.content,
                    score=record.score,
                    metadata=dict(record.metadata),
                )
            )
        return sources

    def _finalize_sources(self, sources: List[RetrievedSource]) -> List[RetrievedSource]:
        """Apply post-retrieval source transformations."""
        if self.parent_store is None or not self.expand_parent_context:
            return sources
        return self._expand_parent_sources(sources)

    def _expand_parent_sources(self, sources: List[RetrievedSource]) -> List[RetrievedSource]:
        """Replace child chunk content with parent content after retrieval."""
        expanded_sources = []
        parent_positions = {}

        for source in sources:
            parent_id = source.metadata.get("parent_id")
            if not parent_id:
                expanded_sources.append(source)
                continue

            parent = self.parent_store.get_parent(str(parent_id))
            if parent is None:
                logger.warning(
                    "Parent chunk not found; keeping child source | parent_id=%s | child_index=%s",
                    parent_id,
                    source.index,
                )
                metadata = dict(source.metadata)
                metadata["parent_lookup_failed"] = True
                expanded_sources.append(
                    RetrievedSource(
                        index=source.index,
                        content=source.content,
                        score=source.score,
                        metadata=metadata,
                    )
                )
                continue

            if parent.id in parent_positions:
                existing = expanded_sources[parent_positions[parent.id]]
                existing.metadata["collapsed_child_count"] += 1
                existing.metadata.setdefault("collapsed_child_ids", []).append(child_id_for(source))
                continue

            metadata = dict(source.metadata)
            metadata.update(
                {
                    "child_id": child_id_for(source),
                    "child_score": source.score,
                    "child_start_char": source.metadata.get("start_char"),
                    "child_end_char": source.metadata.get("end_char"),
                    "parent_id": parent.id,
                    "parent_index": parent.metadata.get("parent_index", source.metadata.get("parent_index")),
                    "parent_start_char": parent.metadata.get("start_char"),
                    "parent_end_char": parent.metadata.get("end_char"),
                    "parent_expanded": True,
                    "collapsed_child_count": 1,
                    "collapsed_child_ids": [child_id_for(source)],
                }
            )
            expanded_source = RetrievedSource(
                index=source.index,
                content=parent.content,
                score=source.score,
                metadata=metadata,
            )
            parent_positions[parent.id] = len(expanded_sources)
            expanded_sources.append(expanded_source)

        return expanded_sources

    def _build_context(self, sources: List[RetrievedSource]) -> str:
        """Build a bounded context block from retrieved sources."""
        if self.context_packing_enabled:
            return self.context_packer.pack(sources).context

        context_blocks = []
        used_chars = 0

        for source in sources:
            source_label = self._format_source_label(source)
            block = f"[{source.index}] {source_label}\n{source.content}".strip()

            if used_chars + len(block) > self.max_context_chars:
                remaining = self.max_context_chars - used_chars
                if remaining <= 0:
                    break
                block = block[:remaining].rstrip()

            context_blocks.append(block)
            used_chars += len(block)

            if used_chars >= self.max_context_chars:
                break

        if not context_blocks:
            return "No relevant context was retrieved."

        return "\n\n".join(context_blocks)

    def _format_source_label(self, source: RetrievedSource) -> str:
        """Format human-readable source metadata."""
        metadata = source.metadata
        source_path = metadata.get("source", "unknown source")
        chunk_index = metadata.get("chunk_index")

        if chunk_index is None:
            return f"source={source_path}; score={source.score:.4f}"

        return f"source={source_path}; chunk={chunk_index}; score={source.score:.4f}"


def extract_chat_content(response: Dict) -> str:
    """Extract assistant text from a chat-completion compatible response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError("Invalid chat response format") from e


def dense_search_results_to_ranked_records(search_results: List[SearchResult]) -> List[RankedRecord]:
    """Convert dense vector search results to stable ranked records."""
    ranked_records = []
    for result in search_results:
        metadata = dict(result.metadata)
        metadata["id"] = result.record.id
        metadata["retrieval_mode"] = "dense"
        metadata["dense_score"] = result.score
        ranked_records.append(
            RankedRecord(
                id=result.record.id,
                score=result.score,
                content=result.content,
                metadata=metadata,
                record=result.record,
            )
        )
    return ranked_records


def retrieved_sources_to_ranked_records(sources: List[RetrievedSource]) -> List[RankedRecord]:
    """Convert retrieved sources to stable ranked records for cross-query RRF."""
    ranked_records = []
    for source in sources:
        metadata = dict(source.metadata)
        record_id = str(metadata.get("id") or metadata.get("child_id") or "")
        if not record_id:
            continue
        ranked_records.append(
            RankedRecord(
                id=record_id,
                score=source.score,
                content=source.content,
                metadata=metadata,
            )
        )
    return ranked_records


def child_id_for(source: RetrievedSource) -> str:
    """Return the stable child id for a retrieved source."""
    metadata = source.metadata or {}
    return str(metadata.get("child_id") or metadata.get("id") or "")
