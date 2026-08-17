"""High-level RAG pipeline orchestration.

``RAGPipeline`` owns configuration and wires the pipeline stages together:

    candidates = retrieval_orchestrator.retrieve(question, ...)
    sources = source_finalizer.finalize(candidates)
    prompt, used_sources = prompt_builder.build(question, sources)
    answer = chat_client.chat(prompt, system_prompt=...)

Stage logic lives in the collaborator modules — ``retrieval_orchestrator``
(dense/hybrid/multi-query retrieval and rerank), ``source_finalizer``
(parent expansion), ``prompt_builder`` (system prompt, scope instructions,
context assembly), and ``question_classifier`` (deterministic question-shape
heuristics). Their public names are re-exported here so existing imports
from ``rag.pipeline`` keep working.
"""

from typing import Dict, List, Optional, Tuple

from embeddings import EmbeddingProvider
from hybrid import BM25Retriever, ReciprocalRankFusion
from logger import setup_logger
from parent_store import ParentStore
from query_rewrite import QueryRewriter
from rerank import Reranker
from tokenization import TokenCounter, create_token_counter
from vector_store import VectorStore

from .context_packing import ContextPacker
from .models import ChatClient, RAGResponse, RetrievedSource, extract_chat_content
from .prompt_builder import (
    CANONICAL_ABSTENTION_RESPONSE,
    RAG_SYSTEM_PROMPT_VERSION,
    PromptBuilder,
)
from .prompt_builder import (
    DEFAULT_SYSTEM_PROMPT as _DEFAULT_SYSTEM_PROMPT,
)
from .question_classifier import (
    ACTION_LIST_MARKERS,
    NAME_ONLY_LOOKUP_MARKERS,
    NAME_ONLY_LOOKUP_TARGETS,
    RELATIONSHIP_ENTITY_MARKERS,
    RELATIONSHIP_ENTITY_TARGETS,
    SCENARIO_PROCEDURE_KEYWORDS,
    SCENARIO_SITUATION_MARKERS,
    SOURCE_RELEVANCE_MARKERS,
    SOURCE_RELEVANCE_TARGETS,
    is_action_list_question,
    is_name_only_lookup_question,
    is_relationship_entity_question,
    is_scenario_procedure_question,
    is_source_relevance_question,
    is_workaround_question,
)
from .retrieval_orchestrator import (
    RetrievalOrchestrator,
    dense_search_results_to_ranked_records,
    retrieved_sources_to_ranked_records,
)
from .source_finalizer import SourceFinalizer, child_id_for

__all__ = [
    "ACTION_LIST_MARKERS",
    "CANONICAL_ABSTENTION_RESPONSE",
    "ChatClient",
    "NAME_ONLY_LOOKUP_MARKERS",
    "NAME_ONLY_LOOKUP_TARGETS",
    "PromptBuilder",
    "RAGPipeline",
    "RAGResponse",
    "RAG_SYSTEM_PROMPT_VERSION",
    "RELATIONSHIP_ENTITY_MARKERS",
    "RELATIONSHIP_ENTITY_TARGETS",
    "RetrievalOrchestrator",
    "RetrievedSource",
    "SCENARIO_PROCEDURE_KEYWORDS",
    "SCENARIO_SITUATION_MARKERS",
    "SOURCE_RELEVANCE_MARKERS",
    "SOURCE_RELEVANCE_TARGETS",
    "SourceFinalizer",
    "child_id_for",
    "dense_search_results_to_ranked_records",
    "extract_chat_content",
    "is_action_list_question",
    "is_name_only_lookup_question",
    "is_relationship_entity_question",
    "is_scenario_procedure_question",
    "is_source_relevance_question",
    "is_workaround_question",
    "retrieved_sources_to_ranked_records",
]

logger = setup_logger(__name__)


class RAGPipeline:
    """Retrieval-augmented generation pipeline orchestrator."""

    DEFAULT_SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT

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
            raise ValueError(
                "query_rewrite_weight_original must be greater than or equal to query_rewrite_weight_variant"
            )

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

        self.retrieval_orchestrator = RetrievalOrchestrator(self)
        self.source_finalizer = SourceFinalizer(self)
        self.prompt_builder = PromptBuilder(self)

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
        candidates = self.retrieval_orchestrator.retrieve(
            question=question,
            requested_top_k=requested_top_k,
            metadata_filter=metadata_filter,
        )
        return self.source_finalizer.finalize(candidates)

    def build_prompt(self, question: str, sources: List[RetrievedSource]) -> str:
        """Build the user prompt sent to the chat client."""
        prompt, _ = self.prompt_builder.build(question, sources)
        return prompt

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> RAGResponse:
        """Retrieve context and generate an answer."""
        sources = self.retrieve(question, top_k=top_k, metadata_filter=metadata_filter)
        prompt, used_sources = self.prompt_builder.build(question, sources)
        raw_response = self.chat_client.chat(prompt, system_prompt=self.system_prompt)
        answer = extract_chat_content(raw_response)

        logger.info(
            "RAG answer generated | sources=%s | answer_chars=%s", len(used_sources), len(answer)
        )
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
        return self.prompt_builder.build(question, sources)

    def _build_context(self, sources: List[RetrievedSource]) -> str:
        """Build a bounded context block from retrieved sources."""
        return self.prompt_builder.build_context(sources)

    def _format_source_label(self, source: RetrievedSource) -> str:
        """Format human-readable source metadata."""
        return self.prompt_builder.format_source_label(source)
