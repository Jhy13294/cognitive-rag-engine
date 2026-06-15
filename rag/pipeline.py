from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol

from embeddings import EmbeddingProvider
from logger import setup_logger
from rerank import Reranker
from vector_store import SearchResult, VectorStore

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


class RAGPipeline:
    """Minimal retrieval-augmented generation pipeline."""

    DEFAULT_SYSTEM_PROMPT = (
        "You are an enterprise knowledge-base assistant. "
        "Answer using only the provided context. "
        "If the context is insufficient, say that the answer is not available in the knowledge base. "
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
    ):
        """Initialize the RAG pipeline."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than 0")
        if fetch_k is not None and fetch_k <= 0:
            raise ValueError("fetch_k must be greater than 0")

        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.chat_client = chat_client
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.reranker = reranker
        self.fetch_k = fetch_k

    def retrieve(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> List[RetrievedSource]:
        """Retrieve source chunks for a question."""
        requested_top_k = top_k or self.top_k
        query_embedding = self.embedding_provider.embed_text(question)
        search_results = self.vector_store.similarity_search(
            query_embedding,
            top_k=self._search_top_k(requested_top_k),
            metadata_filter=metadata_filter,
        )
        sources = self._to_sources(search_results)

        if self.reranker is None:
            return sources

        try:
            return self._rerank_sources(question, sources, requested_top_k)
        except Exception as e:
            logger.warning(
                "Rerank failed; falling back to dense order | error=%s | requested_top_k=%s",
                e,
                requested_top_k,
            )
            return sources[:requested_top_k]

    def build_prompt(self, question: str, sources: List[RetrievedSource]) -> str:
        """Build the user prompt sent to the chat client."""
        context = self._build_context(sources)
        return (
            "Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer with source citations."
        )

    def answer(
        self,
        question: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict] = None,
    ) -> RAGResponse:
        """Retrieve context and generate an answer."""
        sources = self.retrieve(question, top_k=top_k, metadata_filter=metadata_filter)
        prompt = self.build_prompt(question, sources)
        raw_response = self.chat_client.chat(prompt, system_prompt=self.system_prompt)
        answer = extract_chat_content(raw_response)

        logger.info("RAG answer generated | sources=%s | answer_chars=%s", len(sources), len(answer))
        return RAGResponse(
            question=question,
            answer=answer,
            sources=sources,
            prompt=prompt,
            raw_response=raw_response,
        )

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
        if self.reranker is None:
            return requested_top_k
        return max(self.fetch_k or self.reranker.fetch_k, requested_top_k)

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

    def _build_context(self, sources: List[RetrievedSource]) -> str:
        """Build a bounded context block from retrieved sources."""
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
