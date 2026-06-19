import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from api_client import APIClient
from cache import get_default_cache_store
from config import Config
from logger import setup_logger
from rag import RAGPipeline, RAGResponse, RetrievedSource
from rag_cli import build_rag_pipeline_from_index, ingest_documents

from .errors import to_http_exception
from .models import (
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)

logger = setup_logger(__name__)


@dataclass
class ServiceState:
    """Mutable service state for pipeline caching and dependency injection."""

    ingest_callable: Any = ingest_documents
    pipeline_builder: Any = build_rag_pipeline_from_index
    chat_client_factory: Any = None
    cache_store: Any = None
    pipeline_cache: Dict[Tuple, RAGPipeline] = field(default_factory=dict)
    cache_generation: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        """Install the default chat client factory."""
        if self.chat_client_factory is None:
            self.chat_client_factory = default_chat_client_factory
        if self.cache_store is None:
            self.cache_store = get_default_cache_store()

    async def invalidate(self) -> None:
        """Invalidate cached query pipelines after ingestion."""
        async with self.lock:
            self.pipeline_cache.clear()
            self.cache_generation += 1
        if self.cache_store is not None:
            await self.cache_store.bump_corpus_version()


def create_app(state: Optional[ServiceState] = None) -> FastAPI:
    """Create the FastAPI application."""
    Config.validate_cache()
    service_state = state or ServiceState()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Close process-level cache resources on shutdown."""
        try:
            yield
        finally:
            cache_store = service_state.cache_store
            if cache_store is not None and hasattr(cache_store, "aclose"):
                await cache_store.aclose()

    app = FastAPI(
        title="Enterprise RAG Service",
        version="0.1.0",
        description="Async HTTP adapter for the existing ingest/query RAG pipeline.",
        lifespan=lifespan,
    )
    app.state.service_state = service_state

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return service health."""
        return HealthResponse(
            status="ok",
            warning="Authentication is not implemented; do not expose this service publicly.",
        )

    @app.get("/cache/stats")
    async def cache_stats() -> Dict:
        """Return in-process cache counters."""
        service_state = get_state(app)
        if service_state.cache_store is None:
            return {"enabled": False}
        return service_state.cache_store.stats()

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest(request: IngestRequest) -> IngestResponse:
        """Ingest documents through the existing T03 ingest function."""
        service_state = get_state(app)
        try:
            result = await asyncio.to_thread(
                service_state.ingest_callable,
                request.path,
                clean=request.clean,
                recursive=request.recursive,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
                embedding_provider_name=request.embedding_provider,
                embedding_dimension=request.embedding_dimension,
                vector_store_name=request.vector_store,
                parent_child_enabled=request.parent_child,
                parent_chunk_size=request.parent_chunk_size,
                parent_chunk_overlap=request.parent_chunk_overlap,
                child_chunk_size=request.child_chunk_size,
                child_chunk_overlap=request.child_chunk_overlap,
            )
            await service_state.invalidate()
            return IngestResponse(
                records_added=len(result.ids),
                embedding_model=result.embedding_model,
                embedding_dimension=result.embedding_dimension,
                parent_count=result.parent_count,
                cache_invalidated=True,
            )
        except Exception as e:
            raise to_http_exception(e) from e

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        """Run a complete RAG query without blocking the event loop."""
        try:
            pipeline = await get_or_build_pipeline(app, request)
            response = await asyncio.to_thread(
                pipeline.answer,
                request.question,
                top_k=request.top_k,
                metadata_filter=request.metadata_filter,
            )
            return response_to_model(response)
        except HTTPException:
            raise
        except Exception as e:
            raise to_http_exception(e) from e

    @app.post("/query/stream")
    async def query_stream(request: QueryRequest) -> StreamingResponse:
        """Run a streaming RAG query as server-sent events."""
        try:
            pipeline = await get_or_build_pipeline(app, request)
        except HTTPException:
            raise
        except Exception as e:
            raise to_http_exception(e) from e

        return StreamingResponse(
            stream_query_events(app, request, pipeline=pipeline),
            media_type="text/event-stream",
        )

    return app


async def get_or_build_pipeline(app: FastAPI, request: QueryRequest) -> RAGPipeline:
    """Return a cached pipeline for query options, rebuilding when invalidated."""
    service_state = get_state(app)
    cache_key = query_cache_key(request)
    async with service_state.lock:
        cached = service_state.pipeline_cache.get(cache_key)
        if cached is not None:
            return cached

    try:
        pipeline = await asyncio.to_thread(
            service_state.pipeline_builder,
            chat_client=service_state.chat_client_factory(),
            embedding_provider_name=request.embedding_provider,
            embedding_dimension=request.embedding_dimension,
            vector_store_name=request.vector_store,
            rerank_provider_name=request.rerank_provider,
            rerank_fetch_k=request.rerank_fetch_k,
            hybrid_enabled=True if request.hybrid else None,
            hybrid_fetch_k=request.hybrid_fetch_k,
            rrf_k=request.rrf_k,
            hybrid_dense_weight=request.hybrid_dense_weight,
            hybrid_sparse_weight=request.hybrid_sparse_weight,
            bm25_k1=request.bm25_k1,
            bm25_b=request.bm25_b,
            parent_child_enabled=True if request.parent_child else None,
            context_packing_enabled=True if request.context_packing else None,
            context_dedup_enabled=True if request.context_dedup else None,
            context_near_dup_enabled=True if request.context_near_dup else None,
            context_near_dup_threshold=request.context_near_dup_threshold,
            context_max_tokens=request.context_max_tokens,
            tokenizer_encoding=request.tokenizer_encoding,
            query_rewrite_enabled=True if request.multi_query else None,
            query_rewrite_provider_name=request.query_rewrite_provider,
            query_rewrite_fixture_path=request.query_rewrite_fixture,
            query_rewrite_num_queries=request.query_rewrite_num_queries,
            query_rewrite_temperature=request.query_rewrite_temperature,
            query_rewrite_cache_enabled=request.query_rewrite_cache_enabled,
            query_rewrite_weight_original=request.query_rewrite_weight_original,
            query_rewrite_weight_variant=request.query_rewrite_weight_variant,
            top_k=request.top_k or 5,
            max_context_chars=request.max_context_chars,
        )
    except Exception as e:
        raise to_http_exception(e) from e

    async with service_state.lock:
        service_state.pipeline_cache[cache_key] = pipeline
    return pipeline


async def stream_query_events(app: FastAPI, request: QueryRequest, pipeline: Optional[RAGPipeline] = None):
    """Yield SSE events for a streaming RAG query."""
    try:
        if pipeline is None:
            pipeline = await get_or_build_pipeline(app, request)

        cached_getter = getattr(pipeline, "get_cached_answer", None)
        if cached_getter is not None:
            cached_response = await asyncio.to_thread(
                cached_getter,
                request.question,
                request.top_k,
                request.metadata_filter,
            )
            if cached_response is not None:
                yield sse_event(
                    "sources",
                    {
                        "sources": sources_to_models(cached_response.sources),
                        "cached": True,
                        "stream_replay": True,
                    },
                )
                yield sse_event("token", {"delta": cached_response.answer, "cached": True})
                yield sse_event("done", {"answer": cached_response.answer, "cached": True})
                return

        sources = await asyncio.to_thread(
            pipeline.retrieve,
            request.question,
            request.top_k,
            request.metadata_filter,
        )
        prompt, used_sources = await asyncio.to_thread(
            pipeline._build_prompt_and_sources,
            request.question,
            sources,
        )
        yield sse_event("sources", {"sources": sources_to_models(used_sources)})

        if not hasattr(pipeline.chat_client, "stream_chat"):
            raise ValueError("Configured chat client does not support streaming.")

        answer_parts = []
        async for token in pipeline.chat_client.stream_chat(prompt, system_prompt=pipeline.system_prompt):
            answer_parts.append(token)
            yield sse_event("token", {"delta": token})

        answer = "".join(answer_parts)
        store_answer = getattr(pipeline, "store_answer", None)
        if store_answer is not None:
            streamed_response = RAGResponse(
                question=request.question,
                answer=answer,
                sources=used_sources,
                prompt=prompt,
                raw_response={"choices": [{"message": {"content": answer}}], "streamed": True},
            )
            await asyncio.to_thread(store_answer, streamed_response, request.top_k, request.metadata_filter)

        yield sse_event("done", {"answer": answer})
    except Exception as e:
        http_error = to_http_exception(e)
        yield sse_event("error", http_error.detail)


def response_to_model(response: RAGResponse) -> QueryResponse:
    """Convert an internal RAGResponse into an HTTP response model."""
    return QueryResponse(
        question=response.question,
        answer=response.answer,
        sources=sources_to_models(response.sources),
    )


def sources_to_models(sources) -> list:
    """Convert internal source objects into serializable dictionaries."""
    return [
        SourceResponse(
            index=source.index,
            content=source.content,
            score=source.score,
            metadata=dict(source.metadata),
        ).model_dump()
        for source in sources
    ]


def sse_event(event_name: str, payload: Dict) -> str:
    """Format one server-sent event."""
    return f"event: {event_name}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def query_cache_key(request: QueryRequest) -> Tuple:
    """Build a cache key from query options that affect pipeline construction."""
    payload = request.model_dump(exclude={"question", "metadata_filter"})
    return tuple(sorted(payload.items()))


def default_chat_client_factory() -> APIClient:
    """Create the default async-capable chat client."""
    Config.validate()
    return APIClient(Config.API_KEY, Config.API_URL)


def get_state(app: FastAPI) -> ServiceState:
    """Return typed service state from the app."""
    return app.state.service_state


app = create_app()
