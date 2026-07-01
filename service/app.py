import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from api_client import APIClient
from access import ACLAccessError, build_effective_metadata_filter, create_acl_resolver_from_config
from cache import get_default_cache_store
from config import Config
from logger import setup_logger
from observability import QueryObservation, RequestIDMiddleware, create_observability_manager
from rag import RAGPipeline, RAGResponse, RetrievedSource
from rag_cli import build_rag_pipeline_from_index, ingest_documents
from vector_store import check_qdrant_connectivity

from .errors import to_http_exception
from .models import (
    DependencyStatus,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    ReadinessResponse,
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
    acl_resolver: Any = None
    observability: Any = None
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
    service_state = state or ServiceState()
    Config.validate_cache()
    Config.validate_acl(resolver_provided=service_state.acl_resolver is not None)
    Config.validate_observability()
    Config.validate_vector_store()
    Config.validate_embedding()
    Config.validate_readiness()
    if Config.ACL_ENABLED and service_state.acl_resolver is None:
        service_state.acl_resolver = create_acl_resolver_from_config(Config)
    if Config.OBSERVABILITY_ENABLED and service_state.observability is None:
        service_state.observability = create_observability_manager(
            Config,
            cache_store=service_state.cache_store,
        )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        """Close process-level cache resources on shutdown."""
        try:
            yield
        finally:
            observability = service_state.observability
            if observability is not None and hasattr(observability, "close"):
                await asyncio.to_thread(observability.close)
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
    if Config.OBSERVABILITY_ENABLED:
        app.add_middleware(RequestIDMiddleware)

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Return service health."""
        return HealthResponse(
            status="ok",
            warning="Authentication is not implemented; do not expose this service publicly.",
        )

    @app.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
    )
    async def ready() -> JSONResponse:
        """Return dependency readiness for traffic admission."""
        report = await build_readiness_report(service_state)
        status_code = 200 if report.status == "ready" else 503
        return JSONResponse(
            status_code=status_code,
            content=report.model_dump(exclude_none=True),
        )

    @app.get("/cache/stats")
    async def cache_stats() -> Dict:
        """Return in-process cache counters."""
        service_state = get_state(app)
        if service_state.cache_store is None:
            return {"enabled": False}
        return service_state.cache_store.stats()

    if Config.OBSERVABILITY_ENABLED and Config.METRICS_ENABLED:

        @app.get(Config.METRICS_PATH, response_class=PlainTextResponse)
        async def metrics() -> PlainTextResponse:
            """Expose bounded-label metrics in Prometheus text format."""
            observability = get_state(app).observability
            payload = observability.render_metrics() if observability is not None else "# metrics unavailable\n"
            return PlainTextResponse(
                payload,
                media_type="text/plain; version=0.0.4",
            )

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
                acl=request.acl,
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
    async def query(request: QueryRequest, http_request: Request) -> QueryResponse:
        """Run a complete RAG query without blocking the event loop."""
        observation = start_query_observation(app, http_request, request, route="query")
        try:
            principal = extract_trusted_principal(http_request, request.principal)
            metadata_filter = await asyncio.to_thread(resolve_request_metadata_filter, app, request, principal)
            pipeline = await get_or_build_pipeline(app, request)
            bind_query_observation(app, observation, pipeline)
            response = await asyncio.to_thread(
                pipeline.answer,
                request.question,
                top_k=request.top_k,
                metadata_filter=metadata_filter,
            )
            response_model = response_to_model(response)
            finish_query_observation(
                app,
                observation,
                sources=response.sources,
                status_code=200,
                raw_response=response.raw_response,
                prompt=response.prompt,
                answer=response.answer,
            )
            return response_model
        except HTTPException as error:
            finish_query_observation(app, observation, status_code=error.status_code)
            raise
        except Exception as e:
            http_error = to_http_exception(e)
            finish_query_observation(app, observation, status_code=http_error.status_code)
            raise http_error from e

    @app.post("/query/stream")
    async def query_stream(request: QueryRequest, http_request: Request) -> StreamingResponse:
        """Run a streaming RAG query as server-sent events."""
        observation = start_query_observation(app, http_request, request, route="query_stream")
        try:
            principal = extract_trusted_principal(http_request, request.principal)
            metadata_filter = await asyncio.to_thread(resolve_request_metadata_filter, app, request, principal)
            pipeline = await get_or_build_pipeline(app, request)
            bind_query_observation(app, observation, pipeline)
        except HTTPException as error:
            finish_query_observation(app, observation, status_code=error.status_code)
            raise
        except Exception as e:
            http_error = to_http_exception(e)
            finish_query_observation(app, observation, status_code=http_error.status_code)
            raise http_error from e

        return StreamingResponse(
            stream_query_events(
                app,
                request,
                pipeline=pipeline,
                metadata_filter=metadata_filter,
                observation=observation,
            ),
            media_type="text/event-stream",
        )

    return app


async def build_readiness_report(service_state: ServiceState) -> ReadinessResponse:
    """Build a fail-closed readiness report for enabled dependencies."""
    probes: List[Awaitable[DependencyStatus]] = []
    if Config.VECTOR_STORE_PROVIDER.lower() == "qdrant":
        probes.append(
            probe_dependency(
                "qdrant",
                lambda: asyncio.to_thread(
                    check_qdrant_connectivity,
                    host=Config.VECTOR_STORE_HOST,
                    port=Config.VECTOR_STORE_PORT,
                    api_key=Config.VECTOR_STORE_API_KEY,
                    url=Config.VECTOR_STORE_URL,
                    timeout=Config.READINESS_TIMEOUT,
                ),
            )
        )
    if Config.CACHE_ENABLED:
        probes.append(probe_dependency("redis", lambda: ping_redis(service_state)))
    if Config.ACL_ENABLED:
        probes.append(probe_dependency("mysql", lambda: check_mysql(service_state)))

    dependencies = list(await asyncio.gather(*probes)) if probes else []
    dependencies.append(generation_dependency_status())
    required_down = any(
        dependency.required and dependency.status != "up"
        for dependency in dependencies
    )
    return ReadinessResponse(
        status="not_ready" if required_down else "ready",
        dependencies=dependencies,
    )


async def probe_dependency(
    name: str,
    probe_factory: Callable[[], Awaitable[bool]],
) -> DependencyStatus:
    """Run one readiness probe with a bounded timeout."""
    started = time.perf_counter()
    timeout_seconds = Config.READINESS_TIMEOUT
    try:
        ok = await asyncio.wait_for(probe_factory(), timeout=timeout_seconds)
    except asyncio.TimeoutError:
        return DependencyStatus(
            name=name,
            status="down",
            latency_ms=elapsed_ms(started),
            detail=f"Timed out after {timeout_seconds:.2f}s",
        )
    except Exception as error:
        return DependencyStatus(
            name=name,
            status="down",
            latency_ms=elapsed_ms(started),
            detail=f"Connectivity check failed: {type(error).__name__}",
        )

    if not ok:
        return DependencyStatus(
            name=name,
            status="down",
            latency_ms=elapsed_ms(started),
            detail="Connectivity check returned false",
        )
    return DependencyStatus(name=name, status="up", latency_ms=elapsed_ms(started))


async def ping_redis(service_state: ServiceState) -> bool:
    """Ping the configured Redis store without touching its client event loop."""
    cache_store = service_state.cache_store
    if cache_store is None or not hasattr(cache_store, "ping"):
        return False
    return bool(await cache_store.ping(timeout_seconds=Config.READINESS_TIMEOUT))


async def check_mysql(service_state: ServiceState) -> bool:
    """Check MySQL ACL metadata connectivity off the event loop."""
    resolver = service_state.acl_resolver
    if resolver is None or not hasattr(resolver, "check_connectivity"):
        return False
    return bool(await asyncio.to_thread(resolver.check_connectivity, Config.READINESS_TIMEOUT))


def generation_dependency_status() -> DependencyStatus:
    """Expose generation configuration without making it a boot blocker."""
    if Config.API_KEY:
        return DependencyStatus(name="generation", status="up", required=False)
    return DependencyStatus(
        name="generation",
        status="down",
        required=False,
        detail="DEEPSEEK_API_KEY is not configured; generation requests will fail until configured.",
    )


def elapsed_ms(started: float) -> float:
    """Return elapsed milliseconds rounded for readiness output."""
    return round((time.perf_counter() - started) * 1000, 2)


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


async def stream_query_events(
    app: FastAPI,
    request: QueryRequest,
    pipeline: Optional[RAGPipeline] = None,
    metadata_filter: Optional[Dict] = None,
    observation: Optional[QueryObservation] = None,
):
    """Yield SSE events for a streaming RAG query."""
    final_sources = []
    final_raw_response = {}
    final_prompt = ""
    final_answer = ""
    final_status_code = 200
    try:
        if pipeline is None:
            pipeline = await get_or_build_pipeline(app, request)
            bind_query_observation(app, observation, pipeline)
        if metadata_filter is None:
            metadata_filter = await asyncio.to_thread(resolve_request_metadata_filter, app, request)

        cached_getter = getattr(pipeline, "get_cached_answer", None)
        if cached_getter is not None:
            cached_response = await asyncio.to_thread(
                cached_getter,
                request.question,
                request.top_k,
                metadata_filter,
            )
            if cached_response is not None:
                final_sources = cached_response.sources
                final_raw_response = cached_response.raw_response
                final_prompt = cached_response.prompt
                final_answer = cached_response.answer
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
            metadata_filter,
        )
        prompt, used_sources = await asyncio.to_thread(
            pipeline._build_prompt_and_sources,
            request.question,
            sources,
        )
        final_sources = used_sources
        final_prompt = prompt
        yield sse_event("sources", {"sources": sources_to_models(used_sources)})

        if not hasattr(pipeline.chat_client, "stream_chat"):
            raise ValueError("Configured chat client does not support streaming.")

        answer_parts = []
        async for token in pipeline.chat_client.stream_chat(prompt, system_prompt=pipeline.system_prompt):
            answer_parts.append(token)
            yield sse_event("token", {"delta": token})

        answer = "".join(answer_parts)
        final_answer = answer
        final_raw_response = {"choices": [{"message": {"content": answer}}], "streamed": True}
        store_answer = getattr(pipeline, "store_answer", None)
        if store_answer is not None:
            streamed_response = RAGResponse(
                question=request.question,
                answer=answer,
                sources=used_sources,
                prompt=prompt,
                raw_response=final_raw_response,
            )
            await asyncio.to_thread(store_answer, streamed_response, request.top_k, metadata_filter)

        yield sse_event("done", {"answer": answer})
    except (asyncio.CancelledError, GeneratorExit):
        final_status_code = 499
        raise
    except Exception as e:
        http_error = to_http_exception(e)
        final_status_code = http_error.status_code
        yield sse_event("error", http_error.detail)
    finally:
        finish_query_observation(
            app,
            observation,
            sources=final_sources,
            status_code=final_status_code,
            raw_response=final_raw_response,
            prompt=final_prompt,
            answer=final_answer,
        )


def start_query_observation(
    app: FastAPI,
    http_request: Request,
    request: QueryRequest,
    *,
    route: str,
) -> Optional[QueryObservation]:
    """Start fail-open observation state for one query request."""
    observability = get_state(app).observability
    if observability is None:
        return None

    request_id = getattr(http_request.state, "request_id", None) or uuid.uuid4().hex
    principal = http_request.headers.get(Config.ACL_PRINCIPAL_HEADER) or request.principal
    retrieval_mode = "multi_query" if request.multi_query else "hybrid" if request.hybrid else "dense"
    try:
        return observability.start_query(
            request_id=request_id,
            route=route,
            principal=principal,
            question=request.question,
            top_k=request.top_k or 5,
            retrieval_mode=retrieval_mode,
        )
    except Exception as error:
        logger.warning(
            "Observability start failed open | alert_code=observability_start_failed | request_id=%s | error_type=%s",
            request_id,
            type(error).__name__,
        )
        return None


def bind_query_observation(app: FastAPI, observation: Optional[QueryObservation], pipeline: Any) -> None:
    """Bind an existing pipeline to observation state fail-open."""
    if observation is None:
        return
    observability = get_state(app).observability
    if observability is None:
        return
    try:
        observability.bind_pipeline(observation, pipeline)
    except Exception as error:
        logger.warning(
            "Observability pipeline bind failed open | alert_code=observability_bind_failed | request_id=%s | error_type=%s",
            observation.request_id,
            type(error).__name__,
        )


def finish_query_observation(
    app: FastAPI,
    observation: Optional[QueryObservation],
    *,
    sources: Optional[List[Any]] = None,
    status_code: int,
    raw_response: Optional[Dict] = None,
    prompt: str = "",
    answer: str = "",
) -> None:
    """Finalize one observation without allowing failures into the response path."""
    if observation is None:
        return
    observability = get_state(app).observability
    if observability is None:
        return
    try:
        observability.complete_query(
            observation,
            sources=sources,
            status_code=status_code,
            raw_response=raw_response,
            prompt=prompt,
            answer=answer,
        )
    except Exception as error:
        logger.warning(
            "Observability completion failed open | alert_code=observability_complete_failed | request_id=%s | error_type=%s",
            observation.request_id,
            type(error).__name__,
        )


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
    payload = request.model_dump(exclude={"question", "metadata_filter", "principal"})
    return tuple(sorted(payload.items()))


def resolve_request_metadata_filter(app: FastAPI, request: QueryRequest, principal: Optional[str] = None) -> Optional[Dict]:
    """Return the effective server-enforced metadata filter for a query."""
    if not Config.ACL_ENABLED:
        return request.metadata_filter

    if principal is None or not str(principal).strip():
        raise ACLAccessError("Trusted principal is required when ACL is enabled.")

    service_state = get_state(app)
    if service_state.acl_resolver is None:
        raise ACLAccessError("ACL resolver is not configured.")

    allowed_acl = service_state.acl_resolver.allowed_acl_for_principal(str(principal).strip())
    return build_effective_metadata_filter(
        request.metadata_filter,
        allowed_acl,
        metadata_key=Config.ACL_METADATA_KEY,
        default_deny=Config.ACL_DEFAULT_DENY,
    )


def extract_trusted_principal(http_request: Request, body_principal: Optional[str] = None) -> Optional[str]:
    """Extract principal from the trusted upstream header, with explicit local fallback only."""
    if not Config.ACL_ENABLED:
        return body_principal

    header_principal = http_request.headers.get(Config.ACL_PRINCIPAL_HEADER)
    if header_principal and header_principal.strip():
        return header_principal.strip()
    if Config.ACL_ALLOW_BODY_PRINCIPAL and body_principal and body_principal.strip():
        return body_principal.strip()
    if body_principal and body_principal.strip():
        raise ACLAccessError("Principal must come from the trusted upstream header.")
    return None


def default_chat_client_factory() -> APIClient:
    """Create the default async-capable chat client."""
    Config.validate()
    return APIClient(Config.API_KEY, Config.API_URL)


def get_state(app: FastAPI) -> ServiceState:
    """Return typed service state from the app."""
    return app.state.service_state


app = create_app()
