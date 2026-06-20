from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceResponse(BaseModel):
    """Retrieved source returned by the HTTP API."""

    index: int
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """HTTP request body for a complete RAG query."""

    question: str = Field(..., min_length=1)
    principal: Optional[str] = None
    top_k: Optional[int] = Field(default=None, gt=0)
    metadata_filter: Optional[Dict[str, Any]] = None
    max_context_chars: int = Field(default=4000, gt=0)
    embedding_provider: Optional[str] = None
    embedding_dimension: Optional[int] = Field(default=None, gt=0)
    vector_store: Optional[str] = None
    rerank_provider: Optional[str] = None
    rerank_fetch_k: Optional[int] = Field(default=None, gt=0)
    hybrid: bool = False
    hybrid_fetch_k: Optional[int] = Field(default=None, gt=0)
    rrf_k: Optional[int] = Field(default=None, gt=0)
    hybrid_dense_weight: Optional[float] = Field(default=None, ge=0)
    hybrid_sparse_weight: Optional[float] = Field(default=None, ge=0)
    bm25_k1: Optional[float] = Field(default=None, gt=0)
    bm25_b: Optional[float] = Field(default=None, ge=0)
    parent_child: bool = False
    context_packing: bool = False
    context_dedup: bool = False
    context_near_dup: bool = False
    context_near_dup_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    context_max_tokens: Optional[int] = Field(default=None, gt=0)
    tokenizer_encoding: Optional[str] = None
    multi_query: bool = False
    query_rewrite_provider: Optional[str] = None
    query_rewrite_fixture: Optional[str] = None
    query_rewrite_num_queries: Optional[int] = Field(default=None, ge=1)
    query_rewrite_temperature: Optional[float] = Field(default=None, ge=0)
    query_rewrite_cache_enabled: Optional[bool] = None
    query_rewrite_weight_original: Optional[float] = Field(default=None, ge=0)
    query_rewrite_weight_variant: Optional[float] = Field(default=None, ge=0)


class QueryResponse(BaseModel):
    """HTTP response body for a complete RAG query."""

    question: str
    answer: str
    sources: List[SourceResponse]


class IngestRequest(BaseModel):
    """HTTP request body for synchronous-offloaded ingestion."""

    path: str = Field(..., min_length=1)
    acl: Optional[List[str]] = None
    clean: bool = True
    recursive: bool = True
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    embedding_provider: Optional[str] = None
    embedding_dimension: Optional[int] = Field(default=None, gt=0)
    vector_store: Optional[str] = None
    parent_child: bool = False
    parent_chunk_size: Optional[int] = Field(default=None, gt=0)
    parent_chunk_overlap: Optional[int] = Field(default=None, ge=0)
    child_chunk_size: Optional[int] = Field(default=None, gt=0)
    child_chunk_overlap: Optional[int] = Field(default=None, ge=0)


class IngestResponse(BaseModel):
    """HTTP response body for ingestion."""

    records_added: int
    embedding_model: str
    embedding_dimension: int
    parent_count: int
    cache_invalidated: bool


class ErrorResponse(BaseModel):
    """Sanitized HTTP error response."""

    error: Dict[str, str]


class HealthResponse(BaseModel):
    """Health response for the service."""

    status: str
    warning: str
