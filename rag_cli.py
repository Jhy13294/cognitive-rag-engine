import argparse
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from access import build_effective_metadata_filter, create_acl_resolver_from_config, normalize_acl_values
from cache import maybe_wrap_embedding_provider, maybe_wrap_pipeline
from config import Config
from document_loader import Document
from document_loader import load_and_split_documents, load_and_split_documents_hierarchical
from embeddings import EmbeddingProvider, HashEmbeddingProvider, OpenAIEmbeddingProvider
from hybrid import BM25Retriever, RRFConfig, ReciprocalRankFusion
from logger import setup_logger
from parent_store import InMemoryParentStore
from query_rewrite import create_query_rewriter
from rag import EmbeddingSpaceInvalidError, EmbeddingSpaceMismatchError, IndexNotReadyError, RAGPipeline, RAGResponse
from rerank import create_reranker
from tokenization import validate_tokenizer_encoding
from vector_store import create_vector_store
from vector_store.base import VectorRecord, VectorStore, embedded_document_to_record

logger = setup_logger(__name__, level=logging.INFO)

CLI_COMMANDS = {"oneshot", "ingest", "query"}


@dataclass
class IngestResult:
    """Result of an ingest command."""

    ids: List[str]
    records: List[VectorRecord]
    vector_store: VectorStore
    embedding_provider: EmbeddingProvider
    embedding_model: str
    embedding_dimension: int
    parent_count: int = 0


class RAGArgumentParser(argparse.ArgumentParser):
    """Argument parser that preserves legacy path-first CLI compatibility."""

    def parse_args(self, args=None, namespace=None):
        """Parse and normalize subcommands plus legacy oneshot arguments."""
        parsed_args = super().parse_args(args=args, namespace=namespace)
        normalize_cli_command(parsed_args, parser=self)
        return parsed_args


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RAG CLI argument parser."""
    parser = RAGArgumentParser(description="Run a minimal local RAG workflow.")
    parser.add_argument(
        "command_or_path",
        nargs="?",
        help="Subcommand: ingest, query, oneshot. A non-command value keeps legacy oneshot path behavior.",
    )
    parser.add_argument(
        "path_or_question",
        nargs="?",
        help="Path for ingest/oneshot or question for query.",
    )
    parser.add_argument("-q", "--question", help="Question to answer. If omitted, interactive mode starts.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks.")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size for document splitting.")
    parser.add_argument("--chunk-overlap", type=int, default=120, help="Chunk overlap for document splitting.")
    parser.add_argument(
        "--embedding-provider",
        choices=["openai", "hash"],
        default=None,
        help="Embedding provider. Defaults to EMBEDDING_PROVIDER.",
    )
    parser.add_argument("--embedding-dimension", type=int, default=None, help="Embedding dimension override.")
    parser.add_argument(
        "--vector-store",
        choices=["memory", "qdrant"],
        default=None,
        help="Vector store provider. Defaults to VECTOR_STORE_PROVIDER.",
    )
    parser.add_argument(
        "--rerank-provider",
        choices=["none", "deterministic", "cohere"],
        default=None,
        help="Reranker provider. Defaults to RERANK_* configuration.",
    )
    parser.add_argument("--rerank-fetch-k", type=int, default=None, help="Dense candidate count before rerank.")
    parser.add_argument("--hybrid", action="store_true", help="Enable dense + BM25 retrieval with RRF fusion.")
    parser.add_argument("--hybrid-fetch-k", type=int, default=None, help="Candidate count per path before RRF fusion.")
    parser.add_argument("--rrf-k", type=int, default=None, help="RRF rank constant.")
    parser.add_argument("--hybrid-dense-weight", type=float, default=None, help="Dense path RRF weight.")
    parser.add_argument("--hybrid-sparse-weight", type=float, default=None, help="BM25 path RRF weight.")
    parser.add_argument("--bm25-k1", type=float, default=None, help="BM25 k1 parameter.")
    parser.add_argument("--bm25-b", type=float, default=None, help="BM25 b parameter.")
    parser.add_argument("--parent-child", action="store_true", help="Enable parent-child chunking and expansion.")
    parser.add_argument("--parent-chunk-size", type=int, default=None, help="Parent chunk size for generation context.")
    parser.add_argument("--parent-chunk-overlap", type=int, default=None, help="Parent chunk overlap.")
    parser.add_argument("--child-chunk-size", type=int, default=None, help="Child chunk size for retrieval indexing.")
    parser.add_argument("--child-chunk-overlap", type=int, default=None, help="Child chunk overlap.")
    parser.add_argument("--context-packing", action="store_true", help="Enable T08 context packing.")
    parser.add_argument("--context-dedup", action="store_true", help="Enable exact context deduplication.")
    parser.add_argument("--context-near-dup", action="store_true", help="Enable fuzzy near-duplicate context deduplication.")
    parser.add_argument("--context-near-dup-threshold", type=float, default=None, help="Near-duplicate threshold.")
    parser.add_argument("--context-max-tokens", type=int, default=None, help="Maximum context tokens when packing is enabled.")
    parser.add_argument("--tokenizer-encoding", default=None, help="Tokenizer encoding for token budgets.")
    parser.add_argument("--multi-query", action="store_true", help="Enable query rewrite and multi-query RRF.")
    parser.add_argument(
        "--query-rewrite-provider",
        choices=["deterministic", "chat"],
        default=None,
        help="Query rewrite provider. Defaults to QUERY_REWRITE_PROVIDER.",
    )
    parser.add_argument("--query-rewrite-fixture", default=None, help="Deterministic query rewrite fixture path.")
    parser.add_argument("--query-rewrite-num-queries", type=int, default=None, help="Total query variants including original.")
    parser.add_argument("--query-rewrite-temperature", type=float, default=None, help="Chat query rewrite temperature.")
    parser.add_argument("--no-query-rewrite-cache", action="store_true", help="Disable query rewrite cache.")
    parser.add_argument("--query-rewrite-weight-original", type=float, default=None, help="Original query RRF weight.")
    parser.add_argument("--query-rewrite-weight-variant", type=float, default=None, help="Rewritten query RRF weight.")
    parser.add_argument("--max-context-chars", type=int, default=4000, help="Maximum context characters.")
    parser.add_argument("--metadata-filter", help="JSON exact-match metadata filter, for example '{\"file_type\":\"txt\"}'.")
    parser.add_argument("--principal", help="Authenticated principal used for ACL-filtered query commands.")
    parser.add_argument(
        "--acl",
        action="append",
        help="ACL subject for trusted local ingest or CLI query. Repeat or pass comma-separated values.",
    )
    parser.add_argument("--no-clean", action="store_true", help="Disable text cleaning before chunking.")
    parser.add_argument("--non-recursive", action="store_true", help="Disable recursive directory ingestion.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the generated RAG prompt.")
    return parser


def normalize_cli_command(args, parser: Optional[argparse.ArgumentParser] = None) -> None:
    """Normalize subcommand and legacy path-first CLI forms in-place."""
    first = args.command_or_path
    second = args.path_or_question

    if first in CLI_COMMANDS:
        args.command = first
        if args.command == "query":
            if second and args.question:
                _parser_error(parser, "query question was provided both positionally and with --question")
            args.path = None
            args.question = args.question or second
        else:
            args.path = second
    else:
        args.command = "oneshot"
        args.path = first

    if args.command in {"oneshot", "ingest"} and not args.path:
        _parser_error(parser, f"{args.command} requires a document path")
    if args.command == "ingest" and args.question:
        _parser_error(parser, "ingest does not accept --question")


def _parser_error(parser: Optional[argparse.ArgumentParser], message: str) -> None:
    """Raise a parser error when available, otherwise ValueError for tests."""
    if parser is not None:
        parser.error(message)
    raise ValueError(message)


def parse_metadata_filter(raw_filter: Optional[str]) -> Optional[Dict]:
    """Parse a JSON metadata filter."""
    if not raw_filter:
        return None

    try:
        metadata_filter = json.loads(raw_filter)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid metadata filter JSON: {e}") from e

    if not isinstance(metadata_filter, dict):
        raise ValueError("metadata_filter must be a JSON object")

    return metadata_filter


def parse_acl_values(raw_values: Optional[List[str]]) -> List[str]:
    """Parse repeated or comma-separated ACL subjects into a stable list."""
    values = []
    for raw_value in raw_values or []:
        values.extend(part.strip() for part in str(raw_value).split(","))
    return normalize_acl_values(values)


def build_cli_metadata_filter(
    client_filter: Optional[Dict],
    principal: Optional[str] = None,
    allowed_acl: Optional[List[str]] = None,
) -> Optional[Dict]:
    """Build the effective CLI metadata filter, failing closed when ACL is enabled."""
    parsed_acl = normalize_acl_values(allowed_acl or [])
    if not Config.ACL_ENABLED and not parsed_acl:
        return client_filter

    if not parsed_acl:
        Config.validate_acl()
        resolver = create_acl_resolver_from_config(Config)
        parsed_acl = resolver.allowed_acl_for_principal(principal)

    return build_effective_metadata_filter(
        client_filter,
        parsed_acl,
        metadata_key=Config.ACL_METADATA_KEY,
        default_deny=Config.ACL_DEFAULT_DENY,
    )


def apply_acl_metadata(chunks: List[Document], acl: Optional[List[str]]) -> None:
    """Denormalize trusted ACL subjects into chunk metadata for payload filtering."""
    acl_values = normalize_acl_values(acl or [])
    if acl_values:
        for chunk in chunks:
            chunk.metadata[Config.ACL_METADATA_KEY] = list(acl_values)
        return

    if not Config.ACL_INGEST_BINDINGS_ENABLED:
        return

    Config.validate_acl()
    resolver = create_acl_resolver_from_config(Config)
    apply_acl_metadata_from_bindings(chunks, resolver)


def apply_acl_metadata_from_bindings(chunks: List[Document], acl_binding_resolver) -> None:
    """Denormalize MySQL document ACL bindings into chunk metadata."""
    source_acl_cache = {}
    for chunk in chunks:
        source = chunk.metadata.get("source")
        if not source:
            raise ValueError("Document source metadata is required for ACL binding lookup.")
        source_key = str(source)
        if source_key not in source_acl_cache:
            if not hasattr(acl_binding_resolver, "acl_for_source"):
                raise ValueError("ACL binding resolver must implement acl_for_source(source).")
            source_acl_cache[source_key] = normalize_acl_values(
                acl_binding_resolver.acl_for_source(source_key),
                field_name="source_acl",
            )
            if not source_acl_cache[source_key]:
                raise ValueError(f"Document source has no ACL bindings: {source_key}")
        chunk.metadata[Config.ACL_METADATA_KEY] = list(source_acl_cache[source_key])


def validate_query_rewrite_values(
    provider: str,
    num_queries: int,
    temperature: float,
    weight_original: float,
    weight_variant: float,
) -> None:
    """Validate query rewrite values selected by CLI and config."""
    if provider not in {"deterministic", "chat"}:
        raise ValueError(f"Unsupported query rewrite provider: {provider}")
    if num_queries < 1:
        raise ValueError("QUERY_REWRITE_NUM_QUERIES must be greater than or equal to 1.")
    if temperature < 0:
        raise ValueError("QUERY_REWRITE_TEMPERATURE must be non-negative.")
    if weight_original < 0:
        raise ValueError("QUERY_REWRITE_WEIGHT_ORIGINAL must be non-negative.")
    if weight_variant < 0:
        raise ValueError("QUERY_REWRITE_WEIGHT_VARIANT must be non-negative.")
    if weight_original < weight_variant:
        raise ValueError("QUERY_REWRITE_WEIGHT_ORIGINAL must be greater than or equal to QUERY_REWRITE_WEIGHT_VARIANT.")
    if weight_original + weight_variant <= 0:
        raise ValueError("At least one query rewrite weight must be greater than 0.")


def build_loader_kwargs_from_config() -> Dict:
    """Build document-loader keyword arguments from environment configuration."""
    Config.validate_document_loading()
    return {
        "extract_tables": Config.PDF_EXTRACT_TABLES,
        "ocr_enabled": Config.PDF_OCR_ENABLED,
        "ocr_min_chars": Config.PDF_OCR_MIN_CHARS,
        "ocr_dpi": Config.PDF_OCR_DPI,
    }


def ingest_documents(
    path: str,
    clean: bool = True,
    recursive: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embedding_provider_name: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    vector_store_name: Optional[str] = None,
    parent_child_enabled: Optional[bool] = None,
    parent_chunk_size: Optional[int] = None,
    parent_chunk_overlap: Optional[int] = None,
    child_chunk_size: Optional[int] = None,
    child_chunk_overlap: Optional[int] = None,
    embedding_provider: Optional[EmbeddingProvider] = None,
    vector_store: Optional[VectorStore] = None,
    vector_store_overrides: Optional[Dict] = None,
    acl: Optional[List[str]] = None,
) -> IngestResult:
    """Load, split, embed, and upsert documents into a vector store."""
    use_parent_child = Config.PARENT_CHILD_ENABLED if parent_child_enabled is None else parent_child_enabled
    loader_kwargs = build_loader_kwargs_from_config()

    if use_parent_child:
        Config.validate_parent_child()
        hierarchical = load_and_split_documents_hierarchical(
            path,
            recursive=recursive,
            clean=clean,
            parent_chunk_size=parent_chunk_size if parent_chunk_size is not None else Config.PARENT_CHUNK_SIZE,
            parent_chunk_overlap=(
                parent_chunk_overlap if parent_chunk_overlap is not None else Config.PARENT_CHUNK_OVERLAP
            ),
            child_chunk_size=child_chunk_size if child_chunk_size is not None else Config.CHILD_CHUNK_SIZE,
            child_chunk_overlap=child_chunk_overlap if child_chunk_overlap is not None else Config.CHILD_CHUNK_OVERLAP,
            **loader_kwargs,
        )
        chunks = hierarchical.children
        add_parent_payload_to_children(chunks, hierarchical.parents)
        parent_count = len(hierarchical.parents)
    else:
        chunks = load_and_split_documents(
            path,
            recursive=recursive,
            clean=clean,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            **loader_kwargs,
        )
        parent_count = 0

    if not chunks:
        raise ValueError("No supported documents were loaded from the provided path")

    apply_acl_metadata(chunks, acl)
    assign_ingest_sequence(chunks)
    selected_embedding_provider = embedding_provider or create_embedding_provider(
        provider_name=embedding_provider_name,
        embedding_dimension=embedding_dimension,
    )
    embedded_chunks = selected_embedding_provider.embed_documents(chunks)
    vector_records = [embedded_document_to_record(document) for document in embedded_chunks]
    selected_vector_store = vector_store or create_vector_store(
        provider_name=vector_store_name,
        dimension=selected_embedding_provider.dimension,
        **(vector_store_overrides or {}),
    )
    ids = selected_vector_store.add_records(vector_records)

    logger.info(
        "Documents ingested | path=%s | records=%s | parents=%s | embedding_model=%s | embedding_dimension=%s | vector_store=%s",
        path,
        len(ids),
        parent_count,
        selected_embedding_provider.model_name,
        selected_embedding_provider.dimension,
        vector_store_name or Config.VECTOR_STORE_PROVIDER,
    )
    return IngestResult(
        ids=ids,
        records=vector_records,
        vector_store=selected_vector_store,
        embedding_provider=selected_embedding_provider,
        embedding_model=selected_embedding_provider.model_name,
        embedding_dimension=selected_embedding_provider.dimension,
        parent_count=parent_count,
    )


def build_rag_pipeline_from_index(
    chat_client=None,
    embedding_provider_name: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    vector_store_name: Optional[str] = None,
    vector_store: Optional[VectorStore] = None,
    vector_store_overrides: Optional[Dict] = None,
    rerank_provider_name: Optional[str] = None,
    rerank_fetch_k: Optional[int] = None,
    hybrid_enabled: Optional[bool] = None,
    hybrid_fetch_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
    hybrid_dense_weight: Optional[float] = None,
    hybrid_sparse_weight: Optional[float] = None,
    bm25_k1: Optional[float] = None,
    bm25_b: Optional[float] = None,
    parent_child_enabled: Optional[bool] = None,
    context_packing_enabled: Optional[bool] = None,
    context_dedup_enabled: Optional[bool] = None,
    context_near_dup_enabled: Optional[bool] = None,
    context_near_dup_threshold: Optional[float] = None,
    context_max_tokens: Optional[int] = None,
    tokenizer_encoding: Optional[str] = None,
    query_rewrite_enabled: Optional[bool] = None,
    query_rewrite_provider_name: Optional[str] = None,
    query_rewrite_fixture_path: Optional[str] = None,
    query_rewrite_num_queries: Optional[int] = None,
    query_rewrite_temperature: Optional[float] = None,
    query_rewrite_cache_enabled: Optional[bool] = None,
    query_rewrite_weight_original: Optional[float] = None,
    query_rewrite_weight_variant: Optional[float] = None,
    top_k: int = 5,
    max_context_chars: int = 4000,
    embedding_provider: Optional[EmbeddingProvider] = None,
) -> RAGPipeline:
    """Build a RAG pipeline from an existing vector-store index without corpus embedding."""
    selected_embedding_provider = embedding_provider or create_embedding_provider(
        provider_name=embedding_provider_name,
        embedding_dimension=embedding_dimension,
    )
    overrides = dict(vector_store_overrides or {})
    overrides["recreate"] = False
    selected_vector_store = vector_store or create_vector_store(
        provider_name=vector_store_name,
        dimension=selected_embedding_provider.dimension,
        **overrides,
    )
    vector_records = sort_vector_records(selected_vector_store.list_records())
    validate_index_embedding_profile(
        vector_records,
        selected_embedding_provider,
        collection_name=Config.VECTOR_STORE_COLLECTION,
    )
    return build_rag_pipeline_from_records(
        vector_records=vector_records,
        embedding_provider=selected_embedding_provider,
        vector_store=selected_vector_store,
        chat_client=chat_client,
        vector_store_name=vector_store_name,
        rerank_provider_name=rerank_provider_name,
        rerank_fetch_k=rerank_fetch_k,
        hybrid_enabled=hybrid_enabled,
        hybrid_fetch_k=hybrid_fetch_k,
        rrf_k=rrf_k,
        hybrid_dense_weight=hybrid_dense_weight,
        hybrid_sparse_weight=hybrid_sparse_weight,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        parent_child_enabled=parent_child_enabled,
        context_packing_enabled=context_packing_enabled,
        context_dedup_enabled=context_dedup_enabled,
        context_near_dup_enabled=context_near_dup_enabled,
        context_near_dup_threshold=context_near_dup_threshold,
        context_max_tokens=context_max_tokens,
        tokenizer_encoding=tokenizer_encoding,
        query_rewrite_enabled=query_rewrite_enabled,
        query_rewrite_provider_name=query_rewrite_provider_name,
        query_rewrite_fixture_path=query_rewrite_fixture_path,
        query_rewrite_num_queries=query_rewrite_num_queries,
        query_rewrite_temperature=query_rewrite_temperature,
        query_rewrite_cache_enabled=query_rewrite_cache_enabled,
        query_rewrite_weight_original=query_rewrite_weight_original,
        query_rewrite_weight_variant=query_rewrite_weight_variant,
        top_k=top_k,
        max_context_chars=max_context_chars,
    )


def build_rag_pipeline_from_records(
    vector_records: List[VectorRecord],
    embedding_provider: EmbeddingProvider,
    vector_store: VectorStore,
    chat_client=None,
    vector_store_name: Optional[str] = None,
    rerank_provider_name: Optional[str] = None,
    rerank_fetch_k: Optional[int] = None,
    hybrid_enabled: Optional[bool] = None,
    hybrid_fetch_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
    hybrid_dense_weight: Optional[float] = None,
    hybrid_sparse_weight: Optional[float] = None,
    bm25_k1: Optional[float] = None,
    bm25_b: Optional[float] = None,
    parent_child_enabled: Optional[bool] = None,
    parent_store: Optional[InMemoryParentStore] = None,
    context_packing_enabled: Optional[bool] = None,
    context_dedup_enabled: Optional[bool] = None,
    context_near_dup_enabled: Optional[bool] = None,
    context_near_dup_threshold: Optional[float] = None,
    context_max_tokens: Optional[int] = None,
    tokenizer_encoding: Optional[str] = None,
    query_rewrite_enabled: Optional[bool] = None,
    query_rewrite_provider_name: Optional[str] = None,
    query_rewrite_fixture_path: Optional[str] = None,
    query_rewrite_num_queries: Optional[int] = None,
    query_rewrite_temperature: Optional[float] = None,
    query_rewrite_cache_enabled: Optional[bool] = None,
    query_rewrite_weight_original: Optional[float] = None,
    query_rewrite_weight_variant: Optional[float] = None,
    top_k: int = 5,
    max_context_chars: int = 4000,
    acl: Optional[List[str]] = None,
) -> RAGPipeline:
    """Assemble a RAG pipeline from already indexed vector records."""
    embedding_provider = maybe_wrap_embedding_provider(embedding_provider)
    use_parent_child = Config.PARENT_CHILD_ENABLED if parent_child_enabled is None else parent_child_enabled
    use_context_dedup = Config.CONTEXT_DEDUP_ENABLED if context_dedup_enabled is None else context_dedup_enabled
    use_context_near_dup = (
        Config.CONTEXT_NEAR_DUP_ENABLED if context_near_dup_enabled is None else context_near_dup_enabled
    )
    requested_context_packing = (
        Config.CONTEXT_PACKING_ENABLED if context_packing_enabled is None else context_packing_enabled
    )
    use_context_packing = requested_context_packing or use_context_dedup or use_context_near_dup
    selected_context_threshold = (
        context_near_dup_threshold
        if context_near_dup_threshold is not None
        else Config.CONTEXT_NEAR_DUP_THRESHOLD
    )
    selected_context_max_tokens = context_max_tokens if context_max_tokens is not None else Config.CONTEXT_MAX_TOKENS
    selected_tokenizer_encoding = tokenizer_encoding or Config.TOKENIZER_ENCODING
    use_query_rewrite = Config.QUERY_REWRITE_ENABLED if query_rewrite_enabled is None else query_rewrite_enabled
    selected_query_rewrite_provider = query_rewrite_provider_name or Config.QUERY_REWRITE_PROVIDER
    selected_query_rewrite_fixture = query_rewrite_fixture_path or Config.QUERY_REWRITE_FIXTURE_PATH
    selected_query_rewrite_num_queries = (
        query_rewrite_num_queries if query_rewrite_num_queries is not None else Config.QUERY_REWRITE_NUM_QUERIES
    )
    selected_query_rewrite_temperature = (
        query_rewrite_temperature if query_rewrite_temperature is not None else Config.QUERY_REWRITE_TEMPERATURE
    )
    selected_query_rewrite_cache_enabled = (
        query_rewrite_cache_enabled
        if query_rewrite_cache_enabled is not None
        else Config.QUERY_REWRITE_CACHE_ENABLED
    )
    selected_query_rewrite_weight_original = (
        query_rewrite_weight_original
        if query_rewrite_weight_original is not None
        else Config.QUERY_REWRITE_WEIGHT_ORIGINAL
    )
    selected_query_rewrite_weight_variant = (
        query_rewrite_weight_variant
        if query_rewrite_weight_variant is not None
        else Config.QUERY_REWRITE_WEIGHT_VARIANT
    )

    if use_context_packing or use_context_dedup or use_context_near_dup:
        if selected_context_threshold < 0 or selected_context_threshold > 1:
            raise ValueError("CONTEXT_NEAR_DUP_THRESHOLD must be between 0 and 1.")
        if selected_context_max_tokens <= 0:
            raise ValueError("CONTEXT_MAX_TOKENS must be greater than 0.")
        validate_tokenizer_encoding(selected_tokenizer_encoding)

    if use_query_rewrite:
        validate_query_rewrite_values(
            provider=selected_query_rewrite_provider,
            num_queries=selected_query_rewrite_num_queries,
            temperature=selected_query_rewrite_temperature,
            weight_original=selected_query_rewrite_weight_original,
            weight_variant=selected_query_rewrite_weight_variant,
        )

    if use_parent_child and parent_store is None:
        Config.validate_parent_child()
        parent_store = rebuild_parent_store_from_records(vector_records)

    rerank_enabled = None
    if rerank_provider_name == "none":
        rerank_enabled = False
    elif rerank_provider_name is not None:
        rerank_enabled = True

    reranker = create_reranker(
        provider_name=rerank_provider_name,
        enabled=rerank_enabled,
        fetch_k=rerank_fetch_k,
        top_n=top_k,
    )
    use_hybrid = Config.HYBRID_ENABLED if hybrid_enabled is None else hybrid_enabled
    bm25_retriever = None
    rrf = None
    pipeline_fetch_k = rerank_fetch_k

    if use_hybrid:
        Config.validate_hybrid()
        selected_hybrid_fetch_k = hybrid_fetch_k or rerank_fetch_k or Config.RERANK_FETCH_K
        if rerank_fetch_k:
            selected_hybrid_fetch_k = max(selected_hybrid_fetch_k, rerank_fetch_k)
        pipeline_fetch_k = selected_hybrid_fetch_k
        bm25_retriever = BM25Retriever(
            vector_records,
            k1=bm25_k1 if bm25_k1 is not None else Config.BM25_K1,
            b=bm25_b if bm25_b is not None else Config.BM25_B,
        )
        rrf = ReciprocalRankFusion(
            RRFConfig(
                k=rrf_k if rrf_k is not None else Config.RRF_K,
                weights={
                    "dense": hybrid_dense_weight if hybrid_dense_weight is not None else Config.HYBRID_DENSE_WEIGHT,
                    "sparse": hybrid_sparse_weight if hybrid_sparse_weight is not None else Config.HYBRID_SPARSE_WEIGHT,
                },
            )
        )

    if chat_client is None:
        from api_client import APIClient

        Config.validate()
        chat_client = APIClient(Config.API_KEY, Config.API_URL)

    query_rewriter = create_query_rewriter(
        provider_name=selected_query_rewrite_provider,
        enabled=use_query_rewrite,
        num_queries=selected_query_rewrite_num_queries,
        temperature=selected_query_rewrite_temperature,
        cache_enabled=selected_query_rewrite_cache_enabled,
        weight_original=selected_query_rewrite_weight_original,
        weight_variant=selected_query_rewrite_weight_variant,
        fixture_path=selected_query_rewrite_fixture,
        chat_client=chat_client,
    )

    logger.info(
        "RAG pipeline ready | records=%s | parents=%s | embedding_provider=%s | embedding_dimension=%s | vector_store=%s | reranker=%s | hybrid=%s | parent_child=%s | context_packing=%s | context_dedup=%s | query_rewrite=%s",
        len(vector_records),
        parent_store.count() if parent_store is not None else 0,
        embedding_provider.model_name,
        embedding_provider.dimension,
        vector_store_name or Config.VECTOR_STORE_PROVIDER,
        reranker.model_name if reranker else None,
        use_hybrid,
        use_parent_child,
        use_context_packing,
        use_context_dedup,
        use_query_rewrite,
    )
    pipeline = RAGPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_client=chat_client,
        top_k=top_k,
        max_context_chars=max_context_chars,
        reranker=reranker,
        fetch_k=pipeline_fetch_k,
        bm25_retriever=bm25_retriever,
        rrf=rrf,
        parent_store=parent_store,
        context_packing_enabled=use_context_packing,
        context_dedup_enabled=use_context_dedup,
        context_near_dup_enabled=use_context_near_dup,
        context_near_dup_threshold=selected_context_threshold,
        context_max_tokens=selected_context_max_tokens if use_context_packing else None,
        tokenizer_encoding=selected_tokenizer_encoding,
        query_rewriter=query_rewriter,
        query_rewrite_enabled=use_query_rewrite,
        query_rewrite_num_queries=selected_query_rewrite_num_queries,
        query_rewrite_weight_original=selected_query_rewrite_weight_original,
        query_rewrite_weight_variant=selected_query_rewrite_weight_variant,
    )
    return maybe_wrap_pipeline(pipeline, vector_records)


def add_parent_payload_to_children(children: List[Document], parents: List[Document]) -> None:
    """Persist parent text on child metadata for query-time parent-store reconstruction."""
    parents_by_id = {str(parent.metadata.get("parent_id")): parent for parent in parents}
    for child in children:
        parent_id = str(child.metadata.get("parent_id"))
        parent = parents_by_id.get(parent_id)
        if parent is None:
            raise ValueError(f"Parent metadata is missing for child parent_id={parent_id}")
        child.metadata["parent_content"] = parent.content
        child.metadata["parent_source"] = parent.metadata.get("source")


def assign_ingest_sequence(chunks: List[Document]) -> None:
    """Add deterministic ingest order metadata without affecting stable record ids."""
    for index, chunk in enumerate(chunks):
        chunk.metadata["ingest_sequence"] = index


def rebuild_parent_store_from_records(vector_records: List[VectorRecord]) -> InMemoryParentStore:
    """Rebuild the in-memory parent store from persisted child record metadata."""
    parent_documents = []
    seen_parent_ids = set()

    for record in vector_records:
        metadata = dict(record.metadata)
        parent_id = metadata.get("parent_id")
        if not parent_id:
            continue
        parent_id = str(parent_id)
        if parent_id in seen_parent_ids:
            continue

        parent_content = metadata.get("parent_content")
        if not parent_content:
            raise ValueError(
                f"Parent-child query requested but parent content is missing for parent_id={parent_id}"
            )

        parent_metadata = dict(metadata)
        parent_metadata["parent_id"] = parent_id
        parent_metadata["source"] = metadata.get("parent_source") or metadata.get("source")
        parent_metadata["start_char"] = metadata.get("parent_start_char")
        parent_metadata["end_char"] = metadata.get("parent_end_char")
        parent_documents.append(Document(content=str(parent_content), metadata=parent_metadata))
        seen_parent_ids.add(parent_id)

    if not parent_documents:
        raise ValueError("Parent-child query requested but the vector index has no parent metadata.")

    parent_store = InMemoryParentStore()
    parent_store.add_parents(parent_documents)
    return parent_store


def validate_index_embedding_profile(
    vector_records: List[VectorRecord],
    embedding_provider: EmbeddingProvider,
    collection_name: str,
) -> None:
    """Validate that query embeddings match the indexed embedding space."""
    if not vector_records:
        raise IndexNotReadyError(f"Vector store collection is empty or unavailable: {collection_name}")

    models = {str(record.metadata.get("embedding_model")) for record in vector_records if record.metadata.get("embedding_model")}
    dimensions = {
        int(record.metadata.get("embedding_dimension"))
        for record in vector_records
        if record.metadata.get("embedding_dimension") is not None
    }

    if not models:
        raise EmbeddingSpaceInvalidError("Vector index records are missing embedding_model metadata.")
    if not dimensions:
        raise EmbeddingSpaceInvalidError("Vector index records are missing embedding_dimension metadata.")
    if len(models) > 1:
        raise EmbeddingSpaceInvalidError(f"Vector index contains mixed embedding models: {sorted(models)}")
    if len(dimensions) > 1:
        raise EmbeddingSpaceInvalidError(f"Vector index contains mixed embedding dimensions: {sorted(dimensions)}")

    indexed_model = next(iter(models))
    indexed_dimension = next(iter(dimensions))
    if indexed_model != embedding_provider.model_name:
        raise EmbeddingSpaceMismatchError(
            f"Embedding model mismatch for collection {collection_name}: "
            f"index={indexed_model}, query={embedding_provider.model_name}"
        )
    if indexed_dimension != embedding_provider.dimension:
        raise EmbeddingSpaceMismatchError(
            f"Embedding dimension mismatch for collection {collection_name}: "
            f"index={indexed_dimension}, query={embedding_provider.dimension}"
        )


def sort_vector_records(vector_records: List[VectorRecord]) -> List[VectorRecord]:
    """Sort records by persisted ingest order with a stable metadata fallback."""
    return sorted(vector_records, key=vector_record_sort_key)


def vector_record_sort_key(record: VectorRecord) -> tuple:
    """Return a deterministic sort key for records read from a persistent store."""
    metadata = record.metadata
    return (
        numeric_sort_value(metadata.get("ingest_sequence")),
        str(metadata.get("source", "")),
        numeric_sort_value(metadata.get("chunk_index")),
        numeric_sort_value(metadata.get("start_char")),
        numeric_sort_value(metadata.get("end_char")),
        record.id,
    )


def numeric_sort_value(value) -> int:
    """Return an integer sort value, placing missing values after present values."""
    if value is None:
        return 2**63 - 1
    try:
        return int(value)
    except (TypeError, ValueError):
        return 2**63 - 1


def build_rag_pipeline_from_path(
    path: str,
    chat_client=None,
    clean: bool = True,
    recursive: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    embedding_provider_name: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
    vector_store_name: Optional[str] = None,
    rerank_provider_name: Optional[str] = None,
    rerank_fetch_k: Optional[int] = None,
    hybrid_enabled: Optional[bool] = None,
    hybrid_fetch_k: Optional[int] = None,
    rrf_k: Optional[int] = None,
    hybrid_dense_weight: Optional[float] = None,
    hybrid_sparse_weight: Optional[float] = None,
    bm25_k1: Optional[float] = None,
    bm25_b: Optional[float] = None,
    parent_child_enabled: Optional[bool] = None,
    parent_chunk_size: Optional[int] = None,
    parent_chunk_overlap: Optional[int] = None,
    child_chunk_size: Optional[int] = None,
    child_chunk_overlap: Optional[int] = None,
    context_packing_enabled: Optional[bool] = None,
    context_dedup_enabled: Optional[bool] = None,
    context_near_dup_enabled: Optional[bool] = None,
    context_near_dup_threshold: Optional[float] = None,
    context_max_tokens: Optional[int] = None,
    tokenizer_encoding: Optional[str] = None,
    query_rewrite_enabled: Optional[bool] = None,
    query_rewrite_provider_name: Optional[str] = None,
    query_rewrite_fixture_path: Optional[str] = None,
    query_rewrite_num_queries: Optional[int] = None,
    query_rewrite_temperature: Optional[float] = None,
    query_rewrite_cache_enabled: Optional[bool] = None,
    query_rewrite_weight_original: Optional[float] = None,
    query_rewrite_weight_variant: Optional[float] = None,
    top_k: int = 5,
    max_context_chars: int = 4000,
    acl: Optional[List[str]] = None,
) -> RAGPipeline:
    """Ingest documents and build a ready-to-query RAG pipeline."""
    ingest_result = ingest_documents(
        path=path,
        clean=clean,
        recursive=recursive,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_provider_name=embedding_provider_name,
        embedding_dimension=embedding_dimension,
        vector_store_name=vector_store_name,
        parent_child_enabled=parent_child_enabled,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        acl=acl,
    )
    return build_rag_pipeline_from_records(
        vector_records=ingest_result.records,
        embedding_provider=ingest_result.embedding_provider,
        vector_store=ingest_result.vector_store,
        chat_client=chat_client,
        vector_store_name=vector_store_name,
        rerank_provider_name=rerank_provider_name,
        rerank_fetch_k=rerank_fetch_k,
        hybrid_enabled=hybrid_enabled,
        hybrid_fetch_k=hybrid_fetch_k,
        rrf_k=rrf_k,
        hybrid_dense_weight=hybrid_dense_weight,
        hybrid_sparse_weight=hybrid_sparse_weight,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        parent_child_enabled=parent_child_enabled,
        context_packing_enabled=context_packing_enabled,
        context_dedup_enabled=context_dedup_enabled,
        context_near_dup_enabled=context_near_dup_enabled,
        context_near_dup_threshold=context_near_dup_threshold,
        context_max_tokens=context_max_tokens,
        tokenizer_encoding=tokenizer_encoding,
        query_rewrite_enabled=query_rewrite_enabled,
        query_rewrite_provider_name=query_rewrite_provider_name,
        query_rewrite_fixture_path=query_rewrite_fixture_path,
        query_rewrite_num_queries=query_rewrite_num_queries,
        query_rewrite_temperature=query_rewrite_temperature,
        query_rewrite_cache_enabled=query_rewrite_cache_enabled,
        query_rewrite_weight_original=query_rewrite_weight_original,
        query_rewrite_weight_variant=query_rewrite_weight_variant,
        top_k=top_k,
        max_context_chars=max_context_chars,
    )

def create_embedding_provider(
    provider_name: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
):
    """Create an embedding provider for the RAG CLI."""
    provider = (provider_name or Config.EMBEDDING_PROVIDER).lower()

    if provider == "openai":
        Config.validate_embedding()
        return maybe_wrap_embedding_provider(OpenAIEmbeddingProvider(dimensions=embedding_dimension))

    if provider == "hash":
        return maybe_wrap_embedding_provider(HashEmbeddingProvider(dimension=embedding_dimension or 128))

    raise ValueError(f"Unsupported embedding provider: {provider}")


def run_single_question(
    pipeline: RAGPipeline,
    question: str,
    top_k: Optional[int] = None,
    metadata_filter: Optional[Dict] = None,
) -> RAGResponse:
    """Run a single RAG question."""
    return pipeline.answer(question, top_k=top_k, metadata_filter=metadata_filter)


def format_response(response: RAGResponse, show_prompt: bool = False) -> str:
    """Format a RAG response for terminal output."""
    sections = [
        "Answer",
        response.answer,
        "",
        "Sources",
        format_sources(response.sources),
    ]

    if show_prompt:
        sections.extend(["", "Prompt", response.prompt])

    return "\n".join(sections).strip()


def format_sources(sources: List) -> str:
    """Format retrieved sources for terminal output."""
    if not sources:
        return "No sources returned."

    lines = []
    for source in sources:
        metadata = source.metadata
        source_path = metadata.get("source", "unknown source")
        chunk_index = metadata.get("chunk_index", "unknown")
        lines.append(f"[{source.index}] score={source.score:.4f} source={source_path} chunk={chunk_index}")

    return "\n".join(lines)


def interactive_loop(
    pipeline: RAGPipeline,
    top_k: int,
    metadata_filter: Optional[Dict] = None,
    show_prompt: bool = False,
) -> None:
    """Run an interactive terminal loop."""
    print("RAG CLI is ready. Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue

        response = run_single_question(
            pipeline,
            question,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        print("\n" + format_response(response, show_prompt=show_prompt))


def main(argv=None) -> int:
    """Run the RAG CLI."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "ingest":
            ingest_result = ingest_documents(
                args.path,
                clean=not args.no_clean,
                recursive=not args.non_recursive,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                embedding_provider_name=args.embedding_provider,
                embedding_dimension=args.embedding_dimension,
                vector_store_name=args.vector_store,
                parent_child_enabled=True if args.parent_child else None,
                parent_chunk_size=args.parent_chunk_size,
                parent_chunk_overlap=args.parent_chunk_overlap,
                child_chunk_size=args.child_chunk_size,
                child_chunk_overlap=args.child_chunk_overlap,
                acl=parse_acl_values(args.acl),
            )
            print(
                "Ingested "
                f"{len(ingest_result.ids)} records | "
                f"embedding_model={ingest_result.embedding_model} | "
                f"embedding_dimension={ingest_result.embedding_dimension} | "
                f"parents={ingest_result.parent_count}"
            )
            return 0

        metadata_filter = build_cli_metadata_filter(
            parse_metadata_filter(args.metadata_filter),
            principal=args.principal,
            allowed_acl=parse_acl_values(args.acl),
        )
        if args.command == "query":
            pipeline = build_rag_pipeline_from_index(
                embedding_provider_name=args.embedding_provider,
                embedding_dimension=args.embedding_dimension,
                vector_store_name=args.vector_store,
                rerank_provider_name=args.rerank_provider,
                rerank_fetch_k=args.rerank_fetch_k,
                hybrid_enabled=True if args.hybrid else None,
                hybrid_fetch_k=args.hybrid_fetch_k,
                rrf_k=args.rrf_k,
                hybrid_dense_weight=args.hybrid_dense_weight,
                hybrid_sparse_weight=args.hybrid_sparse_weight,
                bm25_k1=args.bm25_k1,
                bm25_b=args.bm25_b,
                parent_child_enabled=True if args.parent_child else None,
                context_packing_enabled=True if args.context_packing else None,
                context_dedup_enabled=True if args.context_dedup else None,
                context_near_dup_enabled=True if args.context_near_dup else None,
                context_near_dup_threshold=args.context_near_dup_threshold,
                context_max_tokens=args.context_max_tokens,
                tokenizer_encoding=args.tokenizer_encoding,
                query_rewrite_enabled=True if args.multi_query else None,
                query_rewrite_provider_name=args.query_rewrite_provider,
                query_rewrite_fixture_path=args.query_rewrite_fixture,
                query_rewrite_num_queries=args.query_rewrite_num_queries,
                query_rewrite_temperature=args.query_rewrite_temperature,
                query_rewrite_cache_enabled=False if args.no_query_rewrite_cache else None,
                query_rewrite_weight_original=args.query_rewrite_weight_original,
                query_rewrite_weight_variant=args.query_rewrite_weight_variant,
                top_k=args.top_k,
                max_context_chars=args.max_context_chars,
            )
        else:
            pipeline = build_rag_pipeline_from_path(
                args.path,
                clean=not args.no_clean,
                recursive=not args.non_recursive,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                embedding_provider_name=args.embedding_provider,
                embedding_dimension=args.embedding_dimension,
                vector_store_name=args.vector_store,
                rerank_provider_name=args.rerank_provider,
                rerank_fetch_k=args.rerank_fetch_k,
                hybrid_enabled=True if args.hybrid else None,
                hybrid_fetch_k=args.hybrid_fetch_k,
                rrf_k=args.rrf_k,
                hybrid_dense_weight=args.hybrid_dense_weight,
                hybrid_sparse_weight=args.hybrid_sparse_weight,
                bm25_k1=args.bm25_k1,
                bm25_b=args.bm25_b,
                parent_child_enabled=True if args.parent_child else None,
                parent_chunk_size=args.parent_chunk_size,
                parent_chunk_overlap=args.parent_chunk_overlap,
                child_chunk_size=args.child_chunk_size,
                child_chunk_overlap=args.child_chunk_overlap,
                context_packing_enabled=True if args.context_packing else None,
                context_dedup_enabled=True if args.context_dedup else None,
                context_near_dup_enabled=True if args.context_near_dup else None,
                context_near_dup_threshold=args.context_near_dup_threshold,
                context_max_tokens=args.context_max_tokens,
                tokenizer_encoding=args.tokenizer_encoding,
                query_rewrite_enabled=True if args.multi_query else None,
                query_rewrite_provider_name=args.query_rewrite_provider,
                query_rewrite_fixture_path=args.query_rewrite_fixture,
                query_rewrite_num_queries=args.query_rewrite_num_queries,
                query_rewrite_temperature=args.query_rewrite_temperature,
                query_rewrite_cache_enabled=False if args.no_query_rewrite_cache else None,
                query_rewrite_weight_original=args.query_rewrite_weight_original,
                query_rewrite_weight_variant=args.query_rewrite_weight_variant,
                top_k=args.top_k,
                max_context_chars=args.max_context_chars,
                acl=parse_acl_values(args.acl),
            )

        if args.question:
            response = run_single_question(
                pipeline,
                args.question,
                top_k=args.top_k,
                metadata_filter=metadata_filter,
            )
            print(format_response(response, show_prompt=args.show_prompt))
        else:
            interactive_loop(
                pipeline,
                top_k=args.top_k,
                metadata_filter=metadata_filter,
                show_prompt=args.show_prompt,
            )

        return 0

    except Exception as e:
        logger.exception("RAG CLI failed | error=%s", e)
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
