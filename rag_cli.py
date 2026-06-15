import argparse
import json
import logging
from typing import Dict, List, Optional

from config import Config
from document_loader import load_and_split_documents
from embeddings import HashEmbeddingProvider, OpenAIEmbeddingProvider
from logger import setup_logger
from rag import RAGPipeline, RAGResponse
from rerank import create_reranker
from vector_store import create_vector_store

logger = setup_logger(__name__, level=logging.INFO)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the RAG CLI argument parser."""
    parser = argparse.ArgumentParser(description="Run a minimal local RAG workflow.")
    parser.add_argument("path", help="Document file or directory to ingest.")
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
    parser.add_argument("--max-context-chars", type=int, default=4000, help="Maximum context characters.")
    parser.add_argument("--metadata-filter", help="JSON exact-match metadata filter, for example '{\"file_type\":\"txt\"}'.")
    parser.add_argument("--no-clean", action="store_true", help="Disable text cleaning before chunking.")
    parser.add_argument("--non-recursive", action="store_true", help="Disable recursive directory ingestion.")
    parser.add_argument("--show-prompt", action="store_true", help="Print the generated RAG prompt.")
    return parser


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
    top_k: int = 5,
    max_context_chars: int = 4000,
) -> RAGPipeline:
    """Ingest documents and build a ready-to-query RAG pipeline."""
    chunks = load_and_split_documents(
        path,
        recursive=recursive,
        clean=clean,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError("No supported documents were loaded from the provided path")

    embedding_provider = create_embedding_provider(
        provider_name=embedding_provider_name,
        embedding_dimension=embedding_dimension,
    )
    embedded_chunks = embedding_provider.embed_documents(chunks)

    vector_store = create_vector_store(
        provider_name=vector_store_name,
        dimension=embedding_provider.dimension,
    )
    vector_store.add_documents(embedded_chunks)
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

    if chat_client is None:
        from api_client import APIClient
        from config import Config

        Config.validate()
        chat_client = APIClient(Config.API_KEY, Config.API_URL)

    logger.info(
        "RAG pipeline ready | chunks=%s | embedding_provider=%s | embedding_dimension=%s | vector_store=%s | reranker=%s",
        len(chunks),
        embedding_provider.model_name,
        embedding_provider.dimension,
        vector_store_name or Config.VECTOR_STORE_PROVIDER,
        reranker.model_name if reranker else None,
    )
    return RAGPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_client=chat_client,
        top_k=top_k,
        max_context_chars=max_context_chars,
        reranker=reranker,
        fetch_k=rerank_fetch_k,
    )


def create_embedding_provider(
    provider_name: Optional[str] = None,
    embedding_dimension: Optional[int] = None,
):
    """Create an embedding provider for the RAG CLI."""
    provider = (provider_name or Config.EMBEDDING_PROVIDER).lower()

    if provider == "openai":
        Config.validate_embedding()
        return OpenAIEmbeddingProvider(dimensions=embedding_dimension)

    if provider == "hash":
        return HashEmbeddingProvider(dimension=embedding_dimension or 128)

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
        metadata_filter = parse_metadata_filter(args.metadata_filter)
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
            top_k=args.top_k,
            max_context_chars=args.max_context_chars,
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
