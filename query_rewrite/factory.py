from typing import Optional

from .base import QueryRewriteConfig, QueryRewriter
from .chat import ChatQueryRewriter, ChatRewriteClient
from .deterministic import DeterministicQueryRewriter


def create_query_rewriter(
    provider_name: str = "deterministic",
    enabled: bool = False,
    num_queries: int = 3,
    temperature: float = 0.1,
    cache_enabled: bool = True,
    weight_original: float = 1.0,
    weight_variant: float = 0.7,
    fixture_path: str = "eval/fixtures/query_rewrites.jsonl",
    chat_client: Optional[ChatRewriteClient] = None,
) -> Optional[QueryRewriter]:
    """Create a query rewriter from configuration."""
    if not enabled:
        return None

    provider = (provider_name or "deterministic").lower()
    config = QueryRewriteConfig(
        provider=provider,
        num_queries=num_queries,
        temperature=temperature,
        cache_enabled=cache_enabled,
        weight_original=weight_original,
        weight_variant=weight_variant,
        fixture_path=fixture_path,
    )

    if provider == "deterministic":
        return DeterministicQueryRewriter(config)

    if provider == "chat":
        if chat_client is None:
            raise ValueError("chat_client is required for ChatQueryRewriter.")
        return ChatQueryRewriter(config=config, chat_client=chat_client)

    raise ValueError(f"Unsupported query rewrite provider: {provider}")
