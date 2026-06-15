from typing import Optional

from config import Config

from .base import Reranker
from .cohere_provider import CohereReranker
from .deterministic import DeterministicReranker


def create_reranker(
    provider_name: Optional[str] = None,
    enabled: Optional[bool] = None,
    **overrides,
) -> Optional[Reranker]:
    """Create an optional reranker from configuration."""
    is_enabled = Config.RERANK_ENABLED if enabled is None else enabled
    provider = (provider_name or Config.RERANK_PROVIDER).lower()

    if not is_enabled or provider in {"none", "off", "disabled"}:
        return None

    if provider == "deterministic":
        return DeterministicReranker(
            model_name=_override(overrides, "model_name", Config.RERANK_MODEL),
            top_n=_override(overrides, "top_n", Config.RERANK_TOP_N),
            fetch_k=_override(overrides, "fetch_k", Config.RERANK_FETCH_K),
            batch_size=_override(overrides, "batch_size", Config.RERANK_BATCH_SIZE),
        )

    if provider == "cohere":
        Config.validate_rerank(provider)
        return CohereReranker(
            api_key=_override(overrides, "api_key", Config.RERANK_API_KEY),
            api_url=_override(overrides, "api_url", Config.RERANK_API_URL),
            model_name=_override(overrides, "model_name", Config.RERANK_MODEL),
            top_n=_override(overrides, "top_n", Config.RERANK_TOP_N),
            fetch_k=_override(overrides, "fetch_k", Config.RERANK_FETCH_K),
            batch_size=_override(overrides, "batch_size", Config.RERANK_BATCH_SIZE),
            timeout=_override(overrides, "timeout", Config.RERANK_TIMEOUT),
            max_retries=_override(overrides, "max_retries", Config.RERANK_MAX_RETRIES),
            base_delay=_override(overrides, "base_delay", Config.RERANK_BASE_DELAY),
            max_delay=_override(overrides, "max_delay", Config.RERANK_MAX_DELAY),
            session=_override(overrides, "session", None),
        )

    raise ValueError(f"Unsupported rerank provider: {provider}")


def _override(overrides: dict, key: str, default):
    """Return an override value when it is explicitly provided."""
    value = overrides.get(key)
    return default if value is None else value

