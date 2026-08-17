import json
from typing import Dict, List, Protocol

from logger import setup_logger

from .base import QueryRewriteConfig, QueryRewriter, normalize_query_variants

logger = setup_logger(__name__)


class ChatRewriteClient(Protocol):
    """Protocol for chat clients used by ChatQueryRewriter."""

    def chat(self, message: str, system_prompt: str = None) -> Dict:
        """Send a chat request and return a provider response."""
        ...


class ChatQueryRewriter(QueryRewriter):
    """Production query rewriter backed by a chat-completion client."""

    SYSTEM_PROMPT = (
        "You rewrite enterprise knowledge-base search queries. "
        "Return only a JSON array of short search query strings. "
        "Preserve product names, policy codes, and acronyms exactly."
    )

    def __init__(self, config: QueryRewriteConfig, chat_client: ChatRewriteClient):
        """Initialize a chat-backed query rewriter."""
        super().__init__(config)
        self.chat_client = chat_client
        self._cache: Dict[str, List[str]] = {}

    @property
    def model_name(self) -> str:
        """Return the production provider name."""
        return "chat-query-rewriter"

    def rewrite(self, question: str) -> List[str]:
        """Return chat-generated query variants, falling back to the original query on failure."""
        cache_key = self._cache_key(question)
        if self.config.cache_enabled and cache_key in self._cache:
            return list(self._cache[cache_key])

        try:
            prompt = self._build_prompt(question)
            response = self.chat_client.chat(prompt, system_prompt=self.SYSTEM_PROMPT)
            raw_text = _extract_chat_content(response)
            parsed_variants = _parse_variants(raw_text)
            variants = normalize_query_variants(question, parsed_variants, self.config.num_queries)
            if len(variants) <= 1:
                logger.warning(
                    "Chat query rewrite returned no variants; using original query | question=%s",
                    question,
                )
                variants = [str(question).strip()]
        except Exception as e:
            logger.warning(
                "Chat query rewrite failed; using original query | error=%s | question=%s",
                e,
                question,
            )
            variants = [str(question).strip()]

        if self.config.cache_enabled:
            self._cache[cache_key] = list(variants)

        logger.info(
            "Query rewritten | provider=%s | original=%s | variants=%s",
            self.model_name,
            question,
            variants,
        )
        return variants

    def _build_prompt(self, question: str) -> str:
        """Build the chat prompt for query rewriting."""
        variant_count = max(self.config.num_queries - 1, 0)
        return (
            f"Original question:\n{question}\n\n"
            f"Generate {variant_count} alternative search queries. "
            "The original question is already included by the caller, so do not repeat it. "
            "Return a JSON array only."
        )

    def _cache_key(self, question: str) -> str:
        """Return a configuration-scoped cache key."""
        return "|".join(
            [
                self.model_name,
                str(self.config.num_queries),
                f"{self.config.temperature:.3f}",
                "cache-v1",
                " ".join(str(question).lower().split()),
            ]
        )


def _extract_chat_content(response: Dict) -> str:
    """Extract assistant text from a chat-completion compatible response."""
    return str(response["choices"][0]["message"]["content"])


def _parse_variants(raw_text: str) -> List[str]:
    """Parse query variants from JSON or newline-delimited text."""
    text = raw_text.strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        if isinstance(parsed, dict):
            values = parsed.get("rewrites", parsed.get("queries", []))
            return [str(item) for item in values] if isinstance(values, list) else []
    except json.JSONDecodeError:
        pass

    variants = []
    for line in text.splitlines():
        cleaned = line.strip().lstrip("-*0123456789. ").strip()
        if cleaned:
            variants.append(cleaned)
    return variants
