import json
from pathlib import Path
from typing import Dict, List

from logger import setup_logger

from .base import QueryRewriteConfig, QueryRewriter, normalize_query_variants

logger = setup_logger(__name__)


class DeterministicQueryRewriter(QueryRewriter):
    """Offline query rewriter backed by a frozen JSONL fixture."""

    def __init__(self, config: QueryRewriteConfig):
        """Load deterministic rewrites from a fixture file."""
        super().__init__(config)
        self._rewrites = self._load_fixture(config.fixture_path)
        self._cache: Dict[str, List[str]] = {}

    @property
    def model_name(self) -> str:
        """Return the deterministic provider name."""
        return "deterministic-query-rewriter"

    def rewrite(self, question: str) -> List[str]:
        """Return fixture-backed query variants for a question."""
        cache_key = self._cache_key(question)
        if self.config.cache_enabled and cache_key in self._cache:
            return list(self._cache[cache_key])

        fixture_variants = self._rewrites.get(_question_key(question), [])
        variants = normalize_query_variants(question, fixture_variants, self.config.num_queries)

        if self.config.cache_enabled:
            self._cache[cache_key] = list(variants)

        logger.info("Query rewritten | provider=%s | original=%s | variants=%s", self.model_name, question, variants)
        return variants

    def _cache_key(self, question: str) -> str:
        """Return a configuration-scoped cache key."""
        return "|".join(
            [
                self.model_name,
                str(self.config.num_queries),
                f"{self.config.temperature:.3f}",
                str(self.config.fixture_path),
                _question_key(question),
            ]
        )

    def _load_fixture(self, fixture_path: str) -> Dict[str, List[str]]:
        """Load a JSONL fixture mapping questions to rewrite variants."""
        path = Path(fixture_path)
        if not path.exists():
            logger.warning("Query rewrite fixture not found | path=%s", fixture_path)
            return {}

        rewrites: Dict[str, List[str]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                question = str(payload.get("question", "")).strip()
                raw_variants = payload.get("rewrites", payload.get("variants", []))
                if not question:
                    raise ValueError(f"Missing question in query rewrite fixture line {line_number}")
                if not isinstance(raw_variants, list):
                    raise ValueError(f"Query rewrite fixture line {line_number} must contain a list of rewrites")
                rewrites[_question_key(question)] = [str(item) for item in raw_variants]

        return rewrites


def _question_key(question: str) -> str:
    """Return a stable fixture lookup key."""
    return " ".join(str(question).lower().split())
