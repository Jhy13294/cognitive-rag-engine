import random
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import requests

from logger import mask_sensitive_info, setup_logger

from .base import RerankConfig, RerankResult, Reranker, RerankerError

if TYPE_CHECKING:
    from rag import RetrievedSource

logger = setup_logger(__name__)

RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class _RetryableRerankError(RerankerError):
    """Internal marker for failures that should be retried."""


class CohereReranker(Reranker):
    """Cohere Rerank API provider with retry handling and mockable session."""

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.cohere.com/v2/rerank",
        model_name: str = "rerank-v3.5",
        top_n: int = 5,
        fetch_k: int = 30,
        batch_size: int = 32,
        timeout: float = 30.0,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        session: Optional[Any] = None,
    ):
        """Initialize the Cohere reranker."""
        if not api_key:
            raise ValueError("api_key is required")

        super().__init__(
            RerankConfig(
                model_name=model_name,
                top_n=top_n,
                fetch_k=fetch_k,
                batch_size=batch_size,
                metadata={"provider": "cohere", "api_url": api_url},
            )
        )
        self.api_key = api_key
        self.api_url = api_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.session = session or requests.Session()

        logger.info(
            "CohereReranker initialized | model=%s | top_n=%s | fetch_k=%s | api_key=%s",
            self.model_name,
            self.top_n,
            self.fetch_k,
            mask_sensitive_info(api_key),
        )

    def rerank(
        self,
        query: str,
        candidates: Sequence["RetrievedSource"],
        top_n: Optional[int] = None,
    ) -> List[RerankResult]:
        """Rerank candidates through Cohere Rerank."""
        candidate_list = list(candidates)
        limit = min(top_n or self.top_n, len(candidate_list))
        if limit <= 0:
            return []

        results = []
        for batch_start in range(0, len(candidate_list), self.batch_size):
            batch_candidates = candidate_list[batch_start : batch_start + self.batch_size]
            payload = {
                "model": self.model_name,
                "query": query,
                "documents": [candidate.content for candidate in batch_candidates],
                "top_n": len(batch_candidates),
            }
            response_data = self._post_with_retries(payload)
            results.extend(self._parse_results(response_data, candidate_list, batch_start))

        results.sort(key=lambda result: (-result.score, result.index))
        return results[:limit]

    def _parse_results(
        self,
        response_data: Dict,
        candidates: Sequence["RetrievedSource"],
        batch_start: int,
    ) -> List[RerankResult]:
        """Parse provider response indexes into original candidate indexes."""
        results = []
        for item in response_data.get("results", []):
            local_index = int(item["index"])
            original_index = batch_start + local_index
            if original_index < 0 or original_index >= len(candidates):
                continue
            score = float(item.get("relevance_score", item.get("score", 0.0)))
            candidate = candidates[original_index]
            results.append(
                RerankResult(
                    index=original_index,
                    score=score,
                    content=candidate.content,
                    metadata={
                        "rerank_provider": "cohere",
                        "dense_score": candidate.score,
                        "dense_rank": original_index + 1,
                    },
                )
            )
        return results

    def _post_with_retries(self, payload: Dict) -> Dict:
        """Send a rerank request with exponential backoff."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in RETRYABLE_STATUS_CODES:
                    raise _RetryableRerankError(f"Retryable Cohere rerank error: HTTP {response.status_code}")
                if response.status_code >= 400:
                    raise RerankerError(f"Cohere rerank error: HTTP {response.status_code}: {response.text}")

                return response.json()

            except (requests.Timeout, requests.ConnectionError, _RetryableRerankError) as e:
                last_error = e
                if attempt >= self.max_retries:
                    raise RerankerError(f"Cohere rerank request failed: {e}") from e

                delay = self._retry_delay(attempt)
                logger.warning(
                    "Retrying Cohere rerank request | attempt=%s | delay=%.2fs | error=%s",
                    attempt + 1,
                    delay,
                    e,
                )
                time.sleep(delay)

        raise RerankerError(f"Cohere rerank request failed: {last_error}")

    def _retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = min(self.max_delay, self.base_delay * (2**attempt))
        jitter = random.uniform(0.0, min(0.25, delay * 0.1))
        return delay + jitter
