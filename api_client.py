import logging
import random
import time
from typing import Dict, Optional, Set

import requests

from config import Config
from logger import mask_sensitive_info, setup_logger

logger = setup_logger(__name__, level=logging.INFO)


class APIClient:
    """HTTP client for chat-completion compatible LLM APIs."""

    RETRYABLE_STATUS_CODES: Set[int] = {429, 500, 502, 503, 504}

    MAX_RETRIES: int = 3
    BASE_DELAY: float = 1.0
    MAX_DELAY: float = 60.0

    def __init__(self, api_key: str, api_url: str, max_retries: int = MAX_RETRIES):
        """Initialize the API client.

        Args:
            api_key: Provider API key.
            api_url: Chat completion endpoint.
            max_retries: Maximum retry attempts for retryable failures.
        """
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.max_retries = max_retries

        masked_key = mask_sensitive_info(api_key)
        logger.info("APIClient initialized | api_key=%s | max_retries=%s", masked_key, max_retries)

    def _should_retry(self, status_code: int) -> bool:
        """Return whether a response status code is retryable."""
        should_retry = status_code in self.RETRYABLE_STATUS_CODES

        if should_retry:
            logger.warning("Retryable API error detected | status_code=%s", status_code)
        else:
            logger.debug("Non-retryable API error detected | status_code=%s", status_code)

        return should_retry

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = self.BASE_DELAY * (2 ** attempt)
        delay += random.uniform(0, 0.1 * delay)
        delay = min(delay, self.MAX_DELAY)

        logger.debug("Retry delay calculated | attempt=%s | delay=%.2fs", attempt + 1, delay)
        return delay

    def chat(self, message: str, system_prompt: Optional[str] = None) -> Dict:
        """Send a chat request.

        Args:
            message: User message.
            system_prompt: Optional system prompt.

        Returns:
            Parsed API response.

        Raises:
            APIError: If the request fails.
        """
        start_time = time.time()
        last_error = None

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        payload = {
            "model": Config.MODEL_NAME,
            "messages": messages,
            "temperature": Config.TEMPERATURE,
            "max_tokens": Config.MAX_TOKENS,
        }

        for attempt in range(self.max_retries + 1):
            try:
                if attempt == 0:
                    logger.info("Sending API request | url=%s", self.api_url)
                else:
                    logger.warning(
                        "Retrying API request | attempt=%s/%s",
                        attempt + 1,
                        self.max_retries + 1,
                    )

                logger.debug("Request payload: %s", payload)

                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30,
                )

                elapsed_time = time.time() - start_time
                self._handle_error(response)

                if attempt > 0:
                    logger.info(
                        "API call succeeded after retry | elapsed=%.2fs | status_code=%s",
                        elapsed_time,
                        response.status_code,
                    )
                else:
                    logger.info(
                        "API call succeeded | elapsed=%.2fs | status_code=%s",
                        elapsed_time,
                        response.status_code,
                    )

                return response.json()

            except APIError as e:
                last_error = e

                if e.status_code and self._should_retry(e.status_code):
                    if attempt < self.max_retries:
                        delay = self._calculate_delay(attempt)
                        logger.warning("Waiting %.2fs before retry", delay)
                        time.sleep(delay)
                        continue

                    logger.error(
                        "Retry limit reached | max_retries=%s | last_error=%s",
                        self.max_retries,
                        e,
                    )
                    raise

                logger.error("Non-retryable API error | error=%s", e)
                raise

            except requests.exceptions.Timeout:
                last_error = APIError("Request timed out", status_code=0, retryable=True)
                logger.warning(
                    "Request timed out | attempt=%s/%s",
                    attempt + 1,
                    self.max_retries + 1,
                )

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning("Waiting %.2fs before retry", delay)
                    time.sleep(delay)
                    continue

                logger.error("Retry limit reached | timeout")
                raise last_error

            except requests.exceptions.RequestException as e:
                last_error = APIError(f"Network error: {e}", status_code=0, retryable=True)
                logger.warning(
                    "Network error | attempt=%s/%s",
                    attempt + 1,
                    self.max_retries + 1,
                )

                if attempt < self.max_retries:
                    delay = self._calculate_delay(attempt)
                    logger.warning("Waiting %.2fs before retry", delay)
                    time.sleep(delay)
                    continue

                logger.error("Retry limit reached | network error")
                raise last_error

        raise last_error

    def _handle_error(self, response) -> None:
        """Raise APIError for non-successful responses."""
        if response.status_code == 200:
            return

        logger.error(
            "API error | status_code=%s | response=%s",
            response.status_code,
            response.text[:200],
        )

        retryable = self._should_retry(response.status_code)

        if response.status_code == 401:
            raise APIError(
                "Authentication failed. Check API key and Authorization header.",
                status_code=401,
                retryable=False,
            )

        if response.status_code == 400:
            raise APIError(
                f"Bad request: {response.text[:200]}",
                status_code=400,
                retryable=False,
            )

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                logger.warning("Server requested retry delay | retry_after=%ss", retry_after)

            raise APIError(
                "Rate limit exceeded.",
                status_code=429,
                retryable=True,
            )

        if response.status_code >= 500:
            raise APIError(
                "Server error.",
                status_code=response.status_code,
                retryable=True,
            )

        raise APIError(
            f"Unknown API error: {response.text[:200]}",
            status_code=response.status_code,
            retryable=retryable,
        )


class APIError(Exception):
    """Custom API exception."""

    def __init__(self, message: str, status_code: int = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.retryable = retryable

    def __str__(self):
        if self.status_code:
            return f"[HTTP {self.status_code}] {self.message}"
        return self.message
