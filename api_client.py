import asyncio
import json
import logging
import random
import threading
import time
from typing import AsyncIterator, Dict, Optional, Set

import httpx

from config import Config
from logger import mask_sensitive_info, setup_logger

logger = setup_logger(__name__, level=logging.INFO)


class APIClient:
    """HTTP client for chat-completion compatible LLM APIs."""

    RETRYABLE_STATUS_CODES: Set[int] = {429, 500, 502, 503, 504}

    MAX_RETRIES: int = 3
    BASE_DELAY: float = 1.0
    MAX_DELAY: float = 60.0

    def __init__(
        self,
        api_key: str,
        api_url: str,
        max_retries: int = MAX_RETRIES,
        temperature: Optional[float] = None,
    ):
        """Initialize the API client.

        Args:
            api_key: Provider API key.
            api_url: Chat completion endpoint.
            max_retries: Maximum retry attempts for retryable failures.
            temperature: Optional per-client generation temperature override.
        """
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.max_retries = max_retries
        self.temperature = temperature
        self.request_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self._usage_lock = threading.Lock()

        masked_key = mask_sensitive_info(api_key)
        logger.info("APIClient initialized | api_key=%s | max_retries=%s", masked_key, max_retries)

    def chat(self, message: str, system_prompt: Optional[str] = None) -> Dict:
        """Send a chat request from synchronous code."""
        return run_async_blocking(self.async_chat(message, system_prompt=system_prompt))

    async def async_chat(self, message: str, system_prompt: Optional[str] = None) -> Dict:
        """Send a chat request without blocking the event loop."""
        payload = self._build_payload(message, system_prompt=system_prompt, stream=False)
        response = await self._post_json_with_retries(payload)
        self._record_usage(response)
        return response

    async def stream_chat(self, message: str, system_prompt: Optional[str] = None) -> AsyncIterator[str]:
        """Stream chat-completion deltas from the upstream provider."""
        payload = self._build_payload(message, system_prompt=system_prompt, stream=True)
        request_recorded = False
        async for token in self._stream_with_retries(payload):
            if not request_recorded:
                self._record_stream_request()
                request_recorded = True
            yield token
        if not request_recorded:
            self._record_stream_request()

    def usage_stats(self) -> Dict[str, int]:
        """Return a thread-safe copy of cumulative provider-reported token usage."""
        with self._usage_lock:
            return {
                "request_count": self.request_count,
                "prompt_tokens": self.total_prompt_tokens,
                "completion_tokens": self.total_completion_tokens,
                "total_tokens": self.total_tokens,
            }

    def _record_usage(self, response: Dict) -> None:
        """Accumulate provider-reported usage for service-level observability."""
        usage = response.get("usage") if isinstance(response, dict) else None
        usage = usage if isinstance(usage, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        with self._usage_lock:
            self.request_count += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += total_tokens

    def _record_stream_request(self) -> None:
        """Count a streaming request whose provider token usage is unavailable."""
        with self._usage_lock:
            self.request_count += 1

    def _build_payload(self, message: str, system_prompt: Optional[str], stream: bool) -> Dict:
        """Build a chat-completion compatible request payload."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        payload = {
            "model": Config.MODEL_NAME,
            "messages": messages,
            "temperature": (
                self.temperature
                if self.temperature is not None
                else Config.TEMPERATURE
            ),
            "max_tokens": Config.MAX_TOKENS,
        }
        if stream:
            payload["stream"] = True
        return payload

    async def _post_json_with_retries(self, payload: Dict) -> Dict:
        """Post a JSON request with async retry handling."""
        start_time = time.monotonic()
        last_error = None

        async with httpx.AsyncClient(timeout=30.0) as client:
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

                    response = await client.post(
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                    )

                    elapsed_time = time.monotonic() - start_time
                    self._handle_error(response)

                    logger.info(
                        "API call succeeded | elapsed=%.2fs | status_code=%s | attempt=%s",
                        elapsed_time,
                        response.status_code,
                        attempt + 1,
                    )
                    return response.json()

                except APIError as e:
                    last_error = e
                    if e.retryable and attempt < self.max_retries:
                        await self._sleep_before_retry(attempt, retry_after=e.retry_after)
                        continue
                    logger.error("API call failed | error=%s", e)
                    raise

                except httpx.TimeoutException as e:
                    last_error = APIError("Request timed out", status_code=0, retryable=True, error_kind="timeout")
                    logger.warning("Request timed out | attempt=%s/%s | error=%s", attempt + 1, self.max_retries + 1, e)
                    if attempt < self.max_retries:
                        await self._sleep_before_retry(attempt)
                        continue
                    raise last_error

                except httpx.RequestError as e:
                    last_error = APIError(f"Network error: {e}", status_code=0, retryable=True, error_kind="network")
                    logger.warning("Network error | attempt=%s/%s | error=%s", attempt + 1, self.max_retries + 1, e)
                    if attempt < self.max_retries:
                        await self._sleep_before_retry(attempt)
                        continue
                    raise last_error

        raise last_error

    async def _stream_with_retries(self, payload: Dict) -> AsyncIterator[str]:
        """Open a streaming request and yield upstream token deltas."""
        last_error = None

        async with httpx.AsyncClient(timeout=None) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    async with client.stream(
                        "POST",
                        self.api_url,
                        headers=self.headers,
                        json=payload,
                    ) as response:
                        if response.status_code != 200:
                            await response.aread()
                        self._handle_error(response)
                        async for line in response.aiter_lines():
                            token = parse_chat_stream_line(line)
                            if token is None:
                                continue
                            if token == STREAM_DONE:
                                return
                            yield token
                        return

                except APIError as e:
                    last_error = e
                    if e.retryable and attempt < self.max_retries:
                        await self._sleep_before_retry(attempt, retry_after=e.retry_after)
                        continue
                    logger.error("Streaming API call failed | error=%s", e)
                    raise

                except httpx.TimeoutException as e:
                    last_error = APIError("Request timed out", status_code=0, retryable=True, error_kind="timeout")
                    logger.warning("Streaming request timed out | attempt=%s/%s | error=%s", attempt + 1, self.max_retries + 1, e)
                    if attempt < self.max_retries:
                        await self._sleep_before_retry(attempt)
                        continue
                    raise last_error

                except httpx.RequestError as e:
                    last_error = APIError(f"Network error: {e}", status_code=0, retryable=True, error_kind="network")
                    logger.warning("Streaming network error | attempt=%s/%s | error=%s", attempt + 1, self.max_retries + 1, e)
                    if attempt < self.max_retries:
                        await self._sleep_before_retry(attempt)
                        continue
                    raise last_error

        raise last_error

    async def _sleep_before_retry(self, attempt: int, response=None, retry_after: Optional[str] = None) -> None:
        """Sleep asynchronously before retrying a failed request."""
        if response is not None and retry_after is None:
            retry_after = response.headers.get("Retry-After")

        if retry_after:
            delay = float(retry_after)
        else:
            delay = self._calculate_delay(attempt)

        logger.warning("Waiting %.2fs before retry", delay)
        await asyncio.sleep(delay)

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

    def _handle_error(self, response) -> None:
        """Raise APIError for non-successful responses."""
        if response.status_code == 200:
            return

        try:
            response_text = getattr(response, "text", "") or ""
        except Exception:
            response_text = ""
        logger.error(
            "API error | status_code=%s | response=%s",
            response.status_code,
            response_text[:200],
        )

        retryable = self._should_retry(response.status_code)

        if response.status_code == 401:
            raise APIError(
                "Authentication failed. Check API key and Authorization header.",
                status_code=401,
                retryable=False,
                error_kind="authentication",
            )

        if response.status_code == 400:
            raise APIError(
                f"Bad request: {response_text[:200]}",
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
                retry_after=retry_after,
                error_kind="rate_limit",
            )

        if response.status_code >= 500:
            raise APIError(
                "Server error.",
                status_code=response.status_code,
                retryable=True,
            )

        raise APIError(
            f"Unknown API error: {response_text[:200]}",
            status_code=response.status_code,
            retryable=retryable,
        )


class APIError(Exception):
    """Custom API exception."""

    def __init__(
        self,
        message: str,
        status_code: int = None,
        retryable: bool = False,
        retry_after: Optional[str] = None,
        error_kind: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.retryable = retryable
        self.retry_after = retry_after
        self.error_kind = error_kind

    def __str__(self):
        if self.status_code:
            return f"[HTTP {self.status_code}] {self.message}"
        return self.message


STREAM_DONE = "__stream_done__"


def parse_chat_stream_line(line: str) -> Optional[str]:
    """Parse one OpenAI-compatible SSE line into a token delta."""
    if not line:
        return None
    if not line.startswith("data:"):
        return None

    raw_data = line[len("data:"):].strip()
    if raw_data == "[DONE]":
        return STREAM_DONE

    try:
        payload = json.loads(raw_data)
    except json.JSONDecodeError:
        logger.warning("Skipping malformed stream payload | payload=%s", raw_data[:200])
        return None

    choices = payload.get("choices") or []
    if not choices:
        return None

    choice = choices[0]
    delta = choice.get("delta") or {}
    content = delta.get("content")
    if content is None:
        message = choice.get("message") or {}
        content = message.get("content")
    return content


def run_async_blocking(coro):
    """Run an async coroutine from synchronous code, including inside a live event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = {}
    error = {}

    def runner():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as e:
            error["value"] = e

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()

    if error:
        raise error["value"]
    return result.get("value")
