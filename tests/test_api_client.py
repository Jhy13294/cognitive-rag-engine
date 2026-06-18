import unittest
from unittest.mock import AsyncMock, patch

from api_client import APIClient, parse_chat_stream_line


class FakeAPIResponse:
    """Minimal httpx response double."""

    def __init__(self, status_code=200, data=None, text="{}", headers=None):
        self.status_code = status_code
        self._data = data or {"choices": [{"message": {"content": "ok"}}]}
        self.text = text
        self.headers = headers or {}

    def json(self):
        return self._data

    async def aread(self):
        return self.text.encode("utf-8")


class FakeStreamResponse(FakeAPIResponse):
    """Streaming httpx response double."""

    def __init__(self, lines, status_code=200, text="{}", headers=None):
        super().__init__(status_code=status_code, text=text, headers=headers)
        self.lines = lines

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class FakeStreamContext:
    """Async context manager returned by httpx.AsyncClient.stream."""

    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeAPIAsyncClient:
    """Async httpx client test double for chat requests."""

    created_count = 0
    responses = []
    post_requests = []
    stream_response = None
    stream_requests = []

    def __init__(self, timeout=None):
        FakeAPIAsyncClient.created_count += 1
        self.timeout = timeout

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, headers=None, json=None):
        FakeAPIAsyncClient.post_requests.append(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": self.timeout,
            }
        )
        return FakeAPIAsyncClient.responses.pop(0)

    def stream(self, method, url, headers=None, json=None):
        FakeAPIAsyncClient.stream_requests.append(
            {
                "method": method,
                "url": url,
                "headers": headers,
                "json": json,
                "timeout": self.timeout,
            }
        )
        return FakeStreamContext(FakeAPIAsyncClient.stream_response)


def reset_fake_api_client(*responses, stream_response=None):
    FakeAPIAsyncClient.created_count = 0
    FakeAPIAsyncClient.responses = list(responses)
    FakeAPIAsyncClient.post_requests = []
    FakeAPIAsyncClient.stream_response = stream_response
    FakeAPIAsyncClient.stream_requests = []


class APIClientTests(unittest.IsolatedAsyncioTestCase):
    def build_client(self, max_retries=1):
        return APIClient(
            api_key="sk-test",
            api_url="https://api.example.test/v1/chat/completions",
            max_retries=max_retries,
        )

    @patch("api_client.httpx.AsyncClient", new=FakeAPIAsyncClient)
    async def test_async_chat_retries_rate_limit_with_async_sleep(self):
        reset_fake_api_client(
            FakeAPIResponse(
                status_code=429,
                text='{"error":{"message":"rate limited"}}',
                headers={"Retry-After": "0"},
            ),
            FakeAPIResponse(data={"choices": [{"message": {"content": "done"}}]}),
        )
        client = self.build_client(max_retries=1)

        with patch("api_client.asyncio.sleep", new_callable=AsyncMock) as sleep_mock:
            response = await client.async_chat("hello")

        self.assertEqual(response["choices"][0]["message"]["content"], "done")
        self.assertEqual(FakeAPIAsyncClient.created_count, 1)
        self.assertEqual(len(FakeAPIAsyncClient.post_requests), 2)
        sleep_mock.assert_awaited_once_with(0.0)
        payload = FakeAPIAsyncClient.post_requests[0]["json"]
        self.assertEqual(payload["messages"][-1]["content"], "hello")
        self.assertNotIn("stream", payload)

    @patch("api_client.httpx.AsyncClient", new=FakeAPIAsyncClient)
    async def test_stream_chat_yields_provider_deltas(self):
        stream_response = FakeStreamResponse(
            [
                'data: {"choices":[{"delta":{"content":"Hel"}}]}',
                'data: {"choices":[{"delta":{"content":"lo"}}]}',
                "data: [DONE]",
            ]
        )
        reset_fake_api_client(stream_response=stream_response)
        client = self.build_client(max_retries=0)

        tokens = [token async for token in client.stream_chat("hello")]

        self.assertEqual(tokens, ["Hel", "lo"])
        self.assertEqual(FakeAPIAsyncClient.stream_requests[0]["method"], "POST")
        self.assertTrue(FakeAPIAsyncClient.stream_requests[0]["json"]["stream"])

    def test_parse_chat_stream_line_handles_done_and_malformed_events(self):
        self.assertEqual(
            parse_chat_stream_line('data: {"choices":[{"delta":{"content":"x"}}]}'),
            "x",
        )
        self.assertEqual(parse_chat_stream_line("data: [DONE]"), "__stream_done__")
        self.assertIsNone(parse_chat_stream_line("data: {bad json"))


if __name__ == "__main__":
    unittest.main()
