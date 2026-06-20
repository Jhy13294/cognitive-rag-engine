import uuid
from typing import Any, Callable

from logger import reset_request_id, set_request_id

DictLike = dict[str, Any]


class RequestIDMiddleware:
    """Attach a generated request ID without buffering streaming responses."""

    def __init__(self, app: Callable, header_name: str = "X-Request-ID"):
        """Initialize the pure ASGI middleware."""
        self.app = app
        self.header_name = header_name
        self.header_name_bytes = header_name.lower().encode("latin-1")

    async def __call__(self, scope: DictLike, receive: Callable, send: Callable) -> None:
        """Generate, expose, and return one request ID for each HTTP request."""
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid.uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        context_token = set_request_id(request_id)

        async def send_with_request_id(message: DictLike) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers = [item for item in headers if item[0].lower() != self.header_name_bytes]
                headers.append((self.header_name_bytes, request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            reset_request_id(context_token)
