from typing import Tuple

from fastapi import HTTPException

from access import ACLAccessError
from api_client import APIError
from embeddings import OpenAIEmbeddingError
from logger import setup_logger
from rag import EmbeddingSpaceInvalidError, EmbeddingSpaceMismatchError, IndexNotReadyError

logger = setup_logger(__name__)


def to_http_exception(error: Exception) -> HTTPException:
    """Map internal errors to sanitized HTTP exceptions."""
    status_code, code, message = classify_exception(error)
    if status_code >= 500:
        logger.error("HTTP service error | code=%s | error_type=%s", code, type(error).__name__)
    else:
        logger.warning("HTTP request rejected | code=%s | error_type=%s", code, type(error).__name__)

    return HTTPException(
        status_code=status_code,
        detail={
            "error": {
                "code": code,
                "message": message,
            }
        },
    )


def classify_exception(error: Exception) -> Tuple[int, str, str]:
    """Classify an exception into HTTP status, code, and safe message."""
    if isinstance(error, (APIError, OpenAIEmbeddingError)):
        return classify_upstream_error(error)

    if isinstance(error, ACLAccessError):
        return 403, "acl_forbidden", "Access denied by ACL policy."

    if isinstance(error, IndexNotReadyError):
        return 409, "index_not_ready", "Vector index is empty or unavailable."
    if isinstance(error, EmbeddingSpaceMismatchError):
        return 422, "embedding_space_mismatch", str(error)
    if isinstance(error, EmbeddingSpaceInvalidError):
        return 422, "embedding_space_invalid", str(error)

    if isinstance(error, ValueError):
        return 400, "bad_request", str(error)

    return 500, "internal_error", "Internal server error."


def classify_upstream_error(error) -> Tuple[int, str, str]:
    """Classify upstream provider errors without leaking credentials."""
    status_code = getattr(error, "status_code", None)
    error_kind = getattr(error, "error_kind", None)

    if error_kind == "rate_limit" or status_code == 429:
        return 429, "upstream_rate_limited", "Upstream provider rate limited the request."
    if error_kind == "authentication" or status_code == 401:
        return 502, "upstream_auth_failed", "Upstream provider authentication failed."
    if error_kind == "timeout" or status_code in {408, 504}:
        return 504, "upstream_timeout", "Upstream provider timed out."
    if error_kind == "network" or status_code == 0:
        return 502, "upstream_network_error", "Upstream provider request failed."
    if status_code is not None and status_code >= 500:
        return 502, "upstream_error", "Upstream provider returned an error."

    return 502, "upstream_error", "Upstream provider request failed."
