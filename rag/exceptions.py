class RAGConfigurationError(ValueError):
    """Base class for typed RAG configuration and index errors."""


class IndexNotReadyError(RAGConfigurationError):
    """Raised when the vector index is empty or unavailable."""


class EmbeddingSpaceMismatchError(RAGConfigurationError):
    """Raised when query-time embedding settings do not match the index."""


class EmbeddingSpaceInvalidError(RAGConfigurationError):
    """Raised when indexed records have missing or mixed embedding metadata."""
