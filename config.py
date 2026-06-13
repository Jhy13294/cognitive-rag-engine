import logging
import os

from dotenv import load_dotenv

from logger import mask_sensitive_info, setup_logger

logger = setup_logger(__name__, level=logging.INFO)

load_dotenv()


def _env_int(name: str, default: int = None) -> int:
    """Read an integer from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    return int(raw_value)


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    return float(raw_value)


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from the environment."""
    raw_value = os.getenv(name)
    if raw_value is None or raw_value == "":
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}


class Config:
    """Application configuration loaded from environment variables."""

    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")

    MODEL_NAME = "deepseek-v4-flash"
    TEMPERATURE = 0.7
    MAX_TOKENS = 2000

    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
    EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
    EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://api.openai.com/v1/embeddings")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-3-small")
    EMBEDDING_DIMENSION = _env_int("EMBEDDING_DIMENSION")
    EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 64)
    EMBEDDING_MAX_BATCH_TOKENS = _env_int("EMBEDDING_MAX_BATCH_TOKENS", 8192)
    EMBEDDING_TIMEOUT = _env_float("EMBEDDING_TIMEOUT", 30.0)
    EMBEDDING_MAX_RETRIES = _env_int("EMBEDDING_MAX_RETRIES", 3)
    EMBEDDING_BASE_DELAY = _env_float("EMBEDDING_BASE_DELAY", 1.0)
    EMBEDDING_MAX_DELAY = _env_float("EMBEDDING_MAX_DELAY", 30.0)
    EMBEDDING_USER = os.getenv("EMBEDDING_USER")

    VECTOR_STORE_PROVIDER = os.getenv("VECTOR_STORE_PROVIDER", "memory")
    VECTOR_STORE_HOST = os.getenv("VECTOR_STORE_HOST") or os.getenv("QDRANT_HOST", "localhost")
    VECTOR_STORE_PORT = _env_int("VECTOR_STORE_PORT", _env_int("QDRANT_PORT", 6333))
    VECTOR_STORE_URL = os.getenv("VECTOR_STORE_URL") or os.getenv("QDRANT_URL")
    VECTOR_STORE_API_KEY = os.getenv("VECTOR_STORE_API_KEY") or os.getenv("QDRANT_API_KEY")
    VECTOR_STORE_COLLECTION = (
        os.getenv("VECTOR_STORE_COLLECTION") or os.getenv("QDRANT_COLLECTION", "enterprise_kb")
    )
    VECTOR_STORE_DISTANCE_METRIC = (
        os.getenv("VECTOR_STORE_DISTANCE_METRIC") or os.getenv("QDRANT_DISTANCE", "cosine")
    )
    VECTOR_STORE_BATCH_SIZE = _env_int("VECTOR_STORE_BATCH_SIZE", _env_int("QDRANT_BATCH_SIZE", 64))
    VECTOR_STORE_TIMEOUT = _env_float("VECTOR_STORE_TIMEOUT", _env_float("QDRANT_TIMEOUT", 30.0))
    VECTOR_STORE_MAX_RETRIES = _env_int(
        "VECTOR_STORE_MAX_RETRIES",
        _env_int("QDRANT_MAX_RETRIES", 3),
    )
    VECTOR_STORE_BASE_DELAY = _env_float(
        "VECTOR_STORE_BASE_DELAY",
        _env_float("QDRANT_BASE_DELAY", 0.5),
    )
    VECTOR_STORE_MAX_DELAY = _env_float(
        "VECTOR_STORE_MAX_DELAY",
        _env_float("QDRANT_MAX_DELAY", 8.0),
    )
    VECTOR_STORE_RECREATE = _env_bool("VECTOR_STORE_RECREATE", _env_bool("QDRANT_RECREATE", False))

    @classmethod
    def validate(cls):
        """Validate required configuration values."""
        if not cls.API_KEY:
            logger.error("API key is not configured")
            raise ValueError("API key is not configured. Check the .env file.")

        masked_key = mask_sensitive_info(cls.API_KEY)
        logger.debug("API key: %s", masked_key)
        logger.debug("API URL: %s", cls.API_URL)
        logger.debug("Model: %s", cls.MODEL_NAME)

        print("Configuration validated")
        return True

    @classmethod
    def validate_embedding(cls):
        """Validate embedding provider configuration."""
        if cls.EMBEDDING_PROVIDER.lower() == "openai" and not cls.EMBEDDING_API_KEY:
            logger.error("Embedding API key is not configured")
            raise ValueError("Embedding API key is not configured. Set EMBEDDING_API_KEY or OPENAI_API_KEY.")

        logger.debug("Embedding provider: %s", cls.EMBEDDING_PROVIDER)
        logger.debug("Embedding API URL: %s", cls.EMBEDDING_API_URL)
        logger.debug("Embedding model: %s", cls.EMBEDDING_MODEL_NAME)
        logger.debug("Embedding dimension: %s", cls.EMBEDDING_DIMENSION)
        return True

    @classmethod
    def validate_vector_store(cls, provider: str = None):
        """Validate vector-store configuration."""
        selected_provider = (provider or cls.VECTOR_STORE_PROVIDER).lower()

        if selected_provider == "memory":
            return True

        if selected_provider != "qdrant":
            raise ValueError(f"Unsupported vector store provider: {selected_provider}")

        if not cls.VECTOR_STORE_COLLECTION:
            raise ValueError("VECTOR_STORE_COLLECTION is required for Qdrant.")
        if cls.VECTOR_STORE_PORT <= 0:
            raise ValueError("VECTOR_STORE_PORT must be greater than 0.")
        if cls.VECTOR_STORE_BATCH_SIZE <= 0:
            raise ValueError("VECTOR_STORE_BATCH_SIZE must be greater than 0.")
        if cls.VECTOR_STORE_DISTANCE_METRIC.lower() not in {"cosine", "dot", "euclid", "euclidean"}:
            raise ValueError("VECTOR_STORE_DISTANCE_METRIC must be one of: cosine, dot, euclid.")

        logger.debug("Vector store provider: %s", selected_provider)
        logger.debug("Vector store host: %s", cls.VECTOR_STORE_HOST)
        logger.debug("Vector store port: %s", cls.VECTOR_STORE_PORT)
        logger.debug("Vector store url: %s", cls.VECTOR_STORE_URL)
        logger.debug("Vector store collection: %s", cls.VECTOR_STORE_COLLECTION)
        logger.debug("Vector store distance metric: %s", cls.VECTOR_STORE_DISTANCE_METRIC)
        return True
