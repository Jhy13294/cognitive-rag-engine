import logging
import os

from dotenv import load_dotenv

from logger import mask_sensitive_info, setup_logger
from tokenization import validate_tokenizer_encoding

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

    RERANK_ENABLED = _env_bool("RERANK_ENABLED", False)
    RERANK_PROVIDER = os.getenv("RERANK_PROVIDER", "deterministic")
    RERANK_MODEL = os.getenv("RERANK_MODEL", "deterministic-lexical-reranker")
    RERANK_FETCH_K = _env_int("RERANK_FETCH_K", 30)
    RERANK_TOP_N = _env_int("RERANK_TOP_N", 5)
    RERANK_BATCH_SIZE = _env_int("RERANK_BATCH_SIZE", 32)
    RERANK_API_KEY = os.getenv("RERANK_API_KEY") or os.getenv("COHERE_API_KEY")
    RERANK_API_URL = os.getenv("RERANK_API_URL", "https://api.cohere.com/v2/rerank")
    RERANK_TIMEOUT = _env_float("RERANK_TIMEOUT", 30.0)
    RERANK_MAX_RETRIES = _env_int("RERANK_MAX_RETRIES", 3)
    RERANK_BASE_DELAY = _env_float("RERANK_BASE_DELAY", 0.5)
    RERANK_MAX_DELAY = _env_float("RERANK_MAX_DELAY", 8.0)

    HYBRID_ENABLED = _env_bool("HYBRID_ENABLED", False)
    HYBRID_DENSE_WEIGHT = _env_float("HYBRID_DENSE_WEIGHT", 0.2)
    HYBRID_SPARSE_WEIGHT = _env_float("HYBRID_SPARSE_WEIGHT", 1.0)
    RRF_K = _env_int("RRF_K", 60)
    BM25_K1 = _env_float("BM25_K1", 1.5)
    BM25_B = _env_float("BM25_B", 0.75)

    PARENT_CHILD_ENABLED = _env_bool("PARENT_CHILD_ENABLED", False)
    PARENT_CHUNK_SIZE = _env_int("PARENT_CHUNK_SIZE", 1600)
    PARENT_CHUNK_OVERLAP = _env_int("PARENT_CHUNK_OVERLAP", 200)
    CHILD_CHUNK_SIZE = _env_int("CHILD_CHUNK_SIZE", 400)
    CHILD_CHUNK_OVERLAP = _env_int("CHILD_CHUNK_OVERLAP", 80)

    CONTEXT_PACKING_ENABLED = _env_bool("CONTEXT_PACKING_ENABLED", False)
    CONTEXT_DEDUP_ENABLED = _env_bool("CONTEXT_DEDUP_ENABLED", False)
    CONTEXT_NEAR_DUP_ENABLED = _env_bool("CONTEXT_NEAR_DUP_ENABLED", False)
    CONTEXT_NEAR_DUP_THRESHOLD = _env_float("CONTEXT_NEAR_DUP_THRESHOLD", 0.9)
    CONTEXT_MAX_TOKENS = _env_int("CONTEXT_MAX_TOKENS", 2048)
    TOKENIZER_ENCODING = os.getenv("TOKENIZER_ENCODING", "cl100k_base")

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

    @classmethod
    def validate_rerank(cls, provider: str = None):
        """Validate reranker configuration."""
        selected_provider = (provider or cls.RERANK_PROVIDER).lower()

        if selected_provider in {"none", "off", "disabled"}:
            return True
        if selected_provider not in {"deterministic", "cohere"}:
            raise ValueError(f"Unsupported rerank provider: {selected_provider}")
        if cls.RERANK_FETCH_K <= 0:
            raise ValueError("RERANK_FETCH_K must be greater than 0.")
        if cls.RERANK_TOP_N <= 0:
            raise ValueError("RERANK_TOP_N must be greater than 0.")
        if cls.RERANK_BATCH_SIZE <= 0:
            raise ValueError("RERANK_BATCH_SIZE must be greater than 0.")
        if selected_provider == "cohere" and not cls.RERANK_API_KEY:
            raise ValueError("RERANK_API_KEY or COHERE_API_KEY is required for Cohere rerank.")

        logger.debug("Rerank enabled: %s", cls.RERANK_ENABLED)
        logger.debug("Rerank provider: %s", selected_provider)
        logger.debug("Rerank model: %s", cls.RERANK_MODEL)
        logger.debug("Rerank fetch_k: %s", cls.RERANK_FETCH_K)
        logger.debug("Rerank top_n: %s", cls.RERANK_TOP_N)
        return True

    @classmethod
    def validate_hybrid(cls):
        """Validate hybrid retrieval configuration."""
        if cls.RRF_K <= 0:
            raise ValueError("RRF_K must be greater than 0.")
        if cls.HYBRID_DENSE_WEIGHT < 0:
            raise ValueError("HYBRID_DENSE_WEIGHT must be non-negative.")
        if cls.HYBRID_SPARSE_WEIGHT < 0:
            raise ValueError("HYBRID_SPARSE_WEIGHT must be non-negative.")
        if cls.HYBRID_DENSE_WEIGHT + cls.HYBRID_SPARSE_WEIGHT <= 0:
            raise ValueError("At least one hybrid retrieval weight must be greater than 0.")
        if cls.BM25_K1 <= 0:
            raise ValueError("BM25_K1 must be greater than 0.")
        if cls.BM25_B < 0 or cls.BM25_B > 1:
            raise ValueError("BM25_B must be between 0 and 1.")

        logger.debug("Hybrid enabled: %s", cls.HYBRID_ENABLED)
        logger.debug("Hybrid dense weight: %s", cls.HYBRID_DENSE_WEIGHT)
        logger.debug("Hybrid sparse weight: %s", cls.HYBRID_SPARSE_WEIGHT)
        logger.debug("RRF k: %s", cls.RRF_K)
        logger.debug("BM25 k1: %s", cls.BM25_K1)
        logger.debug("BM25 b: %s", cls.BM25_B)
        return True

    @classmethod
    def validate_parent_child(cls):
        """Validate parent-child chunking configuration."""
        if cls.PARENT_CHUNK_SIZE <= 0:
            raise ValueError("PARENT_CHUNK_SIZE must be greater than 0.")
        if cls.CHILD_CHUNK_SIZE <= 0:
            raise ValueError("CHILD_CHUNK_SIZE must be greater than 0.")
        if cls.PARENT_CHUNK_OVERLAP < 0:
            raise ValueError("PARENT_CHUNK_OVERLAP cannot be negative.")
        if cls.CHILD_CHUNK_OVERLAP < 0:
            raise ValueError("CHILD_CHUNK_OVERLAP cannot be negative.")
        if cls.PARENT_CHUNK_OVERLAP >= cls.PARENT_CHUNK_SIZE:
            raise ValueError("PARENT_CHUNK_OVERLAP must be smaller than PARENT_CHUNK_SIZE.")
        if cls.CHILD_CHUNK_OVERLAP >= cls.CHILD_CHUNK_SIZE:
            raise ValueError("CHILD_CHUNK_OVERLAP must be smaller than CHILD_CHUNK_SIZE.")
        if cls.CHILD_CHUNK_SIZE >= cls.PARENT_CHUNK_SIZE:
            raise ValueError("CHILD_CHUNK_SIZE must be smaller than PARENT_CHUNK_SIZE.")

        logger.debug("Parent-child enabled: %s", cls.PARENT_CHILD_ENABLED)
        logger.debug("Parent chunk size: %s", cls.PARENT_CHUNK_SIZE)
        logger.debug("Parent chunk overlap: %s", cls.PARENT_CHUNK_OVERLAP)
        logger.debug("Child chunk size: %s", cls.CHILD_CHUNK_SIZE)
        logger.debug("Child chunk overlap: %s", cls.CHILD_CHUNK_OVERLAP)
        return True

    @classmethod
    def validate_context_packing(cls):
        """Validate context packing configuration."""
        if cls.CONTEXT_NEAR_DUP_THRESHOLD < 0 or cls.CONTEXT_NEAR_DUP_THRESHOLD > 1:
            raise ValueError("CONTEXT_NEAR_DUP_THRESHOLD must be between 0 and 1.")
        if cls.CONTEXT_MAX_TOKENS <= 0:
            raise ValueError("CONTEXT_MAX_TOKENS must be greater than 0.")
        validate_tokenizer_encoding(cls.TOKENIZER_ENCODING)

        logger.debug("Context packing enabled: %s", cls.CONTEXT_PACKING_ENABLED)
        logger.debug("Context dedup enabled: %s", cls.CONTEXT_DEDUP_ENABLED)
        logger.debug("Context near-duplicate enabled: %s", cls.CONTEXT_NEAR_DUP_ENABLED)
        logger.debug("Context near-duplicate threshold: %s", cls.CONTEXT_NEAR_DUP_THRESHOLD)
        logger.debug("Context max tokens: %s", cls.CONTEXT_MAX_TOKENS)
        logger.debug("Tokenizer encoding: %s", cls.TOKENIZER_ENCODING)
        return True
