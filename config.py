import logging
import os
import re

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

    MODEL_NAME = "deepseek-v4-pro"
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

    QUERY_REWRITE_ENABLED = _env_bool("QUERY_REWRITE_ENABLED", False)
    QUERY_REWRITE_PROVIDER = os.getenv("QUERY_REWRITE_PROVIDER", "deterministic")
    QUERY_REWRITE_FIXTURE_PATH = os.getenv("QUERY_REWRITE_FIXTURE_PATH", "eval/fixtures/query_rewrites.jsonl")
    QUERY_REWRITE_NUM_QUERIES = _env_int("QUERY_REWRITE_NUM_QUERIES", 3)
    QUERY_REWRITE_TEMPERATURE = _env_float("QUERY_REWRITE_TEMPERATURE", 0.1)
    QUERY_REWRITE_CACHE_ENABLED = _env_bool("QUERY_REWRITE_CACHE_ENABLED", True)
    QUERY_REWRITE_WEIGHT_ORIGINAL = _env_float("QUERY_REWRITE_WEIGHT_ORIGINAL", 1.0)
    QUERY_REWRITE_WEIGHT_VARIANT = _env_float("QUERY_REWRITE_WEIGHT_VARIANT", 0.7)

    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_NAMESPACE = os.getenv("CACHE_NAMESPACE", "rag-cache")
    CACHE_ENABLED = _env_bool("CACHE_ENABLED", False)
    CACHE_EMBEDDING_ENABLED = _env_bool("CACHE_EMBEDDING_ENABLED", True)
    CACHE_RETRIEVAL_ENABLED = _env_bool("CACHE_RETRIEVAL_ENABLED", True)
    CACHE_ANSWER_ENABLED = _env_bool("CACHE_ANSWER_ENABLED", True)
    CACHE_EMBEDDING_TTL = _env_int("CACHE_EMBEDDING_TTL", 604800)
    CACHE_RETRIEVAL_TTL = _env_int("CACHE_RETRIEVAL_TTL", 900)
    CACHE_ANSWER_TTL = _env_int("CACHE_ANSWER_TTL", 300)
    CACHE_TIMEOUT = _env_float("CACHE_TIMEOUT", 0.25)

    ACL_ENABLED = _env_bool("ACL_ENABLED", False)
    ACL_METADATA_KEY = os.getenv("ACL_METADATA_KEY", "acl")
    ACL_DEFAULT_DENY = _env_bool("ACL_DEFAULT_DENY", True)
    ACL_PRINCIPAL_HEADER = os.getenv("ACL_PRINCIPAL_HEADER", "X-Principal")
    ACL_ALLOW_BODY_PRINCIPAL = _env_bool("ACL_ALLOW_BODY_PRINCIPAL", False)
    ACL_INGEST_BINDINGS_ENABLED = _env_bool("ACL_INGEST_BINDINGS_ENABLED", False)
    METADATA_DB_URL = os.getenv("METADATA_DB_URL")
    MYSQL_HOST = os.getenv("MYSQL_HOST")
    MYSQL_PORT = _env_int("MYSQL_PORT", 3306)
    MYSQL_USER = os.getenv("MYSQL_USER")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE")

    OBSERVABILITY_ENABLED = _env_bool("OBSERVABILITY_ENABLED", False)
    AUDIT_ENABLED = _env_bool("AUDIT_ENABLED", False)
    METRICS_ENABLED = _env_bool("METRICS_ENABLED", False)
    AUDIT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")
    AUDIT_LOG_QUERY_TEXT = _env_bool("AUDIT_LOG_QUERY_TEXT", False)
    AUDIT_QUERY_TEXT_MAX_LENGTH = _env_int("AUDIT_QUERY_TEXT_MAX_LENGTH", 128)
    AUDIT_PRINCIPAL_MODE = os.getenv("AUDIT_PRINCIPAL_MODE", "hash")
    AUDIT_HASH_SALT = os.getenv("AUDIT_HASH_SALT")
    AUDIT_LOG_MAX_BYTES = _env_int("AUDIT_LOG_MAX_BYTES", 10 * 1024 * 1024)
    AUDIT_LOG_BACKUP_COUNT = _env_int("AUDIT_LOG_BACKUP_COUNT", 5)
    AUDIT_QUEUE_SIZE = _env_int("AUDIT_QUEUE_SIZE", 10000)
    METRICS_NAMESPACE = os.getenv("METRICS_NAMESPACE", "rag")
    METRICS_PATH = os.getenv("METRICS_PATH", "/metrics")

    RAGAS_ENABLED = _env_bool("RAGAS_ENABLED", False)
    RUN_RAGAS_EVAL = _env_bool("RUN_RAGAS_EVAL", False)
    RAGAS_JUDGE_API_KEY = os.getenv("RAGAS_JUDGE_API_KEY") or API_KEY
    RAGAS_JUDGE_BASE_URL = os.getenv("RAGAS_JUDGE_BASE_URL", "https://api.deepseek.com/v1")
    RAGAS_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", MODEL_NAME)
    RAGAS_JUDGE_TEMPERATURE = _env_float("RAGAS_JUDGE_TEMPERATURE", 0.0)
    RAGAS_JUDGE_TIMEOUT = _env_float("RAGAS_JUDGE_TIMEOUT", 60.0)
    RAGAS_JUDGE_MAX_RETRIES = _env_int("RAGAS_JUDGE_MAX_RETRIES", 3)
    RAGAS_JUDGE_MAX_TOKENS = _env_int("RAGAS_JUDGE_MAX_TOKENS", 4096)
    RAGAS_LIVE_REPETITIONS = _env_int("RAGAS_LIVE_REPETITIONS", 5)
    RAGAS_LIVE_GENERATION_TEMPERATURE = _env_float(
        "RAGAS_LIVE_GENERATION_TEMPERATURE",
        0.0,
    )
    RAGAS_EMBEDDING_PROVIDER = os.getenv("RAGAS_EMBEDDING_PROVIDER", "bge")
    # Local fastembed (ONNX, no torch) mode skips remote key/url and defaults to an
    # English model tuned for the English golden set so answer relevance runs offline
    # without API quota. (Bilingual jina-v2-base-zh was evaluated and rejected: it
    # collapses cosine on English paraphrases with low lexical overlap.)
    _RAGAS_EMBEDDING_IS_LOCAL = RAGAS_EMBEDDING_PROVIDER.strip().lower() == "bge"
    RAGAS_EMBEDDING_API_KEY = (
        os.getenv("RAGAS_EMBEDDING_API_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    RAGAS_EMBEDDING_BASE_URL = os.getenv(
        "RAGAS_EMBEDDING_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta",
    )
    RAGAS_EMBEDDING_MODEL = os.getenv(
        "RAGAS_EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5" if _RAGAS_EMBEDDING_IS_LOCAL else "gemini-embedding-001",
    )
    RAGAS_EMBEDDING_DIMENSION = _env_int(
        "RAGAS_EMBEDDING_DIMENSION",
        384 if _RAGAS_EMBEDDING_IS_LOCAL else 3072,
    )
    RAGAS_EMBEDDING_CACHE_DIR = os.getenv("RAGAS_EMBEDDING_CACHE_DIR") or None
    RAGAS_EMBEDDING_BATCH_SIZE = _env_int("RAGAS_EMBEDDING_BATCH_SIZE", 100)
    RAGAS_EMBEDDING_TIMEOUT = _env_float("RAGAS_EMBEDDING_TIMEOUT", 30.0)
    RAGAS_EMBEDDING_MAX_RETRIES = _env_int("RAGAS_EMBEDDING_MAX_RETRIES", 3)
    RAGAS_EMBEDDING_BASE_DELAY = _env_float("RAGAS_EMBEDDING_BASE_DELAY", 0.5)
    RAGAS_EMBEDDING_MAX_DELAY = _env_float("RAGAS_EMBEDDING_MAX_DELAY", 8.0)
    RAGAS_FIXTURE_PATH = os.getenv("RAGAS_FIXTURE_PATH", "eval/fixtures/ragas_verdicts.jsonl")
    RAGAS_REPORT_DIR = os.getenv("RAGAS_REPORT_DIR", "eval/reports/ragas")
    RAGAS_TOP_K = _env_int("RAGAS_TOP_K", 5)
    RAGAS_FAITHFULNESS_THRESHOLD = _env_float("RAGAS_FAITHFULNESS_THRESHOLD", 0.70)
    RAGAS_ANSWER_RELEVANCE_THRESHOLD = _env_float("RAGAS_ANSWER_RELEVANCE_THRESHOLD", 0.70)
    RAGAS_CONTEXT_PRECISION_THRESHOLD = _env_float("RAGAS_CONTEXT_PRECISION_THRESHOLD", 0.70)
    RAGAS_CONTEXT_RECALL_THRESHOLD = _env_float("RAGAS_CONTEXT_RECALL_THRESHOLD", 0.70)
    RAGAS_NEGATIVE_ABSTENTION_THRESHOLD = _env_float("RAGAS_NEGATIVE_ABSTENTION_THRESHOLD", 1.0)
    RAGAS_FAITHFULNESS_MARGIN = _env_float("RAGAS_FAITHFULNESS_MARGIN", 0.10)
    RAGAS_ANSWER_RELEVANCE_MARGIN = _env_float("RAGAS_ANSWER_RELEVANCE_MARGIN", 0.10)
    RAGAS_CONTEXT_PRECISION_MARGIN = _env_float("RAGAS_CONTEXT_PRECISION_MARGIN", 0.10)
    RAGAS_CONTEXT_RECALL_MARGIN = _env_float("RAGAS_CONTEXT_RECALL_MARGIN", 0.10)
    RAGAS_SIGMA_MULTIPLIER = _env_float("RAGAS_SIGMA_MULTIPLIER", 1.0)

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

    @classmethod
    def validate_query_rewrite(cls, provider: str = None):
        """Validate query rewrite configuration."""
        selected_provider = (provider or cls.QUERY_REWRITE_PROVIDER).lower()
        if selected_provider not in {"deterministic", "chat"}:
            raise ValueError(f"Unsupported query rewrite provider: {selected_provider}")
        if cls.QUERY_REWRITE_NUM_QUERIES < 1:
            raise ValueError("QUERY_REWRITE_NUM_QUERIES must be greater than or equal to 1.")
        if cls.QUERY_REWRITE_TEMPERATURE < 0:
            raise ValueError("QUERY_REWRITE_TEMPERATURE must be non-negative.")
        if cls.QUERY_REWRITE_WEIGHT_ORIGINAL < 0:
            raise ValueError("QUERY_REWRITE_WEIGHT_ORIGINAL must be non-negative.")
        if cls.QUERY_REWRITE_WEIGHT_VARIANT < 0:
            raise ValueError("QUERY_REWRITE_WEIGHT_VARIANT must be non-negative.")
        if cls.QUERY_REWRITE_WEIGHT_ORIGINAL < cls.QUERY_REWRITE_WEIGHT_VARIANT:
            raise ValueError("QUERY_REWRITE_WEIGHT_ORIGINAL must be greater than or equal to QUERY_REWRITE_WEIGHT_VARIANT.")
        if cls.QUERY_REWRITE_WEIGHT_ORIGINAL + cls.QUERY_REWRITE_WEIGHT_VARIANT <= 0:
            raise ValueError("At least one query rewrite weight must be greater than 0.")

        logger.debug("Query rewrite enabled: %s", cls.QUERY_REWRITE_ENABLED)
        logger.debug("Query rewrite provider: %s", selected_provider)
        logger.debug("Query rewrite num queries: %s", cls.QUERY_REWRITE_NUM_QUERIES)
        logger.debug("Query rewrite temperature: %s", cls.QUERY_REWRITE_TEMPERATURE)
        logger.debug("Query rewrite cache enabled: %s", cls.QUERY_REWRITE_CACHE_ENABLED)
        logger.debug("Query rewrite original weight: %s", cls.QUERY_REWRITE_WEIGHT_ORIGINAL)
        logger.debug("Query rewrite variant weight: %s", cls.QUERY_REWRITE_WEIGHT_VARIANT)
        return True

    @classmethod
    def validate_cache(cls):
        """Validate Redis cache configuration."""
        if cls.CACHE_EMBEDDING_TTL <= 0:
            raise ValueError("CACHE_EMBEDDING_TTL must be greater than 0.")
        if cls.CACHE_RETRIEVAL_TTL <= 0:
            raise ValueError("CACHE_RETRIEVAL_TTL must be greater than 0.")
        if cls.CACHE_ANSWER_TTL <= 0:
            raise ValueError("CACHE_ANSWER_TTL must be greater than 0.")
        if cls.CACHE_TIMEOUT <= 0:
            raise ValueError("CACHE_TIMEOUT must be greater than 0.")
        if cls.CACHE_ENABLED and not cls.REDIS_URL:
            raise ValueError("REDIS_URL is required when CACHE_ENABLED is true.")

        logger.debug("Cache enabled: %s", cls.CACHE_ENABLED)
        logger.debug("Cache namespace: %s", cls.CACHE_NAMESPACE)
        logger.debug("Cache embedding enabled: %s", cls.CACHE_EMBEDDING_ENABLED)
        logger.debug("Cache retrieval enabled: %s", cls.CACHE_RETRIEVAL_ENABLED)
        logger.debug("Cache answer enabled: %s", cls.CACHE_ANSWER_ENABLED)
        return True

    @classmethod
    def validate_acl(cls, resolver_provided: bool = False):
        """Validate ACL/RBAC configuration."""
        if not cls.ACL_METADATA_KEY or not cls.ACL_METADATA_KEY.strip():
            raise ValueError("ACL_METADATA_KEY must not be empty.")
        if not cls.ACL_PRINCIPAL_HEADER or not cls.ACL_PRINCIPAL_HEADER.strip():
            raise ValueError("ACL_PRINCIPAL_HEADER must not be empty.")
        if cls.MYSQL_PORT is None or cls.MYSQL_PORT <= 0:
            raise ValueError("MYSQL_PORT must be greater than 0.")

        if (cls.ACL_ENABLED or cls.ACL_INGEST_BINDINGS_ENABLED) and not resolver_provided:
            has_mysql_url = bool(cls.METADATA_DB_URL)
            has_mysql_parts = bool(cls.MYSQL_HOST and cls.MYSQL_USER and cls.MYSQL_DATABASE)
            if not has_mysql_url and not has_mysql_parts:
                raise ValueError(
                    "ACL metadata configuration requires METADATA_DB_URL or MYSQL_HOST/MYSQL_USER/MYSQL_DATABASE."
                )

        logger.debug("ACL enabled: %s", cls.ACL_ENABLED)
        logger.debug("ACL metadata key: %s", cls.ACL_METADATA_KEY)
        logger.debug("ACL default deny: %s", cls.ACL_DEFAULT_DENY)
        logger.debug("ACL principal header: %s", cls.ACL_PRINCIPAL_HEADER)
        logger.debug("ACL allow body principal: %s", cls.ACL_ALLOW_BODY_PRINCIPAL)
        logger.debug("ACL ingest bindings enabled: %s", cls.ACL_INGEST_BINDINGS_ENABLED)
        return True

    @classmethod
    def validate_observability(cls):
        """Validate audit and metrics configuration."""
        if not cls.OBSERVABILITY_ENABLED:
            if cls.AUDIT_ENABLED or cls.METRICS_ENABLED:
                raise ValueError("AUDIT_ENABLED and METRICS_ENABLED require OBSERVABILITY_ENABLED=true.")
            return True

        if not cls.AUDIT_ENABLED and not cls.METRICS_ENABLED:
            raise ValueError("At least one of AUDIT_ENABLED or METRICS_ENABLED must be true.")

        if cls.AUDIT_ENABLED:
            if not cls.AUDIT_LOG_PATH or not cls.AUDIT_LOG_PATH.strip():
                raise ValueError("AUDIT_LOG_PATH is required when audit logging is enabled.")
            if cls.AUDIT_PRINCIPAL_MODE not in {"hash", "mask", "raw"}:
                raise ValueError("AUDIT_PRINCIPAL_MODE must be one of: hash, mask, raw.")
            if not cls.AUDIT_HASH_SALT:
                raise ValueError("AUDIT_HASH_SALT is required when audit logging is enabled.")
            if cls.AUDIT_QUERY_TEXT_MAX_LENGTH <= 0:
                raise ValueError("AUDIT_QUERY_TEXT_MAX_LENGTH must be greater than 0.")
            if cls.AUDIT_LOG_MAX_BYTES <= 0:
                raise ValueError("AUDIT_LOG_MAX_BYTES must be greater than 0.")
            if cls.AUDIT_LOG_BACKUP_COUNT < 0:
                raise ValueError("AUDIT_LOG_BACKUP_COUNT cannot be negative.")
            if cls.AUDIT_QUEUE_SIZE <= 0:
                raise ValueError("AUDIT_QUEUE_SIZE must be greater than 0.")

        if cls.METRICS_ENABLED:
            if not cls.METRICS_NAMESPACE or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cls.METRICS_NAMESPACE) is None:
                raise ValueError("METRICS_NAMESPACE must contain only letters, numbers, and underscores.")
            if not cls.METRICS_PATH or not cls.METRICS_PATH.startswith("/"):
                raise ValueError("METRICS_PATH must start with '/'.")

        logger.debug("Observability enabled: %s", cls.OBSERVABILITY_ENABLED)
        logger.debug("Audit enabled: %s", cls.AUDIT_ENABLED)
        logger.debug("Metrics enabled: %s", cls.METRICS_ENABLED)
        logger.debug("Audit principal mode: %s", cls.AUDIT_PRINCIPAL_MODE)
        logger.debug("Audit query text enabled: %s", cls.AUDIT_LOG_QUERY_TEXT)
        logger.debug("Metrics namespace: %s", cls.METRICS_NAMESPACE)
        logger.debug("Metrics path: %s", cls.METRICS_PATH)
        return True

    @classmethod
    def validate_ragas(cls):
        """Validate deterministic replay and gated live Ragas settings."""
        if cls.RUN_RAGAS_EVAL and not cls.RAGAS_ENABLED:
            raise ValueError("RUN_RAGAS_EVAL requires RAGAS_ENABLED=true.")

        bounded_values = {
            "RAGAS_FAITHFULNESS_THRESHOLD": cls.RAGAS_FAITHFULNESS_THRESHOLD,
            "RAGAS_ANSWER_RELEVANCE_THRESHOLD": cls.RAGAS_ANSWER_RELEVANCE_THRESHOLD,
            "RAGAS_CONTEXT_PRECISION_THRESHOLD": cls.RAGAS_CONTEXT_PRECISION_THRESHOLD,
            "RAGAS_CONTEXT_RECALL_THRESHOLD": cls.RAGAS_CONTEXT_RECALL_THRESHOLD,
            "RAGAS_NEGATIVE_ABSTENTION_THRESHOLD": cls.RAGAS_NEGATIVE_ABSTENTION_THRESHOLD,
            "RAGAS_FAITHFULNESS_MARGIN": cls.RAGAS_FAITHFULNESS_MARGIN,
            "RAGAS_ANSWER_RELEVANCE_MARGIN": cls.RAGAS_ANSWER_RELEVANCE_MARGIN,
            "RAGAS_CONTEXT_PRECISION_MARGIN": cls.RAGAS_CONTEXT_PRECISION_MARGIN,
            "RAGAS_CONTEXT_RECALL_MARGIN": cls.RAGAS_CONTEXT_RECALL_MARGIN,
        }
        for name, value in bounded_values.items():
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be between 0 and 1.")

        if cls.RAGAS_TOP_K <= 0:
            raise ValueError("RAGAS_TOP_K must be greater than 0.")
        # Prefer an odd count so the robust median has one unambiguous center value.
        if cls.RAGAS_LIVE_REPETITIONS < 2:
            raise ValueError("RAGAS_LIVE_REPETITIONS must be at least 2.")
        if cls.RAGAS_JUDGE_TEMPERATURE < 0:
            raise ValueError("RAGAS_JUDGE_TEMPERATURE must be non-negative.")
        if cls.RAGAS_LIVE_GENERATION_TEMPERATURE < 0:
            raise ValueError("RAGAS_LIVE_GENERATION_TEMPERATURE must be non-negative.")
        if cls.RAGAS_JUDGE_TIMEOUT <= 0:
            raise ValueError("RAGAS_JUDGE_TIMEOUT must be greater than 0.")
        if cls.RAGAS_JUDGE_MAX_RETRIES < 0:
            raise ValueError("RAGAS_JUDGE_MAX_RETRIES cannot be negative.")
        if cls.RAGAS_JUDGE_MAX_TOKENS <= 0:
            raise ValueError("RAGAS_JUDGE_MAX_TOKENS must be greater than 0.")
        if cls.RAGAS_EMBEDDING_DIMENSION <= 0:
            raise ValueError("RAGAS_EMBEDDING_DIMENSION must be greater than 0.")
        if cls.RAGAS_EMBEDDING_PROVIDER.strip().lower() not in {"gemini", "bge"}:
            raise ValueError("RAGAS_EMBEDDING_PROVIDER must be one of: gemini, bge.")
        if cls.RAGAS_EMBEDDING_BATCH_SIZE <= 0:
            raise ValueError("RAGAS_EMBEDDING_BATCH_SIZE must be greater than 0.")
        if cls.RAGAS_EMBEDDING_TIMEOUT <= 0:
            raise ValueError("RAGAS_EMBEDDING_TIMEOUT must be greater than 0.")
        if cls.RAGAS_EMBEDDING_MAX_RETRIES < 0:
            raise ValueError("RAGAS_EMBEDDING_MAX_RETRIES cannot be negative.")
        if cls.RAGAS_EMBEDDING_BASE_DELAY < 0:
            raise ValueError("RAGAS_EMBEDDING_BASE_DELAY cannot be negative.")
        if cls.RAGAS_EMBEDDING_MAX_DELAY < cls.RAGAS_EMBEDDING_BASE_DELAY:
            raise ValueError("RAGAS_EMBEDDING_MAX_DELAY must be greater than or equal to base delay.")
        if cls.RAGAS_SIGMA_MULTIPLIER <= 0:
            raise ValueError("RAGAS_SIGMA_MULTIPLIER must be greater than 0.")
        if not cls.RAGAS_JUDGE_MODEL or not cls.RAGAS_JUDGE_MODEL.strip():
            raise ValueError("RAGAS_JUDGE_MODEL is required.")
        if not cls.RAGAS_FIXTURE_PATH or not cls.RAGAS_FIXTURE_PATH.strip():
            raise ValueError("RAGAS_FIXTURE_PATH is required.")
        if not cls.RAGAS_REPORT_DIR or not cls.RAGAS_REPORT_DIR.strip():
            raise ValueError("RAGAS_REPORT_DIR is required.")

        if cls.RUN_RAGAS_EVAL:
            required = {
                "RAGAS_JUDGE_API_KEY": cls.RAGAS_JUDGE_API_KEY,
                "RAGAS_JUDGE_BASE_URL": cls.RAGAS_JUDGE_BASE_URL,
                "RAGAS_EMBEDDING_MODEL": cls.RAGAS_EMBEDDING_MODEL,
                "DEEPSEEK_API_KEY": cls.API_KEY,
            }
            # Remote embedding providers need a key/url; local BGE runs offline.
            if cls.RAGAS_EMBEDDING_PROVIDER.strip().lower() != "bge":
                required["RAGAS_EMBEDDING_API_KEY"] = cls.RAGAS_EMBEDDING_API_KEY
                required["RAGAS_EMBEDDING_BASE_URL"] = cls.RAGAS_EMBEDDING_BASE_URL
            missing = [name for name, value in required.items() if not value]
            if missing:
                raise ValueError("Live Ragas configuration is missing: " + ", ".join(missing))

        logger.debug("Ragas enabled: %s", cls.RAGAS_ENABLED)
        logger.debug("Run live Ragas evaluation: %s", cls.RUN_RAGAS_EVAL)
        logger.debug("Ragas judge model: %s", cls.RAGAS_JUDGE_MODEL)
        logger.debug("Ragas live repetitions: %s", cls.RAGAS_LIVE_REPETITIONS)
        return True
