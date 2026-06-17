import math
from typing import Protocol

from logger import setup_logger

logger = setup_logger(__name__)

KNOWN_TOKENIZER_ENCODINGS = {
    "cl100k_base",
    "o200k_base",
    "p50k_base",
    "r50k_base",
    "gpt2",
}


class TokenCounter(Protocol):
    """Protocol for deterministic token counting."""

    encoding_name: str

    def count(self, text: str) -> int:
        """Return the token count for text."""
        ...


class HeuristicTokenCounter:
    """Offline deterministic token counter used when tiktoken is unavailable."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize the fallback counter."""
        validate_tokenizer_encoding(encoding_name)
        self.encoding_name = encoding_name

    def count(self, text: str) -> int:
        """Estimate tokens with deterministic CJK-aware heuristics."""
        if not text:
            return 1

        cjk_chars = sum(1 for char in text if is_cjk(char))
        non_cjk_chars = len(text) - cjk_chars
        estimated = cjk_chars + math.ceil(non_cjk_chars / 4)
        return max(1, estimated)


class TiktokenCounter:
    """Token counter backed by tiktoken encodings."""

    def __init__(self, encoding_name: str = "cl100k_base"):
        """Initialize a tiktoken-backed counter."""
        validate_tokenizer_encoding(encoding_name)
        try:
            import tiktoken
        except ImportError as e:
            raise RuntimeError("tiktoken is not installed") from e

        self.encoding_name = encoding_name
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        """Return exact token count according to the configured encoding."""
        return max(1, len(self._encoding.encode(text or "")))


def create_token_counter(encoding_name: str = "cl100k_base") -> TokenCounter:
    """Create a tiktoken counter, falling back to deterministic heuristics."""
    validate_tokenizer_encoding(encoding_name)
    try:
        return TiktokenCounter(encoding_name=encoding_name)
    except RuntimeError:
        logger.warning(
            "tiktoken is not installed; using heuristic token counter | encoding=%s",
            encoding_name,
        )
        return HeuristicTokenCounter(encoding_name=encoding_name)


def validate_tokenizer_encoding(encoding_name: str) -> None:
    """Validate tokenizer encoding names before optional imports."""
    if not encoding_name:
        raise ValueError("TOKENIZER_ENCODING is required.")
    if encoding_name not in KNOWN_TOKENIZER_ENCODINGS:
        raise ValueError(
            "TOKENIZER_ENCODING must be one of: "
            + ", ".join(sorted(KNOWN_TOKENIZER_ENCODINGS))
        )


def legacy_len_div_four_count(text: str) -> int:
    """Return the legacy len/4 token estimate used before T08."""
    if not text:
        return 1
    return max(1, math.ceil(len(text) / 4))


def token_count_error_percent(estimated_count: int, true_count: int) -> float:
    """Return absolute percentage error against a reference token count."""
    if true_count <= 0:
        raise ValueError("true_count must be greater than 0")
    return abs(estimated_count - true_count) / true_count * 100.0


def is_cjk(char: str) -> bool:
    """Return whether a character belongs to common CJK ranges."""
    codepoint = ord(char)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0xF900 <= codepoint <= 0xFAFF
    )
