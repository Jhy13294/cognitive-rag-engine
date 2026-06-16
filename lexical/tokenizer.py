import re
from typing import List

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}


def tokenize(text: str) -> List[str]:
    """Tokenize and lightly normalize English words, numbers, and CJK chars."""
    tokens = []
    for raw_token in TOKEN_PATTERN.findall(text.lower()):
        token = normalize_token(raw_token)
        if token and token not in STOPWORDS:
            tokens.append(token)
    return tokens


def normalize_token(token: str) -> str:
    """Apply deterministic light stemming."""
    if len(token) <= 3:
        return token

    for suffix in ("ingly", "edly", "ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            token = token[: -len(suffix)]
            break

    if token.endswith("e") and len(token) > 4:
        token = token[:-1]

    return token
