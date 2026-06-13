import hashlib
import re
from typing import Iterable, List

from .base import EmbeddingConfig, EmbeddingProvider, Vector, normalize_vector


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashing-based embedding provider for tests and local pipelines.

    This provider is not semantic. It exists to validate ingestion, chunking,
    vector metadata, and vector-store plumbing without network access.
    """

    def __init__(
        self,
        dimension: int = 128,
        normalize: bool = True,
        model_name: str = "hash-embedding-local",
        batch_size: int = 32,
    ):
        """Initialize the deterministic hash embedding provider."""
        super().__init__(
            EmbeddingConfig(
                model_name=model_name,
                dimension=dimension,
                normalize=normalize,
                batch_size=batch_size,
            )
        )

    def embed_texts(self, texts: Iterable[str]) -> List[Vector]:
        """Embed multiple text strings."""
        return [self._embed_text(str(text)) for text in texts]

    def _embed_text(self, text: str) -> Vector:
        """Embed text with signed feature hashing."""
        vector = [0.0] * self.dimension
        tokens = self._tokenize(text)

        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            raw_value = int.from_bytes(digest, byteorder="big", signed=False)
            index = raw_value % self.dimension
            sign = 1.0 if (raw_value >> 1) % 2 == 0 else -1.0
            vector[index] += sign

        if self.config.normalize:
            vector = normalize_vector(vector)

        self._validate_vector(vector)
        return vector

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize English words, numbers, underscores, and CJK characters."""
        return re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text.lower())
