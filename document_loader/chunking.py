from dataclasses import dataclass, field
from typing import Iterable, List

from .base import Document
from logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TextSplitter:
    """Recursive text splitter for RAG ingestion."""

    chunk_size: int = 800
    chunk_overlap: int = 120
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", "。", "；", "，", ".", ";", ",", " ", ""]
    )

    def __post_init__(self):
        """Validate splitter configuration."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

    def split_text(self, text: str) -> List[str]:
        """Split plain text into chunks.

        Args:
            text: Source text.

        Returns:
            List of chunk strings.
        """
        if not text:
            return []

        pieces = self._split_recursive(text.strip(), self.separators)
        chunks = self._merge_pieces(pieces)
        logger.debug("Text split completed | chunks=%s", len(chunks))
        return chunks

    def split_document(self, document: Document) -> List[Document]:
        """Split a Document while preserving and extending metadata."""
        chunks = self.split_text(document.content)
        documents = []
        search_start = 0

        for index, chunk in enumerate(chunks):
            start_char = document.content.find(chunk, search_start)
            if start_char == -1:
                start_char = search_start

            end_char = start_char + len(chunk)
            search_start = max(start_char + 1, end_char - self.chunk_overlap)

            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": len(chunks),
                    "start_char": start_char,
                    "end_char": end_char,
                    "chunk_size": len(chunk),
                }
            )
            documents.append(Document(content=chunk, metadata=metadata))

        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split multiple Document objects."""
        chunks = []
        for document in documents:
            chunks.extend(self.split_document(document))
        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text by preferred separators."""
        if len(text) <= self.chunk_size:
            return [text]

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

        pieces = text.split(separator)
        if len(pieces) == 1:
            return self._split_recursive(text, remaining_separators)

        split_pieces = []
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue

            if len(piece) <= self.chunk_size:
                split_pieces.append(piece)
            else:
                split_pieces.extend(self._split_recursive(piece, remaining_separators))

        return split_pieces

    def _merge_pieces(self, pieces: List[str]) -> List[str]:
        """Merge small pieces into chunks near the target size."""
        chunks = []
        current = ""

        for piece in pieces:
            if not piece:
                continue

            candidate = f"{current}\n{piece}".strip() if current else piece

            if len(candidate) <= self.chunk_size:
                current = candidate
                continue

            if current:
                chunks.append(current)
                current = self._build_overlap(current, piece)
            else:
                chunks.append(piece[:self.chunk_size])
                current = piece[self.chunk_size - self.chunk_overlap:]

        if current:
            chunks.append(current)

        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def _build_overlap(self, previous_chunk: str, next_piece: str) -> str:
        """Build overlap context from the previous chunk tail."""
        if self.chunk_overlap == 0:
            return next_piece

        overlap = previous_chunk[-self.chunk_overlap:].strip()
        candidate = f"{overlap}\n{next_piece}".strip()

        if len(candidate) <= self.chunk_size:
            return candidate

        return candidate[-self.chunk_size:]


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """Split plain text with default splitter settings."""
    return TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_text(text)


def split_document(document: Document, chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    """Split a Document with default splitter settings."""
    return TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_document(document)
