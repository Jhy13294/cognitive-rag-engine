from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

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


@dataclass
class ParentChildSplitResult:
    """Hierarchical split output for parent-child RAG ingestion."""

    parents: List[Document]
    children: List[Document]


@dataclass
class ParentChildSplitter:
    """Two-level splitter that indexes child chunks and stores parent chunks."""

    parent_chunk_size: int = 1600
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 80
    separators: List[str] = field(
        default_factory=lambda: ["\n\n", "\n", "。", "；", "，", ".", ";", ",", " ", ""]
    )

    def __post_init__(self):
        """Validate parent-child splitter configuration."""
        if self.parent_chunk_size <= 0:
            raise ValueError("parent_chunk_size must be greater than 0")
        if self.child_chunk_size <= 0:
            raise ValueError("child_chunk_size must be greater than 0")
        if self.parent_chunk_overlap < 0:
            raise ValueError("parent_chunk_overlap cannot be negative")
        if self.child_chunk_overlap < 0:
            raise ValueError("child_chunk_overlap cannot be negative")
        if self.parent_chunk_overlap >= self.parent_chunk_size:
            raise ValueError("parent_chunk_overlap must be smaller than parent_chunk_size")
        if self.child_chunk_overlap >= self.child_chunk_size:
            raise ValueError("child_chunk_overlap must be smaller than child_chunk_size")
        if self.child_chunk_size >= self.parent_chunk_size:
            raise ValueError("child_chunk_size must be smaller than parent_chunk_size")

    def split_document(self, document: Document) -> ParentChildSplitResult:
        """Split one document into parent chunks and child chunks."""
        parent_splitter = TextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            separators=list(self.separators),
        )
        child_splitter = TextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.child_chunk_overlap,
            separators=list(self.separators),
        )
        parent_texts = parent_splitter.split_text(document.content)
        parents = []
        children = []
        parent_search_start = 0
        child_index = 0

        for parent_index, parent_text in enumerate(parent_texts):
            parent_start, parent_end = locate_text_span(
                document.content,
                parent_text,
                parent_search_start,
            )
            parent_text = document.content[parent_start:parent_end]
            parent_search_start = max(parent_start + 1, parent_end - self.parent_chunk_overlap)

            parent_metadata = dict(document.metadata)
            parent_metadata.update(
                {
                    "parent_index": parent_index,
                    "total_parents": len(parent_texts),
                    "start_char": parent_start,
                    "end_char": parent_end,
                    "parent_chunk_size": len(parent_text),
                }
            )
            parent_metadata["parent_id"] = build_parent_id(parent_text, parent_metadata)
            parent = Document(content=parent_text, metadata=parent_metadata)
            parents.append(parent)

            child_texts = child_splitter.split_text(parent_text)
            child_search_start = 0
            for child_text in child_texts:
                relative_start, relative_end = locate_text_span(parent_text, child_text, child_search_start)
                child_text = parent_text[relative_start:relative_end]
                child_search_start = max(relative_start + 1, relative_end - self.child_chunk_overlap)
                child_start = parent_start + relative_start
                child_end = parent_start + relative_end

                child_metadata = dict(document.metadata)
                child_metadata.update(
                    {
                        "chunk_index": child_index,
                        "start_char": child_start,
                        "end_char": child_end,
                        "chunk_size": len(child_text),
                        "parent_id": parent.metadata["parent_id"],
                        "parent_index": parent_index,
                        "parent_start_char": parent_start,
                        "parent_end_char": parent_end,
                    }
                )
                children.append(Document(content=child_text, metadata=child_metadata))
                child_index += 1

        for child in children:
            child.metadata["total_chunks"] = len(children)

        return ParentChildSplitResult(parents=parents, children=children)

    def split_documents(self, documents: Iterable[Document]) -> ParentChildSplitResult:
        """Split multiple documents into parent chunks and child chunks."""
        all_parents = []
        all_children = []

        for document in documents:
            result = self.split_document(document)
            all_parents.extend(result.parents)
            all_children.extend(result.children)

        return ParentChildSplitResult(parents=all_parents, children=all_children)


def locate_text_span(text: str, needle: str, search_start: int = 0) -> Tuple[int, int]:
    """Locate a chunk span in a source text with deterministic fallback."""
    start_char = text.find(needle, search_start)
    if start_char == -1:
        start_char = text.find(needle)
    if start_char == -1:
        start_char = min(max(search_start, 0), len(text))
    end_char = min(start_char + len(needle), len(text))
    return start_char, end_char


def build_parent_id(content: str, metadata: dict) -> str:
    """Build a parent id with the existing deterministic record-id primitive."""
    from vector_store.base import build_record_id

    return build_record_id(content, metadata)


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> List[str]:
    """Split plain text with default splitter settings."""
    return TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_text(text)


def split_document(document: Document, chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    """Split a Document with default splitter settings."""
    return TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap).split_document(document)


def split_document_hierarchical(
    document: Document,
    parent_chunk_size: int = 1600,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 80,
) -> ParentChildSplitResult:
    """Split a Document into parent chunks and child chunks."""
    return ParentChildSplitter(
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
    ).split_document(document)
