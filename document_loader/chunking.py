from dataclasses import dataclass, field
from typing import Iterable, List, Tuple

from logger import setup_logger

from .base import Document

logger = setup_logger(__name__)


@dataclass(frozen=True)
class _TextSpan:
    """Half-open source-text span used while splitting and merging."""

    start: int
    end: int


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

        spans = self._split_text_spans(text)
        logger.debug("Text split completed | chunks=%s", len(spans))
        return [text[span.start : span.end] for span in spans]

    def split_document(self, document: Document) -> List[Document]:
        """Split a Document while preserving and extending metadata."""
        spans = self._split_text_spans(document.content)
        documents = []

        for index, span in enumerate(spans):
            chunk = document.content[span.start : span.end]

            metadata = dict(document.metadata)
            metadata.update(
                {
                    "chunk_index": index,
                    "total_chunks": len(spans),
                    "start_char": span.start,
                    "end_char": span.end,
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

    def _split_text_spans(self, text: str) -> List[_TextSpan]:
        """Return exact source spans for all chunks."""
        start, end = self._trim_span(text, 0, len(text))
        if start >= end:
            return []

        pieces = self._split_recursive_spans(text, start, end, self.separators)
        return self._merge_spans(pieces)

    def _split_recursive_spans(
        self,
        text: str,
        start: int,
        end: int,
        separators: List[str],
    ) -> List[_TextSpan]:
        """Recursively split a source range without rewriting its text."""
        start, end = self._trim_span(text, start, end)
        if start >= end:
            return []
        if end - start <= self.chunk_size:
            return [_TextSpan(start, end)]
        if not separators or separators[0] == "":
            return self._fixed_spans(start, end)

        separator = separators[0]
        remaining_separators = separators[1:]
        ranges = []
        cursor = start
        separator_found = False

        while cursor < end:
            separator_start = text.find(separator, cursor, end)
            if separator_start == -1:
                ranges.append((cursor, end))
                break

            separator_found = True
            ranges.append((cursor, separator_start))
            cursor = separator_start + len(separator)

        if not separator_found:
            return self._split_recursive_spans(text, start, end, remaining_separators)

        spans = []
        for piece_start, piece_end in ranges:
            spans.extend(
                self._split_recursive_spans(
                    text,
                    piece_start,
                    piece_end,
                    remaining_separators,
                )
            )
        return spans

    def _fixed_spans(self, start: int, end: int) -> List[_TextSpan]:
        """Split a range into overlapping fixed-width source spans."""
        spans = []
        cursor = start

        while cursor < end:
            chunk_end = min(cursor + self.chunk_size, end)
            spans.append(_TextSpan(cursor, chunk_end))
            if chunk_end == end:
                break
            cursor = chunk_end - self.chunk_overlap

        return spans

    def _merge_spans(self, pieces: List[_TextSpan]) -> List[_TextSpan]:
        """Merge source spans while keeping every output within the size limit."""
        if not pieces:
            return []

        chunks = []
        current = pieces[0]

        for piece in pieces[1:]:
            candidate = _TextSpan(current.start, max(current.end, piece.end))
            if candidate.end - candidate.start <= self.chunk_size:
                current = candidate
                continue

            chunks.append(current)
            if piece.start < current.end:
                current = piece
                continue

            overlap_start = max(
                current.start,
                current.end - self.chunk_overlap,
                piece.end - self.chunk_size,
            )
            if overlap_start < current.end:
                current = _TextSpan(overlap_start, piece.end)
            else:
                current = piece

        chunks.append(current)
        return chunks

    @staticmethod
    def _trim_span(text: str, start: int, end: int) -> Tuple[int, int]:
        """Trim only a span's boundaries while retaining source coordinates."""
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end


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
        parent_spans = parent_splitter._split_text_spans(document.content)
        parents = []
        children = []
        child_index = 0

        for parent_index, parent_span in enumerate(parent_spans):
            parent_start = parent_span.start
            parent_end = parent_span.end
            parent_text = document.content[parent_start:parent_end]

            parent_metadata = dict(document.metadata)
            parent_metadata.update(
                {
                    "parent_index": parent_index,
                    "total_parents": len(parent_spans),
                    "start_char": parent_start,
                    "end_char": parent_end,
                    "parent_chunk_size": len(parent_text),
                }
            )
            parent_metadata["parent_id"] = build_parent_id(parent_text, parent_metadata)
            parent = Document(content=parent_text, metadata=parent_metadata)
            parents.append(parent)

            child_spans = child_splitter._split_text_spans(parent_text)
            for child_span in child_spans:
                relative_start = child_span.start
                relative_end = child_span.end
                child_text = parent_text[relative_start:relative_end]
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


def locate_text_span(
    text: str,
    needle: str,
    search_start: int = 0,
    document_id: str = "unknown",
) -> Tuple[int, int]:
    """Locate a text span for compatibility, warning if fallback is required."""
    start_char = text.find(needle, search_start)
    if start_char == -1:
        start_char = text.find(needle)
    if start_char == -1:
        logger.warning(
            "Text span lookup failed; using bounded fallback | document_id=%s | "
            "needle_chars=%s | search_start=%s",
            document_id,
            len(needle),
            search_start,
        )
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


def split_document(
    document: Document, chunk_size: int = 800, chunk_overlap: int = 120
) -> List[Document]:
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
