import inspect
from pathlib import Path
from typing import Iterable, List, Optional, Type

from .base import Document, DocumentLoader
from .chunking import ParentChildSplitResult, ParentChildSplitter, TextSplitter
from .md_loader import MDLoader
from .pdf_loader import PDFLoader
from .txt_loader import TXTLoader
from .word_loader import WordLoader
from logger import setup_logger
from text_cleaner import TextCleaner

logger = setup_logger(__name__)


LOADER_REGISTRY = {
    ".txt": TXTLoader,
    ".md": MDLoader,
    ".markdown": MDLoader,
    ".pdf": PDFLoader,
    ".docx": WordLoader,
}


def get_supported_extensions() -> List[str]:
    """Return supported file extensions."""
    return sorted(LOADER_REGISTRY.keys())


def get_document_loader(file_path: str, **loader_kwargs) -> DocumentLoader:
    """Create a document loader based on file extension.

    Args:
        file_path: Source file path.
        **loader_kwargs: Keyword arguments passed to the selected loader.

    Returns:
        A DocumentLoader instance.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    path = Path(file_path)
    extension = path.suffix.lower()
    loader_class: Optional[Type[DocumentLoader]] = LOADER_REGISTRY.get(extension)

    if not loader_class:
        supported = ", ".join(get_supported_extensions())
        raise ValueError(f"Unsupported file extension: {extension}. Supported extensions: {supported}")

    logger.debug("Selected document loader | path=%s | loader=%s", path, loader_class.__name__)
    return loader_class(str(path), **_filter_loader_kwargs(loader_class, loader_kwargs))


def _filter_loader_kwargs(loader_class: Type[DocumentLoader], loader_kwargs: dict) -> dict:
    """Return only kwargs accepted by the selected loader constructor."""
    if not loader_kwargs:
        return {}

    signature = inspect.signature(loader_class.__init__)
    parameters = signature.parameters.values()
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        return dict(loader_kwargs)

    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.name != "self"
        and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return {key: value for key, value in loader_kwargs.items() if key in accepted}


def load_document(file_path: str, clean: bool = False, **loader_kwargs) -> Document:
    """Load a single document with the appropriate loader.

    Args:
        file_path: Source file path.
        clean: Whether to clean loaded text.
        **loader_kwargs: Keyword arguments passed to the selected loader.

    Returns:
        Loaded Document object.
    """
    loader = get_document_loader(file_path, **loader_kwargs)
    document = loader.load_with_metadata()

    if clean:
        document.content = TextCleaner().clean(document.content)
        document.metadata["cleaned"] = True
    else:
        document.metadata["cleaned"] = False

    return document


def iter_supported_files(path: str, recursive: bool = True) -> Iterable[Path]:
    """Yield supported files from a file or directory path."""
    root = Path(path)

    if root.is_file():
        if root.suffix.lower() in LOADER_REGISTRY:
            yield root
        return

    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Path is neither a file nor a directory: {root}")

    pattern = "**/*" if recursive else "*"
    for candidate in root.glob(pattern):
        if candidate.is_file() and candidate.suffix.lower() in LOADER_REGISTRY:
            yield candidate


def load_documents(path: str, recursive: bool = True, clean: bool = False, **loader_kwargs) -> List[Document]:
    """Load all supported documents from a file or directory path."""
    documents = []
    for file_path in iter_supported_files(path, recursive=recursive):
        documents.append(load_document(str(file_path), clean=clean, **loader_kwargs))

    logger.info("Documents loaded | path=%s | count=%s", path, len(documents))
    return documents


def load_and_split_document(
    file_path: str,
    clean: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    **loader_kwargs,
) -> List[Document]:
    """Load, optionally clean, and split a single document."""
    document = load_document(file_path, clean=clean, **loader_kwargs)
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_document(document)

    for chunk in chunks:
        chunk.metadata["loader_entrypoint"] = "load_and_split_document"

    logger.info("Document loaded and split | path=%s | chunks=%s", file_path, len(chunks))
    return chunks


def load_and_split_documents(
    path: str,
    recursive: bool = True,
    clean: bool = True,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    **loader_kwargs,
) -> List[Document]:
    """Load and split all supported documents from a file or directory path."""
    documents = load_documents(path, recursive=recursive, clean=clean, **loader_kwargs)
    splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)

    for chunk in chunks:
        chunk.metadata["loader_entrypoint"] = "load_and_split_documents"

    logger.info("Documents loaded and split | path=%s | chunks=%s", path, len(chunks))
    return chunks


def load_and_split_documents_hierarchical(
    path: str,
    recursive: bool = True,
    clean: bool = True,
    parent_chunk_size: int = 1600,
    parent_chunk_overlap: int = 200,
    child_chunk_size: int = 400,
    child_chunk_overlap: int = 80,
    **loader_kwargs,
) -> ParentChildSplitResult:
    """Load and split documents into parent chunks and child chunks."""
    documents = load_documents(path, recursive=recursive, clean=clean, **loader_kwargs)
    splitter = ParentChildSplitter(
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
    )
    result = splitter.split_documents(documents)

    for parent in result.parents:
        parent.metadata["loader_entrypoint"] = "load_and_split_documents_hierarchical"
    for child in result.children:
        child.metadata["loader_entrypoint"] = "load_and_split_documents_hierarchical"

    logger.info(
        "Documents loaded and split hierarchically | path=%s | parents=%s | children=%s",
        path,
        len(result.parents),
        len(result.children),
    )
    return result
