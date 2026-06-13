from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass
class Document:
    """Text document or document chunk with metadata."""

    content: str
    metadata: Dict = field(default_factory=dict)

    def __repr__(self):
        preview = self.content[:50].replace("\n", " ")
        return f"Document(content='{preview}...', metadata={self.metadata})"


class DocumentLoader(ABC):
    """Base interface for all document loaders."""

    def __init__(self, file_path: str):
        """Initialize a document loader.

        Args:
            file_path: Source file path.
        """
        self.file_path = Path(file_path)
        self._validate_file()

    def _validate_file(self):
        """Validate that the source path exists and points to a file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"File does not exist: {self.file_path}")
        if not self.file_path.is_file():
            raise ValueError(f"Path is not a file: {self.file_path}")

    @abstractmethod
    def load(self) -> str:
        """Load raw text content from the source file."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict:
        """Return metadata for the source file."""
        pass

    def load_with_metadata(self) -> Document:
        """Load content and return it as a Document object."""
        content = self.load()
        metadata = self.get_metadata()
        metadata["source"] = str(self.file_path)
        return Document(content=content, metadata=metadata)
