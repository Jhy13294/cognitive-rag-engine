from typing import Dict

from .base import DocumentLoader
from logger import setup_logger

logger = setup_logger(__name__)


class TXTLoader(DocumentLoader):
    """Plain text document loader with simple encoding detection."""

    def __init__(self, file_path: str, encoding: str = "auto"):
        """Initialize the TXT loader.

        Args:
            file_path: Source file path.
            encoding: File encoding, or "auto" for detection.
        """
        super().__init__(file_path)
        self.encoding = encoding

    def load(self) -> str:
        """Load text content from a TXT file."""
        logger.info("Loading TXT file | path=%s", self.file_path)

        if self.encoding == "auto":
            self.encoding = self._detect_encoding()
            logger.debug("Detected encoding | encoding=%s", self.encoding)

        try:
            with open(self.file_path, "r", encoding=self.encoding) as f:
                content = f.read()

            logger.info("TXT loaded | chars=%s", len(content))
            return content

        except UnicodeDecodeError as e:
            logger.error("TXT decoding failed | error=%s", e)
            raise ValueError(f"Cannot read file with encoding {self.encoding}. Try another encoding.")

        except Exception as e:
            logger.error("TXT loading failed | error=%s", e)
            raise

    def _detect_encoding(self) -> str:
        """Detect file encoding by trying common encodings."""
        encodings = ["utf-8", "gbk", "gb2312", "utf-16", "latin-1"]

        for encoding in encodings:
            try:
                with open(self.file_path, "r", encoding=encoding) as f:
                    f.read(1024)
                return encoding
            except (UnicodeDecodeError, LookupError):
                continue

        return "utf-8"

    def get_metadata(self) -> Dict:
        """Return TXT file metadata."""
        return {
            "file_type": "txt",
            "encoding": self.encoding,
            "file_size": self.file_path.stat().st_size,
        }
