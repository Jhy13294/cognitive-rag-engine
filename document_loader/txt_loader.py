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
        self._raise_if_probably_binary()

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
            raise ValueError(
                f"Cannot read file with encoding {self.encoding}. Try another encoding."
            ) from None

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

    def _raise_if_probably_binary(self) -> None:
        """Reject binary payloads before latin-1 can decode arbitrary bytes."""
        with open(self.file_path, "rb") as f:
            sample = f.read(4096)

        if not sample:
            return
        if sample.startswith((b"\xff\xfe", b"\xfe\xff", b"\xef\xbb\xbf")):
            return
        if b"\x00" in sample:
            raise ValueError("TXT file appears to be binary; refusing to load as text.")

        allowed_controls = {9, 10, 13}
        control_count = sum(1 for byte in sample if byte < 32 and byte not in allowed_controls)
        if control_count / len(sample) > 0.30:
            raise ValueError("TXT file appears to be binary; refusing to load as text.")

    def get_metadata(self) -> Dict:
        """Return TXT file metadata."""
        return {
            "file_type": "txt",
            "encoding": self.encoding,
            "file_size": self.file_path.stat().st_size,
        }
