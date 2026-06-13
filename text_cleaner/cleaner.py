import re
from dataclasses import dataclass

from logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TextCleaner:
    """Conservative text cleaner for RAG ingestion."""

    normalize_whitespace: bool = True
    remove_control_chars: bool = True
    fix_line_breaks: bool = True
    remove_duplicate_blank_lines: bool = True
    strip_text: bool = True

    def clean(self, text: str) -> str:
        """Clean source text without rewriting its meaning.

        Args:
            text: Source text.

        Returns:
            Cleaned text.
        """
        if text is None:
            return ""

        cleaned = str(text)

        if self.remove_control_chars:
            cleaned = self._remove_control_chars(cleaned)

        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        if self.fix_line_breaks:
            cleaned = self._fix_line_breaks(cleaned)

        if self.normalize_whitespace:
            cleaned = self._normalize_whitespace(cleaned)

        if self.remove_duplicate_blank_lines:
            cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        if self.strip_text:
            cleaned = cleaned.strip()

        logger.debug(
            "Text clean completed | original_chars=%s | cleaned_chars=%s",
            len(text),
            len(cleaned),
        )
        return cleaned

    def _remove_control_chars(self, text: str) -> str:
        """Remove invisible control characters while preserving tabs and newlines."""
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    def _fix_line_breaks(self, text: str) -> str:
        """Merge likely accidental line breaks from extracted documents."""
        text = re.sub(r"(?<=[\u4e00-\u9fffA-Za-z0-9，,；;：:])\n(?=[\u4e00-\u9fffA-Za-z0-9])", "", text)
        text = re.sub(r"(?<=-)\n(?=[A-Za-z])", "", text)
        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize spaces and tabs line by line."""
        lines = []
        for line in text.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()
            lines.append(line)
        return "\n".join(lines)


def clean_text(text: str, **kwargs) -> str:
    """Clean text with a temporary TextCleaner instance."""
    return TextCleaner(**kwargs).clean(text)
