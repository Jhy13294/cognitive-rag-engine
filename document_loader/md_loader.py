import re
from typing import Dict, List, Optional, Tuple

from .base import DocumentLoader
from logger import setup_logger

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

logger = setup_logger(__name__)


class MDLoader(DocumentLoader):
    """Markdown document loader for RAG ingestion."""

    _CODE_BLOCK_PATTERN = re.compile(
        r"^```(?P<language>[A-Za-z0-9_+.\-]*)[ \t]*\r?\n"
        r"(?P<code>.*?)^[ \t]*```[ \t]*(?=\r?$)",
        flags=re.DOTALL | re.MULTILINE,
    )

    def __init__(
        self,
        file_path: str,
        keep_markdown_syntax: bool = False,
        extract_code_blocks: Optional[bool] = None,
        remove_links: bool = True,
        use_yaml_lib: bool = True,
        code_block_mode: Optional[str] = None,
    ):
        """Initialize the Markdown loader.

        Args:
            file_path: Source file path.
            keep_markdown_syntax: Whether to keep Markdown markup.
            extract_code_blocks: Compatibility switch. True replaces fenced code
                with metadata placeholders; False drops it. When omitted, code
                bodies remain searchable in the document text.
            remove_links: Whether to remove URLs while keeping link text.
            use_yaml_lib: Whether to prefer PyYAML for Front Matter parsing.
            code_block_mode: Explicit ``preserve``, ``extract``, or ``drop`` mode.
        """
        super().__init__(file_path)
        if code_block_mode is not None and extract_code_blocks is not None:
            raise ValueError("Set either code_block_mode or extract_code_blocks, not both")
        if code_block_mode is None:
            if extract_code_blocks is None:
                code_block_mode = "preserve"
            else:
                code_block_mode = "extract" if extract_code_blocks else "drop"
        if code_block_mode not in {"preserve", "extract", "drop"}:
            raise ValueError("code_block_mode must be one of: preserve, extract, drop")

        self.keep_markdown_syntax = keep_markdown_syntax
        self.code_block_mode = code_block_mode
        self.extract_code_blocks = code_block_mode == "extract"
        self.remove_links = remove_links
        self.use_yaml_lib = use_yaml_lib and HAS_YAML
        self.code_blocks: List[Dict] = []
        self._front_matter: Dict = {}
        self._headings: List[Dict] = []

    def load(self) -> str:
        """Load and normalize Markdown content."""
        logger.info("Loading Markdown file | path=%s", self.file_path)
        self.code_blocks = []
        self._headings = []

        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read()

            logger.debug("Raw Markdown chars | chars=%s", len(content))

            metadata, content = self._extract_front_matter(content)
            self._front_matter = metadata

            if self.keep_markdown_syntax:
                self._record_code_blocks(content)
                processed_content = content.strip()
            else:
                processed_content = self._convert_to_plain_text(content)

            logger.info("Markdown loaded | chars=%s", len(processed_content))
            return processed_content

        except UnicodeDecodeError as e:
            logger.error("Markdown decoding failed | error=%s", e)
            raise

        except Exception as e:
            logger.error("Markdown loading failed | error=%s", e)
            raise

    def get_metadata(self) -> Dict:
        """Return Markdown file metadata."""
        return {
            "file_type": "markdown",
            "file_size": self.file_path.stat().st_size,
            "title": self._front_matter.get("title", self.file_path.stem),
            "has_front_matter": bool(self._front_matter),
            "front_matter": dict(self._front_matter),
            "headings": list(self._headings),
            "code_block_count": len(self.code_blocks),
            "code_blocks": list(self.code_blocks),
        }

    def _extract_front_matter(self, content: str) -> Tuple[Dict, str]:
        """Extract YAML Front Matter from Markdown content."""
        if not content or not content.lstrip().startswith("---"):
            return {}, content

        front_matter_pattern = r"^\s*---\s*\n(.*?)\n---\s*(?:\n|$)"
        match = re.match(front_matter_pattern, content, re.DOTALL)

        if not match:
            return {}, content

        yaml_content = match.group(1)
        remaining_content = content[match.end():].lstrip("\n")
        yaml_stripped = yaml_content.strip()

        if not yaml_stripped or all(
            line.strip().startswith("#") or not line.strip()
            for line in yaml_stripped.split("\n")
        ):
            return {}, remaining_content

        if self.use_yaml_lib:
            return self._parse_with_pyyaml(yaml_content, remaining_content)

        return self._parse_simple_yaml(yaml_content, remaining_content)

    def _parse_with_pyyaml(self, yaml_content: str, remaining_content: str) -> Tuple[Dict, str]:
        """Parse Front Matter with PyYAML."""
        try:
            metadata = yaml.safe_load(yaml_content)
            if metadata is None:
                return {}, remaining_content
            if not isinstance(metadata, dict):
                logger.warning("Front Matter is not a mapping. Ignoring it.")
                return {}, remaining_content

            metadata["has_front_matter"] = True
            metadata["yaml_parser"] = "pyyaml"
            return metadata, remaining_content

        except yaml.YAMLError as e:
            logger.warning("PyYAML parsing failed. Falling back to simple parser. error=%s", e)
            return self._parse_simple_yaml(yaml_content, remaining_content)

    def _parse_simple_yaml(self, yaml_content: str, remaining_content: str) -> Tuple[Dict, str]:
        """Parse simple key-value YAML without nested structures."""
        metadata = {}

        for line in yaml_content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("\"'")

            if not key:
                continue

            if value.startswith("[") and value.endswith("]"):
                value = [item.strip().strip("\"'") for item in value[1:-1].split(",") if item.strip()]
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            elif value.isdigit():
                value = int(value)
            elif self._is_float(value):
                value = float(value)

            metadata[key] = value

        if metadata:
            metadata["has_front_matter"] = True
            metadata["yaml_parser"] = "simple"

        return metadata, remaining_content

    def _is_float(self, value: str) -> bool:
        """Return whether a string represents a float."""
        try:
            float(value)
            return "." in value
        except ValueError:
            return False

    def _convert_to_plain_text(self, content: str) -> str:
        """Convert Markdown content into plain text for indexing."""
        text = content

        protected_code_blocks = {}
        if self.code_block_mode == "extract":
            text = self._extract_and_replace_code_blocks(text)
        elif self.code_block_mode == "drop":
            text = self._remove_code_blocks(text)
        else:
            text, protected_code_blocks = self._protect_code_blocks(text)

        text = self._convert_tables(text)
        text = self._remove_images(text)

        if self.remove_links:
            text = self._remove_links(text)
        else:
            text = self._convert_links(text)

        text = self._convert_headers(text)
        text = self._remove_formatting(text)
        text = self._convert_lists(text)
        text = self._convert_blockquotes(text)
        text = re.sub(r"(?m)^\s*[-*_]{3,}\s*$", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        if protected_code_blocks:
            token_pattern = "|".join(re.escape(token) for token in protected_code_blocks)
            text = re.sub(token_pattern, lambda match: protected_code_blocks[match.group(0)], text)

        return text.strip()

    def _extract_and_replace_code_blocks(self, text: str) -> str:
        """Extract fenced code blocks and replace them with placeholders."""
        def replace_code(match):
            language, _, index = self._record_code_block(match)
            return f"\n[Code block {index + 1}: {language}]\n"

        return self._CODE_BLOCK_PATTERN.sub(replace_code, text)

    def _protect_code_blocks(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Protect code bodies while the surrounding Markdown is normalized."""
        protected = {}

        def protect_code(match):
            _, code, index = self._record_code_block(match)
            token = f"CODEBLOCKPRESERVE{index:08d}TOKEN"
            protected[token] = code
            return f"\n{token}\n"

        return self._CODE_BLOCK_PATTERN.sub(protect_code, text), protected

    def _record_code_blocks(self, text: str) -> None:
        """Record fenced code metadata without changing Markdown content."""
        for match in self._CODE_BLOCK_PATTERN.finditer(text):
            self._record_code_block(match)

    def _record_code_block(self, match) -> Tuple[str, str, int]:
        """Append one fenced block to metadata and return its normalized fields."""
        language = match.group("language") or "unknown"
        code = match.group("code")
        if code.endswith("\r\n"):
            code = code[:-2]
        elif code.endswith("\n"):
            code = code[:-1]
        index = len(self.code_blocks)
        self.code_blocks.append(
            {
                "language": language,
                "code": code,
                "index": index,
                "char_count": len(code),
            }
        )
        return language, code, index

    def _remove_code_blocks(self, text: str) -> str:
        """Remove fenced code blocks."""
        return self._CODE_BLOCK_PATTERN.sub("", text)

    def _convert_tables(self, text: str) -> str:
        """Keep table cell text while removing Markdown table syntax."""
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if re.match(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$", stripped):
                continue
            if "|" in stripped and stripped.count("|") >= 2:
                cells = [cell.strip() for cell in stripped.strip("|").split("|")]
                lines.append(" | ".join(cell for cell in cells if cell))
            else:
                lines.append(line)
        return "\n".join(lines)

    def _remove_links(self, text: str) -> str:
        """Remove link URLs while preserving link text."""
        return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    def _convert_links(self, text: str) -> str:
        """Expand Markdown links into text plus URL."""
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)

    def _remove_images(self, text: str) -> str:
        """Remove image markup while preserving alt text."""
        return re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

    def _convert_headers(self, text: str) -> str:
        """Remove heading markers and record heading metadata."""
        headings = []

        def replace_header(match):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append({"level": level, "title": title})
            return title

        converted = re.sub(r"(?m)^(#{1,6})\s+(.+)$", replace_header, text)
        self._headings = headings
        return converted

    def _remove_formatting(self, text: str) -> str:
        """Remove common inline Markdown formatting markers."""
        text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
        text = re.sub(r"(\*|_)(.*?)\1", r"\2", text)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        text = re.sub(r"~~(.*?)~~", r"\1", text)
        return text

    def _convert_lists(self, text: str) -> str:
        """Remove list markers while preserving item text."""
        text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
        text = re.sub(r"(?m)^\s*\d+\.\s+", "", text)
        return text

    def _convert_blockquotes(self, text: str) -> str:
        """Remove blockquote markers while preserving quote text."""
        return re.sub(r"(?m)^\s*>\s?", "", text)
