from typing import Dict, List

from .base import DocumentLoader
from logger import setup_logger

logger = setup_logger(__name__)


class PDFLoader(DocumentLoader):
    """PDF document loader with optional table extraction."""

    def __init__(self, file_path: str, extract_tables: bool = False):
        """Initialize the PDF loader.

        Args:
            file_path: Source PDF path.
            extract_tables: Whether to extract tables with pdfplumber.
        """
        super().__init__(file_path)
        self.extract_tables = extract_tables

    def load(self) -> str:
        """Load text content from a PDF file."""
        logger.info("Loading PDF file | path=%s", self.file_path)

        try:
            import fitz

            doc = fitz.open(self.file_path)
            page_count = len(doc)
            texts = []

            for page_num, page in enumerate(doc):
                logger.debug("Processing PDF page | page=%s/%s", page_num + 1, page_count)

                text = page.get_text().strip()
                texts.append(f"\n--- Page {page_num + 1} ---\n{text}")

                if self.extract_tables:
                    for table in self._extract_tables_from_page(page_num):
                        texts.append(f"\n[Table]\n{table}\n")

            doc.close()

            full_text = "\n".join(texts).strip()
            logger.info("PDF loaded | pages=%s | chars=%s", page_count, len(full_text))
            return full_text

        except ImportError:
            logger.error("PyMuPDF is not installed. Run: pip install PyMuPDF")
            raise

        except Exception as e:
            logger.error("PDF loading failed | error=%s", e)
            raise

    def _extract_tables_from_page(self, page_num: int) -> List[str]:
        """Extract tables from a PDF page."""
        tables = []

        try:
            import pdfplumber

            with pdfplumber.open(self.file_path) as pdf:
                if page_num >= len(pdf.pages):
                    return tables

                pdf_page = pdf.pages[page_num]
                for table in pdf_page.extract_tables():
                    if table:
                        tables.append(self._table_to_markdown(table))

        except ImportError:
            logger.warning("pdfplumber is not installed. Skipping PDF table extraction.")

        except Exception as e:
            logger.warning("PDF table extraction failed | error=%s", e)

        return tables

    def _table_to_markdown(self, table: List[List[str]]) -> str:
        """Convert a table matrix into Markdown table text."""
        if not table:
            return ""

        lines = []
        for index, row in enumerate(table):
            cleaned_row = [str(cell).strip() if cell else "" for cell in row]
            lines.append("| " + " | ".join(cleaned_row) + " |")

            if index == 0:
                lines.append("| " + " | ".join(["---"] * len(row)) + " |")

        return "\n".join(lines)

    def get_metadata(self) -> Dict:
        """Return PDF file metadata."""
        try:
            import fitz

            doc = fitz.open(self.file_path)
            metadata = {
                "file_type": "pdf",
                "total_pages": len(doc),
                "author": doc.metadata.get("author") or "Unknown",
                "title": doc.metadata.get("title") or self.file_path.stem,
                "file_size": self.file_path.stat().st_size,
            }
            doc.close()
            return metadata

        except ImportError:
            logger.error("PyMuPDF is not installed. Run: pip install PyMuPDF")
            raise
