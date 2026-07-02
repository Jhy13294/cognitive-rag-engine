from typing import Any, Dict, List, Optional

from .base import DocumentLoader
from logger import setup_logger

logger = setup_logger(__name__)


class PDFLoader(DocumentLoader):
    """PDF document loader with optional table extraction and OCR fallback."""

    def __init__(
        self,
        file_path: str,
        extract_tables: bool = False,
        ocr_enabled: bool = False,
        ocr_min_chars: int = 1,
        ocr_dpi: int = 200,
    ):
        """Initialize the PDF loader.

        Args:
            file_path: Source PDF path.
            extract_tables: Whether to extract tables with pdfplumber.
            ocr_enabled: Whether to OCR pages with empty or near-empty text layers.
            ocr_min_chars: Minimum stripped text characters before a page is treated as scanned.
            ocr_dpi: Render DPI used for OCR images.
        """
        super().__init__(file_path)
        self.extract_tables = extract_tables
        self.ocr_enabled = ocr_enabled
        self.ocr_min_chars = ocr_min_chars
        self.ocr_dpi = ocr_dpi
        self._ocr_engine: Optional[Any] = None
        self._scanned_page_count = 0
        self._ocr_page_count = 0

        if self.ocr_min_chars <= 0:
            raise ValueError("ocr_min_chars must be greater than 0.")
        if self.ocr_dpi <= 0:
            raise ValueError("ocr_dpi must be greater than 0.")

    def load(self) -> str:
        """Load text content from a PDF file."""
        logger.info("Loading PDF file | path=%s", self.file_path)

        try:
            import fitz
        except ImportError:
            logger.error("PyMuPDF is not installed. Run: pip install PyMuPDF")
            raise

        doc = None
        pdfplumber_pdf = None

        try:
            doc = fitz.open(self.file_path)
            page_count = len(doc)
            texts = []
            self._scanned_page_count = 0
            self._ocr_page_count = 0

            if self.extract_tables:
                pdfplumber_pdf = self._open_pdfplumber()

            for page_num, page in enumerate(doc):
                logger.debug("Processing PDF page | page=%s/%s", page_num + 1, page_count)

                text = page.get_text().strip()
                if self._is_scanned_page_text(text):
                    self._scanned_page_count += 1
                    if self.ocr_enabled:
                        text = self._ocr_page(page).strip()
                        self._ocr_page_count += 1

                texts.append(f"\n--- Page {page_num + 1} ---\n{text}")

                if self.extract_tables and pdfplumber_pdf is not None:
                    for table in self._extract_tables_from_page(pdfplumber_pdf, page_num):
                        texts.append(f"\n[Table]\n{table}\n")

            full_text = "\n".join(texts).strip()
            if self._scanned_page_count:
                logger.warning(
                    "PDF scanned pages detected | path=%s | scanned_pages=%s | ocr_pages=%s | ocr_enabled=%s",
                    self.file_path,
                    self._scanned_page_count,
                    self._ocr_page_count,
                    self.ocr_enabled,
                )
            logger.info(
                "PDF loaded | pages=%s | chars=%s | scanned_pages=%s | ocr_pages=%s",
                page_count,
                len(full_text),
                self._scanned_page_count,
                self._ocr_page_count,
            )
            return full_text

        except Exception as e:
            logger.error("PDF loading failed | error=%s", e)
            raise

        finally:
            if pdfplumber_pdf is not None:
                try:
                    pdfplumber_pdf.close()
                except Exception as e:
                    logger.debug("PDF table reader close failed | error=%s", e)
            if doc is not None:
                doc.close()

    def _is_scanned_page_text(self, text: str) -> bool:
        """Return True when a page has too little embedded text to trust."""
        return len(text.strip()) < self.ocr_min_chars

    def _ocr_page(self, page) -> str:
        """OCR a rendered PDF page with RapidOCR."""
        try:
            import fitz
            import numpy as np
        except ImportError as e:
            raise ImportError(
                "PDF OCR requires rapidocr-onnxruntime and its image dependencies. "
                "Install optional OCR dependencies with: pip install rapidocr-onnxruntime"
            ) from e

        scale = self.ocr_dpi / 72.0
        pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )
        results, _ = self._get_ocr_engine()(image)
        lines = []
        for result in results or []:
            if len(result) >= 2:
                text = str(result[1]).strip()
                if text:
                    lines.append(text)
        return "\n".join(lines)

    def _get_ocr_engine(self):
        """Create the RapidOCR engine lazily only when OCR is enabled and needed."""
        if self._ocr_engine is None:
            try:
                from rapidocr_onnxruntime import RapidOCR
            except ImportError as e:
                raise ImportError(
                    "PDF OCR requires rapidocr-onnxruntime. "
                    "Install optional OCR dependencies with: pip install rapidocr-onnxruntime"
                ) from e
            self._ocr_engine = RapidOCR()
        return self._ocr_engine

    def _open_pdfplumber(self):
        """Open pdfplumber once for the whole document, or return None when unavailable."""
        try:
            import pdfplumber

            return pdfplumber.open(self.file_path)
        except ImportError:
            logger.warning("pdfplumber is not installed. Skipping PDF table extraction.")
        except Exception as e:
            logger.warning("PDF table extraction failed | error=%s", e)
        return None

    def _extract_tables_from_page(self, pdf, page_num: int) -> List[str]:
        """Extract tables from a PDF page."""
        tables = []

        try:
            if page_num >= len(pdf.pages):
                return tables

            pdf_page = pdf.pages[page_num]
            for table in pdf_page.extract_tables():
                if table:
                    tables.append(self._table_to_markdown(table))

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
                "scanned_page_count": self._scanned_page_count,
                "ocr_page_count": self._ocr_page_count,
            }
            doc.close()
            return metadata

        except ImportError:
            logger.error("PyMuPDF is not installed. Run: pip install PyMuPDF")
            raise
