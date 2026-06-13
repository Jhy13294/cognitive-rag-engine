import tempfile
import unittest
from pathlib import Path

from document_loader import (
    MDLoader,
    PDFLoader,
    TXTLoader,
    WordLoader,
    get_document_loader,
    get_supported_extensions,
    load_and_split_document,
    load_and_split_documents,
    load_document,
    load_documents,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"


class DocumentLoaderEntrypointTests(unittest.TestCase):
    def test_supported_extensions_include_core_formats(self):
        extensions = get_supported_extensions()

        self.assertIn(".txt", extensions)
        self.assertIn(".md", extensions)
        self.assertIn(".markdown", extensions)
        self.assertIn(".pdf", extensions)
        self.assertIn(".docx", extensions)

    def test_get_document_loader_selects_loader_by_extension(self):
        self.assertIsInstance(get_document_loader(str(FIXTURES_DIR / "sample.txt")), TXTLoader)
        self.assertIsInstance(get_document_loader(str(FIXTURES_DIR / "sample.md")), MDLoader)

    def test_get_document_loader_rejects_unsupported_extension(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.csv"
            path.write_text("a,b,c", encoding="utf-8")

            with self.assertRaises(ValueError):
                get_document_loader(str(path))

    def test_load_document_loads_txt_with_metadata(self):
        document = load_document(str(FIXTURES_DIR / "sample.txt"), clean=True)

        self.assertIn("RAG ingestion prototype", document.content)
        self.assertEqual(document.metadata["file_type"], "txt")
        self.assertTrue(document.metadata["cleaned"])
        self.assertIn("source", document.metadata)

    def test_load_document_loads_markdown_front_matter(self):
        document = load_document(str(FIXTURES_DIR / "sample.md"), clean=True)

        self.assertEqual(document.metadata["file_type"], "markdown")
        self.assertEqual(document.metadata["title"], "Sample Markdown Fixture")
        self.assertTrue(document.metadata["has_front_matter"])
        self.assertEqual(document.metadata["code_block_count"], 1)
        self.assertIn("Knowledge Base Notes", document.content)
        self.assertIn("project notes", document.content)

    def test_load_documents_loads_supported_files_from_directory(self):
        documents = load_documents(str(FIXTURES_DIR), clean=True)
        file_types = sorted(document.metadata["file_type"] for document in documents)

        self.assertEqual(file_types, ["markdown", "txt"])

    def test_load_and_split_document_adds_chunk_metadata(self):
        chunks = load_and_split_document(
            str(FIXTURES_DIR / "sample.txt"),
            clean=True,
            chunk_size=80,
            chunk_overlap=10,
        )

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["chunk_index"], 0)
        self.assertEqual(chunks[0].metadata["total_chunks"], len(chunks))
        self.assertIn("start_char", chunks[0].metadata)
        self.assertIn("end_char", chunks[0].metadata)
        self.assertEqual(chunks[0].metadata["loader_entrypoint"], "load_and_split_document")

    def test_load_and_split_documents_splits_directory(self):
        chunks = load_and_split_documents(
            str(FIXTURES_DIR),
            clean=True,
            chunk_size=90,
            chunk_overlap=10,
        )

        sources = {Path(chunk.metadata["source"]).name for chunk in chunks}
        self.assertEqual(sources, {"sample.md", "sample.txt"})
        self.assertTrue(all("chunk_index" in chunk.metadata for chunk in chunks))


class GeneratedDocumentLoaderTests(unittest.TestCase):
    def test_word_loader_with_generated_docx(self):
        try:
            from docx import Document as DocxDocument
        except ImportError:
            self.skipTest("python-docx is not installed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.docx"
            doc = DocxDocument()
            doc.add_paragraph("Generated Word fixture for loader verification.")
            table = doc.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "Key"
            table.cell(0, 1).text = "Value"
            doc.save(path)

            loader = get_document_loader(str(path))
            loaded = loader.load_with_metadata()

            self.assertIsInstance(loader, WordLoader)
            self.assertEqual(loaded.metadata["file_type"], "docx")
            self.assertIn("Generated Word fixture", loaded.content)
            self.assertIn("[Table 1]", loaded.content)

    def test_pdf_loader_with_generated_pdf(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is not installed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample.pdf"
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((72, 72), "Generated PDF fixture for loader verification.")
            doc.save(path)
            doc.close()

            loader = get_document_loader(str(path))
            loaded = loader.load_with_metadata()

            self.assertIsInstance(loader, PDFLoader)
            self.assertEqual(loaded.metadata["file_type"], "pdf")
            self.assertEqual(loaded.metadata["total_pages"], 1)
            self.assertIn("Generated PDF fixture", loaded.content)


if __name__ == "__main__":
    unittest.main()
