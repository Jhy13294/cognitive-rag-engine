import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from rag_cli import (
    apply_acl_metadata_from_bindings,
    build_cli_metadata_filter,
    build_loader_kwargs_from_config,
    build_arg_parser,
    build_rag_pipeline_from_index,
    build_rag_pipeline_from_path,
    create_embedding_provider,
    format_response,
    ingest_documents,
    parse_acl_values,
    parse_metadata_filter,
    run_single_question,
)
from config import Config
from document_loader import Document
from embeddings import HashEmbeddingProvider
from tests.test_document_ingestion import FIXTURES_DIR
from vector_store import InMemoryVectorStore


class FakeChatClient:
    def __init__(self):
        self.calls = []

    def chat(self, message, system_prompt=None):
        self.calls.append({"message": message, "system_prompt": system_prompt})
        return {
            "choices": [
                {
                    "message": {
                        "content": "CLI answer with citation [1].",
                    }
                }
            ]
        }


class CountingHashEmbeddingProvider(HashEmbeddingProvider):
    """Hash embedding provider that records document and query embedding calls."""

    def __init__(self, dimension=64):
        """Initialize the counting provider."""
        super().__init__(dimension=dimension)
        self.embed_documents_calls = 0
        self.embed_text_calls = 0

    def embed_documents(self, documents):
        """Count corpus embedding calls."""
        self.embed_documents_calls += 1
        return super().embed_documents(documents)

    def embed_text(self, text):
        """Count query embedding calls."""
        self.embed_text_calls += 1
        return super().embed_text(text)


class RAGCLITests(unittest.TestCase):
    def test_build_arg_parser_parses_core_options(self):
        parser = build_arg_parser()
        args = parser.parse_args(
            [
                "tests/fixtures",
                "--question",
                "What is this?",
                "--top-k",
                "3",
                "--chunk-size",
                "100",
                "--chunk-overlap",
                "10",
                "--embedding-dimension",
                "64",
                "--embedding-provider",
                "hash",
                "--vector-store",
                "memory",
                "--rerank-provider",
                "none",
                "--rerank-fetch-k",
                "8",
                "--hybrid",
                "--hybrid-fetch-k",
                "12",
                "--rrf-k",
                "50",
                "--hybrid-dense-weight",
                "0.1",
                "--hybrid-sparse-weight",
                "1.0",
                "--bm25-k1",
                "1.4",
                "--bm25-b",
                "0.7",
                "--parent-child",
                "--parent-chunk-size",
                "1600",
                "--parent-chunk-overlap",
                "200",
                "--child-chunk-size",
                "400",
                "--child-chunk-overlap",
                "80",
                "--context-packing",
                "--context-dedup",
                "--context-near-dup",
                "--context-near-dup-threshold",
                "0.85",
                "--context-max-tokens",
                "512",
                "--tokenizer-encoding",
                "cl100k_base",
                "--multi-query",
                "--query-rewrite-provider",
                "deterministic",
                "--query-rewrite-fixture",
                "eval/fixtures/query_rewrites.jsonl",
                "--query-rewrite-num-queries",
                "3",
                "--query-rewrite-temperature",
                "0.1",
                "--no-query-rewrite-cache",
                "--query-rewrite-weight-original",
                "1.0",
                "--query-rewrite-weight-variant",
                "0.7",
                "--principal",
                "alice",
                "--acl",
                "role:finance,role:admin",
                "--acl",
                "role:hr",
                "--no-clean",
                "--non-recursive",
            ]
        )

        self.assertEqual(args.path, "tests/fixtures")
        self.assertEqual(args.command, "oneshot")
        self.assertEqual(args.question, "What is this?")
        self.assertEqual(args.top_k, 3)
        self.assertEqual(args.chunk_size, 100)
        self.assertEqual(args.chunk_overlap, 10)
        self.assertEqual(args.embedding_dimension, 64)
        self.assertEqual(args.embedding_provider, "hash")
        self.assertEqual(args.vector_store, "memory")
        self.assertEqual(args.rerank_provider, "none")
        self.assertEqual(args.rerank_fetch_k, 8)
        self.assertTrue(args.hybrid)
        self.assertEqual(args.hybrid_fetch_k, 12)
        self.assertEqual(args.rrf_k, 50)
        self.assertEqual(args.hybrid_dense_weight, 0.1)
        self.assertEqual(args.hybrid_sparse_weight, 1.0)
        self.assertEqual(args.bm25_k1, 1.4)
        self.assertEqual(args.bm25_b, 0.7)
        self.assertTrue(args.parent_child)
        self.assertEqual(args.parent_chunk_size, 1600)
        self.assertEqual(args.parent_chunk_overlap, 200)
        self.assertEqual(args.child_chunk_size, 400)
        self.assertEqual(args.child_chunk_overlap, 80)
        self.assertTrue(args.context_packing)
        self.assertTrue(args.context_dedup)
        self.assertTrue(args.context_near_dup)
        self.assertEqual(args.context_near_dup_threshold, 0.85)
        self.assertEqual(args.context_max_tokens, 512)
        self.assertEqual(args.tokenizer_encoding, "cl100k_base")
        self.assertTrue(args.multi_query)
        self.assertEqual(args.query_rewrite_provider, "deterministic")
        self.assertEqual(args.query_rewrite_fixture, "eval/fixtures/query_rewrites.jsonl")
        self.assertEqual(args.query_rewrite_num_queries, 3)
        self.assertEqual(args.query_rewrite_temperature, 0.1)
        self.assertTrue(args.no_query_rewrite_cache)
        self.assertEqual(args.query_rewrite_weight_original, 1.0)
        self.assertEqual(args.query_rewrite_weight_variant, 0.7)
        self.assertEqual(args.principal, "alice")
        self.assertEqual(args.acl, ["role:finance,role:admin", "role:hr"])
        self.assertTrue(args.no_clean)
        self.assertTrue(args.non_recursive)

    def test_build_arg_parser_parses_ingest_and_query_subcommands(self):
        parser = build_arg_parser()

        ingest_args = parser.parse_args(["ingest", "tests/fixtures", "--embedding-provider", "hash"])
        query_args = parser.parse_args(["query", "What is indexed?", "--embedding-provider", "hash"])
        oneshot_args = parser.parse_args(["oneshot", "tests/fixtures", "-q", "What is this?"])

        self.assertEqual(ingest_args.command, "ingest")
        self.assertEqual(ingest_args.path, "tests/fixtures")
        self.assertIsNone(ingest_args.question)
        self.assertEqual(query_args.command, "query")
        self.assertIsNone(query_args.path)
        self.assertEqual(query_args.question, "What is indexed?")
        self.assertEqual(oneshot_args.command, "oneshot")
        self.assertEqual(oneshot_args.path, "tests/fixtures")

    def test_parse_metadata_filter_accepts_json_object(self):
        metadata_filter = parse_metadata_filter('{"file_type": "txt"}')

        self.assertEqual(metadata_filter, {"file_type": "txt"})

    def test_parse_metadata_filter_rejects_invalid_json(self):
        with self.assertRaises(ValueError):
            parse_metadata_filter("{bad json")

    def test_parse_metadata_filter_rejects_non_object_json(self):
        with self.assertRaises(ValueError):
            parse_metadata_filter('["txt"]')

    def test_parse_acl_values_accepts_repeated_and_comma_separated_values(self):
        values = parse_acl_values(["role:finance, role:admin", "role:finance"])

        self.assertEqual(values, ["role:admin", "role:finance"])

    def test_build_cli_metadata_filter_adds_acl_when_supplied(self):
        original_enabled = Config.ACL_ENABLED
        try:
            Config.ACL_ENABLED = False
            metadata_filter = build_cli_metadata_filter(
                {"source": "finance.md"},
                principal=None,
                allowed_acl=["role:finance"],
            )

            self.assertEqual(metadata_filter, {"source": "finance.md", "acl": ["role:finance"]})
        finally:
            Config.ACL_ENABLED = original_enabled

    def test_build_loader_kwargs_from_config_uses_pdf_flags(self):
        original_values = {
            "PDF_EXTRACT_TABLES": Config.PDF_EXTRACT_TABLES,
            "PDF_OCR_ENABLED": Config.PDF_OCR_ENABLED,
            "PDF_OCR_MIN_CHARS": Config.PDF_OCR_MIN_CHARS,
            "PDF_OCR_DPI": Config.PDF_OCR_DPI,
        }
        try:
            Config.PDF_EXTRACT_TABLES = True
            Config.PDF_OCR_ENABLED = True
            Config.PDF_OCR_MIN_CHARS = 2
            Config.PDF_OCR_DPI = 220

            self.assertEqual(
                build_loader_kwargs_from_config(),
                {
                    "extract_tables": True,
                    "ocr_enabled": True,
                    "ocr_min_chars": 2,
                    "ocr_dpi": 220,
                },
            )
        finally:
            for name, value in original_values.items():
                setattr(Config, name, value)

    def test_build_loader_kwargs_rejects_invalid_pdf_ocr_threshold(self):
        original_value = Config.PDF_OCR_MIN_CHARS
        try:
            Config.PDF_OCR_MIN_CHARS = 0

            with self.assertRaisesRegex(ValueError, "PDF_OCR_MIN_CHARS"):
                build_loader_kwargs_from_config()
        finally:
            Config.PDF_OCR_MIN_CHARS = original_value

    def test_ingest_documents_passes_pdf_loader_config(self):
        original_values = {
            "PDF_EXTRACT_TABLES": Config.PDF_EXTRACT_TABLES,
            "PDF_OCR_ENABLED": Config.PDF_OCR_ENABLED,
            "PDF_OCR_MIN_CHARS": Config.PDF_OCR_MIN_CHARS,
            "PDF_OCR_DPI": Config.PDF_OCR_DPI,
        }
        captured_kwargs = {}

        def fake_load_and_split_documents(path, **kwargs):
            captured_kwargs.update(kwargs)
            return [
                Document(
                    content="PDF chunk",
                    metadata={
                        "source": "sample.pdf",
                        "chunk_index": 0,
                        "start_char": 0,
                        "end_char": 9,
                    },
                )
            ]

        try:
            Config.PDF_EXTRACT_TABLES = True
            Config.PDF_OCR_ENABLED = True
            Config.PDF_OCR_MIN_CHARS = 3
            Config.PDF_OCR_DPI = 240
            provider = HashEmbeddingProvider(dimension=64)
            store = InMemoryVectorStore(dimension=64)

            with mock.patch("rag_cli.load_and_split_documents", side_effect=fake_load_and_split_documents):
                ingest_documents(
                    "ignored-path",
                    clean=True,
                    recursive=True,
                    chunk_size=100,
                    chunk_overlap=10,
                    embedding_provider=provider,
                    vector_store=store,
                    parent_child_enabled=False,
                )

            self.assertTrue(captured_kwargs["extract_tables"])
            self.assertTrue(captured_kwargs["ocr_enabled"])
            self.assertEqual(captured_kwargs["ocr_min_chars"], 3)
            self.assertEqual(captured_kwargs["ocr_dpi"], 240)
        finally:
            for name, value in original_values.items():
                setattr(Config, name, value)

    def test_ingest_documents_pdf_table_config_reaches_chunks(self):
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF is not installed")

        class FakePage:
            def extract_tables(self):
                return [[["Key", "Value"], ["Owner", "Finance"]]]

        class FakePlumberPDF:
            pages = [FakePage()]

            def close(self):
                return None

        original_values = {
            "PDF_EXTRACT_TABLES": Config.PDF_EXTRACT_TABLES,
            "PDF_OCR_ENABLED": Config.PDF_OCR_ENABLED,
            "PDF_OCR_MIN_CHARS": Config.PDF_OCR_MIN_CHARS,
            "PDF_OCR_DPI": Config.PDF_OCR_DPI,
        }
        try:
            Config.PDF_EXTRACT_TABLES = True
            Config.PDF_OCR_ENABLED = False
            Config.PDF_OCR_MIN_CHARS = 1
            Config.PDF_OCR_DPI = 200

            with tempfile.TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "table.pdf"
                doc = fitz.open()
                page = doc.new_page()
                page.insert_text((72, 72), "PDF table ingest fixture.", fontsize=14)
                doc.save(path)
                doc.close()

                def fake_open(open_path):
                    return FakePlumberPDF()

                provider = HashEmbeddingProvider(dimension=64)
                store = InMemoryVectorStore(dimension=64)
                with mock.patch.dict(sys.modules, {"pdfplumber": types.SimpleNamespace(open=fake_open)}):
                    result = ingest_documents(
                        str(path),
                        clean=False,
                        recursive=True,
                        chunk_size=1000,
                        chunk_overlap=0,
                        embedding_provider=provider,
                        vector_store=store,
                        parent_child_enabled=False,
                    )

            self.assertEqual(len(result.records), 1)
            self.assertIn("[Table]", result.records[0].content)
            self.assertIn("| Key | Value |", result.records[0].content)
            self.assertIn("| Owner | Finance |", result.records[0].content)
        finally:
            for name, value in original_values.items():
                setattr(Config, name, value)

    def test_build_rag_pipeline_from_path_with_fake_client(self):
        chat_client = FakeChatClient()
        pipeline = build_rag_pipeline_from_path(
            str(FIXTURES_DIR),
            chat_client=chat_client,
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider_name="hash",
            embedding_dimension=64,
            vector_store_name="memory",
            rerank_provider_name="none",
            parent_child_enabled=False,
            top_k=2,
            max_context_chars=1000,
        )

        response = run_single_question(pipeline, "What is this project?", top_k=2)

        self.assertEqual(response.answer, "CLI answer with citation [1].")
        self.assertEqual(len(chat_client.calls), 1)
        self.assertGreaterEqual(len(response.sources), 1)

    def test_ingest_query_split_does_not_embed_corpus_during_query(self):
        provider = CountingHashEmbeddingProvider(dimension=64)
        store = InMemoryVectorStore(dimension=64)
        ingest_documents(
            str(FIXTURES_DIR),
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider=provider,
            vector_store=store,
            parent_child_enabled=False,
        )

        pipeline = build_rag_pipeline_from_index(
            chat_client=FakeChatClient(),
            embedding_provider=provider,
            vector_store=store,
            rerank_provider_name="none",
            parent_child_enabled=False,
            top_k=2,
            max_context_chars=1000,
        )
        response = run_single_question(pipeline, "What is this project?", top_k=2)

        self.assertEqual(response.answer, "CLI answer with citation [1].")
        self.assertEqual(provider.embed_documents_calls, 1)
        self.assertEqual(provider.embed_text_calls, 1)

    def test_ingest_documents_denormalizes_acl_metadata(self):
        provider = HashEmbeddingProvider(dimension=64)
        store = InMemoryVectorStore(dimension=64)

        result = ingest_documents(
            str(FIXTURES_DIR),
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider=provider,
            vector_store=store,
            parent_child_enabled=False,
            acl=["role:finance", "role:admin"],
        )

        self.assertGreater(len(result.records), 0)
        for record in result.records:
            self.assertEqual(record.metadata["acl"], ["role:admin", "role:finance"])

    def test_apply_acl_metadata_from_bindings_uses_document_source(self):
        class FakeBindingResolver:
            def __init__(self):
                self.calls = []

            def acl_for_source(self, source):
                self.calls.append(source)
                return ["role:finance", "role:admin"]

        chunks = [
            Document(content="a", metadata={"source": "finance.md", "chunk_index": 0}),
            Document(content="b", metadata={"source": "finance.md", "chunk_index": 1}),
        ]
        resolver = FakeBindingResolver()

        apply_acl_metadata_from_bindings(chunks, resolver)

        self.assertEqual(resolver.calls, ["finance.md"])
        for chunk in chunks:
            self.assertEqual(chunk.metadata["acl"], ["role:admin", "role:finance"])

    def test_double_ingest_is_idempotent_and_keeps_top_k(self):
        provider = HashEmbeddingProvider(dimension=64)
        store = InMemoryVectorStore(dimension=64)

        first = ingest_documents(
            str(FIXTURES_DIR),
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider=provider,
            vector_store=store,
            parent_child_enabled=False,
        )
        query_embedding = provider.embed_text("RAG ingestion prototype")
        first_top_ids = [result.record.id for result in store.similarity_search(query_embedding, top_k=3)]

        second = ingest_documents(
            str(FIXTURES_DIR),
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider=provider,
            vector_store=store,
            parent_child_enabled=False,
        )
        second_top_ids = [result.record.id for result in store.similarity_search(query_embedding, top_k=3)]

        self.assertEqual(store.count(), len(first.records))
        self.assertEqual(len(second.records), len(first.records))
        self.assertEqual(first_top_ids, second_top_ids)

    def test_query_rejects_embedding_dimension_mismatch(self):
        store = InMemoryVectorStore(dimension=64)
        ingest_documents(
            str(FIXTURES_DIR),
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider=HashEmbeddingProvider(dimension=64),
            vector_store=store,
            parent_child_enabled=False,
        )

        with self.assertRaisesRegex(ValueError, "Embedding dimension mismatch"):
            build_rag_pipeline_from_index(
                chat_client=FakeChatClient(),
                embedding_provider=HashEmbeddingProvider(dimension=32),
                vector_store=store,
                rerank_provider_name="none",
                parent_child_enabled=False,
            )

    def test_query_rebuilds_hybrid_and_parent_store_without_corpus_embedding(self):
        provider = CountingHashEmbeddingProvider(dimension=64)
        store = InMemoryVectorStore(dimension=64)
        ingest_documents(
            str(FIXTURES_DIR),
            clean=True,
            recursive=True,
            embedding_provider=provider,
            vector_store=store,
            parent_child_enabled=True,
            parent_chunk_size=160,
            parent_chunk_overlap=20,
            child_chunk_size=80,
            child_chunk_overlap=10,
        )

        pipeline = build_rag_pipeline_from_index(
            chat_client=FakeChatClient(),
            embedding_provider=provider,
            vector_store=store,
            rerank_provider_name="none",
            hybrid_enabled=True,
            hybrid_fetch_k=8,
            parent_child_enabled=True,
            top_k=2,
            max_context_chars=1000,
        )
        sources = pipeline.retrieve("What is this project?", top_k=2)

        self.assertIsNotNone(pipeline.bm25_retriever)
        self.assertIsNotNone(pipeline.parent_store)
        self.assertGreater(pipeline.parent_store.count(), 0)
        self.assertGreaterEqual(len(sources), 1)
        self.assertEqual(provider.embed_documents_calls, 1)
        self.assertEqual(provider.embed_text_calls, 1)

    def test_explicit_rerank_provider_enables_reranker(self):
        pipeline = build_rag_pipeline_from_path(
            str(FIXTURES_DIR),
            chat_client=FakeChatClient(),
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider_name="hash",
            embedding_dimension=64,
            vector_store_name="memory",
            rerank_provider_name="deterministic",
            rerank_fetch_k=8,
            parent_child_enabled=False,
            top_k=2,
            max_context_chars=1000,
        )

        self.assertIsNotNone(pipeline.reranker)
        self.assertEqual(pipeline.fetch_k, 8)

    def test_explicit_hybrid_enables_bm25_retriever(self):
        pipeline = build_rag_pipeline_from_path(
            str(FIXTURES_DIR),
            chat_client=FakeChatClient(),
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider_name="hash",
            embedding_dimension=64,
            vector_store_name="memory",
            rerank_provider_name="none",
            hybrid_enabled=True,
            hybrid_fetch_k=8,
            rrf_k=50,
            hybrid_dense_weight=0.2,
            hybrid_sparse_weight=1.0,
            parent_child_enabled=False,
            top_k=2,
            max_context_chars=1000,
        )

        self.assertIsNotNone(pipeline.bm25_retriever)
        self.assertEqual(pipeline.fetch_k, 8)
        self.assertEqual(pipeline.rrf.config.k, 50)

    def test_explicit_parent_child_enables_parent_store(self):
        pipeline = build_rag_pipeline_from_path(
            str(FIXTURES_DIR),
            chat_client=FakeChatClient(),
            clean=True,
            recursive=True,
            embedding_provider_name="hash",
            embedding_dimension=64,
            vector_store_name="memory",
            rerank_provider_name="none",
            parent_child_enabled=True,
            parent_chunk_size=160,
            parent_chunk_overlap=20,
            child_chunk_size=80,
            child_chunk_overlap=10,
            top_k=2,
            max_context_chars=1000,
        )

        self.assertIsNotNone(pipeline.parent_store)
        self.assertTrue(pipeline.expand_parent_context)
        self.assertGreater(pipeline.parent_store.count(), 0)

    def test_explicit_context_packing_enables_context_packer(self):
        pipeline = build_rag_pipeline_from_path(
            str(FIXTURES_DIR),
            chat_client=FakeChatClient(),
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider_name="hash",
            embedding_dimension=64,
            vector_store_name="memory",
            rerank_provider_name="none",
            parent_child_enabled=False,
            context_packing_enabled=True,
            context_dedup_enabled=True,
            context_near_dup_enabled=False,
            context_max_tokens=512,
            tokenizer_encoding="cl100k_base",
            top_k=2,
            max_context_chars=1000,
        )

        self.assertTrue(pipeline.context_packing_enabled)
        self.assertTrue(pipeline.context_dedup_enabled)
        self.assertIsNotNone(pipeline.context_packer)
        self.assertEqual(pipeline.context_max_tokens, 512)

    def test_explicit_multi_query_enables_query_rewriter(self):
        pipeline = build_rag_pipeline_from_path(
            str(FIXTURES_DIR),
            chat_client=FakeChatClient(),
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider_name="hash",
            embedding_dimension=64,
            vector_store_name="memory",
            rerank_provider_name="none",
            parent_child_enabled=False,
            query_rewrite_enabled=True,
            query_rewrite_provider_name="deterministic",
            query_rewrite_fixture_path="eval/fixtures/query_rewrites.jsonl",
            query_rewrite_num_queries=3,
            query_rewrite_weight_original=1.0,
            query_rewrite_weight_variant=0.7,
            top_k=2,
            max_context_chars=1000,
        )

        self.assertTrue(pipeline.query_rewrite_enabled)
        self.assertIsNotNone(pipeline.query_rewriter)
        self.assertEqual(pipeline.query_rewrite_weight_original, 1.0)
        self.assertGreaterEqual(pipeline.query_rewrite_weight_original, pipeline.query_rewrite_weight_variant)

    def test_format_response_includes_answer_and_sources(self):
        chat_client = FakeChatClient()
        pipeline = build_rag_pipeline_from_path(
            str(FIXTURES_DIR / "sample.txt"),
            chat_client=chat_client,
            clean=True,
            recursive=True,
            chunk_size=100,
            chunk_overlap=10,
            embedding_provider_name="hash",
            embedding_dimension=64,
            vector_store_name="memory",
            rerank_provider_name="none",
            parent_child_enabled=False,
            top_k=2,
            max_context_chars=1000,
        )
        response = run_single_question(pipeline, "What is this project?", top_k=2)

        output = format_response(response, show_prompt=True)

        self.assertIn("Answer", output)
        self.assertIn("CLI answer with citation [1].", output)
        self.assertIn("Sources", output)
        self.assertIn("[1]", output)
        self.assertIn("Prompt", output)

    def test_create_embedding_provider_can_create_hash_provider(self):
        provider = create_embedding_provider(provider_name="hash", embedding_dimension=32)

        self.assertEqual(provider.dimension, 32)


if __name__ == "__main__":
    unittest.main()
