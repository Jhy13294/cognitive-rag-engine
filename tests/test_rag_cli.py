import unittest

from rag_cli import (
    build_arg_parser,
    build_rag_pipeline_from_path,
    create_embedding_provider,
    format_response,
    parse_metadata_filter,
    run_single_question,
)
from tests.test_document_ingestion import FIXTURES_DIR


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
                "--no-clean",
                "--non-recursive",
            ]
        )

        self.assertEqual(args.path, "tests/fixtures")
        self.assertEqual(args.question, "What is this?")
        self.assertEqual(args.top_k, 3)
        self.assertEqual(args.chunk_size, 100)
        self.assertEqual(args.chunk_overlap, 10)
        self.assertEqual(args.embedding_dimension, 64)
        self.assertEqual(args.embedding_provider, "hash")
        self.assertEqual(args.vector_store, "memory")
        self.assertEqual(args.rerank_provider, "none")
        self.assertEqual(args.rerank_fetch_k, 8)
        self.assertTrue(args.no_clean)
        self.assertTrue(args.non_recursive)

    def test_parse_metadata_filter_accepts_json_object(self):
        metadata_filter = parse_metadata_filter('{"file_type": "txt"}')

        self.assertEqual(metadata_filter, {"file_type": "txt"})

    def test_parse_metadata_filter_rejects_invalid_json(self):
        with self.assertRaises(ValueError):
            parse_metadata_filter("{bad json")

    def test_parse_metadata_filter_rejects_non_object_json(self):
        with self.assertRaises(ValueError):
            parse_metadata_filter('["txt"]')

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
            top_k=2,
            max_context_chars=1000,
        )

        response = run_single_question(pipeline, "What is this project?", top_k=2)

        self.assertEqual(response.answer, "CLI answer with citation [1].")
        self.assertEqual(len(chat_client.calls), 1)
        self.assertGreaterEqual(len(response.sources), 1)

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
            top_k=2,
            max_context_chars=1000,
        )

        self.assertIsNotNone(pipeline.reranker)
        self.assertEqual(pipeline.fetch_k, 8)

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
