import unittest

from document_loader import load_and_split_document
from embeddings import HashEmbeddingProvider
from rag import RAGPipeline
from rag.pipeline import extract_chat_content
from tests.test_document_ingestion import FIXTURES_DIR
from vector_store import InMemoryVectorStore


class FakeChatClient:
    def __init__(self, answer="The project is a RAG ingestion prototype [1]."):
        self.answer = answer
        self.calls = []

    def chat(self, message, system_prompt=None):
        self.calls.append({"message": message, "system_prompt": system_prompt})
        return {
            "choices": [
                {
                    "message": {
                        "content": self.answer,
                    }
                }
            ]
        }


def build_test_pipeline():
    chunks = load_and_split_document(
        str(FIXTURES_DIR / "sample.txt"),
        clean=True,
        chunk_size=80,
        chunk_overlap=10,
    )
    embedding_provider = HashEmbeddingProvider(dimension=64)
    embedded_documents = embedding_provider.embed_documents(chunks)
    vector_store = InMemoryVectorStore(dimension=64)
    vector_store.add_documents(embedded_documents)
    chat_client = FakeChatClient()

    return RAGPipeline(
        embedding_provider=embedding_provider,
        vector_store=vector_store,
        chat_client=chat_client,
        top_k=3,
        max_context_chars=1000,
    ), chat_client


class RAGPipelineTests(unittest.TestCase):
    def test_retrieve_returns_sources_with_scores_and_metadata(self):
        pipeline, _ = build_test_pipeline()

        sources = pipeline.retrieve("What is this project?")

        self.assertGreaterEqual(len(sources), 1)
        self.assertEqual(sources[0].index, 1)
        self.assertIsInstance(sources[0].score, float)
        self.assertIn("source", sources[0].metadata)
        self.assertIn("chunk_index", sources[0].metadata)

    def test_build_prompt_includes_context_question_and_citation_numbers(self):
        pipeline, _ = build_test_pipeline()
        sources = pipeline.retrieve("What is this project?", top_k=1)

        prompt = pipeline.build_prompt("What is this project?", sources)

        self.assertIn("Context:", prompt)
        self.assertIn("Question:", prompt)
        self.assertIn("[1]", prompt)
        self.assertIn("What is this project?", prompt)

    def test_answer_calls_chat_client_and_returns_response(self):
        pipeline, chat_client = build_test_pipeline()

        response = pipeline.answer("What is this project?")

        self.assertEqual(response.answer, "The project is a RAG ingestion prototype [1].")
        self.assertEqual(response.question, "What is this project?")
        self.assertGreaterEqual(len(response.sources), 1)
        self.assertEqual(len(chat_client.calls), 1)
        self.assertIn("Use the context below", chat_client.calls[0]["message"])
        self.assertIn("enterprise knowledge-base assistant", chat_client.calls[0]["system_prompt"])

    def test_answer_supports_metadata_filter(self):
        pipeline, _ = build_test_pipeline()
        response = pipeline.answer(
            "What is this project?",
            metadata_filter={"file_type": "txt"},
        )

        self.assertGreaterEqual(len(response.sources), 1)
        self.assertTrue(all(source.metadata["file_type"] == "txt" for source in response.sources))

    def test_prompt_context_is_bounded(self):
        pipeline, _ = build_test_pipeline()
        pipeline.max_context_chars = 60
        sources = pipeline.retrieve("What is this project?", top_k=3)

        prompt = pipeline.build_prompt("What is this project?", sources)

        self.assertIn("Context:", prompt)
        context = prompt.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        self.assertLessEqual(len(context), 70)

    def test_extract_chat_content_rejects_invalid_response(self):
        with self.assertRaises(ValueError):
            extract_chat_content({"invalid": []})


if __name__ == "__main__":
    unittest.main()
