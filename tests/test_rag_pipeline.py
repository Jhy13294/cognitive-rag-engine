import unittest
from dataclasses import asdict

from document_loader import load_and_split_document
from embeddings import HashEmbeddingProvider
from rag import RAGPipeline
from rerank import RerankConfig, RerankResult, Reranker
from rag.pipeline import (
    CANONICAL_ABSTENTION_RESPONSE,
    extract_chat_content,
    is_action_list_question,
    is_name_only_lookup_question,
    is_relationship_entity_question,
    is_scenario_procedure_question,
    is_source_relevance_question,
)
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


class ReverseReranker(Reranker):
    def __init__(self):
        super().__init__(RerankConfig(model_name="reverse-test", top_n=2, fetch_k=3))

    def rerank(self, query, candidates, top_n=None):
        limit = min(top_n or self.top_n, len(candidates))
        results = []
        for index in reversed(range(len(candidates))):
            candidate = candidates[index]
            results.append(
                RerankResult(
                    index=index,
                    score=100.0 - index,
                    content=candidate.content,
                    metadata={"test_reranked": True},
                )
            )
        return results[:limit]


class FailingReranker(Reranker):
    def __init__(self):
        super().__init__(RerankConfig(model_name="failing-test", top_n=2, fetch_k=3))

    def rerank(self, query, candidates, top_n=None):
        raise RuntimeError("planned rerank failure")


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
        self.assertIn("When the question does not ask for source or document names", prompt)
        self.assertIn("not a scenario-based procedure or workaround request", prompt)
        self.assertIn("do not broaden the answer with extra triggers", prompt)

    def test_build_prompt_gates_scenario_scope_instruction_by_question_shape(self):
        pipeline, _ = build_test_pipeline()
        sources = pipeline.retrieve("What is this project?", top_k=1)
        scenario_question = "A new employee arrives late and the building badge is not ready. Where is the workaround?"
        paraphrase_question = "How quickly should the team declare the most serious customer-impacting incident?"

        scenario_prompt = pipeline.build_prompt(scenario_question, sources)
        paraphrase_prompt = pipeline.build_prompt(paraphrase_question, sources)

        self.assertIn("This is a workaround question", scenario_prompt)
        self.assertIn("source-listed workaround actions", scenario_prompt)
        self.assertIn("source-supported condition", scenario_prompt)
        self.assertIn("not a scenario-based procedure or workaround request", paraphrase_prompt)
        self.assertIn("do not broaden the answer with extra triggers", paraphrase_prompt)
        self.assertIn("timing, threshold, approval, or requirement questions", paraphrase_prompt)
        self.assertIn("source-supported condition that makes it apply", paraphrase_prompt)

    def test_scenario_question_classifier_does_not_widen_paraphrase_questions(self):
        self.assertTrue(
            is_scenario_procedure_question(
                "A new employee arrives late and the building badge is not ready. Where is the workaround?"
            )
        )
        self.assertTrue(
            is_scenario_procedure_question(
                "Search results look stale after ingestion. Which checks should support run?"
            )
        )
        self.assertFalse(
            is_scenario_procedure_question(
                "What should a new engineer complete in the first few business days?"
            )
        )
        self.assertFalse(
            is_scenario_procedure_question(
                "How quickly should the team declare the most serious customer-impacting incident?"
            )
        )

    def test_build_prompt_gates_entity_scope_instruction_by_question_shape(self):
        pipeline, _ = build_test_pipeline()
        sources = pipeline.retrieve("What is this project?", top_k=1)

        document_prompt = pipeline.build_prompt(
            "Which document contains KB-209 password reset workflow?",
            sources,
        )
        service_prompt = pipeline.build_prompt(
            "Which service routes telemetry from connected devices into regional ingestion clusters?",
            sources,
        )
        source_relevance_prompt = pipeline.build_prompt(
            "An employee loses a hardware token while traveling. Which sources are relevant?",
            sources,
        )
        task_prompt = pipeline.build_prompt(
            "What should a new engineer complete in the first few business days?",
            sources,
        )
        checks_prompt = pipeline.build_prompt(
            "Search results look stale after ingestion. Which checks should support run?",
            sources,
        )

        self.assertIn("source/document/runbook/policy lookup", document_prompt)
        self.assertIn("preserves the lookup relation and topic", document_prompt)
        self.assertIn("The document/source that contains or defines", document_prompt)
        self.assertIn("Do not answer with only the name or citation", document_prompt)
        self.assertNotIn("only that supported responsibility", document_prompt)
        self.assertIn("which sources or documents are relevant", source_relevance_prompt)
        self.assertIn("source names may be the main answer", source_relevance_prompt)
        self.assertIn("Do not answer with only citation numbers", source_relevance_prompt)
        self.assertNotIn("When the question does not ask for source or document names", source_relevance_prompt)
        self.assertIn("tool, service, or workflow satisfies a described responsibility", service_prompt)
        self.assertIn("preserves the requested responsibility", service_prompt)
        self.assertIn("Do not answer with only the entity name", service_prompt)
        self.assertIn("do not list adjacent responsibilities", service_prompt)
        self.assertIn("actions, tasks, or approvers", task_prompt)
        self.assertIn("do not frame the answer around a checklist or workflow name", task_prompt)
        self.assertIn("source-listed actions or checks using the source's verbs", checks_prompt)
        self.assertIn("avoid adding or rephrasing scenario conditions", checks_prompt)

    def test_entity_and_action_classifiers_cover_regression_shapes(self):
        self.assertTrue(
            is_name_only_lookup_question("Which document contains KB-209 password reset workflow?")
        )
        self.assertTrue(
            is_name_only_lookup_question("Which runbook mentions AtlasDB read replica lag?")
        )
        self.assertTrue(
            is_name_only_lookup_question("What source defines EXP-204 mileage reimbursement?")
        )
        self.assertFalse(
            is_name_only_lookup_question(
                "Which service routes telemetry from connected devices into regional ingestion clusters?"
            )
        )
        self.assertTrue(
            is_source_relevance_question(
                "An employee loses a hardware token while traveling. Which sources are relevant?"
            )
        )
        self.assertFalse(
            is_source_relevance_question("Which document contains KB-209 password reset workflow?")
        )
        self.assertTrue(
            is_relationship_entity_question(
                "Which service routes telemetry from connected devices into regional ingestion clusters?"
            )
        )
        self.assertTrue(
            is_action_list_question("What should a new engineer complete in the first few business days?")
        )
        self.assertTrue(
            is_action_list_question("Who must approve an expensive invoice above the threshold?")
        )

    def test_answer_calls_chat_client_and_returns_response(self):
        pipeline, chat_client = build_test_pipeline()

        response = pipeline.answer("What is this project?")

        self.assertEqual(response.answer, "The project is a RAG ingestion prototype [1].")
        self.assertEqual(response.question, "What is this project?")
        self.assertGreaterEqual(len(response.sources), 1)
        self.assertEqual(len(chat_client.calls), 1)
        self.assertIn("Use the context below", chat_client.calls[0]["message"])
        self.assertIn("enterprise knowledge-base assistant", chat_client.calls[0]["system_prompt"])

    def test_default_prompt_pins_canonical_abstention_response(self):
        pipeline, chat_client = build_test_pipeline()

        pipeline.answer("What is not covered by the knowledge base?")

        system_prompt = chat_client.calls[0]["system_prompt"]
        self.assertIn(f'begin with exactly this sentence: "{CANONICAL_ABSTENTION_RESPONSE}"', system_prompt)
        self.assertIn("directly supported", system_prompt)
        self.assertIn("do not supply an unsupported answer", system_prompt)
        self.assertIn("Match the scope and wording of the user's question", system_prompt)
        self.assertIn("Use the source's action verbs", system_prompt)
        self.assertIn("preserve the lookup relation", system_prompt)
        self.assertIn("never answer with only a bare name", system_prompt)
        self.assertIn("which sources or documents are relevant", system_prompt)
        self.assertIn("preserve the requested responsibility", system_prompt)
        self.assertIn("do not list adjacent responsibilities", system_prompt)

    def test_answer_supports_metadata_filter(self):
        pipeline, _ = build_test_pipeline()
        response = pipeline.answer(
            "What is this project?",
            metadata_filter={"file_type": "txt"},
        )

        self.assertGreaterEqual(len(response.sources), 1)
        self.assertTrue(all(source.metadata["file_type"] == "txt" for source in response.sources))

    def test_retrieve_without_reranker_matches_dense_output(self):
        dense_pipeline, _ = build_test_pipeline()
        default_pipeline, _ = build_test_pipeline()
        default_pipeline.reranker = None
        default_pipeline.fetch_k = 99

        dense_sources = dense_pipeline.retrieve("What is this project?", top_k=3)
        default_sources = default_pipeline.retrieve("What is this project?", top_k=3)

        self.assertEqual([asdict(source) for source in dense_sources], [asdict(source) for source in default_sources])

    def test_retrieve_applies_reranker_and_preserves_dense_observability(self):
        pipeline, _ = build_test_pipeline()
        dense_sources = pipeline.retrieve("What is this project?", top_k=3)
        pipeline.reranker = ReverseReranker()
        pipeline.fetch_k = 3

        reranked_sources = pipeline.retrieve("What is this project?", top_k=2)

        self.assertEqual(len(reranked_sources), 2)
        self.assertEqual(reranked_sources[0].content, dense_sources[2].content)
        self.assertEqual(reranked_sources[0].metadata["dense_rank"], 3)
        self.assertEqual(reranked_sources[0].metadata["dense_score"], dense_sources[2].score)
        self.assertEqual(reranked_sources[0].metadata["rerank_model"], "reverse-test")
        self.assertTrue(reranked_sources[0].metadata["test_reranked"])

    def test_rerank_failure_falls_back_to_dense_order(self):
        pipeline, _ = build_test_pipeline()
        dense_sources = pipeline.retrieve("What is this project?", top_k=2)
        pipeline.reranker = FailingReranker()
        pipeline.fetch_k = 3

        fallback_sources = pipeline.retrieve("What is this project?", top_k=2)

        self.assertEqual([asdict(source) for source in fallback_sources], [asdict(source) for source in dense_sources])

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
