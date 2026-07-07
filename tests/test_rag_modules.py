import hashlib
import unittest
from dataclasses import asdict

import rag.pipeline as pipeline_module
from parent_store import InMemoryParentStore
from rag import PromptBuilder, RAGPipeline, RetrievalOrchestrator, SourceFinalizer
from rag import models, prompt_builder, question_classifier, retrieval_orchestrator, source_finalizer
from vector_store import InMemoryVectorStore, VectorRecord

from document_loader import Document
from embeddings.base import EmbeddingConfig, EmbeddingProvider

# The committed Ragas verdict fixture was recorded against exactly this system
# prompt; a content change without a version bump would silently invalidate
# the frozen generation-quality baseline.
PINNED_SYSTEM_PROMPT_SHA256 = "ea08efc7d91393d39ba941de307d5ddcc134bf14697ef2cab45db5d46b76e6d7"
PINNED_SYSTEM_PROMPT_VERSION = "supporting-relation-and-workaround-answer-v9"
PINNED_ABSTENTION_SENTENCE = "The answer is not available in the knowledge base."


class StaticEmbeddingProvider(EmbeddingProvider):
    """Deterministic query embedding provider for module split tests."""

    def __init__(self):
        super().__init__(EmbeddingConfig(model_name="static-test", dimension=2))

    def embed_texts(self, texts):
        """Return the same query vector for every text."""
        return [[1.0, 0.0] for _ in list(texts)]


class FakeChatClient:
    """Chat client placeholder for pipeline construction."""

    def chat(self, message, system_prompt=None):
        """Return a fixed response."""
        return {"choices": [{"message": {"content": "ok [1]"}}]}


def build_parent_child_pipeline():
    """Build a small parent-child pipeline exercising every split stage."""
    parent = Document(
        content="Full parent content with beta target detail and gamma closing sentence.",
        metadata={"parent_id": "parent-1", "source": "policy.md", "parent_index": 0, "start_char": 0, "end_char": 72},
    )
    parent_store = InMemoryParentStore()
    parent_store.add_parents([parent])

    store = InMemoryVectorStore(dimension=2)
    store.add_records(
        [
            VectorRecord(
                id="child-1",
                content="Beta target detail.",
                embedding=[1.0, 0.0],
                metadata={
                    "id": "child-1",
                    "source": "policy.md",
                    "chunk_index": 0,
                    "start_char": 13,
                    "end_char": 32,
                    "parent_id": "parent-1",
                },
            ),
            VectorRecord(
                id="child-2",
                content="Gamma closing sentence.",
                embedding=[0.9, 0.1],
                metadata={
                    "id": "child-2",
                    "source": "policy.md",
                    "chunk_index": 1,
                    "start_char": 33,
                    "end_char": 56,
                    "parent_id": "parent-1",
                },
            ),
        ]
    )
    return RAGPipeline(
        StaticEmbeddingProvider(),
        store,
        FakeChatClient(),
        top_k=2,
        parent_store=parent_store,
        expand_parent_context=True,
    )


class PipelineModuleSplitTests(unittest.TestCase):
    def test_pipeline_reexports_are_the_module_definitions(self):
        """rag.pipeline compatibility exports must alias the stage modules, not copies."""
        self.assertIs(pipeline_module.RetrievedSource, models.RetrievedSource)
        self.assertIs(pipeline_module.RAGResponse, models.RAGResponse)
        self.assertIs(pipeline_module.extract_chat_content, models.extract_chat_content)
        self.assertIs(pipeline_module.PromptBuilder, PromptBuilder)
        self.assertIs(pipeline_module.RetrievalOrchestrator, RetrievalOrchestrator)
        self.assertIs(pipeline_module.SourceFinalizer, SourceFinalizer)
        self.assertIs(pipeline_module.child_id_for, source_finalizer.child_id_for)
        self.assertIs(
            pipeline_module.dense_search_results_to_ranked_records,
            retrieval_orchestrator.dense_search_results_to_ranked_records,
        )
        self.assertIs(
            pipeline_module.retrieved_sources_to_ranked_records,
            retrieval_orchestrator.retrieved_sources_to_ranked_records,
        )
        for name in (
            "is_action_list_question",
            "is_name_only_lookup_question",
            "is_relationship_entity_question",
            "is_scenario_procedure_question",
            "is_source_relevance_question",
            "is_workaround_question",
        ):
            self.assertIs(getattr(pipeline_module, name), getattr(question_classifier, name))
        self.assertEqual(
            pipeline_module.CANONICAL_ABSTENTION_RESPONSE,
            prompt_builder.CANONICAL_ABSTENTION_RESPONSE,
        )
        self.assertEqual(
            pipeline_module.RAG_SYSTEM_PROMPT_VERSION,
            prompt_builder.RAG_SYSTEM_PROMPT_VERSION,
        )
        self.assertEqual(RAGPipeline.DEFAULT_SYSTEM_PROMPT, prompt_builder.DEFAULT_SYSTEM_PROMPT)

    def test_system_prompt_contract_is_pinned(self):
        """The frozen verdict fixture depends on this exact prompt contract."""
        self.assertEqual(prompt_builder.RAG_SYSTEM_PROMPT_VERSION, PINNED_SYSTEM_PROMPT_VERSION)
        self.assertEqual(prompt_builder.CANONICAL_ABSTENTION_RESPONSE, PINNED_ABSTENTION_SENTENCE)
        self.assertEqual(
            hashlib.sha256(prompt_builder.DEFAULT_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            PINNED_SYSTEM_PROMPT_SHA256,
        )
        self.assertIn(f'"{PINNED_ABSTENTION_SENTENCE}"', prompt_builder.DEFAULT_SYSTEM_PROMPT)

    def test_prompt_builder_backs_pipeline_prompt_entry_points(self):
        """build_prompt and the compatibility helpers must delegate to PromptBuilder."""
        pipeline = build_parent_child_pipeline()
        sources = pipeline.retrieve("beta target detail")

        prompt, used_sources = pipeline.prompt_builder.build("beta target detail", sources)
        self.assertEqual(pipeline.build_prompt("beta target detail", sources), prompt)
        self.assertEqual(
            pipeline._build_prompt_and_sources("beta target detail", sources),
            (prompt, used_sources),
        )
        self.assertEqual(
            pipeline._build_context(sources),
            pipeline.prompt_builder.build_context(sources),
        )

    def test_orchestrator_then_finalizer_matches_pipeline_retrieve(self):
        """pipeline.retrieve must equal the two split stages composed in order."""
        pipeline = build_parent_child_pipeline()
        candidates = pipeline.retrieval_orchestrator.retrieve(
            question="beta target detail",
            requested_top_k=pipeline.top_k,
        )
        staged = pipeline.source_finalizer.finalize(candidates)
        direct = pipeline.retrieve("beta target detail")

        self.assertEqual([asdict(s) for s in staged], [asdict(s) for s in direct])
        self.assertEqual(direct[0].metadata["parent_expanded"], True)
        self.assertEqual(direct[0].metadata["collapsed_child_count"], 2)

    def test_instance_level_retrieve_override_still_feeds_answer(self):
        """Cache wrappers replace pipeline.retrieve on the instance; answer() must honor it."""
        pipeline = build_parent_child_pipeline()
        sentinel_sources = [
            models.RetrievedSource(index=1, content="wrapped content", score=1.0, metadata={"source": "wrapped.md"})
        ]
        calls = []

        def wrapped_retrieve(question, top_k=None, metadata_filter=None):
            calls.append(question)
            return sentinel_sources

        pipeline.retrieve = wrapped_retrieve
        response = pipeline.answer("beta target detail")

        self.assertEqual(calls, ["beta target detail"])
        self.assertEqual(response.sources, sentinel_sources)
        self.assertIn("wrapped content", response.prompt)


if __name__ == "__main__":
    unittest.main()
