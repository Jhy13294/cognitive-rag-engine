"""Prompt construction for the RAG pipeline.

Owns everything that turns a question plus retrieved sources into the final
generation input: the versioned system prompt, per-question-shape scope
instructions, context assembly (packed or legacy character-bounded), and
source label formatting.

The system prompt and the scope instructions are a calibrated contract with
the frozen generation-quality baseline: the committed verdict fixture pins
``RAG_SYSTEM_PROMPT_VERSION``, so any wording change here must bump the
version and re-record the live baseline.
"""

from typing import TYPE_CHECKING, List, Tuple

from .models import RetrievedSource
from .question_classifier import (
    is_action_list_question,
    is_name_only_lookup_question,
    is_relationship_entity_question,
    is_scenario_procedure_question,
    is_source_relevance_question,
    is_workaround_question,
)

if TYPE_CHECKING:
    from .pipeline import RAGPipeline

CANONICAL_ABSTENTION_RESPONSE = "The answer is not available in the knowledge base."
RAG_SYSTEM_PROMPT_VERSION = "supporting-relation-and-workaround-answer-v9"

DEFAULT_SYSTEM_PROMPT = (
    "You are an enterprise knowledge-base assistant. "
    "Answer only with claims directly supported by the provided context. "
    "Do not infer a policy, procedure, value, or current fact from merely related context. "
    "Match the scope and wording of the user's question. "
    "Use the source's action verbs for procedures and checks; do not replace 'check X' with 'run X' "
    "unless the source uses that verb. "
    "For source, document, runbook, or policy lookup questions, preserve the lookup relation in one "
    "short sentence, such as 'The document that contains this topic is Name [1]'; never answer with "
    "only a bare name or citation. "
    "For questions asking which sources or documents are relevant, name each source and give a brief "
    "source-supported reason tied to the scenario. "
    "For tool, service, or workflow selection questions, preserve the requested responsibility in one "
    "short sentence, such as 'The service that performs the requested responsibility is Name [1]'; "
    "do not list adjacent responsibilities. "
    "If the context does not directly support an answer, begin with exactly this sentence: "
    f'"{CANONICAL_ABSTENTION_RESPONSE}" '
    "You may briefly explain what information is missing, but do not supply an unsupported answer. "
    "Cite sources with bracketed numbers like [1], [2]."
)


class PromptBuilder:
    """Build generation prompts from a question and retrieved sources.

    The builder reads context-packing configuration from the owning pipeline
    at call time, so runtime reconfiguration and cache wrappers keep working
    exactly as they do against the pipeline itself.
    """

    def __init__(self, pipeline: "RAGPipeline"):
        self._pipeline = pipeline

    def build(
        self,
        question: str,
        sources: List[RetrievedSource],
    ) -> Tuple[str, List[RetrievedSource]]:
        """Build a prompt and return the sources actually used in context."""
        pipeline = self._pipeline
        if pipeline.context_packing_enabled:
            packed_context = pipeline.context_packer.pack(sources)
            context = packed_context.context
            used_sources = packed_context.used_sources
        else:
            context = self.build_context(sources)
            used_sources = sources

        prompt = (
            "Use the context below to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer with source citations. "
            f"{self._entity_scope_instruction(question)}"
            f"{self._answer_scope_instruction(question)}"
        )
        return prompt, used_sources

    def build_context(self, sources: List[RetrievedSource]) -> str:
        """Build a bounded context block from retrieved sources."""
        pipeline = self._pipeline
        if pipeline.context_packing_enabled:
            return pipeline.context_packer.pack(sources).context

        context_blocks = []
        used_chars = 0

        for source in sources:
            source_label = self.format_source_label(source)
            block = f"[{source.index}] {source_label}\n{source.content}".strip()

            if used_chars + len(block) > pipeline.max_context_chars:
                remaining = pipeline.max_context_chars - used_chars
                if remaining <= 0:
                    break
                block = block[:remaining].rstrip()

            context_blocks.append(block)
            used_chars += len(block)

            if used_chars >= pipeline.max_context_chars:
                break

        if not context_blocks:
            return "No relevant context was retrieved."

        return "\n\n".join(context_blocks)

    def format_source_label(self, source: RetrievedSource) -> str:
        """Format human-readable source metadata."""
        metadata = source.metadata
        source_path = metadata.get("source", "unknown source")
        chunk_index = metadata.get("chunk_index")

        if chunk_index is None:
            return f"source={source_path}; score={source.score:.4f}"

        return f"source={source_path}; chunk={chunk_index}; score={source.score:.4f}"

    def _entity_scope_instruction(self, question: str) -> str:
        """Return an entity-answer instruction scoped to the question shape."""
        if is_source_relevance_question(question):
            return (
                "This asks which sources or documents are relevant; source names may be the main answer. "
                "Name each relevant source and give a brief source-supported reason tied to the question. "
                "Do not answer with only citation numbers. "
            )
        if is_name_only_lookup_question(question):
            return (
                "This is a source/document/runbook/policy lookup; answer in one short sentence that preserves "
                "the lookup relation and topic, for example 'The document/source that contains or defines "
                "the requested topic is Name [n].' Do not answer with only the name or citation. "
                "Do not list workflow steps or unrelated details. "
            )
        if is_relationship_entity_question(question):
            return (
                "This asks which tool, service, or workflow satisfies a described responsibility; answer in "
                "one short sentence that preserves the requested responsibility, for example 'The tool/service "
                "that does the requested thing is Name [n].' Do not answer with only the entity name, and do "
                "not list adjacent responsibilities. "
            )
        return (
            "When the question does not ask for source or document names, do not make a source, checklist, "
            "workflow, or policy name the main answer. "
        )

    def _answer_scope_instruction(self, question: str) -> str:
        """Return a generation instruction scoped to the user's question shape."""
        if is_workaround_question(question):
            return (
                "This is a workaround question; include the source-listed workaround actions and the "
                "source-supported condition that identifies when the workaround applies, while keeping the "
                "wording close to the source."
            )

        if is_scenario_procedure_question(question):
            return (
                "This is a scenario-based procedure/checks question; answer with the source-listed actions "
                "or checks using the source's verbs, and avoid adding or rephrasing scenario conditions unless "
                "they are the answer."
            )

        if is_action_list_question(question):
            return (
                "This asks for actions, tasks, or approvers; answer with the requested actions/items first, "
                "using the source's verbs, and do not frame the answer around a checklist or workflow name."
            )

        return (
            "This is not a scenario-based procedure or workaround request; do not broaden the answer with "
            "extra triggers, preconditions, or workflow details. For timing, threshold, approval, or "
            "requirement questions, include the requested value or action plus the source-supported condition "
            "that makes it apply."
        )
