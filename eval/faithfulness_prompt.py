"""Versioned statement-generation prompt for robust faithfulness evaluation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Sequence

if TYPE_CHECKING:
    from ragas.llms.base import InstructorBaseRagasLLM
    from ragas.metrics.collections import Faithfulness
    from ragas.metrics.collections.faithfulness.util import StatementGeneratorPrompt


STATEMENT_PROMPT_VERSION = "question-independent-rationale-and-scope-preservation-v4"

_INSTRUCTION = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement.

Split genuinely independent claims into separate statements. However, when the answer explicitly frames multiple predicates or actions as one joint requirement, procedure, sequence, or definition, and no individual part alone constitutes the complete claim, preserve the full conjunction as one statement. Do not split such a joint claim into statements that each imply one action alone is the entire requirement. The presence of the word 'and' is not sufficient reason to keep claims together: independent claims must still be split.

Also preserve a relation or assessment claim together with an indispensable rationale introduced by words such as 'because' or 'since' when removing that rationale would make the remaining claim ungrounded or impossible to judge without the original question. Do not split 'Source X is relevant because it directly addresses Y' into a generic 'Source X is relevant' statement and a separate explanation. The preserved statement must include the concrete facts that establish the relation; a generic phrase such as 'addresses the procedure' is not sufficient when the actual procedure is stated in the answer. Independent factual claims inside a rationale may still be split only when the relation statement retains enough concrete evidence to be judged without the original question.

When coordinated clauses carry a trailing temporal or conditional qualifier such as 'before', 'after', 'when', or 'if', keep that qualifier attached to the clause that contains it. Do not move it to an earlier conjunct; preserve the complete compound statement when splitting would change which condition applies to which claim."""

_CONJUNCTIVE_QUESTION = "What is the required account-recovery workaround?"
_CONJUNCTIVE_ANSWER = (
    "The required account-recovery workaround is to call the support hotline "
    "and ask the duty manager to reset the account."
)
_RATIONALE_QUESTION = "Which source is relevant to the lost-token question?"
_RATIONALE_ANSWER = (
    "Source [4] is relevant because it directly describes the procedure for a "
    "hardware token lost during travel: the employee should notify security and "
    "ask the people team for temporary travel access."
)
_SCOPE_QUESTION = "What must happen before the vendor release?"
_SCOPE_ANSWER = (
    "Vendor records must be encrypted, and the compliance owner must archive "
    "the audit ticket before the vendor release."
)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
_RELEVANCE_RATIONALE = re.compile(r"\bis\s+relevant\s+because\b", re.IGNORECASE)
_COORDINATED_CLAUSES = re.compile(r"^(?P<left>.+?),\s+and\s+(?P<right>.+)$", re.IGNORECASE)
_TEMPORAL_QUALIFIER = re.compile(r"\b(?:before|after|when|while|if)\b", re.IGNORECASE)
_CITATION_SUFFIX = re.compile(r"\s*\[\d+\]\s*$")


def _normalize_statement_text(value: str) -> str:
    """Normalize statement text before comparing clause boundaries."""
    without_citation = _CITATION_SUFFIX.sub("", str(value).strip())
    return " ".join(without_citation.rstrip(".!?").split()).casefold()


def preserve_clause_scoped_qualifier(
    answer: str,
    statements: Sequence[str],
) -> List[str]:
    """Keep a trailing temporal qualifier with the clause that owns it."""
    normalized_answer = " ".join(str(answer).split()).strip()
    answer_for_detection = _CITATION_SUFFIX.sub("", normalized_answer).strip()
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(answer_for_detection)
        if sentence.strip()
    ]
    if len(sentences) != 1:
        return list(statements)

    sentence = sentences[0]
    match = _COORDINATED_CLAUSES.match(_CITATION_SUFFIX.sub("", sentence).rstrip(".!?"))
    if match is None:
        return list(statements)

    left_clause = _normalize_statement_text(match.group("left"))
    right_clause = match.group("right")
    qualifier_match = _TEMPORAL_QUALIFIER.search(right_clause)
    if not left_clause or qualifier_match is None:
        return list(statements)

    qualifier = _normalize_statement_text(right_clause[qualifier_match.start() :])
    for statement in statements:
        normalized_statement = _normalize_statement_text(statement)
        if (
            normalized_statement.startswith(left_clause)
            and qualifier
            and qualifier in normalized_statement
        ):
            return [normalized_answer]
    return list(statements)


def preserve_question_dependent_rationale(
    answer: str,
    statements: Sequence[str],
) -> List[str]:
    """Keep a single-sentence relevance rationale independently judgeable."""
    normalized_answer = " ".join(str(answer).split()).strip()
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY.split(normalized_answer)
        if sentence.strip()
    ]
    if len(sentences) == 1 and _RELEVANCE_RATIONALE.search(sentences[0]):
        return [sentences[0]]
    return preserve_clause_scoped_qualifier(answer, statements)


def build_statement_generator_prompt() -> StatementGeneratorPrompt:
    """Build the pinned Ragas prompt without importing Ragas during offline replay."""
    from ragas.metrics.collections.faithfulness.util import (
        StatementGeneratorInput,
        StatementGeneratorOutput,
        StatementGeneratorPrompt,
    )

    class SemanticDependencyStatementGeneratorPrompt(StatementGeneratorPrompt):
        """Preserve predicates and rationales required to judge a complete claim."""

        prompt_version = STATEMENT_PROMPT_VERSION
        instruction = _INSTRUCTION
        examples = [
            *StatementGeneratorPrompt.examples,
            (
                StatementGeneratorInput(
                    question=_CONJUNCTIVE_QUESTION,
                    answer=_CONJUNCTIVE_ANSWER,
                ),
                StatementGeneratorOutput(statements=[_CONJUNCTIVE_ANSWER]),
            ),
            (
                StatementGeneratorInput(
                    question=_RATIONALE_QUESTION,
                    answer=_RATIONALE_ANSWER,
                ),
                StatementGeneratorOutput(statements=[_RATIONALE_ANSWER]),
            ),
            (
                StatementGeneratorInput(
                    question=_SCOPE_QUESTION,
                    answer=_SCOPE_ANSWER,
                ),
                StatementGeneratorOutput(statements=[_SCOPE_ANSWER]),
            ),
        ]

    return SemanticDependencyStatementGeneratorPrompt()


def build_faithfulness_metric(llm: InstructorBaseRagasLLM) -> Faithfulness:
    """Build Faithfulness with the pinned prompt and narrow statement guard."""
    from ragas.metrics.collections import Faithfulness

    class ProjectFaithfulness(Faithfulness):
        """Apply deterministic preservation after probabilistic extraction."""

        async def _create_statements(self, question: str, response: str) -> List[str]:
            statements = await super()._create_statements(question, response)
            return preserve_question_dependent_rationale(response, statements)

    metric = ProjectFaithfulness(llm=llm)
    metric.statement_generator_prompt = build_statement_generator_prompt()
    return metric
