"""Versioned statement-generation prompt for robust faithfulness evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ragas.metrics.collections.faithfulness.util import StatementGeneratorPrompt


STATEMENT_PROMPT_VERSION = "joint-requirement-preservation-v1"

_INSTRUCTION = """Given a question and an answer, analyze the complexity of each sentence in the answer. Break down each sentence into one or more fully understandable statements. Ensure that no pronouns are used in any statement.

Split genuinely independent claims into separate statements. However, when the answer explicitly frames multiple predicates or actions as one joint requirement, procedure, sequence, or definition, and no individual part alone constitutes the complete claim, preserve the full conjunction as one statement. Do not split such a joint claim into statements that each imply one action alone is the entire requirement. The presence of the word 'and' is not sufficient reason to keep claims together: independent claims must still be split."""

_CONJUNCTIVE_QUESTION = "What is the required account-recovery workaround?"
_CONJUNCTIVE_ANSWER = (
    "The required account-recovery workaround is to call the support hotline "
    "and ask the duty manager to reset the account."
)


def build_statement_generator_prompt() -> StatementGeneratorPrompt:
    """Build the pinned Ragas prompt without importing Ragas during offline replay."""
    from ragas.metrics.collections.faithfulness.util import (
        StatementGeneratorInput,
        StatementGeneratorOutput,
        StatementGeneratorPrompt,
    )

    class JointRequirementStatementGeneratorPrompt(StatementGeneratorPrompt):
        """Preserve predicates that jointly constitute one complete claim."""

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
        ]

    return JointRequirementStatementGeneratorPrompt()
