"""Deterministic question-shape heuristics for prompt calibration.

The generation prompt is calibrated per question shape, so these classifiers
decide which scope instruction each question receives. They are intentionally
plain substring heuristics: no LLM calls, no scoring, byte-stable results.
The marker tuples are part of the calibrated generation-prompt contract;
changing them shifts answers and invalidates the frozen generation-quality
baseline, so treat edits like prompt version changes.
"""

SCENARIO_PROCEDURE_KEYWORDS = (
    "workaround",
    "which checks should",
    "what checks should",
    "which steps should",
    "what steps should",
)

SCENARIO_SITUATION_MARKERS = (
    " when ",
    " if ",
    " after ",
    " while ",
    " not ready",
    " delayed",
    " stale",
    " fails",
    " failure",
    " loses ",
    " lost ",
)

NAME_ONLY_LOOKUP_MARKERS = (
    "where is",
    "what source",
    "which source",
    "which document",
    "what document",
    "which runbook",
    "what runbook",
    "what policy",
    "which policy",
)

NAME_ONLY_LOOKUP_TARGETS = (
    "defines",
    "defined",
    "described",
    "contains",
    "mentions",
    "explains",
)

SOURCE_RELEVANCE_MARKERS = (
    "which sources",
    "what sources",
    "which documents",
    "what documents",
    "which policies",
    "what policies",
    "which runbooks",
    "what runbooks",
)

SOURCE_RELEVANCE_TARGETS = (
    " relevant",
    " apply",
    " applies",
    " needed",
    " should be used",
    " should i use",
)

RELATIONSHIP_ENTITY_TARGETS = (
    "tool",
    "service",
    "workflow",
)

RELATIONSHIP_ENTITY_MARKERS = (
    " helps ",
    " routes ",
    " sends ",
    " forwards ",
    " used to ",
    " responsible for ",
)

ACTION_LIST_MARKERS = (
    " what should ",
    " which checks should ",
    " what checks should ",
    " what has to happen ",
    " who must approve ",
)


def is_scenario_procedure_question(question: str) -> bool:
    """Return whether a question asks for a scenario-specific procedure or workaround."""
    normalized = f" {' '.join(question.lower().split())} "
    if any(keyword in normalized for keyword in SCENARIO_PROCEDURE_KEYWORDS):
        return True

    has_procedural_intent = any(
        phrase in normalized
        for phrase in (
            " what should ",
            " how should ",
            " what do ",
            " how do ",
            " how can ",
        )
    )
    has_situation = any(marker in normalized for marker in SCENARIO_SITUATION_MARKERS)
    return has_procedural_intent and has_situation


def is_workaround_question(question: str) -> bool:
    """Return whether the user is asking specifically for a workaround."""
    normalized = f" {' '.join(question.lower().split())} "
    return "workaround" in normalized


def is_name_only_lookup_question(question: str) -> bool:
    """Return whether a question asks for a source/document name rather than details."""
    normalized = f" {' '.join(question.lower().split())} "
    has_lookup_marker = any(marker in normalized for marker in NAME_ONLY_LOOKUP_MARKERS)
    has_lookup_target = any(target in normalized for target in NAME_ONLY_LOOKUP_TARGETS)
    return has_lookup_marker and has_lookup_target


def is_source_relevance_question(question: str) -> bool:
    """Return whether the user asks which sources or documents are relevant."""
    normalized = f" {' '.join(question.lower().split())} "
    has_source_marker = any(marker in normalized for marker in SOURCE_RELEVANCE_MARKERS)
    has_relevance_target = any(target in normalized for target in SOURCE_RELEVANCE_TARGETS)
    return has_source_marker and has_relevance_target


def is_relationship_entity_question(question: str) -> bool:
    """Return whether an entity must be tied to a responsibility to answer."""
    normalized = f" {' '.join(question.lower().split())} "
    has_entity_target = any(target in normalized for target in RELATIONSHIP_ENTITY_TARGETS)
    has_relationship_marker = any(marker in normalized for marker in RELATIONSHIP_ENTITY_MARKERS)
    return has_entity_target and has_relationship_marker


def is_action_list_question(question: str) -> bool:
    """Return whether a question asks for direct actions, tasks, or approvers."""
    normalized = f" {' '.join(question.lower().split())} "
    return any(marker in normalized for marker in ACTION_LIST_MARKERS)
