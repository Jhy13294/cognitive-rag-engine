from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from tokenization import TokenCounter

SENTENCE_TRUNCATION_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? "]


@dataclass
class PackedContext:
    """Context packing output for generation."""

    context: str
    used_sources: List[Any]
    deduped_source_ids: List[str]
    skipped_source_ids: List[str]
    budget_used: int
    budget_limit: int
    budget_unit: str


class ContextPacker:
    """Pack retrieved sources into a bounded generation context."""

    def __init__(
        self,
        max_context_chars: int,
        label_formatter: Callable[[Any], str],
        max_context_tokens: Optional[int] = None,
        token_counter: Optional[TokenCounter] = None,
        dedup_enabled: bool = False,
        near_dup_enabled: bool = False,
        near_dup_threshold: float = 0.9,
    ):
        """Initialize the context packer."""
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than 0")
        if max_context_tokens is not None and max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than 0")
        if near_dup_threshold < 0 or near_dup_threshold > 1:
            raise ValueError("near_dup_threshold must be between 0 and 1")
        if max_context_tokens is not None and token_counter is None:
            raise ValueError("token_counter is required when max_context_tokens is set")

        self.max_context_chars = max_context_chars
        self.max_context_tokens = max_context_tokens
        self.token_counter = token_counter
        self.label_formatter = label_formatter
        self.dedup_enabled = dedup_enabled
        self.near_dup_enabled = near_dup_enabled
        self.near_dup_threshold = near_dup_threshold
        self.truncation_separators = SENTENCE_TRUNCATION_SEPARATORS

    def pack(self, sources: Sequence[Any]) -> PackedContext:
        """Pack sources with include-or-skip semantics."""
        prepared_sources, deduped_source_ids = self._deduplicate(sources)
        context_blocks: List[str] = []
        used_sources: List[Any] = []
        skipped_source_ids: List[str] = []

        for source in prepared_sources:
            packed_index = len(used_sources) + 1
            candidate_source = self._with_context_index(source, packed_index)
            candidate_block = self._format_block(candidate_source)

            if self._fits(context_blocks + [candidate_block]):
                context_blocks.append(candidate_block)
                used_sources.append(candidate_source)
                continue

            if self._measure(candidate_block) <= self._budget_limit():
                skipped_source_ids.append(source_identifier(source))
                continue

            truncated = self._truncate_oversized_source(source, packed_index)
            if truncated is None:
                skipped_source_ids.append(source_identifier(source))
                continue

            truncated_source, truncated_block = truncated
            if self._fits(context_blocks + [truncated_block]):
                context_blocks.append(truncated_block)
                used_sources.append(truncated_source)
            else:
                skipped_source_ids.append(source_identifier(source))

        if not context_blocks:
            context = "No relevant context was retrieved."
        else:
            context = "\n\n".join(context_blocks)

        return PackedContext(
            context=context,
            used_sources=used_sources,
            deduped_source_ids=deduped_source_ids,
            skipped_source_ids=skipped_source_ids,
            budget_used=self._measure(context),
            budget_limit=self._budget_limit(),
            budget_unit=self._budget_unit(),
        )

    def _deduplicate(self, sources: Sequence[Any]) -> Tuple[List[Any], List[str]]:
        """Deduplicate sources without changing retrieval order."""
        if not self.dedup_enabled and not self.near_dup_enabled:
            return [self._copy_source(source) for source in sources], []

        kept_sources: List[Any] = []
        exact_positions: Dict[Tuple, int] = {}
        near_duplicate_shingles: List[Set[str]] = []
        deduped_source_ids: List[str] = []

        for source in sources:
            source_copy = self._copy_source(source)
            duplicate_position = None
            duplicate_reason = None

            if self.dedup_enabled:
                key = exact_dedup_key(source_copy)
                if key is not None and key in exact_positions:
                    duplicate_position = exact_positions[key]
                    duplicate_reason = "exact"

            if duplicate_position is None and self.near_dup_enabled:
                source_shingles = shingles(source_copy.content)
                for index, kept_shingles in enumerate(near_duplicate_shingles):
                    if (
                        jaccard_similarity(source_shingles, kept_shingles)
                        >= self.near_dup_threshold
                    ):
                        duplicate_position = index
                        duplicate_reason = "near"
                        break

            if duplicate_position is not None:
                deduped_id = source_identifier(source_copy)
                deduped_source_ids.append(deduped_id)
                kept = kept_sources[duplicate_position]
                kept.metadata.setdefault("deduped_source_ids", []).append(deduped_id)
                kept.metadata.setdefault("dedup_reasons", []).append(duplicate_reason)
                continue

            kept_position = len(kept_sources)
            key = exact_dedup_key(source_copy)
            if key is not None:
                exact_positions[key] = kept_position
            if self.near_dup_enabled:
                near_duplicate_shingles.append(shingles(source_copy.content))
            kept_sources.append(source_copy)

        return kept_sources, deduped_source_ids

    def _truncate_oversized_source(
        self, source: Any, packed_index: int
    ) -> Optional[Tuple[Any, str]]:
        """Truncate one oversized source only at sentence or paragraph boundaries."""
        boundary_positions = self._boundary_positions(source.content)
        for position in reversed(boundary_positions):
            truncated_content = source.content[:position].strip()
            if not truncated_content:
                continue

            metadata = dict(source.metadata)
            metadata.update(
                {
                    "context_truncated": True,
                    "context_truncation": "sentence_boundary",
                    "original_content_chars": len(source.content),
                    "context_content_chars": len(truncated_content),
                }
            )
            candidate_source = replace(
                source, index=packed_index, content=truncated_content, metadata=metadata
            )
            candidate_block = self._format_block(candidate_source)
            if self._measure(candidate_block) <= self._budget_limit():
                return candidate_source, candidate_block

        return None

    def _boundary_positions(self, text: str) -> List[int]:
        """Return deterministic sentence and paragraph boundary positions."""
        positions = {len(text)}
        for separator in self.truncation_separators:
            start = 0
            while True:
                found_at = text.find(separator, start)
                if found_at == -1:
                    break
                positions.add(found_at + len(separator))
                start = found_at + len(separator)
        return sorted(position for position in positions if position > 0)

    def _with_context_index(self, source: Any, packed_index: int) -> Any:
        """Return a copied source with a contiguous context index."""
        metadata = dict(source.metadata)
        if source.index != packed_index:
            metadata.setdefault("original_index", source.index)
        metadata["context_index"] = packed_index
        return replace(source, index=packed_index, metadata=metadata)

    def _copy_source(self, source: Any) -> Any:
        """Return a source copy with independent metadata."""
        return replace(source, metadata=dict(source.metadata))

    def _format_block(self, source: Any) -> str:
        """Format one context block."""
        source_label = self.label_formatter(source)
        return f"[{source.index}] {source_label}\n{source.content}".strip()

    def _fits(self, blocks: List[str]) -> bool:
        """Return whether blocks fit within the active budget."""
        return self._measure("\n\n".join(blocks)) <= self._budget_limit()

    def _measure(self, text: str) -> int:
        """Measure text in the active budget unit."""
        if self.max_context_tokens is not None:
            return self.token_counter.count(text)
        return len(text)

    def _budget_limit(self) -> int:
        """Return the active budget limit."""
        return (
            self.max_context_tokens
            if self.max_context_tokens is not None
            else self.max_context_chars
        )

    def _budget_unit(self) -> str:
        """Return the active budget unit name."""
        return "tokens" if self.max_context_tokens is not None else "chars"


def exact_dedup_key(source: Any) -> Optional[Tuple]:
    """Return a deterministic exact deduplication key for a source."""
    metadata = source.metadata or {}
    source_path = metadata.get("source")
    start_char = metadata.get("start_char")
    end_char = metadata.get("end_char")
    if source_path is not None and start_char is not None and end_char is not None:
        return ("span", str(source_path), int(start_char), int(end_char))

    record_id = metadata.get("id") or metadata.get("child_id")
    if record_id:
        return ("id", str(record_id))

    parent_id = metadata.get("parent_id")
    if parent_id:
        return ("parent_id", str(parent_id))

    return None


def source_identifier(source: Any) -> str:
    """Return an observable source identifier for diagnostics."""
    metadata = source.metadata or {}
    for key in ("id", "child_id", "parent_id"):
        if metadata.get(key):
            return str(metadata[key])
    if metadata.get("source") is not None:
        return f"{metadata.get('source')}#{metadata.get('start_char')}:{metadata.get('end_char')}"
    return f"source-index-{source.index}"


def shingles(text: str, size: int = 5) -> Set[str]:
    """Return deterministic character shingles for near-duplicate checks."""
    normalized = " ".join((text or "").lower().split())
    if not normalized:
        return set()
    if len(normalized) <= size:
        return {normalized}
    return {normalized[index : index + size] for index in range(0, len(normalized) - size + 1)}


def jaccard_similarity(first: Set[str], second: Set[str]) -> float:
    """Return Jaccard similarity between two shingle sets."""
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)
