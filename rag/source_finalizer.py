"""Post-selection source finalization for the RAG pipeline.

Owns the transformations applied after retrieval has picked the final top-k
candidates: parent context expansion and sibling-child collapse by parent id.
Context packing happens later, at prompt-build time, because it depends on
the generation-side token budget rather than on retrieval.
"""

from typing import TYPE_CHECKING, List

from logger import setup_logger

from .models import RetrievedSource

if TYPE_CHECKING:
    from .pipeline import RAGPipeline

logger = setup_logger(__name__)


class SourceFinalizer:
    """Apply post-retrieval transformations to selected sources.

    The finalizer reads the parent store and expansion flag from the owning
    pipeline at call time, so runtime reconfiguration and cache wrappers keep
    working exactly as they do against the pipeline itself.
    """

    def __init__(self, pipeline: "RAGPipeline"):
        self._pipeline = pipeline

    def finalize(self, sources: List[RetrievedSource]) -> List[RetrievedSource]:
        """Apply post-retrieval source transformations."""
        pipeline = self._pipeline
        if pipeline.parent_store is None or not pipeline.expand_parent_context:
            return sources
        return self._expand_parent_sources(sources)

    def _expand_parent_sources(self, sources: List[RetrievedSource]) -> List[RetrievedSource]:
        """Replace child chunk content with parent content after retrieval."""
        parent_store = self._pipeline.parent_store
        expanded_sources = []
        parent_positions = {}

        for source in sources:
            parent_id = source.metadata.get("parent_id")
            if not parent_id:
                expanded_sources.append(source)
                continue

            parent = parent_store.get_parent(str(parent_id))
            if parent is None:
                logger.warning(
                    "Parent chunk not found; keeping child source | parent_id=%s | child_index=%s",
                    parent_id,
                    source.index,
                )
                metadata = dict(source.metadata)
                metadata["parent_lookup_failed"] = True
                expanded_sources.append(
                    RetrievedSource(
                        index=source.index,
                        content=source.content,
                        score=source.score,
                        metadata=metadata,
                    )
                )
                continue

            if parent.id in parent_positions:
                existing = expanded_sources[parent_positions[parent.id]]
                existing.metadata["collapsed_child_count"] += 1
                existing.metadata.setdefault("collapsed_child_ids", []).append(child_id_for(source))
                continue

            metadata = dict(source.metadata)
            metadata.update(
                {
                    "child_id": child_id_for(source),
                    "child_score": source.score,
                    "child_start_char": source.metadata.get("start_char"),
                    "child_end_char": source.metadata.get("end_char"),
                    "parent_id": parent.id,
                    "parent_index": parent.metadata.get(
                        "parent_index", source.metadata.get("parent_index")
                    ),
                    "parent_start_char": parent.metadata.get("start_char"),
                    "parent_end_char": parent.metadata.get("end_char"),
                    "parent_expanded": True,
                    "collapsed_child_count": 1,
                    "collapsed_child_ids": [child_id_for(source)],
                }
            )
            expanded_source = RetrievedSource(
                index=source.index,
                content=parent.content,
                score=source.score,
                metadata=metadata,
            )
            parent_positions[parent.id] = len(expanded_sources)
            expanded_sources.append(expanded_source)

        return expanded_sources


def child_id_for(source: RetrievedSource) -> str:
    """Return the stable child id for a retrieved source."""
    metadata = source.metadata or {}
    return str(metadata.get("child_id") or metadata.get("id") or "")
