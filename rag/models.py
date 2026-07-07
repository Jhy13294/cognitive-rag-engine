"""Shared data models for the RAG pipeline.

These types are used across the retrieval, source finalization, and prompt
building stages, so they live in a leaf module that every stage can import
without creating circular dependencies.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol


class ChatClient(Protocol):
    """Protocol for chat-completion clients used by the RAG pipeline."""

    def chat(self, message: str, system_prompt: Optional[str] = None) -> Dict:
        """Send a chat request and return a provider response."""
        ...


@dataclass
class RetrievedSource:
    """Source chunk used to answer a query."""

    index: int
    content: str
    score: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Answer returned by the RAG pipeline."""

    question: str
    answer: str
    sources: List[RetrievedSource]
    prompt: str
    raw_response: Dict


def extract_chat_content(response: Dict) -> str:
    """Extract assistant text from a chat-completion compatible response."""
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise ValueError("Invalid chat response format") from e
