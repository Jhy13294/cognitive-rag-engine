import json
import struct
from typing import Dict, List

from rag import RAGResponse, RetrievedSource


def encode_vector(vector: List[float]) -> bytes:
    """Encode a Python float vector as binary float64 values without precision loss."""
    values = [float(value) for value in vector]
    header = struct.pack("!I", len(values))
    if not values:
        return header
    return header + struct.pack(f"!{len(values)}d", *values)


def decode_vector(payload: bytes) -> List[float]:
    """Decode a binary float64 vector produced by encode_vector."""
    if len(payload) < 4:
        raise ValueError("Invalid vector payload")
    length = struct.unpack("!I", payload[:4])[0]
    expected = 4 + (length * 8)
    if len(payload) != expected:
        raise ValueError("Invalid vector payload length")
    if length == 0:
        return []
    return list(struct.unpack(f"!{length}d", payload[4:]))


def source_to_dict(source: RetrievedSource) -> Dict:
    """Serialize a RetrievedSource into JSON-compatible data."""
    return {
        "index": source.index,
        "content": source.content,
        "score": source.score,
        "metadata": dict(source.metadata),
    }


def source_from_dict(payload: Dict) -> RetrievedSource:
    """Deserialize a RetrievedSource and avoid exposing cached aliases."""
    return RetrievedSource(
        index=int(payload["index"]),
        content=str(payload["content"]),
        score=float(payload["score"]),
        metadata=dict(payload.get("metadata") or {}),
    )


def sources_to_payload(sources: List[RetrievedSource]) -> List[Dict]:
    """Serialize retrieved sources for Redis JSON storage."""
    return [source_to_dict(source) for source in sources]


def sources_from_payload(payload: List[Dict]) -> List[RetrievedSource]:
    """Deserialize retrieved sources with fresh objects."""
    return [source_from_dict(item) for item in payload]


def response_to_payload(response: RAGResponse) -> Dict:
    """Serialize a RAGResponse for Redis JSON storage."""
    return {
        "question": response.question,
        "answer": response.answer,
        "sources": sources_to_payload(response.sources),
        "prompt": response.prompt,
        "raw_response": json.loads(json.dumps(response.raw_response, ensure_ascii=False)),
    }


def response_from_payload(payload: Dict) -> RAGResponse:
    """Deserialize a RAGResponse with defensive copies."""
    return RAGResponse(
        question=str(payload["question"]),
        answer=str(payload["answer"]),
        sources=sources_from_payload(payload.get("sources") or []),
        prompt=str(payload.get("prompt") or ""),
        raw_response=dict(payload.get("raw_response") or {}),
    )
