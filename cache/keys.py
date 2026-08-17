import hashlib
import json
from typing import Any, Dict, Optional

CACHE_KEY_VERSION = "cache-v1"


def normalize_question(question: str) -> str:
    """Return a stable normalized question string for exact-key caching."""
    return " ".join(str(question).lower().split())


def stable_json(value: Any) -> str:
    """Serialize a key component deterministically."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest_payload(value: Any) -> str:
    """Return a compact SHA256 digest for a deterministic JSON payload."""
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def namespaced_key(namespace: str, layer: str, payload: Dict[str, Any]) -> str:
    """Build a Redis key with a stable layer-specific payload digest."""
    digest = digest_payload(payload)
    return f"{namespace}:{CACHE_KEY_VERSION}:{layer}:{digest}"


def canonical_metadata_filter(metadata_filter: Optional[Dict]) -> Dict:
    """Return a copyable canonical metadata filter for exact-key matching."""
    return dict(metadata_filter or {})
