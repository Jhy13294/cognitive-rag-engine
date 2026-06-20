from collections.abc import Iterable
from typing import Any, Dict, List, Optional


class ACLAccessError(ValueError):
    """Base class for fail-closed ACL errors."""


class ACLIdentityMissingError(ACLAccessError):
    """Raised when ACL is enabled but no authenticated principal is available."""


class ACLResolutionError(ACLAccessError):
    """Raised when the principal ACL set cannot be resolved."""


class ACLConfigurationError(ACLAccessError):
    """Raised when ACL configuration is invalid or incomplete."""


class ACLDeniedError(ACLAccessError):
    """Raised when an effective ACL filter denies all access."""


def normalize_acl_values(values: Any, *, field_name: str = "acl") -> List[str]:
    """Return stable, non-empty ACL subjects from a scalar or sequence."""
    raw_values = _coerce_sequence(values)
    normalized = []
    for value in raw_values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            normalized.append(text)

    return sorted(set(normalized))


def build_acl_filter(
    allowed_acl: Any,
    *,
    metadata_key: str = "acl",
    default_deny: bool = True,
) -> Dict[str, List[str]]:
    """Build the mandatory server-side ACL metadata filter."""
    if not metadata_key or not str(metadata_key).strip():
        raise ACLConfigurationError("ACL metadata key must not be empty.")

    allowed_values = normalize_acl_values(allowed_acl, field_name="allowed_acl")
    if not allowed_values and default_deny:
        raise ACLDeniedError("Principal has no allowed ACL subjects.")

    return {metadata_key: allowed_values}


def build_effective_metadata_filter(
    client_filter: Optional[Dict[str, Any]],
    allowed_acl: Any,
    *,
    metadata_key: str = "acl",
    default_deny: bool = True,
) -> Dict[str, Any]:
    """AND a client metadata filter with the mandatory server ACL filter."""
    acl_filter = build_acl_filter(
        allowed_acl,
        metadata_key=metadata_key,
        default_deny=default_deny,
    )
    return merge_metadata_filters(client_filter, acl_filter, metadata_key=metadata_key)


def merge_metadata_filters(
    client_filter: Optional[Dict[str, Any]],
    acl_filter: Dict[str, Any],
    *,
    metadata_key: str = "acl",
) -> Dict[str, Any]:
    """Merge client-controlled metadata with server ACL without allowing ACL widening."""
    merged = dict(client_filter or {})
    server_allowed = normalize_acl_values(acl_filter.get(metadata_key), field_name="server_acl")

    if metadata_key in merged:
        client_allowed = normalize_acl_values(merged[metadata_key], field_name="client_acl")
        effective_allowed = sorted(set(client_allowed).intersection(server_allowed))
        if not effective_allowed:
            raise ACLDeniedError("Client ACL filter has no overlap with the authenticated ACL set.")
        merged[metadata_key] = effective_allowed
    else:
        merged[metadata_key] = server_allowed

    return merged


def metadata_matches(metadata: Dict[str, Any], metadata_filter: Dict[str, Any]) -> bool:
    """Return whether metadata satisfies exact scalar filters and list intersection filters."""
    for key, expected in metadata_filter.items():
        actual = metadata.get(key)
        if _is_sequence_filter(expected):
            expected_values = _comparable_set(expected)
            if not expected_values:
                return False
            if not _comparable_set(actual).intersection(expected_values):
                return False
            continue

        if actual != expected:
            return False

    return True


def _coerce_sequence(values: Any) -> List[Any]:
    """Convert a scalar or iterable into a list, treating strings as scalars."""
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, Iterable):
        return list(values)
    return [values]


def _is_sequence_filter(value: Any) -> bool:
    """Return whether a metadata filter value should use MatchAny-style semantics."""
    return isinstance(value, (list, tuple, set))


def _comparable_set(value: Any) -> set:
    """Return a comparable set for scalar or sequence metadata values."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {item for item in value if item is not None}
    return {value}
