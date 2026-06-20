from typing import Dict, List, Protocol

from .filters import ACLIdentityMissingError, ACLResolutionError, normalize_acl_values


class ACLResolver(Protocol):
    """Resolve an authenticated principal into allowed ACL subjects."""

    def allowed_acl_for_principal(self, principal: str) -> List[str]:
        """Return the allowed ACL subjects for a principal."""


class StaticACLResolver:
    """Deterministic ACL resolver for tests and local CLI usage."""

    def __init__(self, mapping: Dict[str, List[str]]):
        """Store a principal-to-ACL mapping."""
        self.mapping = {str(key): normalize_acl_values(value) for key, value in mapping.items()}

    def allowed_acl_for_principal(self, principal: str) -> List[str]:
        """Return allowed ACL subjects or fail closed."""
        if not principal or not str(principal).strip():
            raise ACLIdentityMissingError("Authenticated principal is required when ACL is enabled.")

        principal_key = str(principal).strip()
        if principal_key not in self.mapping:
            raise ACLResolutionError(f"ACL subjects are not configured for principal: {principal_key}")

        allowed = normalize_acl_values(self.mapping[principal_key], field_name="allowed_acl")
        if not allowed:
            raise ACLResolutionError(f"Principal has no allowed ACL subjects: {principal_key}")
        return allowed
