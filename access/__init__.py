"""Access-control helpers for query-time ACL pre-filtering."""

from .filters import (
    ACLAccessError,
    ACLConfigurationError,
    ACLDeniedError,
    ACLIdentityMissingError,
    ACLResolutionError,
    build_acl_filter,
    build_effective_metadata_filter,
    merge_metadata_filters,
    metadata_matches,
    normalize_acl_values,
)
from .mysql import MYSQL_SCHEMA_SQL, MySQLACLResolver, create_acl_resolver_from_config
from .resolver import ACLResolver, StaticACLResolver

__all__ = [
    "ACLAccessError",
    "ACLConfigurationError",
    "ACLDeniedError",
    "ACLIdentityMissingError",
    "ACLResolutionError",
    "ACLResolver",
    "MYSQL_SCHEMA_SQL",
    "MySQLACLResolver",
    "StaticACLResolver",
    "build_acl_filter",
    "build_effective_metadata_filter",
    "create_acl_resolver_from_config",
    "merge_metadata_filters",
    "metadata_matches",
    "normalize_acl_values",
]
