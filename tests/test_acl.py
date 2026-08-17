import unittest

from access import (
    ACLDeniedError,
    ACLIdentityMissingError,
    ACLResolutionError,
    MySQLACLResolver,
    StaticACLResolver,
    build_acl_filter,
    build_effective_metadata_filter,
    merge_metadata_filters,
    metadata_matches,
    normalize_acl_values,
)


class ACLFilterTests(unittest.TestCase):
    def test_normalize_acl_values_is_stable_and_deduplicated(self):
        values = normalize_acl_values([" role:finance ", "role:hr", "role:finance", "", None])

        self.assertEqual(values, ["role:finance", "role:hr"])

    def test_metadata_matches_keeps_scalar_exact_match_semantics(self):
        metadata = {"file_type": "txt", "source": "a.md"}

        self.assertTrue(metadata_matches(metadata, {"file_type": "txt"}))
        self.assertFalse(metadata_matches(metadata, {"file_type": "pdf"}))

    def test_metadata_matches_list_filter_uses_set_intersection(self):
        metadata = {"acl": ["role:finance", "role:admin"]}

        self.assertTrue(metadata_matches(metadata, {"acl": ["role:legal", "role:finance"]}))
        self.assertFalse(metadata_matches(metadata, {"acl": ["role:legal"]}))

    def test_metadata_matches_scalar_record_acl_against_list_filter(self):
        metadata = {"acl": "role:finance"}

        self.assertTrue(metadata_matches(metadata, {"acl": ["role:finance", "role:admin"]}))
        self.assertFalse(metadata_matches(metadata, {"acl": ["role:legal"]}))

    def test_empty_or_missing_acl_is_restricted_under_acl_filter(self):
        self.assertFalse(metadata_matches({}, {"acl": ["role:finance"]}))
        self.assertFalse(metadata_matches({"acl": []}, {"acl": ["role:finance"]}))

    def test_build_acl_filter_fails_closed_for_empty_allowed_set(self):
        with self.assertRaises(ACLDeniedError):
            build_acl_filter([], metadata_key="acl", default_deny=True)

    def test_merge_metadata_filters_intersects_client_acl_with_server_acl(self):
        merged = merge_metadata_filters(
            {"file_type": "txt", "acl": ["role:finance", "role:legal"]},
            {"acl": ["role:finance", "role:admin"]},
            metadata_key="acl",
        )

        self.assertEqual(merged, {"file_type": "txt", "acl": ["role:finance"]})

    def test_merge_metadata_filters_rejects_client_acl_widening(self):
        with self.assertRaises(ACLDeniedError):
            merge_metadata_filters(
                {"acl": ["role:public"]},
                {"acl": ["role:secret"]},
                metadata_key="acl",
            )

    def test_build_effective_metadata_filter_adds_server_acl(self):
        effective = build_effective_metadata_filter(
            {"source": "handbook.md"},
            ["role:hr", "role:finance"],
            metadata_key="acl",
            default_deny=True,
        )

        self.assertEqual(effective, {"source": "handbook.md", "acl": ["role:finance", "role:hr"]})


class StaticACLResolverTests(unittest.TestCase):
    def test_static_resolver_returns_normalized_acl_subjects(self):
        resolver = StaticACLResolver({"alice": ["role:finance", " role:admin "]})

        self.assertEqual(
            resolver.allowed_acl_for_principal("alice"), ["role:admin", "role:finance"]
        )

    def test_static_resolver_fails_closed_without_identity(self):
        resolver = StaticACLResolver({"alice": ["role:finance"]})

        with self.assertRaises(ACLIdentityMissingError):
            resolver.allowed_acl_for_principal("")

    def test_static_resolver_fails_closed_for_unknown_principal(self):
        resolver = StaticACLResolver({"alice": ["role:finance"]})

        with self.assertRaises(ACLResolutionError):
            resolver.allowed_acl_for_principal("bob")


class MySQLACLResolverTests(unittest.TestCase):
    def test_mysql_resolver_reads_principal_memberships(self):
        class FakeCursor:
            def __init__(self):
                self.params = None

            def execute(self, _sql, params):
                self.params = params

            def fetchall(self):
                assert self.params == ("alice",)
                return [("role:finance",), ("role:admin",)]

            def close(self):
                pass

        class FakeConnection:
            def __init__(self):
                self.cursor_instance = FakeCursor()

            def cursor(self):
                return self.cursor_instance

            def close(self):
                pass

        connection = FakeConnection()
        resolver = MySQLACLResolver(connection_factory=lambda: connection)

        allowed = resolver.allowed_acl_for_principal("alice")

        self.assertEqual(allowed, ["role:admin", "role:finance"])

    def test_mysql_resolver_reads_source_acl_bindings(self):
        class FakeCursor:
            def __init__(self):
                self.params = None

            def execute(self, _sql, params):
                self.params = params

            def fetchall(self):
                assert self.params == ("finance.md", "finance.md")
                return [("role:finance",), ("role:admin",)]

            def close(self):
                pass

        class FakeConnection:
            def __init__(self):
                self.cursor_instance = FakeCursor()

            def cursor(self):
                return self.cursor_instance

            def close(self):
                pass

        resolver = MySQLACLResolver(connection_factory=FakeConnection)

        allowed = resolver.acl_for_source("finance.md")

        self.assertEqual(allowed, ["role:admin", "role:finance"])


if __name__ == "__main__":
    unittest.main()
