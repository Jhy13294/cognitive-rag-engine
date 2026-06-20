"""Gated real-MySQL ACL integration tests.

These exercise the production MySQLACLResolver against a real MySQL server
(real mysql.connector, real SQL, real schema) instead of the FakeConnection
unit test. They are skipped unless RUN_MYSQL_ACL_INTEGRATION=1 and MySQL
connection settings are present, mirroring the Qdrant integration gating.

Provision once (matches .env.example defaults):
    CREATE DATABASE IF NOT EXISTS rag_metadata CHARACTER SET utf8mb4;
    CREATE USER IF NOT EXISTS 'rag_user'@'localhost' IDENTIFIED BY 'rag_password';
    GRANT ALL PRIVILEGES ON rag_metadata.* TO 'rag_user'@'localhost';
    FLUSH PRIVILEGES;

Run:
    RUN_MYSQL_ACL_INTEGRATION=1 METADATA_DB_URL=mysql://rag_user:rag_password@localhost:3306/rag_metadata \
        ./.venv/Scripts/python.exe -m unittest tests.test_acl_mysql_integration -v
"""

import os
import unittest

from access import (
    ACLIdentityMissingError,
    ACLResolutionError,
    MySQLACLResolver,
    build_effective_metadata_filter,
)
from access.mysql import MYSQL_SCHEMA_SQL, _parse_mysql_url
from vector_store import VectorRecord
from vector_store.memory_store import InMemoryVectorStore

_TEST_PRINCIPALS = ("acl_it_alice", "acl_it_bob", "acl_it_carol")
_TEST_KB_NAME = "acl_it_kb"
_TEST_SOURCE = "acl_it_finance.md"


def _mysql_settings_present() -> bool:
    """Return whether enough MySQL settings exist to attempt a real connection."""
    if os.getenv("METADATA_DB_URL"):
        return True
    return bool(os.getenv("MYSQL_HOST") and os.getenv("MYSQL_USER") and os.getenv("MYSQL_DATABASE"))


def _connection_options() -> dict:
    """Build mysql.connector options from env (URL preferred, else discrete parts)."""
    url = os.getenv("METADATA_DB_URL")
    if url:
        return _parse_mysql_url(url)
    return {
        "host": os.getenv("MYSQL_HOST"),
        "port": int(os.getenv("MYSQL_PORT", "3306")),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD", ""),
        "database": os.getenv("MYSQL_DATABASE"),
    }


@unittest.skipUnless(
    os.getenv("RUN_MYSQL_ACL_INTEGRATION") and _mysql_settings_present(),
    "Set RUN_MYSQL_ACL_INTEGRATION=1 and METADATA_DB_URL or MYSQL_HOST/USER/DATABASE to run real-MySQL ACL tests.",
)
class MySQLACLResolverIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mysql.connector  # noqa: F401
        except ModuleNotFoundError as e:
            raise unittest.SkipTest("mysql-connector-python is not installed") from e

        cls.options = _connection_options()
        cls._apply_schema_and_seed()

    @classmethod
    def _connect(cls):
        import mysql.connector

        return mysql.connector.connect(connection_timeout=5, **cls.options)

    @classmethod
    def _apply_schema_and_seed(cls):
        """Create the real schema and seed deterministic principal memberships."""
        connection = cls._connect()
        cursor = connection.cursor()
        try:
            for statement in MYSQL_SCHEMA_SQL.split(";"):
                if statement.strip():
                    cursor.execute(statement)
            placeholders = ", ".join(["%s"] * len(_TEST_PRINCIPALS))
            cursor.execute(
                f"DELETE FROM principal_acl_membership WHERE principal IN ({placeholders})",
                _TEST_PRINCIPALS,
            )
            cls._delete_test_documents(cursor)
            cursor.execute("DELETE FROM knowledge_base WHERE name = %s", (_TEST_KB_NAME,))
            cursor.executemany(
                "INSERT INTO principal_acl_membership (principal, acl_subject) VALUES (%s, %s)",
                [
                    ("acl_it_alice", "role:finance"),
                    ("acl_it_alice", "role:admin"),
                    ("acl_it_bob", "role:legal"),
                ],
            )
            cursor.execute("INSERT INTO knowledge_base (name) VALUES (%s)", (_TEST_KB_NAME,))
            knowledge_base_id = cursor.lastrowid
            cursor.execute(
                "INSERT INTO document (knowledge_base_id, source, checksum) VALUES (%s, %s, %s)",
                (knowledge_base_id, _TEST_SOURCE, "acl-it-checksum"),
            )
            document_id = cursor.lastrowid
            cursor.executemany(
                "INSERT INTO acl_binding (document_id, acl_subject) VALUES (%s, %s)",
                [
                    (document_id, "role:finance"),
                    (document_id, "role:admin"),
                ],
            )
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    @classmethod
    def _delete_test_documents(cls, cursor):
        """Delete deterministic test documents and dependent metadata rows."""
        cursor.execute("SELECT id FROM document WHERE source = %s", (_TEST_SOURCE,))
        document_ids = [row[0] for row in cursor.fetchall()]
        if not document_ids:
            return
        placeholders = ", ".join(["%s"] * len(document_ids))
        cursor.execute(f"DELETE FROM acl_binding WHERE document_id IN ({placeholders})", document_ids)
        cursor.execute(f"DELETE FROM chunk WHERE document_id IN ({placeholders})", document_ids)
        cursor.execute(f"DELETE FROM document WHERE id IN ({placeholders})", document_ids)

    @classmethod
    def tearDownClass(cls):
        """Remove seeded rows so the table is left as found."""
        connection = cls._connect()
        cursor = connection.cursor()
        try:
            placeholders = ", ".join(["%s"] * len(_TEST_PRINCIPALS))
            cursor.execute(
                f"DELETE FROM principal_acl_membership WHERE principal IN ({placeholders})",
                _TEST_PRINCIPALS,
            )
            cls._delete_test_documents(cursor)
            cursor.execute("DELETE FROM knowledge_base WHERE name = %s", (_TEST_KB_NAME,))
            connection.commit()
        finally:
            cursor.close()
            connection.close()

    def _resolver(self) -> MySQLACLResolver:
        """Build a resolver that opens a real connection (no connection_factory)."""
        return MySQLACLResolver(**{**self.options, "metadata_db_url": None})

    def test_real_mysql_resolves_seeded_principal_memberships(self):
        resolver = MySQLACLResolver(
            host=self.options.get("host"),
            port=self.options.get("port", 3306),
            user=self.options.get("user"),
            password=self.options.get("password"),
            database=self.options.get("database"),
        )

        self.assertEqual(
            resolver.allowed_acl_for_principal("acl_it_alice"),
            ["role:admin", "role:finance"],
        )
        self.assertEqual(resolver.allowed_acl_for_principal("acl_it_bob"), ["role:legal"])

    def test_real_mysql_resolves_document_source_acl_bindings(self):
        resolver = self._resolver()

        self.assertEqual(resolver.acl_for_source(_TEST_SOURCE), ["role:admin", "role:finance"])

    def test_real_mysql_unknown_principal_fails_closed(self):
        resolver = self._resolver()
        with self.assertRaises(ACLResolutionError):
            resolver.allowed_acl_for_principal("acl_it_carol")

    def test_real_mysql_empty_principal_fails_closed(self):
        resolver = self._resolver()
        with self.assertRaises(ACLIdentityMissingError):
            resolver.allowed_acl_for_principal("   ")

    def test_real_mysql_acl_drives_pre_filter_leakage(self):
        """Resolved real-MySQL ACL must actually gate retrieval (no leak)."""
        resolver = self._resolver()
        allowed_acl = resolver.allowed_acl_for_principal("acl_it_alice")  # finance, admin
        effective = build_effective_metadata_filter(None, allowed_acl, metadata_key="acl", default_deny=True)

        store = InMemoryVectorStore()
        store.add_records([
            VectorRecord(id="finance-doc", content="finance", embedding=[1.0, 0.0],
                         metadata={"acl": ["role:finance"], "source": "fin.md"}),
            VectorRecord(id="secret-doc", content="secret", embedding=[0.99, 0.14],
                         metadata={"acl": ["role:secret"], "source": "sec.md"}),
        ])

        ids = [r.record.id for r in store.similarity_search([1.0, 0.0], top_k=5, metadata_filter=effective)]
        self.assertEqual(ids, ["finance-doc"])

        bob_acl = resolver.allowed_acl_for_principal("acl_it_bob")  # legal only
        bob_effective = build_effective_metadata_filter(None, bob_acl, metadata_key="acl", default_deny=True)
        bob_ids = [r.record.id for r in store.similarity_search([1.0, 0.0], top_k=5, metadata_filter=bob_effective)]
        self.assertEqual(bob_ids, [])


if __name__ == "__main__":
    unittest.main()
