import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from access.mysql import MYSQL_SCHEMA_SQL, MySQLACLResolver
from config import Config


DEMO_KB_NAME = os.getenv("COMPOSE_INIT_KB_NAME", "compose-demo")
DEMO_KB_PATH = Path(os.getenv("COMPOSE_INIT_KNOWLEDGE_BASE_PATH", "/app/eval/fixtures/knowledge_base"))

DEMO_PRINCIPALS: Dict[str, List[str]] = {
    "alice": ["role:finance", "role:employee"],
    "bob": ["role:legal", "role:employee"],
    "ops": ["role:support", "role:employee"],
    "product": ["role:product", "role:employee"],
}

DEMO_FILE_ACL: Dict[str, List[str]] = {
    "finance-policy.md": ["role:finance"],
    "security-policy.md": ["role:legal"],
    "support-runbook.md": ["role:support"],
    "product-glossary.md": ["role:product"],
    "employee-handbook.md": ["role:finance", "role:legal", "role:support", "role:product", "role:employee"],
}


def main() -> int:
    print("compose-init: starting schema, ACL seed, and demo ingest", flush=True)
    connection = _connect_mysql()
    try:
        _create_schema(connection)
        _seed_acl(connection)
    finally:
        connection.close()

    _ingest_demo_documents()
    print("compose-init: completed", flush=True)
    return 0


def _connect_mysql():
    resolver = MySQLACLResolver(
        metadata_db_url=Config.METADATA_DB_URL,
        host=Config.MYSQL_HOST,
        port=Config.MYSQL_PORT,
        user=Config.MYSQL_USER,
        password=Config.MYSQL_PASSWORD,
        database=Config.MYSQL_DATABASE,
    )
    return resolver._connect()


def _create_schema(connection) -> None:
    cursor = connection.cursor()
    try:
        for statement in _split_sql(MYSQL_SCHEMA_SQL):
            cursor.execute(statement)
        connection.commit()
        print("compose-init: mysql schema is ready", flush=True)
    finally:
        cursor.close()


def _seed_acl(connection) -> None:
    cursor = connection.cursor()
    try:
        kb_id = _ensure_knowledge_base(cursor, DEMO_KB_NAME)
        _seed_principal_memberships(cursor)
        for file_name, subjects in DEMO_FILE_ACL.items():
            source_path = str((DEMO_KB_PATH / file_name).resolve())
            document_id = _ensure_document(cursor, kb_id, source_path)
            _seed_document_acl_bindings(cursor, document_id, subjects)
        connection.commit()
        print("compose-init: demo ACL metadata is ready", flush=True)
    finally:
        cursor.close()


def _ensure_knowledge_base(cursor, name: str) -> int:
    cursor.execute("INSERT IGNORE INTO knowledge_base (name) VALUES (%s)", (name,))
    cursor.execute("SELECT id FROM knowledge_base WHERE name = %s", (name,))
    row = cursor.fetchone()
    if not row:
        raise RuntimeError(f"Failed to seed knowledge base: {name}")
    return int(row[0])


def _seed_principal_memberships(cursor) -> None:
    for principal, subjects in DEMO_PRINCIPALS.items():
        for subject in subjects:
            cursor.execute(
                """
                INSERT IGNORE INTO principal_acl_membership (principal, acl_subject)
                VALUES (%s, %s)
                """,
                (principal, subject),
            )


def _ensure_document(cursor, knowledge_base_id: int, source: str) -> int:
    cursor.execute(
        """
        SELECT id
        FROM document
        WHERE knowledge_base_id = %s AND source = %s
        ORDER BY id
        LIMIT 1
        """,
        (knowledge_base_id, source),
    )
    row = cursor.fetchone()
    if row:
        return int(row[0])

    cursor.execute(
        """
        INSERT INTO document (knowledge_base_id, source)
        VALUES (%s, %s)
        """,
        (knowledge_base_id, source),
    )
    return int(cursor.lastrowid)


def _seed_document_acl_bindings(cursor, document_id: int, subjects: Iterable[str]) -> None:
    for subject in subjects:
        cursor.execute(
            """
            SELECT id
            FROM acl_binding
            WHERE document_id = %s AND chunk_id IS NULL AND acl_subject = %s
            LIMIT 1
            """,
            (document_id, subject),
        )
        if cursor.fetchone():
            continue
        cursor.execute(
            """
            INSERT INTO acl_binding (document_id, acl_subject)
            VALUES (%s, %s)
            """,
            (document_id, subject),
        )


def _ingest_demo_documents() -> None:
    if not DEMO_KB_PATH.exists():
        raise FileNotFoundError(f"Demo knowledge base path does not exist: {DEMO_KB_PATH}")

    for file_name, subjects in DEMO_FILE_ACL.items():
        file_path = DEMO_KB_PATH / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Demo document does not exist: {file_path}")

        command = [
            sys.executable,
            "rag_cli.py",
            "ingest",
            str(file_path),
            "--non-recursive",
            "--vector-store",
            "qdrant",
        ]
        for subject in subjects:
            command.extend(["--acl", subject])

        print(f"compose-init: ingesting {file_name} with acl={','.join(subjects)}", flush=True)
        subprocess.run(command, cwd="/app", check=True)


def _split_sql(sql: str) -> List[str]:
    return [statement.strip() for statement in sql.split(";") if statement.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
