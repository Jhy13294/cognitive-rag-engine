from typing import Callable, List, Optional
from urllib.parse import parse_qs, unquote, urlparse

from .filters import ACLConfigurationError, ACLIdentityMissingError, ACLResolutionError, normalize_acl_values

MYSQL_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS knowledge_base (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(255) NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    knowledge_base_id BIGINT NOT NULL,
    source VARCHAR(1024) NOT NULL,
    checksum VARCHAR(128),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (knowledge_base_id) REFERENCES knowledge_base(id)
);

CREATE TABLE IF NOT EXISTS chunk (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    document_id BIGINT NOT NULL,
    vector_record_id VARCHAR(255) NOT NULL UNIQUE,
    start_char INT,
    end_char INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES document(id)
);

CREATE TABLE IF NOT EXISTS acl_binding (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    document_id BIGINT NULL,
    chunk_id BIGINT NULL,
    acl_subject VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (document_id) REFERENCES document(id),
    FOREIGN KEY (chunk_id) REFERENCES chunk(id),
    INDEX idx_acl_binding_subject (acl_subject)
);

CREATE TABLE IF NOT EXISTS principal_acl_membership (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    principal VARCHAR(255) NOT NULL,
    acl_subject VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uniq_principal_subject (principal, acl_subject),
    INDEX idx_principal_acl_principal (principal)
);
"""


class MySQLACLResolver:
    """Resolve principal ACL subjects from MySQL metadata tables."""

    def __init__(
        self,
        *,
        metadata_db_url: Optional[str] = None,
        host: Optional[str] = None,
        port: int = 3306,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        connection_factory: Optional[Callable] = None,
    ):
        """Initialize the resolver without opening a network connection."""
        self.metadata_db_url = metadata_db_url
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection_factory = connection_factory

    def allowed_acl_for_principal(self, principal: str) -> List[str]:
        """Return the ACL subjects granted to a principal."""
        if not principal or not str(principal).strip():
            raise ACLIdentityMissingError("Authenticated principal is required when ACL is enabled.")

        connection = None
        cursor = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT acl_subject
                FROM principal_acl_membership
                WHERE principal = %s
                ORDER BY acl_subject
                """,
                (str(principal).strip(),),
            )
            allowed = normalize_acl_values([row[0] for row in cursor.fetchall()], field_name="allowed_acl")
        except ACLConfigurationError:
            raise
        except Exception as e:
            raise ACLResolutionError("Failed to resolve ACL subjects from MySQL metadata.") from e
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()

        if not allowed:
            raise ACLResolutionError(f"Principal has no allowed ACL subjects: {principal}")
        return allowed

    def acl_for_source(self, source: str) -> List[str]:
        """Return ACL subjects bound to a document source in MySQL metadata."""
        if not source or not str(source).strip():
            raise ACLResolutionError("Document source is required for ACL binding lookup.")

        connection = None
        cursor = None
        try:
            connection = self._connect()
            cursor = connection.cursor()
            cursor.execute(
                """
                SELECT DISTINCT binding.acl_subject
                FROM acl_binding AS binding
                LEFT JOIN document AS doc_binding
                    ON binding.document_id = doc_binding.id
                LEFT JOIN chunk AS chunk_binding
                    ON binding.chunk_id = chunk_binding.id
                LEFT JOIN document AS chunk_doc
                    ON chunk_binding.document_id = chunk_doc.id
                WHERE doc_binding.source = %s OR chunk_doc.source = %s
                ORDER BY binding.acl_subject
                """,
                (str(source).strip(), str(source).strip()),
            )
            acl_values = normalize_acl_values([row[0] for row in cursor.fetchall()], field_name="source_acl")
        except ACLConfigurationError:
            raise
        except Exception as e:
            raise ACLResolutionError("Failed to resolve source ACL bindings from MySQL metadata.") from e
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()

        if not acl_values:
            raise ACLResolutionError(f"Document source has no ACL bindings: {source}")
        return acl_values

    def check_connectivity(self, timeout_seconds: Optional[float] = None) -> bool:
        """Return whether the metadata database accepts a simple read query."""
        connection = None
        cursor = None
        try:
            extra_options = {}
            if timeout_seconds is not None:
                extra_options["connection_timeout"] = max(1, int(timeout_seconds))
            connection = self._connect(extra_options=extra_options)
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            row = cursor.fetchone()
            return bool(row and row[0] == 1)
        except ACLConfigurationError:
            raise
        except Exception as e:
            raise ACLResolutionError("Failed to connect to MySQL metadata store.") from e
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()

    def _connect(self, extra_options: Optional[dict] = None):
        """Open a MySQL connection from config or a test-supplied factory."""
        if self.connection_factory is not None:
            return self.connection_factory()

        try:
            import mysql.connector
        except ModuleNotFoundError as e:
            raise ACLConfigurationError("mysql-connector-python is required for MySQL ACL resolution.") from e

        options = self._connection_options()
        if extra_options:
            options = {**options, **extra_options}
        return mysql.connector.connect(**options)

    def _connection_options(self) -> dict:
        """Build mysql.connector options from URL or discrete settings."""
        if self.metadata_db_url:
            return _parse_mysql_url(self.metadata_db_url)

        if not self.host or not self.user or not self.database:
            raise ACLConfigurationError("MySQL ACL configuration requires METADATA_DB_URL or MYSQL_HOST/USER/DATABASE.")

        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
        }


def create_acl_resolver_from_config(config) -> MySQLACLResolver:
    """Create the production ACL resolver from application configuration."""
    return MySQLACLResolver(
        metadata_db_url=getattr(config, "METADATA_DB_URL", None),
        host=getattr(config, "MYSQL_HOST", None),
        port=getattr(config, "MYSQL_PORT", 3306),
        user=getattr(config, "MYSQL_USER", None),
        password=getattr(config, "MYSQL_PASSWORD", None),
        database=getattr(config, "MYSQL_DATABASE", None),
    )


def _parse_mysql_url(url: str) -> dict:
    """Parse mysql://user:password@host:port/database into connector options."""
    parsed = urlparse(url)
    if parsed.scheme not in {"mysql", "mysql+mysqlconnector"}:
        raise ACLConfigurationError("METADATA_DB_URL must use mysql:// or mysql+mysqlconnector://.")

    database = parsed.path.lstrip("/")
    if not parsed.hostname or not parsed.username or not database:
        raise ACLConfigurationError("METADATA_DB_URL must include host, user, and database.")

    query = parse_qs(parsed.query)
    options = {
        "host": parsed.hostname,
        "port": parsed.port or 3306,
        "user": unquote(parsed.username),
        "password": unquote(parsed.password or ""),
        "database": unquote(database),
    }
    if "ssl_ca" in query:
        options["ssl_ca"] = query["ssl_ca"][0]
    return options
