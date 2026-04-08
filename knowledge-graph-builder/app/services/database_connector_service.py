"""Database Connector Service — PostgreSQL, MySQL, MongoDB ingestion (ORA-77).

Implements the spec from ORA-72:
- Abstract DatabaseConnector interface
- PostgreSQLConnector, MySQLConnector, MongoDBConnector implementations
- Schema-to-KG mapping rules (SQL FK→relationship, MongoDB ObjectId refs)
- Three sync modes: full_snapshot, schema_only, CDC
- Neo4j Connector node lifecycle (register, list, get, delete, update)
- SSRF guard on host registration
- Multi-tenancy: every Cypher query filters by graph_id
"""

from __future__ import annotations

import ipaddress
import json
import re as _re
import socket
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status

from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------


class DatabaseConnectorType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    MONGODB = "mongodb"


class DbSyncMode(str, Enum):
    FULL_SNAPSHOT = "full_snapshot"
    SCHEMA_ONLY = "schema_only"
    CDC = "cdc"


# ---------------------------------------------------------------------------
# Schema introspection data classes
# ---------------------------------------------------------------------------


@dataclass
class ColumnMeta:
    name: str
    data_type: str
    nullable: bool
    is_pk: bool
    is_fk: bool
    fk_table: Optional[str] = None
    fk_column: Optional[str] = None


@dataclass
class TableMeta:
    name: str
    schema_name: str
    columns: List[ColumnMeta] = field(default_factory=list)
    row_count: Optional[int] = None


@dataclass
class SchemaSnapshot:
    connector_type: DatabaseConnectorType
    database: str
    tables: List[TableMeta]
    captured_at: datetime

    def to_json(self) -> str:
        """Serialize snapshot for CDC storage in Neo4j."""
        return json.dumps(
            {
                "connector_type": self.connector_type.value,
                "database": self.database,
                "captured_at": self.captured_at.isoformat(),
                "tables": [
                    {
                        "name": t.name,
                        "schema_name": t.schema_name,
                        "columns": [
                            {
                                "name": c.name,
                                "data_type": c.data_type,
                                "nullable": c.nullable,
                                "is_pk": c.is_pk,
                                "is_fk": c.is_fk,
                                "fk_table": c.fk_table,
                                "fk_column": c.fk_column,
                            }
                            for c in t.columns
                        ],
                    }
                    for t in self.tables
                ],
            }
        )

    @classmethod
    def from_json(cls, data: str) -> "SchemaSnapshot":
        obj = json.loads(data)
        tables = [
            TableMeta(
                name=t["name"],
                schema_name=t["schema_name"],
                columns=[
                    ColumnMeta(
                        name=c["name"],
                        data_type=c["data_type"],
                        nullable=c["nullable"],
                        is_pk=c["is_pk"],
                        is_fk=c["is_fk"],
                        fk_table=c.get("fk_table"),
                        fk_column=c.get("fk_column"),
                    )
                    for c in t["columns"]
                ],
            )
            for t in obj["tables"]
        ]
        return cls(
            connector_type=DatabaseConnectorType(obj["connector_type"]),
            database=obj["database"],
            tables=tables,
            captured_at=datetime.fromisoformat(obj["captured_at"]),
        )


@dataclass
class SampleRow:
    table_name: str
    row_data: Dict[str, Any]


# ---------------------------------------------------------------------------
# Abstract connector interface
# ---------------------------------------------------------------------------


class DatabaseConnector(ABC):
    """Abstract base for all database connector implementations."""

    @abstractmethod
    async def connect(self, user: str, password: str) -> None:
        """Open connection. Credentials injected by caller — never stored."""
        ...

    @abstractmethod
    async def introspect_schema(self) -> SchemaSnapshot:
        """Return full schema metadata without any row data."""
        ...

    @abstractmethod
    async def extract_sample_data(self, table: str, limit: int) -> List[SampleRow]:
        """Return up to `limit` rows from `table` as dict records."""
        ...

    @abstractmethod
    async def detect_schema_changes(
        self, previous_snapshot: SchemaSnapshot
    ) -> Dict[str, Any]:
        """Compare current schema against a previous snapshot.

        Returns dict with keys: added_tables, removed_tables, altered_tables.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release DB connection."""
        ...


# ---------------------------------------------------------------------------
# SSRF guard
# ---------------------------------------------------------------------------

_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("0.0.0.0/8"),
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),
]

_BLOCKED_HOSTNAMES = {
    "localhost",
    "metadata.google.internal",
    "169.254.169.254",
}


def validate_host_ssrf(host: str) -> None:
    """Raise HTTP 422 if host is a private/loopback/link-local address.

    Called at connector registration time to prevent SSRF attacks.
    """
    if host.lower() in _BLOCKED_HOSTNAMES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Host '{host}' is not allowed.",
        )
    try:
        addr = ipaddress.ip_address(host)
        for net in _PRIVATE_NETWORKS:
            if addr in net:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Host '{host}' is in a private/loopback range and is not allowed.",
                )
    except ValueError:
        # hostname — resolve and check resolved IP
        try:
            resolved = socket.gethostbyname(host)
            resolved_addr = ipaddress.ip_address(resolved)
            for net in _PRIVATE_NETWORKS:
                if resolved_addr in net:
                    raise HTTPException(
                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                        detail=f"Host '{host}' resolves to a private/loopback address and is not allowed.",
                    )
        except socket.gaierror:
            pass  # DNS lookup failed — let connect() fail naturally


# ---------------------------------------------------------------------------
# PostgreSQL connector
# ---------------------------------------------------------------------------


class PostgreSQLConnector(DatabaseConnector):
    """Async PostgreSQL connector using asyncpg."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._host: str = config["host"]
        self._port: int = config["port"]
        self._database: str = config["database"]
        self._schema: str = config.get("schema_filter") or "public"
        self._table_filter: Optional[List[str]] = config.get("table_filter")
        self._sample_row_limit: int = config.get("sample_row_limit", 100)
        self._conn = None

    async def connect(self, user: str, password: str) -> None:
        import asyncpg  # type: ignore

        self._conn = await asyncpg.connect(
            host=self._host,
            port=self._port,
            user=user,
            password=password,
            database=self._database,
            timeout=10,
            command_timeout=30,
        )

    async def introspect_schema(self) -> SchemaSnapshot:
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        rows = await self._conn.fetch(
            """
            SELECT
                c.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                CASE WHEN pk.column_name IS NOT NULL THEN TRUE ELSE FALSE END AS is_pk,
                CASE WHEN fk.column_name IS NOT NULL THEN TRUE ELSE FALSE END AS is_fk,
                fk.foreign_table_name AS fk_table,
                fk.foreign_column_name AS fk_column
            FROM information_schema.columns c
            LEFT JOIN (
                SELECT ku.column_name, ku.table_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                  ON tc.constraint_name = ku.constraint_name
                  AND tc.table_schema = ku.table_schema
                WHERE tc.constraint_type = 'PRIMARY KEY'
                  AND tc.table_schema = $1
            ) pk ON pk.table_name = c.table_name AND pk.column_name = c.column_name
            LEFT JOIN (
                SELECT
                    ku.column_name,
                    ku.table_name,
                    ccu.table_name AS foreign_table_name,
                    ccu.column_name AS foreign_column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage ku
                  ON tc.constraint_name = ku.constraint_name
                  AND tc.table_schema = ku.table_schema
                JOIN information_schema.referential_constraints rc
                  ON tc.constraint_name = rc.constraint_name
                  AND tc.table_schema = rc.constraint_schema
                JOIN information_schema.constraint_column_usage ccu
                  ON rc.unique_constraint_name = ccu.constraint_name
                  AND rc.unique_constraint_schema = ccu.constraint_schema
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = $1
            ) fk ON fk.table_name = c.table_name AND fk.column_name = c.column_name
            WHERE c.table_schema = $1
            ORDER BY c.table_name, c.ordinal_position
            """,
            self._schema,
        )

        tables: Dict[str, TableMeta] = {}
        for row in rows:
            tname = row["table_name"]
            if self._table_filter and tname not in self._table_filter:
                continue
            if tname not in tables:
                tables[tname] = TableMeta(name=tname, schema_name=self._schema)
            tables[tname].columns.append(
                ColumnMeta(
                    name=row["column_name"],
                    data_type=row["data_type"],
                    nullable=row["is_nullable"] == "YES",
                    is_pk=bool(row["is_pk"]),
                    is_fk=bool(row["is_fk"]),
                    fk_table=row["fk_table"],
                    fk_column=row["fk_column"],
                )
            )

        return SchemaSnapshot(
            connector_type=DatabaseConnectorType.POSTGRESQL,
            database=self._database,
            tables=list(tables.values()),
            captured_at=datetime.now(timezone.utc),
        )

    async def extract_sample_data(self, table: str, limit: int) -> List[SampleRow]:
        if not self._conn:
            raise RuntimeError("Not connected.")
        # Table/schema identifiers cannot be parameterized in SQL — asyncpg $N
        # placeholders are for values only, not identifiers.  Both values come
        # from DB schema introspection (never direct user input).  Standard SQL
        # double-quoting prevents injection from names containing special chars.
        rows = await self._conn.fetch(
            f'SELECT * FROM "{self._schema}"."{table}" LIMIT $1', limit
        )
        return [SampleRow(table_name=table, row_data=dict(r)) for r in rows]

    async def detect_schema_changes(
        self, previous_snapshot: SchemaSnapshot
    ) -> Dict[str, Any]:
        current = await self.introspect_schema()
        return _diff_snapshots(previous_snapshot, current)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# MySQL connector
# ---------------------------------------------------------------------------


class MySQLConnector(DatabaseConnector):
    """Async MySQL connector using aiomysql."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._host: str = config["host"]
        self._port: int = config["port"]
        self._database: str = config["database"]
        self._table_filter: Optional[List[str]] = config.get("table_filter")
        self._sample_row_limit: int = config.get("sample_row_limit", 100)
        self._conn = None

    async def connect(self, user: str, password: str) -> None:
        import aiomysql  # type: ignore

        self._conn = await aiomysql.connect(
            host=self._host,
            port=self._port,
            user=user,
            password=password,
            db=self._database,
            connect_timeout=10,
        )

    async def introspect_schema(self) -> SchemaSnapshot:
        if not self._conn:
            raise RuntimeError("Not connected.")

        import aiomysql  # type: ignore

        async with self._conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(
                """
                SELECT
                    c.TABLE_NAME     AS table_name,
                    c.COLUMN_NAME    AS column_name,
                    c.DATA_TYPE      AS data_type,
                    c.IS_NULLABLE    AS is_nullable,
                    CASE WHEN c.COLUMN_KEY = 'PRI' THEN 1 ELSE 0 END AS is_pk,
                    CASE WHEN k.REFERENCED_TABLE_NAME IS NOT NULL THEN 1 ELSE 0 END AS is_fk,
                    k.REFERENCED_TABLE_NAME  AS fk_table,
                    k.REFERENCED_COLUMN_NAME AS fk_column
                FROM INFORMATION_SCHEMA.COLUMNS c
                LEFT JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE k
                  ON  c.TABLE_SCHEMA  = k.TABLE_SCHEMA
                  AND c.TABLE_NAME    = k.TABLE_NAME
                  AND c.COLUMN_NAME   = k.COLUMN_NAME
                  AND k.REFERENCED_TABLE_NAME IS NOT NULL
                WHERE c.TABLE_SCHEMA = %s
                ORDER BY c.TABLE_NAME, c.ORDINAL_POSITION
                """,
                (self._database,),
            )
            rows = await cur.fetchall()

        tables: Dict[str, TableMeta] = {}
        for row in rows:
            tname = row["table_name"]
            if self._table_filter and tname not in self._table_filter:
                continue
            if tname not in tables:
                tables[tname] = TableMeta(name=tname, schema_name=self._database)
            tables[tname].columns.append(
                ColumnMeta(
                    name=row["column_name"],
                    data_type=row["data_type"],
                    nullable=row["is_nullable"] == "YES",
                    is_pk=bool(row["is_pk"]),
                    is_fk=bool(row["is_fk"]),
                    fk_table=row["fk_table"],
                    fk_column=row["fk_column"],
                )
            )

        return SchemaSnapshot(
            connector_type=DatabaseConnectorType.MYSQL,
            database=self._database,
            tables=list(tables.values()),
            captured_at=datetime.now(timezone.utc),
        )

    async def extract_sample_data(self, table: str, limit: int) -> List[SampleRow]:
        if not self._conn:
            raise RuntimeError("Not connected.")
        import aiomysql  # type: ignore

        async with self._conn.cursor(aiomysql.DictCursor) as cur:
            # Table/database identifiers cannot be parameterized in SQL — aiomysql
            # %s placeholders are for values only, not identifiers.  Both values
            # come from DB schema introspection (never direct user input).  MySQL
            # backtick-quoting prevents injection from names containing special chars.
            await cur.execute(
                f"SELECT * FROM `{self._database}`.`{table}` LIMIT %s", (limit,)
            )
            rows = await cur.fetchall()
        return [SampleRow(table_name=table, row_data=dict(r)) for r in rows]

    async def detect_schema_changes(
        self, previous_snapshot: SchemaSnapshot
    ) -> Dict[str, Any]:
        current = await self.introspect_schema()
        return _diff_snapshots(previous_snapshot, current)

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None


# ---------------------------------------------------------------------------
# MongoDB connector
# ---------------------------------------------------------------------------

_MONGO_REF_SUFFIXES = ("_id", "Id", "Ref", "ref")


def _infer_mongo_fk_table(field_name: str) -> Optional[str]:
    for suffix in _MONGO_REF_SUFFIXES:
        if field_name.endswith(suffix) and len(field_name) > len(suffix):
            return field_name[: -len(suffix)]
    return None


def _mongo_type_name(value: Any) -> str:
    try:
        from bson import ObjectId  # type: ignore

        if isinstance(value, ObjectId):
            return "objectid"
    except ImportError:
        pass
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "document"
    return type(value).__name__


class MongoDBConnector(DatabaseConnector):
    """Async MongoDB connector using motor."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self._host: str = config["host"]
        self._port: int = config["port"]
        self._database: str = config["database"]
        self._table_filter: Optional[List[str]] = config.get("table_filter")
        self._sample_row_limit: int = config.get("sample_row_limit", 100)
        self._client = None
        self._db = None

    async def connect(self, user: str, password: str) -> None:
        import motor.motor_asyncio  # type: ignore

        uri = (
            f"mongodb://{user}:{password}@{self._host}:{self._port}/"
            f"{self._database}?serverSelectionTimeoutMS=10000"
        )
        self._client = motor.motor_asyncio.AsyncIOMotorClient(uri)
        await self._client.server_info()  # verify connectivity
        self._db = self._client[self._database]

    async def introspect_schema(self) -> SchemaSnapshot:
        if not self._db:
            raise RuntimeError("Not connected.")

        collections = await self._db.list_collection_names()
        tables: List[TableMeta] = []

        for coll_name in collections:
            if self._table_filter and coll_name not in self._table_filter:
                continue
            docs = await self._db[coll_name].find({}).to_list(length=100)

            field_types: Dict[str, str] = {}
            for doc in docs:
                for key, val in doc.items():
                    if key not in field_types:
                        field_types[key] = _mongo_type_name(val)

            columns: List[ColumnMeta] = []
            for fname, ftype in field_types.items():
                is_pk = fname == "_id"
                is_fk = not is_pk and (
                    ftype == "objectid"
                    or any(fname.endswith(s) for s in _MONGO_REF_SUFFIXES)
                )
                fk_table = _infer_mongo_fk_table(fname) if is_fk else None
                columns.append(
                    ColumnMeta(
                        name=fname,
                        data_type=ftype,
                        nullable=True,
                        is_pk=is_pk,
                        is_fk=is_fk,
                        fk_table=fk_table,
                    )
                )
            tables.append(
                TableMeta(name=coll_name, schema_name=self._database, columns=columns)
            )

        return SchemaSnapshot(
            connector_type=DatabaseConnectorType.MONGODB,
            database=self._database,
            tables=tables,
            captured_at=datetime.now(timezone.utc),
        )

    async def extract_sample_data(self, table: str, limit: int) -> List[SampleRow]:
        if not self._db:
            raise RuntimeError("Not connected.")
        docs = await self._db[table].find({}).to_list(length=limit)
        result = []
        for doc in docs:
            row = {
                k: str(v) if v.__class__.__name__ == "ObjectId" else v
                for k, v in doc.items()
            }
            result.append(SampleRow(table_name=table, row_data=row))
        return result

    async def detect_schema_changes(
        self, previous_snapshot: SchemaSnapshot
    ) -> Dict[str, Any]:
        current = await self.introspect_schema()
        return _diff_snapshots(previous_snapshot, current)

    async def close(self) -> None:
        if self._client:
            self._client.close()
            self._client = None


# ---------------------------------------------------------------------------
# Connector factory
# ---------------------------------------------------------------------------


def make_connector(
    connector_type: DatabaseConnectorType, config: Dict[str, Any]
) -> DatabaseConnector:
    if connector_type == DatabaseConnectorType.POSTGRESQL:
        return PostgreSQLConnector(config)
    if connector_type == DatabaseConnectorType.MYSQL:
        return MySQLConnector(config)
    if connector_type == DatabaseConnectorType.MONGODB:
        return MongoDBConnector(config)
    raise ValueError(f"Unsupported connector type: {connector_type}")


# ---------------------------------------------------------------------------
# Schema diff helper (shared by all three connectors)
# ---------------------------------------------------------------------------


def _diff_snapshots(prev: SchemaSnapshot, cur: SchemaSnapshot) -> Dict[str, Any]:
    prev_tables = {t.name: t for t in prev.tables}
    cur_tables = {t.name: t for t in cur.tables}
    added = [t for t in cur_tables if t not in prev_tables]
    removed = [t for t in prev_tables if t not in cur_tables]
    altered = []
    for tname in set(prev_tables) & set(cur_tables):
        prev_cols = {c.name: c for c in prev_tables[tname].columns}
        cur_cols = {c.name: c for c in cur_tables[tname].columns}
        changes: Dict[str, List[str]] = {
            "added_columns": [c for c in cur_cols if c not in prev_cols],
            "removed_columns": [c for c in prev_cols if c not in cur_cols],
            "type_changed": [
                c
                for c in set(prev_cols) & set(cur_cols)
                if prev_cols[c].data_type != cur_cols[c].data_type
            ],
        }
        if any(changes.values()):
            altered.append({"table": tname, **changes})
    return {"added_tables": added, "removed_tables": removed, "altered_tables": altered}


# ---------------------------------------------------------------------------
# KG mapping helpers
# ---------------------------------------------------------------------------


def _to_pascal_case(name: str) -> str:
    return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))


def _entity_label(table_name: str) -> str:
    """Convert table/collection name to PascalCase KG entity label."""
    return _to_pascal_case(table_name.split(".")[-1])


def _rel_type(target_label: str) -> str:
    # Neo4j relationship types cannot be parameterized in Cypher — they must
    # appear as literal identifiers in the query string.  We derive the type
    # from the FK target label (which itself comes from DB schema introspection,
    # never from direct user input) and validate it before interpolation so that
    # a malformed label can never inject arbitrary Cypher tokens.
    safe = f"REFERENCES_{target_label.upper()}"
    if not _re.match(r"^[A-Z_][A-Z0-9_]*$", safe):
        raise ValueError(
            f"Derived relationship type is not a valid Cypher identifier: {safe!r}"
        )
    return safe


# ---------------------------------------------------------------------------
# Neo4j write helper — one table/collection at a time
# ---------------------------------------------------------------------------


async def write_table_to_kg(
    graph_id: str,
    connector_id: str,
    table: TableMeta,
    sync_mode: DbSyncMode,
    sample_rows: List[SampleRow],
    driver=None,  # Optional: pass worker async driver; defaults to fastapi neo4j_client
) -> int:
    """Write entities and relationships for one table into Neo4j.

    Returns count of entity nodes upserted.
    Uses parameterized Cypher only — no string interpolation of user values.
    Every query includes graph_id for multi-tenancy.
    """
    label = _entity_label(table.name)
    pk_cols = [c for c in table.columns if c.is_pk]
    fk_cols = [c for c in table.columns if c.is_fk and not c.is_pk]
    scalar_cols = [c for c in table.columns if not c.is_pk and not c.is_fk]

    async def run_write(cypher: str, params: Dict[str, Any]) -> List[Any]:
        if driver:
            records, _, _ = await driver.execute_query(cypher, params)
            return records
        return await neo4j_client.execute_write_query(cypher, params)

    if sync_mode == DbSyncMode.SCHEMA_ONLY or not sample_rows:
        # Schema-only: create a placeholder entity type node with no row data
        await run_write(
            """
            MERGE (e:__Entity__ {
                graph_id:             $graph_id,
                type:                 $type,
                name:                 $name,
                source_connector_id:  $connector_id,
                source_table:         $table_name,
                source_ingest_mode:   $ingest_mode
            })
            ON CREATE SET e.created_at = datetime().epochMillis,
                          e.updated_at = datetime().epochMillis
            ON MATCH  SET e.updated_at = datetime().epochMillis
            """,
            {
                "graph_id": graph_id,
                "type": label,
                "name": f"[Schema] {label}",
                "connector_id": connector_id,
                "table_name": table.name,
                "ingest_mode": sync_mode.value,
            },
        )
        return 0

    entity_count = 0
    for row in sample_rows:
        pk_parts = [str(row.row_data.get(c.name, "")) for c in pk_cols]
        entity_id = "_".join(pk_parts) if pk_parts else str(uuid.uuid4())

        props = {
            c.name: row.row_data[c.name]
            for c in scalar_cols
            if c.name in row.row_data and row.row_data[c.name] is not None
        }

        await run_write(
            """
            MERGE (e:__Entity__ {
                graph_id:            $graph_id,
                source_connector_id: $connector_id,
                source_table:        $table_name,
                source_pk_value:     $pk_value
            })
            ON CREATE SET
                e.name             = $name,
                e.type             = $label,
                e.entity_id        = $pk_value,
                e.source_ingest_mode = $ingest_mode,
                e.created_at       = datetime().epochMillis,
                e.updated_at       = datetime().epochMillis
            ON MATCH SET
                e.updated_at = datetime().epochMillis
            SET e += $props
            """,
            {
                "graph_id": graph_id,
                "connector_id": connector_id,
                "table_name": table.name,
                "pk_value": entity_id,
                "name": entity_id,
                "label": label,
                "ingest_mode": sync_mode.value,
                "props": props,
            },
        )
        entity_count += 1

        # Create FK → relationship edges
        for fk_col in fk_cols:
            fk_val = row.row_data.get(fk_col.name)
            if not fk_val or not fk_col.fk_table:
                continue
            target_label = _entity_label(fk_col.fk_table)
            await run_write(
                f"""
                MATCH (src:__Entity__ {{
                    graph_id:            $graph_id,
                    source_connector_id: $connector_id,
                    source_table:        $src_table,
                    source_pk_value:     $src_pk
                }})
                MATCH (tgt:__Entity__ {{
                    graph_id:            $graph_id,
                    source_connector_id: $connector_id,
                    source_table:        $tgt_table
                }})
                MERGE (src)-[r:{_rel_type(target_label)}]->(tgt)
                ON CREATE SET
                    r.fk_column  = $fk_col,
                    r.source_table = $src_table,
                    r.graph_id   = $graph_id,
                    r.created_at = datetime().epochMillis
                """,
                {
                    "graph_id": graph_id,
                    "connector_id": connector_id,
                    "src_table": table.name,
                    "src_pk": entity_id,
                    "tgt_table": fk_col.fk_table,
                    "fk_col": fk_col.name,
                },
            )

    return entity_count


# ---------------------------------------------------------------------------
# DatabaseConnectorService — Neo4j connector node lifecycle
# ---------------------------------------------------------------------------


class DatabaseConnectorService:
    """Manages database connector nodes in Neo4j.

    All queries are graph_id-scoped for multi-tenancy.
    Connector credentials are NEVER stored — they live in the credential broker.
    """

    async def ensure_constraints(self) -> None:
        """Create Connector node constraints and indexes (idempotent, IF NOT EXISTS)."""
        for stmt in [
            "CREATE CONSTRAINT connector_id_unique IF NOT EXISTS FOR (c:Connector) REQUIRE c.connector_id IS UNIQUE",
            "CREATE INDEX connector_graph_id IF NOT EXISTS FOR (c:Connector) ON (c.graph_id)",
            "CREATE INDEX connector_sync_error_connector_id IF NOT EXISTS FOR (e:ConnectorSyncError) ON (e.connector_id)",
        ]:
            await neo4j_client.execute_write_query(stmt, {})

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def register(
        self,
        graph_id: str,
        user_id: str,
        display_name: str,
        connector_type: DatabaseConnectorType,
        host: str,
        port: int,
        database: str,
        sync_mode: DbSyncMode,
        schema_filter: Optional[str] = None,
        table_filter: Optional[List[str]] = None,
        sample_row_limit: int = 100,
    ) -> Dict[str, Any]:
        """Register a new database connector node in Neo4j. Returns the connector dict."""
        validate_host_ssrf(host)
        sample_row_limit = min(sample_row_limit, 1000)
        connector_id = str(uuid.uuid4())
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

        result = await neo4j_client.execute_write_query(
            """
            CREATE (c:Connector {
                connector_id:     $connector_id,
                graph_id:         $graph_id,
                user_id:          $user_id,
                connector_type:   $connector_type,
                display_name:     $display_name,
                host:             $host,
                port:             $port,
                database:         $database,
                schema_filter:    $schema_filter,
                table_filter:     $table_filter,
                sample_row_limit: $sample_row_limit,
                sync_mode:        $sync_mode,
                status:           'active',
                created_at:       $now_ms,
                updated_at:       $now_ms
            })
            RETURN c
            """,
            {
                "connector_id": connector_id,
                "graph_id": graph_id,
                "user_id": user_id,
                "connector_type": connector_type.value,
                "display_name": display_name,
                "host": host,
                "port": port,
                "database": database,
                "schema_filter": schema_filter,
                "table_filter": json.dumps(table_filter) if table_filter else None,
                "sample_row_limit": sample_row_limit,
                "sync_mode": sync_mode.value,
                "now_ms": now_ms,
            },
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create connector.",
            )
        return dict(result[0]["c"])

    async def list_connectors(self, graph_id: str) -> List[Dict[str, Any]]:
        result = await neo4j_client.execute_query(
            """
            MATCH (c:Connector {graph_id: $graph_id})
            WHERE c.status <> 'deleted'
            RETURN c
            ORDER BY c.created_at DESC
            """,
            {"graph_id": graph_id},
        )
        return [dict(r["c"]) for r in result]

    async def get_connector(self, graph_id: str, connector_id: str) -> Dict[str, Any]:
        """Get connector with last 5 sync errors."""
        result = await neo4j_client.execute_query(
            """
            MATCH (c:Connector {graph_id: $graph_id, connector_id: $connector_id})
            WHERE c.status <> 'deleted'
            OPTIONAL MATCH (c)-[:HAD_SYNC_ERROR]->(e:ConnectorSyncError)
            WITH c, e ORDER BY e.occurred_at DESC
            RETURN c, collect(e)[..5] AS recent_errors
            """,
            {"graph_id": graph_id, "connector_id": connector_id},
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found."
            )
        data = dict(result[0]["c"])
        data["recent_errors"] = [dict(e) for e in (result[0]["recent_errors"] or [])]
        return data

    async def delete_connector(self, graph_id: str, connector_id: str) -> None:
        """Soft-delete connector and remove INGESTS_INTO relationship."""
        result = await neo4j_client.execute_write_query(
            """
            MATCH (c:Connector {graph_id: $graph_id, connector_id: $connector_id})
            WHERE c.status <> 'deleted'
            SET c.status = 'deleted', c.updated_at = datetime().epochMillis
            WITH c
            OPTIONAL MATCH (c)-[r:INGESTS_INTO]->()
            DELETE r
            RETURN c.connector_id AS id
            """,
            {"graph_id": graph_id, "connector_id": connector_id},
        )
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Connector not found."
            )

    # ------------------------------------------------------------------
    # Sync metadata updates (used by Celery worker via direct driver call)
    # ------------------------------------------------------------------

    async def update_sync_status(
        self,
        graph_id: str,
        connector_id: str,
        sync_status: str,
        row_count: Optional[int] = None,
        error_msg: Optional[str] = None,
    ) -> None:
        await neo4j_client.execute_write_query(
            """
            MATCH (c:Connector {graph_id: $graph_id, connector_id: $connector_id})
            SET c.last_sync_at      = datetime().epochMillis,
                c.last_sync_status  = $sync_status,
                c.last_sync_error   = $error_msg,
                c.last_sync_row_count = $row_count,
                c.updated_at        = datetime().epochMillis
            """,
            {
                "graph_id": graph_id,
                "connector_id": connector_id,
                "sync_status": sync_status,
                "error_msg": error_msg,
                "row_count": row_count,
            },
        )

    async def store_schema_snapshot(
        self, graph_id: str, connector_id: str, snapshot_json: str
    ) -> None:
        await neo4j_client.execute_write_query(
            """
            MATCH (c:Connector {graph_id: $graph_id, connector_id: $connector_id})
            SET c.last_schema_snapshot = $snapshot
            """,
            {
                "graph_id": graph_id,
                "connector_id": connector_id,
                "snapshot": snapshot_json,
            },
        )

    async def record_sync_error(
        self,
        graph_id: str,
        connector_id: str,
        error_type: str,
        error_message: str,
        tables_failed: Optional[List[str]] = None,
    ) -> None:
        """Record a sync error node and prune to keep only the latest 10."""
        error_id = str(uuid.uuid4())
        await neo4j_client.execute_write_query(
            """
            MATCH (c:Connector {graph_id: $graph_id, connector_id: $connector_id})
            CREATE (e:ConnectorSyncError {
                error_id:      $error_id,
                connector_id:  $connector_id,
                graph_id:      $graph_id,
                occurred_at:   datetime().epochMillis,
                error_type:    $error_type,
                error_message: $error_message,
                tables_failed: $tables_failed
            })
            CREATE (c)-[:HAD_SYNC_ERROR {occurred_at: datetime().epochMillis}]->(e)
            WITH c
            MATCH (c)-[:HAD_SYNC_ERROR]->(old:ConnectorSyncError)
            WITH c, old ORDER BY old.occurred_at ASC
            WITH c, collect(old) AS all_errors
            WHERE size(all_errors) > 10
            UNWIND all_errors[..size(all_errors) - 10] AS stale
            DETACH DELETE stale
            """,
            {
                "graph_id": graph_id,
                "connector_id": connector_id,
                "error_id": error_id,
                "error_type": error_type,
                "error_message": error_message,
                "tables_failed": json.dumps(tables_failed) if tables_failed else None,
            },
        )


database_connector_service = DatabaseConnectorService()
