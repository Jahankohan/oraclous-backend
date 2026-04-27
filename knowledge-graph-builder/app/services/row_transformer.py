"""Row Transformer Service — convert relational rows to Neo4j graph nodes/edges (TASK-020).

Consumes GraphMappingRules produced by SchemaMapper (TASK-019) and writes:
- Entity nodes  → __Entity__ nodes via MERGE (entity_table and self_ref_table)
- Relationships → relationship edges via MERGE (junction_table and self_ref_table)

Architecture invariants:
- graph_id on EVERY written node and relationship — no cross-tenant writes
- WorkerNeo4jManager (sync NullPool driver) — NEVER async_driver in Celery tasks
- Parameterized Cypher only — table names come from SchemaSnapshot allowlist
- UNWIND batch writes at 500 rows — no per-row round trips
- Entity tables written BEFORE junction tables (FK targets must exist first)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.services.background_jobs import WorkerNeo4jManager
    from app.services.schema_mapper import GraphMappingRules, TableMapping

logger = get_logger(__name__)

_BATCH_SIZE = 500


class RowTransformer:
    """Write entity nodes and relationship edges to Neo4j from relational rows.

    Parameters
    ----------
    neo4j_manager:
        A connected WorkerNeo4jManager instance (sync NullPool driver).
        Must be used inside a context manager so the driver is available.
    """

    def __init__(self, neo4j_manager: "WorkerNeo4jManager") -> None:
        self._manager = neo4j_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform_table(
        self,
        table_mapping: "TableMapping",
        rows: list[dict[str, Any]],
        graph_id: str,
        connector_id: str,
    ) -> int:
        """Write entity nodes for all rows in *table_mapping*.

        Only called for entity_table and self_ref_table kinds.
        Junction tables are handled by transform_junctions().

        Returns count of rows written.
        """
        if not rows:
            return 0

        driver = self._manager.get_sync_driver()
        total = 0

        for batch in _chunks(rows, _BATCH_SIZE):
            batch_params = [
                _build_entity_row_param(row, table_mapping, graph_id, connector_id)
                for row in batch
            ]
            # Filter out rows where we couldn't determine the PK
            batch_params = [p for p in batch_params if p is not None]
            if not batch_params:
                continue

            with driver.session() as session:
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (e:__Entity__ {id: row.id, graph_id: $graph_id})
                    SET e += row.properties,
                        e.source_table   = $table_name,
                        e.ingestion_time = datetime(),
                        e.label          = $neo4j_label
                    """,
                    {
                        "batch": batch_params,
                        "graph_id": graph_id,
                        "table_name": table_mapping.table_name,
                        "neo4j_label": table_mapping.neo4j_label,
                    },
                )
            total += len(batch_params)

        logger.info(
            "row_transformer: wrote %d entity rows for table %r (graph=%r)",
            total,
            table_mapping.table_name,
            graph_id,
        )
        return total

    def transform_junctions(
        self,
        junction_mappings: list["TableMapping"],
        rows_by_table: dict[str, list[dict[str, Any]]],
        graph_id: str,
        connector_id: str,
    ) -> int:
        """Write relationship edges for junction tables and self-ref FK tables.

        Entity nodes for the referenced tables must already exist in Neo4j
        (call transform_table() for all entity tables first).

        Returns total count of edges written.
        """
        total = 0
        driver = self._manager.get_sync_driver()

        for table_mapping in junction_mappings:
            rows = rows_by_table.get(table_mapping.table_name, [])
            if not rows:
                continue

            for rel_mapping in table_mapping.relationships:
                rel_type = rel_mapping.rel_type
                # Validate rel_type is a safe Cypher identifier (letters, digits, underscores only)
                if not _safe_rel_type(rel_type):
                    logger.warning(
                        "row_transformer: unsafe rel_type %r for table %r — skipping",
                        rel_type,
                        table_mapping.table_name,
                    )
                    continue

                for batch in _chunks(rows, _BATCH_SIZE):
                    if table_mapping.kind == "junction_table":
                        batch_params = [
                            _build_junction_row_param(
                                row, table_mapping, rel_mapping, graph_id, connector_id
                            )
                            for row in batch
                        ]
                    else:
                        # self_ref_table — source and target are the same table
                        batch_params = [
                            _build_self_ref_row_param(
                                row, table_mapping, rel_mapping, graph_id, connector_id
                            )
                            for row in batch
                        ]

                    batch_params = [p for p in batch_params if p is not None]
                    if not batch_params:
                        continue

                    cypher = f"""
                    UNWIND $batch AS row
                    MATCH (a:__Entity__ {{id: row.from_id, graph_id: $graph_id}})
                    MATCH (b:__Entity__ {{id: row.to_id, graph_id: $graph_id}})
                    MERGE (a)-[r:{rel_type} {{graph_id: $graph_id}}]->(b)
                    SET r.ingestion_time = datetime(),
                        r.source_table   = $junction_table
                    """
                    with driver.session() as session:
                        session.run(
                            cypher,
                            {
                                "batch": batch_params,
                                "graph_id": graph_id,
                                "junction_table": table_mapping.table_name,
                            },
                        )
                    total += len(batch_params)

        logger.info(
            "row_transformer: wrote %d relationship edges (graph=%r)", total, graph_id
        )
        return total


# ---------------------------------------------------------------------------
# Row parameter builders
# ---------------------------------------------------------------------------


def _entity_id(connector_id: str, table_name: str, pk_value: Any) -> str:
    """Build the deterministic entity id: {connector_id}:{table_name}:{pk_value}."""
    return f"{connector_id}:{table_name}:{pk_value}"


def _build_entity_row_param(
    row: dict[str, Any],
    table_mapping: "TableMapping",
    graph_id: str,
    connector_id: str,
) -> dict[str, Any] | None:
    """Build a single UNWIND parameter dict for an entity row.

    Returns None if the PK value is missing/None (cannot create a unique node).
    """
    pk_col = table_mapping.pk_column
    pk_value = row.get(pk_col)
    if pk_value is None:
        return None

    entity_id = _entity_id(connector_id, table_mapping.table_name, pk_value)

    # Property columns: non-PK, non-FK columns mapped to neo4j_property names
    props: dict[str, Any] = {}
    for col_mapping in table_mapping.property_columns:
        raw_value = row.get(col_mapping.column_name)
        if raw_value is not None:
            props[col_mapping.neo4j_property] = _coerce_value(raw_value)

    return {"id": entity_id, "properties": props}


def _build_junction_row_param(
    row: dict[str, Any],
    table_mapping: "TableMapping",
    rel_mapping: "RelationshipMapping",  # type: ignore[name-defined]
    graph_id: str,
    connector_id: str,
) -> dict[str, Any] | None:
    """Build a single UNWIND parameter dict for a junction table row.

    from_id is built from the FK column pointing to from_table.
    to_id   is built from the FK column pointing to to_table.
    Returns None if either FK value is missing.
    """
    from_fk_value = row.get(rel_mapping.from_fk_column)
    to_fk_value = row.get(rel_mapping.to_fk_column)
    if from_fk_value is None or to_fk_value is None:
        return None

    from_id = _entity_id(connector_id, rel_mapping.from_table, from_fk_value)
    to_id = _entity_id(connector_id, rel_mapping.to_table, to_fk_value)

    return {"from_id": from_id, "to_id": to_id}


def _build_self_ref_row_param(
    row: dict[str, Any],
    table_mapping: "TableMapping",
    rel_mapping: "RelationshipMapping",  # type: ignore[name-defined]
    graph_id: str,
    connector_id: str,
) -> dict[str, Any] | None:
    """Build a single UNWIND parameter dict for a self-referential FK row.

    Source entity = this row's PK (table_mapping.table_name).
    Target entity = the FK value (also in table_mapping.table_name).
    Returns None if PK or FK value is missing.
    """
    pk_value = row.get(table_mapping.pk_column) if table_mapping.pk_column else None
    fk_value = row.get(rel_mapping.from_fk_column)
    if pk_value is None or fk_value is None:
        return None

    from_id = _entity_id(connector_id, table_mapping.table_name, pk_value)
    to_id = _entity_id(connector_id, table_mapping.table_name, fk_value)

    return {"from_id": from_id, "to_id": to_id}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _chunks(lst: list, size: int):
    """Yield successive fixed-size chunks from lst."""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def _safe_rel_type(rel_type: str) -> bool:
    """Return True if rel_type is a safe Cypher relationship type identifier."""
    import re

    return bool(re.match(r"^[A-Z_][A-Z0-9_]*$", rel_type))


def _coerce_value(value: Any) -> Any:
    """Coerce a row value to a Neo4j-compatible type.

    Neo4j driver supports: bool, int, float, str, list, dict, None.
    Dates and datetimes are serialized to ISO strings.
    """
    import datetime

    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if isinstance(value, (datetime.datetime, datetime.date)):
        return value.isoformat()
    if isinstance(value, (list, tuple)):
        return [_coerce_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _coerce_value(v) for k, v in value.items()}
    # Fallback: stringify
    return str(value)
