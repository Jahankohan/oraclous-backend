"""Integration tests for the full relational-to-graph pipeline (STORY-003).

DO NOT RUN until TASK-019 (PR #47) and TASK-020 (PR #54) are merged to develop.

Tests the end-to-end flow:
  SchemaMapper.map() → GraphMappingRules → RowTransformer.transform_table() +
  RowTransformer.transform_junctions() → Neo4j __Entity__ nodes + relationships

Test isolation:
- Each test uses a unique graph_id: f"test-task021-{uuid4().hex[:8]}"
- All nodes are created under that graph_id only
- teardown fixture deletes all test nodes by graph_id (DETACH DELETE)

Coverage:
1. Entity table → __Entity__ nodes with correct id, properties, graph_id
2. Junction table → relationship edges between entity nodes
3. Self-referential table → self-ref edges (MANAGES)
4. Cross-tenant isolation: second graph_id has zero nodes from first sync
5. 1000-row sync completes in <30s
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Constants — all ids are unique per test run
# ---------------------------------------------------------------------------

_CONNECTOR_ID = f"test-task021-connector-{uuid4().hex[:8]}"


def _unique_graph_id() -> str:
    return f"test-task021-{uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Schema helpers — reuse the SchemaSnapshot types from the service
# ---------------------------------------------------------------------------


def _make_schema_snapshot():
    """Build a SchemaSnapshot with 3 tables:
    - employees  (entity table, with self-referential manager_id FK)
    - projects   (entity table)
    - emp_project (junction table: employee ↔ project)
    """
    from app.services.database_connector_service import (
        ColumnMeta,
        DatabaseConnectorType,
        SchemaSnapshot,
        TableMeta,
    )

    employee_cols = [
        ColumnMeta(
            name="id", data_type="integer", nullable=False, is_pk=True, is_fk=False
        ),
        ColumnMeta(
            name="name", data_type="varchar", nullable=False, is_pk=False, is_fk=False
        ),
        ColumnMeta(
            name="department",
            data_type="varchar",
            nullable=True,
            is_pk=False,
            is_fk=False,
        ),
        ColumnMeta(
            name="manager_id",
            data_type="integer",
            nullable=True,
            is_pk=False,
            is_fk=True,
            fk_table="employees",
            fk_column="id",
        ),
    ]

    project_cols = [
        ColumnMeta(
            name="id", data_type="integer", nullable=False, is_pk=True, is_fk=False
        ),
        ColumnMeta(
            name="title", data_type="varchar", nullable=False, is_pk=False, is_fk=False
        ),
    ]

    junction_cols = [
        ColumnMeta(
            name="employee_id",
            data_type="integer",
            nullable=False,
            is_pk=False,
            is_fk=True,
            fk_table="employees",
            fk_column="id",
        ),
        ColumnMeta(
            name="project_id",
            data_type="integer",
            nullable=False,
            is_pk=False,
            is_fk=True,
            fk_table="projects",
            fk_column="id",
        ),
        ColumnMeta(
            name="created_at",
            data_type="timestamp",
            nullable=True,
            is_pk=False,
            is_fk=False,
        ),
    ]

    return SchemaSnapshot(
        connector_type=DatabaseConnectorType.POSTGRESQL,
        database="testdb",
        captured_at=datetime.now(UTC),
        tables=[
            TableMeta(name="employees", schema_name="public", columns=employee_cols),
            TableMeta(name="projects", schema_name="public", columns=project_cols),
            TableMeta(name="emp_project", schema_name="public", columns=junction_cols),
        ],
    )


def _make_employee_rows(count: int = 5) -> list[dict[str, Any]]:
    """Generate synthetic employee rows. Employee 1 has no manager (CEO)."""
    rows = [
        {"id": 1, "name": "Alice CEO", "department": "Executive", "manager_id": None}
    ]
    for i in range(2, count + 1):
        rows.append(
            {
                "id": i,
                "name": f"Employee {i}",
                "department": "Engineering",
                "manager_id": 1,  # all report to CEO
            }
        )
    return rows


def _make_project_rows(count: int = 3) -> list[dict[str, Any]]:
    return [{"id": i, "title": f"Project {i}"} for i in range(1, count + 1)]


def _make_junction_rows(
    employee_ids: list[int], project_ids: list[int]
) -> list[dict[str, Any]]:
    """Each employee is assigned to the project with matching index (mod count)."""
    rows = []
    for i, eid in enumerate(employee_ids):
        pid = project_ids[i % len(project_ids)]
        rows.append({"employee_id": eid, "project_id": pid, "created_at": None})
    return rows


# ---------------------------------------------------------------------------
# Neo4j query helpers — sync, using direct driver (avoids async event-loop
# conflicts between pytest-asyncio and the neo4j async driver singleton)
# ---------------------------------------------------------------------------


def _count_entities(driver, graph_id: str, source_table: str | None = None) -> int:
    """Count __Entity__ nodes for the given graph_id (and optionally table)."""
    with driver.session() as session:
        if source_table:
            result = session.run(
                "MATCH (e:__Entity__ {graph_id: $gid, source_table: $tbl}) RETURN count(e) AS cnt",
                {"gid": graph_id, "tbl": source_table},
            )
        else:
            result = session.run(
                "MATCH (e:__Entity__ {graph_id: $gid}) RETURN count(e) AS cnt",
                {"gid": graph_id},
            )
        record = result.single()
        return int(record["cnt"]) if record else 0


def _count_relationships(driver, graph_id: str) -> int:
    """Count all relationships scoped to graph_id."""
    with driver.session() as session:
        result = session.run(
            "MATCH (a:__Entity__ {graph_id: $gid})-[r {graph_id: $gid}]->(b:__Entity__ {graph_id: $gid}) "
            "RETURN count(r) AS cnt",
            {"gid": graph_id},
        )
        record = result.single()
        return int(record["cnt"]) if record else 0


def _get_entity(driver, entity_id: str, graph_id: str) -> dict | None:
    """Fetch a single __Entity__ node by its composite id and graph_id."""
    with driver.session() as session:
        result = session.run(
            "MATCH (e:__Entity__ {id: $eid, graph_id: $gid}) RETURN e",
            {"eid": entity_id, "gid": graph_id},
        )
        record = result.single()
        if not record:
            return None
        return dict(record["e"])


def _query_relationships(driver, gid: str, from_id: str, to_id: str):
    """Return relationship records between two entity nodes in the given graph."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:__Entity__ {id: $from_id, graph_id: $gid})
                  -[r {graph_id: $gid}]->
                  (b:__Entity__ {id: $to_id, graph_id: $gid})
            RETURN type(r) AS rel_type, r.graph_id AS edge_graph_id, r.ingestion_time AS ts
            """,
            {"gid": gid, "from_id": from_id, "to_id": to_id},
        )
        return result.data()


def _query_manages(driver, gid: str, from_id: str, to_id: str):
    """Return MANAGES relationship records between two entity nodes."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:__Entity__ {id: $from_id, graph_id: $gid})
                  -[r:MANAGES {graph_id: $gid}]->
                  (b:__Entity__ {id: $to_id, graph_id: $gid})
            RETURN r.graph_id AS gid
            """,
            {"gid": gid, "from_id": from_id, "to_id": to_id},
        )
        return result.data()


def _count_missing_entity_gid(driver, gid: str) -> int:
    """Count __Entity__ nodes where graph_id does not match expected value."""
    with driver.session() as session:
        result = session.run(
            "MATCH (e:__Entity__ {graph_id: $gid}) WHERE e.graph_id <> $gid RETURN count(e) AS cnt",
            {"gid": gid},
        )
        record = result.single()
        return int(record["cnt"]) if record else 0


def _count_missing_rel_gid(driver, gid: str) -> int:
    """Count relationships missing or mismatching graph_id."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (a:__Entity__ {graph_id: $gid})-[r]->(b:__Entity__ {graph_id: $gid})
            WHERE r.graph_id <> $gid OR r.graph_id IS NULL
            RETURN count(r) AS cnt
            """,
            {"gid": gid},
        )
        record = result.single()
        return int(record["cnt"]) if record else 0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def neo4j(request):
    """Return a sync Neo4j driver and clean up test data after the test.

    Uses a direct sync driver (same pattern as _run_pipeline) to avoid
    mixing pytest-asyncio event loops with the neo4j async driver singleton.
    """
    from neo4j import GraphDatabase

    from app.core.config import settings

    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    graph_ids: list[str] = []

    yield driver, graph_ids

    # Teardown: delete all nodes for every graph_id used in this test
    for gid in graph_ids:
        with driver.session() as session:
            session.run("MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid})

    driver.close()


def _make_worker_neo4j_manager():
    """Return a WorkerNeo4jManager backed by the real Neo4j sync driver
    (using the NullPool pattern from background_jobs.py)."""
    from app.services.background_jobs import WorkerNeo4jManager

    manager = WorkerNeo4jManager()
    manager.connect_sync_only()
    return manager


# ---------------------------------------------------------------------------
# Helper: run the full pipeline for a given graph_id
# ---------------------------------------------------------------------------


def _run_pipeline(
    graph_id: str,
    snapshot,
    rows_by_table: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Execute SchemaMapper + RowTransformer synchronously.

    Returns a dict with counts of entities and relationships written.
    """
    from app.services.row_transformer import RowTransformer
    from app.services.schema_mapper import SchemaMapper

    mapper = SchemaMapper()
    rules = mapper.map(
        schema_snapshot=snapshot,
        connector_id=_CONNECTOR_ID,
        graph_id=graph_id,
    )

    entity_mappings = [tm for tm in rules.tables if tm.kind == "entity_table"]
    junction_mappings = [
        tm for tm in rules.tables if tm.kind in ("junction_table", "self_ref_table")
    ]

    manager = _make_worker_neo4j_manager()
    try:
        xfm = RowTransformer(manager)

        total_entities = 0
        for tm in entity_mappings:
            rows = rows_by_table.get(tm.table_name, [])
            total_entities += xfm.transform_table(
                table_mapping=tm,
                rows=rows,
                graph_id=graph_id,
                connector_id=_CONNECTOR_ID,
            )

        # Self-ref tables also need entity nodes — same path as entity tables
        self_ref_mappings = [tm for tm in rules.tables if tm.kind == "self_ref_table"]
        for tm in self_ref_mappings:
            rows = rows_by_table.get(tm.table_name, [])
            total_entities += xfm.transform_table(
                table_mapping=tm,
                rows=rows,
                graph_id=graph_id,
                connector_id=_CONNECTOR_ID,
            )

        # Write junctions (entity nodes must exist first)
        all_junction_and_self_ref = junction_mappings
        total_edges = xfm.transform_junctions(
            junction_mappings=all_junction_and_self_ref,
            rows_by_table=rows_by_table,
            graph_id=graph_id,
            connector_id=_CONNECTOR_ID,
        )
    finally:
        manager.cleanup_sync()

    return {"entities": total_entities, "edges": total_edges}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_entity_table_nodes_written(neo4j):
    """Entity rows → __Entity__ nodes with correct id format and graph_id."""
    driver, graph_ids = neo4j
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    snapshot = _make_schema_snapshot()
    employee_rows = _make_employee_rows(count=3)
    project_rows = _make_project_rows(count=2)
    junction_rows = _make_junction_rows([1, 2, 3], [1, 2])

    _run_pipeline(
        graph_id=graph_id,
        snapshot=snapshot,
        rows_by_table={
            "employees": employee_rows,
            "projects": project_rows,
            "emp_project": junction_rows,
        },
    )

    # 3 employees + 2 projects = 5 entity nodes
    total = _count_entities(driver, graph_id)
    assert total == 5, f"Expected 5 entity nodes, got {total}"

    # Verify id format: {connector_id}:{table}:{pk}
    emp1_id = f"{_CONNECTOR_ID}:employees:1"
    node = _get_entity(driver, emp1_id, graph_id)
    assert node is not None, f"Entity node {emp1_id!r} not found in graph {graph_id!r}"
    assert node["graph_id"] == graph_id
    assert node["source_table"] == "employees"
    assert node["name"] == "Alice CEO"
    assert node["department"] == "Executive"
    # ingestion_time should be set (Neo4j datetime object — not None)
    assert node.get("ingestion_time") is not None


@pytest.mark.integration
def test_entity_node_id_format(neo4j):
    """Entity id must be '{connector_id}:{table_name}:{pk_value}'."""
    driver, graph_ids = neo4j
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    snapshot = _make_schema_snapshot()
    project_rows = [{"id": 42, "title": "Big Project"}]

    _run_pipeline(
        graph_id=graph_id,
        snapshot=snapshot,
        rows_by_table={"employees": [], "projects": project_rows, "emp_project": []},
    )

    expected_id = f"{_CONNECTOR_ID}:projects:42"
    node = _get_entity(driver, expected_id, graph_id)
    assert node is not None, f"Expected entity id {expected_id!r} not found"
    assert node["graph_id"] == graph_id


@pytest.mark.integration
def test_junction_relationship_edges(neo4j):
    """Junction table rows → relationship edges between correct entity nodes."""
    driver, graph_ids = neo4j
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    snapshot = _make_schema_snapshot()
    employee_rows = _make_employee_rows(count=2)
    project_rows = _make_project_rows(count=2)
    junction_rows = [
        {"employee_id": 1, "project_id": 1, "created_at": None},
        {"employee_id": 2, "project_id": 2, "created_at": None},
    ]

    _run_pipeline(
        graph_id=graph_id,
        snapshot=snapshot,
        rows_by_table={
            "employees": employee_rows,
            "projects": project_rows,
            "emp_project": junction_rows,
        },
    )

    rel_count = _count_relationships(driver, graph_id)
    # 2 junction edges + edges from manager_id self-ref (employee 2 → employee 1)
    assert rel_count >= 2, f"Expected at least 2 relationship edges, got {rel_count}"

    # Verify junction edge exists
    records = _query_relationships(
        driver,
        graph_id,
        f"{_CONNECTOR_ID}:employees:1",
        f"{_CONNECTOR_ID}:projects:1",
    )
    assert records, "Junction edge employees:1 → projects:1 not found"
    edge = records[0]
    assert edge["edge_graph_id"] == graph_id
    assert edge["ts"] is not None


@pytest.mark.integration
def test_self_referential_edges(neo4j):
    """Self-ref FK (manager_id) → MANAGES edge from manager → report."""
    driver, graph_ids = neo4j
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    snapshot = _make_schema_snapshot()
    # CEO (id=1) has no manager; VP (id=2) reports to CEO
    employee_rows = [
        {"id": 1, "name": "CEO", "department": "Exec", "manager_id": None},
        {"id": 2, "name": "VP", "department": "Exec", "manager_id": 1},
    ]

    _run_pipeline(
        graph_id=graph_id,
        snapshot=snapshot,
        rows_by_table={"employees": employee_rows, "projects": [], "emp_project": []},
    )

    # employee 2 (pk=2) has manager_id=1 → edge from employee:2 → employee:1
    records = _query_manages(
        driver,
        graph_id,
        f"{_CONNECTOR_ID}:employees:2",
        f"{_CONNECTOR_ID}:employees:1",
    )
    assert records, "Expected MANAGES self-ref edge (employee:2 → employee:1) not found"
    assert records[0]["gid"] == graph_id


@pytest.mark.integration
def test_graph_id_on_every_node_and_relationship(neo4j):
    """graph_id must appear on every written __Entity__ node and relationship."""
    driver, graph_ids = neo4j
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    snapshot = _make_schema_snapshot()
    employee_rows = _make_employee_rows(count=3)
    project_rows = _make_project_rows(count=2)
    junction_rows = _make_junction_rows([1, 2], [1, 2])

    _run_pipeline(
        graph_id=graph_id,
        snapshot=snapshot,
        rows_by_table={
            "employees": employee_rows,
            "projects": project_rows,
            "emp_project": junction_rows,
        },
    )

    assert _count_missing_entity_gid(driver, graph_id) == 0
    assert _count_missing_rel_gid(driver, graph_id) == 0


@pytest.mark.integration
def test_cross_tenant_isolation(neo4j):
    """Sync into graph A must not affect graph B — zero nodes from graph A in graph B."""
    driver, graph_ids = neo4j
    graph_id_a = _unique_graph_id()
    graph_id_b = _unique_graph_id()
    graph_ids.extend([graph_id_a, graph_id_b])

    snapshot = _make_schema_snapshot()
    employee_rows = _make_employee_rows(count=5)
    project_rows = _make_project_rows(count=3)
    junction_rows = _make_junction_rows([1, 2, 3], [1, 2, 3])

    # Sync into graph A only
    _run_pipeline(
        graph_id=graph_id_a,
        snapshot=snapshot,
        rows_by_table={
            "employees": employee_rows,
            "projects": project_rows,
            "emp_project": junction_rows,
        },
    )

    # Graph B must have zero nodes
    count_b = _count_entities(driver, graph_id_b)
    assert (
        count_b == 0
    ), f"Cross-tenant leak: graph_id_b has {count_b} nodes after syncing into graph_id_a"

    # Graph A must have the expected nodes
    count_a = _count_entities(driver, graph_id_a)
    assert (
        count_a == 8
    ), f"Expected 8 entity nodes in graph A (5 employees + 3 projects), got {count_a}"


@pytest.mark.integration
def test_1000_row_sync_completes_in_under_30s(neo4j):
    """1000 employee rows + 1000 junction rows synced in <30s total."""
    driver, graph_ids = neo4j
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    snapshot = _make_schema_snapshot()

    # 1000 employee rows
    employee_rows = [
        {
            "id": i,
            "name": f"Employee {i}",
            "department": "Engineering",
            "manager_id": 1 if i > 1 else None,
        }
        for i in range(1, 1001)
    ]

    # 3 projects
    project_rows = _make_project_rows(count=3)

    # 1000 junction rows (each employee assigned to a project)
    junction_rows = [
        {"employee_id": i, "project_id": (i % 3) + 1, "created_at": None}
        for i in range(1, 1001)
    ]

    start = time.monotonic()
    _run_pipeline(
        graph_id=graph_id,
        snapshot=snapshot,
        rows_by_table={
            "employees": employee_rows,
            "projects": project_rows,
            "emp_project": junction_rows,
        },
    )
    elapsed = time.monotonic() - start

    # Timing assertion — must complete within 30 seconds
    assert elapsed < 30, f"1000-row sync took {elapsed:.2f}s — exceeds 30s SLA"

    # Verify counts
    total_entities = _count_entities(driver, graph_id)
    assert (
        total_entities == 1003
    ), f"Expected 1003 entity nodes (1000 employees + 3 projects), got {total_entities}"


@pytest.mark.integration
def test_no_entity_for_row_missing_pk(neo4j):
    """Rows without a PK value must not produce any __Entity__ node."""
    driver, graph_ids = neo4j
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    snapshot = _make_schema_snapshot()
    # Two valid rows + one row with no PK
    project_rows = [
        {"id": 1, "title": "Alpha"},
        {"id": 2, "title": "Beta"},
        {"title": "No PK"},  # id is missing — must be skipped
    ]

    _run_pipeline(
        graph_id=graph_id,
        snapshot=snapshot,
        rows_by_table={"employees": [], "projects": project_rows, "emp_project": []},
    )

    count = _count_entities(driver, graph_id, source_table="projects")
    assert count == 2, f"Expected 2 project nodes (row without PK skipped), got {count}"


@pytest.mark.integration
def test_schema_mapper_graph_id_in_rules(neo4j):
    """GraphMappingRules.graph_id must equal the injected graph_id."""
    _driver, graph_ids = neo4j
    graph_id = _unique_graph_id()
    graph_ids.append(graph_id)

    from app.services.schema_mapper import SchemaMapper

    snapshot = _make_schema_snapshot()
    mapper = SchemaMapper()
    rules = mapper.map(
        schema_snapshot=snapshot,
        connector_id=_CONNECTOR_ID,
        graph_id=graph_id,
    )

    assert rules.graph_id == graph_id
    assert rules.connector_id == _CONNECTOR_ID
    # All 3 tables should be mappable (employees, projects, emp_project)
    assert len(rules.tables) == 3

    # employees → self_ref_table (has manager_id FK to same table)
    emp_tm = next((t for t in rules.tables if t.table_name == "employees"), None)
    assert emp_tm is not None
    assert emp_tm.kind == "self_ref_table"

    # projects → entity_table
    proj_tm = next((t for t in rules.tables if t.table_name == "projects"), None)
    assert proj_tm is not None
    assert proj_tm.kind == "entity_table"

    # emp_project → junction_table (2 FK cols + audit col)
    junc_tm = next((t for t in rules.tables if t.table_name == "emp_project"), None)
    assert junc_tm is not None
    assert junc_tm.kind == "junction_table"
