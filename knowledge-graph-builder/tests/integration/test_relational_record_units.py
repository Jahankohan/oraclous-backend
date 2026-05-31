"""Integration test — relational `record` units + `fk_target` edges.

TASK-235 / STORY-034 / ADR-022.

TASK-222 shipped a relational primitive that emitted only `:Table` / `:Column`
*structure* units — a relational source therefore produced structure but no
entities, because the recipe engine projects `node` rules over `record` units.
TASK-235 closes that gap: the relational primitive now emits one `record` unit
per row, and the engine resolves `edge resolve_by: "fk_target"`.

These are **integration tests** against the live Docker stack:

  * a test `employees` table (with a self-referencing `manager_id` FK and a
    `department_id` FK, plus a `departments` table) is created in the
    Dockerized Postgres, a few rows inserted, all under a uniquely-named
    schema that is dropped in teardown;
  * the `PostgreSQLConnector` introspects the schema and fetches the rows;
  * the `RelationalPrimitive` decomposes snapshot + rows into a
    `StructuralRepresentation` — asserted to carry one `record` unit per row
    with the correct `parent_id` and field values;
  * the reconciled `org-chart-relational` recipe is run through
    `RecipeExecutionEngine` against the Dockerized Neo4j — asserted to create
    `Employee` / `Department` entity nodes and `REPORTS_TO` / `MEMBER_OF`
    FK edges.

Postgres host: `settings.POSTGRES_URL` uses the Docker-network host `postgres`;
rewritten to `localhost` for a test process outside the compose network.
`TEST_POSTGRES_URL` overrides it. Neo4j defaults match `docker-compose.yml`
(`neo4j`/`password`, `bolt://localhost:7687`); override with `NEO4J_TEST_*`.
The module skips cleanly if either store is unreachable.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

import pytest
import pytest_asyncio
from neo4j import GraphDatabase

from app.core.config import settings
from app.recipes.engine import RecipeExecutionEngine
from app.recipes.primitives.interface import ExtractionMode
from app.recipes.primitives.relational_primitive import RelationalPrimitive
from app.services.database_connector_service import (
    PostgreSQLConnector,
)

# ---------------------------------------------------------------------------
# Connection resolution
# ---------------------------------------------------------------------------

_NEO4J_URI = os.environ.get("NEO4J_TEST_URI", "bolt://localhost:7687")
_NEO4J_USER = os.environ.get("NEO4J_TEST_USER", "neo4j")
_NEO4J_PASSWORD = os.environ.get("NEO4J_TEST_PASSWORD", "password")

_EXAMPLES = Path(__file__).resolve().parents[2] / "app" / "recipes" / "examples"
_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "cypher"
    / "migrations"
    / "2026-05-19_unified_graph_model.cypher"
)


def _resolve_postgres_dsn() -> str:
    """A Postgres URL reachable from the test host.

    `settings.POSTGRES_URL` uses the Docker-network host `postgres`; rewritten
    to `localhost` for a test process running outside the compose network.
    `TEST_POSTGRES_URL` takes precedence if set.
    """
    override = os.getenv("TEST_POSTGRES_URL")
    if override:
        return override
    return settings.POSTGRES_URL.replace("@postgres:", "@localhost:")


def _postgres_parts() -> dict[str, object]:
    """Host / port / database / credentials parsed from the resolved DSN."""
    dsn = _resolve_postgres_dsn().replace("+asyncpg", "")
    parsed = urlparse(dsn)
    return {
        "host": parsed.hostname or "localhost",
        "port": parsed.port or 5432,
        "database": (parsed.path or "/").lstrip("/"),
        "user": parsed.username or "postgres",
        "password": parsed.password or "",
    }


# A test-scoped Postgres schema — created and dropped per test run so the
# fixture tables never collide with real data.
_TEST_SCHEMA = f"task235_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def pg_schema():
    """Create the test schema + employees/departments tables; drop in teardown.

    Skips the test if Postgres is unreachable.
    """
    import asyncpg  # type: ignore

    parts = _postgres_parts()
    try:
        conn = await asyncpg.connect(
            host=parts["host"],
            port=parts["port"],
            user=parts["user"],
            password=parts["password"],
            database=parts["database"],
            timeout=10,
        )
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"live Postgres not reachable: {exc}")

    try:
        await conn.execute(f'CREATE SCHEMA "{_TEST_SCHEMA}"')
        await conn.execute(
            f"""
            CREATE TABLE "{_TEST_SCHEMA}".departments (
                id   INTEGER PRIMARY KEY,
                name TEXT NOT NULL
            )
            """
        )
        await conn.execute(
            f"""
            CREATE TABLE "{_TEST_SCHEMA}".employees (
                id            INTEGER PRIMARY KEY,
                name          TEXT NOT NULL,
                title         TEXT,
                manager_id    INTEGER REFERENCES "{_TEST_SCHEMA}".employees (id),
                department_id INTEGER REFERENCES "{_TEST_SCHEMA}".departments (id)
            )
            """
        )
        await conn.execute(
            f'INSERT INTO "{_TEST_SCHEMA}".departments (id, name) VALUES '
            "(10, 'Engineering'), (20, 'Operations')"
        )
        # Ada is the root (no manager); Alan & Grace report to Ada;
        # Linus reports to Alan. A 4-node, 3-edge reporting tree.
        await conn.execute(
            f'INSERT INTO "{_TEST_SCHEMA}".employees '
            "(id, name, title, manager_id, department_id) VALUES "
            "(1, 'Ada Lovelace', 'CTO', NULL, 10),"
            "(2, 'Alan Turing', 'Principal Engineer', 1, 10),"
            "(3, 'Grace Hopper', 'VP Engineering', 1, 10),"
            "(4, 'Linus Torvalds', 'Staff Engineer', 2, 20)"
        )
    except Exception:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{_TEST_SCHEMA}" CASCADE')
        await conn.close()
        raise

    yield parts

    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{_TEST_SCHEMA}" CASCADE')
    finally:
        await conn.close()


@pytest.fixture(scope="module")
def driver():
    """Live Neo4j driver; skips the module if Neo4j is unreachable."""
    try:
        d = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
        d.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"live Neo4j not reachable at {_NEO4J_URI}: {exc}")
    # The engine MERGEs against the unified-graph-model constraints — make sure
    # they exist (idempotent — a no-op if the migration already ran).
    with d.session() as session:
        body = "\n".join(
            line
            for line in _MIGRATION.read_text().splitlines()
            if not line.lstrip().startswith("//")
        )
        for stmt in (s.strip() for s in body.split(";") if s.strip()):
            session.run(stmt)
    yield d
    d.close()


@pytest.fixture
def graph_id():
    """A fresh, isolated test graph_id per test."""
    return f"test-relational-{uuid.uuid4()}"


@pytest.fixture
def cleanup(driver):
    """Delete every Neo4j node carrying the test graph_id after the test."""
    ids: list[str] = []
    yield ids
    with driver.session() as session:
        for gid in ids:
            session.run("MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid})


@pytest.fixture
def engine():
    return RecipeExecutionEngine()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _introspect_and_fetch(parts: dict[str, object], mode: ExtractionMode):
    """Connect the PostgreSQLConnector, introspect the test schema, fetch rows.

    Returns ``(snapshot, rows_by_table)`` — the input the relational primitive
    consumes.
    """
    connector = PostgreSQLConnector(
        {
            "host": parts["host"],
            "port": parts["port"],
            "database": parts["database"],
            "schema_filter": _TEST_SCHEMA,
        }
    )
    await connector.connect(str(parts["user"]), str(parts["password"]))
    try:
        snapshot = await connector.introspect_schema()
        rows_by_table: dict[str, list[dict]] = {}
        for table in snapshot.tables:
            limit = 1000 if mode == ExtractionMode.FULL else 2
            rows_by_table[table.name] = await connector.fetch_rows(
                table.name, snapshot, limit=limit
            )
    finally:
        await connector.close()
    return snapshot, rows_by_table


# ---------------------------------------------------------------------------
# A — relational primitive emits record units
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_relational_primitive_emits_record_units(pg_schema):
    """The relational primitive emits one `record` unit per row in FULL mode."""
    snapshot, rows = asyncio.run(_introspect_and_fetch(pg_schema, ExtractionMode.FULL))
    rep = RelationalPrimitive().decompose(snapshot, ExtractionMode.FULL, rows)

    records = [u for u in rep.units if u.kind.value == "record"]
    emp_records = [u for u in records if u.name == "employees"]
    dept_records = [u for u in records if u.name == "departments"]

    # 4 employees + 2 departments inserted by the fixture.
    assert len(emp_records) == 4
    assert len(dept_records) == 2

    # Each record is parented at its `:Table` unit and carries its row payload.
    for unit in emp_records:
        assert unit.parent_id == "table:employees"
        assert len(unit.sample_values) == 1
        payload = unit.sample_values[0]
        assert set(payload) == {"id", "name", "title", "manager_id", "department_id"}

    names = {u.sample_values[0]["name"] for u in emp_records}
    assert names == {"Ada Lovelace", "Alan Turing", "Grace Hopper", "Linus Torvalds"}

    # The table/column structure units are unchanged — still emitted.
    assert any(u.kind.value == "table" and u.name == "employees" for u in rep.units)
    fk_cols = [
        u for u in rep.units if u.kind.value == "column" and u.role == "foreign_key"
    ]
    # manager_id + department_id on employees.
    assert {u.name for u in fk_cols} == {"manager_id", "department_id"}
    for col in fk_cols:
        assert "fk_target" in col.metadata


@pytest.mark.integration
def test_relational_primitive_sample_mode_bounds_records(pg_schema):
    """SAMPLE mode emits only the caller-bounded slice of rows."""
    snapshot, rows = asyncio.run(
        _introspect_and_fetch(pg_schema, ExtractionMode.SAMPLE)
    )
    rep = RelationalPrimitive().decompose(snapshot, ExtractionMode.SAMPLE, rows)

    emp_records = [
        u for u in rep.units if u.kind.value == "record" and u.name == "employees"
    ]
    # The fixture inserted 4 employees; SAMPLE fetched a limit of 2.
    assert len(emp_records) == 2


# ---------------------------------------------------------------------------
# B + C — recipe execution: entities + fk_target edges
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_org_chart_recipe_creates_entities_and_fk_edges(
    pg_schema, engine, driver, graph_id, cleanup
):
    """The reconciled org-chart recipe projects Employee/Department entities
    and resolves the REPORTS_TO / MEMBER_OF foreign-key edges."""
    cleanup.append(graph_id)

    snapshot, rows = asyncio.run(_introspect_and_fetch(pg_schema, ExtractionMode.FULL))
    rep = RelationalPrimitive().decompose(snapshot, ExtractionMode.FULL, rows)
    recipe = json.loads((_EXAMPLES / "org-chart-relational.recipe.json").read_text())

    result = engine.execute(recipe, rep, graph_id, driver)

    # 4 employees + 2 departments = 6 entity nodes; salary_history is skipped
    # (and absent from this fixture anyway).
    assert result.nodes_written == 6
    # 3 REPORTS_TO + 4 MEMBER_OF = 7 edges.
    assert result.edges_written == 7
    assert result.warnings == []

    with driver.session() as session:
        employees = session.run(
            "MATCH (e:Employee:__Entity__ {graph_id: $gid}) RETURN count(e) AS c",
            {"gid": graph_id},
        ).single()["c"]
        departments = session.run(
            "MATCH (d:Department:__Entity__ {graph_id: $gid}) RETURN count(d) AS c",
            {"gid": graph_id},
        ).single()["c"]
        assert employees == 4
        assert departments == 2

        # REPORTS_TO — Alan & Grace -> Ada; Linus -> Alan.
        reports = session.run(
            """
            MATCH (e:Employee {graph_id: $gid})-[:REPORTS_TO]->(m:Employee {graph_id: $gid})
            RETURN e.name AS emp, m.name AS mgr ORDER BY emp
            """,
            {"gid": graph_id},
        ).data()
        assert reports == [
            {"emp": "Alan Turing", "mgr": "Ada Lovelace"},
            {"emp": "Grace Hopper", "mgr": "Ada Lovelace"},
            {"emp": "Linus Torvalds", "mgr": "Alan Turing"},
        ]
        # Ada has no manager — no outgoing REPORTS_TO.
        ada_out = session.run(
            """
            MATCH (e:Employee {graph_id: $gid})-[:REPORTS_TO]->()
            WHERE e.name = 'Ada Lovelace'
            RETURN count(*) AS c
            """,
            {"gid": graph_id},
        ).single()["c"]
        assert ada_out == 0

        # MEMBER_OF — Ada/Alan/Grace -> Engineering; Linus -> Operations.
        members = session.run(
            """
            MATCH (e:Employee {graph_id: $gid})-[:MEMBER_OF]->(d:Department {graph_id: $gid})
            RETURN e.name AS emp, d.name AS dept ORDER BY emp
            """,
            {"gid": graph_id},
        ).data()
        assert members == [
            {"emp": "Ada Lovelace", "dept": "Engineering"},
            {"emp": "Alan Turing", "dept": "Engineering"},
            {"emp": "Grace Hopper", "dept": "Engineering"},
            {"emp": "Linus Torvalds", "dept": "Operations"},
        ]


@pytest.mark.integration
def test_org_chart_recipe_is_idempotent(pg_schema, engine, driver, graph_id, cleanup):
    """Re-running the recipe yields the same graph — no duplicate edges/nodes."""
    cleanup.append(graph_id)

    snapshot, rows = asyncio.run(_introspect_and_fetch(pg_schema, ExtractionMode.FULL))
    rep = RelationalPrimitive().decompose(snapshot, ExtractionMode.FULL, rows)
    recipe = json.loads((_EXAMPLES / "org-chart-relational.recipe.json").read_text())

    first = engine.execute(recipe, rep, graph_id, driver)
    second = engine.execute(recipe, rep, graph_id, driver)

    assert first.nodes_written == second.nodes_written == 6
    assert first.edges_written == second.edges_written == 7

    with driver.session() as session:
        employees = session.run(
            "MATCH (e:Employee {graph_id: $gid}) RETURN count(e) AS c",
            {"gid": graph_id},
        ).single()["c"]
        reports = session.run(
            """
            MATCH (:Employee {graph_id: $gid})-[r:REPORTS_TO]->(:Employee {graph_id: $gid})
            RETURN count(r) AS c
            """,
            {"gid": graph_id},
        ).single()["c"]
        members = session.run(
            """
            MATCH (:Employee {graph_id: $gid})-[r:MEMBER_OF]->(:Department {graph_id: $gid})
            RETURN count(r) AS c
            """,
            {"gid": graph_id},
        ).single()["c"]
    # Two runs, still exactly 4 employees / 3 REPORTS_TO / 4 MEMBER_OF.
    assert employees == 4
    assert reports == 3
    assert members == 4
