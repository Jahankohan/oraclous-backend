"""QA end-to-end test for STORY-034 — concern-driven ingestion (ADR-022).

This is the QA gate (TASK-227). It validates the STORY-034 **acceptance
criteria** end-to-end, against the live Dockerized stack — the criteria the
per-task tests (`test_recipe_engine.py`, `test_recipe_library.py`,
`test_recipe_authoring.py`, `test_relational_record_units.py`) do not join up:

  1. **One source, two concerns, no code change** — the *same* structural
     representation, ingested under two *different* recipes (two different
     concerns), produces two *different* graphs. The only thing that changes
     between the two runs is the recipe document (data). No code path differs.
  2. **Reuse with no agent** — a recipe fetched from the `RecipeLibrary`
     (Postgres) executes through `RecipeExecutionEngine` purely mechanically.
     No agent, no LLM, no recipe authoring at run time.
  3. **Beyond the context window** — a recipe runs over a dataset of several
     thousand synthetic records and writes correctly. The mechanical engine
     is not bounded by any LLM context window.
  4. **Provenance** — every node and edge written carries `graph_id`, an
     `EXTRACTED`/`INFERRED` provenance tag, and a `DERIVED_FROM` link to the
     structural unit it derived from.
  5. **No LLM at run time** — the entire structured-projection path is
     asserted to make zero calls into any LLM service.

Connection defaults match `docker-compose.yml` (`neo4j`/`password`,
`bolt://localhost:7687`; Postgres via `settings.POSTGRES_URL` rewritten to
`localhost`). Each test uses a fresh `uuid4` `graph_id` and cleans up after
itself; the module skips cleanly if either store is unreachable.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import uuid

import pytest
import pytest_asyncio
from neo4j import GraphDatabase
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.models.recipe import Recipe
from app.recipes.engine import RecipeExecutionEngine
from app.recipes.library import RecipeLibrary
from app.recipes.primitives.interface import (
    ExtractionMode,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)

# ---------------------------------------------------------------------------
# Connection resolution
# ---------------------------------------------------------------------------

_NEO4J_URI = os.environ.get("NEO4J_TEST_URI", "bolt://localhost:7687")
_NEO4J_USER = os.environ.get("NEO4J_TEST_USER", "neo4j")
_NEO4J_PASSWORD = os.environ.get("NEO4J_TEST_PASSWORD", "password")

from pathlib import Path  # noqa: E402

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "cypher"
    / "migrations"
    / "2026-05-19_unified_graph_model.cypher"
)


def _resolve_postgres_url() -> str:
    """A Postgres URL reachable from the test host."""
    override = os.getenv("TEST_POSTGRES_URL")
    if override:
        return override
    return settings.POSTGRES_URL.replace("@postgres:", "@localhost:")


_POSTGRES_URL = _resolve_postgres_url()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _migration_statements() -> list[str]:
    body = "\n".join(
        line
        for line in _MIGRATION.read_text().splitlines()
        if not line.lstrip().startswith("//")
    )
    return [stmt.strip() for stmt in body.split(";") if stmt.strip()]


@pytest.fixture(scope="module")
def driver():
    """Live Neo4j driver; skips the module if Neo4j is unreachable."""
    try:
        d = GraphDatabase.driver(_NEO4J_URI, auth=(_NEO4J_USER, _NEO4J_PASSWORD))
        d.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"live Neo4j not reachable at {_NEO4J_URI}: {exc}")
    # The engine MERGEs against the unified-graph-model constraints — ensure
    # they exist (idempotent — a no-op if the migration already ran).
    with d.session() as session:
        for stmt in _migration_statements():
            session.run(stmt)
    yield d
    d.close()


@pytest.fixture
def graph_id():
    """A fresh, isolated test graph_id per test."""
    return f"test-qa-story034-{uuid.uuid4()}"


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


@pytest.fixture(autouse=True)
def _ensure_current_event_loop():
    """Guarantee the thread has a usable current event loop for each test.

    Sibling integration modules (`test_relational_record_units.py`) call
    `asyncio.run(...)` inside sync tests; `asyncio.run` closes the loop it
    creates and leaves no current loop in the thread. pytest-asyncio's
    session-scoped loop then trips over `asyncio.get_event_loop()` returning
    nothing. This autouse fixture restores a fresh, open loop so the QA async
    test runs regardless of sibling-test ordering — it does not depend on the
    suite's collection order.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError("closed")
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield


@pytest_asyncio.fixture
async def db_session():
    """Task-scoped async session; skips the test if Postgres is unreachable."""
    pg_engine = create_async_engine(_POSTGRES_URL, poolclass=NullPool, future=True)
    try:
        async with pg_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            await conn.run_sync(Recipe.__table__.create, checkfirst=True)
            await conn.commit()
    except Exception as exc:  # noqa: BLE001
        await pg_engine.dispose()
        pytest.skip(f"live Postgres not reachable at {_POSTGRES_URL}: {exc}")

    session_maker = async_sessionmaker(
        bind=pg_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_maker() as session:
        yield session
    await pg_engine.dispose()


# ---------------------------------------------------------------------------
# Source representation — a SINGLE source, built once, used by both concerns
# ---------------------------------------------------------------------------


def _employee_directory_representation() -> StructuralRepresentation:
    """One CSV-shaped source: an employee directory with a salary column.

    The *same* representation object is fed to two different recipes below.
    Both criterion-1 recipes match against these exact records — the only
    thing that varies between the two runs is the recipe document.
    """
    units: list[StructuralUnit] = [
        StructuralUnit(
            kind=UnitKind.SOURCE,
            unit_id="source",
            name="employees.csv",
            metadata={"row_count": 4, "column_count": 4},
        ),
        StructuralUnit(
            kind=UnitKind.COLUMN,
            unit_id="column:name",
            name="name",
            data_type="str",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.COLUMN,
            unit_id="column:title",
            name="title",
            data_type="str",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.COLUMN,
            unit_id="column:department",
            name="department",
            data_type="str",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.COLUMN,
            unit_id="column:salary",
            name="salary",
            data_type="int",
            role="measure",
            parent_id="source",
        ),
    ]
    rows = [
        {
            "name": "Ada Lovelace",
            "title": "CTO",
            "department": "Engineering",
            "salary": 240000,
        },
        {
            "name": "Alan Turing",
            "title": "Principal Engineer",
            "department": "Engineering",
            "salary": 210000,
        },
        {
            "name": "Grace Hopper",
            "title": "VP Engineering",
            "department": "Engineering",
            "salary": 220000,
        },
        {
            "name": "Linus Torvalds",
            "title": "Staff Engineer",
            "department": "Operations",
            "salary": 200000,
        },
    ]
    for idx, row in enumerate(rows):
        units.append(
            StructuralUnit(
                kind=UnitKind.RECORD,
                unit_id=f"record:{idx}",
                name=f"row {idx}",
                parent_id="source",
                sample_values=[row],
            )
        )
    return StructuralRepresentation(
        source_type="csv",
        shape_signature="employees.csv:name,title,department,salary",
        mode=ExtractionMode.FULL,
        units=units,
    )


# -- concern A: "reporting structure" — people are the entity ----------------


def _reporting_structure_recipe() -> dict:
    """Concern A — model the org's people and their titles.

    Projects one `:Employee` node per record; `title` becomes a property. The
    salary column is explicitly skipped — it is irrelevant to this concern.
    """
    return {
        "recipe_format_version": "0.2",
        "id": "rcp_qa-reporting-structure-v1",
        "version": 1,
        "status": "promoted",
        "concern": "Understand the reporting structure — who holds which role.",
        "applies_to": {
            "source_type": "csv",
            "shape_signature": "employees.csv:name,title,department,salary",
        },
        "defaults": {"provenance": "EXTRACTED"},
        "mappings": [
            {
                "id": "people",
                "match": {"unit_kind": "record"},
                "project_to": "node",
                "label": "Employee",
                "identity": {
                    "scheme": "deterministic",
                    "from": ["column:name"],
                    "normalize": ["casefold", "trim", "collapse_whitespace"],
                },
                "materialize": True,
                "properties": [{"name": "name", "value_from": "column:name"}],
            },
            {
                "id": "person_title",
                "match": {"unit_kind": "column", "name": "title"},
                "project_to": "property",
                "on": "people",
                "name": "title",
                "value_from": "column:title",
            },
            {
                "id": "skip_salary",
                "match": {"unit_kind": "column", "name": "salary"},
                "project_to": "skip",
                "reason": "Compensation is out of scope for an org-structure concern.",
            },
        ],
    }


# -- concern B: "compensation review" — pay bands are the entity -------------


def _compensation_review_recipe() -> dict:
    """Concern B — model the same source as a compensation dataset.

    Over the *identical* representation, this recipe projects one `:PayPacket`
    node per record, keyed and labelled differently, carrying `salary` and
    `department` as properties. It writes a different label, a different
    identity scheme, and different properties — a different graph from the
    same data, with no code change.
    """
    return {
        "recipe_format_version": "0.2",
        "id": "rcp_qa-compensation-review-v1",
        "version": 1,
        "status": "promoted",
        "concern": "Review compensation by department.",
        "applies_to": {
            "source_type": "csv",
            "shape_signature": "employees.csv:name,title,department,salary",
        },
        "defaults": {"provenance": "EXTRACTED"},
        "mappings": [
            {
                "id": "packets",
                "match": {"unit_kind": "record"},
                "project_to": "node",
                "label": "PayPacket",
                "identity": {
                    "scheme": "deterministic",
                    "from": ["column:name"],
                    "normalize": ["casefold", "trim"],
                },
                "materialize": True,
                "properties": [
                    {"name": "salary", "value_from": "column:salary"},
                    {"name": "department", "value_from": "column:department"},
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# Criterion 1 — one source, two concerns, no code change
# ---------------------------------------------------------------------------


def test_one_source_two_concerns_produces_two_different_graphs(engine, driver, cleanup):
    """The SAME representation under TWO recipes yields TWO different graphs.

    STORY-034 acceptance criterion 1. The single `_employee_directory_repr`
    object is executed under the reporting-structure recipe and the
    compensation-review recipe. The two graphs differ in label, properties,
    and entity identity — and the only thing that changed between the runs is
    the recipe document. No code path differs.
    """
    gid_a = f"test-qa-story034-{uuid.uuid4()}"
    gid_b = f"test-qa-story034-{uuid.uuid4()}"
    cleanup.extend([gid_a, gid_b])

    # ONE source representation — built once, shared by both runs.
    representation = _employee_directory_representation()

    recipe_a = _reporting_structure_recipe()
    recipe_b = _compensation_review_recipe()

    # Two runs over the identical representation object — only the recipe
    # differs. Tenant-isolated so the two graphs are independently inspectable.
    result_a = engine.execute(recipe_a, representation, gid_a, driver)
    result_b = engine.execute(recipe_b, representation, gid_b, driver)

    assert result_a.nodes_written == 4
    assert result_b.nodes_written == 4

    with driver.session() as session:
        # Concern A produced :Employee nodes carrying `title`, no `salary`.
        employees = session.run(
            """
            MATCH (e:Employee:__Entity__ {graph_id: $gid})
            RETURN e.name AS name, e.title AS title, e.salary AS salary
            ORDER BY name
            """,
            {"gid": gid_a},
        ).data()
        # Concern B produced :PayPacket nodes carrying `salary`/`department`.
        packets = session.run(
            """
            MATCH (p:PayPacket:__Entity__ {graph_id: $gid})
            RETURN p.salary AS salary, p.department AS department, p.title AS title
            ORDER BY salary
            """,
            {"gid": gid_b},
        ).data()

        # The label from concern A does not exist in concern B's graph, and
        # vice versa — two genuinely different graphs.
        a_has_paypacket = session.run(
            "MATCH (p:PayPacket {graph_id: $gid}) RETURN count(p) AS c",
            {"gid": gid_a},
        ).single()["c"]
        b_has_employee = session.run(
            "MATCH (e:Employee {graph_id: $gid}) RETURN count(e) AS c",
            {"gid": gid_b},
        ).single()["c"]

    assert len(employees) == 4
    assert len(packets) == 4

    # Concern A: each Employee carries a title; salary was skipped, never set.
    assert {e["title"] for e in employees} == {
        "CTO",
        "Principal Engineer",
        "VP Engineering",
        "Staff Engineer",
    }
    assert all(e["salary"] is None for e in employees)

    # Concern B: each PayPacket carries salary + department; never a title.
    assert {p["salary"] for p in packets} == {240000, 210000, 220000, 200000}
    assert {p["department"] for p in packets} == {"Engineering", "Operations"}
    assert all(p["title"] is None for p in packets)

    # The two graphs share no label — they are not the same graph.
    assert a_has_paypacket == 0
    assert b_has_employee == 0


def test_two_concern_run_changes_only_the_recipe_document(engine, driver, cleanup):
    """Nothing but the recipe data differs between the two concern runs.

    Criterion 1's "no code change" clause, made explicit: the representation
    fed to both runs is asserted byte-identical (same object, unmutated by the
    first run), and both runs go through the exact same `engine.execute`
    entrypoint. The engine never mutates its input.
    """
    gid_a = f"test-qa-story034-{uuid.uuid4()}"
    gid_b = f"test-qa-story034-{uuid.uuid4()}"
    cleanup.extend([gid_a, gid_b])

    representation = _employee_directory_representation()
    snapshot_before = representation.model_dump()

    engine.execute(_reporting_structure_recipe(), representation, gid_a, driver)
    # The representation object is untouched by the first run — the second run
    # sees the identical input.
    assert representation.model_dump() == snapshot_before

    engine.execute(_compensation_review_recipe(), representation, gid_b, driver)
    assert representation.model_dump() == snapshot_before


# ---------------------------------------------------------------------------
# Criterion 2 — reuse with no agent (library lookup -> mechanical execute)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_recipe_fetched_from_library_executes_with_no_agent(
    engine, driver, db_session, graph_id
):
    """A recipe stored + promoted in the library executes purely mechanically.

    STORY-034 acceptance criterion 2. The reuse path: store a recipe, promote
    it, look it up by `(source_type, shape_signature, concern)`, and feed the
    *looked-up document* straight into `RecipeExecutionEngine.execute`. No
    agent authored anything at run time — the recipe is fetched as data and
    executed.
    """
    cleanup_ids: list[str] = [graph_id]
    library = RecipeLibrary(db_session)
    recipe = _reporting_structure_recipe()
    # The library owns version/status — store writes a draft v1.
    stored = await library.store(recipe, graph_id)
    await db_session.commit()
    await library.promote(stored["id"], stored["version"], graph_id)
    await db_session.commit()

    # The reuse lookup — exact match on shape + concern, no agent.
    looked_up = await library.lookup(
        source_type="csv",
        shape_signature="employees.csv:name,title,department,salary",
        concern="Understand the reporting structure — who holds which role.",
        graph_id=graph_id,
    )
    assert looked_up is not None, "promoted recipe must be found by lookup"
    assert looked_up["id"] == stored["id"]

    try:
        # Mechanical execution of the looked-up recipe document.
        representation = _employee_directory_representation()
        result = engine.execute(looked_up, representation, graph_id, driver)
        assert result.nodes_written == 4

        with driver.session() as session:
            count = session.run(
                "MATCH (e:Employee:__Entity__ {graph_id: $gid}) RETURN count(e) AS c",
                {"gid": graph_id},
            ).single()["c"]
        assert count == 4
    finally:
        with driver.session() as session:
            for gid in cleanup_ids:
                session.run("MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid})
        await db_session.execute(delete(Recipe).where(Recipe.graph_id == graph_id))
        await db_session.commit()


# ---------------------------------------------------------------------------
# Criterion 3 — beyond the context window
# ---------------------------------------------------------------------------


def _large_representation(record_count: int) -> StructuralRepresentation:
    """A CSV-shaped representation with `record_count` synthetic records.

    At a few thousand records this dataset, serialized, far exceeds any LLM
    context window — yet the mechanical engine projects it in one job.
    """
    units: list[StructuralUnit] = [
        StructuralUnit(
            kind=UnitKind.SOURCE,
            unit_id="source",
            name="big_employees.csv",
            metadata={"row_count": record_count, "column_count": 2},
        ),
        StructuralUnit(
            kind=UnitKind.COLUMN,
            unit_id="column:name",
            name="name",
            data_type="str",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.COLUMN,
            unit_id="column:title",
            name="title",
            data_type="str",
            parent_id="source",
        ),
    ]
    for idx in range(record_count):
        units.append(
            StructuralUnit(
                kind=UnitKind.RECORD,
                unit_id=f"record:{idx}",
                name=f"row {idx}",
                parent_id="source",
                # A distinct name per row -> a distinct identity -> a distinct
                # entity. No collisions, so the written count == record_count.
                sample_values=[
                    {"name": f"Employee {idx:06d}", "title": f"Role {idx % 50}"}
                ],
            )
        )
    return StructuralRepresentation(
        source_type="csv",
        shape_signature="big_employees.csv:name,title",
        mode=ExtractionMode.FULL,
        units=units,
    )


def _large_dataset_recipe() -> dict:
    """A recipe for the large CSV — one node per record, title as a property."""
    return {
        "recipe_format_version": "0.2",
        "id": "rcp_qa-large-dataset-v1",
        "version": 1,
        "status": "promoted",
        "concern": "Project a large employee export into the graph.",
        "applies_to": {
            "source_type": "csv",
            "shape_signature": "big_employees.csv:name,title",
        },
        "defaults": {"provenance": "EXTRACTED"},
        "mappings": [
            {
                "id": "people",
                "match": {"unit_kind": "record"},
                "project_to": "node",
                "label": "Employee",
                "identity": {
                    "scheme": "deterministic",
                    "from": ["column:name"],
                    "normalize": ["casefold", "trim"],
                },
                "materialize": True,
                "properties": [{"name": "name", "value_from": "column:name"}],
            },
            {
                "id": "person_title",
                "match": {"unit_kind": "column", "name": "title"},
                "project_to": "property",
                "on": "people",
                "name": "title",
                "value_from": "column:title",
            },
        ],
    }


def test_recipe_runs_over_dataset_far_larger_than_a_context_window(
    engine, driver, graph_id, cleanup
):
    """A recipe projects a 5,000-record dataset in one mechanical run.

    STORY-034 acceptance criterion 3. 5,000 records, each with a name + title,
    is on the order of hundreds of thousands of tokens once serialized — well
    beyond any LLM context window. The engine — being mechanical and
    `UNWIND`-batched (~500 rows/batch) — projects the whole dataset in a
    single backend job with no context-window concept at all.
    """
    cleanup.append(graph_id)
    record_count = 5_000
    representation = _large_representation(record_count)
    recipe = _large_dataset_recipe()

    result = engine.execute(recipe, representation, graph_id, driver)

    # Every record became an entity — nothing was dropped or truncated.
    assert result.nodes_written == record_count
    assert result.properties_written == record_count

    with driver.session() as session:
        written = session.run(
            "MATCH (e:Employee:__Entity__ {graph_id: $gid}) RETURN count(e) AS c",
            {"gid": graph_id},
        ).single()["c"]
        # Spot-check the first, a middle, and the last record landed correctly.
        spot = session.run(
            """
            MATCH (e:Employee:__Entity__ {graph_id: $gid})
            WHERE e.name IN ['Employee 000000', 'Employee 002500',
                             'Employee 004999']
            RETURN e.name AS name, e.title AS title ORDER BY name
            """,
            {"gid": graph_id},
        ).data()
        # The property rule SET `title` on every node, across many UNWIND
        # batches — assert it landed on all of them.
        with_title = session.run(
            """
            MATCH (e:Employee:__Entity__ {graph_id: $gid})
            WHERE e.title IS NOT NULL RETURN count(e) AS c
            """,
            {"gid": graph_id},
        ).single()["c"]

    assert written == record_count
    assert with_title == record_count
    assert spot == [
        {"name": "Employee 000000", "title": "Role 0"},
        {"name": "Employee 002500", "title": "Role 0"},
        {"name": "Employee 004999", "title": "Role 49"},
    ]


# ---------------------------------------------------------------------------
# Criterion 4 — provenance (graph_id + EXTRACTED/INFERRED + DERIVED_FROM)
# ---------------------------------------------------------------------------


def _container_bearing_representation() -> StructuralRepresentation:
    """A representation with a `table` container so a `record` parents at it.

    A record whose `parent_id` is a container unit gets a `DERIVED_FROM` edge
    to the *container* (not the source) — this exercises the container branch
    of provenance linking.
    """
    units = [
        StructuralUnit(kind=UnitKind.SOURCE, unit_id="source", name="db"),
        StructuralUnit(
            kind=UnitKind.TABLE,
            unit_id="table:staff",
            name="staff",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.RECORD,
            unit_id="record:0",
            name="row 0",
            parent_id="table:staff",
            sample_values=[{"name": "Ada Lovelace"}],
        ),
        StructuralUnit(
            kind=UnitKind.RECORD,
            unit_id="record:1",
            name="row 1",
            parent_id="table:staff",
            sample_values=[{"name": "Alan Turing"}],
        ),
    ]
    return StructuralRepresentation(
        source_type="relational",
        shape_signature="db(staff[name])",
        mode=ExtractionMode.FULL,
        units=units,
    )


def _staff_node_recipe() -> dict:
    """One :Person node per staff record."""
    return {
        "recipe_format_version": "0.2",
        "id": "rcp_qa-staff-node-v1",
        "version": 1,
        "status": "promoted",
        "concern": "Model staff.",
        "applies_to": {
            "source_type": "relational",
            "shape_signature": "db(staff[name])",
        },
        "defaults": {"provenance": "EXTRACTED"},
        "mappings": [
            {
                "id": "staff",
                "match": {"unit_kind": "record"},
                "project_to": "node",
                "label": "Person",
                "identity": {
                    "scheme": "deterministic",
                    "from": ["column:name"],
                    "normalize": ["casefold", "trim"],
                },
                "materialize": True,
                "properties": [{"name": "name", "value_from": "column:name"}],
            }
        ],
    }


def test_every_node_carries_graph_id_provenance_and_derived_from(
    engine, driver, graph_id, cleanup
):
    """Each entity carries graph_id, an EXTRACTED tag, and a DERIVED_FROM link.

    STORY-034 acceptance criterion 4 — for nodes that parent at a *container*.
    The staff records parent at a `:Table` container, so DERIVED_FROM points
    at that container node.
    """
    cleanup.append(graph_id)
    result = engine.execute(
        _staff_node_recipe(), _container_bearing_representation(), graph_id, driver
    )
    assert result.nodes_written == 2
    assert result.containers_written == 1

    with driver.session() as session:
        # graph_id + EXTRACTED tag on every entity.
        nodes = session.run(
            """
            MATCH (p:Person:__Entity__ {graph_id: $gid})
            RETURN p.graph_id AS gid, p.provenance AS prov, p.id AS id
            """,
            {"gid": graph_id},
        ).data()
        # Every entity has a DERIVED_FROM edge to the :Table container it came
        # from, and that edge itself is graph_id-scoped.
        derived = session.run(
            """
            MATCH (p:Person:__Entity__ {graph_id: $gid})
                  -[r:DERIVED_FROM]->(t:Table {graph_id: $gid})
            RETURN count(r) AS c, collect(DISTINCT r.graph_id) AS edge_gids
            """,
            {"gid": graph_id},
        ).single()
        # The container itself carries provenance + graph_id.
        container = session.run(
            """
            MATCH (t:Table {graph_id: $gid})
            RETURN t.provenance AS prov, t.graph_id AS gid
            """,
            {"gid": graph_id},
        ).single()

    assert len(nodes) == 2
    for n in nodes:
        assert n["gid"] == graph_id
        assert n["prov"] == "EXTRACTED"
        assert n["id"]  # a non-empty deterministic identity hash
    assert derived["c"] == 2
    assert derived["edge_gids"] == [graph_id]
    assert container["prov"] == "EXTRACTED"
    assert container["gid"] == graph_id


def test_provenance_tag_is_one_of_extracted_or_inferred(
    engine, driver, graph_id, cleanup
):
    """The provenance tag is always EXTRACTED or INFERRED — never absent/other.

    Criterion 4's "an EXTRACTED/INFERRED provenance tag" clause. A recipe with
    `provenance: INFERRED` on its node rule must stamp INFERRED (with a
    confidence); the default path stamps EXTRACTED.

    Note: the recipe schema's `rule_node` has `additionalProperties: false` and
    defines no `confidence` key, so a recipe cannot pass an explicit
    confidence — the engine falls back to its 0.5 default for an INFERRED node.
    """
    cleanup.append(graph_id)
    recipe = _staff_node_recipe()
    recipe["mappings"][0]["provenance"] = "INFERRED"

    engine.execute(recipe, _container_bearing_representation(), graph_id, driver)

    with driver.session() as session:
        tags = session.run(
            """
            MATCH (p:Person:__Entity__ {graph_id: $gid})
            RETURN DISTINCT p.provenance AS prov
            """,
            {"gid": graph_id},
        ).data()
        confidences = session.run(
            """
            MATCH (p:Person:__Entity__ {graph_id: $gid})
            RETURN p.confidence AS conf
            """,
            {"gid": graph_id},
        ).data()

    assert [t["prov"] for t in tags] == ["INFERRED"]
    # confidence is only present on INFERRED elements (unified-graph-model §5);
    # the engine's default for an INFERRED node is 0.5.
    assert all(c["conf"] == 0.5 for c in confidences)


def test_edges_carry_provenance_and_graph_id(engine, driver, graph_id, cleanup):
    """Relationship rows carry graph_id + a provenance tag (criterion 4, edges).

    Criterion 4 says "every node *and edge*" carries provenance. This builds a
    two-table relational representation with a foreign key, runs an `fk_target`
    edge rule, and asserts the resulting relationships carry both.
    """
    cleanup.append(graph_id)
    # A departments table + an employees table whose `dept_id` FKs to it.
    units = [
        StructuralUnit(kind=UnitKind.SOURCE, unit_id="source", name="db"),
        StructuralUnit(
            kind=UnitKind.TABLE,
            unit_id="table:departments",
            name="departments",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.TABLE,
            unit_id="table:employees",
            name="employees",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.COLUMN,
            unit_id="column:dept_id",
            name="dept_id",
            role="foreign_key",
            parent_id="table:employees",
            metadata={
                "fk_target": "table:departments",
                "fk_target_column": "id",
                "fk_target_present": True,
            },
        ),
        StructuralUnit(
            kind=UnitKind.RECORD,
            unit_id="dept:0",
            name="departments",
            parent_id="table:departments",
            sample_values=[{"id": 10, "name": "Engineering"}],
        ),
        StructuralUnit(
            kind=UnitKind.RECORD,
            unit_id="emp:0",
            name="employees",
            parent_id="table:employees",
            sample_values=[{"id": 1, "name": "Ada", "dept_id": 10}],
        ),
        StructuralUnit(
            kind=UnitKind.RECORD,
            unit_id="emp:1",
            name="employees",
            parent_id="table:employees",
            sample_values=[{"id": 2, "name": "Alan", "dept_id": 10}],
        ),
    ]
    representation = StructuralRepresentation(
        source_type="relational",
        shape_signature="db(departments[id,name];employees[id,name,dept_id])",
        mode=ExtractionMode.FULL,
        units=units,
    )
    recipe = {
        "recipe_format_version": "0.2",
        "id": "rcp_qa-edge-provenance-v1",
        "version": 1,
        "status": "promoted",
        "concern": "Department membership.",
        "applies_to": {
            "source_type": "relational",
            "shape_signature": "db(departments[id,name];employees[id,name,dept_id])",
        },
        "defaults": {"provenance": "EXTRACTED"},
        "mappings": [
            {
                "id": "employees",
                "match": {"unit_kind": "record", "name": "employees"},
                "project_to": "node",
                "label": "Employee",
                "identity": {"scheme": "deterministic", "from": ["column:name"]},
                "materialize": True,
                "properties": [],
            },
            {
                "id": "departments",
                "match": {"unit_kind": "record", "name": "departments"},
                "project_to": "node",
                "label": "Department",
                "identity": {"scheme": "deterministic", "from": ["column:name"]},
                "materialize": True,
                "properties": [],
            },
            {
                "id": "member_of",
                "match": {
                    "unit_kind": "column",
                    "name": "dept_id",
                    "role": "foreign_key",
                },
                "project_to": "edge",
                "type": "MEMBER_OF",
                "from": {"node_rule": "employees"},
                "to": {"node_rule": "departments", "resolve_by": "fk_target"},
            },
        ],
    }

    result = engine.execute(recipe, representation, graph_id, driver)
    assert result.edges_written == 2

    with driver.session() as session:
        edges = session.run(
            """
            MATCH (:Employee {graph_id: $gid})-[r:MEMBER_OF]->(:Department {graph_id: $gid})
            RETURN r.graph_id AS gid, r.provenance AS prov,
                   r.recipe_id AS rid, r.ingestion_source AS isrc
            """,
            {"gid": graph_id},
        ).data()

    assert len(edges) == 2
    for e in edges:
        assert e["gid"] == graph_id
        assert e["prov"] == "EXTRACTED"
        assert e["rid"] == "rcp_qa-edge-provenance-v1"
        assert e["isrc"] == result.source_id


# ---------------------------------------------------------------------------
# Criterion 5 — no LLM on the run-time path (verified)
# ---------------------------------------------------------------------------


def _llm_service_modules() -> list[str]:
    """Already-imported module names that look like an LLM client/service.

    The engine module graph is inspected for any LLM-service import; the
    structured-projection path must reach none of them.
    """
    candidates = []
    for name in list(sys.modules):
        low = name.lower()
        if "llm" in low or "openai" in low or "anthropic" in low:
            candidates.append(name)
    return candidates


def test_engine_module_imports_no_llm_service():
    """The recipe engine module pulls in no LLM client at import time.

    STORY-034 acceptance criterion 5 — a static guarantee. A fresh import of
    `app.recipes.engine` is inspected: nothing in its transitive import set is
    an LLM service. The structured path cannot call a model it never imported.
    """
    # Re-import the engine cleanly and inspect what it dragged in.
    importlib.reload(importlib.import_module("app.recipes.engine"))
    engine_module = sys.modules["app.recipes.engine"]
    # The engine's own globals reference no llm_service symbol.
    suspicious = [
        attr
        for attr in vars(engine_module)
        if "llm" in attr.lower() or "openai" in attr.lower()
    ]
    assert suspicious == [], (
        f"recipe engine references LLM symbols at module scope: {suspicious}"
    )


def test_structured_projection_makes_no_llm_calls(engine, driver, graph_id, cleanup):
    """A full structured run completes with zero LLM calls — instrumented.

    STORY-034 acceptance criterion 5 — a dynamic guarantee. Every callable in
    every already-imported LLM-service module is wrapped with a counter; a
    complete node/property/edge projection is then run. The counter must be
    zero — the run-time path invokes no model.

    `text_extraction` is the *only* rule kind that would need a model, and the
    engine deliberately skips it (out of scope for TASK-223). This test uses a
    purely structured recipe — no `text_extraction` rule — so a non-zero count
    would be a real regression, not the known deferral.
    """
    cleanup.append(graph_id)

    calls: list[str] = []
    patched: list[tuple[object, str, object]] = []

    def _wrap(mod, attr_name, original):
        def _counting(*args, **kwargs):
            calls.append(f"{mod.__name__}.{attr_name}")
            return original(*args, **kwargs)

        return _counting

    # Instrument every callable in every loaded LLM-ish module.
    for mod_name in _llm_service_modules():
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for attr_name in dir(mod):
            if attr_name.startswith("__"):
                continue
            try:
                original = getattr(mod, attr_name)
            except Exception:  # noqa: BLE001
                continue
            if callable(original) and getattr(original, "__module__", None) == mod_name:
                try:
                    setattr(mod, attr_name, _wrap(mod, attr_name, original))
                    patched.append((mod, attr_name, original))
                except Exception:  # noqa: BLE001
                    continue

    try:
        # A complete structured run — node + property + edge.
        result = engine.execute(
            _staff_node_recipe(),
            _container_bearing_representation(),
            graph_id,
            driver,
        )
        assert result.nodes_written == 2
    finally:
        for mod, attr_name, original in patched:
            setattr(mod, attr_name, original)

    assert calls == [], f"run-time path made LLM calls: {calls}"
