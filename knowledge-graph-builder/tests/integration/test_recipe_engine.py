"""Integration test — the recipe execution engine (TASK-223, STORY-034).

The engine writes to Neo4j, so these are **integration tests** against the live
Dockerized Neo4j (the by-dependency testing policy). Each test:

  * uses a fresh `uuid4` test `graph_id` — never real data;
  * runs `RecipeExecutionEngine.execute` with an example recipe plus a
    hand-built `StructuralRepresentation`;
  * asserts the resulting nodes / edges via Cypher — labels, provenance fields,
    `graph_id` scoping;
  * re-runs `execute` and asserts idempotency — no duplicate nodes or edges;
  * deletes its test-`graph_id` nodes in teardown.

Connection defaults match `docker-compose.yml` (`neo4j`/`password`,
`bolt://localhost:7687`); override with `NEO4J_TEST_*` env vars.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest
from neo4j import GraphDatabase

from app.recipes.engine import RecipeExecutionEngine, RecipeValidationError
from app.recipes.primitives.interface import (
    ExtractionMode,
    StructuralRepresentation,
    StructuralUnit,
    UnitKind,
)

_URI = os.environ.get("NEO4J_TEST_URI", "bolt://localhost:7687")
_USER = os.environ.get("NEO4J_TEST_USER", "neo4j")
_PASSWORD = os.environ.get("NEO4J_TEST_PASSWORD", "password")

_EXAMPLES = Path(__file__).resolve().parents[2] / "app" / "recipes" / "examples"
_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "cypher"
    / "migrations"
    / "2026-05-19_unified_graph_model.cypher"
)


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
    try:
        d = GraphDatabase.driver(_URI, auth=(_USER, _PASSWORD))
        d.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"live Neo4j not reachable at {_URI}: {exc}")
    # The engine MERGEs against the unified-graph-model constraints — make sure
    # they exist (idempotent — a no-op if the migration already ran).
    with d.session() as session:
        for stmt in _migration_statements():
            session.run(stmt)
    yield d
    d.close()


@pytest.fixture
def graph_id():
    """A fresh, isolated test graph_id per test."""
    return f"test-recipe-{uuid.uuid4()}"


@pytest.fixture
def cleanup(driver):
    """Delete every node carrying the test graph_id after the test."""
    ids: list[str] = []
    yield ids
    with driver.session() as session:
        for gid in ids:
            session.run("MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid})


@pytest.fixture
def engine():
    return RecipeExecutionEngine()


def _load_recipe(name: str) -> dict:
    return json.loads((_EXAMPLES / name).read_text())


# ---------------------------------------------------------------------------
# Hand-built representations
# ---------------------------------------------------------------------------


def _eurail_representation() -> StructuralRepresentation:
    """A JSON-shaped representation for the eurail free-text recipe.

    Two findings (record units) plus the source and field units the JSON
    primitive emits. The `raw` field is the free-text the recipe's
    text_extraction rule targets — out of scope for TASK-223.
    """
    units = [
        StructuralUnit(
            kind=UnitKind.SOURCE,
            unit_id="source",
            name="findings.json",
            metadata={"record_count": 2},
        ),
        StructuralUnit(
            kind=UnitKind.FIELD,
            unit_id="field:finding_id",
            name="finding_id",
            data_type="string",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.FIELD,
            unit_id="field:source_id",
            name="source_id",
            data_type="string",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.FIELD,
            unit_id="field:raw",
            name="raw",
            data_type="string",
            role="free_text",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.RECORD,
            unit_id="record:0",
            name="record 0",
            parent_id="source",
            sample_values=[
                {
                    "finding_id": "F-001",
                    "source_id": "S-100",
                    "raw": "SBB runs ERTMS on the Gotthard line.",
                }
            ],
        ),
        StructuralUnit(
            kind=UnitKind.RECORD,
            unit_id="record:1",
            name="record 1",
            parent_id="source",
            sample_values=[
                {
                    "finding_id": "F-002",
                    "source_id": "S-101",
                    "raw": "ÖBB adopted OSDM for ticketing.",
                }
            ],
        ),
    ]
    return StructuralRepresentation(
        source_type="json",
        shape_signature="findings[]{finding_id,source_id,raw}",
        mode=ExtractionMode.FULL,
        units=units,
    )


def _org_chart_csv_recipe() -> dict:
    """A CSV-shaped org-chart recipe.

    The shipped org-chart example matches `node` on `unit_kind: table` — the
    relational primitive emits no row units, so it materializes no entities on
    its own. This variant matches `node`/`property`/`edge` on the `record`
    units a CSV/JSON primitive emits, which is the path the engine projects
    entities over. It exercises node, property, and an `identity`-resolved edge.
    """
    return {
        "recipe_format_version": "0.2",
        "id": "rcp_org-chart-csv-test-v1",
        "version": 1,
        "status": "promoted",
        "concern": "Reporting structure from a flat employee export.",
        "applies_to": {
            "source_type": "csv",
            "shape_signature": "employees.csv:name,title",
        },
        "defaults": {"provenance": "EXTRACTED", "materialize_fine_grain": True},
        "mappings": [
            {
                "id": "employees",
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
                "id": "employee_title",
                "match": {"unit_kind": "column", "name": "title"},
                "project_to": "property",
                "on": "employees",
                "name": "title",
                "value_from": "column:title",
            },
        ],
    }


def _org_chart_csv_representation() -> StructuralRepresentation:
    """A CSV representation: 3 employee rows, columns name/title."""
    units = [
        StructuralUnit(
            kind=UnitKind.SOURCE,
            unit_id="source",
            name="employees.csv",
            metadata={"row_count": 3, "column_count": 2},
        ),
        StructuralUnit(
            kind=UnitKind.COLUMN,
            unit_id="column:name",
            name="name",
            data_type="str",
            role="free_text",
            parent_id="source",
        ),
        StructuralUnit(
            kind=UnitKind.COLUMN,
            unit_id="column:title",
            name="title",
            data_type="str",
            role="free_text",
            parent_id="source",
        ),
    ]
    rows = [
        {"name": "  Ada  Lovelace ", "title": "CTO"},
        {"name": "Alan Turing", "title": "Principal Engineer"},
        {"name": "Grace Hopper", "title": "VP Engineering"},
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
        shape_signature="employees.csv:name,title",
        mode=ExtractionMode.FULL,
        units=units,
    )


# ---------------------------------------------------------------------------
# Recipe validation
# ---------------------------------------------------------------------------


def test_invalid_recipe_is_rejected(engine):
    """A recipe missing required fields is refused, not coerced."""
    with pytest.raises(RecipeValidationError):
        engine._validate_recipe({"recipe_format_version": "0.2"})


def test_unsafe_label_recipe_is_rejected(engine):
    """A node rule with a Cypher-unsafe domain label is refused (recipe §5.5)."""
    recipe = _org_chart_csv_recipe()
    recipe["mappings"][0]["label"] = "Employee`) DETACH DELETE n //"
    with pytest.raises(RecipeValidationError):
        engine._validate_recipe(recipe)


def test_reserved_namespace_label_is_rejected(engine):
    """A node rule declaring a reserved __wrapped__ label is refused (ADR-015)."""
    recipe = _org_chart_csv_recipe()
    recipe["mappings"][0]["label"] = "__Entity__"
    with pytest.raises(RecipeValidationError):
        engine._validate_recipe(recipe)


def test_container_label_collision_is_rejected(engine):
    """A node rule declaring a platform container label is refused."""
    recipe = _org_chart_csv_recipe()
    recipe["mappings"][0]["label"] = "Table"
    with pytest.raises(RecipeValidationError):
        engine._validate_recipe(recipe)


def test_example_recipes_validate(engine):
    """Both shipped example recipes validate against the schema."""
    engine._validate_recipe(_load_recipe("org-chart-relational.recipe.json"))
    engine._validate_recipe(_load_recipe("eurail-evidence-freetext.recipe.json"))


# ---------------------------------------------------------------------------
# Structure materialization + node projection
# ---------------------------------------------------------------------------


def test_execute_materializes_source_and_entities(engine, driver, graph_id, cleanup):
    """A run writes a :Source node and one :__Entity__ per record."""
    cleanup.append(graph_id)
    recipe = _org_chart_csv_recipe()
    rep = _org_chart_csv_representation()

    result = engine.execute(recipe, rep, graph_id, driver)

    assert result.nodes_written == 3
    assert result.containers_written == 0  # CSV has no container-kind units

    with driver.session() as session:
        src = session.run(
            "MATCH (s:Source {graph_id: $gid}) RETURN count(s) AS c",
            {"gid": graph_id},
        ).single()["c"]
        assert src == 1

        ents = session.run(
            """
            MATCH (e:Employee:__Entity__ {graph_id: $gid})
            RETURN e.name AS name, e.provenance AS prov,
                   e.recipe_id AS rid, e.ingestion_source AS isrc,
                   e.ingestion_time AS itime
            ORDER BY name
            """,
            {"gid": graph_id},
        ).data()
    assert len(ents) == 3
    for e in ents:
        assert e["prov"] == "EXTRACTED"
        assert e["rid"] == "rcp_org-chart-csv-test-v1"
        assert e["isrc"] == result.source_id
        assert e["itime"] is not None
    # "  Ada  Lovelace " property value is preserved verbatim (normalization
    # only affects identity, never the stored property).
    assert "  Ada  Lovelace " in {e["name"] for e in ents}


def test_entities_link_back_to_source_via_derived_from(
    engine, driver, graph_id, cleanup
):
    """Every entity carries a DERIVED_FROM edge to its source/container."""
    cleanup.append(graph_id)
    engine.execute(
        _org_chart_csv_recipe(), _org_chart_csv_representation(), graph_id, driver
    )

    with driver.session() as session:
        cnt = session.run(
            """
            MATCH (e:Employee:__Entity__ {graph_id: $gid})
                  -[:DERIVED_FROM]->(s:Source {graph_id: $gid})
            RETURN count(e) AS c
            """,
            {"gid": graph_id},
        ).single()["c"]
    assert cnt == 3


def test_property_rule_sets_property_on_node(engine, driver, graph_id, cleanup):
    """A `property` rule SETs its value on the `on` node-rule's entities."""
    cleanup.append(graph_id)
    engine.execute(
        _org_chart_csv_recipe(), _org_chart_csv_representation(), graph_id, driver
    )

    with driver.session() as session:
        titles = session.run(
            """
            MATCH (e:Employee:__Entity__ {graph_id: $gid})
            RETURN e.name AS name, e.title AS title ORDER BY name
            """,
            {"gid": graph_id},
        ).data()
    by_title = {t["title"] for t in titles}
    assert by_title == {"CTO", "Principal Engineer", "VP Engineering"}


def test_graph_id_scoping_isolates_tenants(engine, driver, cleanup):
    """Two graph_ids running the same recipe do not see each other's nodes."""
    gid_a = f"test-recipe-{uuid.uuid4()}"
    gid_b = f"test-recipe-{uuid.uuid4()}"
    cleanup.extend([gid_a, gid_b])

    engine.execute(
        _org_chart_csv_recipe(), _org_chart_csv_representation(), gid_a, driver
    )
    engine.execute(
        _org_chart_csv_recipe(), _org_chart_csv_representation(), gid_b, driver
    )

    with driver.session() as session:
        a = session.run(
            "MATCH (e:Employee {graph_id: $gid}) RETURN count(e) AS c",
            {"gid": gid_a},
        ).single()["c"]
        b = session.run(
            "MATCH (e:Employee {graph_id: $gid}) RETURN count(e) AS c",
            {"gid": gid_b},
        ).single()["c"]
        # An identity hash includes graph_id — so the two tenants' nodes are
        # distinct even though the identity *keys* are identical.
        cross = session.run(
            """
            MATCH (ea:Employee {graph_id: $a}), (eb:Employee {graph_id: $b})
            WHERE ea.id = eb.id RETURN count(*) AS c
            """,
            {"a": gid_a, "b": gid_b},
        ).single()["c"]
    assert a == 3 and b == 3
    assert cross == 0


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_execute_is_idempotent(engine, driver, graph_id, cleanup):
    """Re-running execute yields the same graph — no duplicate nodes/edges."""
    cleanup.append(graph_id)
    recipe = _org_chart_csv_recipe()
    rep = _org_chart_csv_representation()

    first = engine.execute(recipe, rep, graph_id, driver)
    second = engine.execute(recipe, rep, graph_id, driver)

    assert first.nodes_written == second.nodes_written == 3

    with driver.session() as session:
        ents = session.run(
            "MATCH (e:Employee:__Entity__ {graph_id: $gid}) RETURN count(e) AS c",
            {"gid": graph_id},
        ).single()["c"]
        srcs = session.run(
            "MATCH (s:Source {graph_id: $gid}) RETURN count(s) AS c",
            {"gid": graph_id},
        ).single()["c"]
        derived = session.run(
            """
            MATCH (:Employee {graph_id: $gid})-[r:DERIVED_FROM]->()
            RETURN count(r) AS c
            """,
            {"gid": graph_id},
        ).single()["c"]
    # Two runs, still exactly 3 employees / 1 source / 3 DERIVED_FROM edges.
    assert ents == 3
    assert srcs == 1
    assert derived == 3


# ---------------------------------------------------------------------------
# Containers + PART_OF
# ---------------------------------------------------------------------------


def test_container_units_become_container_nodes(engine, driver, graph_id, cleanup):
    """A `table` / `chunk` unit becomes a container node wired PART_OF the source."""
    cleanup.append(graph_id)
    units = [
        StructuralUnit(kind=UnitKind.SOURCE, unit_id="source", name="repo"),
        StructuralUnit(
            kind=UnitKind.FILE, unit_id="file:a.py", name="a.py", parent_id="source"
        ),
        StructuralUnit(
            kind=UnitKind.TABLE,
            unit_id="table:employees",
            name="employees",
            parent_id="source",
        ),
    ]
    rep = StructuralRepresentation(
        source_type="code",
        shape_signature="repo(a.py;employees)",
        mode=ExtractionMode.FULL,
        units=units,
    )
    # A minimal recipe with a single skip rule — valid, projects no entities.
    recipe = {
        "recipe_format_version": "0.2",
        "id": "rcp_container-only-test-v1",
        "version": 1,
        "status": "promoted",
        "concern": "Structure only.",
        "applies_to": {"source_type": "code", "shape_signature": "x"},
        "mappings": [
            {
                "id": "skip_all",
                "match": {"unit_kind": "source"},
                "project_to": "skip",
                "reason": "structure-only test",
            }
        ],
    }
    result = engine.execute(recipe, rep, graph_id, driver)
    assert result.containers_written == 2

    with driver.session() as session:
        part_of = session.run(
            """
            MATCH (c:__KGBuilder__ {graph_id: $gid})-[:PART_OF]->(s:Source {graph_id: $gid})
            RETURN count(c) AS c
            """,
            {"gid": graph_id},
        ).single()["c"]
        files = session.run(
            "MATCH (f:File {graph_id: $gid}) RETURN count(f) AS c",
            {"gid": graph_id},
        ).single()["c"]
        tables = session.run(
            "MATCH (t:Table {graph_id: $gid}) RETURN count(t) AS c",
            {"gid": graph_id},
        ).single()["c"]
    assert part_of == 2
    assert files == 1
    assert tables == 1


# ---------------------------------------------------------------------------
# text_extraction — out of scope for TASK-223
# ---------------------------------------------------------------------------


def test_text_extraction_rule_is_skipped_with_warning(
    engine, driver, graph_id, cleanup
):
    """A text_extraction rule is recognised, skipped, and reported — no LLM."""
    cleanup.append(graph_id)
    recipe = _load_recipe("eurail-evidence-freetext.recipe.json")
    rep = _eurail_representation()

    result = engine.execute(recipe, rep, graph_id, driver)

    # The `finding` node rule still runs deterministically — 2 records.
    assert result.nodes_written == 2
    # The text_extraction rule matched the free-text `raw` field and was
    # skipped with a warning rather than invoking a model.
    assert any("text_extraction" in w for w in result.warnings)

    with driver.session() as session:
        findings = session.run(
            """
            MATCH (f:Finding:__Entity__ {graph_id: $gid})
            RETURN f.sourceId AS sid ORDER BY sid
            """,
            {"gid": graph_id},
        ).data()
        # No INFERRED entities — the LLM path did not run.
        inferred = session.run(
            """
            MATCH (e:__Entity__ {graph_id: $gid, provenance: 'INFERRED'})
            RETURN count(e) AS c
            """,
            {"gid": graph_id},
        ).single()["c"]
    assert [f["sid"] for f in findings] == ["S-100", "S-101"]
    assert inferred == 0
