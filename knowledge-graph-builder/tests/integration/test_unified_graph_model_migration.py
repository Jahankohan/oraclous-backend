"""Integration test — the unified graph model migration (TASK-221, STORY-034).

Runs against the **live Dockerized Neo4j** (the by-dependency testing policy:
this test has a real runtime dependency, so it validates against the live
stack). It applies the migration, asserts its constraints exist, re-applies it,
and asserts the re-apply is idempotent.

Connection defaults match `docker-compose.yml` (`neo4j`/`password`,
`bolt://localhost:7687`) and can be overridden by `NEO4J_TEST_*` env vars.
"""

import os
from pathlib import Path

import pytest
from neo4j import GraphDatabase

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "cypher"
    / "migrations"
    / "2026-05-19_unified_graph_model.cypher"
)
_EXPECTED_CONSTRAINTS = {
    "ugm_source_key",
    "ugm_table_key",
    "ugm_sheet_key",
    "ugm_file_key",
    "ugm_chunk_key",
    "ugm_entity_key",
}
_URI = os.environ.get("NEO4J_TEST_URI", "bolt://localhost:7687")
_USER = os.environ.get("NEO4J_TEST_USER", "neo4j")
_PASSWORD = os.environ.get("NEO4J_TEST_PASSWORD", "password")


def _statements(path: Path) -> list[str]:
    """Split a .cypher migration into statements, dropping // comment lines."""
    body = "\n".join(
        line
        for line in path.read_text().splitlines()
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
    yield d
    d.close()


def _apply_migration(driver) -> None:
    with driver.session() as session:
        for stmt in _statements(_MIGRATION):
            session.run(stmt)


def _ugm_constraints(driver) -> set[str]:
    with driver.session() as session:
        result = session.run(
            "SHOW CONSTRAINTS YIELD name WHERE name STARTS WITH 'ugm_' RETURN name"
        )
        return {record["name"] for record in result}


def test_migration_file_is_non_empty():
    assert _statements(_MIGRATION), "migration file has no executable statements"


def test_migration_creates_unified_model_constraints(driver):
    _apply_migration(driver)
    assert _ugm_constraints(driver) == _EXPECTED_CONSTRAINTS


def test_migration_is_idempotent(driver):
    _apply_migration(driver)
    _apply_migration(driver)  # second apply must not raise
    assert _ugm_constraints(driver) == _EXPECTED_CONSTRAINTS
