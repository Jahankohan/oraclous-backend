"""
TASK-067 — Assessment substrate schema migration tests.

Covers the four DoD acceptance items:

  1. Running the migration on a fresh Neo4j creates every declared constraint
     and index without error.
  2. Re-running the migration is idempotent — no duplicate-constraint errors.
  3. A duplicate-ID insert fails with the expected constraint-violation.
  4. Each declared index is online and the query planner uses it (verified by
     `EXPLAIN` on representative reads).

Pure unit tests (no Neo4j required) live at the top and validate the migration
file structure + the splitter logic. Integration tests use the existing
`neo4j_test_driver` / `clean_test_graph` fixtures from `tests/conftest.py`.

Run unit slice only:
    pytest tests/cypher/test_assessment_migration.py -m "not integration"

Run full suite against a live Neo4j:
    pytest tests/cypher/test_assessment_migration.py -m "integration"
"""

from __future__ import annotations

import pytest

# --- Pure unit-tests: migration file structure -------------------------------
from app.db.assessment_schema_init import (
    _split_statements,
    load_migration_statements,
)

# Expected schema items per TASK-067 §What and ADR-019
EXPECTED_CONSTRAINTS = {
    "assessment_template_id_unique",
    "module_id_unique",
    "subject_id_unique",
    "assessment_run_id_unique",
    "module_run_id_unique",
    "finding_id_unique",
    "source_id_unique",
    "conflict_id_unique",
    "deliverable_id_unique",
    "unresolved_question_id_unique",
    "registry_item_id_unique",
}

EXPECTED_INDEXES = {
    "assessment_run_graph_status_idx",
    "module_run_run_status_idx",
    "module_run_run_wave_idx",
    "finding_run_idx",
    "finding_run_label_confidence_idx",
    "source_url_normalized_idx",
    "unresolved_question_run_status_idx",
    "deliverable_run_kind_idx",
    "registry_item_kind_slug_version_idx",
    "registry_item_owner_visibility_idx",
}


def test_migration_file_exists():
    """The canonical migration file must ship in the repo."""
    statements = load_migration_statements()
    assert statements, "migration file produced zero statements"


def test_migration_declares_all_expected_constraints():
    """Every label called out by TASK-067 §What must have a uniqueness constraint."""
    stmts = load_migration_statements()
    constraint_names = {
        # `CREATE CONSTRAINT <name> IF NOT EXISTS FOR …`
        s.split()[2]
        for s in stmts
        if s.upper().startswith("CREATE CONSTRAINT")
    }
    missing = EXPECTED_CONSTRAINTS - constraint_names
    assert not missing, f"missing constraints: {missing}"


def test_migration_declares_all_expected_indexes():
    """Every read pattern called out by TASK-067 §What must have an index."""
    stmts = load_migration_statements()
    index_names = {
        # `CREATE INDEX <name> IF NOT EXISTS FOR …`
        s.split()[2]
        for s in stmts
        if s.upper().startswith("CREATE INDEX")
    }
    missing = EXPECTED_INDEXES - index_names
    assert not missing, f"missing indexes: {missing}"


def test_migration_is_idempotent_by_construction():
    """Every constraint/index uses `IF NOT EXISTS`; every write uses `MERGE`."""
    stmts = load_migration_statements()
    for stmt in stmts:
        upper = stmt.upper()
        if upper.startswith("CREATE CONSTRAINT") or upper.startswith("CREATE INDEX"):
            assert "IF NOT EXISTS" in upper, (
                f"statement is not idempotent (missing IF NOT EXISTS): {stmt[:80]!r}"
            )
        elif upper.startswith("CREATE "):
            pytest.fail(
                f"raw CREATE (not MERGE, not IF NOT EXISTS) found: {stmt[:80]!r}"
            )


def test_migration_bootstraps_catalog_anchors():
    """The two `__system__` catalog anchors must be MERGEd, not CREATEd."""
    stmts = load_migration_statements()
    catalog_anchors = [s for s in stmts if "__assessments_catalog__" in s]
    registry_anchors = [s for s in stmts if "__registry__" in s]
    assert len(catalog_anchors) == 1, (
        f"expected exactly one __assessments_catalog__ anchor, got {len(catalog_anchors)}"
    )
    assert len(registry_anchors) == 1, (
        f"expected exactly one __registry__ anchor, got {len(registry_anchors)}"
    )
    for stmt in catalog_anchors + registry_anchors:
        # ADR-015 — anchor must carry both :Graph and :__Rebac__ markers
        assert stmt.upper().startswith("MERGE"), (
            f"catalog anchor must be MERGE, not CREATE: {stmt[:80]!r}"
        )
        assert ":__Rebac__" in stmt, (
            f"catalog anchor missing __Rebac__ marker per ADR-015: {stmt[:80]!r}"
        )
        assert "namespace: '__system__'" in stmt or 'namespace: "__system__"' in stmt, (
            f"catalog anchor missing namespace='__system__' per ADR-015: {stmt[:80]!r}"
        )


def test_splitter_handles_comments_and_blank_lines():
    """The splitter must strip `//` comments and blank lines reliably."""
    sample = """
    // a header comment
    CREATE CONSTRAINT foo IF NOT EXISTS
    FOR (x:Foo) REQUIRE x.id IS UNIQUE;

    // another comment
    CREATE INDEX bar_idx IF NOT EXISTS
    FOR (b:Bar) ON (b.k);
    """
    stmts = _split_statements(sample)
    assert len(stmts) == 2
    assert stmts[0].startswith("CREATE CONSTRAINT foo")
    assert stmts[1].startswith("CREATE INDEX bar_idx")
    for s in stmts:
        assert not s.endswith(";"), "splitter must trim trailing ';'"


# --- Integration tests: against a live Neo4j --------------------------------

pytestmark_integration = [pytest.mark.integration, pytest.mark.neo4j]


@pytest.mark.integration
@pytest.mark.neo4j
class TestMigrationAppliesCleanly:
    """DoD: migration runs on fresh Neo4j without error; re-running is idempotent."""

    async def test_first_run_succeeds(self, neo4j_test_driver):
        from app.db.assessment_schema_init import ensure_assessment_schema

        await ensure_assessment_schema(neo4j_test_driver)

    async def test_second_run_is_idempotent(self, neo4j_test_driver):
        from app.db.assessment_schema_init import ensure_assessment_schema

        # Two consecutive runs — must not raise.
        await ensure_assessment_schema(neo4j_test_driver)
        await ensure_assessment_schema(neo4j_test_driver)

    async def test_all_constraints_present_after_run(self, neo4j_test_driver):
        from app.db.assessment_schema_init import ensure_assessment_schema

        await ensure_assessment_schema(neo4j_test_driver)
        async with neo4j_test_driver.session() as session:
            res = await session.run("SHOW CONSTRAINTS YIELD name")
            names = {row["name"] async for row in res}
        missing = EXPECTED_CONSTRAINTS - names
        assert not missing, f"constraints missing from Neo4j after migration: {missing}"

    async def test_all_indexes_present_and_online(self, neo4j_test_driver):
        from app.db.assessment_schema_init import ensure_assessment_schema

        await ensure_assessment_schema(neo4j_test_driver)
        async with neo4j_test_driver.session() as session:
            res = await session.run(
                "SHOW INDEXES YIELD name, state WHERE name IN $names",
                {"names": list(EXPECTED_INDEXES)},
            )
            rows = [row.data() async for row in res]
        found = {row["name"]: row["state"].upper() for row in rows}
        missing = EXPECTED_INDEXES - set(found)
        assert not missing, f"indexes missing after migration: {missing}"
        for name, state in found.items():
            assert state == "ONLINE", f"index {name!r} state={state!r}, expected ONLINE"


@pytest.mark.integration
@pytest.mark.neo4j
class TestCatalogAnchors:
    """DoD: catalog graph_id `__assessments_catalog__` is created/asserted."""

    async def test_assessments_catalog_anchor_created(self, neo4j_test_driver):
        from app.db.assessment_schema_init import ensure_assessment_schema

        await ensure_assessment_schema(neo4j_test_driver)
        async with neo4j_test_driver.session() as session:
            res = await session.run(
                """
                MATCH (g:Graph:__Rebac__ {
                    graph_id: '__assessments_catalog__',
                    namespace: '__system__'
                })
                RETURN g.graph_id AS gid, g.namespace AS ns
                """
            )
            rows = [row.data() async for row in res]
        assert len(rows) == 1, (
            "expected exactly one __assessments_catalog__ anchor, "
            f"got {len(rows)}: {rows}"
        )
        assert rows[0]["gid"] == "__assessments_catalog__"
        assert rows[0]["ns"] == "__system__"

    async def test_registry_anchor_created(self, neo4j_test_driver):
        """ADR-019: the registry catalog anchor must also be present."""
        from app.db.assessment_schema_init import ensure_assessment_schema

        await ensure_assessment_schema(neo4j_test_driver)
        async with neo4j_test_driver.session() as session:
            res = await session.run(
                """
                MATCH (g:Graph:__Rebac__ {
                    graph_id: '__registry__',
                    namespace: '__system__'
                })
                RETURN count(g) AS cnt
                """
            )
            row = (await res.single()).data()
        assert row["cnt"] == 1, "expected exactly one __registry__ anchor"

    async def test_anchors_are_idempotent(self, neo4j_test_driver):
        from app.db.assessment_schema_init import ensure_assessment_schema

        # Run three times — anchor count must stay at 1.
        for _ in range(3):
            await ensure_assessment_schema(neo4j_test_driver)
        async with neo4j_test_driver.session() as session:
            res = await session.run(
                """
                MATCH (g:Graph:__Rebac__ {graph_id: '__assessments_catalog__'})
                RETURN count(g) AS cnt
                """
            )
            cnt = (await res.single()).data()["cnt"]
        assert cnt == 1, f"anchor duplicated after re-runs: count={cnt}"


@pytest.mark.integration
@pytest.mark.neo4j
class TestConstraintEnforcement:
    """DoD: duplicate-ID inserts fail with the expected constraint violation."""

    @pytest.fixture(autouse=True)
    async def _setup_and_cleanup(self, neo4j_test_driver):
        from app.db.assessment_schema_init import ensure_assessment_schema

        await ensure_assessment_schema(neo4j_test_driver)
        yield
        # Cleanup — only nodes inserted by this suite, identified by a sentinel
        # graph_id so we never touch real data.
        async with neo4j_test_driver.session() as session:
            await session.run(
                "MATCH (n {graph_id: 'test-task067-constraint'}) DETACH DELETE n"
            )

    async def test_duplicate_assessment_run_id_rejected(self, neo4j_test_driver):
        """Inserting two :AssessmentRun nodes with the same run_id must fail."""
        from neo4j.exceptions import ClientError

        run_id = "test-task067-run-001"
        async with neo4j_test_driver.session() as session:
            await session.run(
                """
                CREATE (r:AssessmentRun:__Platform__ {
                    run_id: $run_id,
                    graph_id: 'test-task067-constraint',
                    status: 'planned'
                })
                """,
                {"run_id": run_id},
            )
            with pytest.raises(ClientError) as exc_info:
                result = await session.run(
                    """
                    CREATE (r:AssessmentRun:__Platform__ {
                        run_id: $run_id,
                        graph_id: 'test-task067-constraint',
                        status: 'planned'
                    })
                    """,
                    {"run_id": run_id},
                )
                await result.consume()
            assert "ConstraintValidationFailed" in str(exc_info.value) or (
                "already exists" in str(exc_info.value).lower()
            )

    async def test_duplicate_finding_id_rejected(self, neo4j_test_driver):
        from neo4j.exceptions import ClientError

        finding_id = "test-task067-finding-001"
        async with neo4j_test_driver.session() as session:
            await session.run(
                """
                CREATE (f:Finding:__Platform__ {
                    finding_id: $fid,
                    run_id: 'r1',
                    graph_id: 'test-task067-constraint'
                })
                """,
                {"fid": finding_id},
            )
            with pytest.raises(ClientError):
                result = await session.run(
                    """
                    CREATE (f:Finding:__Platform__ {
                        finding_id: $fid,
                        run_id: 'r2',
                        graph_id: 'test-task067-constraint'
                    })
                    """,
                    {"fid": finding_id},
                )
                await result.consume()

    async def test_duplicate_registry_item_id_rejected(self, neo4j_test_driver):
        """ADR-019: :RegistryItem.item_id is the uniqueness key."""
        from neo4j.exceptions import ClientError

        item_id = "test-task067-item-001"
        async with neo4j_test_driver.session() as session:
            await session.run(
                """
                CREATE (i:RegistryItem:__Platform__ {
                    item_id: $iid,
                    kind: 'skill',
                    slug: 'foo',
                    version: '1.0.0',
                    visibility: 'private',
                    owner_user_id: 'u-test',
                    graph_id: 'test-task067-constraint'
                })
                """,
                {"iid": item_id},
            )
            with pytest.raises(ClientError):
                result = await session.run(
                    """
                    CREATE (i:RegistryItem:__Platform__ {
                        item_id: $iid,
                        kind: 'skill',
                        slug: 'bar',
                        version: '1.0.0',
                        visibility: 'private',
                        owner_user_id: 'u-test',
                        graph_id: 'test-task067-constraint'
                    })
                    """,
                    {"iid": item_id},
                )
                await result.consume()


@pytest.mark.integration
@pytest.mark.neo4j
class TestIndexUsedByPlanner:
    """DoD: representative queries' EXPLAIN plans show the new indexes in use."""

    @pytest.fixture(autouse=True)
    async def _migrate(self, neo4j_test_driver):
        from app.db.assessment_schema_init import ensure_assessment_schema

        await ensure_assessment_schema(neo4j_test_driver)

    async def test_assessment_run_status_query_uses_index(self, neo4j_test_driver):
        async with neo4j_test_driver.session() as session:
            res = await session.run(
                """
                EXPLAIN
                MATCH (r:AssessmentRun)
                WHERE r.graph_id = $gid AND r.status = $status
                RETURN r
                """,
                {"gid": "test-task067-explain", "status": "running"},
            )
            summary = await res.consume()
        plan_str = str(summary.plan or summary).lower()
        assert (
            "indexseek" in plan_str
            or "nodeindexseek" in plan_str
            or "nodeindex" in plan_str
        ), (
            f"AssessmentRun (graph_id, status) query did not use an index: {plan_str[:400]}"
        )

    async def test_module_run_wave_query_uses_index(self, neo4j_test_driver):
        async with neo4j_test_driver.session() as session:
            res = await session.run(
                """
                EXPLAIN
                MATCH (mr:ModuleRun)
                WHERE mr.run_id = $rid AND mr.wave = $wave
                RETURN mr
                """,
                {"rid": "test-task067-explain", "wave": 1},
            )
            summary = await res.consume()
        plan_str = str(summary.plan or summary).lower()
        assert (
            "indexseek" in plan_str
            or "nodeindexseek" in plan_str
            or "nodeindex" in plan_str
        ), f"ModuleRun (run_id, wave) query did not use an index: {plan_str[:400]}"

    async def test_source_url_normalized_query_uses_index(self, neo4j_test_driver):
        async with neo4j_test_driver.session() as session:
            res = await session.run(
                """
                EXPLAIN
                MATCH (s:Source)
                WHERE s.url_normalized = $url
                RETURN s
                """,
                {"url": "https://example.com/foo"},
            )
            summary = await res.consume()
        plan_str = str(summary.plan or summary).lower()
        assert (
            "indexseek" in plan_str
            or "nodeindexseek" in plan_str
            or "nodeindex" in plan_str
        ), f"Source url_normalized query did not use an index: {plan_str[:400]}"

    async def test_registry_item_lookup_uses_index(self, neo4j_test_driver):
        """ADR-019: (kind, slug, version) is the resolver key."""
        async with neo4j_test_driver.session() as session:
            res = await session.run(
                """
                EXPLAIN
                MATCH (ri:RegistryItem)
                WHERE ri.kind = $k AND ri.slug = $s AND ri.version = $v
                RETURN ri
                """,
                {"k": "skill", "s": "assess", "v": "1.0.0"},
            )
            summary = await res.consume()
        plan_str = str(summary.plan or summary).lower()
        assert (
            "indexseek" in plan_str
            or "nodeindexseek" in plan_str
            or "nodeindex" in plan_str
        ), (
            f"RegistryItem (kind, slug, version) lookup did not use an index: {plan_str[:400]}"
        )
