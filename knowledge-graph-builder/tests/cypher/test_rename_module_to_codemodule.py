"""
TASK-075 — Tests for the `:Module` → `:CodeModule` rename migration.

Covers the DoD acceptance items:

  1. The migration file exists and parses cleanly through the splitter.
  2. Statement structure: drops the colliding constraint, drops the colliding
     fulltext index, promotes `:Module` rows owned by the code-parser to
     `:CodeModule`, then re-declares the constraint and fulltext index.
  3. Integration: on a Neo4j containing pre-existing code-parser `:Module`
     rows AND assessment-substrate `:Module:__Platform__` rows, the migration
     promotes only the code-parser rows. The count of `:CodeModule` after
     equals the count of code-parser `:Module` before; the count of
     `:Module:__Platform__` is unchanged.
  4. Idempotency: running `ensure_code_schema` twice in a row leaves the
     database in the same state and does not raise.
  5. Regression: no `:Module` rows remain on the code-parser side; no
     `:CodeModule` rows are accidentally added to the assessment side.

Run unit slice only:
    pytest tests/cypher/test_rename_module_to_codemodule.py -m "not integration"

Run full suite against a live Neo4j:
    pytest tests/cypher/test_rename_module_to_codemodule.py -m "integration"
"""

from __future__ import annotations

import uuid

import pytest

# --- Pure unit tests: migration file structure -------------------------------
from app.services.code_parser_service import (
    _RENAME_MIGRATION_PATH,
    _load_rename_migration_statements,
    _split_cypher_statements,
)


def test_migration_file_exists_on_disk():
    """The rename migration `.cypher` file must ship in the repo."""
    assert _RENAME_MIGRATION_PATH.is_file(), (
        f"rename migration not present at {_RENAME_MIGRATION_PATH}"
    )


def test_migration_parses_to_nonempty_statement_list():
    statements = _load_rename_migration_statements()
    assert statements, "rename migration produced zero statements"


def test_migration_declares_expected_statements():
    """The migration must drop the colliding constraint+index, promote labels,
    and re-declare them on `:CodeModule`."""
    stmts = _load_rename_migration_statements()
    upper = [s.upper() for s in stmts]
    # 1. Drop old constraint
    assert any(
        u.startswith("DROP CONSTRAINT MODULE_UNIQUE") and "IF EXISTS" in u
        for u in upper
    ), "missing `DROP CONSTRAINT module_unique IF EXISTS`"
    # 2. Drop old fulltext index
    assert any(
        u.startswith("DROP INDEX CODE_SYMBOL_SEARCH") and "IF EXISTS" in u
        for u in upper
    ), "missing `DROP INDEX code_symbol_search IF EXISTS`"
    # 3. Label promotion — explicit SET …:CodeModule REMOVE …:Module
    assert any("SET M:CODEMODULE" in u and "REMOVE M:MODULE" in u for u in upper), (
        "missing `SET m:CodeModule REMOVE m:Module` statement"
    )
    # 4. Re-declare constraint on `:CodeModule`
    assert any(
        u.startswith("CREATE CONSTRAINT CODE_MODULE_UNIQUE")
        and "(M:CODEMODULE)" in u.replace(" ", "")
        and "IF NOT EXISTS" in u
        for u in upper
    ), (
        "missing `CREATE CONSTRAINT code_module_unique IF NOT EXISTS FOR (m:CodeModule) …`"
    )
    # 5. Re-declare fulltext index referencing `:CodeModule` (not `:Module`)
    fulltext_stmts = [u for u in upper if "FULLTEXT INDEX CODE_SYMBOL_SEARCH" in u]
    assert fulltext_stmts, "missing `CREATE FULLTEXT INDEX code_symbol_search …`"
    for ft in fulltext_stmts:
        assert "CODEMODULE" in ft, (
            "fulltext index must reference :CodeModule after rename"
        )
        # The label list must NOT include bare `:Module` anymore.
        # Tolerate `:CodeModule` matches by checking for `|MODULE|` or
        # trailing `|MODULE` patterns specifically.
        assert "|MODULE|" not in ft and not ft.rstrip(")]").endswith("|MODULE"), (
            "fulltext index still references the old `:Module` label"
        )


def test_migration_label_promotion_filters_out_assessment_modules():
    """The label-promotion `MATCH` must scope to non-`:__Platform__` rows so
    the assessment substrate's `:Module:__Platform__` nodes are untouched."""
    # Use the parsed (post-comment-strip) statement list so we never
    # false-match on prose in `//` comment blocks.
    stmts = _load_rename_migration_statements()
    promotion_stmts = [
        s for s in stmts if "SET m:CodeModule" in s and "REMOVE m:Module" in s
    ]
    assert len(promotion_stmts) == 1, (
        f"expected exactly one promotion statement, got {len(promotion_stmts)}"
    )
    promotion = promotion_stmts[0]
    assert "__Platform__" in promotion, (
        "promotion statement must filter on `__Platform__` so it skips "
        f"assessment-substrate Module nodes — got:\n{promotion}"
    )
    # Filter polarity must be negative — we want NOT m:__Platform__.
    assert "NOT m:__Platform__" in promotion or "NOT (m:__Platform__)" in promotion, (
        "promotion statement must use `NOT m:__Platform__` to skip "
        f"assessment Modules — got:\n{promotion}"
    )


def test_migration_is_idempotent_by_construction():
    """Every constraint/index drop uses `IF EXISTS`; every create uses
    `IF NOT EXISTS`. The label-promotion MATCH naturally idempotent because
    re-running finds zero matches on the code-parser-but-not-platform set
    after the first pass."""
    stmts = _load_rename_migration_statements()
    for stmt in stmts:
        upper = stmt.upper()
        if upper.startswith("DROP CONSTRAINT") or upper.startswith("DROP INDEX"):
            assert "IF EXISTS" in upper, (
                f"DROP not idempotent (missing IF EXISTS): {stmt[:80]!r}"
            )
        elif upper.startswith("CREATE CONSTRAINT") or upper.startswith(
            "CREATE FULLTEXT INDEX"
        ):
            assert "IF NOT EXISTS" in upper, (
                f"CREATE not idempotent (missing IF NOT EXISTS): {stmt[:80]!r}"
            )


def test_splitter_handles_comments_and_blank_lines():
    """The shared `_split_cypher_statements` helper mirrors
    `assessment_schema_init._split_statements` — exercise the same edge
    cases here so a future drift between the two breaks loudly."""
    sample = """
    // a header comment
    DROP CONSTRAINT foo IF EXISTS;

    // another comment
    CREATE CONSTRAINT bar IF NOT EXISTS
    FOR (b:Bar) REQUIRE b.id IS UNIQUE;
    """
    stmts = _split_cypher_statements(sample)
    assert len(stmts) == 2
    assert stmts[0].startswith("DROP CONSTRAINT foo")
    assert stmts[1].startswith("CREATE CONSTRAINT bar")
    for s in stmts:
        assert not s.endswith(";"), "splitter must trim trailing ';'"


# --- Integration tests: against a live Neo4j --------------------------------


@pytest.mark.integration
@pytest.mark.neo4j
class TestRenameMigrationAppliesCleanly:
    """The migration runs on a fresh Neo4j without error; re-running is a no-op."""

    async def test_first_run_succeeds(self, neo4j_test_driver):
        from app.services.code_parser_service import ensure_code_schema

        await ensure_code_schema(neo4j_test_driver)

    async def test_second_run_is_idempotent(self, neo4j_test_driver):
        from app.services.code_parser_service import ensure_code_schema

        await ensure_code_schema(neo4j_test_driver)
        await ensure_code_schema(neo4j_test_driver)

    async def test_code_module_constraint_present_after_run(self, neo4j_test_driver):
        from app.services.code_parser_service import ensure_code_schema

        await ensure_code_schema(neo4j_test_driver)
        async with neo4j_test_driver.session() as session:
            res = await session.run("SHOW CONSTRAINTS YIELD name")
            names = {row["name"] async for row in res}
        assert "code_module_unique" in names, (
            f"code_module_unique constraint missing after migration: {sorted(names)}"
        )
        # The old constraint must be gone.
        assert "module_unique" not in names, (
            "old `module_unique` constraint still present — rename did not drop it"
        )


@pytest.mark.integration
@pytest.mark.neo4j
class TestRenameDoesNotTouchAssessmentModules:
    """The migration must promote ONLY code-parser-owned `:Module` rows; the
    assessment substrate's `:Module:__Platform__` rows must be untouched."""

    @pytest.fixture(autouse=True)
    async def _setup_and_cleanup(self, neo4j_test_driver):
        # Use a sentinel graph_id so the test never touches real data.
        self._sentinel_tenant = f"test-task075-tenant-{uuid.uuid4().hex[:8]}"
        self._sentinel_catalog = f"test-task075-catalog-{uuid.uuid4().hex[:8]}"

        async def _wipe():
            async with neo4j_test_driver.session() as session:
                await session.run(
                    "MATCH (n {graph_id: $gid}) DETACH DELETE n",
                    {"gid": self._sentinel_tenant},
                )
                await session.run(
                    "MATCH (n {graph_id: $gid}) DETACH DELETE n",
                    {"gid": self._sentinel_catalog},
                )

        await _wipe()
        yield
        await _wipe()

    async def test_only_code_parser_modules_get_promoted(self, neo4j_test_driver):
        from app.services.code_parser_service import ensure_code_schema

        # Seed BOTH kinds of pre-existing data BEFORE the migration runs:
        #   1. Three "old code-parser style" `:Module` nodes (no __Platform__),
        #      in a tenant graph — these should be promoted to `:CodeModule`.
        #   2. Two "assessment substrate" `:Module:__Platform__` nodes in a
        #      catalog graph — these must stay as `:Module:__Platform__`.
        #
        # To seed nodes carrying the legacy `:Module` label without the new
        # constraint conflicting, we drop any existing `module_unique`
        # constraint up-front. The migration itself does this as its first
        # statement; doing it explicitly in the test just makes the seed
        # deterministic regardless of prior test state.
        async with neo4j_test_driver.session() as session:
            await session.run("DROP CONSTRAINT module_unique IF EXISTS")
            # Old code-parser data
            for name in ("pkg.mod_a", "pkg.mod_b", "pkg.mod_c"):
                await session.run(
                    """
                    CREATE (m:Module {
                        graph_id: $gid,
                        name: $name,
                        language: 'python'
                    })
                    """,
                    {"gid": self._sentinel_tenant, "name": name},
                )
            # Assessment-substrate data
            for i in range(2):
                await session.run(
                    """
                    CREATE (m:Module:__Platform__ {
                        graph_id: $gid,
                        module_id: $mid,
                        slug: 'sentinel-slug',
                        name: 'Sentinel Module',
                        template_id: 'test-task075',
                        wave: 1,
                        ordinal: 1,
                        kind: 'research'
                    })
                    """,
                    {"gid": self._sentinel_catalog, "mid": f"int-test-075-{i}"},
                )

        # Run the migration.
        await ensure_code_schema(neo4j_test_driver)

        async with neo4j_test_driver.session() as session:
            # 1. All three code-parser modules are now :CodeModule (not :Module).
            res = await session.run(
                """
                MATCH (m:CodeModule {graph_id: $gid})
                RETURN count(m) AS c
                """,
                {"gid": self._sentinel_tenant},
            )
            row = await res.single()
            assert row["c"] == 3, (
                f"expected 3 :CodeModule rows for {self._sentinel_tenant}, got {row['c']}"
            )

            # No code-parser rows still carry the old `:Module` label.
            res = await session.run(
                """
                MATCH (m:Module {graph_id: $gid})
                RETURN count(m) AS c
                """,
                {"gid": self._sentinel_tenant},
            )
            row = await res.single()
            assert row["c"] == 0, (
                f"expected 0 leftover :Module rows for {self._sentinel_tenant}, got {row['c']}"
            )

            # 2. Assessment-substrate modules are unchanged.
            res = await session.run(
                """
                MATCH (m:Module:__Platform__ {graph_id: $gid})
                RETURN count(m) AS c
                """,
                {"gid": self._sentinel_catalog},
            )
            row = await res.single()
            assert row["c"] == 2, (
                f"expected 2 :Module:__Platform__ rows in catalog, got {row['c']} "
                "— migration leaked into the assessment substrate"
            )

            # No assessment-substrate module was accidentally promoted.
            res = await session.run(
                """
                MATCH (m:CodeModule {graph_id: $gid})
                RETURN count(m) AS c
                """,
                {"gid": self._sentinel_catalog},
            )
            row = await res.single()
            assert row["c"] == 0, (
                f"expected 0 :CodeModule rows in catalog graph, got {row['c']} "
                "— assessment module was incorrectly promoted"
            )

    async def test_rerun_is_a_noop(self, neo4j_test_driver):
        """After the first promotion, a second `ensure_code_schema` call
        must not double-promote or otherwise change the row counts."""
        from app.services.code_parser_service import ensure_code_schema

        async with neo4j_test_driver.session() as session:
            await session.run("DROP CONSTRAINT module_unique IF EXISTS")
            await session.run(
                """
                CREATE (m:Module {graph_id: $gid, name: 'pkg.solo', language: 'python'})
                """,
                {"gid": self._sentinel_tenant},
            )

        await ensure_code_schema(neo4j_test_driver)
        await ensure_code_schema(neo4j_test_driver)

        async with neo4j_test_driver.session() as session:
            res = await session.run(
                "MATCH (m:CodeModule {graph_id: $gid}) RETURN count(m) AS c",
                {"gid": self._sentinel_tenant},
            )
            row = await res.single()
            assert row["c"] == 1, (
                f"expected exactly 1 :CodeModule after two ensure runs, got {row['c']}"
            )
            res = await session.run(
                "MATCH (m:Module {graph_id: $gid}) RETURN count(m) AS c",
                {"gid": self._sentinel_tenant},
            )
            row = await res.single()
            assert row["c"] == 0, "leftover :Module after re-run"
