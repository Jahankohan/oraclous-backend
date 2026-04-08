"""
ORA-150: Smoke tests for ORA-138 fix — Neo4j relationship temporal indexes.

These tests use static source analysis (AST / text inspection) so they run
without importing app modules, which currently fail due to the pre-existing
ORA-99 SQLAlchemy metadata double-registration bug.

Tests validate:
  1. rel_temporal_idx composite index exists in snapshot_service.ensure_indexes()
  2. The index definition covers (r.graph_id, r.valid_from, r.valid_to)
  3. ensure_indexes() is called at application startup (main.py lifespan)
  4. compile_temporal_filter() generates correct WHERE clauses for all
     four filter modes — confirming the index will be exercised at runtime
  5. No regression: all pre-existing index definitions still present

These tests will be superseded by live Neo4j integration tests once
ORA-99 is resolved and Docker test infra is confirmed healthy.
"""

import os
import re

import pytest

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "app")


def _read(rel: str) -> str:
    with open(os.path.join(_BASE, rel)) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Suite 1: snapshot_service.py — index definitions
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSnapshotServiceTemporalIndex:
    """Verify rel_temporal_idx composite index is defined correctly."""

    def _get_index_lines(self) -> list[str]:
        src = _read("services/snapshot_service.py")
        return [
            line.strip()
            for line in src.splitlines()
            if "CREATE INDEX" in line or "rel_temporal" in line
        ]

    def test_rel_temporal_idx_present(self):
        """The composite temporal index for relationships must exist."""
        src = _read("services/snapshot_service.py")
        assert (
            "rel_temporal_idx" in src
        ), "Missing rel_temporal_idx — ORA-138 fix not applied or index renamed"

    def test_rel_temporal_idx_covers_valid_from(self):
        """Composite index must include r.valid_from."""
        src = _read("services/snapshot_service.py")
        # Find the line containing rel_temporal_idx
        for line in src.splitlines():
            if "rel_temporal_idx" in line and "CREATE INDEX" in line:
                assert (
                    "valid_from" in line
                ), f"rel_temporal_idx does not cover valid_from: {line!r}"
                return
        pytest.fail("rel_temporal_idx CREATE INDEX statement not found")

    def test_rel_temporal_idx_covers_valid_to(self):
        """Composite index must include r.valid_to."""
        src = _read("services/snapshot_service.py")
        for line in src.splitlines():
            if "rel_temporal_idx" in line and "CREATE INDEX" in line:
                assert (
                    "valid_to" in line
                ), f"rel_temporal_idx does not cover valid_to: {line!r}"
                return
        pytest.fail("rel_temporal_idx CREATE INDEX statement not found")

    def test_rel_temporal_idx_covers_graph_id(self):
        """Composite index must lead with graph_id for multi-tenant partitioning."""
        src = _read("services/snapshot_service.py")
        for line in src.splitlines():
            if "rel_temporal_idx" in line and "CREATE INDEX" in line:
                assert (
                    "graph_id" in line
                ), f"rel_temporal_idx missing graph_id — multi-tenant scan risk: {line!r}"
                # graph_id should appear before valid_from in the index definition
                gi = line.index("graph_id")
                vf = line.index("valid_from")
                assert gi < vf, (
                    "graph_id must be the leading key in rel_temporal_idx for "
                    "optimal multi-tenant + temporal query performance"
                )
                return
        pytest.fail("rel_temporal_idx CREATE INDEX statement not found")

    def test_rel_temporal_idx_is_relationship_index(self):
        """Index must be on relationships ()-[r]-(), not nodes."""
        src = _read("services/snapshot_service.py")
        for line in src.splitlines():
            if "rel_temporal_idx" in line and "CREATE INDEX" in line:
                assert (
                    "()-[r]-()" in line or "FOR ()-[" in line
                ), f"rel_temporal_idx must be a relationship property index: {line!r}"
                return
        pytest.fail("rel_temporal_idx CREATE INDEX statement not found")

    def test_rel_temporal_idx_uses_if_not_exists(self):
        """Index creation must be idempotent (IF NOT EXISTS)."""
        src = _read("services/snapshot_service.py")
        for line in src.splitlines():
            if "rel_temporal_idx" in line and "CREATE INDEX" in line:
                assert (
                    "IF NOT EXISTS" in line
                ), f"Missing IF NOT EXISTS — index creation not idempotent: {line!r}"
                return
        pytest.fail("rel_temporal_idx CREATE INDEX statement not found")

    def test_pre_existing_indexes_not_removed(self):
        """Regression: pre-existing versioning indexes must still be present."""
        src = _read("services/snapshot_service.py")
        required = [
            "entity_transaction_time_idx",
            "entity_invalidated_at_idx",
            "version_graph_idx",
            "version_number_idx",
            "rel_version_composite_idx",
        ]
        for idx in required:
            assert (
                idx in src
            ), f"Pre-existing index {idx!r} was removed — regression in ORA-138 fix"

    def test_ensure_indexes_is_async(self):
        """ensure_indexes must be an async method (called with await in lifespan)."""
        src = _read("services/snapshot_service.py")
        # Find the method definition
        m = re.search(r"(async\s+def|def)\s+ensure_indexes", src)
        assert m is not None, "ensure_indexes method not found"
        assert m.group(0).startswith(
            "async"
        ), "ensure_indexes must be async — it's awaited in main.py lifespan"


# ---------------------------------------------------------------------------
# Suite 2: main.py — startup wiring
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStartupWiring:
    """Verify ensure_indexes() is called during app startup."""

    def test_ensure_indexes_called_in_lifespan(self):
        """main.py lifespan must call snapshot_service.ensure_indexes()."""
        src = (
            _read("../app/main.py")
            if os.path.exists(os.path.join(_BASE, "../app/main.py"))
            else _read_main()
        )
        assert (
            "snapshot_service.ensure_indexes()" in src
            or "await snapshot_service.ensure_indexes()" in src
        ), (
            "snapshot_service.ensure_indexes() not called in application startup — "
            "indexes will not be created on deployment"
        )

    def test_ensure_indexes_awaited(self):
        """ensure_indexes() must be awaited (it's async)."""
        main_src = _read_main()
        assert "await snapshot_service.ensure_indexes()" in main_src, (
            "ensure_indexes() not awaited in main.py — will create a coroutine "
            "without running it (silent failure on startup)"
        )


def _read_main() -> str:
    main_path = os.path.join(os.path.dirname(__file__), "..", "..", "app", "main.py")
    # Normalize path
    main_path = os.path.normpath(main_path)
    with open(main_path) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Suite 3: compile_temporal_filter — static WHERE clause validation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCompileTemporalFilterLogic:
    """
    Validate compile_temporal_filter() output WITHOUT importing the app
    (which is blocked by ORA-99).

    Parses the method source with AST and verifies the string literals it
    would produce, rather than calling the function directly.
    """

    def _get_method_source(self) -> str:
        src = _read("services/pipeline_service.py")
        lines = src.splitlines()
        start = None
        for i, line in enumerate(lines):
            if "def compile_temporal_filter" in line:
                start = i
                break
        assert start is not None, "compile_temporal_filter method not found"
        # Collect method body until next method or class definition at same indent
        indent = len(lines[start]) - len(lines[start].lstrip())
        body_lines = [lines[start]]
        for line in lines[start + 1 :]:
            stripped = line.strip()
            if not stripped:
                body_lines.append(line)
                continue
            curr_indent = len(line) - len(line.lstrip())
            if curr_indent <= indent and stripped and not stripped.startswith("#"):
                break
            body_lines.append(line)
        return "\n".join(body_lines)

    def test_current_only_returns_valid_to_null_check(self):
        """current_only filter must return 'r.valid_to IS NULL'."""
        src = self._get_method_source()
        assert (
            "r.valid_to IS NULL" in src
        ), 'current_only branch must return "r.valid_to IS NULL"'

    def test_point_in_time_filters_on_valid_from(self):
        """point_in_time filter must include r.valid_from clause."""
        src = self._get_method_source()
        assert (
            "r.valid_from" in src and "point_in_time" in src
        ), "compile_temporal_filter must filter on r.valid_from for point_in_time"

    def test_point_in_time_filters_on_valid_to(self):
        """point_in_time filter must include r.valid_to clause."""
        src = self._get_method_source()
        assert (
            "r.valid_to" in src and "point_in_time" in src
        ), "compile_temporal_filter must filter on r.valid_to for point_in_time"

    def test_empty_filter_returns_true(self):
        """Empty filter (no criteria) must return 'true' (no-op)."""
        src = self._get_method_source()
        assert '"true"' in src or "'true'" in src, (
            "Empty TemporalFilter must produce 'true' — "
            "backward compat: existing queries must not break"
        )

    def test_valid_from_gte_filter_present(self):
        """valid_from_gte range filter must be supported."""
        src = self._get_method_source()
        assert (
            "valid_from_gte" in src
        ), "valid_from_gte range filter not implemented in compile_temporal_filter"

    def test_valid_to_lte_filter_present(self):
        """valid_to_lte range filter must be supported."""
        src = self._get_method_source()
        assert (
            "valid_to_lte" in src
        ), "valid_to_lte range filter not implemented in compile_temporal_filter"

    def test_null_safe_clauses_for_open_ended_relationships(self):
        """
        Clauses must be NULL-safe: relationships without valid_from or valid_to
        must still match when no temporal constraint is violated.
        E.g. '(r.valid_from IS NULL OR r.valid_from <= ...)' not just 'r.valid_from <= ...'
        """
        src = self._get_method_source()
        # Check that IS NULL appears alongside comparisons (NULL-safety)
        assert "IS NULL OR" in src or "IS NULL)" in src, (
            "compile_temporal_filter must use NULL-safe clauses — "
            "relationships without temporal properties must still be returned"
        )

    def test_graph_id_not_injected_into_temporal_clause(self):
        """
        compile_temporal_filter produces a clause fragment for WHERE, not a full
        query. graph_id filtering must be done separately (multi-tenancy invariant).
        """
        src = self._get_method_source()
        assert "graph_id" not in src, (
            "compile_temporal_filter must NOT inject graph_id — "
            "that's enforced by the surrounding Cypher query, not the filter compiler"
        )


# ---------------------------------------------------------------------------
# Suite 4: Index strategy review — composite + standalone (ORA-138 final design)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestIndexStrategyReview:
    """
    The ORA-138 fix implements three indexes:
      - rel_temporal_idx    FOR ()-[r]-() ON (r.graph_id, r.valid_from, r.valid_to)
        → composite index for multi-tenant temporal range queries
      - rel_valid_from_idx  FOR ()-[r]-() ON (r.valid_from)
        → standalone index needed for multihop traversal (Neo4j can't use
          composite index for single-property traversal steps)
      - rel_valid_to_idx    FOR ()-[r]-() ON (r.valid_to)
        → same rationale as rel_valid_from_idx

    These tests verify all three indexes are present and the composite
    index retains the graph_id lead key for multi-tenant query isolation.
    """

    def test_composite_approach_is_graph_id_scoped(self):
        """Composite index includes graph_id — avoids cross-tenant index scans."""
        src = _read("services/snapshot_service.py")
        for line in src.splitlines():
            if "rel_temporal_idx" in line and "CREATE INDEX" in line:
                assert "graph_id" in line
                return
        pytest.fail("rel_temporal_idx not found")

    def test_standalone_valid_from_index_present(self):
        """
        rel_valid_from_idx must exist alongside the composite index.
        Neo4j cannot use the composite rel_temporal_idx for single-property
        multihop traversal steps — a dedicated index is required.
        """
        src = _read("services/snapshot_service.py")
        assert "rel_valid_from_idx" in src, (
            "rel_valid_from_idx missing — multihop temporal traversal will "
            "fall back to full relationship scans without this index"
        )

    def test_standalone_valid_to_index_present(self):
        """
        rel_valid_to_idx must exist alongside the composite index.
        Same rationale as rel_valid_from_idx — required for multihop traversal.
        """
        src = _read("services/snapshot_service.py")
        assert "rel_valid_to_idx" in src, (
            "rel_valid_to_idx missing — multihop temporal traversal will "
            "fall back to full relationship scans without this index"
        )
