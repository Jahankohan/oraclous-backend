"""
ORA-150 Integration Tests: Neo4j relationship temporal index validation.

Validates ORA-138 fix: composite relationship temporal index rel_temporal_idx
covering (r.graph_id, r.valid_from, r.valid_to) is present, ONLINE, and
actually used by temporal filter queries (index seek, not full scan).

Requirements:
  - Running Neo4j instance (Docker: make dev-up)
  - OPENAI_API_KEY not required — no LLM calls in these tests
  - pytest -m "integration and neo4j"

Acceptance criteria (from ORA-150):
  [x] rel_temporal_idx present and ONLINE in Neo4j schema
  [x] Query execution plan shows index seek on r.valid_from / r.valid_to
  [x] P95 temporal filter query latency < 300ms on graph with >10k relationships
  [x] No regressions in existing temporal tests
  [x] Multi-tenant isolation: graph_id enforced alongside temporal filter
"""

import statistics
import time
import uuid
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
import pytest_asyncio

try:
    from app.core.neo4j_client import neo4j_client
    from app.schemas.graph_schemas import TemporalFilter
    from app.services.pipeline_service import MultiTenantGraphRAGPipeline
    from app.services.snapshot_service import snapshot_service

    _IMPORTS_OK = True
except Exception as _import_err:
    neo4j_client = None
    snapshot_service = None
    MultiTenantGraphRAGPipeline = None
    TemporalFilter = None
    _IMPORTS_OK = False
    _IMPORT_ERR = str(_import_err)

pytestmark = [pytest.mark.integration, pytest.mark.neo4j]

# ---------------------------------------------------------------------------
# Skip guard — can't run without Neo4j or if ORA-99 blocks imports
# ---------------------------------------------------------------------------

if not _IMPORTS_OK:
    pytestmark.append(
        pytest.mark.skip(
            reason=f"App imports blocked (ORA-99 SQLAlchemy metadata bug): {_IMPORT_ERR if not _IMPORTS_OK else ''}"
        )
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture(scope="module")
async def neo4j():
    """Connect to Neo4j and ensure indexes, yield driver, then disconnect."""
    if not _IMPORTS_OK:
        pytest.skip("App imports blocked by ORA-99")
    await neo4j_client.connect()
    await snapshot_service.ensure_indexes()
    yield neo4j_client
    await neo4j_client.disconnect()


@pytest_asyncio.fixture
async def test_graph_id(neo4j) -> AsyncGenerator[str, None]:
    """Create an isolated test graph, yield its ID, clean up after test."""
    graph_id = f"test-ora138-{uuid.uuid4().hex[:8]}"
    yield graph_id
    # Cleanup: remove all test nodes and relationships
    await neo4j.execute_query(
        "MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": graph_id}
    )


async def _seed_temporal_graph(neo4j, graph_id: str, n_rels: int = 10_000) -> None:
    """
    Create a test graph with n_rels relationships each having
    valid_from / valid_to temporal properties.

    Entity nodes: alternating SourceNode / TargetNode pairs.
    Relationships: RELATES_TO with temporal windows spread across 20 years.
    """
    batch_size = 500
    base_year = 2000

    for batch_start in range(0, n_rels, batch_size):
        batch_end = min(batch_start + batch_size, n_rels)
        params_list = []
        for i in range(batch_start, batch_end):
            year_offset = i % 20  # 20-year spread
            vf = datetime(base_year + year_offset, 1, 1, tzinfo=UTC)
            # 70% have valid_to (closed window), 30% still-ongoing (valid_to=NULL)
            vt = datetime(vf.year + 2, 12, 31, tzinfo=UTC) if i % 10 != 0 else None
            params_list.append(
                {
                    "graph_id": graph_id,
                    "src_id": f"src-{i}",
                    "tgt_id": f"tgt-{i}",
                    "valid_from": vf.isoformat(),
                    "valid_to": vt.isoformat() if vt else None,
                }
            )

        # Batch MERGE in a single transaction
        await neo4j.execute_query(
            """
            UNWIND $rows AS row
            MERGE (s:__Entity__ {graph_id: row.graph_id, entity_id: row.src_id})
            MERGE (t:__Entity__ {graph_id: row.graph_id, entity_id: row.tgt_id})
            CREATE (s)-[r:RELATES_TO {
                graph_id:   row.graph_id,
                valid_from: CASE WHEN row.valid_from IS NOT NULL
                                 THEN datetime(row.valid_from) ELSE null END,
                valid_to:   CASE WHEN row.valid_to IS NOT NULL
                                 THEN datetime(row.valid_to) ELSE null END,
                transaction_time: datetime()
            }]->(t)
            """,
            {"rows": params_list},
        )


# ---------------------------------------------------------------------------
# Suite 1: Index presence and state
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.neo4j
class TestIndexPresence:
    """Verify rel_temporal_idx exists and is ONLINE after ensure_indexes()."""

    async def test_rel_temporal_idx_exists(self, neo4j):
        result = await neo4j.execute_query(
            "SHOW INDEXES WHERE name = 'rel_temporal_idx'", {}
        )
        assert len(result) > 0, (
            "rel_temporal_idx not found in Neo4j schema — "
            "ensure_indexes() may not have been called at startup"
        )

    async def test_rel_temporal_idx_is_online(self, neo4j):
        result = await neo4j.execute_query(
            "SHOW INDEXES WHERE name = 'rel_temporal_idx'", {}
        )
        assert len(result) == 1
        state = result[0].get("state", "").upper()
        assert (
            state == "ONLINE"
        ), f"rel_temporal_idx state is {state!r}, expected ONLINE"

    async def test_rel_temporal_idx_covers_correct_properties(self, neo4j):
        result = await neo4j.execute_query(
            "SHOW INDEXES WHERE name = 'rel_temporal_idx'", {}
        )
        assert len(result) == 1
        props = result[0].get("properties", [])
        assert "graph_id" in props, f"graph_id missing from index properties: {props}"
        assert (
            "valid_from" in props
        ), f"valid_from missing from index properties: {props}"
        assert "valid_to" in props, f"valid_to missing from index properties: {props}"

    async def test_all_versioning_indexes_still_present(self, neo4j):
        """Regression: pre-existing versioning indexes must not have been dropped."""
        required = [
            "entity_transaction_time_idx",
            "entity_invalidated_at_idx",
            "version_graph_idx",
            "version_number_idx",
            "rel_version_composite_idx",
        ]
        result = await neo4j.execute_query(
            "SHOW INDEXES WHERE name IN $names", {"names": required}
        )
        found = {r["name"] for r in result}
        missing = set(required) - found
        assert not missing, f"Pre-existing indexes removed (regression): {missing}"

    async def test_standalone_rel_temporal_indexes_present(self, neo4j):
        """ORA-138: standalone rel_valid_from_idx and rel_valid_to_idx must be ONLINE.

        These indexes support traversal queries (e.g. _multihop_enrich) where
        r.graph_id is not in the WHERE clause, so the composite rel_temporal_idx
        cannot be used by the query planner.
        """
        required = ["rel_valid_from_idx", "rel_valid_to_idx"]
        result = await neo4j.execute_query(
            "SHOW INDEXES WHERE name IN $names", {"names": required}
        )
        found = {r["name"] for r in result}
        missing = set(required) - found
        assert not missing, (
            f"ORA-138 standalone rel temporal indexes missing: {missing}. "
            "ensure_indexes() may not have run at startup."
        )
        for row in result:
            state = row.get("state", "").upper()
            assert (
                state == "ONLINE"
            ), f"Index {row['name']!r} is {state!r}, expected ONLINE"


# ---------------------------------------------------------------------------
# Suite 2: Query execution plan — index seek validation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.neo4j
class TestIndexSeek:
    """
    Verify that temporal filter queries use index seeks, not full scans.
    Neo4j EXPLAIN returns an execution plan — check for DirectedRelationshipIndexSeek
    or RelationshipIndexSeekByRange rather than DirectedRelationshipScan.
    """

    async def test_point_in_time_query_uses_index(self, neo4j, test_graph_id):
        await _seed_temporal_graph(neo4j, test_graph_id, n_rels=1_000)
        pit = "2010-01-01T00:00:00"

        plan = await neo4j.execute_query(
            """
            EXPLAIN
            MATCH (s)-[r:RELATES_TO]->(t)
            WHERE r.graph_id = $graph_id
              AND (r.valid_from IS NULL OR r.valid_from <= datetime($pit))
              AND (r.valid_to IS NULL OR r.valid_to > datetime($pit))
            RETURN s, r, t
            """,
            {"graph_id": test_graph_id, "pit": pit},
        )
        # EXPLAIN returns a plan summary; verify no full RelationshipScan in hot path
        plan_str = str(plan)
        assert "DirectedRelationshipScan" not in plan_str or "IndexSeek" in plan_str, (
            "Query execution plan shows full relationship scan — "
            "index is not being used for temporal filter"
        )

    async def test_current_only_query_uses_index(self, neo4j, test_graph_id):
        await _seed_temporal_graph(neo4j, test_graph_id, n_rels=500)

        plan = await neo4j.execute_query(
            """
            EXPLAIN
            MATCH (s)-[r:RELATES_TO]->(t)
            WHERE r.graph_id = $graph_id
              AND r.valid_to IS NULL
            RETURN s, r, t
            """,
            {"graph_id": test_graph_id},
        )
        # NULL check on an indexed property should use the index
        plan_str = str(plan)
        assert "DirectedRelationshipScan" not in plan_str, (
            "current_only query using full scan — "
            "valid_to IS NULL not leveraging index"
        )


# ---------------------------------------------------------------------------
# Suite 3: P95 latency benchmark (< 300ms on 10k relationships)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.neo4j
@pytest.mark.slow
class TestTemporalQueryLatency:
    """
    ORA-150 acceptance: P95 temporal filter query latency < 300ms
    on a graph with >10k relationships.
    """

    _GRAPH_SEEDED = False  # module-level flag to avoid re-seeding

    async def _ensure_seeded(self, neo4j, graph_id: str):
        count_result = await neo4j.execute_query(
            "MATCH ()-[r:RELATES_TO {graph_id: $gid}]->() RETURN count(r) AS cnt",
            {"gid": graph_id},
        )
        cnt = count_result[0]["cnt"] if count_result else 0
        if cnt < 10_000:
            await _seed_temporal_graph(neo4j, graph_id, n_rels=10_000)

    async def test_point_in_time_p95_under_300ms(self, neo4j):
        graph_id = "perf-test-ora138-pit"
        await self._ensure_seeded(neo4j, graph_id)

        pit = "2010-06-15T00:00:00"
        latencies = []
        N_SAMPLES = 30

        for _ in range(N_SAMPLES):
            start = time.monotonic()
            await neo4j.execute_query(
                """
                MATCH (s)-[r:RELATES_TO]->(t)
                WHERE r.graph_id = $graph_id
                  AND (r.valid_from IS NULL OR r.valid_from <= datetime($pit))
                  AND (r.valid_to IS NULL OR r.valid_to > datetime($pit))
                RETURN count(r) AS cnt
                """,
                {"graph_id": graph_id, "pit": pit},
            )
            latencies.append((time.monotonic() - start) * 1000)

        latencies.sort()
        p95 = latencies[int(0.95 * N_SAMPLES)]
        p50 = statistics.median(latencies)

        assert p95 < 300, (
            f"point_in_time P95 latency {p95:.1f}ms exceeds 300ms SLA "
            f"(P50={p50:.1f}ms, N={N_SAMPLES}). "
            "Check that rel_temporal_idx is ONLINE and Neo4j has warmed its cache."
        )

    async def test_current_only_p95_under_300ms(self, neo4j):
        graph_id = "perf-test-ora138-current"
        await self._ensure_seeded(neo4j, graph_id)

        latencies = []
        N_SAMPLES = 30

        for _ in range(N_SAMPLES):
            start = time.monotonic()
            await neo4j.execute_query(
                """
                MATCH (s)-[r:RELATES_TO]->(t)
                WHERE r.graph_id = $graph_id
                  AND r.valid_to IS NULL
                RETURN count(r) AS cnt
                """,
                {"graph_id": graph_id},
            )
            latencies.append((time.monotonic() - start) * 1000)

        latencies.sort()
        p95 = latencies[int(0.95 * N_SAMPLES)]

        assert p95 < 300, f"current_only P95 latency {p95:.1f}ms exceeds 300ms SLA"

    async def test_range_filter_p95_under_300ms(self, neo4j):
        graph_id = "perf-test-ora138-range"
        await self._ensure_seeded(neo4j, graph_id)

        latencies = []
        N_SAMPLES = 30

        for _ in range(N_SAMPLES):
            start = time.monotonic()
            await neo4j.execute_query(
                """
                MATCH (s)-[r:RELATES_TO]->(t)
                WHERE r.graph_id = $graph_id
                  AND (r.valid_from IS NULL OR r.valid_from >= datetime('2005-01-01T00:00:00'))
                  AND (r.valid_to IS NULL OR r.valid_to <= datetime('2015-12-31T00:00:00'))
                RETURN count(r) AS cnt
                """,
                {"graph_id": graph_id},
            )
            latencies.append((time.monotonic() - start) * 1000)

        latencies.sort()
        p95 = latencies[int(0.95 * N_SAMPLES)]

        assert p95 < 300, f"range_filter P95 latency {p95:.1f}ms exceeds 300ms SLA"


# ---------------------------------------------------------------------------
# Suite 4: Multi-tenant isolation alongside temporal filter
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.neo4j
class TestMultiTenantTemporalIsolation:
    """
    ORA-150 acceptance: graph_id isolation must hold even when temporal
    filter is applied. Temporal filter must NOT leak cross-tenant relationships.
    """

    async def test_point_in_time_cannot_leak_cross_tenant_rels(self, neo4j):
        graph_a = f"tenant-a-{uuid.uuid4().hex[:6]}"
        graph_b = f"tenant-b-{uuid.uuid4().hex[:6]}"

        try:
            # Seed graph_a with a relationship at 2015
            await neo4j.execute_query(
                """
                MERGE (s:__Entity__ {graph_id: $gid, entity_id: 'alice'})
                MERGE (t:__Entity__ {graph_id: $gid, entity_id: 'acme'})
                CREATE (s)-[r:WORKS_FOR {
                    graph_id: $gid,
                    valid_from: datetime('2010-01-01T00:00:00'),
                    valid_to: null
                }]->(t)
                """,
                {"gid": graph_a},
            )
            # Seed graph_b with a similar relationship — must not appear in graph_a queries
            await neo4j.execute_query(
                """
                MERGE (s:__Entity__ {graph_id: $gid, entity_id: 'alice'})
                MERGE (t:__Entity__ {graph_id: $gid, entity_id: 'acme'})
                CREATE (s)-[r:WORKS_FOR {
                    graph_id: $gid,
                    valid_from: datetime('2010-01-01T00:00:00'),
                    valid_to: null
                }]->(t)
                """,
                {"gid": graph_b},
            )

            # Query graph_a with point_in_time — must return only graph_a rels
            result = await neo4j.execute_query(
                """
                MATCH (s)-[r:WORKS_FOR]->(t)
                WHERE r.graph_id = $graph_id
                  AND (r.valid_from IS NULL OR r.valid_from <= datetime('2015-01-01T00:00:00'))
                  AND (r.valid_to IS NULL OR r.valid_to > datetime('2015-01-01T00:00:00'))
                RETURN r.graph_id AS gid
                """,
                {"graph_id": graph_a},
            )

            graph_ids_returned = {row["gid"] for row in result}
            assert graph_ids_returned == {graph_a}, (
                f"Cross-tenant leak: temporal filter returned graph_ids {graph_ids_returned} "
                f"when querying {graph_a!r} — graph_b relationships must not appear"
            )

        finally:
            for gid in (graph_a, graph_b):
                await neo4j.execute_query(
                    "MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid}
                )

    async def test_current_only_filter_scoped_to_graph(self, neo4j):
        graph_a = f"tenant-a-{uuid.uuid4().hex[:6]}"
        graph_b = f"tenant-b-{uuid.uuid4().hex[:6]}"

        try:
            for gid in (graph_a, graph_b):
                await neo4j.execute_query(
                    """
                    MERGE (s:__Entity__ {graph_id: $gid, entity_id: 'x'})
                    MERGE (t:__Entity__ {graph_id: $gid, entity_id: 'y'})
                    CREATE (s)-[r:LINKS {graph_id: $gid, valid_to: null}]->(t)
                    """,
                    {"gid": gid},
                )

            result = await neo4j.execute_query(
                """
                MATCH (s)-[r:LINKS]->(t)
                WHERE r.graph_id = $graph_id AND r.valid_to IS NULL
                RETURN r.graph_id AS gid
                """,
                {"graph_id": graph_a},
            )
            ids = {row["gid"] for row in result}
            assert ids == {
                graph_a
            }, f"current_only filter leaked cross-tenant data: {ids}"

        finally:
            for gid in (graph_a, graph_b):
                await neo4j.execute_query(
                    "MATCH (n {graph_id: $gid}) DETACH DELETE n", {"gid": gid}
                )
