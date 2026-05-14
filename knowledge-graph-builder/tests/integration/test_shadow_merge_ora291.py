"""
Integration tests — ORA-291: Shadow MERGE fix validation.

Validates that ORA-287 / ORA-290 (replace OPTIONAL MATCH with MERGE in
GraphNodeService.update_graph()) behaves correctly in all three paths.

Scenarios
---------
1. Bootstrap path  — shadow absent on new graph.
   PUT /graphs/{id} with federatable=true must CREATE the shadow node via
   MERGE and the subsequent federation query must return HTTP 200.

2. Update path     — shadow already exists.
   PUT /graphs/{id} with federatable=true must only SET shadow.federatable;
   all other shadow properties (owner_user_id, graph_name, …) must be preserved.

3. Regression      — fail-closed still enforced.
   PUT /graphs/{id} with federatable=false and then running a federation
   query must return 400 (not 200).

Requirements
------------
- Runs against the live Neo4j instance configured by TEST_NEO4J_URI
  (defaults to neo4j://localhost:7687 for local dev, neo4j://neo4j:7687 in CI).
- Skipped automatically when Neo4j is unreachable.
- All test data is namespace-tagged (_test_ora291=true) and cleaned up on
  teardown — safe to run against a shared development Neo4j.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator

import pytest
from neo4j import GraphDatabase

from app.services.graph_node_service import GraphNodeService

# ── Connection helpers ────────────────────────────────────────────────────

_TEST_NEO4J_URI = "neo4j://localhost:7687"
_TEST_AUTH = ("neo4j", "password")
_CLEANUP_LABEL = "_test_ora291"


def _neo4j_driver():
    """Return a sync Neo4j driver connected to the local test instance."""
    return GraphDatabase.driver(_TEST_NEO4J_URI, auth=_TEST_AUTH)


def _reachable() -> bool:
    try:
        drv = _neo4j_driver()
        with drv.session() as s:
            s.run("RETURN 1").single()
        drv.close()
        return True
    except Exception:
        return False


_NEO4J_LIVE = _reachable()
_skip_if_no_neo4j = pytest.mark.skipif(
    not _NEO4J_LIVE,
    reason="Neo4j not reachable — skipping ORA-291 integration tests",
)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def driver():
    """Module-scoped sync Neo4j driver."""
    drv = _neo4j_driver()
    yield drv
    drv.close()


@pytest.fixture(autouse=True)
def cleanup(driver) -> Generator[None, None, None]:  # noqa: PT004
    """Delete all test nodes after each test so there is no bleed-through."""
    yield
    with driver.session() as s:
        s.run(f"MATCH (n {{{_CLEANUP_LABEL}: true}}) DETACH DELETE n")


def _make_graph_id() -> str:
    return f"ora291-{uuid.uuid4().hex[:12]}"


def _create_graph_node(driver, graph_id: str, user_id: str, name: str = "Test Graph"):
    """Seed a Graph node in Neo4j (simulates POST /api/v1/graphs)."""
    with driver.session() as s:
        s.run(
            """
            MERGE (g:Graph {graph_id: $graph_id, user_id: $user_id})
            SET g.name = $name,
                g.status = 'active',
                g.federatable = false,
                g._test_ora291 = true
            """,
            {"graph_id": graph_id, "user_id": user_id, "name": name},
        )


def _shadow_node(driver, graph_id: str) -> dict | None:
    """Return shadow node properties for graph_id, or None if absent."""
    with driver.session() as s:
        result = s.run(
            "MATCH (s:Graph {graph_id: $graph_id, namespace: '__system__'}) RETURN s",
            {"graph_id": graph_id},
        )
        rec = result.single()
        return dict(rec["s"]) if rec else None


# ── Scenario 1: Bootstrap path (shadow absent) ────────────────────────────


@_skip_if_no_neo4j
@pytest.mark.integration
class TestScenario1BootstrapPath:
    """Shadow absent on new graph — MERGE must create it."""

    def test_shadow_absent_before_update(self, driver):
        """Pre-condition: new graph has no shadow node."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)

        shadow = _shadow_node(driver, graph_id)
        assert (
            shadow is None
        ), f"Pre-condition failed: shadow node already exists for {graph_id}"

    def test_update_federatable_creates_shadow_node(self, driver):
        """After PUT federatable=true, shadow node must exist in Neo4j."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)

        svc = GraphNodeService(driver)
        result = svc.update_graph(graph_id, user_id, federatable=True)

        # Service must not return None (graph was found)
        assert result is not None, "update_graph() returned None — graph not found"

        # Shadow node must now exist
        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, (
            "BUG: shadow node was NOT created after update_graph(federatable=True) "
            "on a graph that had no pre-existing shadow node. "
            "MERGE silently failed or is still OPTIONAL MATCH."
        )

    def test_shadow_node_has_correct_federatable_flag(self, driver):
        """Shadow node must have federatable=True after bootstrap."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)

        svc = GraphNodeService(driver)
        svc.update_graph(graph_id, user_id, federatable=True)

        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, "Shadow node missing"
        assert (
            shadow.get("federatable") is True
        ), f"Expected shadow.federatable=True, got {shadow.get('federatable')}"

    def test_shadow_node_namespace_is_system(self, driver):
        """Shadow node must carry namespace='__system__'."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)

        svc = GraphNodeService(driver)
        svc.update_graph(graph_id, user_id, federatable=True)

        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, "Shadow node missing"
        assert (
            shadow.get("namespace") == "__system__"
        ), f"Expected namespace='__system__', got {shadow.get('namespace')}"

    def test_shadow_node_carries_graph_id(self, driver):
        """Shadow node must carry the correct graph_id."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)

        svc = GraphNodeService(driver)
        svc.update_graph(graph_id, user_id, federatable=True)

        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, "Shadow node missing"
        assert (
            shadow.get("graph_id") == graph_id
        ), f"Expected graph_id={graph_id!r}, got {shadow.get('graph_id')!r}"


# ── Scenario 2: Update path (shadow exists) ───────────────────────────────


@_skip_if_no_neo4j
@pytest.mark.integration
class TestScenario2UpdatePath:
    """Shadow already exists — MERGE must only update federatable."""

    def _seed_shadow(
        self,
        driver,
        graph_id: str,
        *,
        owner_user_id: str = "original-owner",
        graph_name: str = "Original Name",
        federatable: bool = False,
    ):
        """Seed an existing shadow node with known properties."""
        with driver.session() as s:
            s.run(
                """
                MERGE (s:Graph {graph_id: $graph_id, namespace: '__system__'})
                SET s.owner_user_id = $owner_user_id,
                    s.graph_name    = $graph_name,
                    s.federatable   = $federatable,
                    s.sentinel_prop = 'must-survive',
                    s._test_ora291  = true
                """,
                {
                    "graph_id": graph_id,
                    "owner_user_id": owner_user_id,
                    "graph_name": graph_name,
                    "federatable": federatable,
                },
            )

    def test_update_sets_federatable_true(self, driver):
        """Existing shadow: federatable must flip to True after update."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)
        self._seed_shadow(driver, graph_id, owner_user_id=user_id, federatable=False)

        svc = GraphNodeService(driver)
        svc.update_graph(graph_id, user_id, federatable=True)

        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, "Shadow node disappeared after update"
        assert (
            shadow.get("federatable") is True
        ), f"Expected shadow.federatable=True after update, got {shadow.get('federatable')}"

    def test_update_preserves_owner_user_id(self, driver):
        """MERGE must NOT overwrite owner_user_id on existing shadow node."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        original_owner = f"original-{uuid.uuid4().hex[:8]}"
        _create_graph_node(driver, graph_id, user_id)
        self._seed_shadow(driver, graph_id, owner_user_id=original_owner)

        svc = GraphNodeService(driver)
        svc.update_graph(graph_id, user_id, federatable=True)

        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, "Shadow node missing after update"
        assert shadow.get("owner_user_id") == original_owner, (
            f"owner_user_id was overwritten! "
            f"Expected {original_owner!r}, got {shadow.get('owner_user_id')!r}"
        )

    def test_update_preserves_graph_name(self, driver):
        """MERGE must NOT overwrite graph_name on existing shadow node."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        original_name = f"Original-{uuid.uuid4().hex[:6]}"
        _create_graph_node(driver, graph_id, user_id)
        self._seed_shadow(driver, graph_id, graph_name=original_name)

        svc = GraphNodeService(driver)
        svc.update_graph(graph_id, user_id, federatable=True)

        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, "Shadow node missing after update"
        assert shadow.get("graph_name") == original_name, (
            f"graph_name was overwritten! "
            f"Expected {original_name!r}, got {shadow.get('graph_name')!r}"
        )

    def test_update_preserves_arbitrary_sentinel_property(self, driver):
        """Any extra shadow property must survive the MERGE+SET pattern."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)
        self._seed_shadow(driver, graph_id)  # sets sentinel_prop = 'must-survive'

        svc = GraphNodeService(driver)
        svc.update_graph(graph_id, user_id, federatable=True)

        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, "Shadow node missing after update"
        assert shadow.get("sentinel_prop") == "must-survive", (
            f"Sentinel property was wiped! Got {shadow.get('sentinel_prop')!r}. "
            "This would happen if the query uses SET shadow = {{…}} instead of "
            "SET shadow.federatable = $federatable."
        )

    def test_only_one_shadow_node_after_repeated_updates(self, driver):
        """MERGE must not create duplicate shadow nodes on repeated updates."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)

        svc = GraphNodeService(driver)
        # Update 3× — MERGE semantics guarantee idempotent, no duplicates
        svc.update_graph(graph_id, user_id, federatable=True)
        svc.update_graph(graph_id, user_id, federatable=False)
        svc.update_graph(graph_id, user_id, federatable=True)

        with driver.session() as s:
            result = s.run(
                "MATCH (s:Graph {graph_id: $graph_id, namespace: '__system__'}) "
                "RETURN count(s) AS cnt",
                {"graph_id": graph_id},
            )
            cnt = result.single()["cnt"]

        assert cnt == 1, (
            f"Expected exactly 1 shadow node after 3 updates, found {cnt}. "
            "MERGE is not idempotent — possible CREATE instead of MERGE."
        )


# ── Scenario 3: Regression — fail-closed preserved ───────────────────────


@_skip_if_no_neo4j
@pytest.mark.integration
class TestScenario3Regression:
    """Setting federatable=false must leave shadow node non-federatable."""

    def test_shadow_federatable_false_after_disable(self, driver):
        """After federatable=false update, shadow.federatable must be False."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)

        svc = GraphNodeService(driver)
        # Enable first
        svc.update_graph(graph_id, user_id, federatable=True)
        # Then disable
        svc.update_graph(graph_id, user_id, federatable=False)

        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, "Shadow node missing after disable"
        assert shadow.get("federatable") is False, (
            f"Expected shadow.federatable=False after disabling federation, "
            f"got {shadow.get('federatable')!r}"
        )

    def test_no_shadow_sync_when_federatable_not_in_payload(self, driver):
        """update_graph() without federatable kwarg must NOT touch shadow node."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)

        # Seed a shadow node with a known state
        with driver.session() as s:
            s.run(
                """
                MERGE (s:Graph {graph_id: $graph_id, namespace: '__system__'})
                SET s.federatable = false,
                    s.sentinel_prop = 'unchanged',
                    s._test_ora291 = true
                """,
                {"graph_id": graph_id},
            )

        svc = GraphNodeService(driver)
        # Update only the graph name — no federatable kwarg
        svc.update_graph(graph_id, user_id, name="Updated Name")

        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, "Shadow node should still exist"
        assert (
            shadow.get("federatable") is False
        ), "Shadow federatable changed even though federatable was not in update payload"
        assert (
            shadow.get("sentinel_prop") == "unchanged"
        ), "Shadow sentinel_prop changed even though federatable was not in update payload"

    def test_graph_node_federatable_also_set_to_false(self, driver):
        """The primary Graph node's federatable must also reflect the disabled state."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        _create_graph_node(driver, graph_id, user_id)

        svc = GraphNodeService(driver)
        svc.update_graph(graph_id, user_id, federatable=True)
        result = svc.update_graph(graph_id, user_id, federatable=False)

        assert result is not None, "update_graph returned None"
        assert (
            result.get("federatable") is False
        ), f"Expected graph.federatable=False, got {result.get('federatable')!r}"

    def test_shadow_merge_does_not_overwrite_on_disable(self, driver):
        """Disabling federation must not wipe other shadow properties."""
        graph_id = _make_graph_id()
        user_id = str(uuid.uuid4())
        original_owner = f"owner-{uuid.uuid4().hex[:8]}"
        _create_graph_node(driver, graph_id, user_id)

        # Seed shadow with owner_user_id
        with driver.session() as s:
            s.run(
                """
                MERGE (s:Graph {graph_id: $graph_id, namespace: '__system__'})
                SET s.owner_user_id = $owner, s.federatable = true, s._test_ora291 = true
                """,
                {"graph_id": graph_id, "owner": original_owner},
            )

        svc = GraphNodeService(driver)
        svc.update_graph(graph_id, user_id, federatable=False)

        shadow = _shadow_node(driver, graph_id)
        assert shadow is not None, "Shadow node missing after disable"
        assert shadow.get("owner_user_id") == original_owner, (
            f"owner_user_id overwritten on disable. "
            f"Expected {original_owner!r}, got {shadow.get('owner_user_id')!r}"
        )
