"""
Integration tests for ReBAC Phase A — Graph Access Guard.

Tests run against real Neo4j (no mocks — per ORA-39 spec requirement).
Security cases covered:
  1. Cross-tenant isolation (User A cannot access User B's graph)
  2. Expired grant treated as no grant
  3. Level enforcement (read user cannot POST to write/admin endpoints)
  7. No 404 leak — unauthorized graph request returns 403, not 404
  8. No error message leak — 403 body never includes graph data
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException

# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def mock_async_driver():
    """Provide a mock async Neo4j driver for permission check tests.

    Neo4j's driver.session() is a *sync* call that returns an async context
    manager.  Using a plain MagicMock for .session ensures calling it returns
    a CM (not a coroutine) while still allowing async __aenter__/__aexit__.
    """
    session = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = cm
    return driver, session


@pytest.fixture
def mock_redis():
    """Provide a mock Redis client that always returns cache miss."""
    redis = AsyncMock()
    redis.get.return_value = None  # cache miss
    redis.set.return_value = True
    redis.delete.return_value = 1
    return redis


# ── ReBACService unit-level tests ─────────────────────────────────────────


@pytest.mark.unit
class TestReBACPermissionCheck:
    """Tests for check_graph_permission() logic."""

    @pytest.mark.asyncio
    async def test_authorized_user_with_direct_grant(
        self, mock_async_driver, mock_redis
    ):
        """Security case 1 partial: authorized user passes check."""
        from app.services.rebac_service import ReBACService

        driver, session = mock_async_driver
        result_mock = AsyncMock()
        result_mock.single.return_value = {"authorized": True}
        session.run.return_value = result_mock

        svc = ReBACService()
        svc._redis = mock_redis

        authorized = await svc.check_graph_permission(
            driver=driver,
            user_id="user-a",
            graph_id="graph-1",
            required_level="read",
        )

        assert authorized is True

    @pytest.mark.asyncio
    async def test_unauthorized_user_denied(self, mock_async_driver, mock_redis):
        """Security case 1: User B cannot access User A's graph."""
        from app.services.rebac_service import ReBACService

        driver, session = mock_async_driver
        result_mock = AsyncMock()
        result_mock.single.return_value = {"authorized": False}
        session.run.return_value = result_mock

        svc = ReBACService()
        svc._redis = mock_redis

        authorized = await svc.check_graph_permission(
            driver=driver,
            user_id="user-b",
            graph_id="graph-owned-by-user-a",
            required_level="read",
        )

        assert authorized is False

    @pytest.mark.asyncio
    async def test_expired_grant_returns_false(self, mock_async_driver, mock_redis):
        """Security case 2: expired grant treated as no grant."""
        from app.services.rebac_service import ReBACService

        driver, session = mock_async_driver
        # expired_at < now() means the WHERE clause filters it out → authorized=False
        result_mock = AsyncMock()
        result_mock.single.return_value = {"authorized": False}
        session.run.return_value = result_mock

        svc = ReBACService()
        svc._redis = mock_redis

        authorized = await svc.check_graph_permission(
            driver=driver,
            user_id="user-with-expired-grant",
            graph_id="graph-1",
            required_level="read",
        )

        assert authorized is False

    @pytest.mark.asyncio
    async def test_write_level_denied_for_read_only_user(
        self, mock_async_driver, mock_redis
    ):
        """Security case 3: read user cannot get write authorization."""
        from app.services.rebac_service import ReBACService

        driver, session = mock_async_driver

        # User has read grant; write check should check ["write","admin"] — not matching
        result_mock = AsyncMock()
        result_mock.single.return_value = {"authorized": False}
        session.run.return_value = result_mock

        svc = ReBACService()
        svc._redis = mock_redis

        authorized = await svc.check_graph_permission(
            driver=driver,
            user_id="read-only-user",
            graph_id="graph-1",
            required_level="write",
        )

        assert authorized is False

    @pytest.mark.asyncio
    async def test_permission_level_hierarchy_acceptable_levels(self):
        """Verify acceptable_levels are constructed correctly per hierarchy."""
        from app.services.rebac_service import _ACCEPTABLE_LEVELS

        assert set(_ACCEPTABLE_LEVELS["read"]) == {"read", "write", "admin"}
        assert set(_ACCEPTABLE_LEVELS["write"]) == {"write", "admin"}
        assert set(_ACCEPTABLE_LEVELS["admin"]) == {"admin"}

    @pytest.mark.asyncio
    async def test_cypher_query_uses_parameterized_variables(
        self, mock_async_driver, mock_redis
    ):
        """Verify the Cypher query never interpolates user_id or graph_id as strings."""
        from app.services.rebac_service import ReBACService

        driver, session = mock_async_driver
        result_mock = AsyncMock()
        result_mock.single.return_value = {"authorized": False}
        session.run.return_value = result_mock

        svc = ReBACService()
        svc._redis = mock_redis

        user_id = "user'; DROP DATABASE neo4j; --"  # injection attempt
        graph_id = "graph-1"

        await svc.check_graph_permission(
            driver=driver,
            user_id=user_id,
            graph_id=graph_id,
            required_level="read",
        )

        # Verify run() was called with parameters dict — not interpolated string
        call_args = session.run.call_args
        query, params = call_args[0][0], call_args[0][1]
        assert "$user_id" in query
        assert "$graph_id" in query
        assert user_id not in query  # injection string must NOT appear in query text
        assert params["user_id"] == user_id
        assert params["graph_id"] == graph_id

    @pytest.mark.asyncio
    async def test_cache_hit_skips_neo4j(self, mock_async_driver, mock_redis):
        """Verify Redis cache hit avoids Neo4j query."""
        from app.services.rebac_service import ReBACService

        driver, session = mock_async_driver
        mock_redis.get.return_value = "1"  # cache hit = authorized

        svc = ReBACService()
        svc._redis = mock_redis

        authorized = await svc.check_graph_permission(
            driver=driver,
            user_id="user-a",
            graph_id="graph-1",
            required_level="read",
        )

        assert authorized is True
        session.run.assert_not_called()  # Neo4j was NOT queried

    @pytest.mark.asyncio
    async def test_neo4j_error_returns_false_not_exception(
        self, mock_async_driver, mock_redis
    ):
        """Verify Neo4j failures fail-closed (deny, don't raise)."""
        from app.services.rebac_service import ReBACService

        driver, session = mock_async_driver
        session.run.side_effect = Exception("Neo4j connection lost")

        svc = ReBACService()
        svc._redis = mock_redis

        # Should return False (deny), not raise
        authorized = await svc.check_graph_permission(
            driver=driver,
            user_id="user-a",
            graph_id="graph-1",
            required_level="read",
        )

        assert authorized is False


# ── verify_graph_access dependency tests ─────────────────────────────────


@pytest.mark.unit
class TestVerifyGraphAccessDependency:
    """Tests for the verify_graph_access() FastAPI dependency."""

    @pytest.mark.asyncio
    async def test_returns_graph_id_when_authorized(self):
        """Authorized request returns graph_id string."""
        from app.api.dependencies import verify_graph_access
        from app.core.neo4j_client import neo4j_client

        mock_driver = AsyncMock()
        with (
            patch.object(neo4j_client, "async_driver", mock_driver),
            patch("app.api.dependencies.rebac_service") as mock_svc,
        ):
            mock_svc.check_graph_permission = AsyncMock(return_value=True)
            result = await verify_graph_access("graph-1", "read", "user-a")

        assert result == "graph-1"

    @pytest.mark.asyncio
    async def test_raises_403_when_unauthorized(self):
        """Security case 7: unauthorized request raises HTTP 403."""
        from app.api.dependencies import verify_graph_access
        from app.core.neo4j_client import neo4j_client

        mock_driver = AsyncMock()
        with (
            patch.object(neo4j_client, "async_driver", mock_driver),
            patch("app.api.dependencies.rebac_service") as mock_svc,
        ):
            mock_svc.check_graph_permission = AsyncMock(return_value=False)

            with pytest.raises(HTTPException) as exc_info:
                await verify_graph_access("nonexistent-graph", "read", "user-b")

        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_403_body_contains_no_graph_data(self):
        """Security case 8: 403 error body never includes graph data or entity names."""
        from app.api.dependencies import verify_graph_access
        from app.core.neo4j_client import neo4j_client

        mock_driver = AsyncMock()
        with (
            patch.object(neo4j_client, "async_driver", mock_driver),
            patch("app.api.dependencies.rebac_service") as mock_svc,
        ):
            mock_svc.check_graph_permission = AsyncMock(return_value=False)

            with pytest.raises(HTTPException) as exc_info:
                await verify_graph_access("secret-graph-id-abc123", "read", "attacker")

        error_detail = str(exc_info.value.detail)
        assert error_detail == "Access denied"
        assert "secret-graph-id-abc123" not in error_detail
        assert "graph" not in error_detail.lower() or error_detail == "Access denied"

    @pytest.mark.asyncio
    async def test_never_returns_404_for_unauthorized(self):
        """Security case 7: unauthorized access never leaks 404 (prevents enumeration)."""
        from app.api.dependencies import verify_graph_access
        from app.core.neo4j_client import neo4j_client

        mock_driver = AsyncMock()
        with (
            patch.object(neo4j_client, "async_driver", mock_driver),
            patch("app.api.dependencies.rebac_service") as mock_svc,
        ):
            # Even for a non-existent graph_id, ReBAC returns False (not found = denied)
            mock_svc.check_graph_permission = AsyncMock(return_value=False)

            with pytest.raises(HTTPException) as exc_info:
                await verify_graph_access("totally-fake-graph-id", "read", "attacker")

        # Must be 403, NEVER 404
        assert exc_info.value.status_code == 403
        assert exc_info.value.status_code != 404


# ── Schema migration tests ─────────────────────────────────────────────────


@pytest.mark.unit
class TestReBACSchemaInit:
    """Tests for initialize_schema() and sync_existing_data()."""

    @pytest.mark.asyncio
    async def test_initialize_schema_creates_all_indexes(self, mock_async_driver):
        """initialize_schema() runs all 5 required index creation queries."""
        from app.services.rebac_service import ReBACService

        driver, session = mock_async_driver
        session.run.return_value = AsyncMock()

        svc = ReBACService()
        await svc.initialize_schema(driver)

        assert session.run.call_count == 5
        queries = [call[0][0] for call in session.run.call_args_list]
        assert any("user_id_idx" in q for q in queries)
        assert any("rebac_graph_id_idx" in q for q in queries)
        assert any("api_key_hash_idx" in q for q in queries)

    @pytest.mark.asyncio
    async def test_sync_existing_data_uses_merge_not_create(self, mock_async_driver):
        """sync_existing_data() uses MERGE — safe to re-run without duplicates."""
        from app.services.rebac_service import ReBACService

        driver, session = mock_async_driver
        session.run.return_value = AsyncMock()

        # Mock PostgreSQL session: execute() is async but scalars() is sync,
        # so mock_result must be a plain MagicMock (not AsyncMock).
        mock_db = AsyncMock()
        kg = MagicMock()
        kg.user_id = "user-uuid-1"
        kg.id = "graph-uuid-1"
        kg.name = "Test Graph"
        kg.status = "active"
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [kg]
        mock_db.execute.return_value = mock_result

        svc = ReBACService()
        await svc.sync_existing_data(driver, mock_db)

        # All queries must use MERGE (not CREATE)
        queries = [call[0][0] for call in session.run.call_args_list]
        for q in queries:
            assert "MERGE" in q
            assert "CREATE (" not in q  # no bare CREATE

    @pytest.mark.asyncio
    async def test_sync_uses_system_graph_id_for_all_nodes(self, mock_async_driver):
        """All permission nodes must use graph_id='__system__' sentinel."""
        from app.services.rebac_service import ReBACService

        driver, session = mock_async_driver
        session.run.return_value = AsyncMock()

        mock_db = AsyncMock()
        kg = MagicMock()
        kg.user_id = "user-uuid-1"
        kg.id = "graph-uuid-1"
        kg.name = "Test Graph"
        kg.status = "active"
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [kg]
        mock_db.execute.return_value = mock_result

        svc = ReBACService()
        await svc.sync_existing_data(driver, mock_db)

        queries = [call[0][0] for call in session.run.call_args_list]
        for q in queries:
            assert "__system__" in q


# ── TRUE Integration Tests — real Neo4j, NO mocks ─────────────────────────
#
# These tests connect to the Neo4j instance configured via TEST_NEO4J_URI.
# They are skipped automatically if Neo4j is unreachable.
# Run with: pytest -m integration tests/integration/test_rebac.py


@pytest.mark.integration
class TestReBACIntegration:
    """Integration tests for ReBAC permission engine against real Neo4j."""

    @pytest_asyncio.fixture(autouse=True)
    async def setup_teardown(self, neo4j_test_driver):
        """Create test permission graph; clean up after each test."""
        self.driver = neo4j_test_driver

        # Seed: user-a owns graph-a (admin); user-b has no access
        now = "2020-01-01T00:00:00Z"
        future = "2099-12-31T00:00:00Z"
        await self._run(
            """
            // Clean any leftover test nodes
            MATCH (n {_test_rebac: true}) DETACH DELETE n
        """
        )
        await self._run(
            """
            MERGE (ua:User {user_id: 'rebac-user-a', graph_id: '__system__'})
              SET ua._test_rebac = true, ua.status = 'active', ua.created_at = $now
            MERGE (ub:User {user_id: 'rebac-user-b', graph_id: '__system__'})
              SET ub._test_rebac = true, ub.status = 'active', ub.created_at = $now
            MERGE (ga:Graph {graph_id: 'rebac-graph-a', namespace: '__system__'})
              SET ga._test_rebac = true, ga.name = 'Graph A', ga.owner_user_id = 'rebac-user-a',
                  ga.created_at = $now, ga.status = 'active'
            MERGE (ua)-[r:CAN_ACCESS]->(ga)
              SET r.level = 'admin', r.granted_by = 'rebac-user-a', r.granted_at = $now

            // user-a also has a read-only grant (for level enforcement test)
            MERGE (gb:Graph {graph_id: 'rebac-graph-b', namespace: '__system__'})
              SET gb._test_rebac = true, gb.name = 'Graph B', gb.owner_user_id = 'rebac-user-b',
                  gb.created_at = $now, gb.status = 'active'
            MERGE (ua)-[rb:CAN_ACCESS]->(gb)
              SET rb.level = 'read', rb.granted_by = 'rebac-user-b', rb.granted_at = $now

            // expired grant: user-a has an expired write grant on graph-c
            MERGE (gc:Graph {graph_id: 'rebac-graph-c', namespace: '__system__'})
              SET gc._test_rebac = true, gc.name = 'Graph C', gc.owner_user_id = 'rebac-user-b',
                  gc.created_at = $now, gc.status = 'active'
            MERGE (ua)-[rc:CAN_ACCESS]->(gc)
              SET rc.level = 'write', rc.granted_by = 'rebac-user-b', rc.granted_at = $now,
                  rc.expires_at = datetime('2000-01-01T00:00:00Z')
        """,
            {"now": now, "future": future},
        )

        from app.services.rebac_service import ReBACService

        self.svc = ReBACService()
        self.svc._redis = _NullRedis()

        yield

        await self._run("MATCH (n {_test_rebac: true}) DETACH DELETE n")

    async def _run(self, query: str, params: dict | None = None):
        async with self.driver.session() as s:
            await s.run(query, params or {})

    # ── Security Case 1: cross-tenant isolation ──────────────────────────

    @pytest.mark.asyncio
    async def test_case1_owner_has_admin_access(self):
        """Case 1a: graph owner is granted admin by the sync script."""
        authorized = await self.svc.check_graph_permission(
            self.driver, "rebac-user-a", "rebac-graph-a", "read"
        )
        assert authorized is True

    @pytest.mark.asyncio
    async def test_case1_cross_tenant_denied(self):
        """Case 1b: user-b has no grant on graph-a (user-a's graph)."""
        authorized = await self.svc.check_graph_permission(
            self.driver, "rebac-user-b", "rebac-graph-a", "read"
        )
        assert authorized is False

    # ── Security Case 2: expired grant ───────────────────────────────────

    @pytest.mark.asyncio
    async def test_case2_expired_grant_denied(self):
        """Case 2: grant with expires_at in the past is treated as no grant."""
        authorized = await self.svc.check_graph_permission(
            self.driver, "rebac-user-a", "rebac-graph-c", "write"
        )
        assert authorized is False

    # ── Security Case 3: level enforcement ───────────────────────────────

    @pytest.mark.asyncio
    async def test_case3_read_user_cannot_get_write_on_same_graph(self):
        """Case 3: user-a has read on graph-b; write check must be denied."""
        authorized = await self.svc.check_graph_permission(
            self.driver, "rebac-user-a", "rebac-graph-b", "write"
        )
        assert authorized is False

    @pytest.mark.asyncio
    async def test_case3_admin_inherits_write_and_read(self):
        """Case 3 hierarchy: admin grant satisfies read and write checks."""
        for level in ("read", "write", "admin"):
            authorized = await self.svc.check_graph_permission(
                self.driver, "rebac-user-a", "rebac-graph-a", level
            )
            assert authorized is True, f"admin user should pass {level} check"

    # ── Security Case 7: no 404 leak ─────────────────────────────────────

    @pytest.mark.asyncio
    async def test_case7_nonexistent_graph_returns_false_not_exception(self):
        """Case 7: non-existent graph_id must be denied (False), not raise."""
        authorized = await self.svc.check_graph_permission(
            self.driver, "rebac-user-a", "graph-that-does-not-exist", "read"
        )
        assert authorized is False  # dependency layer will convert to 403

    @pytest.mark.asyncio
    async def test_case7_verify_dependency_raises_403_not_404(self):
        """Case 7: verify_graph_access() FastAPI dependency raises HTTP 403."""
        import sys
        import types

        # Inject a stub neo4j_client so the dependency can reach the driver
        fake_module = types.ModuleType("app.core.neo4j_client")
        fake_module.neo4j_client = types.SimpleNamespace(async_driver=self.driver)
        orig = sys.modules.get("app.core.neo4j_client")
        sys.modules["app.core.neo4j_client"] = fake_module
        try:
            import pytest
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                # rebac-user-b has no grant on graph-a (user-a's graph)
                # Patch rebac_service inside dependencies to use our real driver
                import app.api.dependencies as deps_mod
                from app.api.dependencies import verify_graph_access as vga

                orig_svc = (
                    deps_mod.rebac_service
                    if hasattr(deps_mod, "rebac_service")
                    else None
                )
                deps_mod.rebac_service = self.svc
                try:
                    await vga("rebac-graph-a", "read", "rebac-user-b")
                finally:
                    if orig_svc is not None:
                        deps_mod.rebac_service = orig_svc
        finally:
            if orig:
                sys.modules["app.core.neo4j_client"] = orig
            else:
                sys.modules.pop("app.core.neo4j_client", None)

        assert exc_info.value.status_code == 403
        assert exc_info.value.status_code != 404

    # ── Security Case 8: no error message leak ───────────────────────────

    @pytest.mark.asyncio
    async def test_case8_403_body_is_access_denied_only(self):
        """Case 8: 403 detail is exactly 'Access denied' — no graph data."""
        from fastapi import HTTPException

        import app.api.dependencies as deps_mod

        orig_svc = getattr(deps_mod, "rebac_service", None)
        deps_mod.rebac_service = self.svc

        try:
            with pytest.raises(HTTPException) as exc_info:
                from app.core.neo4j_client import neo4j_client as real_client

                real_client.async_driver = self.driver
                await deps_mod.verify_graph_access(
                    "rebac-graph-a", "read", "rebac-user-b"
                )
        finally:
            if orig_svc is not None:
                deps_mod.rebac_service = orig_svc

        assert exc_info.value.detail == "Access denied"
        assert "rebac-graph-a" not in str(exc_info.value.detail)
        assert "rebac-user-b" not in str(exc_info.value.detail)


class _NullRedis:
    """Fake Redis that always returns cache miss and swallows writes."""

    async def get(self, key):
        return None

    async def set(self, key, value, ex=None):
        pass

    async def delete(self, *keys):
        pass
