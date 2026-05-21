"""
Unit tests for ReBAC Phase B — ORA-48 Role/Permission/SubGraph model.

Covers ORA-48 test criteria T1-T9 at the service level (mocked Neo4j).
All tests use mock drivers — no live Neo4j required.

Marked @pytest.mark.unit.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_driver(single_return=None, iter_return=None):
    """
    Build a mock async Neo4j driver.

    single_return: value returned by result.single()
    iter_return:   list of row dicts returned when iterating over result
    """
    session = AsyncMock()

    result = AsyncMock()
    result.single = AsyncMock(return_value=single_return)

    async def _aiter(self):
        for row in iter_return or []:
            yield MagicMock(**row, __getitem__=lambda s, k, _row=row: _row[k])

    result.__aiter__ = _aiter
    session.run = AsyncMock(return_value=result)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = cm
    return driver, session, result


def _null_redis():
    redis = AsyncMock()
    redis.get.return_value = None
    redis.set.return_value = True
    redis.delete.return_value = 1
    return redis


# ── T9 — graph_id validation ───────────────────────────────────────────────


@pytest.mark.unit
class TestGraphIdValidation:
    """T9: Any permission check missing graph_id raises ValueError."""

    @pytest.mark.asyncio
    async def test_check_permission_raises_without_graph_id(self):
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        svc._redis = _null_redis()
        driver, _, _ = _make_driver()
        with pytest.raises(ValueError, match="graph_id is required"):
            await svc.check_graph_permission(driver, "user-a", "", "read")

    @pytest.mark.asyncio
    async def test_grant_role_raises_without_graph_id(self):
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        driver, _, _ = _make_driver()
        with pytest.raises(ValueError):
            await svc.grant_role(driver, "", "user-a", "viewer", "admin-user")

    @pytest.mark.asyncio
    async def test_revoke_role_raises_without_graph_id(self):
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        driver, _, _ = _make_driver()
        with pytest.raises(ValueError):
            await svc.revoke_role(driver, "", "user-a", "viewer")

    @pytest.mark.asyncio
    async def test_list_members_raises_without_graph_id(self):
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        driver, _, _ = _make_driver()
        with pytest.raises(ValueError):
            await svc.list_graph_members(driver, "")

    @pytest.mark.asyncio
    async def test_create_subgraph_raises_without_graph_id(self):
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        driver, _, _ = _make_driver()
        with pytest.raises(ValueError):
            await svc.create_subgraph(driver, "", "sg-name")


# ── T1/T2 — owner read / viewer no write ──────────────────────────────────


@pytest.mark.unit
class TestPermissionCheckPhaseB:
    """
    Phase B check_graph_permission: HAS_ROLE traversal via mocked driver.
    The driver session.run is called twice:
      1. Permission check query → returns authorized
      2. Role existence check → returns cnt > 0 (Phase B data present)
    """

    def _svc_with_phase_b(self, perm_authorized: bool):
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        svc._redis = _null_redis()

        # Two sequential session.run calls per check_graph_permission
        perm_result = AsyncMock()
        perm_result.single = AsyncMock(return_value={"authorized": perm_authorized})

        role_result = AsyncMock()
        role_result.single = AsyncMock(return_value={"cnt": 1})  # Phase B has data

        session = AsyncMock()
        session.run = AsyncMock(side_effect=[perm_result, role_result])

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session.return_value = cm
        return svc, driver

    @pytest.mark.asyncio
    async def test_t1_owner_can_read(self):
        """T1: Graph owner with graph:read permission returns True."""
        svc, driver = self._svc_with_phase_b(True)
        result = await svc.check_graph_permission(
            driver, "owner-user", "graph-1", "read"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_t2_viewer_cannot_write(self):
        """T2: Viewer without graph:write permission returns False."""
        svc, driver = self._svc_with_phase_b(False)
        result = await svc.check_graph_permission(
            driver, "viewer-user", "graph-1", "write"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_phase_a_fallback_when_no_phase_b_data(self):
        """When Phase B has no Role nodes (cnt=0), fall back to Phase A CAN_ACCESS."""
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        svc._redis = _null_redis()

        # Phase B: perm check returns None (no rows), role count = 0
        perm_result = AsyncMock()
        perm_result.single = AsyncMock(return_value={"authorized": False})

        role_result = AsyncMock()
        role_result.single = AsyncMock(return_value={"cnt": 0})  # no Phase B data

        # Phase A returns authorized = True (legacy CAN_ACCESS)
        phase_a_result = AsyncMock()
        phase_a_result.single = AsyncMock(return_value={"authorized": True})

        session = AsyncMock()
        session.run = AsyncMock(side_effect=[perm_result, role_result, phase_a_result])

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session.return_value = cm

        result = await svc.check_graph_permission(
            driver, "legacy-user", "graph-1", "read"
        )
        assert result is True  # Phase A fallback authorized

    @pytest.mark.asyncio
    async def test_cypher_uses_parameterized_queries(self):
        """T9 injection: user_id must not appear as literal in the Cypher query string."""
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        svc._redis = _null_redis()

        perm_result = AsyncMock()
        perm_result.single = AsyncMock(return_value={"authorized": False})
        role_result = AsyncMock()
        role_result.single = AsyncMock(return_value={"cnt": 0})
        phase_a_result = AsyncMock()
        phase_a_result.single = AsyncMock(return_value={"authorized": False})

        session = AsyncMock()
        session.run = AsyncMock(side_effect=[perm_result, role_result, phase_a_result])

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session.return_value = cm

        injection = "admin'; DROP DATABASE neo4j; --"
        await svc.check_graph_permission(driver, injection, "graph-1", "read")

        for call in session.run.call_args_list:
            query = call[0][0]
            assert injection not in query, (
                "Injection string must never appear in query text"
            )
            assert "$user_id" in query or "$acceptable" in query or "graph_id" in query

    @pytest.mark.asyncio
    async def test_neo4j_error_returns_false_fail_closed(self):
        """Neo4j failure must deny access (fail-closed) — never raise."""
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        svc._redis = _null_redis()

        session = AsyncMock()
        session.run.side_effect = Exception("Neo4j connection lost")

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session.return_value = cm

        result = await svc.check_graph_permission(driver, "user-a", "graph-1", "read")
        assert result is False

    @pytest.mark.asyncio
    async def test_cache_hit_skips_neo4j(self):
        """Redis cache hit must bypass Neo4j entirely."""
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        redis = _null_redis()
        redis.get.return_value = "1"  # cache hit = authorized
        svc._redis = redis

        driver, session, _ = _make_driver()
        result = await svc.check_graph_permission(driver, "user-a", "graph-1", "read")

        assert result is True
        session.run.assert_not_called()


# ── bootstrap_graph_roles ─────────────────────────────────────────────────


@pytest.mark.unit
class TestBootstrapGraphRoles:
    """T8: New graph must get 5 system roles + HAS_PERMISSION + HAS_ROLE for owner."""

    @pytest.mark.asyncio
    async def test_bootstrap_calls_run_for_each_role(self):
        """Bootstrap runs at least 5 MERGE role queries."""
        from app.services.rebac_service import _SYSTEM_ROLES, ReBACService

        svc = ReBACService()
        svc._redis = _null_redis()

        session = AsyncMock()
        result_mock = AsyncMock()
        result_mock.single = AsyncMock(return_value={"role_id": "some-id"})
        session.run = AsyncMock(return_value=result_mock)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session.return_value = cm

        await svc.bootstrap_graph_roles(driver, "graph-1", owner_user_id="user-a")

        call_count = session.run.call_count
        # At minimum: 5 roles + N perm edges + 5 inheritance + 1 owner grant
        assert call_count >= len(_SYSTEM_ROLES) + 1

    @pytest.mark.asyncio
    async def test_bootstrap_all_queries_include_graph_id_param(self):
        """T8 + Rule #4: every query in bootstrap must pass graph_id as parameter."""
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        svc._redis = _null_redis()

        session = AsyncMock()
        result_mock = AsyncMock()
        result_mock.single = AsyncMock(return_value={"role_id": "some-id"})
        session.run = AsyncMock(return_value=result_mock)

        cm = MagicMock()
        cm.__aenter__ = AsyncMock(return_value=session)
        cm.__aexit__ = AsyncMock(return_value=False)
        driver = MagicMock()
        driver.session.return_value = cm

        await svc.bootstrap_graph_roles(
            driver, "test-graph-123", owner_user_id="user-a"
        )

        for call in session.run.call_args_list:
            params = call[0][1] if len(call[0]) > 1 else {}
            assert "graph_id" in params, f"Query missing graph_id param: {call}"
            assert params["graph_id"] == "test-graph-123"


# ── grant_role / revoke_role ───────────────────────────────────────────────


@pytest.mark.unit
class TestRoleManagement:
    @pytest.mark.asyncio
    async def test_grant_role_runs_merge_query(self):
        """grant_role issues a single MERGE/SET query parameterized correctly."""
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        svc._redis = _null_redis()
        driver, session, _ = _make_driver(single_return=None)

        await svc.grant_role(
            driver, "graph-1", "user-b", "viewer", "admin-user", email="b@test.com"
        )

        assert session.run.called
        call = session.run.call_args_list[0]
        query, params = call[0][0], call[0][1]
        assert "$user_id" in query
        assert "$graph_id" in query
        assert "$role_name" in query
        assert params["graph_id"] == "graph-1"
        assert params["user_id"] == "user-b"
        assert params["role_name"] == "viewer"

    @pytest.mark.asyncio
    async def test_revoke_role_returns_zero_when_not_found(self):
        """revoke_role returns 0 when no matching edge exists."""
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        svc._redis = _null_redis()
        driver, session, _ = _make_driver(single_return={"revoked_count": 0})

        count = await svc.revoke_role(driver, "graph-1", "user-b", "viewer")
        assert count == 0

    @pytest.mark.asyncio
    async def test_t6_revoke_invalidates_cache(self):
        """T6: After revoke, permission cache is invalidated for user+graph."""
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        redis = _null_redis()
        svc._redis = redis
        driver, session, _ = _make_driver(single_return={"revoked_count": 1})

        await svc.revoke_role(driver, "graph-1", "user-b", "editor")

        assert redis.delete.called
        deleted_keys = [str(c[0]) for c in redis.delete.call_args_list]
        # at least one delete call should reference user-b and graph-1
        combined = " ".join(deleted_keys)
        assert "user-b" in combined
        assert "graph-1" in combined

    @pytest.mark.asyncio
    async def test_t7_cross_tenant_isolation_via_graph_id_param(self):
        """T7: grant_role on graph-A must not affect graph-B (graph_id passed correctly)."""
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        svc._redis = _null_redis()
        driver, session, _ = _make_driver(single_return=None)

        await svc.grant_role(driver, "graph-A", "user-x", "viewer", "admin-A")

        call = session.run.call_args_list[0]
        params = call[0][1]
        assert params["graph_id"] == "graph-A"
        # graph-B must never appear anywhere in the params
        assert "graph-B" not in str(params)


# ── list_graph_members ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestListGraphMembers:
    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_members(self):
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        driver, session, result = _make_driver(iter_return=[])

        members = await svc.list_graph_members(driver, "graph-1")
        assert members == []

    @pytest.mark.asyncio
    async def test_query_includes_graph_id(self):
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        driver, session, _ = _make_driver(iter_return=[])

        await svc.list_graph_members(driver, "graph-xyz")

        call = session.run.call_args_list[0]
        query, params = call[0][0], call[0][1]
        assert "$graph_id" in query
        assert params["graph_id"] == "graph-xyz"


# ── SubGraph management ───────────────────────────────────────────────────


@pytest.mark.unit
class TestSubGraphManagement:
    @pytest.mark.asyncio
    async def test_create_subgraph_returns_dict_with_required_keys(self):
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        driver, session, _ = _make_driver(
            single_return={
                "subgraph_id": "sg-001",
                "name": "HR Confidential",
                "description": "HR data",
                "created_at": "2026-01-01T00:00:00",
            }
        )

        result = await svc.create_subgraph(
            driver, "graph-1", "HR Confidential", "HR data", "user-a"
        )
        assert result["subgraph_id"] == "sg-001"
        assert result["graph_id"] == "graph-1"
        assert result["name"] == "HR Confidential"

    @pytest.mark.asyncio
    async def test_create_subgraph_query_includes_graph_id(self):
        from app.services.rebac_service import ReBACService

        svc = ReBACService()
        driver, session, _ = _make_driver(
            single_return={
                "subgraph_id": "sg-001",
                "name": "Finance",
                "description": None,
                "created_at": None,
            }
        )

        await svc.create_subgraph(driver, "graph-fin", "Finance")

        call = session.run.call_args_list[0]
        params = call[0][1]
        assert params["graph_id"] == "graph-fin"


# ── Acceptable levels (Phase A backward compat) ────────────────────────────


@pytest.mark.unit
class TestAcceptableLevels:
    def test_level_hierarchy(self):
        from app.services.rebac_service import _ACCEPTABLE_LEVELS

        assert set(_ACCEPTABLE_LEVELS["read"]) == {"read", "write", "admin"}
        assert set(_ACCEPTABLE_LEVELS["write"]) == {"write", "admin"}
        assert set(_ACCEPTABLE_LEVELS["admin"]) == {"admin"}
