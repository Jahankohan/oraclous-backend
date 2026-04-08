"""Integration tests for Service Account API endpoints.

Covers 6 scenarios from ORA-106 acceptance criteria:
1. Full lifecycle — create → get → rotate-key → revoke via HTTP
2. SA JWT auth path — service_account principal_type routes to SA permission check
3. Cross-graph grant — add → list → verify access → delete → access denied
4. List requires admin — non-admin caller gets 403
5. Cross-tenant isolation — user cannot create SA on another user's graph (403)
6. Revoked SA token — 401 on subsequent requests after token is invalidated

All tests use mocked Neo4j, auth-service, and service layer — no live services required.
"""

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_ID = str(uuid.uuid4())
USER_B_ID = str(uuid.uuid4())  # Different tenant user
SA_ID = str(uuid.uuid4())
HOME_GRAPH_ID = str(uuid.uuid4())
TARGET_GRAPH_ID = str(uuid.uuid4())
TENANT_ID = str(uuid.uuid4())
TENANT_B_ID = str(uuid.uuid4())

_NOW = datetime(2026, 4, 8, 12, 0, 0, tzinfo=timezone.utc).isoformat()

# Minimal user principal returned by auth_service.verify_token for a human user
FAKE_USER = {
    "id": USER_ID,
    "email": "user@example.com",
    "principal_type": "user",
    "tenant_id": TENANT_ID,
}

# SA principal — principal_type=service_account, carries tenant_id in JWT
FAKE_SA_PRINCIPAL = {
    "id": SA_ID,
    "principal_type": "service_account",
    "tenant_id": TENANT_ID,
}

# Mock SA record as returned by service_account_service methods
_SA_RECORD = {
    "service_account_id": SA_ID,
    "name": "integration-test-sa",
    "description": "created in integration tests",
    "home_graph_id": HOME_GRAPH_ID,
    "tenant_id": TENANT_ID,
    "status": "active",
    "key_prefix": "osk_testkey1",
    "created_at": _NOW,
    "last_used_at": None,
}

_SA_CREATED_RECORD = {**_SA_RECORD, "api_key": "osk_testkey1abc123def456gh"}

_SA_ROTATED_RECORD = {
    "service_account_id": SA_ID,
    "key_prefix": "osk_newkey12",
    "api_key": "osk_newkey12abc123def456",
    "rotated_at": _NOW,
}

_GRANT_RECORD = {
    "graph_id": TARGET_GRAPH_ID,
    "graph_name": "Target Graph",
    "level": "reader",
    "source": "explicit",
    "granted_by": USER_ID,
    "granted_at": _NOW,
    "expires_at": None,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_auth(principal: dict):
    """Patch auth_service.verify_token to return the given principal."""
    p = patch("app.api.dependencies.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(return_value=principal)
    return p


def _auth_headers() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _make_mock_driver():
    """Return a mock AsyncDriver that does nothing (session not reached for most paths)."""
    mock_driver = MagicMock()
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)
    mock_driver.session = MagicMock(return_value=mock_session)
    return mock_driver


def _patch_neo4j(mock_driver=None):
    """
    Patch neo4j_client in both the endpoint module and the dependencies module.
    Returns a tuple of (endpoint_patch, deps_patch) that must each be stopped.
    """
    if mock_driver is None:
        mock_driver = _make_mock_driver()

    ep = patch("app.api.v1.endpoints.service_accounts.neo4j_client")
    deps = patch("app.core.neo4j_client.neo4j_client")

    mock_ep = ep.start()
    mock_ep.async_driver = mock_driver

    mock_deps = deps.start()
    mock_deps.async_driver = mock_driver

    return ep, deps


# ---------------------------------------------------------------------------
# Scenario 1: Full lifecycle — create → get → rotate-key → revoke
# ---------------------------------------------------------------------------


class TestServiceAccountFullLifecycle:
    """HTTP lifecycle: create, retrieve, rotate key, revoke."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_create_service_account_returns_201_with_api_key(self, async_client):
        """POST /graphs/{graphId}/service-accounts → 201 with api_key in body."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.api.dependencies.rebac_service") as mock_rebac,
            ):
                mock_svc.create_service_account = AsyncMock(
                    return_value=_SA_CREATED_RECORD
                )
                mock_rebac.check_graph_permission = AsyncMock(return_value=True)

                response = await async_client.post(
                    f"/api/v1/graphs/{HOME_GRAPH_ID}/service-accounts",
                    json={
                        "name": "integration-test-sa",
                        "description": "created in integration tests",
                        "level": "reader",
                    },
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 201
        data = response.json()
        assert data["service_account_id"] == SA_ID
        assert data["name"] == "integration-test-sa"
        assert "api_key" in data
        assert data["api_key"].startswith("osk_")
        assert data["status"] == "active"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_service_account_returns_200_with_metadata(self, async_client):
        """GET /service-accounts/{accountId} → 200 with SA metadata (no api_key)."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.api.dependencies.rebac_service") as mock_rebac,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                mock_rebac.check_graph_permission = AsyncMock(return_value=True)

                response = await async_client.get(
                    f"/api/v1/service-accounts/{SA_ID}",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["service_account_id"] == SA_ID
        assert "api_key" not in data  # Never returned after creation
        assert data["status"] == "active"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_rotate_key_returns_200_with_new_api_key(self, async_client):
        """POST /service-accounts/{accountId}/rotate-key → 200 with new api_key."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.api.dependencies.rebac_service") as mock_rebac,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                mock_svc.rotate_key = AsyncMock(return_value=_SA_ROTATED_RECORD)
                mock_rebac.check_graph_permission = AsyncMock(return_value=True)

                response = await async_client.post(
                    f"/api/v1/service-accounts/{SA_ID}/rotate-key",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["service_account_id"] == SA_ID
        assert "api_key" in data
        assert data["api_key"] != _SA_CREATED_RECORD["api_key"]  # New key, not old
        assert data["api_key"].startswith("osk_")
        assert "rotated_at" in data

    @pytest.mark.integration
    @pytest.mark.api
    async def test_revoke_service_account_returns_204(self, async_client):
        """DELETE /service-accounts/{accountId} → 204 No Content."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.api.dependencies.rebac_service") as mock_rebac,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                mock_svc.revoke_service_account = AsyncMock(return_value=True)
                mock_rebac.check_graph_permission = AsyncMock(return_value=True)

                response = await async_client.delete(
                    f"/api/v1/service-accounts/{SA_ID}",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 204
        assert response.content == b""  # No body on 204


# ---------------------------------------------------------------------------
# Scenario 2: SA JWT auth path works (vs. user JWT)
# ---------------------------------------------------------------------------


class TestServiceAccountJWTAuthPath:
    """SA principal (principal_type=service_account) routes through SA ACL check."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_sa_principal_can_get_own_metadata(self, async_client):
        """SA JWT calling GET /service-accounts/{id} → routes to SA permission check."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_SA_PRINCIPAL)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.services.service_account_service.service_account_service") as mock_dep_svc,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                # SA permission check (via verify_graph_access → SA path)
                mock_dep_svc.check_sa_graph_permission = AsyncMock(return_value=True)

                response = await async_client.get(
                    f"/api/v1/service-accounts/{SA_ID}",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["service_account_id"] == SA_ID

    @pytest.mark.integration
    @pytest.mark.api
    async def test_sa_principal_denied_without_permission(self, async_client):
        """SA JWT with no CAN_ACCESS edge → verify_graph_access returns 403."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_SA_PRINCIPAL)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.services.service_account_service.service_account_service") as mock_dep_svc,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                # SA lacks permission on home graph
                mock_dep_svc.check_sa_graph_permission = AsyncMock(return_value=False)

                response = await async_client.get(
                    f"/api/v1/service-accounts/{SA_ID}",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_sa_principal_cannot_self_grant(self, async_client):
        """SA JWT calling POST /service-accounts/{id}/graph-grants → 403 (no self-elevation)."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_SA_PRINCIPAL)
        try:
            with patch(
                "app.api.v1.endpoints.service_accounts.service_account_service"
            ) as mock_svc:
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)

                response = await async_client.post(
                    f"/api/v1/service-accounts/{SA_ID}/graph-grants",
                    json={"graph_id": TARGET_GRAPH_ID, "level": "reader"},
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 403
        assert "cannot grant" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Scenario 3: Cross-graph grant lifecycle
# ---------------------------------------------------------------------------


class TestCrossGraphGrant:
    """Add cross-graph grant → list grants → revoke grant."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_add_graph_grant_returns_201(self, async_client):
        """POST /service-accounts/{id}/graph-grants → 201 with grant details."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.api.dependencies.rebac_service") as mock_rebac,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                mock_svc.add_graph_grant = AsyncMock(return_value=_GRANT_RECORD)
                mock_rebac.check_graph_permission = AsyncMock(return_value=True)

                response = await async_client.post(
                    f"/api/v1/service-accounts/{SA_ID}/graph-grants",
                    json={"graph_id": TARGET_GRAPH_ID, "level": "reader"},
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 201
        data = response.json()
        assert data["graph_id"] == TARGET_GRAPH_ID
        assert data["level"] == "reader"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_list_graph_grants_returns_all_grants(self, async_client):
        """GET /service-accounts/{id}/graph-grants → 200 with list of grants."""
        second_grant = {
            "graph_id": str(uuid.uuid4()),
            "graph_name": "Another Graph",
            "level": "writer",
            "source": "explicit",
            "granted_by": USER_ID,
            "granted_at": _NOW,
            "expires_at": None,
        }
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.api.dependencies.rebac_service") as mock_rebac,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                mock_svc.list_graph_grants = AsyncMock(
                    return_value=[_GRANT_RECORD, second_grant]
                )
                mock_rebac.check_graph_permission = AsyncMock(return_value=True)

                response = await async_client.get(
                    f"/api/v1/service-accounts/{SA_ID}/graph-grants",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        graph_ids = {g["graph_id"] for g in data}
        assert TARGET_GRAPH_ID in graph_ids

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_graph_grant_returns_204(self, async_client):
        """DELETE /service-accounts/{id}/graph-grants/{graphId} → 204."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.api.dependencies.rebac_service") as mock_rebac,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                mock_svc.delete_graph_grant = AsyncMock(return_value=True)
                mock_rebac.check_graph_permission = AsyncMock(return_value=True)

                response = await async_client.delete(
                    f"/api/v1/service-accounts/{SA_ID}/graph-grants/{TARGET_GRAPH_ID}",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 204

    @pytest.mark.integration
    @pytest.mark.api
    async def test_sa_cannot_modify_grants_after_deletion(self, async_client):
        """DELETE grant then SA (SA principal) loses access → 403 on subsequent verify."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        # SA calls an endpoint — permission now denied (grant was revoked)
        auth_p = _patch_auth(FAKE_SA_PRINCIPAL)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.services.service_account_service.service_account_service") as mock_dep_svc,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                # check_sa_graph_permission returns False — grant was revoked
                mock_dep_svc.check_sa_graph_permission = AsyncMock(return_value=False)

                response = await async_client.get(
                    f"/api/v1/service-accounts/{SA_ID}",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Scenario 4: List endpoint requires admin permission (non-admin gets 403)
# ---------------------------------------------------------------------------


class TestListRequiresAdmin:
    """GET /graphs/{graphId}/service-accounts requires admin level."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_list_sa_returns_200_for_admin(self, async_client):
        """Admin caller → GET /graphs/{id}/service-accounts returns 200."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.api.dependencies.rebac_service") as mock_rebac,
            ):
                mock_svc.list_service_accounts = AsyncMock(return_value=[_SA_RECORD])
                # Admin check passes
                mock_rebac.check_graph_permission = AsyncMock(return_value=True)

                response = await async_client.get(
                    f"/api/v1/graphs/{HOME_GRAPH_ID}/service-accounts",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 200
        assert isinstance(response.json(), list)

    @pytest.mark.integration
    @pytest.mark.api
    async def test_list_sa_returns_403_for_non_admin(self, async_client):
        """Non-admin caller → GET /graphs/{id}/service-accounts returns 403."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with patch("app.api.dependencies.rebac_service") as mock_rebac:
                # Admin check fails — user is only reader/writer
                mock_rebac.check_graph_permission = AsyncMock(return_value=False)

                response = await async_client.get(
                    f"/api/v1/graphs/{HOME_GRAPH_ID}/service-accounts",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_list_sa_403_does_not_reveal_graph_existence(self, async_client):
        """403 response must not include graph_id or internal info."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with patch("app.api.dependencies.rebac_service") as mock_rebac:
                mock_rebac.check_graph_permission = AsyncMock(return_value=False)

                response = await async_client.get(
                    f"/api/v1/graphs/{HOME_GRAPH_ID}/service-accounts",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 403
        detail = response.json()["detail"]
        # Error detail must not leak graph_id or internal paths
        assert HOME_GRAPH_ID not in detail
        assert "/" not in detail


# ---------------------------------------------------------------------------
# Scenario 5: Cross-tenant isolation — user cannot create SA on another graph
# ---------------------------------------------------------------------------


class TestCrossTenantIsolation:
    """Multi-tenancy: SA operations are scoped to the caller's tenant."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_user_cannot_create_sa_on_another_users_graph(self, async_client):
        """User A (tenant A) cannot create SA on Graph B (tenant B) → 403."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        # User A is authenticated
        auth_p = _patch_auth(FAKE_USER)
        try:
            with patch("app.api.dependencies.rebac_service") as mock_rebac:
                # User A has no write access to Graph B
                mock_rebac.check_graph_permission = AsyncMock(return_value=False)

                response = await async_client.post(
                    f"/api/v1/graphs/{TARGET_GRAPH_ID}/service-accounts",
                    json={"name": "hijack-sa", "level": "reader"},
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.api
    async def test_user_cannot_get_sa_from_another_tenant(self, async_client):
        """User A cannot fetch SA metadata belonging to tenant B → 403."""
        # SA belongs to tenant B
        sa_tenant_b = {**_SA_RECORD, "tenant_id": TENANT_B_ID}
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)  # User A, tenant A
        try:
            with patch(
                "app.api.v1.endpoints.service_accounts.service_account_service"
            ) as mock_svc:
                # Service returns None — SA not found for this tenant (tenant isolation enforced in Cypher)
                mock_svc.get_service_account = AsyncMock(return_value=None)

                response = await async_client.get(
                    f"/api/v1/service-accounts/{SA_ID}",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        # Must return 403, not 404 — never reveal SA existence
        assert response.status_code == 403
        assert response.json()["detail"] == "Access denied"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_403_not_404_for_inaccessible_sa(self, async_client):
        """Access denial for unknown SA always returns 403, never 404."""
        unknown_sa_id = str(uuid.uuid4())
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with patch(
                "app.api.v1.endpoints.service_accounts.service_account_service"
            ) as mock_svc:
                mock_svc.get_service_account = AsyncMock(return_value=None)

                response = await async_client.get(
                    f"/api/v1/service-accounts/{unknown_sa_id}",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 403
        assert response.status_code != 404

    @pytest.mark.integration
    @pytest.mark.api
    async def test_update_sa_requires_write_access_to_home_graph(self, async_client):
        """PATCH /service-accounts/{id} requires write access — read-only user gets 403."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_USER)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.api.dependencies.rebac_service") as mock_rebac,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                # Caller has only read access, not write
                mock_rebac.check_graph_permission = AsyncMock(return_value=False)

                response = await async_client.patch(
                    f"/api/v1/service-accounts/{SA_ID}",
                    json={"name": "updated-name"},
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 403


# ---------------------------------------------------------------------------
# Scenario 6: Revoked SA token returns 401
# ---------------------------------------------------------------------------


class TestRevokedSAToken:
    """After revocation, the SA's token must be rejected at the auth boundary."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_revoked_sa_token_returns_401(self, async_client):
        """auth_service.verify_token raises 401 for revoked SA → endpoint returns 401."""
        from fastapi import HTTPException
        from fastapi import status as http_status

        auth_p = patch("app.api.dependencies.auth_service")
        mock_auth = auth_p.start()
        # Revoked token → auth-service returns 401
        mock_auth.verify_token = AsyncMock(
            side_effect=HTTPException(
                status_code=http_status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        )
        try:
            response = await async_client.get(
                f"/api/v1/service-accounts/{SA_ID}",
                headers=_auth_headers(),
            )
        finally:
            auth_p.stop()

        assert response.status_code == 401

    @pytest.mark.integration
    @pytest.mark.api
    async def test_missing_auth_header_returns_403(self, async_client):
        """Request with no Authorization header → 403 (HTTPBearer rejects it)."""
        response = await async_client.get(f"/api/v1/service-accounts/{SA_ID}")
        # HTTPBearer returns 403 when no credentials provided
        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.api
    async def test_active_sa_token_returns_200_before_revocation(self, async_client):
        """Sanity check: valid active SA token successfully authenticates."""
        mock_driver = _make_mock_driver()
        ep_p, deps_p = _patch_neo4j(mock_driver)
        auth_p = _patch_auth(FAKE_SA_PRINCIPAL)
        try:
            with (
                patch(
                    "app.api.v1.endpoints.service_accounts.service_account_service"
                ) as mock_svc,
                patch("app.services.service_account_service.service_account_service") as mock_dep_svc,
            ):
                mock_svc.get_service_account = AsyncMock(return_value=_SA_RECORD)
                mock_dep_svc.check_sa_graph_permission = AsyncMock(return_value=True)

                response = await async_client.get(
                    f"/api/v1/service-accounts/{SA_ID}",
                    headers=_auth_headers(),
                )
        finally:
            auth_p.stop()
            ep_p.stop()
            deps_p.stop()

        assert response.status_code == 200
