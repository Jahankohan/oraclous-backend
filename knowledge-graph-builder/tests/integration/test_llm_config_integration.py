"""Integration tests for LLM Config endpoints (STORY-021 / TASK-040).

Runs against the Docker Neo4j instance. The credential-broker HTTP calls are
mocked via FastAPI dependency_overrides so no live broker is required.

Tests cover:
1. Create org-level config → 201, config_id returned, api_key never in response
2. List org-level configs → includes the created config
3. Delete org-level config → 204; second delete → 404
4. Create project-level config → 201 (verify_graph_access mocked)
5. List project-level configs → includes created config
6. Cross-tenant isolation — org A's config not visible to org B
7. Broker error → 400 surfaced to caller
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

try:
    from neo4j import AsyncDriver
    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────

_ORG_A = f"integ-llmcfg-orgA-{uuid.uuid4().hex[:8]}"
_ORG_B = f"integ-llmcfg-orgB-{uuid.uuid4().hex[:8]}"
_GID_A = f"integ-llmcfg-graph-{uuid.uuid4().hex[:8]}"
_USER = "integ-llmcfg-user-001"

pytestmark = pytest.mark.skipif(
    not _NEO4J_AVAILABLE, reason="neo4j driver not installed"
)

_CREATE_BODY = {
    "provider": "openrouter",
    "model": "openai/gpt-4o",
    "api_key": "sk-or-live-test-1234",
    "base_url": "https://openrouter.ai/api/v1",
}

# The llm_configs router is mounted under /api/v1 inside the api_router which is
# itself mounted at /api/v1 in main.py — resulting in the double prefix.
_ORG_PREFIX = "/api/v1/api/v1/org/llm-configs"
_GRAPH_PREFIX = "/api/v1/api/v1/graphs"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_verify():
    return patch(
        "app.api.v1.endpoints.llm_configs.verify_graph_access",
        new=AsyncMock(return_value=None),
    )


def _make_mock_broker(cred_id: str = "cred-mock-001") -> AsyncMock:
    mock = AsyncMock()
    mock.store_api_key = AsyncMock(return_value=cred_id)
    mock.retrieve_api_key = AsyncMock(return_value="sk-or-live-test-1234")
    mock.delete_credential = AsyncMock(return_value=None)
    return mock


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _cleanup(neo4j_test_driver: AsyncDriver):
    async def _wipe():
        await neo4j_test_driver.execute_query(
            "MATCH (n:LLMConfig) WHERE n.org_id IN [$a, $b] OR n.graph_id = $g DETACH DELETE n",
            {"a": _ORG_A, "b": _ORG_B, "g": _GID_A},
        )
    await _wipe()
    yield
    await _wipe()


@pytest.fixture(autouse=True)
def _override_auth(async_client):
    from app.main import app
    from app.api.dependencies import get_current_user, get_current_user_id

    app.dependency_overrides[get_current_user] = lambda: {
        "id": _USER,
        "tenant_id": _ORG_A,
    }
    app.dependency_overrides[get_current_user_id] = lambda: _USER
    yield
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_user_id, None)


@pytest.fixture(autouse=True)
def _override_llm_config_service(neo4j_test_driver: AsyncDriver):
    from app.main import app
    from app.api.v1.endpoints.llm_configs import _llm_config_service
    from app.services.llm_config_service import LLMConfigService

    app.dependency_overrides[_llm_config_service] = lambda: LLMConfigService(neo4j_test_driver)
    yield
    app.dependency_overrides.pop(_llm_config_service, None)


@pytest.fixture(autouse=True)
def _override_broker():
    from app.main import app
    from app.api.v1.endpoints.llm_configs import _broker

    mock = _make_mock_broker()
    app.dependency_overrides[_broker] = lambda: mock
    yield mock
    app.dependency_overrides.pop(_broker, None)


# ── Org-level tests ───────────────────────────────────────────────────────────


class TestOrgLevelConfig:
    async def test_create_returns_201_with_config_id(self, async_client):
        resp = await async_client.post(_ORG_PREFIX, json=_CREATE_BODY)
        assert resp.status_code == 201
        body = resp.json()
        assert "config_id" in body
        assert len(body["config_id"]) == 36  # UUID

    async def test_create_api_key_never_in_response(self, async_client):
        resp = await async_client.post(_ORG_PREFIX, json=_CREATE_BODY)
        assert resp.status_code == 201
        text = resp.text
        assert "sk-or-live-test" not in text
        assert "api_key" not in text.lower().replace("api_key_masked", "")

    async def test_list_returns_created_config(self, async_client):
        create = await async_client.post(_ORG_PREFIX, json=_CREATE_BODY)
        config_id = create.json()["config_id"]

        resp = await async_client.get(_ORG_PREFIX)
        assert resp.status_code == 200
        ids = [c["config_id"] for c in resp.json()]
        assert config_id in ids

    async def test_list_never_returns_api_key(self, async_client):
        await async_client.post(_ORG_PREFIX, json=_CREATE_BODY)

        resp = await async_client.get(_ORG_PREFIX)
        assert resp.status_code == 200
        assert "sk-or-live-test" not in resp.text
        for cfg in resp.json():
            assert "api_key" not in cfg or cfg.get("api_key") is None

    async def test_delete_returns_204(self, async_client):
        create = await async_client.post(_ORG_PREFIX, json=_CREATE_BODY)
        config_id = create.json()["config_id"]

        resp = await async_client.delete(f"{_ORG_PREFIX}/{config_id}")
        assert resp.status_code == 204

    async def test_delete_twice_returns_404(self, async_client):
        create = await async_client.post(_ORG_PREFIX, json=_CREATE_BODY)
        config_id = create.json()["config_id"]

        await async_client.delete(f"{_ORG_PREFIX}/{config_id}")
        resp = await async_client.delete(f"{_ORG_PREFIX}/{config_id}")
        assert resp.status_code == 404

    async def test_delete_nonexistent_returns_404(self, async_client):
        resp = await async_client.delete(f"{_ORG_PREFIX}/{uuid.uuid4()}")
        assert resp.status_code == 404

    async def test_list_excludes_deleted_config(self, async_client):
        create = await async_client.post(_ORG_PREFIX, json=_CREATE_BODY)
        config_id = create.json()["config_id"]

        await async_client.delete(f"{_ORG_PREFIX}/{config_id}")

        resp = await async_client.get(_ORG_PREFIX)
        ids = [c["config_id"] for c in resp.json()]
        assert config_id not in ids


# ── Project-level tests ────────────────────────────────────────────────────────


class TestProjectLevelConfig:
    async def test_create_returns_201_with_config_id(self, async_client):
        with _mock_verify():
            resp = await async_client.post(
                f"{_GRAPH_PREFIX}/{_GID_A}/llm-configs", json=_CREATE_BODY
            )
        assert resp.status_code == 201
        assert "config_id" in resp.json()

    async def test_list_returns_created_config(self, async_client):
        with _mock_verify():
            create = await async_client.post(
                f"{_GRAPH_PREFIX}/{_GID_A}/llm-configs", json=_CREATE_BODY
            )
        config_id = create.json()["config_id"]

        with _mock_verify():
            resp = await async_client.get(f"{_GRAPH_PREFIX}/{_GID_A}/llm-configs")
        assert resp.status_code == 200
        ids = [c["config_id"] for c in resp.json()]
        assert config_id in ids

    async def test_create_api_key_never_in_response(self, async_client):
        with _mock_verify():
            resp = await async_client.post(
                f"{_GRAPH_PREFIX}/{_GID_A}/llm-configs", json=_CREATE_BODY
            )
        assert resp.status_code == 201
        assert "sk-or-live-test" not in resp.text


# ── Cross-tenant isolation ────────────────────────────────────────────────────


class TestCrossTenantIsolation:
    async def test_org_b_cannot_see_org_a_configs(self, async_client):
        create = await async_client.post(_ORG_PREFIX, json=_CREATE_BODY)
        config_id = create.json()["config_id"]

        from app.main import app
        from app.api.dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: {
            "id": "other-user",
            "tenant_id": _ORG_B,
        }
        try:
            resp = await async_client.get(_ORG_PREFIX)
            ids = [c["config_id"] for c in resp.json()]
            assert config_id not in ids
        finally:
            app.dependency_overrides[get_current_user] = lambda: {
                "id": _USER,
                "tenant_id": _ORG_A,
            }

    async def test_org_b_delete_of_org_a_config_returns_404(self, async_client):
        create = await async_client.post(_ORG_PREFIX, json=_CREATE_BODY)
        config_id = create.json()["config_id"]

        from app.main import app
        from app.api.dependencies import get_current_user

        app.dependency_overrides[get_current_user] = lambda: {
            "id": "attacker",
            "tenant_id": _ORG_B,
        }
        try:
            resp = await async_client.delete(f"{_ORG_PREFIX}/{config_id}")
            assert resp.status_code == 404
        finally:
            app.dependency_overrides[get_current_user] = lambda: {
                "id": _USER,
                "tenant_id": _ORG_A,
            }


# ── Broker error propagation ──────────────────────────────────────────────────


class TestBrokerError:
    async def test_broker_error_surfaces_as_400(self, async_client):
        from app.main import app
        from app.api.v1.endpoints.llm_configs import _broker
        from app.services.credential_broker_client import CredentialBrokerError

        failing_broker = AsyncMock()
        failing_broker.store_api_key = AsyncMock(
            side_effect=CredentialBrokerError("vault unavailable")
        )

        app.dependency_overrides[_broker] = lambda: failing_broker
        try:
            resp = await async_client.post(_ORG_PREFIX, json=_CREATE_BODY)
        finally:
            mock = _make_mock_broker()
            app.dependency_overrides[_broker] = lambda: mock

        assert resp.status_code == 400
        assert "vault unavailable" in resp.json()["detail"]
