"""Integration tests for STORY-022: agent publish/unpublish/rotate-key and public chat.

Runs against the Docker Neo4j instance. Auth and Neo4j service dependencies
are overridden via FastAPI dependency_overrides and patch().

Coverage:
1. publish → 201, integration_key returned once (oak- prefix), key_hash stored not plaintext
2. get published → 201 key never re-exposed; key_last4 only
3. unpublish → 204; second unpublish → 404
4. rotate-key → new oak- key, old key rejected by public endpoint
5. public chat → 200 with mock agent executor
6. public chat → 401 on missing key
7. public chat → 401 on wrong key (same message as missing slug — no info leak)
8. public chat → 429 when rate limit exceeded
9. cross-tenant isolation — graph_id comes from :PublishedAgent, not caller
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

try:
    from neo4j import AsyncDriver

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _NEO4J_AVAILABLE, reason="neo4j driver not installed"
)

# ── Constants ──────────────────────────────────────────────────────────────────

_RUN = uuid.uuid4().hex[:8]
_GRAPH_ID = f"integ-story022-graph-{_RUN}"
_AGENT_ID = f"integ-story022-agent-{_RUN}"
_USER_ID = f"integ-story022-user-{_RUN}"
_SLUG = f"test-slug-{_RUN}"

# Integration router is mounted under /api/v1 inside api_router which is itself
# mounted at /api/v1 — resulting in the double prefix.
_PREFIX = f"/api/v1/api/v1/graphs/{_GRAPH_ID}/agents/{_AGENT_ID}"
_PUBLIC_PREFIX = f"/public/agents/{_SLUG}"

_PUBLISH_BODY = {
    "slug": _SLUG,
    "cors_origins": [],
    "rate_limit_rpm": 60,
    "egress_url": None,
}


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_executor_mock(response_text: str = "Hello from agent") -> MagicMock:
    result = MagicMock()
    result.response = response_text
    result.session_id = "sess-123"
    result.provenance = None

    mock_executor = MagicMock()
    mock_executor.run = AsyncMock(return_value=result)
    return mock_executor


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture(autouse=True)
async def _cleanup(neo4j_test_driver: AsyncDriver):
    async def _wipe():
        await neo4j_test_driver.execute_query(
            "MATCH (n:PublishedAgent) WHERE n.graph_id = $g DETACH DELETE n",
            {"g": _GRAPH_ID},
        )
        await neo4j_test_driver.execute_query(
            "MATCH (n:AuditEvent) WHERE n.graph_id = $g DETACH DELETE n",
            {"g": _GRAPH_ID},
        )
        await neo4j_test_driver.execute_query(
            "MATCH (n:Agent {agent_id: $a, graph_id: $g}) DETACH DELETE n",
            {"a": _AGENT_ID, "g": _GRAPH_ID},
        )

    await _wipe()
    # Seed the :Agent node required by publish_agent Cypher
    await neo4j_test_driver.execute_query(
        """
        MERGE (a:Agent {agent_id: $agent_id, graph_id: $graph_id})
        ON CREATE SET a.name = 'Test Agent', a.created_at = timestamp()
        """,
        {"agent_id": _AGENT_ID, "graph_id": _GRAPH_ID},
    )
    yield
    await _wipe()


@pytest.fixture(autouse=True)
def _override_auth(async_client):
    from app.api.dependencies import get_current_user, get_current_user_id
    from app.main import app

    app.dependency_overrides[get_current_user] = lambda: {
        "id": _USER_ID,
        "tenant_id": "org1",
    }
    app.dependency_overrides[get_current_user_id] = lambda: _USER_ID
    yield
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_current_user_id, None)


@pytest.fixture(autouse=True)
def _override_integration_service(neo4j_test_driver: AsyncDriver):
    from app.api.v1.endpoints.integration import _integration_service
    from app.main import app
    from app.services.integration_key_service import IntegrationKeyService

    app.dependency_overrides[_integration_service] = lambda: IntegrationKeyService(
        neo4j_test_driver
    )
    yield
    app.dependency_overrides.pop(_integration_service, None)


@pytest.fixture(autouse=True)
def _mock_verify_graph_access():
    with patch(
        "app.api.v1.endpoints.integration.verify_graph_access",
        new=AsyncMock(return_value=None),
    ):
        yield


@pytest.fixture(autouse=True)
def _override_neo4j_client_for_public(neo4j_test_driver: AsyncDriver):
    """Patch neo4j_client.async_driver so public endpoints use the test driver."""
    with patch("app.api.public.endpoints.public_agents.neo4j_client") as mock_client:
        mock_client.async_driver = neo4j_test_driver
        yield


# ── Publish / unpublish / rotate tests ────────────────────────────────────────


class TestPublishAgent:
    async def test_publish_returns_201_with_oak_key(self, async_client):
        resp = await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        assert resp.status_code == 201
        body = resp.json()
        assert body["slug"] == _SLUG
        assert body["integration_key"].startswith("oak-")
        assert body["key_last4"] == body["integration_key"][-4:]

    async def test_publish_stores_hash_not_plaintext(
        self, async_client, neo4j_test_driver
    ):
        resp = await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        assert resp.status_code == 201
        key = resp.json()["integration_key"]

        result = await neo4j_test_driver.execute_query(
            "MATCH (p:PublishedAgent {graph_id: $g}) RETURN p",
            {"g": _GRAPH_ID},
        )
        assert len(result.records) == 1
        node = result.records[0]["p"]
        assert node["key_hash"] != key
        assert not node["key_hash"].startswith("oak-")
        assert len(node["key_hash"]) == 64  # SHA-256 hex

    async def test_publish_slug_conflict_returns_409(self, async_client):
        # First publish succeeds
        resp1 = await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        assert resp1.status_code == 201

        # Second publish with same slug on a different (fake) agent
        other_prefix = (
            f"/api/v1/api/v1/graphs/{_GRAPH_ID}/agents/other-agent-99/publish"
        )
        resp2 = await async_client.post(other_prefix, json=_PUBLISH_BODY)
        assert resp2.status_code == 409

    async def test_get_published_does_not_return_full_key(self, async_client):
        await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        resp = await async_client.get(f"{_PREFIX}/publish")
        assert resp.status_code == 200
        body = resp.json()
        assert "integration_key" not in body
        assert "key_last4" in body
        assert len(body["key_last4"]) == 4

    async def test_unpublish_returns_204(self, async_client):
        await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        resp = await async_client.delete(f"{_PREFIX}/publish")
        assert resp.status_code == 204

    async def test_unpublish_twice_returns_404(self, async_client):
        await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        await async_client.delete(f"{_PREFIX}/publish")
        resp = await async_client.delete(f"{_PREFIX}/publish")
        assert resp.status_code == 404

    async def test_rotate_key_returns_new_oak_key(self, async_client):
        publish_resp = await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        old_key = publish_resp.json()["integration_key"]

        rotate_resp = await async_client.post(f"{_PREFIX}/rotate-key")
        assert rotate_resp.status_code == 200
        new_key = rotate_resp.json()["integration_key"]

        assert new_key.startswith("oak-")
        assert new_key != old_key
        assert rotate_resp.json()["key_last4"] == new_key[-4:]


# ── Public chat tests ─────────────────────────────────────────────────────────


class TestPublicChat:
    async def test_valid_key_returns_200_with_response(self, async_client):
        publish_resp = await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        key = publish_resp.json()["integration_key"]

        mock_executor = _make_executor_mock("Hi there!")
        with patch("app.api.public.endpoints.public_agents.AgentExecutor") as MockExec:
            MockExec.from_neo4j = AsyncMock(return_value=mock_executor)
            resp = await async_client.post(
                _PUBLIC_PREFIX + "/chat",
                json={"message": "hello"},
                headers={"Authorization": f"Bearer {key}"},
            )

        assert resp.status_code == 200
        assert resp.json()["response"] == "Hi there!"

    async def test_missing_key_returns_401(self, async_client):
        await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        resp = await async_client.post(
            _PUBLIC_PREFIX + "/chat",
            json={"message": "hello"},
        )
        assert resp.status_code == 401

    async def test_wrong_key_returns_401_same_message(self, async_client):
        await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        resp = await async_client.post(
            _PUBLIC_PREFIX + "/chat",
            json={"message": "hello"},
            headers={"Authorization": "Bearer oak-wrongkeyvalue"},
        )
        assert resp.status_code == 401
        # Same detail as "missing key" to avoid information leak
        assert "Invalid or revoked" in resp.json()["detail"]

    async def test_rotated_old_key_rejected(self, async_client):
        publish_resp = await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        old_key = publish_resp.json()["integration_key"]

        await async_client.post(f"{_PREFIX}/rotate-key")

        resp = await async_client.post(
            _PUBLIC_PREFIX + "/chat",
            json={"message": "hello"},
            headers={"Authorization": f"Bearer {old_key}"},
        )
        assert resp.status_code == 401

    async def test_rate_limit_returns_429(self, async_client):
        publish_resp = await async_client.post(
            f"{_PREFIX}/publish",
            json={**_PUBLISH_BODY, "rate_limit_rpm": 1},
        )
        key = publish_resp.json()["integration_key"]

        mock_executor = _make_executor_mock()
        with patch("app.api.public.endpoints.public_agents.AgentExecutor") as MockExec:
            MockExec.from_neo4j = AsyncMock(return_value=mock_executor)

            # First request: should succeed
            r1 = await async_client.post(
                _PUBLIC_PREFIX + "/chat",
                json={"message": "hello"},
                headers={"Authorization": f"Bearer {key}"},
            )
            assert r1.status_code == 200

            # Second request: should be rate-limited
            r2 = await async_client.post(
                _PUBLIC_PREFIX + "/chat",
                json={"message": "hello"},
                headers={"Authorization": f"Bearer {key}"},
            )
            assert r2.status_code == 429
            assert "Retry-After" in r2.headers

    async def test_cors_blocked_origin_returns_403(self, async_client):
        publish_resp = await async_client.post(
            f"{_PREFIX}/publish",
            json={**_PUBLISH_BODY, "cors_origins": ["https://allowed.example.com"]},
        )
        key = publish_resp.json()["integration_key"]

        mock_executor = _make_executor_mock()
        with patch("app.api.public.endpoints.public_agents.AgentExecutor") as MockExec:
            MockExec.from_neo4j = AsyncMock(return_value=mock_executor)
            resp = await async_client.post(
                _PUBLIC_PREFIX + "/chat",
                json={"message": "hello"},
                headers={
                    "Authorization": f"Bearer {key}",
                    "origin": "https://evil.example.com",
                },
            )

        assert resp.status_code == 403

    async def test_graph_id_sourced_from_published_not_caller(self, async_client):
        """Agent execution is scoped to the :PublishedAgent node's graph_id — caller cannot override."""
        publish_resp = await async_client.post(f"{_PREFIX}/publish", json=_PUBLISH_BODY)
        key = publish_resp.json()["integration_key"]

        captured_graph_ids: list[str] = []

        async def _spy_from_neo4j(driver, graph_id, agent_id):
            captured_graph_ids.append(graph_id)
            return _make_executor_mock()

        with patch("app.api.public.endpoints.public_agents.AgentExecutor") as MockExec:
            MockExec.from_neo4j = AsyncMock(side_effect=_spy_from_neo4j)
            await async_client.post(
                _PUBLIC_PREFIX + "/chat",
                json={"message": "hello"},
                headers={"Authorization": f"Bearer {key}"},
            )

        assert captured_graph_ids == [_GRAPH_ID]
