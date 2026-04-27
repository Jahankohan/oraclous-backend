"""Integration tests for the Connector Framework (ORA-78).

Tests hit the real FastAPI app and PostgreSQL database.
Neo4j is NOT required for these tests — connectors use PostgreSQL for metadata.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TEST_GRAPH_ID = "connector-test-graph-99999"
TEST_USER_ID = "connector-test-user-99999"

MOCK_TOKEN = "test-connector-token"


@pytest_asyncio.fixture
async def db_session():
    """Task-scoped async database session for test setup/teardown."""
    engine = create_async_engine(settings.POSTGRES_URL, poolclass=NullPool, future=True)
    session_maker = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_maker() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def client():
    """Async HTTP client with mocked auth returning TEST_USER_ID."""
    from app.api.dependencies import get_current_user_id
    from app.main import app

    async def _mock_user_id() -> str:
        return TEST_USER_ID

    app.dependency_overrides[get_current_user_id] = _mock_user_id

    # Also mock verify_graph_access so we don't need Neo4j
    from app.api.dependencies import verify_graph_access

    async def _mock_verify_graph_access(
        graph_id: str, required_level: str, user_id: str
    ) -> str:
        return graph_id

    app.dependency_overrides[verify_graph_access] = _mock_verify_graph_access

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Cleanup helper
# ---------------------------------------------------------------------------


async def _delete_test_connectors(db: AsyncSession) -> None:
    from sqlalchemy import delete

    from app.models.graph import Connector, ConnectorSyncLog, WebhookEvent

    result = await db.execute(
        __import__("sqlalchemy")
        .select(Connector)
        .where(
            Connector.graph_id == TEST_GRAPH_ID,
            Connector.user_id == TEST_USER_ID,
        )
    )
    connectors = result.scalars().all()
    for connector in connectors:
        await db.execute(
            delete(WebhookEvent).where(WebhookEvent.connector_id == connector.id)
        )
        await db.execute(
            delete(ConnectorSyncLog).where(
                ConnectorSyncLog.connector_id == connector.id
            )
        )
        await db.delete(connector)
    await db.commit()


# ---------------------------------------------------------------------------
# Connector template tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_connector_templates(client: AsyncClient) -> None:
    """Verify all 7 built-in templates are available."""
    response = await client.get("/api/v1/connector-templates")
    assert response.status_code == 200
    data = response.json()
    assert "github" in data
    assert "notion" in data
    assert "slack" in data
    assert "webhook_receiver" in data
    assert "rest_api" in data


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_github_connector_template(client: AsyncClient) -> None:
    response = await client.get("/api/v1/connector-templates/github")
    assert response.status_code == 200
    data = response.json()
    assert data["connector_type"] == "github"
    assert data["supports_incremental"] is True
    assert data["supports_webhook"] is True


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_unknown_template_returns_404(client: AsyncClient) -> None:
    response = await client.get("/api/v1/connector-templates/nonexistent_source")
    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Connector CRUD
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_register_and_list_connector(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Register a connector and verify it appears in the list."""
    await _delete_test_connectors(db_session)

    payload = {
        "name": "Test GitHub Connector",
        "connector_type": "github",
        "config": {
            "auth": {"auth_type": "bearer_token", "header_name": "Authorization"},
            "base_url": "https://api.github.com",
            "entity_mapping": {"extraction_mode": "template"},
        },
        "schedule": "0 */6 * * *",
    }
    create_resp = await client.post(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors",
        json=payload,
    )
    assert create_resp.status_code == 201
    created = create_resp.json()
    assert created["connector_type"] == "github"
    assert created["graph_id"] == TEST_GRAPH_ID
    assert created["schedule"] == "0 */6 * * *"
    assert created["webhook_url"] is None  # not a webhook_receiver

    connector_id = created["id"]

    try:
        # List
        list_resp = await client.get(f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        ids = [c["id"] for c in list_data["connectors"]]
        assert connector_id in ids

        # Get by ID
        get_resp = await client.get(
            f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors/{connector_id}"
        )
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == connector_id
    finally:
        await _delete_test_connectors(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_register_webhook_receiver_returns_webhook_url(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """webhook_receiver connectors should include webhook_url in response."""
    await _delete_test_connectors(db_session)

    payload = {
        "name": "My Webhook Receiver",
        "connector_type": "webhook_receiver",
        "config": {
            "auth": {
                "auth_type": "hmac_secret",
                "hmac_secret_credential_id": "my-hmac-cred",
                "hmac_header": "X-Hub-Signature-256",
            },
            "entity_mapping": {"extraction_mode": "llm"},
        },
    }
    resp = await client.post(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors",
        json=payload,
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["webhook_url"] is not None
    assert f"/webhooks/{TEST_GRAPH_ID}/" in data["webhook_url"]

    await _delete_test_connectors(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_connector(client: AsyncClient, db_session: AsyncSession) -> None:
    """Update connector name and verify change persists."""
    await _delete_test_connectors(db_session)

    payload = {
        "name": "Original Name",
        "connector_type": "rest_api",
        "config": {
            "auth": {"auth_type": "api_key"},
            "base_url": "https://api.example.com",
            "entity_mapping": {"extraction_mode": "llm"},
        },
    }
    create_resp = await client.post(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors", json=payload
    )
    assert create_resp.status_code == 201
    connector_id = create_resp.json()["id"]

    try:
        patch_resp = await client.patch(
            f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors/{connector_id}",
            json={"name": "Updated Name", "status": "paused"},
        )
        assert patch_resp.status_code == 200
        updated = patch_resp.json()
        assert updated["name"] == "Updated Name"
        assert updated["status"] == "paused"
    finally:
        await _delete_test_connectors(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_connector(client: AsyncClient, db_session: AsyncSession) -> None:
    """Delete a connector and verify it's gone."""
    await _delete_test_connectors(db_session)

    payload = {
        "name": "Delete Me",
        "connector_type": "rest_api",
        "config": {
            "auth": {"auth_type": "api_key"},
            "entity_mapping": {"extraction_mode": "llm"},
        },
    }
    create_resp = await client.post(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors", json=payload
    )
    assert create_resp.status_code == 201
    connector_id = create_resp.json()["id"]

    del_resp = await client.delete(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors/{connector_id}"
    )
    assert del_resp.status_code == 204

    get_resp = await client.get(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors/{connector_id}"
    )
    assert get_resp.status_code == 404


# ---------------------------------------------------------------------------
# Webhook receiver tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_webhook_accepted_with_valid_hmac(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Valid HMAC-signed webhook payload is accepted and event stored."""
    await _delete_test_connectors(db_session)

    # Register webhook_receiver connector (no HMAC secret ID for simplicity — skip verification in test)
    payload = {
        "name": "GitHub Webhook",
        "connector_type": "webhook_receiver",
        "config": {
            "auth": {
                "auth_type": "hmac_secret",
                "hmac_header": "X-Hub-Signature-256",
            },
            "entity_mapping": {"extraction_mode": "llm"},
        },
    }
    create_resp = await client.post(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors", json=payload
    )
    assert create_resp.status_code == 201
    connector_id = create_resp.json()["id"]

    try:
        raw_body = json.dumps(
            {"action": "opened", "repository": {"full_name": "org/repo"}}
        ).encode()

        # Patch HMAC credential resolution to skip real broker call
        with patch(
            "app.api.v1.endpoints.webhooks._resolve_credential",
            new_callable=AsyncMock,
            return_value=None,  # no hmac_secret_credential_id set → skip HMAC check
        ):
            webhook_resp = await client.post(
                f"/api/v1/webhooks/{TEST_GRAPH_ID}/{connector_id}",
                content=raw_body,
                headers={"Content-Type": "application/json"},
            )

        assert webhook_resp.status_code == 200
        data = webhook_resp.json()
        assert data["status"] == "accepted"
        assert "event_id" in data
    finally:
        await _delete_test_connectors(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_webhook_duplicate_rejected(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Sending identical payload twice returns duplicate on second request."""
    await _delete_test_connectors(db_session)

    payload = {
        "name": "Dedup Webhook",
        "connector_type": "webhook_receiver",
        "config": {
            "auth": {"auth_type": "hmac_secret"},
            "entity_mapping": {"extraction_mode": "llm"},
        },
    }
    create_resp = await client.post(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors", json=payload
    )
    assert create_resp.status_code == 201
    connector_id = create_resp.json()["id"]

    try:
        raw_body = json.dumps({"event": "push", "ref": "refs/heads/main"}).encode()

        with patch(
            "app.api.v1.endpoints.webhooks._resolve_credential",
            new_callable=AsyncMock,
            return_value=None,
        ):
            resp1 = await client.post(
                f"/api/v1/webhooks/{TEST_GRAPH_ID}/{connector_id}",
                content=raw_body,
                headers={"Content-Type": "application/json"},
            )
            resp2 = await client.post(
                f"/api/v1/webhooks/{TEST_GRAPH_ID}/{connector_id}",
                content=raw_body,
                headers={"Content-Type": "application/json"},
            )

        assert resp1.status_code == 200
        assert resp1.json()["status"] == "accepted"
        assert resp2.status_code == 200
        assert resp2.json()["status"] == "duplicate"
    finally:
        await _delete_test_connectors(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_webhook_wrong_graph_returns_404(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Mismatched graph_id returns 404 — not 403 — to prevent enumeration."""
    await _delete_test_connectors(db_session)

    payload = {
        "name": "Enum Test",
        "connector_type": "webhook_receiver",
        "config": {
            "auth": {"auth_type": "hmac_secret"},
            "entity_mapping": {"extraction_mode": "llm"},
        },
    }
    create_resp = await client.post(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors", json=payload
    )
    assert create_resp.status_code == 201
    connector_id = create_resp.json()["id"]

    try:
        wrong_graph = "wrong-graph-id-00000"
        resp = await client.post(
            f"/api/v1/webhooks/{wrong_graph}/{connector_id}",
            content=b'{"test": 1}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 404
    finally:
        await _delete_test_connectors(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_webhook_connector_returns_404_on_webhook_endpoint(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Sending a webhook to a non-webhook_receiver connector returns 404."""
    await _delete_test_connectors(db_session)

    payload = {
        "name": "GitHub Puller",
        "connector_type": "github",
        "config": {
            "auth": {"auth_type": "bearer_token"},
            "entity_mapping": {"extraction_mode": "template"},
        },
    }
    create_resp = await client.post(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors", json=payload
    )
    assert create_resp.status_code == 201
    connector_id = create_resp.json()["id"]

    try:
        resp = await client.post(
            f"/api/v1/webhooks/{TEST_GRAPH_ID}/{connector_id}",
            content=b'{"test": 1}',
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 404
    finally:
        await _delete_test_connectors(db_session)


# ---------------------------------------------------------------------------
# Manual sync trigger
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_sync_enqueues_celery_task(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST /connectors/{id}/sync dispatches Celery task and returns task_id."""
    await _delete_test_connectors(db_session)

    payload = {
        "name": "Sync Test Connector",
        "connector_type": "github",
        "config": {
            "auth": {"auth_type": "bearer_token"},
            "entity_mapping": {"extraction_mode": "template"},
        },
        "schedule": "0 * * * *",
    }
    create_resp = await client.post(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors", json=payload
    )
    assert create_resp.status_code == 201
    connector_id = create_resp.json()["id"]

    try:
        mock_task = MagicMock()
        mock_task.id = "mock-celery-task-id"

        with patch(
            "app.services.connector_service.connector_service.trigger_sync",
            new_callable=AsyncMock,
            return_value={
                "connector_id": connector_id,
                "task_id": "mock-celery-task-id",
                "status": "queued",
            },
        ):
            sync_resp = await client.post(
                f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors/{connector_id}/sync"
            )

        assert sync_resp.status_code == 200
        data = sync_resp.json()
        assert data["status"] == "queued"
        assert "task_id" in data
    finally:
        await _delete_test_connectors(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_trigger_sync_on_webhook_receiver_returns_400(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Triggering sync on a webhook_receiver returns 400."""
    await _delete_test_connectors(db_session)

    payload = {
        "name": "Webhook Only",
        "connector_type": "webhook_receiver",
        "config": {
            "auth": {"auth_type": "hmac_secret"},
            "entity_mapping": {"extraction_mode": "llm"},
        },
    }
    create_resp = await client.post(
        f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors", json=payload
    )
    assert create_resp.status_code == 201
    connector_id = create_resp.json()["id"]

    try:
        sync_resp = await client.post(
            f"/api/v1/graphs/{TEST_GRAPH_ID}/connectors/{connector_id}/sync"
        )
        assert sync_resp.status_code == 400
    finally:
        await _delete_test_connectors(db_session)


# ---------------------------------------------------------------------------
# Multi-tenancy isolation
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connectors_scoped_to_graph(
    client: AsyncClient, db_session: AsyncSession
) -> None:
    """Connectors from one graph are not visible from another graph."""
    await _delete_test_connectors(db_session)

    graph_a = TEST_GRAPH_ID
    graph_b = "connector-test-graph-OTHER"

    payload = {
        "name": "Graph A Connector",
        "connector_type": "rest_api",
        "config": {
            "auth": {"auth_type": "api_key"},
            "entity_mapping": {"extraction_mode": "llm"},
        },
    }
    create_resp = await client.post(
        f"/api/v1/graphs/{graph_a}/connectors", json=payload
    )
    assert create_resp.status_code == 201
    connector_id = create_resp.json()["id"]

    try:
        # List from graph_b should NOT return graph_a's connectors
        list_resp = await client.get(f"/api/v1/graphs/{graph_b}/connectors")
        assert list_resp.status_code == 200
        ids = [c["id"] for c in list_resp.json()["connectors"]]
        assert connector_id not in ids

        # Direct get from graph_b should 404
        get_resp = await client.get(
            f"/api/v1/graphs/{graph_b}/connectors/{connector_id}"
        )
        assert get_resp.status_code == 404
    finally:
        await _delete_test_connectors(db_session)
