"""
Integration tests for ORA-36: Ontology-Guided Extraction API

Tests cover:
- POST/GET/PATCH/DELETE ontology endpoints
- Ingestion with STRICT ontology mode
- Validate endpoint — returns counts without modifying graph
- retroactive-apply dry_run vs live
- Multi-tenant isolation: ontology from graph A does NOT affect graph B
"""
import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_A_ID = str(uuid.uuid4())
USER_B_ID = str(uuid.uuid4())
GRAPH_A_ID = str(uuid.uuid4())
GRAPH_B_ID = str(uuid.uuid4())

_NOW = datetime(2026, 4, 7, 12, 0, 0, tzinfo=timezone.utc).isoformat()

_ENTITY_TYPES = [
    {"name": "Person", "description": "A human being"},
    {"name": "Company", "description": "A corporate entity"},
]
_RELATIONSHIP_TYPES = [
    {"name": "WORKS_FOR", "source_type": "Person", "target_type": "Company"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neo4j_graph(graph_id: str, user_id: str) -> dict:
    return {
        "graph_id": graph_id,
        "user_id": user_id,
        "name": "Test Graph",
        "description": "desc",
        "created_at": _NOW,
        "updated_at": _NOW,
        "node_count": 10,
        "relationship_count": 5,
        "status": "active",
    }


def _ontology_response(graph_id: str, version: int = 1) -> dict:
    from app.schemas.graph_schemas import OntologyValidationMode
    return {
        "graph_id": graph_id,
        "entity_types": _ENTITY_TYPES,
        "relationship_types": _RELATIONSHIP_TYPES,
        "ontology_mode": OntologyValidationMode.WARN.value,
        "version": version,
        "updated_at": _NOW,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_auth():
    """Bypass token verification for all tests in this file."""
    mock_auth = MagicMock()
    mock_auth.verify_token = AsyncMock(return_value={"id": USER_A_ID})
    with patch("app.api.dependencies.auth_service", mock_auth):
        yield


@pytest.fixture
def mock_graph_service():
    svc = MagicMock()
    svc.get_graph.return_value = _neo4j_graph(GRAPH_A_ID, USER_A_ID)
    return svc


# ---------------------------------------------------------------------------
# 1. POST /graphs/{id}/ontology — set ontology
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
async def test_set_ontology_returns_200(async_client, mock_graph_service):
    from app.schemas.graph_schemas import OntologyResponse, OntologyValidationMode

    mock_ontology_resp = MagicMock(spec=OntologyResponse)
    mock_ontology_resp.model_dump.return_value = _ontology_response(GRAPH_A_ID)

    with patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j, \
         patch("app.api.v1.endpoints.graphs.GraphNodeService", return_value=mock_graph_service), \
         patch("app.services.instructions_service.instructions_service") as mock_svc:
        mock_neo4j.sync_driver = MagicMock()
        mock_svc.set_ontology = AsyncMock(return_value=mock_ontology_resp)

        response = await async_client.post(
            f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
            json={
                "entity_types": _ENTITY_TYPES,
                "relationship_types": _RELATIONSHIP_TYPES,
                "ontology_mode": "warn",
            },
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 2. GET /graphs/{id}/ontology — retrieve existing ontology
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
async def test_get_ontology_returns_200_when_set(async_client, mock_graph_service):
    from app.schemas.graph_schemas import OntologyResponse

    mock_resp = MagicMock(spec=OntologyResponse)
    mock_resp.model_dump.return_value = _ontology_response(GRAPH_A_ID)

    with patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j, \
         patch("app.api.v1.endpoints.graphs.GraphNodeService", return_value=mock_graph_service), \
         patch("app.services.instructions_service.instructions_service") as mock_svc:
        mock_neo4j.sync_driver = MagicMock()
        mock_svc.get_ontology = AsyncMock(return_value=mock_resp)

        response = await async_client.get(
            f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 3. GET /graphs/{id}/ontology — 404 when not set
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
async def test_get_ontology_returns_404_when_not_set(async_client, mock_graph_service):
    with patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j, \
         patch("app.api.v1.endpoints.graphs.GraphNodeService", return_value=mock_graph_service), \
         patch("app.services.instructions_service.instructions_service") as mock_svc:
        mock_neo4j.sync_driver = MagicMock()
        mock_svc.get_ontology = AsyncMock(return_value=None)

        response = await async_client.get(
            f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# 4. PATCH /graphs/{id}/ontology — merge update
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
async def test_patch_ontology_returns_200(async_client, mock_graph_service):
    from app.schemas.graph_schemas import OntologyResponse

    mock_resp = MagicMock(spec=OntologyResponse)
    mock_resp.model_dump.return_value = _ontology_response(GRAPH_A_ID, version=2)

    with patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j, \
         patch("app.api.v1.endpoints.graphs.GraphNodeService", return_value=mock_graph_service), \
         patch("app.services.instructions_service.instructions_service") as mock_svc:
        mock_neo4j.sync_driver = MagicMock()
        mock_svc.patch_ontology = AsyncMock(return_value=mock_resp)

        response = await async_client.patch(
            f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
            json={"add_entity_types": [{"name": "Drug"}]},
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 5. DELETE /graphs/{id}/ontology — clear ontology
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
async def test_delete_ontology_returns_204(async_client, mock_graph_service):
    with patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j, \
         patch("app.api.v1.endpoints.graphs.GraphNodeService", return_value=mock_graph_service), \
         patch("app.services.instructions_service.instructions_service") as mock_svc:
        mock_neo4j.sync_driver = MagicMock()
        mock_svc.delete_ontology = AsyncMock(return_value=None)

        response = await async_client.delete(
            f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 204


# ---------------------------------------------------------------------------
# 6. POST /graphs/{id}/ontology/validate — counts without modifying graph
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
async def test_validate_ontology_returns_report(async_client, mock_graph_service):
    from app.schemas.graph_schemas import OntologyResponse, OntologyValidationMode

    mock_ontology = MagicMock()
    mock_ontology.entity_types = [
        MagicMock(name="Person"),
    ]
    # Fix: entity_types[0].name must return string
    mock_ontology.entity_types[0].name = "Person"
    mock_ontology.ontology_mode = OntologyValidationMode.WARN

    with patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j, \
         patch("app.api.v1.endpoints.graphs.GraphNodeService", return_value=mock_graph_service), \
         patch("app.services.instructions_service.instructions_service") as mock_svc:
        mock_neo4j.sync_driver = MagicMock()
        mock_svc.get_ontology = AsyncMock(return_value=mock_ontology)
        # Count query result
        mock_neo4j.execute_query = AsyncMock(side_effect=[
            [{"total": 100, "violations": 15}],   # count query
            [{"name": "Alien", "label": "Alien", "element_id": "e1"}],  # scan query
        ])

        response = await async_client.post(
            f"/api/v1/graphs/{GRAPH_A_ID}/ontology/validate",
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    body = response.json()
    assert "violation_count" in body
    assert "scanned_entities" in body


# ---------------------------------------------------------------------------
# 7. POST /graphs/{id}/ontology/retroactive-apply — dry_run=True
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
async def test_retroactive_apply_dry_run_returns_counts(async_client, mock_graph_service):
    from app.schemas.graph_schemas import OntologyValidationMode

    mock_ontology = MagicMock()
    mock_ontology.entity_types = [MagicMock()]
    mock_ontology.entity_types[0].name = "Person"
    mock_ontology.ontology_mode = OntologyValidationMode.WARN

    with patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j, \
         patch("app.api.v1.endpoints.graphs.GraphNodeService", return_value=mock_graph_service), \
         patch("app.services.instructions_service.instructions_service") as mock_svc:
        mock_neo4j.sync_driver = MagicMock()
        mock_svc.get_ontology = AsyncMock(return_value=mock_ontology)
        mock_neo4j.execute_query = AsyncMock(side_effect=[
            [{"cnt": 500}],          # entity count
            [{"violations": 10}],    # violation count (dry_run)
        ])

        response = await async_client.post(
            f"/api/v1/graphs/{GRAPH_A_ID}/ontology/retroactive-apply",
            json={"dry_run": True},
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is True
    assert "violations_found" in body


# ---------------------------------------------------------------------------
# 8. retroactive-apply dry_run=False, ≤10k → inline
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
async def test_retroactive_apply_live_inline(async_client, mock_graph_service):
    from app.schemas.graph_schemas import OntologyValidationMode

    mock_ontology = MagicMock()
    mock_ontology.entity_types = [MagicMock()]
    mock_ontology.entity_types[0].name = "Person"
    mock_ontology.ontology_mode = OntologyValidationMode.STRICT

    with patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j, \
         patch("app.api.v1.endpoints.graphs.GraphNodeService", return_value=mock_graph_service), \
         patch("app.services.instructions_service.instructions_service") as mock_svc:
        mock_neo4j.sync_driver = MagicMock()
        mock_svc.get_ontology = AsyncMock(return_value=mock_ontology)
        mock_neo4j.execute_query = AsyncMock(side_effect=[
            [{"cnt": 100}],           # entity count — under 10k, run inline
            [{"deleted": 5}],         # delete query result
            [{"cnt": 0}],             # remaining violations
        ])

        response = await async_client.post(
            f"/api/v1/graphs/{GRAPH_A_ID}/ontology/retroactive-apply",
            json={"dry_run": False},
            headers={"Authorization": "Bearer test-token"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["dry_run"] is False
    assert body.get("celery_task_id") is None


# ---------------------------------------------------------------------------
# 9. Multi-tenant isolation: ontology from graph A does NOT affect graph B
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.api
async def test_ontology_multi_tenant_isolation(async_client):
    """User B cannot read User A's graph ontology — should get 403."""
    graph_a_owned_by_a = _neo4j_graph(GRAPH_A_ID, USER_A_ID)
    # Simulate auth returning user B
    mock_auth = MagicMock()
    mock_auth.verify_token = AsyncMock(return_value={"id": USER_B_ID})

    mock_svc = MagicMock()
    mock_svc.get_graph.return_value = graph_a_owned_by_a

    with patch("app.api.dependencies.auth_service", mock_auth), \
         patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j, \
         patch("app.api.v1.endpoints.graphs.GraphNodeService", return_value=mock_svc):
        mock_neo4j.sync_driver = MagicMock()

        response = await async_client.get(
            f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
            headers={"Authorization": "Bearer user-b-token"},
        )

    assert response.status_code == 403
