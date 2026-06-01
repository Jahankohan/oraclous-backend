"""
Integration tests — ORA-451: validate ORA-448 InstructionsService async_driver DI.

Scope (9 endpoints):
  PUT    /api/v1/graphs/{id}/instructions          — set instructions
  GET    /api/v1/graphs/{id}/instructions          — get instructions
  DELETE /api/v1/graphs/{id}/instructions          — delete instructions
  POST   /api/v1/graphs/{id}/ontology              — set ontology
  GET    /api/v1/graphs/{id}/ontology              — get ontology
  PATCH  /api/v1/graphs/{id}/ontology              — patch ontology
  DELETE /api/v1/graphs/{id}/ontology              — delete ontology
  POST   /api/v1/graphs/{id}/ontology/validate     — validate (dry-run scan)
  POST   /api/v1/graphs/{id}/ontology/retroactive-apply — apply to existing entities

Invariants verified:
  - InstructionsService is constructed per-request with the DI-injected driver
    (no module-level singleton exists after ORA-448)
  - All 9 endpoints return 503 when get_neo4j_async_driver raises (Neo4j unavailable)
  - Access-level requirements preserved: write/read/admin on instructions; ownership
    check on ontology
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi import status as fastapi_status

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

USER_A_ID = str(uuid.uuid4())
GRAPH_A_ID = str(uuid.uuid4())

_NOW = datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC)

_ENTITY_TYPES_JSON = [{"name": "Person", "description": "A human being"}]
_RELATIONSHIP_TYPES_JSON = [
    {"name": "WORKS_FOR", "source_type": "Person", "target_type": "Company"}
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _auth_headers() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _patch_auth(user_id: str = USER_A_ID):
    """Patch auth_service to return the given user id."""
    p = patch("app.api.dependencies.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(return_value={"id": user_id})
    return p


def _make_instructions_response():
    """Build a real GraphInstructionsResponse for mock return values."""
    from uuid import UUID as _UUID

    from app.schemas.graph_schemas import GraphInstructions, GraphInstructionsResponse

    return GraphInstructionsResponse(
        graph_id=_UUID(GRAPH_A_ID),
        instructions=GraphInstructions(domain="hr org chart"),
        version=1,
        updated_at=_NOW,
    )


def _make_ontology_response(version: int = 1):
    """Build a real OntologyResponse for mock return values."""
    from uuid import UUID as _UUID

    from app.schemas.graph_schemas import (
        EntityTypeDefinition,
        OntologyResponse,
        OntologyValidationMode,
        RelationshipTypeDefinition,
    )

    return OntologyResponse(
        graph_id=_UUID(GRAPH_A_ID),
        entity_types=[EntityTypeDefinition(name="Person", description="A human being")],
        relationship_types=[
            RelationshipTypeDefinition(
                name="WORKS_FOR", source_type="Person", target_type="Company"
            )
        ],
        ontology_mode=OntologyValidationMode.WARN,
        version=version,
        updated_at=_NOW,
    )


def _make_mock_ontology():
    """Build a minimal mock ontology suitable for validate/retroactive-apply scans."""
    from app.schemas.graph_schemas import OntologyValidationMode

    ontology = MagicMock()
    ontology.entity_types = [MagicMock()]
    ontology.entity_types[0].name = "Person"
    ontology.ontology_mode = OntologyValidationMode.WARN
    return ontology


# ---------------------------------------------------------------------------
# 1.  Instructions CRUD  (PUT / GET / DELETE)
# ---------------------------------------------------------------------------


class TestInstructionsCRUD:
    """Verify PUT/GET/DELETE /instructions work with DI-injected driver."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_set_instructions_returns_200(self, async_client):
        """PUT /instructions → 200; InstructionsService constructed with DI driver."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.set_instructions = AsyncMock(
            return_value=_make_instructions_response()
        )

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ) as MockISvc,
            ):
                mock_vga.return_value = GRAPH_A_ID

                response = await async_client.put(
                    f"/api/v1/graphs/{GRAPH_A_ID}/instructions",
                    json={"domain": "hr org chart"},
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 200
        # Service must be constructed with the injected driver — not a singleton
        MockISvc.assert_called_once_with(mock_driver)
        # Write-level access required
        assert mock_vga.call_args.args[1] == "write"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_instructions_returns_200_when_set(self, async_client):
        """GET /instructions → 200 when instructions exist."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.get_instructions = AsyncMock(
            return_value=_make_instructions_response()
        )

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ),
            ):
                mock_vga.return_value = GRAPH_A_ID

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/instructions",
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 200
        # Read-level access required
        assert mock_vga.call_args.args[1] == "read"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_instructions_returns_404_when_not_set(self, async_client):
        """GET /instructions → 404 when no instructions configured on this graph."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.get_instructions = AsyncMock(return_value=None)

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ),
            ):
                mock_vga.return_value = GRAPH_A_ID

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/instructions",
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 404

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_instructions_returns_204(self, async_client):
        """DELETE /instructions → 204; admin-level access required."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.delete_instructions = AsyncMock(return_value=None)

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ) as MockISvc,
            ):
                mock_vga.return_value = GRAPH_A_ID

                response = await async_client.delete(
                    f"/api/v1/graphs/{GRAPH_A_ID}/instructions",
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 204
        # Service is DI-injected and delete was called with the correct graph id
        MockISvc.assert_called_once_with(mock_driver)
        mock_svc.delete_instructions.assert_called_once_with(GRAPH_A_ID)
        # Admin-level access required for delete
        assert mock_vga.call_args.args[1] == "admin"


# ---------------------------------------------------------------------------
# 2.  Ontology CRUD  (POST / GET / PATCH / DELETE)
# ---------------------------------------------------------------------------


class TestOntologyCRUDDI:
    """POST/GET/PATCH/DELETE /ontology — DI pattern; old singleton is gone."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_set_ontology_returns_200(self, async_client):
        """POST /ontology → 200; InstructionsService constructed with DI driver."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.set_ontology = AsyncMock(return_value=_make_ontology_response())

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ) as MockISvc,
            ):
                mock_vga.return_value = GRAPH_A_ID

                response = await async_client.post(
                    f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
                    json={
                        "entity_types": _ENTITY_TYPES_JSON,
                        "relationship_types": _RELATIONSHIP_TYPES_JSON,
                        "ontology_mode": "warn",
                    },
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 200
        MockISvc.assert_called_once_with(mock_driver)

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_ontology_returns_200_when_set(self, async_client):
        """GET /ontology → 200 when an ontology has been configured."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.get_ontology = AsyncMock(return_value=_make_ontology_response())

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ),
            ):
                mock_vga.return_value = GRAPH_A_ID

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 200

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_ontology_returns_404_when_not_set(self, async_client):
        """GET /ontology → 404 when no ontology configured on this graph."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.get_ontology = AsyncMock(return_value=None)

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ),
            ):
                mock_vga.return_value = GRAPH_A_ID

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 404

    @pytest.mark.integration
    @pytest.mark.api
    async def test_patch_ontology_returns_200(self, async_client):
        """PATCH /ontology → 200 with merge update (add one entity type)."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.patch_ontology = AsyncMock(
            return_value=_make_ontology_response(version=2)
        )

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ),
            ):
                mock_vga.return_value = GRAPH_A_ID

                response = await async_client.patch(
                    f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
                    json={"add_entity_types": [{"name": "Company"}]},
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 200
        body = response.json()
        assert body["version"] == 2

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_ontology_returns_204(self, async_client):
        """DELETE /ontology → 204 (no body)."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.delete_ontology = AsyncMock(return_value=None)

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ),
            ):
                mock_vga.return_value = GRAPH_A_ID

                response = await async_client.delete(
                    f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 204
        assert response.content == b""


# ---------------------------------------------------------------------------
# 3.  Ontology validate and retroactive-apply
# ---------------------------------------------------------------------------


class TestOntologySpecialEndpoints:
    """POST /ontology/validate and POST /ontology/retroactive-apply."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_validate_ontology_returns_200_with_report(self, async_client):
        """POST /ontology/validate → 200 with violation_count and scanned_entities."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.get_ontology = AsyncMock(return_value=_make_mock_ontology())

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ),
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j_c,
            ):
                mock_vga.return_value = GRAPH_A_ID
                mock_neo4j_c.execute_query = AsyncMock(
                    side_effect=[
                        [{"total": 100, "violations": 5}],
                        [{"name": "Alien", "label": "Alien", "element_id": "e1"}],
                    ]
                )

                response = await async_client.post(
                    f"/api/v1/graphs/{GRAPH_A_ID}/ontology/validate",
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 200
        body = response.json()
        assert "violation_count" in body
        assert "scanned_entities" in body

    @pytest.mark.integration
    @pytest.mark.api
    async def test_retroactive_apply_dry_run_returns_200(self, async_client):
        """POST /ontology/retroactive-apply {dry_run:true} → 200 with counts."""
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        mock_driver = MagicMock()
        mock_svc = MagicMock()
        mock_svc.get_ontology = AsyncMock(return_value=_make_mock_ontology())

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = lambda: mock_driver
        try:
            with (
                patch(
                    "app.api.v1.endpoints.graphs.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.graphs.InstructionsService",
                    return_value=mock_svc,
                ),
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j_c,
            ):
                mock_vga.return_value = GRAPH_A_ID
                mock_neo4j_c.execute_query = AsyncMock(
                    side_effect=[
                        [{"cnt": 500}],
                        [{"violations": 10}],
                    ]
                )

                response = await async_client.post(
                    f"/api/v1/graphs/{GRAPH_A_ID}/ontology/retroactive-apply",
                    json={"dry_run": True},
                    headers=_auth_headers(),
                )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 200
        body = response.json()
        assert body["dry_run"] is True
        assert "violations_found" in body


# ---------------------------------------------------------------------------
# 4.  503 propagation — all 9 endpoints fail fast when Neo4j unavailable
# ---------------------------------------------------------------------------


def _raise_503():
    """Dependency override that simulates Neo4j being unreachable."""
    raise HTTPException(
        status_code=fastapi_status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Neo4j connection not available",
    )


class TestNeo4j503PropagationAllEndpoints:
    """
    get_neo4j_async_driver raises 503 → endpoint never runs → 503 returned to caller.

    This verifies that the DI check (ORA-428 pattern) gates all 9 endpoints,
    replacing the old inline `if not neo4j_client.async_driver` guards.
    """

    @pytest.mark.integration
    @pytest.mark.api
    async def test_set_instructions_503(self, async_client):
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = _raise_503
        try:
            response = await async_client.put(
                f"/api/v1/graphs/{GRAPH_A_ID}/instructions",
                json={"domain": "test"},
                headers=_auth_headers(),
            )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 503

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_instructions_503(self, async_client):
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = _raise_503
        try:
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_A_ID}/instructions",
                headers=_auth_headers(),
            )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 503

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_instructions_503(self, async_client):
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = _raise_503
        try:
            response = await async_client.delete(
                f"/api/v1/graphs/{GRAPH_A_ID}/instructions",
                headers=_auth_headers(),
            )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 503

    @pytest.mark.integration
    @pytest.mark.api
    async def test_set_ontology_503(self, async_client):
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = _raise_503
        try:
            response = await async_client.post(
                f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
                json={"entity_types": _ENTITY_TYPES_JSON},
                headers=_auth_headers(),
            )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 503

    @pytest.mark.integration
    @pytest.mark.api
    async def test_get_ontology_503(self, async_client):
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = _raise_503
        try:
            response = await async_client.get(
                f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
                headers=_auth_headers(),
            )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 503

    @pytest.mark.integration
    @pytest.mark.api
    async def test_patch_ontology_503(self, async_client):
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = _raise_503
        try:
            response = await async_client.patch(
                f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
                json={"add_entity_types": [{"name": "Drug"}]},
                headers=_auth_headers(),
            )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 503

    @pytest.mark.integration
    @pytest.mark.api
    async def test_delete_ontology_503(self, async_client):
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = _raise_503
        try:
            response = await async_client.delete(
                f"/api/v1/graphs/{GRAPH_A_ID}/ontology",
                headers=_auth_headers(),
            )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 503

    @pytest.mark.integration
    @pytest.mark.api
    async def test_validate_ontology_503(self, async_client):
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = _raise_503
        try:
            response = await async_client.post(
                f"/api/v1/graphs/{GRAPH_A_ID}/ontology/validate",
                headers=_auth_headers(),
            )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 503

    @pytest.mark.integration
    @pytest.mark.api
    async def test_retroactive_apply_503(self, async_client):
        from app.core.dependencies import get_neo4j_async_driver
        from app.main import app

        auth = _patch_auth()
        app.dependency_overrides[get_neo4j_async_driver] = _raise_503
        try:
            response = await async_client.post(
                f"/api/v1/graphs/{GRAPH_A_ID}/ontology/retroactive-apply",
                json={"dry_run": True},
                headers=_auth_headers(),
            )
        finally:
            auth.stop()
            app.dependency_overrides.pop(get_neo4j_async_driver, None)

        assert response.status_code == 503


# ---------------------------------------------------------------------------
# 5.  No regression: instructions_service singleton removed
# ---------------------------------------------------------------------------


class TestSingletonRemoved:
    """
    After ORA-448, the module-level instructions_service singleton no longer exists.
    This test confirms the instructions_service module does not export that name,
    ensuring the old patch target is gone and tests that relied on it must be updated.
    """

    @pytest.mark.integration
    def test_instructions_service_singleton_does_not_exist(self):
        """instructions_service global singleton must not exist after ORA-448."""
        import app.services.instructions_service as m

        assert not hasattr(m, "instructions_service"), (
            "Module-level instructions_service singleton still exists — "
            "ORA-448 DI migration incomplete"
        )

    @pytest.mark.integration
    def test_instructions_resolver_singleton_does_not_exist(self):
        """instructions_resolver global singleton must not exist after ORA-448."""
        import app.services.instructions_service as m

        assert not hasattr(m, "instructions_resolver"), (
            "Module-level instructions_resolver singleton still exists — "
            "ORA-448 DI migration incomplete"
        )

    @pytest.mark.integration
    def test_instructions_compiler_still_exists(self):
        """instructions_compiler is a stateless helper — it must still be accessible."""
        import app.services.instructions_service as m

        assert hasattr(m, "instructions_compiler"), (
            "instructions_compiler was accidentally removed; it is still needed"
        )
