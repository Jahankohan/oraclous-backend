"""Integration tests for GET /api/v1/graphs/{graph_id}/entities (ORA-372)."""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

USER_ID = str(uuid.uuid4())
GRAPH_ID = str(uuid.uuid4())
FAKE_USER = {"id": USER_ID, "email": "test@example.com"}


def _patch_auth():
    p = patch("app.api.dependencies.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(return_value=FAKE_USER)
    return p


def _auth_headers() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _entity_item(
    id: str = "e-1",
    name: str = "Acme Corp",
    type: str = "Organization",
    confidence: float = 0.9,
    community_id: str | None = "c-1",
    degree: int = 5,
) -> dict:
    return {
        "id": id,
        "name": name,
        "type": type,
        "confidence": confidence,
        "community_id": community_id,
        "degree": degree,
    }


def _list_result(
    items: list[dict], total: int = -1, page: int = 1, page_size: int = 50
) -> dict:
    return {
        "items": items,
        "total": total if total >= 0 else len(items),
        "page": page,
        "page_size": page_size,
    }


class TestListEntitiesEndpoint:
    """GET /api/v1/graphs/{graph_id}/entities"""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_happy_path_returns_entity_list_response(self, async_client):
        """200 with items matching EntityListResponse schema."""
        items = [
            _entity_item("e-1", "Acme Corp", "Organization"),
            _entity_item("e-2", "Alice", "Person", community_id=None, degree=2),
        ]
        service_result = _list_result(items, total=42)

        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                mock_vga.return_value = GRAPH_ID
                MockSvc.return_value.list_entities = AsyncMock(
                    return_value=service_result
                )

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/entities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 42
        assert data["page"] == 1
        assert data["page_size"] == 50
        assert len(data["items"]) == 2

        first = data["items"][0]
        assert first["id"] == "e-1"
        assert first["name"] == "Acme Corp"
        assert first["type"] == "Organization"
        assert first["confidence"] == 0.9
        assert first["community_id"] == "c-1"
        assert first["degree"] == 5

        # community_id may be None
        second = data["items"][1]
        assert second["community_id"] is None

    @pytest.mark.integration
    @pytest.mark.api
    async def test_unauthenticated_returns_401(self, async_client):
        """No token → 401 Unauthorized."""
        response = await async_client.get(f"/api/v1/graphs/{GRAPH_ID}/entities")
        assert response.status_code == 401

    @pytest.mark.integration
    @pytest.mark.api
    async def test_forbidden_graph_returns_403(self, async_client):
        """verify_graph_access raises HTTPException 403 → propagated."""
        from fastapi import HTTPException

        auth_patch = _patch_auth()
        try:
            with patch(
                "app.api.v1.endpoints.entities.verify_graph_access",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=403, detail="Forbidden"),
            ):
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/entities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.api
    async def test_q_parameter_forwarded_to_service(self, async_client):
        """?q=alice is passed as q= to list_entities()."""
        service_result = _list_result([_entity_item("e-3", "Alice", "Person")])

        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                mock_vga.return_value = GRAPH_ID
                mock_list = AsyncMock(return_value=service_result)
                MockSvc.return_value.list_entities = mock_list

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/entities?q=alice",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        _call_kwargs = mock_list.call_args.kwargs
        assert _call_kwargs["q"] == "alice"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_multi_value_type_filter_forwarded(self, async_client):
        """?type=concept&type=entity → types=["concept", "entity"]."""
        service_result = _list_result([])

        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                mock_vga.return_value = GRAPH_ID
                mock_list = AsyncMock(return_value=service_result)
                MockSvc.return_value.list_entities = mock_list

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/entities?type=concept&type=entity",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        _call_kwargs = mock_list.call_args.kwargs
        assert sorted(_call_kwargs["types"]) == ["concept", "entity"]

    @pytest.mark.integration
    @pytest.mark.api
    async def test_invalid_sort_returns_422(self, async_client):
        """An unrecognised sort value → 422 Unprocessable Entity."""
        auth_patch = _patch_auth()
        try:
            with patch(
                "app.api.v1.endpoints.entities.verify_graph_access",
                new_callable=AsyncMock,
                return_value=GRAPH_ID,
            ):
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/entities?sort=bad_sort",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 422

    @pytest.mark.integration
    @pytest.mark.api
    async def test_page_size_clamped_at_200(self, async_client):
        """page_size > 200 → 422 (FastAPI ge/le validation)."""
        auth_patch = _patch_auth()
        try:
            with patch(
                "app.api.v1.endpoints.entities.verify_graph_access",
                new_callable=AsyncMock,
                return_value=GRAPH_ID,
            ):
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/entities?page_size=201",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 422

    @pytest.mark.integration
    @pytest.mark.api
    async def test_empty_result_returns_zero_total(self, async_client):
        """Empty graph returns items=[], total=0."""
        service_result = _list_result([], total=0)

        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                mock_vga.return_value = GRAPH_ID
                MockSvc.return_value.list_entities = AsyncMock(
                    return_value=service_result
                )

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/entities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
