"""
Integration tests for the communities listing endpoint (TASK-050).

Endpoint: GET /api/v1/graphs/{graph_id}/communities
Schema: list[Community] = [{community_id, level, label, size, summary?}]
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

USER_ID = str(uuid.uuid4())
GRAPH_ID = str(uuid.uuid4())
FAKE_USER = {"id": USER_ID, "email": "test@example.com"}


def _patch_auth():
    """Patch auth_service.verify_token to return a fake user."""
    p = patch("app.api.dependencies.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(return_value=FAKE_USER)
    return p


def _auth_headers() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _community_record(
    community_id: str = "c-1",
    level: int = 0,
    entity_count: int = 5,
    summary: str | None = "Tech companies in the Bay Area",
) -> dict:
    return {
        "community_id": community_id,
        "level": level,
        "entity_count": entity_count,
        "weight": 0.42,
        "parent_id": None,
        "status": "active",
        "summary": summary,
    }


class TestListCommunitiesEndpoint:
    """GET /api/v1/graphs/{graph_id}/communities — flat Community[] response."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_returns_flat_list_with_label_and_size(self, async_client):
        """200 with a list whose items have community_id, level, label, size, summary."""
        analytics_result = {
            "communities": [
                _community_record(
                    "c-aaa",
                    level=0,
                    entity_count=12,
                    summary="Tech companies in the Bay Area. Includes startups.",
                ),
                _community_record("c-bbb", level=1, entity_count=4, summary=None),
            ],
            "total": 2,
            "detection_status": "active",
            "last_detected_at": "2026-04-28T00:00:00Z",
        }

        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.communities.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.communities.GraphAnalyticsService"
                ) as MockSvc,
            ):
                mock_vga.return_value = GRAPH_ID
                MockSvc.return_value.get_communities_list = AsyncMock(
                    return_value=analytics_result
                )

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/communities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

        first, second = data
        # Required fields per the frontend Community interface
        for entry in (first, second):
            assert set(entry.keys()) >= {"community_id", "level", "label", "size"}
            assert "summary" in entry

        # Field mapping
        assert first["community_id"] == "c-aaa"
        assert first["level"] == 0
        assert first["size"] == 12
        # label is derived from the first sentence of summary
        assert first["label"] == "Tech companies in the Bay Area"
        assert first["summary"].startswith("Tech companies")

        # No summary → synthetic label
        assert second["community_id"] == "c-bbb"
        assert second["size"] == 4
        assert second["label"].startswith("Community ")
        assert second["summary"] is None

    @pytest.mark.integration
    @pytest.mark.api
    async def test_returns_empty_list_when_no_communities(self, async_client):
        """200 with [] when no community data exists for the graph."""
        analytics_result = {
            "communities": [],
            "total": 0,
            "detection_status": "not_detected",
            "last_detected_at": None,
        }

        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.communities.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.communities.GraphAnalyticsService"
                ) as MockSvc,
            ):
                mock_vga.return_value = GRAPH_ID
                MockSvc.return_value.get_communities_list = AsyncMock(
                    return_value=analytics_result
                )

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/communities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        assert response.json() == []

    @pytest.mark.integration
    @pytest.mark.api
    async def test_returns_403_when_no_read_access(self, async_client):
        """403 when caller lacks read access to the graph."""
        from fastapi import HTTPException
        from fastapi import status as fastapi_status

        auth_patch = _patch_auth()
        try:
            with patch(
                "app.api.v1.endpoints.communities.verify_graph_access",
                new_callable=AsyncMock,
            ) as mock_vga:
                mock_vga.side_effect = HTTPException(
                    status_code=fastapi_status.HTTP_403_FORBIDDEN,
                    detail="Access denied",
                )

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/communities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.api
    async def test_requires_auth(self, async_client):
        """No token → 401/403."""
        response = await async_client.get(f"/api/v1/graphs/{GRAPH_ID}/communities")
        assert response.status_code in (401, 403)

    @pytest.mark.integration
    @pytest.mark.api
    async def test_passes_level_query_param(self, async_client):
        """Optional ?level=N is passed through to the analytics service."""
        analytics_result = {
            "communities": [],
            "total": 0,
            "detection_status": "active",
            "last_detected_at": None,
        }
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.communities.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.communities.GraphAnalyticsService"
                ) as MockSvc,
            ):
                mock_vga.return_value = GRAPH_ID
                svc = MockSvc.return_value
                svc.get_communities_list = AsyncMock(return_value=analytics_result)

                await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/communities?level=2",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        call_kwargs = svc.get_communities_list.call_args.kwargs
        assert call_kwargs.get("level") == 2

    @pytest.mark.integration
    @pytest.mark.api
    async def test_uses_read_level_rebac(self, async_client):
        """ReBAC must be called with required_level='read'."""
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.communities.verify_graph_access",
                    new_callable=AsyncMock,
                ) as mock_vga,
                patch(
                    "app.api.v1.endpoints.communities.GraphAnalyticsService"
                ) as MockSvc,
            ):
                mock_vga.return_value = GRAPH_ID
                MockSvc.return_value.get_communities_list = AsyncMock(
                    return_value={
                        "communities": [],
                        "total": 0,
                        "detection_status": "active",
                        "last_detected_at": None,
                    }
                )
                await async_client.get(
                    f"/api/v1/graphs/{GRAPH_ID}/communities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        called = mock_vga.call_args
        assert called.args[1] == "read" or called.kwargs.get("required_level") == "read"
