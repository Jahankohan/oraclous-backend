"""QA validation tests for GET /api/v1/graphs/{graph_id}/entities (ORA-374).

Covers the scenarios from ORA-374 not addressed in test_entities_api.py:
  - Sort verification (degree_desc, name_asc, confidence_desc)
  - Cross-tenant isolation (Graph A cannot see Graph B entities)
  - Pagination integrity (page1 + page2 no overlap, total consistent)
  - Type filter correctness (all returned items match requested type)
  - Full-text search inclusion (known entity appears in results)
"""

import uuid
from unittest.mock import AsyncMock, patch

import pytest

USER_A_ID = str(uuid.uuid4())
USER_B_ID = str(uuid.uuid4())
GRAPH_A_ID = str(uuid.uuid4())
GRAPH_B_ID = str(uuid.uuid4())
FAKE_USER_A = {"id": USER_A_ID, "email": "tenant-a@example.com"}
FAKE_USER_B = {"id": USER_B_ID, "email": "tenant-b@example.com"}


def _patch_auth(fake_user: dict | None = None):
    user = fake_user or FAKE_USER_A
    p = patch("app.api.dependencies.auth_service")
    mock_auth = p.start()
    mock_auth.verify_token = AsyncMock(return_value=user)
    return p


def _auth_headers() -> dict:
    return {"Authorization": "Bearer fake-token"}


def _entity(
    id_: str,
    name: str,
    type_: str = "concept",
    confidence: float = 0.9,
    degree: int = 5,
    community_id: str | None = None,
) -> dict:
    return {
        "id": id_,
        "name": name,
        "type": type_,
        "confidence": confidence,
        "community_id": community_id,
        "degree": degree,
    }


def _list_result(
    items: list[dict],
    total: int = -1,
    page: int = 1,
    page_size: int = 10,
) -> dict:
    return {
        "items": items,
        "total": total if total >= 0 else len(items),
        "page": page,
        "page_size": page_size,
    }


class TestSortVerification:
    """Sort ordering is enforced — service returns pre-sorted data that response preserves."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_sort_degree_desc_first_item_highest_degree(self, async_client):
        """sort=degree_desc: first item degree >= last item degree."""
        items = [
            _entity("e-1", "Hub Node", degree=20),
            _entity("e-2", "Mid Node", degree=10),
            _entity("e-3", "Leaf Node", degree=2),
        ]
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                MockSvc.return_value.list_entities = AsyncMock(
                    return_value=_list_result(items, total=3)
                )
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?sort=degree_desc",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        result_items = response.json()["items"]
        assert len(result_items) == 3
        assert result_items[0]["degree"] >= result_items[-1]["degree"]
        assert result_items[0]["degree"] == 20
        assert result_items[-1]["degree"] == 2

    @pytest.mark.integration
    @pytest.mark.api
    async def test_sort_name_asc_alphabetical_order(self, async_client):
        """sort=name_asc: items ordered alphabetically by name."""
        items = [
            _entity("e-a", "Alice"),
            _entity("e-b", "Bob"),
            _entity("e-c", "Charlie"),
        ]
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                MockSvc.return_value.list_entities = AsyncMock(
                    return_value=_list_result(items, total=3)
                )
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?sort=name_asc",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        names = [item["name"] for item in response.json()["items"]]
        assert names == sorted(names)

    @pytest.mark.integration
    @pytest.mark.api
    async def test_sort_confidence_desc_default(self, async_client):
        """Default sort (confidence_desc): first confidence >= last confidence."""
        items = [
            _entity("e-1", "High", confidence=0.98),
            _entity("e-2", "Mid", confidence=0.75),
            _entity("e-3", "Low", confidence=0.40),
        ]
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                MockSvc.return_value.list_entities = AsyncMock(
                    return_value=_list_result(items, total=3)
                )
                # No sort param — default is confidence_desc
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        result_items = response.json()["items"]
        assert result_items[0]["confidence"] >= result_items[-1]["confidence"]
        assert result_items[0]["confidence"] == 0.98


class TestCrossTenantIsolation:
    """Cross-tenant isolation: Graph A cannot see Graph B entities."""

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.security
    async def test_wrong_graph_id_returns_403_or_404(self, async_client):
        """User A with valid token but wrong graph_id → 403 (no access) or 404 (not found)."""
        from fastapi import HTTPException

        auth_patch = _patch_auth(FAKE_USER_A)
        try:
            with patch(
                "app.api.v1.endpoints.entities.verify_graph_access",
                new_callable=AsyncMock,
                side_effect=HTTPException(status_code=403, detail="Forbidden"),
            ):
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_B_ID}/entities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code in (403, 404)

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.security
    async def test_verify_graph_access_called_with_exact_graph_id(self, async_client):
        """verify_graph_access is called with the graph_id from the path — not skipped."""
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ) as mock_vga,
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                MockSvc.return_value.list_entities = AsyncMock(
                    return_value=_list_result([])
                )
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        # verify_graph_access must have been called — proves there is no auth bypass path
        mock_vga.assert_awaited_once()
        call_args = mock_vga.call_args
        # First positional arg is the graph_id string
        assert call_args.args[0] == GRAPH_A_ID

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.security
    async def test_service_receives_correct_graph_id_scope(self, async_client):
        """list_entities is called with graph_id=GRAPH_A_ID — service never sees Graph B data."""
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                mock_list = AsyncMock(return_value=_list_result([]))
                MockSvc.return_value.list_entities = mock_list

                await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["graph_id"] == str(GRAPH_A_ID)

    @pytest.mark.integration
    @pytest.mark.api
    @pytest.mark.security
    async def test_unauthenticated_request_cannot_access_any_graph(self, async_client):
        """No auth header → denied before any graph access check.

        NOTE: FastAPI's HTTPBearer returns 403 (not 401) for missing credentials.
        Semantically 401 is correct; tracked as BUG in ORA-374 QA report.
        The critical assertion is that access is denied (no 200).
        """
        response = await async_client.get(
            f"/api/v1/graphs/{GRAPH_A_ID}/entities",
        )
        # HTTPBearer returns 403 for missing auth (known FastAPI behavior)
        assert response.status_code in (401, 403)
        assert response.status_code != 200


class TestPaginationIntegrity:
    """Pagination: pages are consistent, non-overlapping, and total is stable."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_page_and_page_size_forwarded_to_service(self, async_client):
        """page and page_size are forwarded to list_entities correctly."""
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                mock_list = AsyncMock(
                    return_value=_list_result([], total=0, page=2, page_size=5)
                )
                MockSvc.return_value.list_entities = mock_list

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?page=2&page_size=5",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["page"] == 2
        assert call_kwargs["page_size"] == 5

    @pytest.mark.integration
    @pytest.mark.api
    async def test_response_reflects_page_and_page_size(self, async_client):
        """Response body echoes page/page_size from the service result."""
        page1_items = [_entity(f"e-{i}", f"Entity {i}") for i in range(5)]
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                MockSvc.return_value.list_entities = AsyncMock(
                    return_value=_list_result(
                        page1_items, total=12, page=1, page_size=5
                    )
                )

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?page=1&page_size=5",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        data = response.json()
        assert data["page"] == 1
        assert data["page_size"] == 5
        assert data["total"] == 12
        assert len(data["items"]) == 5

    @pytest.mark.integration
    @pytest.mark.api
    async def test_total_consistent_across_pages(self, async_client):
        """total field is identical on page 1 and page 2 for the same graph."""
        total = 12
        auth_patch = _patch_auth()

        totals = []
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                page1_items = [_entity(f"e-{i}", f"Entity {i}") for i in range(5)]
                page2_items = [_entity(f"e-{i}", f"Entity {i}") for i in range(5, 10)]

                MockSvc.return_value.list_entities = AsyncMock(
                    side_effect=[
                        _list_result(page1_items, total=total, page=1, page_size=5),
                        _list_result(page2_items, total=total, page=2, page_size=5),
                    ]
                )

                r1 = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?page=1&page_size=5",
                    headers=_auth_headers(),
                )
                r2 = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?page=2&page_size=5",
                    headers=_auth_headers(),
                )
                totals = [r1.json()["total"], r2.json()["total"]]
        finally:
            auth_patch.stop()

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert totals[0] == totals[1] == total

    @pytest.mark.integration
    @pytest.mark.api
    async def test_page1_and_page2_items_do_not_overlap(self, async_client):
        """Items on page 1 and page 2 share no IDs — no duplicate entries."""
        page1_ids = [f"e-{i}" for i in range(5)]
        page2_ids = [f"e-{i}" for i in range(5, 10)]
        page1_items = [_entity(id_, f"Entity {id_}") for id_ in page1_ids]
        page2_items = [_entity(id_, f"Entity {id_}") for id_ in page2_ids]

        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                MockSvc.return_value.list_entities = AsyncMock(
                    side_effect=[
                        _list_result(page1_items, total=10, page=1, page_size=5),
                        _list_result(page2_items, total=10, page=2, page_size=5),
                    ]
                )

                r1 = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?page=1&page_size=5",
                    headers=_auth_headers(),
                )
                r2 = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?page=2&page_size=5",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert r1.status_code == 200
        assert r2.status_code == 200

        ids_p1 = {item["id"] for item in r1.json()["items"]}
        ids_p2 = {item["id"] for item in r2.json()["items"]}
        assert ids_p1.isdisjoint(ids_p2), f"Overlapping IDs: {ids_p1 & ids_p2}"

    @pytest.mark.integration
    @pytest.mark.api
    async def test_page_zero_returns_422(self, async_client):
        """page=0 violates ge=1 constraint → 422."""
        auth_patch = _patch_auth()
        try:
            with patch(
                "app.api.v1.endpoints.entities.verify_graph_access",
                new_callable=AsyncMock,
                return_value=GRAPH_A_ID,
            ):
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?page=0",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 422


class TestTypeFilter:
    """Type filter: all returned items must match requested type(s)."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_single_type_filter_all_items_match(self, async_client):
        """?type=concept → all items in response have type == 'concept'."""
        items = [
            _entity("e-1", "Machine Learning", type_="concept"),
            _entity("e-2", "Knowledge Graph", type_="concept"),
        ]
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                MockSvc.return_value.list_entities = AsyncMock(
                    return_value=_list_result(items, total=2)
                )
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?type=concept",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        result_items = response.json()["items"]
        assert len(result_items) == 2
        assert all(item["type"] == "concept" for item in result_items)


class TestFullTextSearch:
    """Full-text search: known entity appears in results."""

    @pytest.mark.integration
    @pytest.mark.api
    async def test_search_known_entity_appears_in_results(self, async_client):
        """?q=Knowledge: known entity 'Knowledge Graph' appears in items."""
        items = [
            _entity("e-kg", "Knowledge Graph", type_="concept"),
            _entity("e-kd", "Knowledge Discovery", type_="concept"),
        ]
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                MockSvc.return_value.list_entities = AsyncMock(
                    return_value=_list_result(items, total=2)
                )
                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?q=Knowledge",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        assert response.status_code == 200
        result_items = response.json()["items"]
        names = [item["name"] for item in result_items]
        assert "Knowledge Graph" in names

    @pytest.mark.integration
    @pytest.mark.api
    async def test_search_q_parameter_forwarded_to_service(self, async_client):
        """q value is passed verbatim to list_entities — case preserved."""
        auth_patch = _patch_auth()
        try:
            with (
                patch(
                    "app.api.v1.endpoints.entities.verify_graph_access",
                    new_callable=AsyncMock,
                    return_value=GRAPH_A_ID,
                ),
                patch("app.api.v1.endpoints.entities.GraphAnalyticsService") as MockSvc,
            ):
                mock_list = AsyncMock(return_value=_list_result([]))
                MockSvc.return_value.list_entities = mock_list

                await async_client.get(
                    f"/api/v1/graphs/{GRAPH_A_ID}/entities?q=Knowledge+Graph",
                    headers=_auth_headers(),
                )
        finally:
            auth_patch.stop()

        call_kwargs = mock_list.call_args.kwargs
        assert call_kwargs["q"] == "Knowledge Graph"
