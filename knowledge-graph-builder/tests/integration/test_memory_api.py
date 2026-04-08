"""
Integration tests for the Agent Memory API.

Tests the full request → endpoint → memory_service → response pipeline,
mocking only the Neo4j driver calls and auth dependencies.

Endpoints tested:
  POST   /api/v1/graphs/{graphId}/memories
  GET    /api/v1/graphs/{graphId}/memories/search
  GET    /api/v1/graphs/{graphId}/memories/context
  PATCH  /api/v1/graphs/{graphId}/memories/{memoryId}
  DELETE /api/v1/graphs/{graphId}/memories/{memoryId}
  POST   /api/v1/graphs/{graphId}/memories/consolidate
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app

GRAPH_ID = "test-graph-memory-api"
MEMORY_ID = "mem-00000000-0000-0000-0000-000000000001"

# ------------------------------------------------------------------ #
# Auth + ReBAC bypass
# ------------------------------------------------------------------ #

def _patch_auth():
    """Patch auth so every request is authenticated as user-1."""
    return patch(
        "app.api.dependencies.auth_service.verify_token",
        new=AsyncMock(return_value={"id": "user-1", "email": "test@example.com"}),
    )


def _patch_rebac():
    """Patch ReBAC so graph access is always granted."""
    return patch(
        "app.api.dependencies.rebac_service.check_graph_permission",
        new=AsyncMock(return_value=True),
    )


def _patch_neo4j_driver():
    """Provide a dummy async_driver so the rebac dependency doesn't 503."""
    mock_driver = MagicMock()
    return patch("app.api.dependencies.neo4j_client.async_driver", mock_driver)


# ------------------------------------------------------------------ #
# POST /memories — store
# ------------------------------------------------------------------ #

class TestStoreMemoryEndpoint:
    @pytest.mark.integration
    async def test_store_memory_returns_201(self, async_client):
        store_result = MagicMock()
        store_result.memory_id = MEMORY_ID
        store_result.importance_score = 0.8
        store_result.contradictions_detected = []
        store_result.entity_linked = None

        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.memory_service.store_memory",
                 new=AsyncMock(return_value=store_result),
             ):
            resp = await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/memories",
                json={
                    "type": "semantic",
                    "content": "Reza is the CEO of DeAgenticAI",
                    "subject": "Reza",
                    "predicate": "IS_CEO_OF",
                    "object": "DeAgenticAI",
                    "confidence": 0.95,
                    "scope": "organization",
                },
                headers={"Authorization": "Bearer test-token"},
            )

        assert resp.status_code == 201
        data = resp.json()
        assert data["memory_id"] == MEMORY_ID
        assert data["importance_score"] == pytest.approx(0.8)
        assert data["contradictions_detected"] == []

    @pytest.mark.integration
    async def test_store_memory_with_contradictions(self, async_client):
        store_result = MagicMock()
        store_result.memory_id = MEMORY_ID
        store_result.importance_score = 0.8
        store_result.contradictions_detected = [
            MagicMock(
                model_dump=lambda: {
                    "conflict_memory_id": "old-id",
                    "content": "Bob is CEO",
                    "resolution": "new_wins",
                }
            )
        ]
        store_result.entity_linked = None

        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.memory_service.store_memory",
                 new=AsyncMock(return_value=store_result),
             ):
            resp = await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/memories",
                json={"type": "semantic", "content": "Alice is CEO", "confidence": 0.9},
                headers={"Authorization": "Bearer test-token"},
            )

        assert resp.status_code == 201

    @pytest.mark.integration
    async def test_store_memory_missing_content_returns_422(self, async_client):
        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver():
            resp = await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/memories",
                json={"type": "semantic"},  # missing content
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code == 422


# ------------------------------------------------------------------ #
# GET /memories/search
# ------------------------------------------------------------------ #

class TestSearchMemoriesEndpoint:
    @pytest.mark.integration
    async def test_search_returns_200(self, async_client):
        search_result = MagicMock()
        search_result.memories = []
        search_result.graph_facts = []
        search_result.total = 0

        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.memory_service.search_memories",
                 new=AsyncMock(return_value=search_result),
             ):
            resp = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/memories/search",
                params={"query": "Reza CEO"},
                headers={"Authorization": "Bearer test-token"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "memories" in data
        assert "total" in data

    @pytest.mark.integration
    async def test_search_missing_query_returns_422(self, async_client):
        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver():
            resp = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/memories/search",
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code == 422

    @pytest.mark.integration
    async def test_search_passes_filters(self, async_client):
        search_result = MagicMock()
        search_result.memories = []
        search_result.graph_facts = []
        search_result.total = 0

        mock_search = AsyncMock(return_value=search_result)

        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.memory_service.search_memories",
                 new=mock_search,
             ):
            await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/memories/search",
                params={
                    "query": "CEO",
                    "type": "semantic",
                    "scope": "organization",
                    "min_confidence": "0.7",
                    "limit": "10",
                },
                headers={"Authorization": "Bearer test-token"},
            )

        call_kwargs = mock_search.call_args[1]
        assert call_kwargs["graph_id"] == GRAPH_ID
        assert call_kwargs["min_confidence"] == pytest.approx(0.7)
        assert call_kwargs["limit"] == 10


# ------------------------------------------------------------------ #
# GET /memories/context
# ------------------------------------------------------------------ #

class TestGetContextEndpoint:
    @pytest.mark.integration
    async def test_context_returns_200(self, async_client):
        ctx_result = MagicMock()
        ctx_result.context_block = "## Relevant Memory\n\n**Facts:**\n- Reza is CEO"
        ctx_result.memories_used = [MEMORY_ID]
        ctx_result.token_estimate = 42
        ctx_result.retrieval_ms = 15

        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.memory_service.get_context",
                 new=AsyncMock(return_value=ctx_result),
             ):
            resp = await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/memories/context",
                params={"query": "help with Q4 planning"},
                headers={"Authorization": "Bearer test-token"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "context_block" in data
        assert "memories_used" in data
        assert "token_estimate" in data

    @pytest.mark.integration
    async def test_context_passes_scope_list(self, async_client):
        ctx_result = MagicMock()
        ctx_result.context_block = ""
        ctx_result.memories_used = []
        ctx_result.token_estimate = 0
        ctx_result.retrieval_ms = 5

        mock_ctx = AsyncMock(return_value=ctx_result)

        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.memory_service.get_context",
                 new=mock_ctx,
             ):
            await async_client.get(
                f"/api/v1/graphs/{GRAPH_ID}/memories/context",
                params={"query": "test", "scope": "user,organization"},
                headers={"Authorization": "Bearer test-token"},
            )

        call_kwargs = mock_ctx.call_args[1]
        assert call_kwargs["scopes"] == ["user", "organization"]


# ------------------------------------------------------------------ #
# PATCH /memories/{memoryId}
# ------------------------------------------------------------------ #

class TestUpdateMemoryEndpoint:
    @pytest.mark.integration
    async def test_update_returns_200(self, async_client):
        from datetime import datetime, timezone

        update_result = MagicMock()
        update_result.old_memory_id = MEMORY_ID
        update_result.new_memory_id = "mem-new-id"
        update_result.superseded_at = datetime.now(timezone.utc)

        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.memory_service.update_memory",
                 new=AsyncMock(return_value=update_result),
             ):
            resp = await async_client.patch(
                f"/api/v1/graphs/{GRAPH_ID}/memories/{MEMORY_ID}",
                json={"content": "Reza stepped down, now Chairman", "reason": "correction"},
                headers={"Authorization": "Bearer test-token"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["old_memory_id"] == MEMORY_ID
        assert data["new_memory_id"] == "mem-new-id"

    @pytest.mark.integration
    async def test_update_not_found_returns_404(self, async_client):
        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.memory_service.update_memory",
                 new=AsyncMock(side_effect=ValueError("Memory nonexistent not found in graph")),
             ):
            resp = await async_client.patch(
                f"/api/v1/graphs/{GRAPH_ID}/memories/nonexistent",
                json={"content": "new content"},
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code == 404


# ------------------------------------------------------------------ #
# DELETE /memories/{memoryId}
# ------------------------------------------------------------------ #

class TestDeleteMemoryEndpoint:
    @pytest.mark.integration
    async def test_soft_delete_returns_204(self, async_client):
        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.memory_service.delete_memory",
                 new=AsyncMock(return_value=None),
             ):
            resp = await async_client.delete(
                f"/api/v1/graphs/{GRAPH_ID}/memories/{MEMORY_ID}",
                headers={"Authorization": "Bearer test-token"},
            )
        assert resp.status_code == 204

    @pytest.mark.integration
    async def test_hard_delete_passes_flag(self, async_client):
        mock_delete = AsyncMock(return_value=None)

        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.memory_service.delete_memory",
                 new=mock_delete,
             ):
            await async_client.delete(
                f"/api/v1/graphs/{GRAPH_ID}/memories/{MEMORY_ID}",
                params={"hard": "true"},
                headers={"Authorization": "Bearer test-token"},
            )

        call_kwargs = mock_delete.call_args[1]
        assert call_kwargs["hard"] is True


# ------------------------------------------------------------------ #
# POST /memories/consolidate
# ------------------------------------------------------------------ #

class TestConsolidateEndpoint:
    @pytest.mark.integration
    async def test_consolidate_queues_task(self, async_client):
        mock_task = MagicMock()
        mock_task.id = "celery-task-id-1"

        with _patch_auth(), _patch_rebac(), _patch_neo4j_driver(), \
             patch(
                 "app.api.v1.endpoints.memories.consolidate_memories_task"
             ) as mock_celery_task:
            mock_celery_task.delay.return_value = mock_task
            resp = await async_client.post(
                f"/api/v1/graphs/{GRAPH_ID}/memories/consolidate",
                headers={"Authorization": "Bearer test-token"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "celery-task-id-1"
        assert "queued" in data["message"].lower() or GRAPH_ID in data["message"]
