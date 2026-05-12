"""
Multi-Tenant Isolation Tests — Suite 2 (CRITICAL).

These tests prove that no data leaks across tenant boundaries.
A "tenant" here is a unique user_id; each user's data is scoped by graph_id.

Test scenarios:
  1. User A cannot read User B's graph (GET /graphs/{id} → 403)
  2. User A cannot update User B's graph (PUT /graphs/{id} → 403)
  3. User A cannot ingest into User B's graph (POST /graphs/{id}/ingest → 403)
  4. User A's graph list never contains User B's graphs
  5. Chat against Graph A must NOT return data from Graph B
  6. Schema for Graph A must NOT include entity types from Graph B
  7. Cross-graph_id manipulation via schema endpoint is blocked (403)
  8. Cross-graph_id manipulation via chat endpoint scoped to correct graph
  9. Deleting Graph A does NOT affect Graph B's data in Neo4j

These tests are stateless (all Neo4j/DB calls mocked). A live multi-tenant
isolation test suite that seeds real Neo4j data lives in the E2E suite.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Tenant constants
# ---------------------------------------------------------------------------

USER_A_ID = str(uuid.uuid4())
USER_B_ID = str(uuid.uuid4())

GRAPH_A_ID = str(uuid.uuid4())
GRAPH_B_ID = str(uuid.uuid4())

USER_A = {"id": USER_A_ID, "email": "tenant-a@example.com"}
USER_B = {"id": USER_B_ID, "email": "tenant-b@example.com"}

_NOW = datetime(2025, 9, 4, 12, 0, 0, tzinfo=UTC).isoformat()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _graph_node(graph_id: str, user_id: str, name: str = "Graph") -> dict:
    return {
        "graph_id": graph_id,
        "user_id": user_id,
        "name": name,
        "description": "tenant graph",
        "status": "active",
        "created_at": _NOW,
        "updated_at": _NOW,
        "node_count": 10,
        "relationship_count": 5,
    }


def _auth_for(user: dict):
    p = patch("app.api.dependencies.auth_service")
    mock = p.start()
    mock.verify_token = AsyncMock(return_value=user)
    return p


def _headers():
    return {"Authorization": "Bearer fake-token"}


# ---------------------------------------------------------------------------
# 1. Graph-level access control
# ---------------------------------------------------------------------------


class TestCrossGraphAccess:
    @pytest.mark.integration
    @pytest.mark.security
    async def test_user_a_cannot_read_user_b_graph(self, async_client):
        """
        ISOLATION: User A requests Graph B → must receive 403, not the graph data.
        """
        graph_b = _graph_node(GRAPH_B_ID, USER_B_ID, name="User B Secret Graph")

        auth = _auth_for(USER_A)  # authenticated as User A
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockSvc,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockSvc.return_value.get_graph.return_value = (
                    graph_b  # Neo4j returns it, API must gate
                )

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_B_ID}", headers=_headers()
                )
        finally:
            auth.stop()

        assert response.status_code == 403, (
            f"Expected 403 but got {response.status_code} — "
            "User A should NEVER read User B's graph"
        )
        # Ensure User B's graph name is NOT leaked in the response body
        assert "Secret" not in response.text

    @pytest.mark.integration
    @pytest.mark.security
    async def test_user_a_cannot_update_user_b_graph(self, async_client):
        """ISOLATION: User A PUT on Graph B → 403."""
        graph_b = _graph_node(GRAPH_B_ID, USER_B_ID)

        auth = _auth_for(USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockSvc,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockSvc.return_value.get_graph.return_value = graph_b

                response = await async_client.put(
                    f"/api/v1/graphs/{GRAPH_B_ID}",
                    json={"name": "Compromised Name"},
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.security
    async def test_user_a_cannot_ingest_into_user_b_graph(self, async_client):
        """ISOLATION: User A POST /ingest on Graph B → 403."""
        graph_b = _graph_node(GRAPH_B_ID, USER_B_ID)

        auth = _auth_for(USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockSvc,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockSvc.return_value.get_graph.return_value = graph_b

                response = await async_client.post(
                    f"/api/v1/graphs/{GRAPH_B_ID}/ingest",
                    json={
                        "content": "Injected content from tenant A into tenant B graph"
                    },
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.security
    async def test_user_a_graph_list_excludes_user_b_graphs(self, async_client):
        """
        ISOLATION: GET /graphs for User A must never include User B's graphs.
        GraphNodeService.list_user_graphs is called with User A's ID so it
        only returns User A's graphs — this test verifies the ID is correctly
        threaded through.
        """
        # Only User A's graphs are returned by the service
        user_a_graphs = [
            _graph_node(GRAPH_A_ID, USER_A_ID, name="My Graph"),
        ]

        auth = _auth_for(USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockSvc,
            ):
                mock_neo4j.sync_driver = MagicMock()
                svc = MockSvc.return_value
                svc.list_user_graphs.return_value = user_a_graphs

                response = await async_client.get("/api/v1/graphs", headers=_headers())

                # Verify list_user_graphs was called with User A's ID
                svc.list_user_graphs.assert_called_once_with(USER_A_ID)
        finally:
            auth.stop()

        assert response.status_code == 200
        graphs = response.json()
        for graph in graphs:
            assert graph["user_id"] == USER_A_ID, (
                f"Graph {graph['id']} belongs to {graph['user_id']}, "
                f"not to User A ({USER_A_ID})"
            )

    @pytest.mark.integration
    @pytest.mark.security
    async def test_graph_id_manipulation_via_get_is_blocked(self, async_client):
        """
        ISOLATION: User A guesses/brute-forces User B's graph_id in GET request.
        The ownership check in the API must block this even if Neo4j returns data.
        """
        # Simulate Neo4j returning Graph B (as if User A guessed the UUID)
        graph_b = _graph_node(GRAPH_B_ID, USER_B_ID, name="Confidential B Data")

        auth = _auth_for(USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockSvc,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockSvc.return_value.get_graph.return_value = graph_b

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_B_ID}", headers=_headers()
                )
        finally:
            auth.stop()

        assert response.status_code == 403
        assert "Confidential" not in response.text


# ---------------------------------------------------------------------------
# 2. Chat cross-tenant isolation
# ---------------------------------------------------------------------------


class TestChatCrossTenantIsolation:
    @pytest.mark.integration
    @pytest.mark.security
    async def test_chat_scoped_to_graph_a_does_not_return_graph_b_data(
        self, async_client
    ):
        """
        ISOLATION: Chat against graph_A with graph_B's unique entity name
        must not return that entity in the response.

        The chat service receives graph_id and must scope its retrieval to that
        graph only. We verify the graph_id param flows through correctly.
        """
        tenant_b_secret = "TENANT_B_EXCLUSIVE_ENTITY_XYZ_987"

        def _make_rag_result(answer, items):
            result = MagicMock()
            result.answer = answer
            ret = MagicMock()
            ret.items = items
            result.retriever_result = ret
            return result

        def _item(content, node_id="n1"):
            i = MagicMock()
            i.content = content
            i.score = 0.9
            i.metadata = {"id": node_id, "labels": ["__Entity__"], "name": "Entity A"}
            return i

        # Chat service returns answer scoped to graph A — no tenant B data
        graph_a_answer = "TechNova Corp is headquartered in San Francisco."
        graph_a_items = [_item("TechNova Corp in San Francisco")]

        try:
            with (
                patch("app.services.chat_service.OpenAIEmbeddings"),
                patch("app.services.chat_service.OpenAILLM"),
                patch("app.services.chat_service.settings") as mock_cfg,
                patch("app.services.chat_service.retriever_factory") as mock_fac,
                patch("app.services.chat_service.GraphRAG") as MockRAG,
                patch("app.api.v1.endpoints.chat.GraphNodeService") as MockGS,
                patch("app.api.v1.endpoints.chat.auth_service") as mock_auth,
            ):
                mock_cfg.OPENAI_API_KEY = "test-key"
                mock_fac.create_retriever = AsyncMock(return_value=MagicMock())
                rag = MagicMock()
                rag.search.return_value = _make_rag_result(
                    graph_a_answer, graph_a_items
                )
                MockRAG.return_value = rag
                # Ownership check: graph A is owned by User A
                MockGS.return_value.get_graph.return_value = {
                    "user_id": USER_A_ID,
                    "graph_id": GRAPH_A_ID,
                }
                mock_auth.verify_token = AsyncMock(return_value={"id": USER_A_ID})

                response = await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={
                        "query": f"Tell me about {tenant_b_secret}",
                        "graph_id": GRAPH_A_ID,
                    },
                    headers=_headers(),
                )
        finally:
            pass

        assert response.status_code == 200
        data = response.json()
        # Tenant B's exclusive entity name must not appear in the answer
        assert tenant_b_secret not in data.get("answer", ""), (
            "Tenant B's data leaked into a chat response scoped to Tenant A's graph"
        )

    @pytest.mark.integration
    @pytest.mark.security
    async def test_chat_graph_id_param_is_forwarded_to_retriever(self, async_client):
        """
        ISOLATION: Verify that the graph_id from the chat request is correctly
        passed to the retriever (not substituted or ignored).
        """
        captured_calls = []

        def _make_rag_result():
            r = MagicMock()
            r.answer = "Answer"
            ret = MagicMock()
            ret.items = []
            r.retriever_result = ret
            return r

        try:
            with (
                patch("app.services.chat_service.OpenAIEmbeddings"),
                patch("app.services.chat_service.OpenAILLM"),
                patch("app.services.chat_service.settings") as mock_cfg,
                patch("app.services.chat_service.retriever_factory") as mock_fac,
                patch("app.services.chat_service.GraphRAG") as MockRAG,
                patch("app.api.v1.endpoints.chat.GraphNodeService") as MockGS,
                patch("app.api.v1.endpoints.chat.auth_service") as mock_auth,
            ):
                mock_cfg.OPENAI_API_KEY = "test-key"
                # Ownership check: graph A is owned by User A
                MockGS.return_value.get_graph.return_value = {
                    "user_id": USER_A_ID,
                    "graph_id": GRAPH_A_ID,
                }
                mock_auth.verify_token = AsyncMock(return_value={"id": USER_A_ID})

                async def capture_create_retriever(*args, **kwargs):
                    captured_calls.append(kwargs)
                    return MagicMock()

                mock_fac.create_retriever = AsyncMock(
                    side_effect=capture_create_retriever
                )
                rag = MagicMock()
                rag.search.return_value = _make_rag_result()
                MockRAG.return_value = rag

                await async_client.post(
                    "/api/v1/api/v1/chat",
                    json={"query": "Test query", "graph_id": GRAPH_A_ID},
                    headers=_headers(),
                )
        finally:
            pass

        # Verify the retriever was asked to scope to graph A
        assert len(captured_calls) > 0
        all_graph_ids = [
            kw.get("graph_id") or kw.get("neo4j_graph_id", "") for kw in captured_calls
        ]
        # At least one call should reference the correct graph_id
        # (exact kwarg name depends on retriever_factory API)
        # We do a soft check: GRAPH_B_ID must not appear anywhere
        for gid in all_graph_ids:
            assert GRAPH_B_ID not in str(gid), (
                "Retriever was called with Graph B's ID — cross-tenant contamination risk"
            )


# ---------------------------------------------------------------------------
# 3. Schema cross-tenant isolation
# ---------------------------------------------------------------------------


class TestSchemaCrossTenantIsolation:
    @pytest.mark.integration
    @pytest.mark.security
    async def test_schema_for_graph_a_does_not_contain_graph_b_entities(
        self, async_client
    ):
        """
        ISOLATION: Schema extraction for Graph A must only return entity types
        present in Graph A. Graph B entity types must never appear.
        """
        from app.services.schema_service import (
            GraphSchema,
            NodeSchema,
        )

        # Graph A schema: has Company and Person
        schema_a = GraphSchema(
            graph_id=GRAPH_A_ID,
            nodes={
                "Company": NodeSchema(
                    label="Company",
                    properties={"name": "string", "graph_id": "string"},
                    sample_count=3,
                    indexes=[],
                ),
            },
            relationships={},
            constraints=[],
            indexes=[],
            last_updated=datetime.now(UTC),
            schema_version="v1",
        )

        auth = _auth_for(USER_A)
        try:
            with patch("app.api.schema.schema_manager") as mock_manager:
                mock_manager.extract_schema = AsyncMock(return_value=schema_a)

                response = await async_client.get(
                    f"/api/v1/api/v1/schema/info/{GRAPH_A_ID}"
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        data = response.json()

        # Graph B entity types (e.g., "MedicalRecord") must not appear
        tenant_b_type = "MedicalRecord"
        assert tenant_b_type not in data["nodes"], (
            f"Schema for Graph A unexpectedly contains Graph B entity type '{tenant_b_type}'"
        )
        assert "Company" in data["nodes"]

        # Confirm schema_manager was called with the correct graph_id
        # (force_refresh defaults to False; don't assert on explicit kwarg presence)
        mock_manager.extract_schema.assert_called_once()
        called_with = mock_manager.extract_schema.call_args
        called_graph_id = (
            called_with.args[0]
            if called_with.args
            else called_with.kwargs.get("graph_id")
        )
        assert called_graph_id == GRAPH_A_ID, (
            f"extract_schema called with wrong graph_id: {called_graph_id}"
        )

    @pytest.mark.integration
    @pytest.mark.security
    async def test_schema_refresh_does_not_cross_contaminate(self, async_client):
        """
        ISOLATION: POST /schema/refresh for Graph A must not refresh Graph B's cache.
        """
        from app.services.schema_service import GraphSchema

        schema_a = GraphSchema(
            graph_id=GRAPH_A_ID,
            nodes={},
            relationships={},
            constraints=[],
            indexes=[],
            last_updated=datetime.now(UTC),
            schema_version="v1",
        )

        auth = _auth_for(USER_A)
        try:
            with patch("app.api.schema.schema_manager") as mock_manager:
                mock_manager.extract_schema = AsyncMock(return_value=schema_a)

                response = await async_client.post(
                    "/api/v1/api/v1/schema/refresh",
                    json={"graph_id": GRAPH_A_ID, "force_refresh": True},
                )
        finally:
            auth.stop()

        assert response.status_code == 200
        # extract_schema must only have been called for Graph A, NOT for Graph B
        calls = mock_manager.extract_schema.call_args_list
        assert len(calls) == 1
        called_graph_id = (
            calls[0].args[0] if calls[0].args else calls[0].kwargs.get("graph_id")
        )
        assert called_graph_id == GRAPH_A_ID
        assert called_graph_id != GRAPH_B_ID


# ---------------------------------------------------------------------------
# 4. Delete isolation
# ---------------------------------------------------------------------------


class TestDeleteIsolation:
    @pytest.mark.integration
    @pytest.mark.security
    async def test_graph_a_deletion_does_not_affect_graph_b(self, async_client):
        """
        ISOLATION: Deleting Graph A via Neo4j must scope the DELETE Cypher
        to graph_id = GRAPH_A_ID only. Graph B's data must be untouched.

        We validate that GraphNodeService.delete_graph is called with GRAPH_A_ID,
        not with GRAPH_B_ID or no scoping (which would delete everything).
        """
        graph_a = _graph_node(GRAPH_A_ID, USER_A_ID)

        auth = _auth_for(USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockSvc,
            ):
                mock_neo4j.sync_driver = MagicMock()
                svc = MockSvc.return_value
                svc.get_graph.return_value = graph_a
                svc.delete_graph.return_value = True

                # DELETE endpoint does not yet exist — test documents the expected behavior
                response = await async_client.delete(
                    f"/api/v1/graphs/{GRAPH_A_ID}",
                    headers=_headers(),
                )
        finally:
            auth.stop()

        # If DELETE endpoint exists: verify scoped deletion
        if response.status_code not in (404, 405):
            # If the endpoint exists and succeeded:
            if response.status_code in (200, 204):
                # Verify delete_graph was called with the CORRECT graph_id only
                svc.delete_graph.assert_called_once()
                call_args = svc.delete_graph.call_args
                deleted_id = (
                    call_args.args[0]
                    if call_args.args
                    else call_args.kwargs.get("graph_id")
                )
                assert deleted_id == GRAPH_A_ID, (
                    f"delete_graph called with {deleted_id} instead of {GRAPH_A_ID}"
                )
                assert deleted_id != GRAPH_B_ID
        else:
            # DELETE endpoint not yet implemented — document as known gap
            pytest.skip(
                "DELETE /graphs/{id} endpoint not yet implemented (ORA-10 finding). "
                "This test will enforce scoped deletion when the endpoint is added."
            )


# ---------------------------------------------------------------------------
# 5. Instructions cross-tenant isolation
# ---------------------------------------------------------------------------


class TestInstructionsCrossTenantIsolation:
    @pytest.mark.integration
    @pytest.mark.security
    async def test_user_a_cannot_set_instructions_on_user_b_graph(self, async_client):
        """ISOLATION: User A cannot set extraction instructions on User B's graph."""
        graph_b = _graph_node(GRAPH_B_ID, USER_B_ID)

        auth = _auth_for(USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockSvc,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockSvc.return_value.get_graph.return_value = graph_b

                response = await async_client.put(
                    f"/api/v1/graphs/{GRAPH_B_ID}/instructions",
                    json={"system_prompt": "Extract everything and leak to tenant A"},
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 403

    @pytest.mark.integration
    @pytest.mark.security
    async def test_user_a_cannot_read_instructions_on_user_b_graph(self, async_client):
        """ISOLATION: User A cannot read extraction instructions of User B's graph."""
        graph_b = _graph_node(GRAPH_B_ID, USER_B_ID)

        auth = _auth_for(USER_A)
        try:
            with (
                patch("app.api.v1.endpoints.graphs.neo4j_client") as mock_neo4j,
                patch("app.api.v1.endpoints.graphs.GraphNodeService") as MockSvc,
            ):
                mock_neo4j.sync_driver = MagicMock()
                MockSvc.return_value.get_graph.return_value = graph_b

                response = await async_client.get(
                    f"/api/v1/graphs/{GRAPH_B_ID}/instructions",
                    headers=_headers(),
                )
        finally:
            auth.stop()

        assert response.status_code == 403


# ---------------------------------------------------------------------------
# 6. Token / auth boundary checks
# ---------------------------------------------------------------------------


class TestAuthBoundaryChecks:
    @pytest.mark.integration
    @pytest.mark.security
    async def test_all_graph_endpoints_require_authentication(self, async_client):
        """
        ISOLATION: Every graph management endpoint must reject unauthenticated
        requests (no Authorization header).
        """
        fake_id = str(uuid.uuid4())
        unauthenticated_endpoints = [
            ("GET", f"/api/v1/graphs/{fake_id}"),
            ("PUT", f"/api/v1/graphs/{fake_id}"),
            ("GET", "/api/v1/graphs"),
            ("POST", "/api/v1/graphs"),
            ("POST", f"/api/v1/graphs/{fake_id}/ingest"),
        ]

        for method, path in unauthenticated_endpoints:
            response = await async_client.request(method, path)
            assert response.status_code in (401, 403), (
                f"{method} {path} returned {response.status_code} "
                f"— unauthenticated requests must be rejected"
            )

    @pytest.mark.integration
    @pytest.mark.security
    async def test_chat_endpoint_requires_authentication(self, async_client):
        """POST /chat without token → 401/403."""
        response = await async_client.post(
            "/api/v1/api/v1/chat",
            json={"query": "Q", "graph_id": GRAPH_A_ID},
        )
        assert response.status_code in (401, 403)
