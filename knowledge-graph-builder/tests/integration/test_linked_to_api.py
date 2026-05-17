"""Integration tests for the cross-subgraph LINKED_TO API (TASK-204).

Covers:
  1. POST /graphs/{src}/linked-to — link two same-org subgraphs (201)
  2. POST a cross-org link is rejected (400)
  3. POST a self-link is rejected (400)
  4. POST /graphs/{src}/entities/{eid}/linked-to — entity link (201)
  5. GET /graphs/{id}/linked-to — visible to an admin/editor on the source,
     HIDDEN from a viewer when min_role=editor (ADR-021 §4)
  6. DELETE /graphs/{src}/linked-to/{tgt} — removes a link (204)
  7. DELETE of an unknown link → 404

These tests hit the real FastAPI app and Neo4j. Auth is mocked via
dependency_overrides (per-client ``X-Test-User`` header), mirroring
``test_org_members_api.py``.

Each test creates :Graph:__Platform__ nodes wired with
``(:Organization)-[:OWNS]->(:Graph)`` directly in Neo4j, sets each graph's
``org_id``, bootstraps the 5 ReBAC :Role nodes per graph, and grants the
test users the ReBAC roles they need so the real ``verify_graph_access``
checks pass. All test nodes use the ``__lttest__`` name prefix for teardown.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from fastapi import Request
from httpx import ASGITransport, AsyncClient

from app.services.rebac_service import rebac_service

# ---------------------------------------------------------------------------
# Constants — test principals
# ---------------------------------------------------------------------------

ADMIN_USER_ID = str(uuid.uuid4())  # admin/owner on source graphs
VIEWER_USER_ID = str(uuid.uuid4())  # viewer-only on source graphs

_TEST_USER_IDS = [ADMIN_USER_ID, VIEWER_USER_ID]


# ---------------------------------------------------------------------------
# Auth override / clients
# ---------------------------------------------------------------------------


async def _mock_user_id(request: Request) -> str:
    """Resolve the test user from the per-client ``X-Test-User`` header."""
    return request.headers.get("x-test-user", ADMIN_USER_ID)


def _client_for(user_id: str) -> AsyncClient:
    """Build an async HTTP client that authenticates as *user_id*."""
    from app.api.dependencies import get_current_user_id
    from app.main import app

    app.dependency_overrides[get_current_user_id] = _mock_user_id
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-Test-User": user_id},
    )


@pytest_asyncio.fixture
async def admin_client():
    """Async HTTP client authenticated as ADMIN_USER_ID."""
    client = _client_for(ADMIN_USER_ID)
    async with client as c:
        yield c
    from app.main import app

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def viewer_client():
    """Async HTTP client authenticated as VIEWER_USER_ID."""
    client = _client_for(VIEWER_USER_ID)
    async with client as c:
        yield c
    from app.main import app

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Neo4j fixture helpers
# ---------------------------------------------------------------------------


async def _clean_neo4j(driver) -> None:
    """Remove __lttest__ :Graph / :Organization / :__Entity__ / :Role nodes
    plus test :User nodes (and all their edges)."""
    async with driver.session() as session:
        await session.run(
            "MATCH (u:User) WHERE u.user_id IN $ids DETACH DELETE u",
            {"ids": _TEST_USER_IDS},
        )
        await session.run(
            "MATCH (n) WHERE n.name STARTS WITH '__lttest__' DETACH DELETE n",
        )
        # Test entities are keyed by id prefix, not name.
        await session.run(
            "MATCH (e:__Entity__) WHERE e.id STARTS WITH '__lttest__' DETACH DELETE e",
        )
        # :Role nodes belong to the test graphs — match by graph_id set below.


async def _make_org(driver, org_id: str) -> None:
    """Create an :Organization:__Platform__ node with the given org_id."""
    async with driver.session() as session:
        await session.run(
            """
            CREATE (o:Organization:__Platform__ {
                org_id: $org_id,
                name: $name,
                created_at: datetime()
            })
            """,
            {"org_id": org_id, "name": f"__lttest__ org {org_id[:8]}"},
        )


async def _make_graph(driver, org_id: str, owner_user_id: str) -> str:
    """Create a :Graph:__Platform__ OWNED by *org_id*, carrying ``org_id``.

    Bootstraps the 5 ReBAC :Role nodes and returns the new graph_id.
    """
    graph_id = str(uuid.uuid4())
    async with driver.session() as session:
        await session.run(
            """
            MATCH (o:Organization:__Platform__ {org_id: $org_id})
            CREATE (g:Graph:__Platform__ {
                graph_id: $graph_id,
                name: $name,
                org_id: $org_id,
                user_id: $owner_user_id,
                status: 'active',
                created_at: datetime(),
                updated_at: datetime()
            })
            MERGE (o)-[:OWNS]->(g)
            """,
            {
                "org_id": org_id,
                "graph_id": graph_id,
                "name": f"__lttest__ graph {graph_id[:8]}",
                "owner_user_id": owner_user_id,
            },
        )
    # grant_role MATCHes pre-existing :Role nodes — bootstrap them first.
    await rebac_service.bootstrap_graph_roles(driver, graph_id, owner_user_id)
    return graph_id


async def _make_entity(driver, graph_id: str, entity_id: str) -> None:
    """Create a :__Entity__ node keyed by (graph_id, id) — matches the
    structured_ingest entity MERGE key."""
    async with driver.session() as session:
        await session.run(
            """
            CREATE (e:TestThing:__Entity__ {
                id: $id, graph_id: $graph_id, name: $name
            })
            """,
            {
                "id": entity_id,
                "graph_id": graph_id,
                "name": f"__lttest__ entity {entity_id}",
            },
        )


async def _grant(driver, graph_id: str, user_id: str, role: str) -> None:
    """Grant *user_id* the named ReBAC role on *graph_id*."""
    await rebac_service.grant_role(
        driver,
        graph_id=graph_id,
        target_user_id=user_id,
        role_name=role,
        granted_by=ADMIN_USER_ID,
    )


async def _drop_test_roles(driver, graph_ids: list[str]) -> None:
    """Delete :Role nodes for the given test graphs (teardown)."""
    async with driver.session() as session:
        await session.run(
            "MATCH (r:Role) WHERE r.graph_id IN $gids DETACH DELETE r",
            {"gids": graph_ids},
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_graph_link_same_org(
    admin_client: AsyncClient, neo4j_test_driver
) -> None:
    """A subgraph link between two same-org graphs is created (201)."""
    await _clean_neo4j(neo4j_test_driver)
    org_id = str(uuid.uuid4())
    created: list[str] = []
    try:
        await _make_org(neo4j_test_driver, org_id)
        src = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        tgt = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        created = [src, tgt]
        await _grant(neo4j_test_driver, src, ADMIN_USER_ID, "owner")

        resp = await admin_client.post(
            f"/api/v1/graphs/{src}/linked-to",
            json={"target_graph_id": tgt, "min_role": "viewer"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["source_graph_id"] == src
        assert body["target_graph_id"] == tgt
        assert body["min_role"] == "viewer"
        assert body["created_by"] == ADMIN_USER_ID
        assert body["created_at"]
    finally:
        await _clean_neo4j(neo4j_test_driver)
        await _drop_test_roles(neo4j_test_driver, created)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reject_cross_org_link(
    admin_client: AsyncClient, neo4j_test_driver
) -> None:
    """A link whose graphs belong to different orgs is rejected (400)."""
    await _clean_neo4j(neo4j_test_driver)
    org_a = str(uuid.uuid4())
    org_b = str(uuid.uuid4())
    created: list[str] = []
    try:
        await _make_org(neo4j_test_driver, org_a)
        await _make_org(neo4j_test_driver, org_b)
        src = await _make_graph(neo4j_test_driver, org_a, ADMIN_USER_ID)
        tgt = await _make_graph(neo4j_test_driver, org_b, ADMIN_USER_ID)
        created = [src, tgt]
        await _grant(neo4j_test_driver, src, ADMIN_USER_ID, "owner")

        resp = await admin_client.post(
            f"/api/v1/graphs/{src}/linked-to",
            json={"target_graph_id": tgt, "min_role": "viewer"},
        )
        assert resp.status_code == 400, resp.text
        assert "organization" in resp.json()["detail"].lower()
    finally:
        await _clean_neo4j(neo4j_test_driver)
        await _drop_test_roles(neo4j_test_driver, created)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_reject_self_link(admin_client: AsyncClient, neo4j_test_driver) -> None:
    """A subgraph link from a graph to itself is rejected (400)."""
    await _clean_neo4j(neo4j_test_driver)
    org_id = str(uuid.uuid4())
    created: list[str] = []
    try:
        await _make_org(neo4j_test_driver, org_id)
        src = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        created = [src]
        await _grant(neo4j_test_driver, src, ADMIN_USER_ID, "owner")

        resp = await admin_client.post(
            f"/api/v1/graphs/{src}/linked-to",
            json={"target_graph_id": src, "min_role": "viewer"},
        )
        assert resp.status_code == 400, resp.text
        assert "itself" in resp.json()["detail"].lower()
    finally:
        await _clean_neo4j(neo4j_test_driver)
        await _drop_test_roles(neo4j_test_driver, created)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_entity_link(admin_client: AsyncClient, neo4j_test_driver) -> None:
    """An entity link across two same-org subgraphs is created (201)."""
    await _clean_neo4j(neo4j_test_driver)
    org_id = str(uuid.uuid4())
    created: list[str] = []
    try:
        await _make_org(neo4j_test_driver, org_id)
        src = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        tgt = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        created = [src, tgt]
        src_eid = "__lttest__ent-src"
        tgt_eid = "__lttest__ent-tgt"
        await _make_entity(neo4j_test_driver, src, src_eid)
        await _make_entity(neo4j_test_driver, tgt, tgt_eid)
        await _grant(neo4j_test_driver, src, ADMIN_USER_ID, "owner")

        resp = await admin_client.post(
            f"/api/v1/graphs/{src}/entities/{src_eid}/linked-to",
            json={
                "target_graph_id": tgt,
                "target_entity_id": tgt_eid,
                "min_role": "viewer",
            },
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["source_graph_id"] == src
        assert body["source_entity_id"] == src_eid
        assert body["target_graph_id"] == tgt
        assert body["target_entity_id"] == tgt_eid
        assert body["min_role"] == "viewer"

        # GET shows it to the admin.
        listed = await admin_client.get(
            f"/api/v1/graphs/{src}/entities/{src_eid}/linked-to"
        )
        assert listed.status_code == 200, listed.text
        assert len(listed.json()) == 1
        assert listed.json()[0]["target_entity_id"] == tgt_eid
    finally:
        await _clean_neo4j(neo4j_test_driver)
        await _drop_test_roles(neo4j_test_driver, created)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_min_role_visibility_filter(
    admin_client: AsyncClient, viewer_client: AsyncClient, neo4j_test_driver
) -> None:
    """A link with min_role=editor is visible to admin/editor on the source
    but HIDDEN from a viewer (ADR-021 §4 — source-subgraph role gate)."""
    await _clean_neo4j(neo4j_test_driver)
    org_id = str(uuid.uuid4())
    created: list[str] = []
    try:
        await _make_org(neo4j_test_driver, org_id)
        src = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        tgt = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        created = [src, tgt]
        # Admin user: owner role on src (can create + see anything).
        await _grant(neo4j_test_driver, src, ADMIN_USER_ID, "owner")
        # Viewer user: viewer role on src — has read access but role index
        # 3 > editor index 2, so an editor-gated link must be hidden.
        await _grant(neo4j_test_driver, src, VIEWER_USER_ID, "viewer")

        create = await admin_client.post(
            f"/api/v1/graphs/{src}/linked-to",
            json={"target_graph_id": tgt, "min_role": "editor"},
        )
        assert create.status_code == 201, create.text

        # Admin (owner role) sees the link.
        admin_list = await admin_client.get(f"/api/v1/graphs/{src}/linked-to")
        assert admin_list.status_code == 200, admin_list.text
        assert len(admin_list.json()) == 1
        assert admin_list.json()[0]["target_graph_id"] == tgt

        # Viewer (viewer role) has read access but the editor-gated link
        # is filtered out — empty list, not 403.
        viewer_list = await viewer_client.get(f"/api/v1/graphs/{src}/linked-to")
        assert viewer_list.status_code == 200, viewer_list.text
        assert viewer_list.json() == []
    finally:
        await _clean_neo4j(neo4j_test_driver)
        await _drop_test_roles(neo4j_test_driver, created)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_graph_link(admin_client: AsyncClient, neo4j_test_driver) -> None:
    """DELETE removes an existing subgraph link (204)."""
    await _clean_neo4j(neo4j_test_driver)
    org_id = str(uuid.uuid4())
    created: list[str] = []
    try:
        await _make_org(neo4j_test_driver, org_id)
        src = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        tgt = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        created = [src, tgt]
        await _grant(neo4j_test_driver, src, ADMIN_USER_ID, "owner")

        create = await admin_client.post(
            f"/api/v1/graphs/{src}/linked-to",
            json={"target_graph_id": tgt, "min_role": "viewer"},
        )
        assert create.status_code == 201, create.text

        delete = await admin_client.delete(f"/api/v1/graphs/{src}/linked-to/{tgt}")
        assert delete.status_code == 204, delete.text

        # Link is gone.
        listed = await admin_client.get(f"/api/v1/graphs/{src}/linked-to")
        assert listed.json() == []
    finally:
        await _clean_neo4j(neo4j_test_driver)
        await _drop_test_roles(neo4j_test_driver, created)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_delete_unknown_link_404(
    admin_client: AsyncClient, neo4j_test_driver
) -> None:
    """DELETE of a non-existent subgraph link returns 404."""
    await _clean_neo4j(neo4j_test_driver)
    org_id = str(uuid.uuid4())
    created: list[str] = []
    try:
        await _make_org(neo4j_test_driver, org_id)
        src = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        tgt = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        created = [src, tgt]
        await _grant(neo4j_test_driver, src, ADMIN_USER_ID, "owner")

        # No link was ever created between src and tgt.
        delete = await admin_client.delete(f"/api/v1/graphs/{src}/linked-to/{tgt}")
        assert delete.status_code == 404, delete.text
    finally:
        await _clean_neo4j(neo4j_test_driver)
        await _drop_test_roles(neo4j_test_driver, created)
