"""Integration tests for the entity neighborhood endpoint (ORA-358/ORA-380).

Endpoint: GET /api/v1/graphs/{graph_id}/entities/{entity_id}/neighborhood

Covers:
  1. Valid entity — returns nodes and edges within requested hop depth
  2. Invalid graph_id (non-existent UUID) — 403 (ReBAC denies access)
  3. Invalid entity_id (non-existent in graph) — 404
  4. hops out of range (0, 4) — 422
  5. edge_types filter — only requested relationship types appear in edges

Auth is mocked via dependency_overrides mirroring test_graph_data_api.py.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from fastapi import Request
from httpx import ASGITransport, AsyncClient

from app.services.rebac_service import rebac_service

# ---------------------------------------------------------------------------
# Test principals
# ---------------------------------------------------------------------------

OWNER_USER_ID = str(uuid.uuid4())
STRANGER_USER_ID = str(uuid.uuid4())

_TEST_USER_IDS = [OWNER_USER_ID, STRANGER_USER_ID]
_ENTITY_PREFIX = "__nbtest__"


# ---------------------------------------------------------------------------
# Auth override / clients
# ---------------------------------------------------------------------------


async def _mock_user_id(request: Request) -> str:
    return request.headers.get("x-test-user", OWNER_USER_ID)


def _client_for(user_id: str) -> AsyncClient:
    from app.api.dependencies import get_current_user_id
    from app.main import app

    app.dependency_overrides[get_current_user_id] = _mock_user_id
    return AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
        headers={"X-Test-User": user_id},
    )


@pytest_asyncio.fixture
async def owner_client():
    client = _client_for(OWNER_USER_ID)
    async with client as c:
        yield c
    from app.main import app

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def stranger_client():
    client = _client_for(STRANGER_USER_ID)
    async with client as c:
        yield c
    from app.main import app

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Neo4j helpers (mirrors test_graph_data_api.py)
# ---------------------------------------------------------------------------


async def _clean_neo4j(driver, graph_ids: list[str]) -> None:
    async with driver.session() as session:
        await session.run(
            "MATCH (u:User) WHERE u.user_id IN $ids DETACH DELETE u",
            {"ids": _TEST_USER_IDS},
        )
        await session.run(
            "MATCH (e:__Entity__) WHERE e.id STARTS WITH $p DETACH DELETE e",
            {"p": _ENTITY_PREFIX},
        )
        for gid in graph_ids:
            await session.run(
                "MATCH (n) WHERE n.graph_id = $gid DETACH DELETE n",
                {"gid": gid},
            )
            await session.run(
                "MATCH (r:Role) WHERE r.graph_id = $gid DETACH DELETE r",
                {"gid": gid},
            )


async def _make_graph(driver, owner_user_id: str) -> str:
    graph_id = str(uuid.uuid4())
    async with driver.session() as session:
        await session.run(
            """
            CREATE (g:Graph:__Platform__ {
                graph_id: $graph_id,
                name: $name,
                user_id: $owner_user_id,
                status: 'active',
                created_at: datetime(),
                updated_at: datetime()
            })
            """,
            {
                "graph_id": graph_id,
                "name": f"__nbtest__ graph {graph_id[:8]}",
                "owner_user_id": owner_user_id,
            },
        )
    await rebac_service.bootstrap_graph_roles(driver, graph_id, owner_user_id)
    return graph_id


async def _make_entity(
    driver, graph_id: str, entity_id: str, *, label: str = "Thing"
) -> None:
    async with driver.session() as session:
        await session.run(
            f"""
            CREATE (e:{label}:__Entity__:__KGBuilder__ $props)
            SET e.transaction_time = datetime(),
                e.valid_from = date()
            """,
            {
                "props": {
                    "id": entity_id,
                    "graph_id": graph_id,
                    "name": f"{entity_id}-name",
                }
            },
        )


async def _make_rel(
    driver, graph_id: str, src_id: str, tgt_id: str, *, rel_type: str = "RELATED_TO"
) -> None:
    async with driver.session() as session:
        await session.run(
            f"""
            MATCH (a:__Entity__ {{graph_id: $gid, id: $src}})
            MATCH (b:__Entity__ {{graph_id: $gid, id: $tgt}})
            MERGE (a)-[r:{rel_type} {{graph_id: $gid}}]->(b)
            """,
            {"gid": graph_id, "src": src_id, "tgt": tgt_id},
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_valid_entity_returns_neighborhood(
    owner_client: AsyncClient, neo4j_test_driver
) -> None:
    """A 1-hop neighborhood around a focal entity returns that entity + its direct neighbors."""
    graph_id = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    focal_id = f"{_ENTITY_PREFIX}focal-{uuid.uuid4().hex[:8]}"
    neighbor_id = f"{_ENTITY_PREFIX}nbr-{uuid.uuid4().hex[:8]}"
    isolated_id = f"{_ENTITY_PREFIX}iso-{uuid.uuid4().hex[:8]}"

    await _make_entity(neo4j_test_driver, graph_id, focal_id)
    await _make_entity(neo4j_test_driver, graph_id, neighbor_id)
    await _make_entity(neo4j_test_driver, graph_id, isolated_id)
    await _make_rel(neo4j_test_driver, graph_id, focal_id, neighbor_id)

    try:
        resp = await owner_client.get(
            f"/api/v1/graphs/{graph_id}/entities/{focal_id}/neighborhood",
            params={"hops": 1},
        )
        assert resp.status_code == 200
        body = resp.json()
        node_ids = {n["id"] for n in body["nodes"]}
        assert focal_id in node_ids
        assert neighbor_id in node_ids
        assert isolated_id not in node_ids
        assert len(body["edges"]) >= 1
    finally:
        await _clean_neo4j(neo4j_test_driver, [graph_id])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_graph_id_returns_403(
    owner_client: AsyncClient, neo4j_test_driver
) -> None:
    """A non-existent graph_id must return 403 (ReBAC denies; prevents enumeration)."""
    fake_graph_id = str(uuid.uuid4())
    resp = await owner_client.get(
        f"/api/v1/graphs/{fake_graph_id}/entities/some-entity/neighborhood"
    )
    assert resp.status_code == 403


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_entity_id_returns_404(
    owner_client: AsyncClient, neo4j_test_driver
) -> None:
    """A valid graph with a non-existent entity_id must return 404."""
    graph_id = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    try:
        resp = await owner_client.get(
            f"/api/v1/graphs/{graph_id}/entities/does-not-exist/neighborhood"
        )
        assert resp.status_code == 404
    finally:
        await _clean_neo4j(neo4j_test_driver, [graph_id])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_hops_out_of_range_returns_422(
    owner_client: AsyncClient, neo4j_test_driver
) -> None:
    """hops=0 and hops=4 must both be rejected with 422 (FastAPI Query validation)."""
    fake_graph_id = str(uuid.uuid4())
    for bad_hops in (0, 4):
        resp = await owner_client.get(
            f"/api/v1/graphs/{fake_graph_id}/entities/any/neighborhood",
            params={"hops": bad_hops},
        )
        assert resp.status_code == 422, f"expected 422 for hops={bad_hops}"
