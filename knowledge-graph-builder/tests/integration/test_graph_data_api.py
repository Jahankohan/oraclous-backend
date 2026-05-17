"""Integration tests for the Graph Explorer data API (TASK-210).

Endpoint: GET /api/v1/graphs/{graph_id}/graph-data

Covers:
  1. basic node/edge return for a small graph
  2. `limit` cap + `truncated:true` when the match set exceeds the cap
  3. `node_type` label filter
  4. `min_degree` filter
  5. `community_id` filter (via :__Community__ + IN_COMMUNITY edges)
  6. induced edges only — no dangling edge to a node outside the set
  7. a caller with no `read` role gets 403
  8. node `properties` excludes `embedding` (and graph_id / id)
  9. `degree` and `community_id` are populated on returned nodes

These hit the real FastAPI app and Neo4j. Auth is mocked via
``dependency_overrides`` (per-client ``X-Test-User`` header), mirroring
``test_linked_to_api.py``. Each test creates a :Graph:__Platform__ node,
bootstraps the 5 ReBAC :Role nodes, grants the test user a role, and ingests
:__Entity__ nodes + relationships directly via Cypher. All test entities use
the ``__gdtest__`` id prefix for teardown.
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

OWNER_USER_ID = str(uuid.uuid4())  # owner role on the test graph
STRANGER_USER_ID = str(uuid.uuid4())  # no role on the test graph

_TEST_USER_IDS = [OWNER_USER_ID, STRANGER_USER_ID]
_ENTITY_PREFIX = "__gdtest__"


# ---------------------------------------------------------------------------
# Auth override / clients
# ---------------------------------------------------------------------------


async def _mock_user_id(request: Request) -> str:
    """Resolve the test user from the per-client ``X-Test-User`` header."""
    return request.headers.get("x-test-user", OWNER_USER_ID)


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
async def owner_client():
    """Async HTTP client authenticated as OWNER_USER_ID."""
    client = _client_for(OWNER_USER_ID)
    async with client as c:
        yield c
    from app.main import app

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def stranger_client():
    """Async HTTP client authenticated as STRANGER_USER_ID (no graph role)."""
    client = _client_for(STRANGER_USER_ID)
    async with client as c:
        yield c
    from app.main import app

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Neo4j fixture helpers
# ---------------------------------------------------------------------------


async def _clean_neo4j(driver, graph_ids: list[str]) -> None:
    """Remove test :__Entity__ / :__Community__ nodes, :Graph nodes, :Role
    nodes and :User nodes (and all their edges)."""
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
    """Create a :Graph:__Platform__ node and bootstrap its ReBAC roles."""
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
                "name": f"__gdtest__ graph {graph_id[:8]}",
                "owner_user_id": owner_user_id,
            },
        )
    await rebac_service.bootstrap_graph_roles(driver, graph_id, owner_user_id)
    return graph_id


async def _grant(driver, graph_id: str, user_id: str, role: str) -> None:
    """Grant *user_id* the named ReBAC role on *graph_id*."""
    await rebac_service.grant_role(
        driver,
        graph_id=graph_id,
        target_user_id=user_id,
        role_name=role,
        granted_by=OWNER_USER_ID,
    )


async def _make_entity(
    driver,
    graph_id: str,
    entity_id: str,
    *,
    label: str = "Thing",
    entity_type: str | None = None,
    extra: dict | None = None,
) -> None:
    """Create a :__Entity__ node with one domain label, keyed by (graph_id, id)."""
    props: dict = {
        "id": entity_id,
        "graph_id": graph_id,
        "name": f"{entity_id}-name",
    }
    if entity_type is not None:
        props["type"] = entity_type
    if extra:
        props.update(extra)
    async with driver.session() as session:
        # Real extracted entities also carry :__KGBuilder__ (the neo4j_graphrag
        # marker). Include it so the suite exercises realistic nodes — the
        # graph-data node filter must NOT exclude :__KGBuilder__.
        await session.run(
            f"CREATE (e:{label}:__Entity__:__KGBuilder__ $props)",
            {"props": props},
        )


async def _make_rel(
    driver,
    graph_id: str,
    src_id: str,
    tgt_id: str,
    *,
    rel_type: str = "RELATED_TO",
    count: int | None = None,
) -> None:
    """Create an entity-to-entity relationship carrying graph_id."""
    rel_props: dict = {"graph_id": graph_id}
    if count is not None:
        rel_props["count"] = count
    async with driver.session() as session:
        await session.run(
            f"""
            MATCH (a:__Entity__ {{graph_id: $gid, id: $src}})
            MATCH (b:__Entity__ {{graph_id: $gid, id: $tgt}})
            MERGE (a)-[r:{rel_type} {{graph_id: $gid}}]->(b)
            SET r += $props
            """,
            {"gid": graph_id, "src": src_id, "tgt": tgt_id, "props": rel_props},
        )


async def _make_community(
    driver, graph_id: str, community_id: str, member_ids: list[str], level: int = 0
) -> None:
    """Create an active :__Community__ and IN_COMMUNITY edges for its members."""
    async with driver.session() as session:
        await session.run(
            """
            CREATE (c:__Community__ {
                id: $cid, graph_id: $gid, status: 'active', level: $level
            })
            """,
            {"cid": community_id, "gid": graph_id, "level": level},
        )
        for mid in member_ids:
            await session.run(
                """
                MATCH (e:__Entity__ {graph_id: $gid, id: $mid})
                MATCH (c:__Community__ {graph_id: $gid, id: $cid})
                MERGE (e)-[:IN_COMMUNITY {graph_id: $gid, level: $level}]->(c)
                """,
                {"gid": graph_id, "mid": mid, "cid": community_id, "level": level},
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_basic_nodes_and_edges(
    owner_client: AsyncClient, neo4j_test_driver
) -> None:
    """A small graph returns its entity nodes and entity-to-entity edges."""
    await _clean_neo4j(neo4j_test_driver, [])
    gid = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    try:
        a, b, c = f"{_ENTITY_PREFIX}a", f"{_ENTITY_PREFIX}b", f"{_ENTITY_PREFIX}c"
        await _make_entity(neo4j_test_driver, gid, a, label="Person")
        await _make_entity(neo4j_test_driver, gid, b, label="Company")
        await _make_entity(neo4j_test_driver, gid, c, label="Person")
        await _make_rel(neo4j_test_driver, gid, a, b)
        await _make_rel(neo4j_test_driver, gid, b, c)

        resp = await owner_client.get(f"/api/v1/graphs/{gid}/graph-data")
        assert resp.status_code == 200, resp.text
        body = resp.json()

        ids = {n["id"] for n in body["nodes"]}
        assert ids == {a, b, c}
        assert body["truncated"] is False
        assert len(body["edges"]) == 2
        for edge in body["edges"]:
            assert edge["source"] in ids and edge["target"] in ids
            assert edge["type"] == "RELATED_TO"
            assert isinstance(edge["weight"], float)
        # label is the domain label, not a reserved marker
        labels = {n["id"]: n["label"] for n in body["nodes"]}
        assert labels[a] == "Person"
        assert labels[b] == "Company"
    finally:
        await _clean_neo4j(neo4j_test_driver, [gid])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_limit_cap_and_truncated(
    owner_client: AsyncClient, neo4j_test_driver
) -> None:
    """`limit` caps the node count and sets truncated:true when exceeded."""
    await _clean_neo4j(neo4j_test_driver, [])
    gid = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    try:
        for i in range(5):
            await _make_entity(neo4j_test_driver, gid, f"{_ENTITY_PREFIX}{i}")

        resp = await owner_client.get(f"/api/v1/graphs/{gid}/graph-data?limit=3")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert len(body["nodes"]) == 3
        assert body["truncated"] is True

        # Asking for everything → not truncated.
        resp_all = await owner_client.get(f"/api/v1/graphs/{gid}/graph-data?limit=100")
        body_all = resp_all.json()
        assert len(body_all["nodes"]) == 5
        assert body_all["truncated"] is False
    finally:
        await _clean_neo4j(neo4j_test_driver, [gid])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_node_type_filter(owner_client: AsyncClient, neo4j_test_driver) -> None:
    """`node_type` keeps only nodes carrying that label."""
    await _clean_neo4j(neo4j_test_driver, [])
    gid = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    try:
        p1 = f"{_ENTITY_PREFIX}p1"
        p2 = f"{_ENTITY_PREFIX}p2"
        co = f"{_ENTITY_PREFIX}co"
        await _make_entity(neo4j_test_driver, gid, p1, label="Person")
        await _make_entity(neo4j_test_driver, gid, p2, label="Person")
        await _make_entity(neo4j_test_driver, gid, co, label="Company")

        resp = await owner_client.get(
            f"/api/v1/graphs/{gid}/graph-data?node_type=Person"
        )
        assert resp.status_code == 200, resp.text
        ids = {n["id"] for n in resp.json()["nodes"]}
        assert ids == {p1, p2}
    finally:
        await _clean_neo4j(neo4j_test_driver, [gid])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_min_degree_filter(owner_client: AsyncClient, neo4j_test_driver) -> None:
    """`min_degree` keeps only nodes whose entity-to-entity degree is high."""
    await _clean_neo4j(neo4j_test_driver, [])
    gid = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    try:
        # hub connects to a, b, c (degree 3). a/b/c each have degree 1.
        # iso has degree 0.
        hub = f"{_ENTITY_PREFIX}hub"
        a, b, c = f"{_ENTITY_PREFIX}a", f"{_ENTITY_PREFIX}b", f"{_ENTITY_PREFIX}c"
        iso = f"{_ENTITY_PREFIX}iso"
        for n in (hub, a, b, c, iso):
            await _make_entity(neo4j_test_driver, gid, n)
        await _make_rel(neo4j_test_driver, gid, hub, a)
        await _make_rel(neo4j_test_driver, gid, hub, b)
        await _make_rel(neo4j_test_driver, gid, hub, c)

        resp = await owner_client.get(f"/api/v1/graphs/{gid}/graph-data?min_degree=2")
        assert resp.status_code == 200, resp.text
        nodes = resp.json()["nodes"]
        assert {n["id"] for n in nodes} == {hub}
        assert nodes[0]["degree"] == 3
    finally:
        await _clean_neo4j(neo4j_test_driver, [gid])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_community_filter_and_population(
    owner_client: AsyncClient, neo4j_test_driver
) -> None:
    """`community_id` filters to members; community_id/degree are populated."""
    await _clean_neo4j(neo4j_test_driver, [])
    gid = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    try:
        m1, m2 = f"{_ENTITY_PREFIX}m1", f"{_ENTITY_PREFIX}m2"
        out = f"{_ENTITY_PREFIX}out"
        for n in (m1, m2, out):
            await _make_entity(neo4j_test_driver, gid, n)
        await _make_rel(neo4j_test_driver, gid, m1, m2)
        cid = f"{_ENTITY_PREFIX}comm-1"
        await _make_community(neo4j_test_driver, gid, cid, [m1, m2])

        resp = await owner_client.get(
            f"/api/v1/graphs/{gid}/graph-data?community_id={cid}"
        )
        assert resp.status_code == 200, resp.text
        nodes = resp.json()["nodes"]
        assert {n["id"] for n in nodes} == {m1, m2}
        for n in nodes:
            assert n["community_id"] == cid
            assert n["degree"] == 1
    finally:
        await _clean_neo4j(neo4j_test_driver, [gid])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_edges_are_induced(owner_client: AsyncClient, neo4j_test_driver) -> None:
    """No edge dangles to a node outside the returned (capped) node set."""
    await _clean_neo4j(neo4j_test_driver, [])
    gid = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    try:
        # 5 nodes in a chain n0-n1-n2-n3-n4. Cap to 3 → only 3 nodes returned,
        # so any edge to a dropped node must NOT appear.
        ids = [f"{_ENTITY_PREFIX}n{i}" for i in range(5)]
        for n in ids:
            await _make_entity(neo4j_test_driver, gid, n)
        for i in range(4):
            await _make_rel(neo4j_test_driver, gid, ids[i], ids[i + 1])

        resp = await owner_client.get(f"/api/v1/graphs/{gid}/graph-data?limit=3")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        returned = {n["id"] for n in body["nodes"]}
        assert len(returned) == 3
        for edge in body["edges"]:
            assert edge["source"] in returned
            assert edge["target"] in returned
    finally:
        await _clean_neo4j(neo4j_test_driver, [gid])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_properties_exclude_embedding(
    owner_client: AsyncClient, neo4j_test_driver
) -> None:
    """Node `properties` drops embedding / graph_id / id but keeps the rest."""
    await _clean_neo4j(neo4j_test_driver, [])
    gid = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    try:
        eid = f"{_ENTITY_PREFIX}e1"
        await _make_entity(
            neo4j_test_driver,
            gid,
            eid,
            label="Place",
            entity_type="City",
            extra={
                "embedding": [0.1, 0.2, 0.3],
                "description": "a test city",
                "lat": 51.5,
                "lng": -0.12,
            },
        )

        resp = await owner_client.get(f"/api/v1/graphs/{gid}/graph-data")
        assert resp.status_code == 200, resp.text
        node = resp.json()["nodes"][0]
        props = node["properties"]
        assert "embedding" not in props
        assert "graph_id" not in props
        assert "id" not in props
        # user-ingested fields survive
        assert props["description"] == "a test city"
        assert props["lat"] == 51.5
        assert props["lng"] == -0.12
        assert node["type"] == "City"
    finally:
        await _clean_neo4j(neo4j_test_driver, [gid])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_read_caller_gets_403(
    stranger_client: AsyncClient, neo4j_test_driver
) -> None:
    """A caller with no role on the graph is denied with 403."""
    await _clean_neo4j(neo4j_test_driver, [])
    gid = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    try:
        await _make_entity(neo4j_test_driver, gid, f"{_ENTITY_PREFIX}x")
        resp = await stranger_client.get(f"/api/v1/graphs/{gid}/graph-data")
        assert resp.status_code == 403, resp.text
    finally:
        await _clean_neo4j(neo4j_test_driver, [gid])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_entities_without_id_property(
    owner_client: AsyncClient, neo4j_test_driver
) -> None:
    """Extraction-path entities carry no `id` property — graph-data must still
    return them (elementId fallback) with their relationships and weights."""
    await _clean_neo4j(neo4j_test_driver, [])
    gid = await _make_graph(neo4j_test_driver, OWNER_USER_ID)
    try:
        # No `id` property — exactly as neo4j_graphrag-extracted entities are.
        async with neo4j_test_driver.session() as session:
            await session.run(
                """
                CREATE (a:Organization:__Entity__:__KGBuilder__
                          {graph_id: $gid, name: '__gdtest__noid-a'})
                CREATE (b:Organization:__Entity__:__KGBuilder__
                          {graph_id: $gid, name: '__gdtest__noid-b'})
                MERGE (a)-[:CITES {graph_id: $gid, weight: 4}]->(b)
                """,
                {"gid": gid},
            )

        resp = await owner_client.get(f"/api/v1/graphs/{gid}/graph-data")
        assert resp.status_code == 200, resp.text
        body = resp.json()

        assert len(body["nodes"]) == 2
        node_ids = {n["id"] for n in body["nodes"]}
        # every node has a real, non-empty id (elementId fallback)
        assert all(nid for nid in node_ids)
        names = {n["properties"]["name"] for n in body["nodes"]}
        assert names == {"__gdtest__noid-a", "__gdtest__noid-b"}

        # the relationship is returned and its endpoints match the node ids
        assert len(body["edges"]) == 1
        edge = body["edges"][0]
        assert edge["source"] in node_ids and edge["target"] in node_ids
        assert edge["weight"] == 4.0  # r.weight, via the coalesce
    finally:
        await _clean_neo4j(neo4j_test_driver, [gid])
