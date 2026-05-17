"""Integration tests for the Organization Agent registry (TASK-203 Part 2).

Covers:
  1. ``AgentService.create_agent`` sets ``org_id`` on the :Agent node and wires
     ``(:Organization)-[:HAS_AGENT]->(:Agent)``.
  2. POST .../agents/{aid}/subgraph-grants — grant on specific subgraphs.
  3. POST .../agents/{aid}/subgraph-grants — grant on "all" subgraphs.
  4. ``list_agent_grants`` reflects the grants made.
  5. ``check_agent_graph_permission`` — True for the home graph and for granted
     graphs at/above level, False otherwise (fail-closed).
  6. DELETE .../agents/{aid}/subgraph-grants/{gid} — revoke a grant.
  7. GET /organizations/{org_id}/agents — lists the org's agents.
  8. A non-owner gets 404 on the org-agent routes.

These tests hit the real FastAPI app, PostgreSQL, and Neo4j. Auth is mocked via
dependency_overrides (per-client ``X-Test-User`` header), mirroring
``test_org_members_api.py``.

The :Agent is created by calling ``AgentService.create_agent`` directly against
the test Neo4j driver — that exercises the new org_id / HAS_AGENT wiring without
needing the agents-endpoint plumbing. The org-agent grant/registry endpoints are
exercised through the HTTP API.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from fastapi import Request
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.schemas.agent_schemas import AgentCreate, RetrieverConfig
from app.services import org_agent_service
from app.services.agent_service import AgentService

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OWNER_USER_ID = str(uuid.uuid4())
OTHER_USER_ID = str(uuid.uuid4())

_TEST_USER_IDS = [OWNER_USER_ID, OTHER_USER_ID]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_session():
    """Task-scoped async database session for test setup/teardown."""
    engine = create_async_engine(settings.POSTGRES_URL, poolclass=NullPool, future=True)
    session_maker = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_maker() as session:
        yield session
    await engine.dispose()


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
async def other_client():
    """Async HTTP client authenticated as a different (non-owner) user."""
    client = _client_for(OTHER_USER_ID)
    async with client as c:
        yield c
    from app.main import app

    app.dependency_overrides.clear()


async def _delete_test_orgs(db: AsyncSession) -> None:
    """Remove organizations created by the test users."""
    from app.models.organization import Organization

    await db.execute(
        delete(Organization).where(
            Organization.owner_user_id.in_([OWNER_USER_ID, OTHER_USER_ID])
        )
    )
    await db.commit()


async def _delete_neo4j_state(driver) -> None:
    """Remove :Organization / :User / :Graph / :Agent test nodes from Neo4j."""
    async with driver.session() as session:
        await session.run(
            "MATCH (u:User) WHERE u.user_id IN $ids DETACH DELETE u",
            {"ids": _TEST_USER_IDS},
        )
        await session.run(
            """
            MATCH (o:Organization)
            WHERE o.name STARTS WITH '__agenttest__'
            DETACH DELETE o
            """,
        )
        await session.run(
            """
            MATCH (g:Graph) WHERE g.name STARTS WITH '__agenttest__'
            DETACH DELETE g
            """,
        )
        await session.run(
            """
            MATCH (a:Agent) WHERE a.name STARTS WITH '__agenttest__'
            DETACH DELETE a
            """,
        )


async def _create_org(client: AsyncClient, name: str) -> str:
    """Create an organization via the API and return its org_id."""
    resp = await client.post(
        "/api/v1/organizations",
        json={"name": name, "description": "agent-registry-test"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def _attach_subgraph(driver, org_id: str, owner_user_id: str) -> str:
    """Create a :Graph node OWNED by *org_id*. Returns the new graph_id.

    The graph is named with the ``__agenttest__`` prefix so teardown finds it.
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
                "name": f"__agenttest__ graph {graph_id[:8]}",
                "owner_user_id": owner_user_id,
            },
        )
    return graph_id


async def _create_agent(driver, graph_id: str, user_id: str) -> str:
    """Create an :Agent on *graph_id* via AgentService. Returns the agent_id."""
    svc = AgentService(driver)
    data = AgentCreate(
        name=f"__agenttest__ agent {uuid.uuid4().hex[:8]}",
        description="org registry test agent",
        system_prompt="You are a test agent.",
        retriever=RetrieverConfig(),
        tools=["graph_search"],
    )
    return await svc.create_agent(graph_id, user_id, data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_agent_sets_org_id_and_has_agent_edge(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """create_agent stamps org_id and wires (:Organization)-[:HAS_AGENT]->(:Agent)."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__agenttest__ CreateWiring")
    try:
        home_graph = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        agent_id = await _create_agent(neo4j_test_driver, home_graph, OWNER_USER_ID)

        # org_id property is stamped on the :Agent node.
        async with neo4j_test_driver.session() as session:
            result = await session.run(
                "MATCH (a:Agent:__Platform__ {agent_id: $aid}) "
                "RETURN a.org_id AS org_id, a.graph_id AS graph_id",
                {"aid": agent_id},
            )
            record = await result.single()
        assert record is not None
        assert record["org_id"] == org_id
        assert record["graph_id"] == home_graph

        # HAS_AGENT edge exists.
        async with neo4j_test_driver.session() as session:
            result = await session.run(
                """
                MATCH (:Organization:__Platform__ {org_id: $org_id})
                      -[:HAS_AGENT]->(a:Agent:__Platform__ {agent_id: $aid})
                RETURN count(a) AS c
                """,
                {"org_id": org_id, "aid": agent_id},
            )
            record = await result.single()
        assert record["c"] == 1
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_grant_agent_specific_subgraphs(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """Grant an agent CAN_ACCESS on a specific subgraph; list_agent_grants sees it."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__agenttest__ GrantSpecific")
    try:
        home_graph = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        graph_b = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        graph_c = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        agent_id = await _create_agent(neo4j_test_driver, home_graph, OWNER_USER_ID)

        grant = await owner_client.post(
            f"/api/v1/organizations/{org_id}/agents/{agent_id}/subgraph-grants",
            json={"level": "writer", "graph_ids": [graph_b]},
        )
        assert grant.status_code == 200, grant.text
        granted = grant.json()
        assert len(granted) == 1
        assert granted[0]["graph_id"] == graph_b
        assert granted[0]["level"] == "writer"

        # list_agent_grants reflects exactly the granted subgraph.
        grants = await org_agent_service.list_agent_grants(
            neo4j_test_driver, org_id, agent_id
        )
        assert {g["graph_id"] for g in grants} == {graph_b}
        assert grants[0]["level"] == "writer"

        # GET endpoint agrees.
        listed = await owner_client.get(
            f"/api/v1/organizations/{org_id}/agents/{agent_id}/subgraph-grants"
        )
        assert listed.status_code == 200, listed.text
        assert {g["graph_id"] for g in listed.json()} == {graph_b}

        # graph_c was never granted.
        assert graph_c not in {g["graph_id"] for g in grants}
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_grant_agent_all_subgraphs(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """Grant an agent 'all' subgraphs; every owned graph gets a CAN_ACCESS edge."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__agenttest__ GrantAll")
    try:
        home_graph = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        graph_b = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        agent_id = await _create_agent(neo4j_test_driver, home_graph, OWNER_USER_ID)

        grant = await owner_client.post(
            f"/api/v1/organizations/{org_id}/agents/{agent_id}/subgraph-grants",
            json={"level": "reader", "graph_ids": "all"},
        )
        assert grant.status_code == 200, grant.text
        granted = {g["graph_id"] for g in grant.json()}
        # "all" resolves to every owned graph — home_graph included.
        assert granted == {home_graph, graph_b}

        grants = await org_agent_service.list_agent_grants(
            neo4j_test_driver, org_id, agent_id
        )
        assert {g["graph_id"] for g in grants} == {home_graph, graph_b}
        assert all(g["level"] == "reader" for g in grants)
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_check_agent_graph_permission_fail_closed(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """check_agent_graph_permission: home graph + granted graphs allowed, else denied."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__agenttest__ PermCheck")
    try:
        home_graph = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        granted_graph = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        ungranted_graph = await _attach_subgraph(
            neo4j_test_driver, org_id, OWNER_USER_ID
        )
        agent_id = await _create_agent(neo4j_test_driver, home_graph, OWNER_USER_ID)

        # Grant 'writer' on granted_graph.
        await owner_client.post(
            f"/api/v1/organizations/{org_id}/agents/{agent_id}/subgraph-grants",
            json={"level": "writer", "graph_ids": [granted_graph]},
        )

        check = org_agent_service.check_agent_graph_permission

        # Home graph: always allowed, at every level.
        assert await check(neo4j_test_driver, agent_id, home_graph, "reader") is True
        assert await check(neo4j_test_driver, agent_id, home_graph, "admin") is True

        # Granted graph at 'writer': reader/writer allowed, admin denied.
        assert await check(neo4j_test_driver, agent_id, granted_graph, "reader") is True
        assert await check(neo4j_test_driver, agent_id, granted_graph, "writer") is True
        assert await check(neo4j_test_driver, agent_id, granted_graph, "admin") is False

        # Ungranted graph: denied at every level (fail-closed).
        assert (
            await check(neo4j_test_driver, agent_id, ungranted_graph, "reader") is False
        )

        # Unknown agent / unknown graph / unknown level: all denied.
        assert (
            await check(neo4j_test_driver, "no-such-agent", home_graph, "reader")
            is False
        )
        assert (
            await check(neo4j_test_driver, agent_id, "no-such-graph", "reader") is False
        )
        assert (
            await check(neo4j_test_driver, agent_id, home_graph, "superuser") is False
        )
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_revoke_agent_grant(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """DELETE removes a grant (204); a second DELETE on the same grant is 404."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__agenttest__ Revoke")
    try:
        home_graph = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        graph_b = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        agent_id = await _create_agent(neo4j_test_driver, home_graph, OWNER_USER_ID)

        await owner_client.post(
            f"/api/v1/organizations/{org_id}/agents/{agent_id}/subgraph-grants",
            json={"level": "reader", "graph_ids": [graph_b]},
        )

        revoke = await owner_client.delete(
            f"/api/v1/organizations/{org_id}/agents/{agent_id}"
            f"/subgraph-grants/{graph_b}"
        )
        assert revoke.status_code == 204, revoke.text

        # Grant is gone.
        grants = await org_agent_service.list_agent_grants(
            neo4j_test_driver, org_id, agent_id
        )
        assert grants == []

        # check_agent_graph_permission now denies graph_b.
        assert (
            await org_agent_service.check_agent_graph_permission(
                neo4j_test_driver, agent_id, graph_b, "reader"
            )
            is False
        )

        # A second revoke of the now-absent grant is 404.
        revoke_again = await owner_client.delete(
            f"/api/v1/organizations/{org_id}/agents/{agent_id}"
            f"/subgraph-grants/{graph_b}"
        )
        assert revoke_again.status_code == 404, revoke_again.text
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_org_agents(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """GET /organizations/{org_id}/agents lists the org's agents."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__agenttest__ ListAgents")
    try:
        home_graph = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        agent_a = await _create_agent(neo4j_test_driver, home_graph, OWNER_USER_ID)
        agent_b = await _create_agent(neo4j_test_driver, home_graph, OWNER_USER_ID)

        listed = await owner_client.get(f"/api/v1/organizations/{org_id}/agents")
        assert listed.status_code == 200, listed.text
        agents = listed.json()
        by_id = {a["agent_id"]: a for a in agents}
        assert agent_a in by_id
        assert agent_b in by_id
        assert by_id[agent_a]["org_id"] == org_id
        assert by_id[agent_a]["graph_id"] == home_graph
        assert by_id[agent_a]["active"] is True
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_owner_gets_404_on_agent_routes(
    owner_client: AsyncClient,
    other_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """A non-owner gets 404 on every org-agent route — existence is masked."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__agenttest__ NonOwner")
    try:
        home_graph = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        agent_id = await _create_agent(neo4j_test_driver, home_graph, OWNER_USER_ID)

        # GET agents
        resp = await other_client.get(f"/api/v1/organizations/{org_id}/agents")
        assert resp.status_code == 404, resp.text

        # POST subgraph-grants
        resp = await other_client.post(
            f"/api/v1/organizations/{org_id}/agents/{agent_id}/subgraph-grants",
            json={"level": "reader", "graph_ids": [home_graph]},
        )
        assert resp.status_code == 404, resp.text

        # GET subgraph-grants
        resp = await other_client.get(
            f"/api/v1/organizations/{org_id}/agents/{agent_id}/subgraph-grants"
        )
        assert resp.status_code == 404, resp.text

        # DELETE subgraph-grant
        resp = await other_client.delete(
            f"/api/v1/organizations/{org_id}/agents/{agent_id}"
            f"/subgraph-grants/{home_graph}"
        )
        assert resp.status_code == 404, resp.text
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)
