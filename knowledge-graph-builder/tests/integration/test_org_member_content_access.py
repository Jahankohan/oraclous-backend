"""Integration tests for member read access to org content listings (TASK-209).

Extends the TASK-208 member-readable pattern from ``GET /organizations/{id}``
to the three org content-listing routes:

  - GET /organizations/{org_id}/graphs
  - GET /organizations/{org_id}/members
  - GET /organizations/{org_id}/agents

Covers:
  1. A non-owner member granted a ReBAC role on one of the org's two subgraphs
     sees only that subgraph on .../graphs (not both).
  2. That member sees the full roster on .../members.
  3. That member sees only agents touching the accessible subgraph on
     .../agents.
  4. The owner still sees everything on all three routes.
  5. A user with no BELONGS_TO edge gets 404 on all three routes.
  6. A mutating route (POST .../members) still 404s for the non-owner member.

These tests hit the real FastAPI app, PostgreSQL, and Neo4j. Auth is mocked via
dependency_overrides (per-client ``X-Test-User`` header), mirroring
``test_org_members_api.py`` and ``test_org_agents_api.py``. Test :Graph nodes
are wired ``(:Organization)-[:OWNS]->(:Graph)`` directly in Neo4j and their
5 ReBAC :Role nodes are bootstrapped via ``rebac_service.bootstrap_graph_roles``
(``grant_role`` MATCHes pre-existing :Role nodes, so the bootstrap is required).
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
from app.services.agent_service import AgentService
from app.services.rebac_service import rebac_service

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OWNER_USER_ID = str(uuid.uuid4())
MEMBER_USER_ID = str(uuid.uuid4())
STRANGER_USER_ID = str(uuid.uuid4())

_TEST_USER_IDS = [OWNER_USER_ID, MEMBER_USER_ID, STRANGER_USER_ID]

# Name prefix shared by all test :Organization / :Graph / :Agent nodes.
_PREFIX = "__memcontenttest__"


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
async def member_client():
    """Async HTTP client authenticated as MEMBER_USER_ID."""
    client = _client_for(MEMBER_USER_ID)
    async with client as c:
        yield c
    from app.main import app

    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def stranger_client():
    """Async HTTP client authenticated as a user with no org membership."""
    client = _client_for(STRANGER_USER_ID)
    async with client as c:
        yield c
    from app.main import app

    app.dependency_overrides.clear()


async def _delete_test_orgs(db: AsyncSession) -> None:
    """Remove organizations created by the test users."""
    from app.models.organization import Organization

    await db.execute(
        delete(Organization).where(Organization.owner_user_id.in_(_TEST_USER_IDS))
    )
    await db.commit()


async def _delete_neo4j_state(driver) -> None:
    """Remove :Organization / :User / :Graph / :Role / :Agent test nodes."""
    async with driver.session() as session:
        await session.run(
            "MATCH (u:User) WHERE u.user_id IN $ids DETACH DELETE u",
            {"ids": _TEST_USER_IDS},
        )
        await session.run(
            "MATCH (o:Organization) WHERE o.name STARTS WITH $p DETACH DELETE o",
            {"p": _PREFIX},
        )
        await session.run(
            "MATCH (g:Graph) WHERE g.name STARTS WITH $p DETACH DELETE g",
            {"p": _PREFIX},
        )
        await session.run(
            "MATCH (a:Agent) WHERE a.name STARTS WITH $p DETACH DELETE a",
            {"p": _PREFIX},
        )


async def _create_org(client: AsyncClient, name: str) -> str:
    """Create an organization via the API and return its org_id."""
    resp = await client.post(
        "/api/v1/organizations",
        json={"name": name, "description": "member-content-test"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def _attach_subgraph(driver, org_id: str, owner_user_id: str) -> str:
    """Create a :Graph OWNED by *org_id* and bootstrap its ReBAC roles.

    Returns the new graph_id. The graph is named with the ``_PREFIX`` so
    teardown can find it.
    """
    graph_id = str(uuid.uuid4())
    async with driver.session() as session:
        await session.run(
            """
            MATCH (o:Organization:__Platform__ {org_id: $org_id})
            CREATE (g:Graph:__Platform__ {
                graph_id: $graph_id,
                name: $name,
                description: '',
                org_id: $org_id,
                user_id: $owner_user_id,
                status: 'active',
                node_count: 0,
                relationship_count: 0,
                federatable: false,
                created_at: datetime(),
                updated_at: datetime()
            })
            MERGE (o)-[:OWNS]->(g)
            """,
            {
                "org_id": org_id,
                "graph_id": graph_id,
                "name": f"{_PREFIX} graph {graph_id[:8]}",
                "owner_user_id": owner_user_id,
            },
        )
    # grant_role MATCHes pre-existing :Role nodes — bootstrap them first.
    await rebac_service.bootstrap_graph_roles(driver, graph_id, owner_user_id)
    return graph_id


async def _create_agent(driver, graph_id: str, user_id: str) -> str:
    """Create an :Agent on *graph_id* via AgentService. Returns the agent_id."""
    svc = AgentService(driver)
    data = AgentCreate(
        name=f"{_PREFIX} agent {uuid.uuid4().hex[:8]}",
        description="org content access test agent",
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
async def test_member_graphs_scoped_to_rebac_access(
    owner_client: AsyncClient,
    member_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """A non-owner member sees only the org subgraphs they have a HAS_ROLE on."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, f"{_PREFIX} GraphsScope")
    try:
        graph_a = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        graph_b = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)

        # Add the member, granted a ReBAC role on graph_a only.
        add = await owner_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={
                "user_id": MEMBER_USER_ID,
                "org_role": "member",
                "subgraph_grants": {"role": "viewer", "graph_ids": [graph_a]},
            },
        )
        assert add.status_code == 201, add.text

        # Member sees only graph_a.
        resp = await member_client.get(f"/api/v1/organizations/{org_id}/graphs")
        assert resp.status_code == 200, resp.text
        ids = {g["id"] for g in resp.json()}
        assert ids == {graph_a}, f"expected only graph_a, got {ids}"

        # Owner sees both.
        resp = await owner_client.get(f"/api/v1/organizations/{org_id}/graphs")
        assert resp.status_code == 200, resp.text
        owner_ids = {g["id"] for g in resp.json()}
        assert {graph_a, graph_b}.issubset(owner_ids)
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_member_sees_full_roster(
    owner_client: AsyncClient,
    member_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """A non-owner member sees the full org roster on GET .../members."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, f"{_PREFIX} Roster")
    try:
        add = await owner_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={"user_id": MEMBER_USER_ID, "org_role": "member"},
        )
        assert add.status_code == 201, add.text

        resp = await member_client.get(f"/api/v1/organizations/{org_id}/members")
        assert resp.status_code == 200, resp.text
        roster = {m["user_id"] for m in resp.json()}
        # Full roster: owner + the member.
        assert OWNER_USER_ID in roster
        assert MEMBER_USER_ID in roster
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_member_agents_scoped_to_rebac_access(
    owner_client: AsyncClient,
    member_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """A non-owner member sees only agents touching an accessible subgraph."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, f"{_PREFIX} AgentsScope")
    try:
        graph_a = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        graph_b = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)

        # agent_a has home graph graph_a (accessible to the member).
        agent_a = await _create_agent(neo4j_test_driver, graph_a, OWNER_USER_ID)
        # agent_b has home graph graph_b (NOT accessible to the member).
        agent_b = await _create_agent(neo4j_test_driver, graph_b, OWNER_USER_ID)
        # agent_c has home graph graph_b but a CAN_ACCESS grant to graph_a.
        agent_c = await _create_agent(neo4j_test_driver, graph_b, OWNER_USER_ID)
        grant = await owner_client.post(
            f"/api/v1/organizations/{org_id}/agents/{agent_c}/subgraph-grants",
            json={"level": "reader", "graph_ids": [graph_a]},
        )
        assert grant.status_code == 200, grant.text

        # Add the member, granted a ReBAC role on graph_a only.
        add = await owner_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={
                "user_id": MEMBER_USER_ID,
                "org_role": "member",
                "subgraph_grants": {"role": "viewer", "graph_ids": [graph_a]},
            },
        )
        assert add.status_code == 201, add.text

        # Member sees agent_a (home graph) and agent_c (CAN_ACCESS), not agent_b.
        resp = await member_client.get(f"/api/v1/organizations/{org_id}/agents")
        assert resp.status_code == 200, resp.text
        member_agents = {a["agent_id"] for a in resp.json()}
        assert member_agents == {
            agent_a,
            agent_c,
        }, f"expected {{agent_a, agent_c}}, got {member_agents}"

        # Owner sees all three.
        resp = await owner_client.get(f"/api/v1/organizations/{org_id}/agents")
        assert resp.status_code == 200, resp.text
        owner_agents = {a["agent_id"] for a in resp.json()}
        assert {agent_a, agent_b, agent_c}.issubset(owner_agents)
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_member_gets_404_on_all_listings(
    owner_client: AsyncClient,
    stranger_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """A user with no BELONGS_TO edge gets 404 on all three content listings."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, f"{_PREFIX} NonMember")
    try:
        await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)

        for route in ("graphs", "members", "agents"):
            resp = await stranger_client.get(f"/api/v1/organizations/{org_id}/{route}")
            assert resp.status_code == 404, f"{route}: {resp.text}"
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mutating_route_still_owner_only(
    owner_client: AsyncClient,
    member_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """A non-owner member still gets 404 on a mutating route (POST .../members)."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, f"{_PREFIX} MutatingGuard")
    try:
        add = await owner_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={"user_id": MEMBER_USER_ID, "org_role": "member"},
        )
        assert add.status_code == 201, add.text

        # The member can read the roster (TASK-209)...
        read = await member_client.get(f"/api/v1/organizations/{org_id}/members")
        assert read.status_code == 200, read.text

        # ...but cannot add a member — that route is still owner-only.
        resp = await member_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={"user_id": str(uuid.uuid4()), "org_role": "member"},
        )
        assert resp.status_code == 404, resp.text
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)
