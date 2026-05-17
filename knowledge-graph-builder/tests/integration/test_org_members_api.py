"""Integration tests for the Organization Member API (TASK-203 Part 1).

Covers:
  1. POST   /organizations/{id}/members           — adds a member
  2. GET    /organizations/{id}/members           — lists members + org_role
  3. POST  .../members/{uid}/subgraph-grants      — grant on specific graphs
  4. POST  .../members/{uid}/subgraph-grants      — grant on "all" graphs
  5. Granted member appears in rebac_service.list_graph_members
  6. DELETE .../members/{uid}                     — removes a member
  7. Removing the last owner is refused (409)
  8. A non-owner gets 404 on every member route

These tests hit the real FastAPI app, PostgreSQL, and Neo4j. Auth is mocked
via dependency_overrides (per-client ``X-Test-User`` header), mirroring
``test_organizations_api.py``.

Each test that needs subgraphs creates :Graph:__Platform__ nodes wired with
``(:Organization)-[:OWNS]->(:Graph)`` directly in Neo4j, and bootstraps the
5 ReBAC :Role nodes for each test graph via ``rebac_service.bootstrap_graph_roles``
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
from app.services.rebac_service import rebac_service

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OWNER_USER_ID = str(uuid.uuid4())
OTHER_USER_ID = str(uuid.uuid4())
MEMBER_USER_ID = str(uuid.uuid4())

# All test-created user ids — for Neo4j teardown.
_TEST_USER_IDS = [OWNER_USER_ID, OTHER_USER_ID, MEMBER_USER_ID]


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
    """Remove :Organization / :User / :Graph / :Role test nodes from Neo4j."""
    async with driver.session() as session:
        # Test :User nodes (and all their edges).
        await session.run(
            "MATCH (u:User) WHERE u.user_id IN $ids DETACH DELETE u",
            {"ids": _TEST_USER_IDS},
        )
        # :Organization nodes whose owner is a test user (matched via the
        # BELONGS_TO edge that survived only if owner node still existed —
        # so also match orgs by name prefix as a fallback).
        await session.run(
            """
            MATCH (o:Organization)
            WHERE o.name STARTS WITH '__memtest__'
            DETACH DELETE o
            """,
        )
        # Test :Graph + :Role nodes.
        await session.run(
            """
            MATCH (g:Graph) WHERE g.name STARTS WITH '__memtest__'
            DETACH DELETE g
            """,
        )


async def _create_org(client: AsyncClient, name: str) -> str:
    """Create an organization via the API and return its org_id."""
    resp = await client.post(
        "/api/v1/organizations",
        json={"name": name, "description": "member-test"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


async def _attach_subgraph(driver, org_id: str, owner_user_id: str) -> str:
    """Create a :Graph node OWNED by *org_id* and bootstrap its ReBAC roles.

    Returns the new graph_id. The graph is named with the ``__memtest__``
    prefix so teardown can find it.
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
                "name": f"__memtest__ graph {graph_id[:8]}",
                "owner_user_id": owner_user_id,
            },
        )
    # grant_role MATCHes pre-existing :Role nodes — bootstrap them first.
    await rebac_service.bootstrap_graph_roles(driver, graph_id, owner_user_id)
    return graph_id


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_add_and_list_member(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """POST a member, then GET /members shows it with the correct org_role."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__memtest__ Add+List")
    try:
        add = await owner_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={
                "user_id": MEMBER_USER_ID,
                "org_role": "admin",
                "email": "member@example.com",
            },
        )
        assert add.status_code == 201, add.text
        body = add.json()
        assert body["user_id"] == MEMBER_USER_ID
        assert body["org_role"] == "admin"
        assert body["email"] == "member@example.com"
        assert body["since"]
        assert body["subgraph_grants"] == []

        listed = await owner_client.get(f"/api/v1/organizations/{org_id}/members")
        assert listed.status_code == 200, listed.text
        members = listed.json()
        by_user = {m["user_id"]: m for m in members}
        # The owner is auto-added at org creation; the new member is present.
        assert OWNER_USER_ID in by_user
        assert by_user[OWNER_USER_ID]["org_role"] == "owner"
        assert MEMBER_USER_ID in by_user
        assert by_user[MEMBER_USER_ID]["org_role"] == "admin"
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_grant_member_specific_subgraphs(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """Grant a member a ReBAC role on a specific subgraph; ReBAC sees them."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__memtest__ GrantSpecific")
    try:
        graph_a = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        graph_b = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)

        await owner_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={"user_id": MEMBER_USER_ID, "org_role": "member"},
        )

        grant = await owner_client.post(
            f"/api/v1/organizations/{org_id}/members/{MEMBER_USER_ID}/subgraph-grants",
            json={"role": "editor", "graph_ids": [graph_a]},
        )
        assert grant.status_code == 200, grant.text
        granted = grant.json()
        assert granted == [{"graph_id": graph_a, "role": "editor"}]

        # The member is now an active ReBAC member of graph_a, not graph_b.
        members_a = await rebac_service.list_graph_members(neo4j_test_driver, graph_a)
        assert any(
            m["user_id"] == MEMBER_USER_ID and m["role"] == "editor" for m in members_a
        )
        members_b = await rebac_service.list_graph_members(neo4j_test_driver, graph_b)
        assert not any(m["user_id"] == MEMBER_USER_ID for m in members_b)

        # GET /members reflects the grant.
        listed = await owner_client.get(f"/api/v1/organizations/{org_id}/members")
        by_user = {m["user_id"]: m for m in listed.json()}
        assert {"graph_id": graph_a, "role": "editor"} in by_user[MEMBER_USER_ID][
            "subgraph_grants"
        ]
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_grant_member_all_subgraphs(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """Grant a member 'all' subgraphs; ReBAC sees them on every owned graph."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__memtest__ GrantAll")
    try:
        graph_a = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)
        graph_b = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)

        # Add the member with an inline subgraph_grants spec at creation time.
        add = await owner_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={
                "user_id": MEMBER_USER_ID,
                "org_role": "member",
                "subgraph_grants": {"role": "viewer", "graph_ids": "all"},
            },
        )
        assert add.status_code == 201, add.text
        granted = {(g["graph_id"], g["role"]) for g in add.json()["subgraph_grants"]}
        assert granted == {(graph_a, "viewer"), (graph_b, "viewer")}

        for graph_id in (graph_a, graph_b):
            members = await rebac_service.list_graph_members(
                neo4j_test_driver, graph_id
            )
            assert any(
                m["user_id"] == MEMBER_USER_ID and m["role"] == "viewer"
                for m in members
            ), f"member missing from {graph_id}"
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_member(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """DELETE /members/{uid} removes the membership and revokes ReBAC grants."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__memtest__ Remove")
    try:
        graph_a = await _attach_subgraph(neo4j_test_driver, org_id, OWNER_USER_ID)

        await owner_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={
                "user_id": MEMBER_USER_ID,
                "org_role": "member",
                "subgraph_grants": {"role": "editor", "graph_ids": [graph_a]},
            },
        )

        remove = await owner_client.delete(
            f"/api/v1/organizations/{org_id}/members/{MEMBER_USER_ID}"
        )
        assert remove.status_code == 204, remove.text

        # Membership is gone.
        listed = await owner_client.get(f"/api/v1/organizations/{org_id}/members")
        assert all(m["user_id"] != MEMBER_USER_ID for m in listed.json()), (
            "removed member still listed"
        )

        # The ReBAC grant on graph_a is soft-revoked (no longer active).
        members_a = await rebac_service.list_graph_members(neo4j_test_driver, graph_a)
        assert not any(m["user_id"] == MEMBER_USER_ID for m in members_a)
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_remove_last_owner_refused(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """Removing the org's only owner is refused with 409."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__memtest__ LastOwner")
    try:
        remove = await owner_client.delete(
            f"/api/v1/organizations/{org_id}/members/{OWNER_USER_ID}"
        )
        assert remove.status_code == 409, remove.text
        assert "owner" in remove.json()["detail"].lower()
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_owner_gets_404_on_member_routes(
    owner_client: AsyncClient,
    other_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """A non-owner gets 404 on every member route — existence is masked."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__memtest__ NonOwner")
    try:
        # POST member
        resp = await other_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={"user_id": MEMBER_USER_ID, "org_role": "member"},
        )
        assert resp.status_code == 404, resp.text

        # GET members
        resp = await other_client.get(f"/api/v1/organizations/{org_id}/members")
        assert resp.status_code == 404, resp.text

        # DELETE member
        resp = await other_client.delete(
            f"/api/v1/organizations/{org_id}/members/{MEMBER_USER_ID}"
        )
        assert resp.status_code == 404, resp.text

        # POST subgraph-grants
        resp = await other_client.post(
            f"/api/v1/organizations/{org_id}/members/{MEMBER_USER_ID}/subgraph-grants",
            json={"role": "viewer", "graph_ids": "all"},
        )
        assert resp.status_code == 404, resp.text
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)
