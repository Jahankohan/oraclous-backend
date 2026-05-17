"""Integration tests for subgraph (knowledge graph) ownership by an
Organization (TASK-202).

Covers:
  1. POST /graphs with an explicit org_id the caller owns → 201, body.org_id set
  2. POST /graphs with no org_id → 201, a default org is created/used
  3. POST /graphs with an org_id the caller does NOT own → 404
  4. Creation wires the (:Organization)-[:OWNS]->(:Graph) and
     (:User)-[:CREATED]->(:Graph) Neo4j edges
  5. GET /organizations/{org_id}/graphs returns the org's graphs;
     a non-owner gets 404

These tests hit the real FastAPI app, PostgreSQL, and Neo4j. Auth is mocked
via dependency_overrides — no live auth-service is required. The per-client
``X-Test-User`` header lets two clients share one app-global override.
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OWNER_USER_ID = str(uuid.uuid4())
OTHER_USER_ID = str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _sync_neo4j_driver():
    """Ensure ``neo4j_client.sync_driver`` is connected.

    The graph-create endpoint uses the *sync* Neo4j driver, which is normally
    initialised by the FastAPI lifespan handler. The test ASGI client does not
    run lifespan startup, so we connect it explicitly here.
    """
    from app.core.neo4j_client import neo4j_client

    if neo4j_client.sync_driver is None:
        neo4j_client.connect_sync()
    yield


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
    """Remove :Graph / :Organization / :User test nodes from Neo4j."""
    async with driver.session() as session:
        # Graphs owned by either test user.
        await session.run(
            """
            MATCH (g:Graph:__Platform__)
            WHERE g.user_id IN $users
            DETACH DELETE g
            """,
            {"users": [OWNER_USER_ID, OTHER_USER_ID]},
        )
        # Organizations belonging to either test user, then the users.
        for user in (OWNER_USER_ID, OTHER_USER_ID):
            await session.run(
                """
                MATCH (o:Organization)
                WHERE o.org_id IS NOT NULL
                  AND EXISTS {
                      MATCH (:User {user_id: $user})-[:BELONGS_TO]->(o)
                  }
                DETACH DELETE o
                """,
                {"user": user},
            )
            await session.run(
                "MATCH (u:User {user_id: $user}) DETACH DELETE u",
                {"user": user},
            )


async def _cleanup(db: AsyncSession, driver) -> None:
    await _delete_test_orgs(db)
    await _delete_neo4j_state(driver)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_graph_with_explicit_org(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """POST /graphs with an org_id the caller owns → 201, response carries org_id."""
    await _cleanup(db_session, neo4j_test_driver)

    org = await owner_client.post(
        "/api/v1/organizations",
        json={"name": "Explicit Org", "description": "owns graphs"},
    )
    assert org.status_code == 201, org.text
    org_id = org.json()["id"]

    try:
        resp = await owner_client.post(
            "/api/v1/graphs",
            json={"name": "Graph In Explicit Org", "org_id": org_id},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["org_id"] == org_id
        assert body["name"] == "Graph In Explicit Org"
    finally:
        await _cleanup(db_session, neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_graph_without_org_uses_default(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """POST /graphs with no org_id → 201, a default org is created and used."""
    await _cleanup(db_session, neo4j_test_driver)

    try:
        resp = await owner_client.post(
            "/api/v1/graphs",
            json={"name": "Graph With Default Org"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["org_id"], "expected a default org_id to be assigned"

        # The default org must be a real org owned by the caller.
        org_resp = await owner_client.get(f"/api/v1/organizations/{body['org_id']}")
        assert org_resp.status_code == 200, org_resp.text
        assert org_resp.json()["owner_user_id"] == OWNER_USER_ID

        # A second graph with no org_id reuses the same default org (idempotent).
        resp2 = await owner_client.post(
            "/api/v1/graphs",
            json={"name": "Second Default-Org Graph"},
        )
        assert resp2.status_code == 201, resp2.text
        assert resp2.json()["org_id"] == body["org_id"]
    finally:
        await _cleanup(db_session, neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_graph_with_unowned_org_returns_404(
    owner_client: AsyncClient,
    other_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """POST /graphs with an org_id the caller does NOT own → 404."""
    await _cleanup(db_session, neo4j_test_driver)

    org = await owner_client.post(
        "/api/v1/organizations",
        json={"name": "Owner-Only Org", "description": "private"},
    )
    assert org.status_code == 201, org.text
    org_id = org.json()["id"]

    try:
        # other_client does not own this org.
        resp = await other_client.post(
            "/api/v1/graphs",
            json={"name": "Hijack Attempt", "org_id": org_id},
        )
        assert resp.status_code == 404, resp.text

        # A non-existent org_id is also a 404 (existence is not leaked).
        resp_missing = await owner_client.post(
            "/api/v1/graphs",
            json={"name": "Missing Org Graph", "org_id": str(uuid.uuid4())},
        )
        assert resp_missing.status_code == 404, resp_missing.text
    finally:
        await _cleanup(db_session, neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_graph_wires_neo4j_ownership_edges(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """Graph creation wires (:Organization)-[:OWNS]->(:Graph) and
    (:User)-[:CREATED]->(:Graph)."""
    await _cleanup(db_session, neo4j_test_driver)

    org = await owner_client.post(
        "/api/v1/organizations",
        json={"name": "Edge-Check Org", "description": "neo4j edges"},
    )
    assert org.status_code == 201, org.text
    org_id = org.json()["id"]

    try:
        resp = await owner_client.post(
            "/api/v1/graphs",
            json={"name": "Edge-Check Graph", "org_id": org_id},
        )
        assert resp.status_code == 201, resp.text
        graph_id = resp.json()["id"]

        async with neo4j_test_driver.session() as session:
            result = await session.run(
                """
                MATCH (o:Organization {org_id: $org_id})
                      -[:OWNS]->(g:Graph:__Platform__ {graph_id: $graph_id})
                MATCH (u:User {user_id: $owner})-[:CREATED]->(g)
                RETURN g.org_id AS org_id
                """,
                {"org_id": org_id, "graph_id": graph_id, "owner": OWNER_USER_ID},
            )
            record = await result.single()

        assert record is not None, "expected OWNS + CREATED edges to exist"
        assert record["org_id"] == org_id
    finally:
        await _cleanup(db_session, neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_organization_graphs(
    owner_client: AsyncClient,
    other_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """GET /organizations/{org_id}/graphs returns the org's graphs;
    a non-owner gets 404."""
    await _cleanup(db_session, neo4j_test_driver)

    org = await owner_client.post(
        "/api/v1/organizations",
        json={"name": "Listing Org", "description": "has graphs"},
    )
    assert org.status_code == 201, org.text
    org_id = org.json()["id"]

    try:
        g1 = await owner_client.post(
            "/api/v1/graphs",
            json={"name": "Org Graph One", "org_id": org_id},
        )
        g2 = await owner_client.post(
            "/api/v1/graphs",
            json={"name": "Org Graph Two", "org_id": org_id},
        )
        assert g1.status_code == 201, g1.text
        assert g2.status_code == 201, g2.text
        created_ids = {g1.json()["id"], g2.json()["id"]}

        list_resp = await owner_client.get(f"/api/v1/organizations/{org_id}/graphs")
        assert list_resp.status_code == 200, list_resp.text
        listed = list_resp.json()
        listed_ids = {g["id"] for g in listed}
        assert created_ids <= listed_ids
        for g in listed:
            assert g["org_id"] == org_id

        # A non-owner must get 404 — org existence is not leaked.
        other_resp = await other_client.get(f"/api/v1/organizations/{org_id}/graphs")
        assert other_resp.status_code == 404, other_resp.text
    finally:
        await _cleanup(db_session, neo4j_test_driver)
