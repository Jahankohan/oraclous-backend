"""Integration tests for the Organization API (TASK-201).

Covers:
  1. POST /organizations          — 201 + correct body
  2. GET  /organizations/{id}     — retrieves the created org
  3. GET  /organizations          — lists orgs owned by the caller
  4. PATCH /organizations/{id}    — mutates fields
  5. Non-owner gets 404 on GET and PATCH (existence is masked)
  6. Create wires the Neo4j :Organization node + BELONGS_TO {org_role:"owner"}

These tests hit the real FastAPI app, PostgreSQL, and Neo4j. Auth is mocked
via dependency_overrides so no live auth-service is required.
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
    """Resolve the test user from the per-client ``X-Test-User`` header.

    A single shared override serves every client; each client carries its own
    user id in a header. This is required because ``app.dependency_overrides``
    is app-global — two clients overriding the same key would otherwise clobber
    one another, and the last fixture to initialise would win for both.
    """
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


async def _delete_neo4j_orgs(driver) -> None:
    """Remove :Organization / :User test nodes from Neo4j."""
    async with driver.session() as session:
        await session.run(
            """
            MATCH (u:User {user_id: $owner})
            DETACH DELETE u
            """,
            {"owner": OWNER_USER_ID},
        )
        await session.run(
            """
            MATCH (o:Organization)
            WHERE o.org_id IS NOT NULL
              AND EXISTS {
                  MATCH (:User {user_id: $owner})-[:BELONGS_TO]->(o)
              }
            DETACH DELETE o
            """,
            {"owner": OWNER_USER_ID},
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_organization_returns_201(
    owner_client: AsyncClient, db_session: AsyncSession
) -> None:
    """POST /organizations → 201 with the created org body."""
    await _delete_test_orgs(db_session)

    resp = await owner_client.post(
        "/api/v1/organizations",
        json={
            "name": "Acme Rail Cooperative",
            "description": "A test organization",
            "settings": {"plan": "enterprise"},
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["name"] == "Acme Rail Cooperative"
    assert data["description"] == "A test organization"
    assert data["owner_user_id"] == OWNER_USER_ID
    assert data["settings"] == {"plan": "enterprise"}
    assert data["status"] == "active"
    assert "id" in data
    assert data["created_at"]
    assert data["updated_at"]

    await _delete_test_orgs(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_and_list_organization(
    owner_client: AsyncClient, db_session: AsyncSession
) -> None:
    """GET /organizations/{id} and GET /organizations return the created org."""
    await _delete_test_orgs(db_session)

    create = await owner_client.post(
        "/api/v1/organizations",
        json={"name": "Org For Get", "description": "desc"},
    )
    assert create.status_code == 201, create.text
    org_id = create.json()["id"]

    try:
        get_resp = await owner_client.get(f"/api/v1/organizations/{org_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["id"] == org_id
        assert get_resp.json()["name"] == "Org For Get"

        list_resp = await owner_client.get("/api/v1/organizations")
        assert list_resp.status_code == 200
        ids = [o["id"] for o in list_resp.json()]
        assert org_id in ids
    finally:
        await _delete_test_orgs(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_organization_mutates_fields(
    owner_client: AsyncClient, db_session: AsyncSession
) -> None:
    """PATCH /organizations/{id} updates name/description/settings."""
    await _delete_test_orgs(db_session)

    create = await owner_client.post(
        "/api/v1/organizations",
        json={"name": "Before", "description": "old"},
    )
    assert create.status_code == 201, create.text
    org_id = create.json()["id"]

    try:
        patch_resp = await owner_client.patch(
            f"/api/v1/organizations/{org_id}",
            json={"name": "After", "settings": {"k": "v"}},
        )
        assert patch_resp.status_code == 200, patch_resp.text
        data = patch_resp.json()
        assert data["name"] == "After"
        assert data["description"] == "old"  # untouched
        assert data["settings"] == {"k": "v"}
    finally:
        await _delete_test_orgs(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_owner_gets_404_on_get_and_patch(
    owner_client: AsyncClient,
    other_client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """A non-owner user must get 404 (not 403) on GET and PATCH."""
    await _delete_test_orgs(db_session)

    create = await owner_client.post(
        "/api/v1/organizations",
        json={"name": "Private Org", "description": "secret"},
    )
    assert create.status_code == 201, create.text
    org_id = create.json()["id"]

    try:
        get_resp = await other_client.get(f"/api/v1/organizations/{org_id}")
        assert get_resp.status_code == 404

        patch_resp = await other_client.patch(
            f"/api/v1/organizations/{org_id}",
            json={"name": "Hijacked"},
        )
        assert patch_resp.status_code == 404
    finally:
        await _delete_test_orgs(db_session)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_wires_neo4j_organization_node(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """Creating an org wires the :Organization node + BELONGS_TO owner edge."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_orgs(neo4j_test_driver)

    create = await owner_client.post(
        "/api/v1/organizations",
        json={"name": "Graph-Synced Org", "description": "neo4j check"},
    )
    assert create.status_code == 201, create.text
    org_id = create.json()["id"]

    try:
        async with neo4j_test_driver.session() as session:
            result = await session.run(
                """
                MATCH (u:User {user_id: $owner})
                      -[r:BELONGS_TO]->(o:Organization {org_id: $org_id})
                RETURN o.name AS name,
                       o.status AS status,
                       o.graph_id AS graph_id,
                       r.org_role AS org_role
                """,
                {"owner": OWNER_USER_ID, "org_id": org_id},
            )
            record = await result.single()

        assert record is not None, "expected :Organization node + BELONGS_TO edge"
        assert record["name"] == "Graph-Synced Org"
        assert record["status"] == "active"
        assert record["graph_id"] == "__system__"
        assert record["org_role"] == "owner"
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_orgs(neo4j_test_driver)
