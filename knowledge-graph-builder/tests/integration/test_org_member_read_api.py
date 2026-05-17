"""Integration tests for member-side organization read access (TASK-208).

Covers:
  1. A user added as a member sees that org in GET /organizations, with the
     org_role reflecting their granted role.
  2. That member can GET /organizations/{id} and gets 200 + their org_role.
  3. A user with no BELONGS_TO edge still gets 404 on GET /organizations/{id}.
  4. The owner sees org_role="owner" on GET (single) and GET (list).

These tests hit the real FastAPI app, PostgreSQL, and Neo4j. Auth is mocked
via dependency_overrides (per-client ``X-Test-User`` header), mirroring
``test_organizations_api.py`` and ``test_org_members_api.py``.
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
MEMBER_USER_ID = str(uuid.uuid4())
STRANGER_USER_ID = str(uuid.uuid4())

# All test-created user ids — for Neo4j teardown.
_TEST_USER_IDS = [OWNER_USER_ID, MEMBER_USER_ID, STRANGER_USER_ID]


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
    """Remove :Organization / :User test nodes from Neo4j."""
    async with driver.session() as session:
        await session.run(
            "MATCH (u:User) WHERE u.user_id IN $ids DETACH DELETE u",
            {"ids": _TEST_USER_IDS},
        )
        await session.run(
            """
            MATCH (o:Organization)
            WHERE o.name STARTS WITH '__memreadtest__'
            DETACH DELETE o
            """,
        )


async def _create_org(client: AsyncClient, name: str) -> str:
    """Create an organization via the API and return its org_id."""
    resp = await client.post(
        "/api/v1/organizations",
        json={"name": name, "description": "member-read-test"},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_member_sees_org_in_list(
    owner_client: AsyncClient,
    member_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """A user added as a member sees the org in GET /organizations w/ org_role."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__memreadtest__ List")
    try:
        add = await owner_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={"user_id": MEMBER_USER_ID, "org_role": "member"},
        )
        assert add.status_code == 201, add.text

        listed = await member_client.get("/api/v1/organizations")
        assert listed.status_code == 200, listed.text
        by_id = {o["id"]: o for o in listed.json()}
        assert org_id in by_id, "member should see the org they belong to"
        assert by_id[org_id]["org_role"] == "member"
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_member_can_get_org(
    owner_client: AsyncClient,
    member_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """A member can GET /organizations/{id} and gets 200 + their org_role."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__memreadtest__ Get")
    try:
        add = await owner_client.post(
            f"/api/v1/organizations/{org_id}/members",
            json={"user_id": MEMBER_USER_ID, "org_role": "admin"},
        )
        assert add.status_code == 201, add.text

        get_resp = await member_client.get(f"/api/v1/organizations/{org_id}")
        assert get_resp.status_code == 200, get_resp.text
        data = get_resp.json()
        assert data["id"] == org_id
        assert data["name"] == "__memreadtest__ Get"
        assert data["org_role"] == "admin"
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_non_member_gets_404_on_get(
    owner_client: AsyncClient,
    stranger_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """A user with no BELONGS_TO edge still gets 404 on GET /organizations/{id}."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    org_id = await _create_org(owner_client, "__memreadtest__ NonMember")
    try:
        get_resp = await stranger_client.get(f"/api/v1/organizations/{org_id}")
        assert get_resp.status_code == 404, get_resp.text

        # The non-member also does not see it in their list.
        listed = await stranger_client.get("/api/v1/organizations")
        assert listed.status_code == 200, listed.text
        assert org_id not in [o["id"] for o in listed.json()]
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_owner_sees_org_role_owner(
    owner_client: AsyncClient,
    db_session: AsyncSession,
    neo4j_test_driver,
) -> None:
    """The owner sees org_role='owner' on GET (single), GET (list), POST."""
    await _delete_test_orgs(db_session)
    await _delete_neo4j_state(neo4j_test_driver)

    create = await owner_client.post(
        "/api/v1/organizations",
        json={"name": "__memreadtest__ OwnerRole", "description": "x"},
    )
    assert create.status_code == 201, create.text
    org_id = create.json()["id"]
    # POST response carries org_role.
    assert create.json()["org_role"] == "owner"

    try:
        get_resp = await owner_client.get(f"/api/v1/organizations/{org_id}")
        assert get_resp.status_code == 200, get_resp.text
        assert get_resp.json()["org_role"] == "owner"

        listed = await owner_client.get("/api/v1/organizations")
        assert listed.status_code == 200, listed.text
        by_id = {o["id"]: o for o in listed.json()}
        assert org_id in by_id
        assert by_id[org_id]["org_role"] == "owner"
    finally:
        await _delete_test_orgs(db_session)
        await _delete_neo4j_state(neo4j_test_driver)
