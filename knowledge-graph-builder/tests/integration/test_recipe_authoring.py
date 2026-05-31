"""Integration test — `save_recipe_draft`, step 3 of the authoring loop.

TASK-225 / STORY-034 / ADR-022.

`save_recipe_draft` is a thin wrapper over `RecipeLibrary.store`, which is
Postgres-backed (STORY-034 open question 2; ADR-020 — a recipe is operational
configuration, not graph content). So this is an **integration test** against
the live Dockerized Postgres.

Each test:

  * ensures the `recipes` table exists — `Recipe.__table__.create(checkfirst)`,
    mirroring `alembic/versions/add_recipes_table.py`;
  * uses a fresh `uuid4` test `graph_id` — never real data;
  * exercises `save_recipe_draft` and asserts it stored a draft;
  * deletes its own rows in teardown;
  * skips cleanly when Postgres is unreachable.

Connection: `settings.POSTGRES_URL` points at the Docker-network host
`postgres`; from the test host that is `localhost`. `TEST_POSTGRES_URL`
overrides it. The whole module skips if no engine connects — the same pattern
as `tests/integration/test_recipe_library.py`.
"""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.models.recipe import Recipe
from app.recipes.authoring import save_recipe_draft
from app.recipes.engine import RecipeValidationError

# ---------------------------------------------------------------------------
# Connection — resolve a host reachable from the test process
# ---------------------------------------------------------------------------


def _resolve_postgres_url() -> str:
    """A Postgres URL reachable from the test host.

    `settings.POSTGRES_URL` uses the Docker-network host `postgres`; rewritten
    to `localhost` for a test process running outside the compose network.
    `TEST_POSTGRES_URL` takes precedence if set.
    """
    override = os.getenv("TEST_POSTGRES_URL")
    if override:
        return override
    return settings.POSTGRES_URL.replace("@postgres:", "@localhost:")


_POSTGRES_URL = _resolve_postgres_url()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_session():
    """Task-scoped async session; skips the test if Postgres is unreachable."""
    engine = create_async_engine(_POSTGRES_URL, poolclass=NullPool, future=True)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            # Ensure the recipes table exists — equivalent to running the
            # add_recipes_table migration; a no-op if it is already present.
            await conn.run_sync(Recipe.__table__.create, checkfirst=True)
            await conn.commit()
    except Exception as exc:  # noqa: BLE001
        await engine.dispose()
        pytest.skip(f"live Postgres not reachable at {_POSTGRES_URL}: {exc}")

    session_maker = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_maker() as session:
        yield session
    await engine.dispose()


@pytest_asyncio.fixture
async def graph_id(db_session: AsyncSession):
    """A dedicated test tenant id; its rows are deleted in teardown."""
    gid = f"recipe-authoring-test-{uuid.uuid4()}"
    yield gid
    await db_session.execute(delete(Recipe).where(Recipe.graph_id == gid))
    await db_session.commit()


# ---------------------------------------------------------------------------
# Recipe builder — a minimal document valid against recipe.schema.json
# ---------------------------------------------------------------------------


def _make_recipe(
    *,
    source_type: str = "csv",
    shape_signature: str = "people.csv{id,name,team}",
    concern: str = "Map team membership.",
) -> dict:
    """A minimal recipe document — the kind the data-specialist agent authors."""
    return {
        "recipe_format_version": "0.2",
        "id": f"rcp_{uuid.uuid4()}",
        "version": 1,
        "status": "draft",
        "concern": concern,
        "applies_to": {
            "source_type": source_type,
            "shape_signature": shape_signature,
        },
        "defaults": {"provenance": "EXTRACTED", "materialize_fine_grain": False},
        "authoring": {
            "authored_by": "data-specialist",
            "created": "2026-05-22",
        },
        "mappings": [
            {
                "id": "people",
                "match": {"unit_kind": "record"},
                "project_to": "node",
                "label": "Person",
                "identity": {
                    "scheme": "deterministic",
                    "from": ["column:name"],
                    "normalize": ["casefold", "trim"],
                },
                "materialize": True,
                "properties": [],
            }
        ],
    }


# ---------------------------------------------------------------------------
# save_recipe_draft
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_recipe_draft_stores_version_1_draft(
    db_session: AsyncSession, graph_id: str
) -> None:
    """An agent-authored recipe is stored as version 1, status draft."""
    recipe = _make_recipe()

    stored = await save_recipe_draft(recipe, graph_id, db_session)
    await db_session.commit()

    assert stored["id"] == recipe["id"]
    assert stored["version"] == 1
    assert stored["status"] == "draft"

    # The row is actually persisted under the asking tenant.
    row = await db_session.get(Recipe, (recipe["id"], 1))
    assert row is not None
    assert row.graph_id == graph_id
    assert row.status == "draft"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_recipe_draft_bumps_version_on_resave(
    db_session: AsyncSession, graph_id: str
) -> None:
    """Re-saving the same recipe id writes the next version — every draft kept."""
    recipe = _make_recipe()

    s1 = await save_recipe_draft(recipe, graph_id, db_session)
    s2 = await save_recipe_draft(recipe, graph_id, db_session)
    await db_session.commit()

    assert [s1["version"], s2["version"]] == [1, 2]
    assert s1["status"] == "draft" and s2["status"] == "draft"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_save_recipe_draft_rejects_schema_invalid_recipe(
    db_session: AsyncSession, graph_id: str
) -> None:
    """An invalid recipe is refused — validation happens inside the library."""
    bad = _make_recipe()
    del bad["mappings"]  # `mappings` is required by recipe.schema.json

    with pytest.raises(RecipeValidationError):
        await save_recipe_draft(bad, graph_id, db_session)

    # Nothing was persisted.
    row = await db_session.get(Recipe, (bad["id"], 1))
    assert row is None
