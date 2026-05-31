"""Integration test — the ingestion-recipe library (TASK-224, STORY-034).

The recipe library is Postgres-backed (STORY-034 open question 2; ADR-020 — a
recipe is operational configuration, not graph content), so these are
**integration tests** against the live Dockerized Postgres.

Each test:

  * ensures the `recipes` table exists — `Recipe.__table__.create(checkfirst)`,
    which mirrors the migration in `alembic/versions/add_recipes_table.py`;
  * uses a fresh `uuid4` test `graph_id` — never real data;
  * exercises store -> version-bump -> promote -> get -> lookup (hit + miss);
  * deletes its own rows in teardown;
  * skips cleanly when Postgres is unreachable.

Postgres connection: `settings.POSTGRES_URL` points at the Docker-network host
`postgres`; from the test host that is `localhost`. `TEST_POSTGRES_URL` (or the
`localhost` rewrite) overrides it. The whole module skips if no engine connects.
"""

from __future__ import annotations

import copy
import os
import uuid

import pytest
import pytest_asyncio
from sqlalchemy import delete, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.models.recipe import Recipe
from app.recipes.engine import RecipeValidationError
from app.recipes.library import RecipeLibrary

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
    gid = f"recipe-lib-test-{uuid.uuid4()}"
    yield gid
    await db_session.execute(delete(Recipe).where(Recipe.graph_id == gid))
    await db_session.commit()


# ---------------------------------------------------------------------------
# Recipe builders — minimal documents valid against recipe.schema.json
# ---------------------------------------------------------------------------


def _make_recipe(
    *,
    source_type: str = "relational",
    shape_signature: str = "employees(id,name)",
    concern: str = "Understand the reporting structure.",
) -> dict:
    """A minimal recipe document that validates against recipe.schema.json."""
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
                "id": "employees",
                "match": {"unit_kind": "table", "name": "employees"},
                "project_to": "node",
                "label": "Employee",
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
# store + versioning
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_new_recipe_is_version_1_draft(
    db_session: AsyncSession, graph_id: str
) -> None:
    """A brand-new recipe id is stored at version 1, status draft."""
    lib = RecipeLibrary(db_session)
    recipe = _make_recipe()

    stored = await lib.store(recipe, graph_id)
    await db_session.commit()

    assert stored["version"] == 1
    assert stored["status"] == "draft"
    assert stored["id"] == recipe["id"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_existing_id_bumps_version(
    db_session: AsyncSession, graph_id: str
) -> None:
    """Storing the same id again writes max(version) + 1 — every version kept."""
    lib = RecipeLibrary(db_session)
    recipe = _make_recipe()

    s1 = await lib.store(recipe, graph_id)
    s2 = await lib.store(recipe, graph_id)
    s3 = await lib.store(recipe, graph_id)
    await db_session.commit()

    assert [s1["version"], s2["version"], s3["version"]] == [1, 2, 3]
    # All three versions persist — promotion would not overwrite them.
    assert s2["status"] == "draft" and s3["status"] == "draft"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_store_rejects_schema_invalid_recipe(
    db_session: AsyncSession, graph_id: str
) -> None:
    """An invalid recipe document is refused — RecipeValidationError, nothing stored."""
    lib = RecipeLibrary(db_session)
    bad = _make_recipe()
    del bad["mappings"]  # `mappings` is required by the schema

    with pytest.raises(RecipeValidationError):
        await lib.store(bad, graph_id)

    # Nothing was persisted.
    got = await lib.get(bad["id"], graph_id)
    assert got is None


# ---------------------------------------------------------------------------
# promote
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_promote_marks_version_promoted(
    db_session: AsyncSession, graph_id: str
) -> None:
    """promote flips one version to promoted; siblings stay draft."""
    lib = RecipeLibrary(db_session)
    recipe = _make_recipe()

    await lib.store(recipe, graph_id)  # v1
    await lib.store(recipe, graph_id)  # v2
    await db_session.commit()

    await lib.promote(recipe["id"], 1, graph_id)
    await db_session.commit()

    v1 = await lib.get(recipe["id"], graph_id, version=1)
    v2 = await lib.get(recipe["id"], graph_id, version=2)
    assert v1["status"] == "promoted"
    assert v2["status"] == "draft"  # promotion never touched the sibling


@pytest.mark.integration
@pytest.mark.asyncio
async def test_promote_unknown_version_raises(
    db_session: AsyncSession, graph_id: str
) -> None:
    """Promoting a non-existent (id, version) raises KeyError — no row created."""
    lib = RecipeLibrary(db_session)
    with pytest.raises(KeyError):
        await lib.promote("rcp_does-not-exist", 7, graph_id)


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_latest_version_when_version_omitted(
    db_session: AsyncSession, graph_id: str
) -> None:
    """get with version=None returns the highest version."""
    lib = RecipeLibrary(db_session)
    recipe = _make_recipe()

    await lib.store(recipe, graph_id)  # v1
    await lib.store(recipe, graph_id)  # v2
    await db_session.commit()

    latest = await lib.get(recipe["id"], graph_id)
    assert latest["version"] == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_miss_returns_none(db_session: AsyncSession, graph_id: str) -> None:
    """get for an unknown id returns None."""
    lib = RecipeLibrary(db_session)
    assert await lib.get("rcp_nope", graph_id) is None


# ---------------------------------------------------------------------------
# lookup — the reuse path
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lookup_hit_returns_latest_promoted(
    db_session: AsyncSession, graph_id: str
) -> None:
    """lookup returns the latest promoted recipe matching shape + concern."""
    lib = RecipeLibrary(db_session)
    recipe = _make_recipe(
        source_type="relational",
        shape_signature="employees(id,name)",
        concern="Understand the reporting structure.",
    )

    await lib.store(recipe, graph_id)  # v1
    await lib.store(recipe, graph_id)  # v2
    await db_session.commit()
    await lib.promote(recipe["id"], 1, graph_id)
    await lib.promote(recipe["id"], 2, graph_id)
    await db_session.commit()

    hit = await lib.lookup(
        source_type="relational",
        shape_signature="employees(id,name)",
        concern="Understand the reporting structure.",
        graph_id=graph_id,
    )
    assert hit is not None
    assert hit["id"] == recipe["id"]
    assert hit["version"] == 2  # latest promoted wins
    assert hit["status"] == "promoted"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lookup_ignores_draft_versions(
    db_session: AsyncSession, graph_id: str
) -> None:
    """A recipe with only draft versions is never returned by lookup."""
    lib = RecipeLibrary(db_session)
    recipe = _make_recipe(
        shape_signature="orders(id,total)",
        concern="Track order volume.",
    )
    await lib.store(recipe, graph_id)  # draft only — never promoted
    await db_session.commit()

    miss = await lib.lookup(
        source_type="relational",
        shape_signature="orders(id,total)",
        concern="Track order volume.",
        graph_id=graph_id,
    )
    assert miss is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lookup_miss_on_shape_or_concern_mismatch(
    db_session: AsyncSession, graph_id: str
) -> None:
    """lookup matches exactly on source_type, shape_signature, and concern."""
    lib = RecipeLibrary(db_session)
    recipe = _make_recipe(
        source_type="relational",
        shape_signature="employees(id,name)",
        concern="Understand the reporting structure.",
    )
    await lib.store(recipe, graph_id)
    await db_session.commit()
    await lib.promote(recipe["id"], 1, graph_id)
    await db_session.commit()

    # Wrong shape_signature.
    assert (
        await lib.lookup(
            "relational",
            "different(shape)",
            "Understand the reporting structure.",
            graph_id,
        )
        is None
    )
    # Wrong concern.
    assert (
        await lib.lookup(
            "relational", "employees(id,name)", "A different concern.", graph_id
        )
        is None
    )
    # Wrong source_type.
    assert (
        await lib.lookup(
            "csv",
            "employees(id,name)",
            "Understand the reporting structure.",
            graph_id,
        )
        is None
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_lookup_is_tenant_scoped(db_session: AsyncSession, graph_id: str) -> None:
    """A promoted recipe authored by one tenant is invisible to another.

    Cross-tenant recipe sharing is deliberately deferred (RecipeLibrary docstring).
    """
    lib = RecipeLibrary(db_session)
    recipe = _make_recipe(
        shape_signature="invoices(id,amount)",
        concern="Audit invoice totals.",
    )
    await lib.store(recipe, graph_id)
    await db_session.commit()
    await lib.promote(recipe["id"], 1, graph_id)
    await db_session.commit()

    other_graph = f"recipe-lib-test-{uuid.uuid4()}"
    try:
        # Same shape + concern, different tenant -> miss.
        miss = await lib.lookup(
            "relational", "invoices(id,amount)", "Audit invoice totals.", other_graph
        )
        assert miss is None
        # get is tenant-scoped too.
        assert await lib.get(recipe["id"], other_graph) is None
    finally:
        await db_session.execute(delete(Recipe).where(Recipe.graph_id == other_graph))
        await db_session.commit()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_store_promote_get_lookup(
    db_session: AsyncSession, graph_id: str
) -> None:
    """Full path: store -> version-bump -> promote -> get -> lookup hit + miss."""
    lib = RecipeLibrary(db_session)
    recipe = _make_recipe(
        source_type="csv",
        shape_signature="people.csv{name,team}",
        concern="Map team membership.",
    )

    # store -> v1 draft
    s1 = await lib.store(recipe, graph_id)
    assert s1["version"] == 1 and s1["status"] == "draft"

    # store again -> v2 draft (version bump; v1 untouched)
    s2 = await lib.store(recipe, graph_id)
    assert s2["version"] == 2
    await db_session.commit()

    # lookup before promotion -> miss (drafts are not executed)
    assert (
        await lib.lookup(
            "csv", "people.csv{name,team}", "Map team membership.", graph_id
        )
        is None
    )

    # promote v2
    await lib.promote(recipe["id"], 2, graph_id)
    await db_session.commit()

    # get latest -> v2 promoted
    latest = await lib.get(recipe["id"], graph_id)
    assert latest["version"] == 2 and latest["status"] == "promoted"

    # get explicit v1 -> still draft
    v1 = await lib.get(recipe["id"], graph_id, version=1)
    assert v1["version"] == 1 and v1["status"] == "draft"

    # lookup hit -> the promoted v2
    hit = await lib.lookup(
        "csv", "people.csv{name,team}", "Map team membership.", graph_id
    )
    assert hit is not None and hit["version"] == 2

    # lookup miss on a concern this tenant never authored
    miss = await lib.lookup(
        "csv", "people.csv{name,team}", "An unrelated concern.", graph_id
    )
    assert miss is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stored_document_is_independent_copy(
    db_session: AsyncSession, graph_id: str
) -> None:
    """store does not mutate the caller's recipe dict's id/version identity."""
    lib = RecipeLibrary(db_session)
    recipe = _make_recipe()
    original = copy.deepcopy(recipe)

    await lib.store(recipe, graph_id)
    await db_session.commit()

    # The caller's recipe id is unchanged; the library assigned its own version.
    assert recipe["id"] == original["id"]
