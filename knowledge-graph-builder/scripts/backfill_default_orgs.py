"""Backfill default organizations onto pre-TASK-202 knowledge graphs.

Before TASK-202, :Graph:__Platform__ nodes were created without an ``org_id``
and without an owning ``(:Organization)-[:OWNS]->(:Graph)`` edge. This script
brings legacy graphs up to the TASK-202 model:

  1. Find every distinct ``user_id`` that owns :Graph:__Platform__ nodes with a
     null / missing ``org_id``.
  2. For each such user, resolve their default (personal) organization via
     ``organization_service.get_or_create_default_org`` — creating a
     "Personal Organization" if they own none.
  3. For each of that user's org-less graphs:
       - set ``g.org_id``
       - MERGE ``(:Organization)-[:OWNS]->(:Graph)``
       - MERGE ``(:User)-[:CREATED]->(:Graph)``
  4. (TASK-203 Part 2) For every :Agent:__Platform__ that still lacks an
     ``org_id``, copy the ``org_id`` from its home graph and MERGE
     ``(:Organization)-[:HAS_AGENT]->(:Agent)``.

Idempotent — safe to re-run. Graphs that already have an ``org_id`` are skipped
because step 1 only selects graphs where ``org_id`` IS NULL; the agent step
only touches agents whose ``org_id`` is null/absent.

Run with:
    python scripts/backfill_default_orgs.py
  or
    python -m scripts.backfill_default_orgs

Env vars required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, POSTGRES_URL
(the same values the app uses).
"""

from __future__ import annotations

import asyncio
import os
import sys

# Allow running from project root (`python scripts/backfill_default_orgs.py`)
# as well as module mode (`python -m scripts.backfill_default_orgs`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j import AsyncGraphDatabase  # noqa: E402
from sqlalchemy.ext.asyncio import (  # noqa: E402
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.logging import get_logger  # noqa: E402
from app.services import organization_service  # noqa: E402

logger = get_logger(__name__)


async def _users_with_orgless_graphs(driver) -> list[str]:
    """Return distinct user_ids owning :Graph nodes that have no org_id."""
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (g:Graph:__Platform__)
            WHERE g.org_id IS NULL AND g.user_id IS NOT NULL
            RETURN DISTINCT g.user_id AS user_id
            """
        )
        return [record["user_id"] async for record in result]


async def _backfill_user_graphs(driver, user_id: str, org_id: str) -> int:
    """Wire every org-less graph of *user_id* to *org_id*. Returns count touched.

    MERGE makes both the org_id assignment and the OWNS / CREATED edges
    idempotent. Only graphs that still lack an org_id are affected.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (g:Graph:__Platform__ {user_id: $user_id})
            WHERE g.org_id IS NULL
            SET g.org_id = $org_id
            WITH g
            MATCH (o:Organization {org_id: $org_id})
            MERGE (o)-[:OWNS]->(g)
            WITH g
            MERGE (u:User:__Platform__ {
                user_id: $user_id, graph_id: "__system__"
            })
            MERGE (u)-[:CREATED]->(g)
            RETURN count(g) AS touched
            """,
            {"user_id": user_id, "org_id": org_id},
        )
        record = await result.single()
        return int(record["touched"]) if record else 0


async def _backfill_agent_orgs(driver) -> int:
    """Wire every org-less :Agent to its home graph's organization.

    For each :Agent:__Platform__ whose ``org_id`` is null/absent, copy the
    ``org_id`` from its home graph (``a.graph_id``) and MERGE
    ``(:Organization)-[:HAS_AGENT]->(:Agent)``. Agents whose home graph still
    lacks an ``org_id`` are skipped (run the graph backfill first).

    Idempotent — the WHERE clause only selects agents that still need it, and
    the HAS_AGENT edge is MERGE-d. Returns the count of agents touched.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (a:Agent:__Platform__)
            WHERE a.org_id IS NULL
            MATCH (g:Graph:__Platform__ {graph_id: a.graph_id})
            WHERE g.org_id IS NOT NULL
            SET a.org_id = g.org_id
            WITH a, g
            MATCH (o:Organization:__Platform__ {org_id: g.org_id})
            MERGE (o)-[:HAS_AGENT]->(a)
            RETURN count(DISTINCT a) AS touched
            """
        )
        record = await result.single()
        return int(record["touched"]) if record else 0


async def run_backfill() -> None:
    """Backfill default orgs across all users with org-less graphs."""
    engine = create_async_engine(settings.POSTGRES_URL, poolclass=NullPool, future=True)
    session_maker = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )

    try:
        await driver.verify_connectivity()

        user_ids = await _users_with_orgless_graphs(driver)
        print(f"Users with org-less graphs: {len(user_ids)}")

        total_graphs = 0
        for user_id in user_ids:
            async with session_maker() as db:
                org_id = await organization_service.get_or_create_default_org(
                    db, driver, user_id
                )
            touched = await _backfill_user_graphs(driver, user_id, org_id)
            total_graphs += touched
            print(f"  user {user_id}: org {org_id} -> {touched} graph(s)")

        # TASK-203 Part 2 — bring pre-registry :Agent nodes up to the
        # org-aware model. Runs after the graph backfill so every graph that
        # can have an org_id already does.
        agents_touched = await _backfill_agent_orgs(driver)

        print("\nBackfill complete.")
        print(f"  -> {len(user_ids)} user(s) processed.")
        print(f"  -> {total_graphs} graph(s) assigned an owning organization.")
        print(f"  -> {agents_touched} agent(s) assigned an owning organization.")
    finally:
        await driver.close()
        await engine.dispose()


def main() -> None:
    print("TASK-202 / TASK-203 — Default Organization + Agent Backfill")
    print("=" * 50)
    asyncio.run(run_backfill())


if __name__ == "__main__":
    main()
