"""Integration tests for cross-subgraph retrieval over LINKED_TO (TASK-205).

A knowledge-graph query against subgraph X must also retrieve from every
subgraph X is ``LINKED_TO`` *that the requesting user can see*. The visible
link set comes from ``linked_to_service.list_graph_links`` (the ADR-021 §4
``min_role`` gate is applied inside it); the effective retrieval set is
``[source_graph_id] + [link.target_graph_id for link in visible_links]``.

Backward compatibility is the top priority — with no link, retrieval against
A returns *only* A's data, byte-identical to pre-TASK-205 behaviour.

Coverage
--------
  1. ``AgentExecutor._effective_graph_ids`` returns source-only when there
     is no link (backward-compat).
  2. With an A→B ``LINKED_TO``, the effective set is ``[A, B]`` and a
     ``graph_search`` over that set returns B's chunk alongside A's.
  3. With NO link, ``graph_search`` against A returns only A's chunk.
  4. A link whose ``min_role`` the caller does not meet contributes
     nothing — the caller's effective set stays source-only.
  5. ``requesting_user_id=None`` (legacy callers) → source-only fallback,
     even when a visible link exists.

These tests hit real Neo4j. They seed two same-org subgraphs with one
``:Chunk`` each (deterministic 3072-d embeddings — no OpenAI dependency),
exercise the real ``graph_id IN $graph_ids`` Cypher, and tear everything
down. All test nodes carry the ``__lsrtest__`` name/id prefix.
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from app.services import linked_to_service
from app.services.agent_executor import AgentExecutor
from app.services.agent_tools import AgentToolkit
from app.services.rebac_service import rebac_service

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ADMIN_USER_ID = str(uuid.uuid4())  # owner on the source graph
VIEWER_USER_ID = str(uuid.uuid4())  # viewer-only on the source graph

_TEST_USER_IDS = [ADMIN_USER_ID, VIEWER_USER_ID]

_EMBED_DIM = 3072  # matches the text_embeddings_primary index config

# A deterministic embedding for the seeded chunks AND the query — every
# vector is identical, so cosine similarity is 1.0 for all candidates and
# the vector index returns them all. The retrieval-set filter
# (`graph_id IN $graph_ids`) is what the test is actually exercising.
_UNIT_VEC = [1.0] + [0.0] * (_EMBED_DIM - 1)


# ---------------------------------------------------------------------------
# Stub embedder
# ---------------------------------------------------------------------------


class _StubEmbedder:
    """Returns a fixed unit vector for any query — keeps the test free of
    network calls and fully deterministic."""

    def embed_query(self, text: str) -> list[float]:
        return list(_UNIT_VEC)


# ---------------------------------------------------------------------------
# Neo4j fixture helpers
# ---------------------------------------------------------------------------


async def _clean_neo4j(driver) -> None:
    """Remove all __lsrtest__ nodes plus the test :User nodes."""
    async with driver.session() as session:
        await session.run(
            "MATCH (u:User) WHERE u.user_id IN $ids DETACH DELETE u",
            {"ids": _TEST_USER_IDS},
        )
        await session.run(
            "MATCH (n) WHERE n.name STARTS WITH '__lsrtest__' DETACH DELETE n",
        )
        await session.run(
            "MATCH (c:Chunk) WHERE c.id STARTS WITH '__lsrtest__' DETACH DELETE c",
        )


async def _ensure_vector_index(driver) -> None:
    """Create the text_embeddings_primary vector index if missing.

    Idempotent — the index is normally created at app startup; this guard
    lets the test run against a fresh DB.
    """
    async with driver.session() as session:
        await session.run(
            f"""
            CREATE VECTOR INDEX text_embeddings_primary IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {_EMBED_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
            """
        )


async def _make_org(driver, org_id: str) -> None:
    async with driver.session() as session:
        await session.run(
            """
            CREATE (o:Organization:__Platform__ {
                org_id: $org_id, name: $name, created_at: datetime()
            })
            """,
            {"org_id": org_id, "name": f"__lsrtest__ org {org_id[:8]}"},
        )


async def _make_graph(driver, org_id: str, owner_user_id: str) -> str:
    """Create a :Graph:__Platform__ OWNED by *org_id* and bootstrap roles."""
    graph_id = str(uuid.uuid4())
    async with driver.session() as session:
        await session.run(
            """
            MATCH (o:Organization:__Platform__ {org_id: $org_id})
            CREATE (g:Graph:__Platform__ {
                graph_id: $graph_id, name: $name, org_id: $org_id,
                user_id: $owner_user_id, status: 'active',
                created_at: datetime(), updated_at: datetime()
            })
            MERGE (o)-[:OWNS]->(g)
            """,
            {
                "org_id": org_id,
                "graph_id": graph_id,
                "name": f"__lsrtest__ graph {graph_id[:8]}",
                "owner_user_id": owner_user_id,
            },
        )
    await rebac_service.bootstrap_graph_roles(driver, graph_id, owner_user_id)
    return graph_id


async def _make_chunk(driver, graph_id: str, chunk_id: str, text: str) -> None:
    """Seed a :Chunk node with a deterministic embedding in *graph_id*."""
    async with driver.session() as session:
        await session.run(
            """
            CREATE (c:Chunk {
                id: $id, graph_id: $graph_id, name: $name,
                text: $text, embedding: $embedding
            })
            """,
            {
                "id": chunk_id,
                "graph_id": graph_id,
                "name": f"__lsrtest__ chunk {chunk_id}",
                "text": text,
                "embedding": list(_UNIT_VEC),
            },
        )


async def _grant(driver, graph_id: str, user_id: str, role: str) -> None:
    await rebac_service.grant_role(
        driver,
        graph_id=graph_id,
        target_user_id=user_id,
        role_name=role,
        granted_by=ADMIN_USER_ID,
    )


async def _drop_test_roles(driver, graph_ids: list[str]) -> None:
    async with driver.session() as session:
        await session.run(
            "MATCH (r:Role) WHERE r.graph_id IN $gids DETACH DELETE r",
            {"gids": graph_ids},
        )


# ---------------------------------------------------------------------------
# Fixture: two same-org subgraphs, one chunk each
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def two_graphs(neo4j_test_driver):
    """Yield ``(graph_a, graph_b)`` — two same-org subgraphs, one
    distinct ``:Chunk`` each. The admin user is owner on both. The viewer
    user is granted only the ``viewer`` role on graph A.
    """
    await _clean_neo4j(neo4j_test_driver)
    await _ensure_vector_index(neo4j_test_driver)
    org_id = str(uuid.uuid4())
    created: list[str] = []
    try:
        await _make_org(neo4j_test_driver, org_id)
        graph_a = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        graph_b = await _make_graph(neo4j_test_driver, org_id, ADMIN_USER_ID)
        created = [graph_a, graph_b]
        await _make_chunk(
            neo4j_test_driver, graph_a, "__lsrtest__chunk-a", "Alpha graph content"
        )
        await _make_chunk(
            neo4j_test_driver, graph_b, "__lsrtest__chunk-b", "Beta graph content"
        )
        await _grant(neo4j_test_driver, graph_a, ADMIN_USER_ID, "owner")
        await _grant(neo4j_test_driver, graph_a, VIEWER_USER_ID, "viewer")
        yield graph_a, graph_b
    finally:
        await _clean_neo4j(neo4j_test_driver)
        await _drop_test_roles(neo4j_test_driver, created)


def _toolkit(driver) -> AgentToolkit:
    """An AgentToolkit permitted to run graph_search, with the stub embedder."""
    return AgentToolkit(driver, ["graph_search"], embedder=_StubEmbedder())


def _executor(driver, graph_id: str, requesting_user_id: str | None) -> AgentExecutor:
    """Build a bare AgentExecutor wired with the driver + requesting user.

    Only ``_effective_graph_ids`` / ``resolve_effective_graph_ids`` are
    exercised here, so the LLM and toolkit are placeholders.
    """
    agent_def = {"graph_id": graph_id, "tools": ["graph_search"]}
    return AgentExecutor(
        agent_def=agent_def,
        toolkit=_toolkit(driver),
        llm=None,  # not used by _effective_graph_ids
        model="test-model",
        driver=driver,
        requesting_user_id=requesting_user_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_effective_graph_ids_source_only_when_no_link(
    two_graphs, neo4j_test_driver
) -> None:
    """Backward-compat: with no LINKED_TO, the effective set is exactly the
    source graph."""
    graph_a, _graph_b = two_graphs
    ex = _executor(neo4j_test_driver, graph_a, ADMIN_USER_ID)
    effective = await ex.resolve_effective_graph_ids()
    assert effective == [graph_a]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_effective_graph_ids_spans_visible_link(
    two_graphs, neo4j_test_driver
) -> None:
    """With an A→B LINKED_TO the caller can see, the effective set is
    ``[A, B]`` — source graph first."""
    graph_a, graph_b = two_graphs
    await linked_to_service.create_graph_link(
        neo4j_test_driver, graph_a, graph_b, "viewer", ADMIN_USER_ID
    )
    ex = _executor(neo4j_test_driver, graph_a, ADMIN_USER_ID)
    effective = await ex.resolve_effective_graph_ids()
    assert effective == [graph_a, graph_b]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_search_spans_linked_subgraph(
    two_graphs, neo4j_test_driver
) -> None:
    """A graph_search over the effective set ``[A, B]`` returns B's chunk
    alongside A's — the core TASK-205 behaviour."""
    graph_a, graph_b = two_graphs
    await linked_to_service.create_graph_link(
        neo4j_test_driver, graph_a, graph_b, "viewer", ADMIN_USER_ID
    )
    ex = _executor(neo4j_test_driver, graph_a, ADMIN_USER_ID)
    effective = await ex.resolve_effective_graph_ids()

    results = await _toolkit(neo4j_test_driver).graph_search(effective, "any query")
    ids = {r.id for r in results}
    assert "__lsrtest__chunk-a" in ids, "source graph chunk must be present"
    assert "__lsrtest__chunk-b" in ids, "linked graph chunk must be present"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_graph_search_no_link_returns_only_source(
    two_graphs, neo4j_test_driver
) -> None:
    """BACKWARD-COMPAT: with NO link, retrieval against A returns only A's
    chunk — byte-identical to pre-TASK-205 single-graph behaviour."""
    graph_a, _graph_b = two_graphs
    ex = _executor(neo4j_test_driver, graph_a, ADMIN_USER_ID)
    effective = await ex.resolve_effective_graph_ids()
    assert effective == [graph_a]

    results = await _toolkit(neo4j_test_driver).graph_search(effective, "any query")
    ids = {r.id for r in results}
    assert "__lsrtest__chunk-a" in ids
    assert "__lsrtest__chunk-b" not in ids, "no link → linked graph must NOT leak"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_link_below_caller_min_role_contributes_nothing(
    two_graphs, neo4j_test_driver
) -> None:
    """A link gated at min_role=editor is invisible to a viewer — the
    viewer's effective set stays source-only and retrieval does not span
    into B."""
    graph_a, graph_b = two_graphs
    # Link gated at editor: a viewer (role index 3) cannot see it.
    await linked_to_service.create_graph_link(
        neo4j_test_driver, graph_a, graph_b, "editor", ADMIN_USER_ID
    )

    # Admin (owner) DOES see the link → effective set spans both.
    admin_ex = _executor(neo4j_test_driver, graph_a, ADMIN_USER_ID)
    assert await admin_ex.resolve_effective_graph_ids() == [graph_a, graph_b]

    # Viewer does NOT meet min_role=editor → effective set is source-only.
    viewer_ex = _executor(neo4j_test_driver, graph_a, VIEWER_USER_ID)
    effective = await viewer_ex.resolve_effective_graph_ids()
    assert effective == [graph_a]

    results = await _toolkit(neo4j_test_driver).graph_search(effective, "any query")
    ids = {r.id for r in results}
    assert "__lsrtest__chunk-a" in ids
    assert "__lsrtest__chunk-b" not in ids, "min_role gate must hide linked data"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_no_requesting_user_falls_back_to_source(
    two_graphs, neo4j_test_driver
) -> None:
    """Legacy callers (requesting_user_id=None) never span linked
    subgraphs, even when a visible link exists — fail-safe fallback."""
    graph_a, graph_b = two_graphs
    await linked_to_service.create_graph_link(
        neo4j_test_driver, graph_a, graph_b, "viewer", ADMIN_USER_ID
    )
    ex = _executor(neo4j_test_driver, graph_a, requesting_user_id=None)
    effective = await ex.resolve_effective_graph_ids()
    assert effective == [graph_a]

    results = await _toolkit(neo4j_test_driver).graph_search(effective, "any query")
    ids = {r.id for r in results}
    assert "__lsrtest__chunk-b" not in ids
