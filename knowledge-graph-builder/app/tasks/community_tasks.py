"""
Community Detection Celery Task

Implements hierarchical community detection via the Leiden algorithm
(leidenalg Python library), LLM summarisation, and embedding generation.

Architecture rules:
- Uses SYNC Neo4j driver with minimal pool (NullPool pattern) — Celery worker context
- Never shares drivers with FastAPI
- Every Cypher query filters by graph_id (multi-tenancy)
- Redis lock enforces 1-job-per-graph concurrency
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any

import redis
from neo4j import GraphDatabase
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Celery app (reuse existing app from background_jobs to share broker/backend)
# ---------------------------------------------------------------------------

# Import the existing celery_app rather than creating a new one
from app.services.background_jobs import celery_app  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMMUNITY_DETECTION_MIN_ENTITIES: int = getattr(
    settings, "COMMUNITY_DETECTION_MIN_ENTITIES", 50
)
COMMUNITY_DETECTION_CONCURRENCY: int = getattr(
    settings, "COMMUNITY_DETECTION_CONCURRENCY", 3
)
LLM_SUMMARY_CONCURRENCY: int = getattr(settings, "LLM_SUMMARY_CONCURRENCY", 5)

DEFAULT_LEVELS = [0, 1, 2]
DEFAULT_RESOLUTIONS = [0.5, 1.0, 2.0]

# Prompt template version — bump this to invalidate all cached summaries
PROMPT_TEMPLATE_VERSION = "v1"

SUMMARY_SYSTEM_PROMPT = (
    "You are a knowledge graph analyst. Generate a concise, informative summary "
    "of a cluster of related entities from a knowledge graph."
)

SUMMARY_USER_TEMPLATE = """\
Here is a community of {entity_count} related entities from a knowledge graph:

Entities:
{entity_list}

Key relationships between members:
{relationship_list}

Generate a 2-3 sentence summary that:
1. Describes what these entities have in common (the theme or domain)
2. Identifies the most central/important entity if one exists
3. Notes any interesting structural pattern (e.g., hierarchy, network, timeline)

Summary (2-3 sentences, no bullet points, plain text):"""

# ---------------------------------------------------------------------------
# Deterministic community ID
# ---------------------------------------------------------------------------


def make_community_id(
    graph_id: str, level: int, resolution: float, member_ids: list[str]
) -> str:
    """Deterministic SHA-256 based community ID (first 16 hex chars)."""
    sorted_ids = sorted(member_ids)
    content = f"{graph_id}|L{level}|R{resolution}|" + "|".join(sorted_ids)
    hash_hex = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"community_{hash_hex}"


def make_summary_hash(member_ids: list[str]) -> str:
    """Hash used to skip redundant LLM calls."""
    sorted_ids = sorted(member_ids)
    content = PROMPT_TEMPLATE_VERSION + "|" + "|".join(sorted_ids)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Celery task entry point
# ---------------------------------------------------------------------------


@celery_app.task(bind=True, name="community_tasks.detect_communities_task")
def detect_communities_task(
    self,
    graph_id: str,
    levels: list[int] | None = None,
    resolutions: list[float] | None = None,
    force_rebuild: bool = False,
) -> dict[str, Any]:
    """
    Celery task: hierarchical community detection for a single graph.

    Args:
        graph_id: UUID string of the target graph
        levels: Hierarchy level indices (default [0, 1, 2])
        resolutions: Leiden resolution γ per level (default [0.5, 1.0, 2.0])
        force_rebuild: If True, run even when status == 'active'
    """
    from app.services.task_executor import AsyncTaskExecutor

    return AsyncTaskExecutor.run_async_task(
        _detect_communities_async,
        self,
        graph_id,
        levels or DEFAULT_LEVELS,
        resolutions or DEFAULT_RESOLUTIONS,
        force_rebuild,
    )


# ---------------------------------------------------------------------------
# Async implementation
# ---------------------------------------------------------------------------


async def _detect_communities_async(
    task,
    graph_id: str,
    levels: list[int],
    resolutions: list[float],
    force_rebuild: bool,
) -> dict[str, Any]:
    start_ts = time.time()
    lock_key = f"community_detect:{graph_id}"
    redis_client = redis.Redis.from_url(settings.REDIS_URL)

    # -----------------------------------------------------------------------
    # Step 1 — Pre-flight
    # -----------------------------------------------------------------------
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
        max_connection_pool_size=1,
    )
    pg_engine = create_engine(
        settings.POSTGRES_URL.replace("+asyncpg", ""),
        poolclass=NullPool,
    )

    try:
        # Count entities
        with driver.session() as session:
            entity_count = _count_entities(session, graph_id)

        if entity_count < COMMUNITY_DETECTION_MIN_ENTITIES:
            return {
                "status": "skipped",
                "reason": f"Entity count {entity_count} < minimum {COMMUNITY_DETECTION_MIN_ENTITIES}",
                "graph_id": graph_id,
            }

        # Check current status and Redis lock
        current_status = _get_communities_status_pg(pg_engine, graph_id)
        if current_status == "rebuilding" and not force_rebuild:
            return {
                "status": "skipped",
                "reason": "Community detection already in progress",
                "graph_id": graph_id,
            }

        # Acquire Redis distributed lock (10 min TTL)
        acquired = redis_client.set(lock_key, task.request.id, nx=True, ex=600)
        if not acquired:
            return {
                "status": "skipped",
                "reason": "Another detection job holds the Redis lock",
                "graph_id": graph_id,
            }

        # Mark rebuilding in Postgres
        _update_communities_status_pg(pg_engine, graph_id, "rebuilding")

        # -----------------------------------------------------------------------
        # Step 2 — Extract adjacency list
        # -----------------------------------------------------------------------
        with driver.session() as session:
            edges = _extract_adjacency_list(session, graph_id)
            entity_ids = _get_entity_ids(session, graph_id)

        if not entity_ids:
            _update_communities_status_pg(pg_engine, graph_id, "not_detected")
            return {
                "status": "skipped",
                "reason": "No entities found",
                "graph_id": graph_id,
            }

        # -----------------------------------------------------------------------
        # Step 3 — Run Leiden per level
        # -----------------------------------------------------------------------
        communities_map = _run_leiden(entity_ids, edges, levels, resolutions)

        # -----------------------------------------------------------------------
        # Step 4 — Build hierarchy (assign parent_id)
        # -----------------------------------------------------------------------
        communities_map = _build_hierarchy(communities_map, levels)

        # -----------------------------------------------------------------------
        # Step 5+6 — LLM summarisation + embeddings
        # -----------------------------------------------------------------------
        communities_map = await _summarise_and_embed(communities_map, graph_id, driver)

        # -----------------------------------------------------------------------
        # Step 7 — Write to Neo4j
        # -----------------------------------------------------------------------
        total_written = 0
        communities_per_level: dict[str, int] = {}
        for level in levels:
            comms = communities_map.get(level, {})
            _upsert_communities(driver, graph_id, level, comms, resolutions[level])
            communities_per_level[str(level)] = len(comms)
            total_written += len(comms)

        # -----------------------------------------------------------------------
        # Step 8 — Post-run Postgres update
        # -----------------------------------------------------------------------
        duration_ms = int((time.time() - start_ts) * 1000)
        _update_communities_status_pg(
            pg_engine,
            graph_id,
            "active",
            communities_count=total_written,
            entity_count_at_detection=entity_count,
        )

        logger.info(
            f"Community detection complete for graph {graph_id}: "
            f"{total_written} communities across {len(levels)} levels in {duration_ms}ms"
        )

        return {
            "status": "completed",
            "graph_id": graph_id,
            "communities_per_level": communities_per_level,
            "total_communities": total_written,
            "entities_processed": entity_count,
            "duration_ms": duration_ms,
        }

    except Exception as exc:
        logger.exception(f"Community detection failed for graph {graph_id}: {exc}")
        _update_communities_status_pg(pg_engine, graph_id, "stale")
        raise

    finally:
        redis_client.delete(lock_key)
        driver.close()
        pg_engine.dispose()


# ---------------------------------------------------------------------------
# Neo4j helpers (sync, run inside driver.session())
# ---------------------------------------------------------------------------


def _count_entities(session, graph_id: str) -> int:
    result = session.run(
        "MATCH (e:__Entity__ {graph_id: $graph_id}) RETURN count(e) AS cnt",
        {"graph_id": graph_id},
    )
    return result.single()["cnt"]


def _get_entity_ids(session, graph_id: str) -> list[str]:
    result = session.run(
        "MATCH (e:__Entity__ {graph_id: $graph_id}) RETURN e.id AS eid",
        {"graph_id": graph_id},
    )
    return [r["eid"] for r in result if r["eid"]]


def _extract_adjacency_list(session, graph_id: str) -> list[tuple[str, str, int]]:
    """Return (src_id, tgt_id, weight) triples — both sides filtered by graph_id."""
    result = session.run(
        """
        MATCH (a:__Entity__ {graph_id: $graph_id})-[r]->(b:__Entity__ {graph_id: $graph_id})
        WHERE a.id IS NOT NULL AND b.id IS NOT NULL
        RETURN a.id AS src, b.id AS tgt, count(r) AS weight
        """,
        {"graph_id": graph_id},
    )
    return [(r["src"], r["tgt"], r["weight"]) for r in result]


def _get_entity_names_and_types(
    session, graph_id: str, entity_ids: list[str]
) -> dict[str, dict]:
    """Fetch name + labels for a list of entity IDs."""
    result = session.run(
        """
        MATCH (e:__Entity__ {graph_id: $graph_id})
        WHERE e.id IN $entity_ids
        RETURN e.id AS eid, e.name AS name, labels(e) AS lbls
        """,
        {"graph_id": graph_id, "entity_ids": entity_ids},
    )
    return {
        r["eid"]: {
            "name": r["name"] or r["eid"],
            "type": next((lbl for lbl in r["lbls"] if lbl != "__Entity__"), "Entity"),
        }
        for r in result
    }


def _get_relationships_between(
    session, graph_id: str, entity_ids: list[str], limit: int = 10
) -> list[dict]:
    """Fetch relationships between the given entities (for summarisation)."""
    result = session.run(
        """
        MATCH (a:__Entity__ {graph_id: $graph_id})-[r]->(b:__Entity__ {graph_id: $graph_id})
        WHERE a.id IN $entity_ids AND b.id IN $entity_ids
        RETURN a.name AS src, type(r) AS rel, b.name AS tgt
        LIMIT $lim
        """,
        {"graph_id": graph_id, "entity_ids": entity_ids, "lim": limit},
    )
    return [{"src": r["src"], "rel": r["rel"], "tgt": r["tgt"]} for r in result]


def _get_existing_summary_hashes(session, graph_id: str) -> dict[str, str]:
    """Return {community_id: summary_prompt_hash} for existing communities."""
    result = session.run(
        """
        MATCH (c:__Community__ {graph_id: $graph_id})
        WHERE c.summary_prompt IS NOT NULL
        RETURN c.id AS cid, c.summary_prompt AS hash, c.summary AS summary
        """,
        {"graph_id": graph_id},
    )
    return {r["cid"]: {"hash": r["hash"], "summary": r["summary"]} for r in result}


# ---------------------------------------------------------------------------
# Leiden algorithm
# ---------------------------------------------------------------------------


def _run_leiden(
    entity_ids: list[str],
    edges: list[tuple[str, str, int]],
    levels: list[int],
    resolutions: list[float],
) -> dict[int, dict[str, list[str]]]:
    """Run Leiden at each resolution; return {level: {community_id: [entity_ids]}}."""
    import igraph as ig
    import leidenalg

    # Map entity string IDs → integer indices (igraph requirement)
    id_to_idx = {eid: i for i, eid in enumerate(entity_ids)}
    idx_to_id = {i: eid for eid, i in id_to_idx.items()}
    n = len(entity_ids)

    # Build igraph graph
    edge_list = [
        (id_to_idx[src], id_to_idx[tgt])
        for src, tgt, _ in edges
        if src in id_to_idx and tgt in id_to_idx
    ]
    weights = [
        float(w) for src, tgt, w in edges if src in id_to_idx and tgt in id_to_idx
    ]

    g = ig.Graph(n=n, edges=edge_list, directed=False)
    if weights:
        g.es["weight"] = weights

    result: dict[int, dict[str, list[str]]] = {}
    for level, resolution in zip(levels, resolutions, strict=False):
        partition = leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight" if weights else None,
            resolution_parameter=resolution,
            n_iterations=10,
            seed=42,
        )

        # Build {community_id: [entity_id_strings]}
        level_map: dict[str, list[str]] = {}
        for vertex_idx, community_idx in enumerate(partition.membership):
            eid = idx_to_id[vertex_idx]
            # Temporary community key (we'll derive deterministic ID later)
            key = f"tmp_{community_idx}"
            level_map.setdefault(key, []).append(eid)

        # Replace tmp keys with deterministic IDs
        final_map: dict[str, list[str]] = {}
        for _tmp_key, members in level_map.items():
            cid = make_community_id(
                graph_id="__placeholder__",  # filled during upsert
                level=level,
                resolution=resolution,
                member_ids=members,
            )
            final_map[cid] = members

        result[level] = final_map

    return result


# ---------------------------------------------------------------------------
# Hierarchy: assign parent_id
# ---------------------------------------------------------------------------


def _build_hierarchy(
    communities_map: dict[int, dict[str, list[str]]],
    levels: list[int],
) -> dict[int, dict[str, list[str]]]:
    """
    Enrich each community dict with a 'parent_id' key.
    For level > 0, majority-vote which parent community owns each community.
    Returns the same structure with extra metadata dicts.
    """
    # Convert simple list values to richer dicts
    enriched: dict[int, dict[str, dict]] = {}
    for level in levels:
        enriched[level] = {
            cid: {"members": members, "parent_id": None}
            for cid, members in communities_map.get(level, {}).items()
        }

    # Build entity → community lookup for each level
    entity_to_community: dict[int, dict[str, str]] = {}
    for level in levels:
        entity_to_community[level] = {}
        for cid, data in enriched[level].items():
            for eid in data["members"]:
                entity_to_community[level][eid] = cid

    # Assign parent_id for level > 0
    for i, level in enumerate(levels):
        if i == 0:
            continue
        parent_level = levels[i - 1]
        parent_lookup = entity_to_community.get(parent_level, {})

        for _, data in enriched[level].items():
            # Majority vote
            vote: dict[str, int] = {}
            for eid in data["members"]:
                parent_cid = parent_lookup.get(eid)
                if parent_cid:
                    vote[parent_cid] = vote.get(parent_cid, 0) + 1
            if vote:
                data["parent_id"] = max(vote, key=vote.__getitem__)

    return enriched  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# LLM Summarisation + Embeddings
# ---------------------------------------------------------------------------


async def _summarise_and_embed(
    communities_map: dict[int, dict[str, dict]],
    graph_id: str,
    driver,
) -> dict[int, dict[str, dict]]:
    """
    For each community:
      1. Check if summary_prompt hash matches — if so, reuse existing summary.
      2. Otherwise call LLM (up to LLM_SUMMARY_CONCURRENCY concurrent calls).
      3. Embed the summary.
    """
    from openai import AsyncOpenAI

    openai_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    embedding_model = settings.EMBEDDING_MODEL or "text-embedding-3-large"
    llm_model = "gpt-4o-mini"

    # Fetch existing hashes (for cache check)
    with driver.session() as session:
        existing_hashes = _get_existing_summary_hashes(session, graph_id)

    # Collect all communities that need LLM calls
    tasks: list[tuple[int, str, dict]] = []
    for level, comms in communities_map.items():
        for cid, data in comms.items():
            member_ids = data["members"]
            new_hash = make_summary_hash(member_ids)
            cached = existing_hashes.get(cid)
            if cached and cached["hash"] == new_hash:
                # Reuse cached summary, skip LLM
                data["summary"] = cached["summary"]
                data["summary_prompt"] = new_hash
                data["embedding"] = None  # will be fetched from Neo4j node
            else:
                data["summary_prompt"] = new_hash
                tasks.append((level, cid, data))

    # Fetch entity metadata for summarisation in one batch
    all_member_ids = list(
        {
            eid
            for comms in communities_map.values()
            for data in comms.values()
            for eid in data["members"]
        }
    )
    with driver.session() as session:
        entity_meta = _get_entity_names_and_types(session, graph_id, all_member_ids)

    # LLM calls with concurrency cap
    semaphore = asyncio.Semaphore(LLM_SUMMARY_CONCURRENCY)

    async def summarise_one(level: int, cid: str, data: dict) -> None:
        async with semaphore:
            member_ids = data["members"]
            entity_list = "\n".join(
                f"- {entity_meta.get(eid, {}).get('name', eid)} ({entity_meta.get(eid, {}).get('type', 'Entity')})"
                for eid in member_ids[:20]
            )
            with driver.session() as session:
                rels = _get_relationships_between(session, graph_id, member_ids[:20])
            rel_list = (
                "\n".join(f"- {r['src']} --[{r['rel']}]--> {r['tgt']}" for r in rels)
                or "(no direct relationships found)"
            )

            prompt = SUMMARY_USER_TEMPLATE.format(
                entity_count=len(member_ids),
                entity_list=entity_list,
                relationship_list=rel_list,
            )
            try:
                response = await openai_client.chat.completions.create(
                    model=llm_model,
                    messages=[
                        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,
                    max_tokens=200,
                )
                data["summary"] = response.choices[0].message.content.strip()
            except Exception as exc:
                logger.warning(f"LLM summarisation failed for community {cid}: {exc}")
                names = [
                    entity_meta.get(eid, {}).get("name", eid) for eid in member_ids[:5]
                ]
                data["summary"] = (
                    f"Community of {len(member_ids)} entities including: {', '.join(names)}"
                )

    await asyncio.gather(*(summarise_one(lv, cid, data) for lv, cid, data in tasks))

    # Embed all summaries
    all_embed_tasks: list[tuple[int, str, dict]] = []
    for level, comms in communities_map.items():
        for cid, data in comms.items():
            if data.get("embedding") is None and data.get("summary"):
                all_embed_tasks.append((level, cid, data))

    if all_embed_tasks:
        await _embed_summaries(all_embed_tasks, openai_client, embedding_model)

    return communities_map


async def _embed_summaries(
    tasks: list[tuple[int, str, dict]],
    openai_client,
    embedding_model: str,
) -> None:
    """Batch embed community summaries and store in data dict."""
    semaphore = asyncio.Semaphore(LLM_SUMMARY_CONCURRENCY)

    async def embed_one(level: int, cid: str, data: dict) -> None:
        async with semaphore:
            try:
                response = await openai_client.embeddings.create(
                    model=embedding_model,
                    input=data["summary"],
                )
                data["embedding"] = response.data[0].embedding
            except Exception as exc:
                logger.warning(f"Embedding failed for community {cid}: {exc}")
                data["embedding"] = None

    await asyncio.gather(*(embed_one(lv, cid, data) for lv, cid, data in tasks))


# ---------------------------------------------------------------------------
# Neo4j upsert
# ---------------------------------------------------------------------------


def _upsert_communities(
    driver,
    graph_id: str,
    level: int,
    communities: dict[str, dict],
    resolution: float,
) -> None:
    """MERGE all communities at a given level and their IN_COMMUNITY relationships."""
    total_entities = sum(len(d["members"]) for d in communities.values())

    with driver.session() as session:
        for _cid, data in communities.items():
            members = data["members"]
            # Re-derive deterministic ID with real graph_id
            real_cid = make_community_id(graph_id, level, resolution, members)

            weight = len(members) / max(total_entities, 1)
            session.run(
                """
                MERGE (c:__Community__ {id: $id, graph_id: $graph_id})
                SET c.level = $level,
                    c.resolution = $resolution,
                    c.algorithm = 'leiden',
                    c.summary = $summary,
                    c.summary_prompt = $summary_prompt,
                    c.entity_count = $entity_count,
                    c.weight = $weight,
                    c.parent_id = $parent_id,
                    c.status = 'active',
                    c.last_updated = datetime()
                """,
                {
                    "id": real_cid,
                    "graph_id": graph_id,
                    "level": level,
                    "resolution": resolution,
                    "summary": data.get("summary", ""),
                    "summary_prompt": data.get("summary_prompt"),
                    "entity_count": len(members),
                    "weight": weight,
                    "parent_id": data.get("parent_id"),
                },
            )

            if data.get("embedding"):
                session.run(
                    """
                    MATCH (c:__Community__ {id: $id, graph_id: $graph_id})
                    SET c.embedding = $embedding
                    """,
                    {
                        "id": real_cid,
                        "graph_id": graph_id,
                        "embedding": data["embedding"],
                    },
                )

            # MERGE IN_COMMUNITY relationships
            for eid in members:
                session.run(
                    """
                    MATCH (e:__Entity__ {id: $eid, graph_id: $graph_id})
                    MATCH (c:__Community__ {id: $cid, graph_id: $graph_id})
                    MERGE (e)-[r:IN_COMMUNITY {graph_id: $graph_id, level: $level}]->(c)
                    """,
                    {
                        "eid": eid,
                        "graph_id": graph_id,
                        "cid": real_cid,
                        "level": level,
                    },
                )

        # MERGE PARENT_COMMUNITY relationships (level > 0)
        if level > 0:
            for _cid, data in communities.items():
                parent_id = data.get("parent_id")
                if not parent_id:
                    continue
                real_cid = make_community_id(
                    graph_id, level, resolution, data["members"]
                )
                session.run(
                    """
                    MATCH (child:__Community__ {id: $child_id, graph_id: $graph_id})
                    MATCH (parent:__Community__ {id: $parent_id, graph_id: $graph_id})
                    MERGE (child)-[:PARENT_COMMUNITY {graph_id: $graph_id}]->(parent)
                    """,
                    {
                        "child_id": real_cid,
                        "parent_id": parent_id,
                        "graph_id": graph_id,
                    },
                )


# ---------------------------------------------------------------------------
# PostgreSQL helpers (sync SQLAlchemy, NullPool)
# ---------------------------------------------------------------------------


def _get_communities_status_pg(engine, graph_id: str) -> str:
    """Return communities_status for a graph, or 'not_detected' if column missing."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT communities_status FROM knowledge_graphs WHERE id = :gid"),
                {"gid": graph_id},
            ).fetchone()
            return row[0] if row and row[0] else "not_detected"
    except Exception:
        return "not_detected"


def _update_communities_status_pg(
    engine,
    graph_id: str,
    status: str,
    communities_count: int | None = None,
    entity_count_at_detection: int | None = None,
) -> None:
    try:
        sets = ["communities_status = :status"]
        params: dict[str, Any] = {"status": status, "gid": graph_id}

        if status == "active":
            sets.append("communities_detected_at = now()")
            sets.append("entity_delta_since_detection = 0")
            if communities_count is not None:
                sets.append("communities_count = :cc")
                params["cc"] = communities_count
            if entity_count_at_detection is not None:
                sets.append("entity_count_at_detection = :ecd")
                params["ecd"] = entity_count_at_detection

        sql = f"UPDATE knowledge_graphs SET {', '.join(sets)} WHERE id = :gid"
        with engine.connect() as conn:
            conn.execute(text(sql), params)
            conn.commit()
    except Exception as exc:
        logger.warning(f"Failed to update communities_status in Postgres: {exc}")
