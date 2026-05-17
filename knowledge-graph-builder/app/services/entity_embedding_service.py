"""Entity embedding — ingest-time + backfill embedding of ``:__Entity__`` nodes.

The ingestion pipeline embeds text *chunks* but never the *entities* it
extracts. As a result the ``entity_embeddings`` vector index stays empty for
every graph the pipeline produces, and entity-level features that read it —
the entity deduplicator's embedding pass and federation entity-resolution —
silently degrade.

This service computes a 3072-dim embedding for every ``:__Entity__`` that
lacks one and writes it to ``e.embedding``. It has two callers:

  * the ingest pipeline calls it as a best-effort post-dedup stage so new
    graphs are born with embedded entities;
  * the one-time backfill calls it for graphs ingested before that stage.

It opens its own task-scoped async Neo4j driver. The pipeline runs inside a
Celery worker whose ``self.driver`` is a *sync* driver — passing that to an
``async with driver.session()`` raises "Session object does not support the
asynchronous context manager protocol". Self-connecting keeps the service
caller-agnostic and every connection inside the current event loop.

All Cypher is ``graph_id``-scoped and parameterised.
"""

from __future__ import annotations

from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Embeddings are requested in batches. The OpenAI/OpenRouter API accepts far
# larger batches, but 256 keeps each request payload modest and bounds the
# write transaction that follows it.
_EMBED_BATCH = 256

# Marker labels that are not a node's display identity — skipped when picking
# the human-readable label that goes into the embedding text.
_RESERVED_LABELS: set[str] = {
    "__Entity__",
    "__KGBuilder__",
    "__Platform__",
    "__Chat__",
    "__Community__",
    "__Rebac__",
    "__System__",
}


def _embedding_text(
    name: str | None, labels: list[str] | None, description: str | None
) -> str:
    """Build the text embedded for an entity: ``name (Label) — description``.

    Extraction-built entities usually carry only ``name``; ``description`` is
    appended when present (structured-ingest entities have it).
    """
    label = next(
        (lbl for lbl in (labels or []) if lbl not in _RESERVED_LABELS), "Entity"
    )
    parts = [name or "", f"({label})"]
    if description:
        parts.append(f"— {description}")
    return " ".join(p for p in parts if p).strip()


async def embed_graph_entities(
    *,
    graph_id: str,
    only_missing: bool = True,
) -> dict[str, int]:
    """Embed ``:__Entity__`` nodes of *graph_id* and write ``e.embedding``.

    Opens (and closes) its own task-scoped async Neo4j driver — see the
    module docstring for why it does not take a driver argument.

    Args:
        graph_id: Tenant graph to embed. Every MATCH is scoped to it.
        only_missing: When ``True`` (default) only entities without an
            ``embedding`` are processed — cheap and safe to call on every
            ingest. ``False`` re-embeds the whole graph.

    Returns:
        ``{"total": N, "embedded": M}`` — entities considered and written.
    """
    driver = AsyncGraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    try:
        missing_clause = "AND e.embedding IS NULL" if only_missing else ""
        select_query = f"""
        MATCH (e:__Entity__ {{graph_id: $graph_id}})
        WHERE e.name IS NOT NULL {missing_clause}
        RETURN elementId(e) AS eid, e.name AS name,
               labels(e) AS labels, e.description AS description
        """
        async with driver.session() as session:
            result = await session.run(select_query, {"graph_id": graph_id})
            rows = await result.data()

        if not rows:
            logger.info("entity-embedding: graph %s — nothing to embed", graph_id)
            return {"total": 0, "embedded": 0}

        client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        model = settings.EMBEDDING_MODEL or "text-embedding-3-large"
        embedded = 0
        try:
            for start in range(0, len(rows), _EMBED_BATCH):
                batch = rows[start : start + _EMBED_BATCH]
                texts = [
                    _embedding_text(r["name"], r["labels"], r.get("description"))
                    for r in batch
                ]
                response = await client.embeddings.create(model=model, input=texts)
                # Pair each vector back to its row by the API's `index` field —
                # never assume `response.data` preserves input order.
                vectors = [
                    d.embedding for d in sorted(response.data, key=lambda d: d.index)
                ]
                payload = [
                    {"eid": r["eid"], "emb": vec}
                    for r, vec in zip(batch, vectors, strict=True)
                ]
                async with driver.session() as session:
                    await session.run(
                        """
                        UNWIND $rows AS row
                        MATCH (e:__Entity__ {graph_id: $graph_id})
                        WHERE elementId(e) = row.eid
                        SET e.embedding = row.emb
                        """,
                        {"rows": payload, "graph_id": graph_id},
                    )
                embedded += len(payload)
        finally:
            await client.close()

        logger.info(
            "entity-embedding: graph %s — embedded %d/%d entities",
            graph_id,
            embedded,
            len(rows),
        )
        return {"total": len(rows), "embedded": embedded}
    finally:
        await driver.close()
