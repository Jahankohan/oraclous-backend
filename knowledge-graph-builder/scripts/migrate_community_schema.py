"""
Community Schema Migration

Migrates the Neo4j schema and PostgreSQL tables to support hierarchical
community detection (Leiden algorithm, LLM summaries, embeddings).

All schema changes are ADDITIVE — existing data is preserved.

Run with:
    python scripts/migrate_community_schema.py

Env vars required: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, POSTGRES_URL
"""

import asyncio
import os
import sys

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncpg
from neo4j import GraphDatabase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Neo4j Migration
# ---------------------------------------------------------------------------

NEO4J_MIGRATIONS = [
    # Step 1: Additive indexes
    (
        "community_graph_level index",
        """
        CREATE INDEX community_graph_level IF NOT EXISTS
            FOR (c:__Community__) ON (c.graph_id, c.level)
        """,
    ),
    (
        "community_status index",
        """
        CREATE INDEX community_status IF NOT EXISTS
            FOR (c:__Community__) ON (c.graph_id, c.status)
        """,
    ),
    (
        "community_parent index",
        """
        CREATE INDEX community_parent IF NOT EXISTS
            FOR (c:__Community__) ON (c.parent_id)
        """,
    ),
    (
        "in_community_graph_level relationship index",
        """
        CREATE INDEX in_community_graph_level IF NOT EXISTS
            FOR ()-[r:IN_COMMUNITY]-() ON (r.graph_id, r.level)
        """,
    ),
    (
        "community_summaries fulltext index",
        """
        CREATE FULLTEXT INDEX community_summaries IF NOT EXISTS
            FOR (c:__Community__) ON EACH [c.summary]
        """,
    ),
    # Step 2: Backfill existing community nodes (stale, level=1)
    (
        "backfill existing __Community__ nodes",
        """
        MATCH (c:__Community__)
        WHERE c.level IS NULL
        SET c.level = 1,
            c.status = 'stale',
            c.algorithm = 'louvain_fallback',
            c.parent_id = null
        """,
    ),
    # Step 3: Backfill IN_COMMUNITY relationship level property
    (
        "backfill IN_COMMUNITY relationship level property",
        """
        MATCH ()-[r:IN_COMMUNITY]->()
        WHERE r.level IS NULL
        SET r.level = 1
        """,
    ),
]


def run_neo4j_migrations(embedding_dim: int) -> None:
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    try:
        with driver.session() as session:
            # Check existing vector index dimension
            existing_dim = _get_existing_vector_index_dim(session)
            logger.info(f"Existing community vector index dimension: {existing_dim}")
            logger.info(f"Target embedding dimension from config: {embedding_dim}")

            # Step 3: Drop + recreate vector index only if dimension mismatch
            if existing_dim is not None and existing_dim != embedding_dim:
                logger.info(
                    f"Dimension mismatch ({existing_dim} → {embedding_dim}): "
                    "dropping and recreating community_embeddings index"
                )
                session.run("DROP INDEX community_embeddings IF EXISTS")
                logger.info("Dropped old community_embeddings index")

            if existing_dim is None or existing_dim != embedding_dim:
                session.run(
                    f"""
                    CREATE VECTOR INDEX community_embeddings IF NOT EXISTS
                        FOR (c:__Community__) ON (c.embedding)
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {embedding_dim},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                    """
                )
                logger.info(
                    f"Created community_embeddings vector index (dim={embedding_dim})"
                )
            else:
                logger.info(
                    "community_embeddings vector index already at correct dimension — skipping"
                )

            # Run all other migrations
            for name, query in NEO4J_MIGRATIONS:
                try:
                    session.run(query)
                    logger.info(f"  ✓ {name}")
                except Exception as e:
                    logger.warning(f"  ⚠ {name}: {e}")

        logger.info("Neo4j migrations complete")
    finally:
        driver.close()


def _get_existing_vector_index_dim(session) -> int | None:
    """Return dimension of existing community_embeddings vector index, or None."""
    try:
        result = session.run(
            "SHOW INDEXES WHERE name = 'community_embeddings' AND type = 'VECTOR'"
        )
        records = list(result)
        if not records:
            return None
        # options dict contains vector.dimensions
        options = records[0].get("options", {}) or {}
        index_config = options.get("indexConfig", {}) or {}
        return index_config.get("vector.dimensions")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PostgreSQL Migration
# ---------------------------------------------------------------------------

POSTGRES_MIGRATIONS = [
    (
        "knowledge_graphs.communities_detected_at",
        "ALTER TABLE knowledge_graphs ADD COLUMN IF NOT EXISTS communities_detected_at TIMESTAMP",
    ),
    (
        "knowledge_graphs.communities_status",
        "ALTER TABLE knowledge_graphs ADD COLUMN IF NOT EXISTS communities_status VARCHAR(20) DEFAULT 'not_detected'",
    ),
    (
        "knowledge_graphs.entity_count_at_detection",
        "ALTER TABLE knowledge_graphs ADD COLUMN IF NOT EXISTS entity_count_at_detection INTEGER DEFAULT 0",
    ),
    (
        "knowledge_graphs.entity_delta_since_detection",
        "ALTER TABLE knowledge_graphs ADD COLUMN IF NOT EXISTS entity_delta_since_detection INTEGER DEFAULT 0",
    ),
]


async def run_postgres_migrations() -> None:
    # Strip async driver prefix if present
    pg_url = settings.POSTGRES_URL.replace("+asyncpg", "")
    conn = await asyncpg.connect(pg_url)
    try:
        for name, sql in POSTGRES_MIGRATIONS:
            try:
                await conn.execute(sql)
                logger.info(f"  ✓ {name}")
            except Exception as e:
                logger.warning(f"  ⚠ {name}: {e}")
        logger.info("PostgreSQL migrations complete")
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


async def main() -> None:
    logger.info("=== Community Schema Migration ===")
    logger.info(f"Neo4j:    {settings.NEO4J_URI}")
    logger.info(f"Postgres: {settings.POSTGRES_URL[:50]}...")

    embedding_dim = settings.VECTOR_INDEX_DIMENSIONS
    logger.info(f"Embedding dim (from config): {embedding_dim}")

    logger.info("\n--- Neo4j ---")
    run_neo4j_migrations(embedding_dim)

    logger.info("\n--- PostgreSQL ---")
    await run_postgres_migrations()

    logger.info("\n=== Migration complete ===")


if __name__ == "__main__":
    asyncio.run(main())
