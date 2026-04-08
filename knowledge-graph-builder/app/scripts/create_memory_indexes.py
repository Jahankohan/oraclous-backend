#!/usr/bin/env python
"""
Create Neo4j indexes for the Agent Memory API.

4 indexes required:
  1. memory_embedding_idx     — Vector index for semantic similarity search
  2. memory_content_idx       — Fulltext index for keyword recall
  3. memory_graph_scope_idx   — Composite index for graph-scoped lookup
  4. memory_content_hash_idx  — Index for deduplication

Run once during deployment or after a Neo4j upgrade:
    python -m app.scripts.create_memory_indexes
"""
import asyncio
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from app.core.logging import get_logger
from app.core.neo4j_client import neo4j_client

logger = get_logger(__name__)


async def create_memory_indexes() -> None:
    await neo4j_client.connect()

    index_queries = [
        # 1. Vector index — semantic similarity search on Memory nodes
        #    Dimensions: 3072 to match text-embedding-3-large used elsewhere.
        """
        CREATE VECTOR INDEX memory_embedding_idx IF NOT EXISTS
        FOR (m:Memory) ON m.embedding
        OPTIONS {indexConfig: {
          `vector.dimensions`: 3072,
          `vector.similarity_function`: 'cosine'
        }}
        """,
        # 2. Fulltext index — keyword recall across memory content
        """
        CREATE FULLTEXT INDEX memory_content_idx IF NOT EXISTS
        FOR (m:Memory) ON EACH [m.content]
        """,
        # 3. Composite lookup — graph-scoped queries by scope/type/validity
        """
        CREATE INDEX memory_graph_scope_idx IF NOT EXISTS
        FOR (m:Memory) ON (m.graph_id, m.scope, m.memory_type, m.valid_to)
        """,
        # 4. Deduplication — content hash within graph
        """
        CREATE INDEX memory_content_hash_idx IF NOT EXISTS
        FOR (m:Memory) ON (m.graph_id, m.content_hash)
        """,
    ]

    logger.info("Creating Memory API Neo4j indexes...")
    for i, query in enumerate(index_queries, start=1):
        try:
            await neo4j_client.execute_write_query(query)
            logger.info(f"OK index {i}/{len(index_queries)}")
        except Exception as e:
            logger.warning(f"Index {i} creation failed (may already exist): {e}")

    # Verify
    indexes = await neo4j_client.execute_query(
        "SHOW INDEXES WHERE labelsOrTypes = ['Memory']"
    )
    logger.info(f"Memory indexes present: {len(indexes)}")
    for idx in indexes:
        logger.info(f"  {idx.get('name')}: {idx.get('type')} — {idx.get('state')}")

    logger.info("Memory index creation complete.")
    await neo4j_client.close()


if __name__ == "__main__":
    asyncio.run(create_memory_indexes())
