#!/usr/bin/env python
"""
Create missing vector indexes for Neo4j GraphRAG chat functionality
"""
import asyncio
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

logger = get_logger(__name__)


async def create_vector_indexes():
    """Create vector indexes required for GraphRAG chat"""
    
    try:
        # Connect to Neo4j
        await neo4j_client.connect()
        
        # Vector indexes for different node types
        index_queries = [
            # Primary text embeddings index for chunks
            """
            CREATE VECTOR INDEX text_embeddings_primary IF NOT EXISTS
            FOR (c:Chunk) ON (c.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 3072,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            
            # Entity embeddings index for semantic entity search
            """
            CREATE VECTOR INDEX entity_embeddings IF NOT EXISTS
            FOR (e:__Entity__) ON (e.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 3072,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            
            # DocumentChunk embeddings (alternative chunk format)
            """
            CREATE VECTOR INDEX chunk_embeddings IF NOT EXISTS
            FOR (c:DocumentChunk) ON (c.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 3072,
                `vector.similarity_function`: 'cosine'
            }}
            """,
            
            # Entity embeddings (alternative entity format)
            """
            CREATE VECTOR INDEX entity_embeddings_alt IF NOT EXISTS
            FOR (e:Entity) ON (e.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 3072,
                `vector.similarity_function`: 'cosine'
            }}
            """
        ]
        
        # Fulltext indexes for hybrid search
        fulltext_queries = [
            """
            CREATE FULLTEXT INDEX chunk_text_fulltext IF NOT EXISTS
            FOR (c:Chunk) ON EACH [c.text]
            """,
            
            """
            CREATE FULLTEXT INDEX entity_text_fulltext IF NOT EXISTS
            FOR (e:__Entity__) ON EACH [e.name, e.description]
            """,
            
            """
            CREATE FULLTEXT INDEX entity_alt_fulltext IF NOT EXISTS
            FOR (e:Entity) ON EACH [e.name, e.description]
            """
        ]
        
        # Create vector indexes
        logger.info("Creating vector indexes...")
        for i, query in enumerate(index_queries):
            try:
                await neo4j_client.execute_write_query(query)
                logger.info(f"✅ Created vector index {i+1}/{len(index_queries)}")
            except Exception as e:
                logger.warning(f"⚠️ Vector index {i+1} creation failed (may already exist): {e}")
        
        # Create fulltext indexes  
        logger.info("Creating fulltext indexes...")
        for i, query in enumerate(fulltext_queries):
            try:
                await neo4j_client.execute_write_query(query)
                logger.info(f"✅ Created fulltext index {i+1}/{len(fulltext_queries)}")
            except Exception as e:
                logger.warning(f"⚠️ Fulltext index {i+1} creation failed (may already exist): {e}")
        
        # List all indexes to verify
        logger.info("Listing all indexes...")
        list_query = "SHOW INDEXES"
        indexes = await neo4j_client.execute_query(list_query)
        
        logger.info(f"Found {len(indexes)} indexes:")
        for idx in indexes:
            logger.info(f"  - {idx.get('name', 'unknown')}: {idx.get('type', 'unknown')} on {idx.get('labelsOrTypes', 'unknown')}")
        
        logger.info("✅ Vector index creation completed!")
        
    except Exception as e:
        logger.error(f"❌ Failed to create vector indexes: {e}")
        raise
    finally:
        await neo4j_client.close()


if __name__ == "__main__":
    asyncio.run(create_vector_indexes())
