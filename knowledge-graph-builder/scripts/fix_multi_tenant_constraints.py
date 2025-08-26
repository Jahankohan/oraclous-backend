#!/usr/bin/env python3
"""
Emergency fix for multi-tenant constraint violations
Run this ONCE to fix the broken global uniqueness constraints
"""

import asyncio
from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

logger = get_logger(__name__)

async def fix_multi_tenant_constraints():
    """Fix broken global uniqueness constraints"""
    
    logger.info("🚨 Starting emergency multi-tenant constraint fix...")
    
    try:
        await neo4j_client.connect()
        
        # Step 1: Get existing problematic constraints
        constraints_query = """
        SHOW CONSTRAINTS 
        YIELD name, labelsOrTypes, properties
        WHERE any(prop IN properties WHERE prop = 'id')
        AND size(properties) = 1
        """
        
        existing_constraints = await neo4j_client.execute_query(constraints_query)
        logger.info(f"Found {len(existing_constraints)} problematic constraints")
        
        # Step 2: Drop broken constraints
        for constraint in existing_constraints:
            constraint_name = constraint["name"]
            logger.info(f"Dropping constraint: {constraint_name}")
            
            drop_query = f"DROP CONSTRAINT {constraint_name} IF EXISTS"
            await neo4j_client.execute_write_query(drop_query)
            
        # Step 3: Create composite constraints for common entity types
        entity_types = ["Entity", "Person", "Organization", "Location", "Concept", "Chunk", "Document"]
        
        for entity_type in entity_types:
            constraint_name = f"{entity_type.lower()}_multi_tenant_unique"
            
            constraint_query = f"""
            CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
            FOR (n:{entity_type}) 
            REQUIRE (n.graph_id, n.id) IS UNIQUE
            """
            
            try:
                await neo4j_client.execute_write_query(constraint_query)
                logger.info(f"✅ Created composite constraint for {entity_type}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create constraint for {entity_type}: {e}")
        
        # Step 4: Create performance indexes
        index_queries = [
            "CREATE INDEX entity_graph_id_index IF NOT EXISTS FOR (n:Entity) ON (n.graph_id)",
            "CREATE INDEX person_graph_id_index IF NOT EXISTS FOR (n:Person) ON (n.graph_id)",
            "CREATE INDEX chunk_graph_id_index IF NOT EXISTS FOR (n:Chunk) ON (n.graph_id)"
        ]
        
        for query in index_queries:
            try:
                await neo4j_client.execute_write_query(query)
                logger.info("✅ Created graph_id performance index")
            except Exception as e:
                logger.debug(f"Index might already exist: {e}")
        
        # Step 5: Validate no constraint violations exist
        await validate_constraint_compliance()
        
        logger.info("🎉 Multi-tenant constraint fix completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"💥 Migration failed: {e}")
        return False

async def validate_constraint_compliance():
    """Check for any existing constraint violations"""
    
    # Check for duplicate (graph_id, id) pairs
    validation_query = """
    MATCH (n)
    WHERE n.graph_id IS NOT NULL AND n.id IS NOT NULL
    WITH n.graph_id as graph_id, n.id as node_id, count(*) as count, collect(n) as nodes
    WHERE count > 1
    RETURN graph_id, node_id, count, [node IN nodes | {labels: labels(node), properties: keys(node)}] as node_details
    LIMIT 10
    """
    
    violations = await neo4j_client.execute_query(validation_query)
    
    if violations:
        logger.error(f"⚠️ Found {len(violations)} constraint violations that need manual cleanup:")
        for violation in violations:
            logger.error(f"  Graph {violation['graph_id']}, ID '{violation['node_id']}': {violation['count']} duplicates")
        return False
    
    logger.info("✅ No constraint violations found")
    return True

if __name__ == "__main__":
    success = asyncio.run(fix_multi_tenant_constraints())
    if success:
        print("✅ Multi-tenant constraints fixed successfully!")
    else:
        print("❌ Fix failed - check logs for details")
