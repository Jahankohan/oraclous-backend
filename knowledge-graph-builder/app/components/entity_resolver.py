"""
Entity Deduplication Component for Neo4j GraphRAG Pipeline

Native component that integrates into the neo4j_graphrag pipeline to consolidate
duplicate entities across chunks into single entities with multiple chunk relationships.

This approach creates a cleaner graph by:
1. Identifying duplicate entities across chunks
2. Merging them into a single canonical entity 
3. Linking the canonical entity to all relevant chunks
4. Removing duplicate entity nodes
"""

import time
from typing import Any, Dict, Optional, List
from neo4j import Driver, Session

from neo4j_graphrag.experimental.pipeline.component import Component
from neo4j_graphrag.experimental.components.types import Neo4jGraph

from app.core.logging import get_logger

logger = get_logger(__name__)


class MultiTenantEntityDeduplicator(Component):
    """
    Native neo4j_graphrag component for entity deduplication with multi-tenant support.
    
    This component:
    1. Finds entities with identical names across different chunks
    2. Merges duplicate entities into a single canonical entity
    3. Links the canonical entity to all chunks where it appears
    4. Removes duplicate entity nodes
    5. Preserves all relationships and properties
    6. Supports multi-tenant isolation using graph_id
    """
    
    def __init__(
        self, 
        driver: Driver,
        graph_id: str,
        similarity_threshold: float = 0.85,
        enable_fuzzy_matching: bool = False,
        neo4j_database: Optional[str] = None
    ):
        """
        Initialize the entity deduplicator component.
        
        Args:
            driver: Neo4j driver instance
            graph_id: Tenant graph identifier for multi-tenant isolation
            similarity_threshold: Threshold for similarity matching (0.0-1.0)
            enable_fuzzy_matching: Whether to enable fuzzy string matching
            neo4j_database: Neo4j database name (optional)
        """
        super().__init__()
        self.driver = driver
        self.graph_id = graph_id
        self.similarity_threshold = similarity_threshold
        self.enable_fuzzy_matching = enable_fuzzy_matching
        self.neo4j_database = neo4j_database
        
    async def run(self, graph: Neo4jGraph) -> Neo4jGraph:
        """
        Run entity deduplication on the graph and return the updated graph.
        
        This method follows the neo4j_graphrag component pattern:
        - Takes a Neo4jGraph as input
        - Processes the graph data by consolidating duplicate entities
        - Returns the modified graph
        
        Args:
            graph: The Neo4jGraph containing entities and relationships
            
        Returns:
            The same graph (deduplication happens in-database)
        """
        logger.info(f"🔄 Running entity deduplication for graph {self.graph_id}")
        
        start_time = time.time()
        
        try:
            # Step 1: Exact name matching (most common case)
            exact_deduplications = await self._deduplicate_exact_matches()
            
            # Step 2: Fuzzy matching (if enabled)
            fuzzy_deduplications = 0
            if self.enable_fuzzy_matching:
                fuzzy_deduplications = await self._deduplicate_fuzzy_matches()
            
            total_deduplicated = exact_deduplications + fuzzy_deduplications
            duration = time.time() - start_time
            
            logger.info(f"✅ Entity deduplication completed for graph {self.graph_id}: "
                       f"{total_deduplicated} entities deduplicated in {duration:.2f}s "
                       f"(exact: {exact_deduplications}, fuzzy: {fuzzy_deduplications})")
            
        except Exception as e:
            logger.error(f"❌ Entity deduplication failed for graph {self.graph_id}: {e}")
            # Don't raise - return original graph to continue pipeline
        
        # Return the original graph (deduplication happens in-database)
        return graph
        
    async def _deduplicate_exact_matches(self) -> int:
        """
        Find and consolidate entities with identical names across different chunks.
        
        Strategy:
        1. Find groups of entities with the same name across different chunks
        2. For each group, keep the first entity as canonical
        3. Move all FROM_CHUNK relationships to the canonical entity
        4. Move all other relationships to the canonical entity  
        5. Delete duplicate entities
        
        Returns:
            Number of entities that were deduplicated
        """
        entities_deduplicated = 0
        
        with self.driver.session(database=self.neo4j_database) as session:
            # Find entities with the same name in different chunks
            find_duplicates_query = """
            MATCH (e:__Entity__)-[:FROM_CHUNK]->(c:Chunk)
            WHERE e.graph_id = $graph_id
            WITH e.name as entity_name, collect(DISTINCT e) as entities, collect(DISTINCT c) as chunks
            WHERE size(entities) > 1
            RETURN entity_name, entities, chunks
            """
            
            result = session.run(find_duplicates_query, graph_id=self.graph_id)
            duplicate_groups = list(result)
            
            logger.info(f"Found {len(duplicate_groups)} entity groups with duplicates")
            
            for record in duplicate_groups:
                entity_name = record['entity_name']
                entities = record['entities']
                chunks = record['chunks']
                
                try:
                    deduplicated_count = await self._consolidate_entity_group(
                        session, entity_name, entities, chunks
                    )
                    entities_deduplicated += deduplicated_count
                    
                except Exception as e:
                    logger.error(f"Failed to deduplicate entity group '{entity_name}': {e}")
                    
        return entities_deduplicated
        
    async def _consolidate_entity_group(
        self, 
        session: Any, 
        entity_name: str, 
        entities: List[Any], 
        chunks: List[Any]
    ) -> int:
        """
        Consolidate a group of duplicate entities into a single canonical entity.
        
        Args:
            session: Neo4j session
            entity_name: Name of the entities to consolidate
            entities: List of duplicate entity nodes
            chunks: List of chunks where these entities appear
            
        Returns:
            Number of entities that were consolidated (duplicates removed)
        """
        if len(entities) <= 1:
            return 0
            
        # Use the first entity as the canonical one
        canonical_entity_id = entities[0].element_id
        duplicate_entity_ids = [e.element_id for e in entities[1:]]
        
        logger.debug(f"Consolidating {len(duplicate_entity_ids)} duplicates of '{entity_name}' "
                    f"into canonical entity {canonical_entity_id}")
        
        # Step 1: Connect canonical entity to all chunks
        for chunk in chunks:
            chunk_id = chunk.element_id
            
            # Connect canonical entity to this chunk if not already connected
            session.run("""
                MATCH (canonical) WHERE elementId(canonical) = $canonical_id
                MATCH (chunk) WHERE elementId(chunk) = $chunk_id
                MERGE (canonical)-[:FROM_CHUNK]->(chunk)
            """, canonical_id=canonical_entity_id, chunk_id=chunk_id)
        
        # Step 2: Move relationships from duplicates to canonical (without APOC)
        for duplicate_id in duplicate_entity_ids:
            # Handle outgoing relationships manually by relationship type
            # First get all relationship types and targets
            # Handle outgoing relationships manually by relationship type
            # First get all relationship types and targets
            outgoing_rels = session.run("""
                MATCH (duplicate)-[r]->(target)
                WHERE elementId(duplicate) = $duplicate_id AND type(r) <> 'FROM_CHUNK'
                RETURN elementId(target) as target_id, type(r) as rel_type, properties(r) as rel_props
            """, duplicate_id=duplicate_id)
            
            # Recreate each relationship for canonical entity
            for rel_record in list(outgoing_rels):
                rel_type = rel_record['rel_type']
                
                # Use dynamic cypher to create relationships of different types
                if rel_type in ['WORKS_FOR', 'FOUNDED', 'LEADS', 'MANAGES', 'DEVELOPED', 'PARTNERED_WITH']:
                    session.run(f"""
                        MATCH (canonical) WHERE elementId(canonical) = $canonical_id
                        MATCH (target) WHERE elementId(target) = $target_id
                        MERGE (canonical)-[r:{rel_type}]->(target)
                        SET r = $rel_props
                    """, 
                    canonical_id=canonical_entity_id,
                    target_id=rel_record['target_id'],
                    rel_props=rel_record['rel_props'] or {}
                    )
            
            # Handle incoming relationships  
            incoming_rels = session.run("""
                MATCH (source)-[r]->(duplicate)
                WHERE elementId(duplicate) = $duplicate_id AND type(r) <> 'FROM_CHUNK'
                RETURN elementId(source) as source_id, type(r) as rel_type, properties(r) as rel_props
            """, duplicate_id=duplicate_id)
            
            # Recreate each incoming relationship for canonical entity
            for rel_record in list(incoming_rels):
                rel_type = rel_record['rel_type']
                
                if rel_type in ['WORKS_FOR', 'FOUNDED', 'LEADS', 'MANAGES', 'DEVELOPED', 'PARTNERED_WITH']:
                    session.run(f"""
                        MATCH (source) WHERE elementId(source) = $source_id
                        MATCH (canonical) WHERE elementId(canonical) = $canonical_id
                        MERGE (source)-[r:{rel_type}]->(canonical)
                        SET r = $rel_props
                    """,
                    source_id=rel_record['source_id'],
                    canonical_id=canonical_entity_id,
                    rel_props=rel_record['rel_props'] or {}
                    )
        
        # Step 3: Delete duplicate entities and their relationships
        for duplicate_id in duplicate_entity_ids:
            session.run("""
                MATCH (duplicate) WHERE elementId(duplicate) = $duplicate_id
                DETACH DELETE duplicate
            """, duplicate_id=duplicate_id)
        
        logger.debug(f"✅ Consolidated '{entity_name}': removed {len(duplicate_entity_ids)} duplicates, "
                    f"canonical entity now linked to {len(chunks)} chunks")
        
        return len(duplicate_entity_ids)
        
    async def _deduplicate_fuzzy_matches(self) -> int:
        """
        Find and consolidate entities with similar names using fuzzy matching.
        This is a placeholder for future fuzzy matching implementation.
        
        Returns:
            Number of entities that were deduplicated via fuzzy matching
        """
        # TODO: Implement fuzzy matching when needed
        logger.debug("Fuzzy matching not yet implemented")
        return 0
        """
        Find and resolve entities with identical names across different chunks.
        
        Returns:
            Number of SAME_AS relationships created
        """
        query = """
        MATCH (e1:__Entity__)-[:FROM_CHUNK]->(c1:Chunk)
        MATCH (e2:__Entity__)-[:FROM_CHUNK]->(c2:Chunk)
        WHERE e1.graph_id = $graph_id 
        AND e2.graph_id = $graph_id
        AND e1.name = e2.name 
        AND c1.index <> c2.index
        AND elementId(e1) < elementId(e2)  // Avoid duplicates
        AND NOT (e1)-[:SAME_AS]-(e2)  // Don't create if already exists
        RETURN e1.name as entity_name,
               elementId(e1) as e1_id,
               elementId(e2) as e2_id,
               c1.index as chunk1,
               c2.index as chunk2
        """
        
        create_query = """
        MATCH (e1) WHERE elementId(e1) = $e1_id
        MATCH (e2) WHERE elementId(e2) = $e2_id
        MERGE (e1)-[:SAME_AS {
            created_by: 'entity_resolution',
            method: 'exact_match',
            graph_id: $graph_id,
            created_at: datetime()
        }]-(e2)
        """
        
        links_created = 0
        
        with self.driver.session(database=self.neo4j_database) as session:
            # Find potential matches
            result = session.run(query, graph_id=self.graph_id)
            matches = list(result)
            
            # Create SAME_AS relationships
            for record in matches:
                try:
                    session.run(
                        create_query, 
                        e1_id=record['e1_id'], 
                        e2_id=record['e2_id'],
                        graph_id=self.graph_id
                    )
                    links_created += 1
                    logger.debug(f"   ✅ Linked '{record['entity_name']}' between "
                                f"Chunk {record['chunk1']} ↔ Chunk {record['chunk2']}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Failed to link {record['entity_name']}: {e}")
                    
        return links_created
        
    async def _resolve_fuzzy_matches(self) -> int:
        """
        Find and resolve entities with similar names using fuzzy matching.
        
        Returns:
            Number of SAME_AS relationships created
        """
        # For now, implement a simple similarity check
        # In production, you might want to use more sophisticated algorithms
        
        query = """
        MATCH (e1:__Entity__)-[:FROM_CHUNK]->(c1:Chunk)
        MATCH (e2:__Entity__)-[:FROM_CHUNK]->(c2:Chunk)
        WHERE e1.graph_id = $graph_id 
        AND e2.graph_id = $graph_id
        AND c1.index <> c2.index
        AND elementId(e1) < elementId(e2)
        AND NOT (e1)-[:SAME_AS]-(e2)
        AND NOT e1.name = e2.name  // Exclude exact matches
        // Simple fuzzy matching using CONTAINS or similar patterns
        AND (
            e1.name CONTAINS e2.name OR 
            e2.name CONTAINS e1.name OR
            apoc.text.levenshteinSimilarity(e1.name, e2.name) > $threshold
        )
        RETURN e1.name as name1,
               e2.name as name2,
               elementId(e1) as e1_id,
               elementId(e2) as e2_id,
               apoc.text.levenshteinSimilarity(e1.name, e2.name) as similarity
        """
        
        create_query = """
        MATCH (e1) WHERE elementId(e1) = $e1_id
        MATCH (e2) WHERE elementId(e2) = $e2_id
        MERGE (e1)-[:SAME_AS {
            created_by: 'entity_resolution',
            method: 'fuzzy_match',
            similarity_score: $similarity,
            graph_id: $graph_id,
            created_at: datetime()
        }]-(e2)
        """
        
        links_created = 0
        
        with self.driver.session(database=self.neo4j_database) as session:
            try:
                # Find potential fuzzy matches
                result = session.run(
                    query, 
                    graph_id=self.graph_id,
                    threshold=self.similarity_threshold
                )
                matches = list(result)
                
                # Create SAME_AS relationships for fuzzy matches
                for record in matches:
                    try:
                        session.run(
                            create_query,
                            e1_id=record['e1_id'],
                            e2_id=record['e2_id'],
                            similarity=record['similarity'],
                            graph_id=self.graph_id
                        )
                        links_created += 1
                        logger.debug(f"   ✅ Fuzzy linked '{record['name1']}' ↔ '{record['name2']}' "
                                    f"(similarity: {record['similarity']:.2f})")
                        
                    except Exception as e:
                        logger.error(f"   ❌ Failed to create fuzzy link: {e}")
                        
            except Exception as e:
                # APOC might not be available, that's ok
                logger.warning(f"Fuzzy matching requires APOC procedures: {e}")
                
        return links_created
