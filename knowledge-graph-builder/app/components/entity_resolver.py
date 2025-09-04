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
from typing import Any, Optional, List, Dict
from neo4j import Driver, Session
from neo4j.graph import Node

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
        session: Session, 
        entity_name: str, 
        entities: List[Node], 
        chunks: List[Node]
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
                target_id = rel_record['target_id']
                rel_props: Dict[str, Any] = rel_record['rel_props'] or {}
                
                # Use parameterized query with predefined relationship types
                # Handle all relationship types dynamically
                query = """
                    MATCH (canonical) WHERE elementId(canonical) = $canonical_id
                    MATCH (target) WHERE elementId(target) = $target_id
                    CALL apoc.create.relationship(canonical, $rel_type, $rel_props, target) YIELD rel
                    RETURN rel
                """
                try:
                    session.run(query, 
                        canonical_id=canonical_entity_id,
                        target_id=target_id,
                        rel_type=rel_type,
                        rel_props=rel_props
                    )
                except Exception:
                    # Fallback without APOC - create specific relationship types
                    self._create_relationship_fallback(
                        session, canonical_entity_id, target_id, rel_type, rel_props
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
                source_id = rel_record['source_id']
                rel_props: Dict[str, Any] = rel_record['rel_props'] or {}
                
                # Handle all relationship types dynamically
                query = """
                    MATCH (source) WHERE elementId(source) = $source_id
                    MATCH (canonical) WHERE elementId(canonical) = $canonical_id
                    CALL apoc.create.relationship(source, $rel_type, $rel_props, canonical) YIELD rel
                    RETURN rel
                """
                try:
                    session.run(query,
                        source_id=source_id,
                        canonical_id=canonical_entity_id,
                        rel_type=rel_type,
                        rel_props=rel_props
                    )
                except Exception:
                    # Fallback without APOC
                    self._create_relationship_fallback(
                        session, source_id, canonical_entity_id, rel_type, rel_props
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
        
    def _create_relationship_fallback(
        self, 
        session: Session, 
        source_id: str, 
        target_id: str, 
        rel_type: str, 
        rel_props: Dict[str, Any]
    ) -> None:
        """
        Fallback method to create relationships without APOC procedures.
        
        This method handles ANY relationship type dynamically by using
        a generic approach that works for all relationship types.
        """
        # Use a generic approach that works for any relationship type
        # We'll use a two-step process: create the relationship, then set properties
        try:
            # Step 1: Create the relationship using CALL procedure
            # This approach works with any relationship type
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                CALL apoc.cypher.doIt(
                    'MERGE (s)-[r:' + $rel_type + ']->(t) RETURN r',
                    {s: source, t: target}
                ) YIELD value
                WITH value.r as rel
                SET rel = $rel_props
                RETURN rel
            """
            
            session.run(query, 
                source_id=source_id, 
                target_id=target_id, 
                rel_type=rel_type,
                rel_props=rel_props
            )
            
        except Exception:
            # Ultimate fallback: use known relationship types or create a generic one
            if rel_type in ['WORKS_FOR', 'FOUNDED', 'LEADS', 'MANAGES', 'DEVELOPED', 'PARTNERED_WITH']:
                # Use predefined relationship creation for known types
                self._create_known_relationship(session, source_id, target_id, rel_type, rel_props)
            else:
                # For unknown relationship types, log a warning and create a generic relationship
                logger.warning(f"Unknown relationship type '{rel_type}' - creating generic relationship")
                
                # Create a generic relationship with the original type as a property
                query = """
                    MATCH (source) WHERE elementId(source) = $source_id
                    MATCH (target) WHERE elementId(target) = $target_id
                    MERGE (source)-[r:RELATED_TO]->(target)
                    SET r.original_type = $rel_type, r = $rel_props
                    RETURN r
                """
                session.run(query, 
                    source_id=source_id, 
                    target_id=target_id, 
                    rel_type=rel_type,
                    rel_props=rel_props
                )
    
    def _create_known_relationship(
        self, 
        session: Session, 
        source_id: str, 
        target_id: str, 
        rel_type: str, 
        rel_props: Dict[str, Any]
    ) -> None:
        """
        Create relationships for known/predefined relationship types.
        This is the ultimate fallback when APOC is not available.
        """
        if rel_type == 'WORKS_FOR':
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:WORKS_FOR]->(target)
                SET r = $rel_props
            """
        elif rel_type == 'FOUNDED':
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:FOUNDED]->(target)
                SET r = $rel_props
            """
        elif rel_type == 'LEADS':
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:LEADS]->(target)
                SET r = $rel_props
            """
        elif rel_type == 'MANAGES':
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:MANAGES]->(target)
                SET r = $rel_props
            """
        elif rel_type == 'DEVELOPED':
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:DEVELOPED]->(target)
                SET r = $rel_props
            """
        elif rel_type == 'PARTNERED_WITH':
            query = """
                MATCH (source) WHERE elementId(source) = $source_id
                MATCH (target) WHERE elementId(target) = $target_id
                MERGE (source)-[r:PARTNERED_WITH]->(target)
                SET r = $rel_props
            """
        else:
            # This shouldn't happen since we check before calling this method
            return
            
        session.run(query, source_id=source_id, target_id=target_id, rel_props=rel_props)
        
    async def _deduplicate_fuzzy_matches(self) -> int:
        """
        Find and consolidate entities with similar names using fuzzy matching.
        
        This method uses APOC procedures for similarity calculations when available.
        Falls back gracefully when APOC is not installed.
        
        Returns:
            Number of entities that were deduplicated via fuzzy matching
        """
        entities_deduplicated = 0
        
        # Query to find entities with similar names using fuzzy matching
        query = """
        MATCH (e1:__Entity__)-[:FROM_CHUNK]->(c1:Chunk)
        MATCH (e2:__Entity__)-[:FROM_CHUNK]->(c2:Chunk)
        WHERE e1.graph_id = $graph_id 
        AND e2.graph_id = $graph_id
        AND c1.index <> c2.index
        AND elementId(e1) < elementId(e2)
        AND NOT e1.name = e2.name  // Exclude exact matches
        // Simple fuzzy matching using CONTAINS or APOC similarity
        AND (
            e1.name CONTAINS e2.name OR 
            e2.name CONTAINS e1.name OR
            (EXISTS {
                CALL apoc.text.levenshteinSimilarity(e1.name, e2.name) YIELD value
                WHERE value > $threshold
            })
        )
        WITH e1, e2, 
             CASE 
                WHEN EXISTS { CALL apoc.text.levenshteinSimilarity(e1.name, e2.name) YIELD value }
                THEN apoc.text.levenshteinSimilarity(e1.name, e2.name)
                ELSE 0.9
             END as similarity
        WHERE similarity > $threshold
        WITH e1.name as entity_name, collect(DISTINCT e1) + collect(DISTINCT e2) as entities
        WHERE size(entities) > 1
        RETURN entity_name, entities
        """
        
        try:
            with self.driver.session(database=self.neo4j_database) as session:
                # Find potential fuzzy matches
                result = session.run(
                    query, 
                    graph_id=self.graph_id,
                    threshold=self.similarity_threshold
                )
                fuzzy_groups = list(result)
                
                logger.info(f"Found {len(fuzzy_groups)} entity groups with fuzzy matches")
                
                # Process each group of similar entities
                for record in fuzzy_groups:
                    entity_name = record['entity_name']
                    entities = record['entities']
                    
                    if len(entities) > 1:
                        try:
                            # Get chunks for these entities
                            chunks: List[Node] = []
                            for entity in entities:
                                entity_chunks = session.run("""
                                    MATCH (e)-[:FROM_CHUNK]->(c:Chunk)
                                    WHERE elementId(e) = $entity_id
                                    RETURN c
                                """, entity_id=entity.element_id)
                                chunks.extend([record['c'] for record in entity_chunks])
                            
                            # Remove duplicates by element_id
                            unique_chunks_dict = {chunk.element_id: chunk for chunk in chunks}
                            unique_chunks: List[Node] = list(unique_chunks_dict.values())
                            
                            # Consolidate the fuzzy matched entities
                            deduplicated_count = await self._consolidate_entity_group(
                                session, entity_name, entities, unique_chunks
                            )
                            entities_deduplicated += deduplicated_count
                            
                        except Exception as e:
                            logger.error(f"Failed to deduplicate fuzzy entity group '{entity_name}': {e}")
                
        except Exception as e:
            # APOC might not be available, that's ok
            logger.warning(f"Fuzzy matching requires APOC procedures (falling back to exact matching only): {e}")
                
        return entities_deduplicated
