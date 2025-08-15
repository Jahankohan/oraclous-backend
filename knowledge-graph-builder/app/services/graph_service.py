import logging
from typing import List, Dict, Any, Optional, Tuple
import asyncio

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import ServiceError
from app.config.settings import get_settings
from app.models.responses import GraphVisualization, GraphNode, GraphRelationship, DuplicateNode
from app.services.embedding_service import EmbeddingService
from app.utils.llm_clients import LLMClientFactory

logger = logging.getLogger(__name__)

class GraphService:
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.llm_factory = LLMClientFactory()
    
    async def get_graph_visualization(
        self, 
        file_names: Optional[List[str]] = None,
        limit: int = 100
    ) -> GraphVisualization:
        """Get graph visualization data"""
        try:
            # Build query based on file filters
            if file_names:
                query = """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ENTITY]->(e:Entity)
                WHERE d.fileName IN $fileNames
                WITH collect(DISTINCT e) as entities
                UNWIND entities as e1
                MATCH (e1)-[r]-(e2)
                WHERE e2 IN entities
                RETURN DISTINCT e1, r, e2
                LIMIT $limit
                """
                params = {"fileNames": file_names, "limit": limit}
            else:
                query = """
                MATCH (e1:Entity)-[r]-(e2:Entity)
                RETURN DISTINCT e1, r, e2
                LIMIT $limit
                """
                params = {"limit": limit}
            
            result = self.neo4j.execute_query(query, params)
            
            nodes = {}
            relationships = []
            
            for record in result:
                # Process nodes
                for node_key in ['e1', 'e2']:
                    node_data = record[node_key]
                    node_id = str(node_data.get('id', node_data.get('element_id')))
                    
                    if node_id not in nodes:
                        nodes[node_id] = GraphNode(
                            id=node_id,
                            labels=['Entity'],  # Default label
                            properties=dict(node_data)
                        )
                
                # Process relationship
                rel_data = record['r']
                relationships.append(GraphRelationship(
                    id=str(rel_data.get('id', rel_data.get('element_id'))),
                    type=rel_data.get('type', 'RELATED'),
                    start_node_id=str(record['e1'].get('id', record['e1'].get('element_id'))),
                    end_node_id=str(record['e2'].get('id', record['e2'].get('element_id'))),
                    properties=dict(rel_data) if hasattr(rel_data, '__iter__') else {}
                ))
            
            return GraphVisualization(
                nodes=list(nodes.values()),
                relationships=relationships
            )
            
        except Exception as e:
            logger.error(f"Error getting graph visualization: {e}")
            raise ServiceError(f"Failed to get graph visualization: {e}")
    
    async def get_node_neighbors(self, node_id: str, depth: int = 1) -> GraphVisualization:
        """Get neighbors of a specific node"""
        try:
            query = f"""
            MATCH (n)-[r*1..{depth}]-(neighbor)
            WHERE n.id = $nodeId OR elementId(n) = $nodeId
            WITH n, r, neighbor
            UNWIND r as rel
            RETURN DISTINCT n, rel, neighbor
            """
            
            result = self.neo4j.execute_query(query, {"nodeId": node_id})
            
            nodes = {}
            relationships = []
            
            for record in result:
                # Add center node
                center_node = record['n']
                center_id = str(center_node.get('id', center_node.get('element_id')))
                
                if center_id not in nodes:
                    nodes[center_id] = GraphNode(
                        id=center_id,
                        labels=['Entity'],
                        properties=dict(center_node)
                    )
                
                # Add neighbor node
                neighbor_node = record['neighbor']
                neighbor_id = str(neighbor_node.get('id', neighbor_node.get('element_id')))
                
                if neighbor_id not in nodes:
                    nodes[neighbor_id] = GraphNode(
                        id=neighbor_id,
                        labels=['Entity'],
                        properties=dict(neighbor_node)
                    )
                
                # Add relationship
                rel = record['rel']
                relationships.append(GraphRelationship(
                    id=str(rel.get('id', rel.get('element_id'))),
                    type=rel.get('type', 'RELATED'),
                    start_node_id=center_id,
                    end_node_id=neighbor_id,
                    properties=dict(rel) if hasattr(rel, '__iter__') else {}
                ))
            
            return GraphVisualization(
                nodes=list(nodes.values()),
                relationships=relationships
            )
            
        except Exception as e:
            logger.error(f"Error getting node neighbors: {e}")
            raise ServiceError(f"Failed to get node neighbors: {e}")
    
    async def delete_documents(self, file_names: List[str], delete_entities: bool = False) -> int:
        """Delete documents and optionally their entities"""
        try:
            if delete_entities:
                # Delete everything related to the documents
                query = """
                MATCH (d:Document)
                WHERE d.fileName IN $fileNames
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e:Entity)
                DETACH DELETE d, c, e
                RETURN count(d) as deletedCount
                """
            else:
                # Delete only documents and chunks, keep entities
                query = """
                MATCH (d:Document)
                WHERE d.fileName IN $fileNames
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                DETACH DELETE d, c
                RETURN count(d) as deletedCount
                """
            
            result = self.neo4j.execute_write_query(query, {"fileNames": file_names})
            return result[0]["deletedCount"] if result else 0
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise ServiceError(f"Failed to delete documents: {e}")
    
    async def get_duplicate_nodes(self) -> List[DuplicateNode]:
        """Find potential duplicate nodes based on similarity"""
        try:
            query = """
            MATCH (e1:Entity), (e2:Entity)
            WHERE e1 <> e2 AND e1.name IS NOT NULL AND e2.name IS NOT NULL
            WITH e1, e2, 
                 gds.similarity.cosine(
                     apoc.text.sorensenDiceSimilarity(toLower(e1.name), toLower(e2.name)),
                     1.0
                 ) as similarity
            WHERE similarity > $threshold
            RETURN e1.id as id1, e1.name as name1, labels(e1) as labels1,
                   e2.id as id2, e2.name as name2, labels(e2) as labels2,
                   similarity
            ORDER BY similarity DESC
            LIMIT 100
            """
            
            result = self.neo4j.execute_query(query, {
                "threshold": self.settings.duplicate_score_threshold
            })
            
            duplicates = []
            processed_pairs = set()
            
            for record in result:
                pair = tuple(sorted([record["id1"], record["id2"]]))
                if pair not in processed_pairs:
                    processed_pairs.add(pair)
                    
                    duplicates.extend([
                        DuplicateNode(
                            id=record["id1"],
                            name=record["name1"],
                            labels=record["labels1"],
                            similarity_score=record["similarity"]
                        ),
                        DuplicateNode(
                            id=record["id2"],
                            name=record["name2"],
                            labels=record["labels2"],
                            similarity_score=record["similarity"]
                        )
                    ])
            
            return duplicates[:50]  # Return top 50 duplicates
            
        except Exception as e:
            logger.error(f"Error finding duplicates: {e}")
            return []
    
    async def merge_duplicate_nodes(self, node_ids: List[str], target_node_id: str) -> None:
        """Merge duplicate nodes into target node"""
        try:
            query = """
            MATCH (target:Entity {id: $targetId})
            MATCH (duplicate:Entity) 
            WHERE duplicate.id IN $duplicateIds AND duplicate <> target
            
            // Merge properties
            SET target += duplicate
            
            // Redirect all relationships from duplicates to target
            MATCH (duplicate)-[r]-(other)
            WHERE NOT (target)-[type(r)]-(other)
            CREATE (target)-[newR:type(r)]->(other)
            SET newR += properties(r)
            
            // Delete duplicate relationships and nodes
            MATCH (duplicate)-[r]-()
            DELETE r
            DELETE duplicate
            
            RETURN count(duplicate) as mergedCount
            """
            
            # Note: This query needs refinement for proper relationship handling
            result = self.neo4j.execute_write_query(query, {
                "targetId": target_node_id,
                "duplicateIds": [nid for nid in node_ids if nid != target_node_id]
            })
            
            logger.info(f"Merged {len(node_ids) - 1} duplicate nodes into {target_node_id}")
            
        except Exception as e:
            logger.error(f"Error merging duplicates: {e}")
            raise ServiceError(f"Failed to merge duplicates: {e}")
    
    async def get_unconnected_nodes(self) -> List[Dict[str, Any]]:
        """Get list of unconnected entity nodes"""
        try:
            query = """
            MATCH (e:Entity)
            WHERE NOT (e)-[]-(:Entity)
            RETURN e.id as id, 
                   e.name as name, 
                   labels(e) as labels,
                   properties(e) as properties
            LIMIT 100
            """
            
            result = self.neo4j.execute_query(query)
            return result
            
        except Exception as e:
            logger.error(f"Error getting unconnected nodes: {e}")
            return []
    
    async def delete_unconnected_nodes(self, node_ids: List[str]) -> int:
        """Delete unconnected entity nodes"""
        try:
            query = """
            MATCH (e:Entity)
            WHERE e.id IN $nodeIds AND NOT (e)-[]-(:Entity)
            DETACH DELETE e
            RETURN count(e) as deletedCount
            """
            
            result = self.neo4j.execute_write_query(query, {"nodeIds": node_ids})
            deleted_count = result[0]["deletedCount"] if result else 0
            
            logger.info(f"Deleted {deleted_count} unconnected nodes")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error deleting unconnected nodes: {e}")
            raise ServiceError(f"Failed to delete unconnected nodes: {e}")
    
    async def post_process_graph(self) -> Dict[str, Any]:
        """Run post-processing tasks on the graph"""
        try:
            results = {}
            
            # 1. Create similarity relationships between chunks
            chunk_similarity_query = """
            MATCH (c1:Chunk), (c2:Chunk)
            WHERE c1 <> c2 AND c1.embedding IS NOT NULL AND c2.embedding IS NOT NULL
            WITH c1, c2, gds.similarity.cosine(c1.embedding, c2.embedding) as similarity
            WHERE similarity > $minScore
            MERGE (c1)-[r:SIMILAR_TO]-(c2)
            SET r.score = similarity
            RETURN count(r) as similarityRelationships
            """
            
            similarity_result = self.neo4j.execute_write_query(chunk_similarity_query, {
                "minScore": self.settings.knn_min_score
            })
            results["chunk_similarities"] = similarity_result[0]["similarityRelationships"] if similarity_result else 0
            
            # 2. Create entity similarity relationships
            entity_similarity_query = """
            MATCH (e1:Entity), (e2:Entity)
            WHERE e1 <> e2 AND e1.name IS NOT NULL AND e2.name IS NOT NULL
            WITH e1, e2, 
                 apoc.text.sorensenDiceSimilarity(toLower(e1.name), toLower(e2.name)) as similarity
            WHERE similarity > $threshold AND similarity < 1.0
            MERGE (e1)-[r:SIMILAR_ENTITY]-(e2)
            SET r.score = similarity
            RETURN count(r) as entitySimilarities
            """
            
            entity_result = self.neo4j.execute_write_query(entity_similarity_query, {
                "threshold": self.settings.duplicate_score_threshold * 0.8  # Lower threshold for similarity
            })
            results["entity_similarities"] = entity_result[0]["entitySimilarities"] if entity_result else 0
            
            # 3. Detect communities (if GDS is available)
            try:
                community_query = """
                CALL gds.louvain.stream('entityGraph')
                YIELD nodeId, communityId
                MATCH (e:Entity) WHERE id(e) = nodeId
                SET e.community = communityId
                RETURN count(e) as entitiesWithCommunities
                """
                
                # This would require a projected graph - placeholder for now
                results["communities"] = 0
                
            except Exception:
                logger.info("Community detection skipped - GDS not available or configured")
                results["communities"] = 0
            
            # 4. Calculate centrality metrics
            try:
                centrality_query = """
                MATCH (e:Entity)-[r]-()
                WITH e, count(r) as degree
                SET e.degree = degree
                RETURN count(e) as entitiesWithDegree
                """
                
                centrality_result = self.neo4j.execute_write_query(centrality_query)
                results["centrality_metrics"] = centrality_result[0]["entitiesWithDegree"] if centrality_result else 0
                
            except Exception as e:
                logger.warning(f"Centrality calculation failed: {e}")
                results["centrality_metrics"] = 0
            
            # 5. Create fulltext indexes
            try:
                fulltext_query = """
                CREATE FULLTEXT INDEX chunkText IF NOT EXISTS 
                FOR (n:Chunk) ON EACH [n.text]
                """
                
                self.neo4j.execute_write_query(fulltext_query)
                results["fulltext_index"] = "created"
                
            except Exception as e:
                logger.warning(f"Fulltext index creation failed: {e}")
                results["fulltext_index"] = "failed"
            
            return results
            
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            raise ServiceError(f"Post-processing failed: {e}")
        
    async def find_duplicate_nodes(self, threshold: float = None) -> List[DuplicateNode]:
        """Find potential duplicate entity nodes"""
        try:
            score_threshold = threshold or self.settings.duplicate_score_threshold
            
            query = """
            MATCH (e1:Entity), (e2:Entity)
            WHERE e1.id < e2.id
            AND (
                toLower(e1.name) = toLower(e2.name)
                OR (
                    e1.embedding IS NOT NULL 
                    AND e2.embedding IS NOT NULL
                    AND gds.similarity.cosine(e1.embedding, e2.embedding) >= $threshold
                )
            )
            RETURN e1.id as id1, 
                   e1.name as name1,
                   e2.id as id2, 
                   e2.name as name2,
                   CASE 
                       WHEN e1.embedding IS NOT NULL AND e2.embedding IS NOT NULL
                       THEN gds.similarity.cosine(e1.embedding, e2.embedding)
                       ELSE 1.0
                   END as similarity
            ORDER BY similarity DESC
            LIMIT 100
            """
            
            result = self.neo4j.execute_query(query, {"threshold": score_threshold})
            
            duplicates = []
            for record in result:
                duplicates.append(DuplicateNode(
                    id=record["id1"],
                    name=record["name1"],
                    labels=["Entity"],
                    similarity_score=record["similarity"]
                ))
                duplicates.append(DuplicateNode(
                    id=record["id2"],
                    name=record["name2"],
                    labels=["Entity"],
                    similarity_score=record["similarity"]
                ))
            
            # Remove duplicates from the list
            seen = set()
            unique_duplicates = []
            for dup in duplicates:
                if dup.id not in seen:
                    seen.add(dup.id)
                    unique_duplicates.append(dup)
            
            return unique_duplicates
            
        except Exception as e:
            logger.error(f"Error finding duplicate nodes: {e}")
            raise ServiceError(f"Failed to find duplicate nodes: {e}")
    
    async def merge_duplicate_nodes(self, node_ids: List[str], target_node_id: str) -> bool:
        """Merge duplicate nodes into a target node"""
        try:
            if target_node_id not in node_ids:
                raise ValueError("Target node ID must be in the list of nodes to merge")
            
            # Get nodes to merge (excluding target)
            source_node_ids = [nid for nid in node_ids if nid != target_node_id]
            
            # Merge relationships
            query = """
            MATCH (target:Entity {id: $targetId})
            UNWIND $sourceIds as sourceId
            MATCH (source:Entity {id: sourceId})
            
            // Copy outgoing relationships
            OPTIONAL MATCH (source)-[r]->(other)
            WHERE NOT (target)-[]->(other)
            FOREACH (rel in CASE WHEN r IS NOT NULL THEN [r] ELSE [] END |
                CREATE (target)-[newRel:RELATED]->(other)
                SET newRel = properties(r)
            )
            
            // Copy incoming relationships
            OPTIONAL MATCH (other)-[r]->(source)
            WHERE NOT (other)-[]->(target)
            FOREACH (rel in CASE WHEN r IS NOT NULL THEN [r] ELSE [] END |
                CREATE (other)-[newRel:RELATED]->(target)
                SET newRel = properties(r)
            )
            
            // Merge properties
            SET target += properties(source)
            
            // Delete source node
            DETACH DELETE source
            
            RETURN count(source) as mergedCount
            """
            
            result = self.neo4j.execute_write_query(query, {
                "targetId": target_node_id,
                "sourceIds": source_node_ids
            })
            
            merged_count = result[0]["mergedCount"] if result else 0
            logger.info(f"Merged {merged_count} nodes into {target_node_id}")
            
            return merged_count > 0
            
        except Exception as e:
            logger.error(f"Error merging duplicate nodes: {e}")
            raise ServiceError(f"Failed to merge duplicate nodes: {e}")
    
    async def post_process_graph(self) -> Dict[str, Any]:
        """Run post-processing tasks on the graph"""
        try:
            results = {}
            
            # 1. Create entity embeddings
            entity_count = await self._create_entity_embeddings()
            results["entities_embedded"] = entity_count
            
            # 2. Create similarity relationships
            similarity_count = await self._create_similarity_relationships()
            results["similarity_relationships"] = similarity_count
            
            # 3. Detect communities
            community_count = await self._detect_communities()
            results["communities_detected"] = community_count
            
            # 4. Generate entity descriptions
            description_count = await self._generate_entity_descriptions()
            results["descriptions_generated"] = description_count
            
            return results
            
        except Exception as e:
            logger.error(f"Error in post-processing: {e}")
            raise ServiceError(f"Post-processing failed: {e}")
    
    async def _create_entity_embeddings(self) -> int:
        """Create embeddings for entities without embeddings"""
        query = """
        MATCH (e:Entity)
        WHERE e.embedding IS NULL AND e.name IS NOT NULL
        RETURN e.id as id, e.name as name
        LIMIT 1000
        """
        
        entities = self.neo4j.execute_query(query)
        
        if not entities:
            return 0
        
        # Generate embeddings in batches
        batch_size = 100
        total_embedded = 0
        
        for i in range(0, len(entities), batch_size):
            batch = entities[i:i + batch_size]
            texts = [e["name"] for e in batch]
            
            embeddings = await self.embedding_service.generate_embeddings(texts)
            
            # Update entities with embeddings
            for entity, embedding in zip(batch, embeddings):
                update_query = """
                MATCH (e:Entity {id: $id})
                SET e.embedding = $embedding
                """
                
                self.neo4j.execute_write_query(update_query, {
                    "id": entity["id"],
                    "embedding": embedding
                })
                
                total_embedded += 1
        
        return total_embedded
    
    async def _create_similarity_relationships(self) -> int:
        """Create similarity relationships between entities"""
        query = """
        MATCH (e1:Entity), (e2:Entity)
        WHERE e1.id < e2.id
        AND e1.embedding IS NOT NULL
        AND e2.embedding IS NOT NULL
        AND NOT (e1)-[:SIMILAR_TO]-(e2)
        WITH e1, e2, gds.similarity.cosine(e1.embedding, e2.embedding) as similarity
        WHERE similarity >= $threshold
        CREATE (e1)-[r:SIMILAR_TO {score: similarity}]->(e2)
        RETURN count(r) as created
        """
        
        result = self.neo4j.execute_write_query(query, {
            "threshold": self.settings.duplicate_score_threshold
        })
        
        return result[0]["created"] if result else 0
    
    async def _detect_communities(self) -> int:
        """Detect communities in the graph"""
        # This would use graph algorithms like Louvain
        # For now, return 0 as placeholder
        return 0
    
    async def _generate_entity_descriptions(self) -> int:
        """Generate descriptions for entities without descriptions"""
        query = """
        MATCH (e:Entity)
        WHERE e.description IS NULL AND e.name IS NOT NULL
        OPTIONAL MATCH (e)<-[:HAS_ENTITY]-(c:Chunk)
        WITH e, collect(c.text)[0..3] as contexts
        WHERE size(contexts) > 0
        RETURN e.id as id, e.name as name, contexts
        LIMIT 100
        """
        
        entities = self.neo4j.execute_query(query)
        
        if not entities:
            return 0
        
        llm = self.llm_factory.get_llm(self.settings.default_llm_model)
        total_generated = 0
        
        for entity in entities:
            # Generate description using LLM
            prompt = f"""Generate a brief description for the entity "{entity['name']}" 
            based on the following contexts:
            
            {' '.join(entity['contexts'][:3])}
            
            Description (one sentence):"""
            
            try:
                description = await llm.generate_completion(prompt, temperature=0.3, max_tokens=100)
                
                # Update entity
                update_query = """
                MATCH (e:Entity {id: $id})
                SET e.description = $description
                """
                
                self.neo4j.execute_write_query(update_query, {
                    "id": entity["id"],
                    "description": description.strip()
                })
                
                total_generated += 1
                
            except Exception as e:
                logger.error(f"Failed to generate description for entity {entity['id']}: {e}")
                continue
        
        return total_generated
    
    async def generate_schema_suggestions(self, text: str, model: str = None) -> Dict[str, List[str]]:
        """Generate schema suggestions from sample text"""
        try:
            llm_model = model or self.settings.default_llm_model
            llm = self.llm_factory.get_llm(llm_model)
            
            prompt = f"""Analyze the following text and suggest:
            1. Entity types (node labels) that should be extracted
            2. Relationship types that connect these entities
            3. Key properties for each entity type
            
            Text: {text[:2000]}
            
            Return the response in this JSON format:
            {{
                "node_labels": ["Person", "Organization", ...],
                "relationship_types": ["WORKS_FOR", "KNOWS", ...],
                "properties": {{
                    "Person": ["name", "age", "role"],
                    "Organization": ["name", "type", "location"]
                }}
            }}
            """
            
            response = await llm.generate_completion(prompt, temperature=0.3)
            
            # Parse response
            import json
            try:
                schema = json.loads(response)
                return schema
            except json.JSONDecodeError:
                # Fallback to basic extraction
                return {
                    "node_labels": ["Entity"],
                    "relationship_types": ["RELATED_TO"],
                    "properties": {"Entity": ["name", "type"]}
                }
                
        except Exception as e:
            logger.error(f"Error generating schema suggestions: {e}")
            raise ServiceError(f"Failed to generate schema suggestions: {e}")
