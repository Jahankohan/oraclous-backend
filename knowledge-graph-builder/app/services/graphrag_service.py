from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import json
import asyncio

from app.core.neo4j_client import neo4j_client
from app.services.search_service import search_service
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.core.logging import get_logger

logger = get_logger(__name__)

class GraphRAGService:
    """Advanced Graph Retrieval-Augmented Generation service"""
    
    def __init__(self):
        self.max_entities = 10
        self.max_chunks = 5
        self.max_depth = 2
    
    async def graph_augmented_retrieval(
        self,
        query: str,
        graph_id: UUID,
        user_id: str,
        retrieval_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive GraphRAG retrieval"""
        
        config = retrieval_config or {}
        max_entities = config.get("max_entities", self.max_entities)
        max_chunks = config.get("max_chunks", self.max_chunks)
        max_depth = config.get("max_depth", self.max_depth)
        
        try:
            # Step 1: Entity extraction from query
            logger.info(f"Starting GraphRAG for query: {query[:50]}...")
            query_entities = await self._extract_query_entities(query, user_id)
            
            # Step 2: Multi-modal retrieval
            retrieval_tasks = await self._execute_parallel_retrieval(
                query, graph_id, max_entities, max_chunks
            )
            
            # Step 3: Entity neighborhood expansion
            neighborhoods = await self._expand_entity_neighborhoods(
                retrieval_tasks["similar_entities"], graph_id, max_depth
            )
            
            # Step 4: Relationship path discovery
            paths = await self._discover_relationship_paths(
                query_entities, retrieval_tasks["similar_entities"], graph_id
            )
            
            # Step 5: Context construction
            context = self._construct_rich_context(
                query_entities, 
                retrieval_tasks, 
                neighborhoods, 
                paths
            )
            
            return {
                "query": query,
                "query_entities": query_entities,
                "similar_entities": retrieval_tasks["similar_entities"],
                "text_chunks": retrieval_tasks["text_chunks"],
                "neighborhoods": neighborhoods,
                "paths": paths,
                "context": context,
                "metadata": {
                    "entities_found": len(retrieval_tasks["similar_entities"]),
                    "chunks_found": len(retrieval_tasks["text_chunks"]),
                    "neighborhoods_expanded": len(neighborhoods),
                    "paths_discovered": len(paths)
                }
            }
            
        except Exception as e:
            logger.error(f"GraphRAG retrieval failed: {e}")
            return {
                "query": query,
                "error": str(e),
                "context": "GraphRAG retrieval failed. Using fallback context.",
                "metadata": {"status": "failed"}
            }
    
    async def _extract_query_entities(self, query: str, user_id: str) -> List[str]:
        """Extract entities mentioned in the user query"""
        
        try:
            if not llm_service.is_initialized():
                await llm_service.initialize_llm(user_id, "openai")
            
            entity_prompt = f"""
            Extract named entities (people, organizations, places, concepts, products) from this question:
            "{query}"
            
            Return only a JSON array of entity names. If no clear entities, return [].
            Examples:
            - "Who is John Doe?" → ["John Doe"]
            - "What does OpenAI do?" → ["OpenAI"]
            - "How are Tesla and SpaceX related?" → ["Tesla", "SpaceX"]
            
            Response:
            """
            
            response = await llm_service.llm.ainvoke(entity_prompt)
            entities = json.loads(response.content.strip())
            
            logger.info(f"Extracted entities from query: {entities}")
            return entities if isinstance(entities, list) else []
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    async def _execute_parallel_retrieval(
        self,
        query: str,
        graph_id: UUID,
        max_entities: int,
        max_chunks: int
    ) -> Dict[str, List]:
        """Execute multiple retrieval methods in parallel"""
        
        try:
            # Create parallel tasks
            tasks = {}
            
            # Entity similarity search
            if embedding_service.is_initialized():
                tasks["similar_entities"] = search_service.similarity_search_entities(
                    query=query,
                    graph_id=graph_id,
                    k=max_entities,
                    threshold=0.6
                )
            
            # Text chunk search
            if embedding_service.is_initialized():
                tasks["text_chunks"] = search_service.similarity_search_chunks(
                    query=query,
                    graph_id=graph_id,
                    k=max_chunks,
                    threshold=0.6
                )
            
            # Keyword search fallback
            tasks["keyword_entities"] = search_service.fulltext_search_entities(
                query=query,
                graph_id=graph_id,
                limit=max_entities // 2
            )
            
            # Execute all tasks
            results = {}
            for task_name, task in tasks.items():
                try:
                    results[task_name] = await task
                except Exception as e:
                    logger.warning(f"Task {task_name} failed: {e}")
                    results[task_name] = []
            
            # Merge entity results
            all_entities = results.get("similar_entities", []) + results.get("keyword_entities", [])
            results["similar_entities"] = self._deduplicate_entities(all_entities)[:max_entities]
            
            return results
            
        except Exception as e:
            logger.error(f"Parallel retrieval failed: {e}")
            return {"similar_entities": [], "text_chunks": [], "keyword_entities": []}
    
    async def _expand_entity_neighborhoods(
        self,
        entities: List[Dict],
        graph_id: UUID,
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Expand entity neighborhoods to get related context"""
        
        neighborhoods = []
        
        for entity in entities[:5]:  # Limit to top 5 entities
            try:
                neighborhood = await self._get_entity_neighborhood_detailed(
                    entity["id"], graph_id, max_depth
                )
                if neighborhood:
                    neighborhoods.append({
                        "center_entity": entity,
                        "neighborhood": neighborhood
                    })
                    
            except Exception as e:
                logger.warning(f"Neighborhood expansion failed for {entity.get('id')}: {e}")
        
        return neighborhoods
    
    async def _get_entity_neighborhood_detailed(
        self,
        entity_id: str,
        graph_id: UUID,
        max_depth: int
    ) -> Dict[str, Any]:
        """Get detailed neighborhood information for an entity"""
        
        try:
            query = f"""
            MATCH (center {{id: $entity_id, graph_id: $graph_id}})
            CALL {{
                WITH center
                MATCH path = (center)-[*1..{max_depth}]-(neighbor)
                WHERE neighbor.graph_id = $graph_id
                WITH neighbor, relationships(path) as path_rels
                RETURN DISTINCT neighbor.id as neighbor_id,
                       neighbor.name as neighbor_name,
                       labels(neighbor) as neighbor_labels,
                       [r in path_rels | type(r)] as relationship_types,
                       length(path_rels) as distance
                ORDER BY distance, neighbor_name
                LIMIT 15
            }}
            RETURN collect({{
                id: neighbor_id,
                name: neighbor_name,
                labels: neighbor_labels,
                relationships: relationship_types,
                distance: distance
            }}) as neighbors
            """
            
            result = await neo4j_client.execute_query(query, {
                "entity_id": entity_id,
                "graph_id": str(graph_id)
            })
            
            return result[0]["neighbors"] if result else []
            
        except Exception as e:
            logger.warning(f"Detailed neighborhood query failed: {e}")
            return []
    
    async def _discover_relationship_paths(
        self,
        query_entities: List[str],
        similar_entities: List[Dict],
        graph_id: UUID
    ) -> List[Dict[str, Any]]:
        """Discover interesting relationship paths between entities"""
        
        paths = []
        all_entity_names = query_entities + [e.get("name", "") for e in similar_entities]
        
        # Find paths between pairs of entities
        for i, entity1 in enumerate(all_entity_names[:5]):
            for entity2 in all_entity_names[i+1:6]:  # Limit combinations
                try:
                    path = await self._find_shortest_path(entity1, entity2, graph_id)
                    if path:
                        paths.append(path)
                        
                except Exception as e:
                    logger.debug(f"Path discovery failed between {entity1} and {entity2}: {e}")
        
        return paths[:5]  # Return top 5 paths
    
    async def _find_shortest_path(
        self,
        entity1_name: str,
        entity2_name: str,
        graph_id: UUID
    ) -> Optional[Dict[str, Any]]:
        """Find shortest path between two entities"""
        
        try:
            query = """
            MATCH (start {graph_id: $graph_id})
            WHERE toLower(start.name) CONTAINS toLower($entity1)
            MATCH (end {graph_id: $graph_id})
            WHERE toLower(end.name) CONTAINS toLower($entity2)
            MATCH path = shortestPath((start)-[*1..4]-(end))
            WHERE start <> end
            RETURN nodes(path) as path_nodes,
                   relationships(path) as path_relationships,
                   length(path) as path_length
            LIMIT 1
            """
            
            result = await neo4j_client.execute_query(query, {
                "entity1": entity1_name,
                "entity2": entity2_name,
                "graph_id": str(graph_id)
            })
            
            if result:
                path_data = result[0]
                return {
                    "start_entity": entity1_name,
                    "end_entity": entity2_name,
                    "path_length": path_data["path_length"],
                    "nodes": [
                        {"id": node.get("id"), "name": node.get("name")} 
                        for node in path_data["path_nodes"]
                    ],
                    "relationships": [
                        {"type": rel.type, "properties": dict(rel)}
                        for rel in path_data["path_relationships"]
                    ]
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Shortest path query failed: {e}")
            return None
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities based on ID or name"""
        
        seen_ids = set()
        seen_names = set()
        deduplicated = []
        
        for entity in entities:
            entity_id = entity.get("id", "")
            entity_name = entity.get("name", "").lower()
            
            if entity_id and entity_id not in seen_ids:
                seen_ids.add(entity_id)
                deduplicated.append(entity)
            elif entity_name and entity_name not in seen_names and not entity_id:
                seen_names.add(entity_name)
                deduplicated.append(entity)
        
        return deduplicated
    
    def _construct_rich_context(
        self,
        query_entities: List[str],
        retrieval_results: Dict[str, List],
        neighborhoods: List[Dict],
        paths: List[Dict]
    ) -> str:
        """Construct comprehensive context for LLM"""
        
        context_sections = []
        
        # Query entities section
        if query_entities:
            context_sections.append(f"Query mentions: {', '.join(query_entities)}")
        
        # Similar entities section
        similar_entities = retrieval_results.get("similar_entities", [])
        if similar_entities:
            entities_info = []
            for entity in similar_entities[:8]:
                score = entity.get("score", 0)
                name = entity.get("name", "Unknown")
                entity_type = entity.get("labels", ["Unknown"])[0] if entity.get("labels") else "Unknown"
                entities_info.append(f"- {name} ({entity_type}, relevance: {score:.2f})")
            
            context_sections.append(f"Relevant entities:\n" + "\n".join(entities_info))
        
        # Text chunks section
        text_chunks = retrieval_results.get("text_chunks", [])
        if text_chunks:
            chunks_info = []
            for chunk in text_chunks[:3]:
                chunk_text = chunk.get("text", "")[:200]
                chunks_info.append(f"- {chunk_text}...")
            
            context_sections.append(f"Relevant text passages:\n" + "\n".join(chunks_info))
        
        # Neighborhoods section
        if neighborhoods:
            neighborhood_info = []
            for neighborhood in neighborhoods[:3]:
                center = neighborhood["center_entity"]["name"]
                neighbors = neighborhood["neighborhood"][:5]
                neighbor_names = [n.get("name", "Unknown") for n in neighbors]
                neighborhood_info.append(f"- {center} is connected to: {', '.join(neighbor_names)}")
            
            context_sections.append(f"Entity relationships:\n" + "\n".join(neighborhood_info))
        
        # Paths section
        if paths:
            path_info = []
            for path in paths[:3]:
                start = path["start_entity"]
                end = path["end_entity"]
                length = path["path_length"]
                path_info.append(f"- {start} → {end} (path length: {length})")
            
            context_sections.append(f"Entity connections:\n" + "\n".join(path_info))
        
        return "\n\n".join(context_sections)
    
    async def generate_graphrag_answer(
        self,
        query: str,
        retrieval_result: Dict[str, Any],
        user_id: str
    ) -> str:
        """Generate final answer using GraphRAG context"""
        
        try:
            if not llm_service.is_initialized():
                await llm_service.initialize_llm(user_id, "openai")
            
            context = retrieval_result.get("context", "")
            metadata = retrieval_result.get("metadata", {})
            
            graphrag_prompt = f"""
            You are an AI assistant with access to a comprehensive knowledge graph. Use the provided context to answer the user's question thoroughly.
            
            Question: {query}
            
            Knowledge Graph Context:
            {context}
            
            Context Statistics:
            - Entities found: {metadata.get('entities_found', 0)}
            - Text chunks: {metadata.get('chunks_found', 0)}
            - Neighborhoods explored: {metadata.get('neighborhoods_expanded', 0)}
            - Connection paths: {metadata.get('paths_discovered', 0)}
            
            Instructions:
            1. Provide a comprehensive answer using the graph context
            2. Mention specific entities and relationships when relevant
            3. If the context provides multiple perspectives, synthesize them
            4. Be clear about the confidence level of your answer
            5. If information is incomplete, acknowledge what's missing
            
            Answer:
            """
            
            response = await llm_service.llm.ainvoke(graphrag_prompt)
            return response.content
            
        except Exception as e:
            logger.error(f"GraphRAG answer generation failed: {e}")
            return f"I found relevant information but encountered an error generating the answer: {str(e)}"

# Global GraphRAG service
graphrag_service = GraphRAGService()
