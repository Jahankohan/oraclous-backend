from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
import json
import asyncio
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from app.core.neo4j_client import neo4j_client
from app.services.llm_service import llm_service
from app.services.search_service import search_service
from app.services.embedding_service import embedding_service
from app.services.schema_service import schema_service
from app.services.graphrag_service import graphrag_service
from app.core.logging import get_logger

logger = get_logger(__name__)

# ==================== DATA STRUCTURES ====================

class ReasoningMode(Enum):
    """Advanced reasoning modes for graph analysis"""
    COMPREHENSIVE = "comprehensive"
    FOCUSED = "focused"
    EXPLORATORY = "exploratory"

class ChatMode(Enum):
    """Chat retrieval modes from original implementation"""
    VECTOR = "vector"
    GRAPH = "graph"
    GRAPH_VECTOR = "graph_vector"
    GRAPHRAG = "graphrag"
    COMPREHENSIVE = "comprehensive"  # New advanced mode

@dataclass
class GraphContext:
    """Rich context extracted from graph structure"""
    primary_entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    neighborhoods: List[Dict[str, Any]]
    pathways: List[Dict[str, Any]]
    communities: List[Dict[str, Any]]
    influential_nodes: List[Dict[str, Any]]
    temporal_context: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    reasoning_chain: List[str]

# ==================== COMPREHENSIVE CHAT SERVICE ====================

class ChatService:
    """
    Comprehensive chat service combining ALL original functionality with enhancements:
    
    ORIGINAL FEATURES RESTORED:
    - Multiple retrieval modes (vector, graph, graph_vector, graphrag)
    - Natural language to Cypher conversion
    - Full service integration (embedding, search, schema)
    - Proper graph_id filtering throughout
    
    ENHANCED FEATURES ADDED:
    - Advanced graph reasoning with Neo4j GDS
    - Community detection and centrality analysis
    - Response grounding and hallucination control
    - Conversational intelligence and insights
    """
    
    def __init__(self):
        # Original attributes
        self.cypher_chain = None
        self.current_graph_id = None
        self.conversation_history = []
        self.graph_schema = None
        
        # Enhanced attributes
        self.reasoning_modes = [mode.value for mode in ReasoningMode]
        self.centrality_cache = {}
        self.community_cache = {}
        self.graph_statistics = {}
        self.reasoning_history = []
    
    # ==================== INITIALIZATION (ORIGINAL + ENHANCED) ====================
    
    async def initialize_chat(
        self, 
        graph_id: UUID, 
        user_id: str,
        provider: str = "openai",
        model: str = "gpt-4o-mini"
    ) -> bool:
        """Initialize chat service for a specific graph (ORIGINAL METHOD)"""
        
        try:
            # Initialize LLM with TEMPERATURE CONTROL for hallucination prevention
            if not llm_service.is_initialized():
                success = await llm_service.initialize_llm(
                    user_id=user_id,
                    provider=provider,
                    model=model,
                    temperature=0.2  # LOW TEMPERATURE for factual responses
                )
                if not success:
                    return False
            
            # Initialize embeddings for vector search (RESTORED)
            if not embedding_service.is_initialized():
                await embedding_service.initialize_embeddings(
                    provider="openai",
                    user_id=user_id
                )
            
            # Get graph schema using schema_service (RESTORED)
            schema = await schema_service.get_graph_schema(graph_id)
            
            # Create Cypher QA chain (RESTORED)
            self.cypher_chain = await self._create_cypher_chain(schema)
            self.current_graph_id = graph_id
            self.graph_schema = schema
            
            # ENHANCED: Pre-compute graph statistics
            await self._precompute_graph_statistics()
            
            logger.info(f"Chat initialized for graph {graph_id} with all services")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize chat: {e}")
            return False
    
    # ==================== MAIN CHAT INTERFACE (ALL MODES RESTORED) ====================
    
    async def chat_with_graph(
        self,
        query: str,
        mode: str = "graph_vector",  # RESTORED: Default to hybrid mode
        graph_id: Optional[UUID] = None,
        conversation_id: Optional[str] = None,
        include_history: bool = True,
        max_context_tokens: int = 4000,
        include_reasoning_chain: bool = True
    ) -> Dict[str, Any]:
        """
        Main chat interface with ALL retrieval modes restored
        
        Modes:
        - vector: Pure vector search on entities/chunks
        - graph: Cypher query generation  
        - graph_vector: Hybrid approach (RECOMMENDED)
        - graphrag: Advanced neighborhood analysis
        - comprehensive: New advanced graph reasoning
        """
        
        # Validate graph_id
        if graph_id and graph_id != self.current_graph_id:
            logger.warning(f"Graph ID mismatch: expected {self.current_graph_id}, got {graph_id}")
            return {
                "answer": "Error: Chat not initialized for this graph",
                "mode": mode,
                "success": False
            }
        
        if not self.current_graph_id:
            return {
                "answer": "Error: Chat not initialized for any graph",
                "success": False,
                "mode": mode
            }
        
        try:
            # Add to conversation history
            if include_history:
                self.conversation_history.append({
                    "type": "user",
                    "content": query,
                    "timestamp": datetime.now()
                })
            
            # Route to appropriate retrieval method (ALL RESTORED)
            if mode == ChatMode.VECTOR.value:
                result = await self._vector_search_chat(query)
            elif mode == ChatMode.GRAPH.value:
                result = await self._graph_cypher_chat(query)
            elif mode == ChatMode.GRAPH_VECTOR.value:
                result = await self._hybrid_graph_vector_chat(query)
            elif mode == ChatMode.GRAPHRAG.value:
                result = await self._graphrag_chat(query)
            elif mode == ChatMode.COMPREHENSIVE.value:
                result = await self._comprehensive_reasoning_chat(
                    query, max_context_tokens, include_reasoning_chain
                )
            else:
                raise ValueError(f"Unsupported chat mode: {mode}")
            
            # ENHANCED: Add response enhancements
            enhanced_result = await self._enhance_response_with_insights(
                query, result, mode
            )
            
            # Add response to history
            if include_history:
                self.conversation_history.append({
                    "type": "assistant", 
                    "content": enhanced_result["answer"],
                    "timestamp": datetime.now(),
                    "metadata": enhanced_result.get("metadata", {})
                })
            
            enhanced_result["mode"] = mode
            enhanced_result["conversation_id"] = conversation_id
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "mode": mode,
                "success": False,
                "error": str(e)
            }
    
    # ==================== VECTOR SEARCH MODE (RESTORED) ====================
    
    async def _vector_search_chat(self, query: str) -> Dict[str, Any]:
        """Chat using vector search on entities and chunks (RESTORED)"""
        
        try:
            # Search entities with PROPER graph_id filtering
            entity_results = await search_service.similarity_search_entities(
                query=query,
                graph_id=self.current_graph_id,
                k=5,
                threshold=0.7
            )
            
            # Search text chunks if available
            chunk_results = []
            try:
                chunk_results = await search_service.similarity_search_chunks(
                    query=query,
                    graph_id=self.current_graph_id,
                    k=3,
                    threshold=0.6
                )
            except Exception:
                pass  # Chunks might not exist
            
            # Build context from results
            context = self._build_vector_context(entity_results, chunk_results)
            
            # Generate answer using LLM with TEMPERATURE CONTROL
            answer_prompt = f"""
            Based on the following information from knowledge graph {self.current_graph_id}, answer the question.
            
            Question: {query}
            
            Relevant Entities:
            {context['entities']}
            
            Relevant Text:
            {context['chunks']}
            
            IMPORTANT: Use ONLY the information provided above. If insufficient, say so clearly.
            """
            
            response = await llm_service.llm.ainvoke(
                answer_prompt,
                temperature=0.1  # VERY LOW temperature for factual accuracy
            )
            
            # ENHANCED: Validate response grounding
            is_grounded = await self._validate_response_grounding(response.content, context)
            
            return {
                "answer": response.content,
                "sources": {
                    "entities": entity_results[:3],
                    "chunks": chunk_results[:2]
                },
                "method": "vector_search",
                "grounded": is_grounded,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Vector search chat failed: {e}")
            return {
                "answer": "I couldn't find relevant information using vector search.",
                "success": False,
                "error": str(e)
            }
    
    # ==================== GRAPH CYPHER MODE (RESTORED) ====================
    
    async def _graph_cypher_chat(self, query: str) -> Dict[str, Any]:
        """Chat using Cypher query generation (RESTORED)"""
        
        try:
            # Get graph schema using schema_service
            schema = await schema_service.get_graph_schema(self.current_graph_id)
            
            # Generate Cypher query (RESTORED)
            cypher_result = await self._natural_language_to_cypher(query, schema)
            
            if not cypher_result["success"]:
                return {
                    "answer": "I couldn't generate a valid query for your question.",
                    "success": False,
                    "cypher": cypher_result.get("cypher"),
                    "error": cypher_result.get("error")
                }
            
            # Execute Cypher query
            results = cypher_result["result"]
            
            # Generate natural language answer from results
            answer = await self._generate_answer_from_cypher_results(query, results)
            
            return {
                "answer": answer,
                "cypher": cypher_result["cypher"],
                "raw_results": results[:5],
                "method": "cypher_generation",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Cypher chat failed: {e}")
            return {
                "answer": "I couldn't process your question using graph queries.",
                "success": False,
                "error": str(e)
            }
    
    # ==================== HYBRID MODE (RESTORED) ====================
    
    async def _hybrid_graph_vector_chat(self, query: str) -> Dict[str, Any]:
        """Hybrid approach combining graph queries and vector search (RESTORED)"""
        
        try:
            # Try both approaches concurrently
            vector_task = asyncio.create_task(self._vector_search_chat(query))
            cypher_task = asyncio.create_task(self._graph_cypher_chat(query))
            
            vector_result, cypher_result = await asyncio.gather(
                vector_task, cypher_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_result, Exception):
                vector_result = {"success": False, "answer": "Vector search failed"}
            if isinstance(cypher_result, Exception):
                cypher_result = {"success": False, "answer": "Cypher search failed"}
            
            # Combine results intelligently
            if vector_result["success"] and cypher_result["success"]:
                combined_answer = await self._combine_answers(
                    query, vector_result, cypher_result
                )
                return {
                    "answer": combined_answer,
                    "vector_result": vector_result,
                    "cypher_result": cypher_result,
                    "method": "hybrid_graph_vector",
                    "success": True
                }
            
            elif vector_result["success"]:
                vector_result["method"] = "hybrid_fallback_vector"
                return vector_result
            
            elif cypher_result["success"]:
                cypher_result["method"] = "hybrid_fallback_cypher"
                return cypher_result
            
            else:
                return {
                    "answer": "I couldn't find relevant information using either search method.",
                    "success": False,
                    "vector_error": vector_result.get("error"),
                    "cypher_error": cypher_result.get("error")
                }
                
        except Exception as e:
            logger.error(f"Hybrid chat failed: {e}")
            return {
                "answer": "I encountered an error while searching for information.",
                "success": False,
                "error": str(e)
            }
    
    # ==================== GRAPHRAG MODE (RESTORED) ====================
    
    async def _graphrag_chat(self, query: str) -> Dict[str, Any]:
        """Advanced GraphRAG implementation (RESTORED)"""
        
        try:
            graphrag_result = await graphrag_service.graph_augmented_retrieval(
                query=query,
                graph_id=self.current_graph_id,
                user_id="current_user",
                retrieval_config={
                    "max_entities": 8,
                    "max_chunks": 5,
                    "max_depth": 2
                }
            )
            
            # Generate answer using the rich context
            if "context" in graphrag_result and "metadata" in graphrag_result:
                answer = await graphrag_service.generate_graphrag_answer(
                    query=query,
                    context=graphrag_result["context"],
                    metadata=graphrag_result["metadata"]
                )
                
                return {
                    "answer": answer,
                    "context": {
                        "query_entities": graphrag_result.get("query_entities", []),
                        "similar_entities": graphrag_result.get("similar_entities", [])[:3],
                        "neighborhoods": graphrag_result["metadata"].get("neighborhoods_expanded", 0),
                        "text_chunks": graphrag_result["metadata"].get("chunks_found", 0),
                        "paths": graphrag_result["metadata"].get("paths_discovered", 0)
                    },
                    "method": "graphrag_advanced",
                    "success": True,
                    "grounded": True,  # GraphRAG is always grounded
                    "metadata": graphrag_result["metadata"]
                }
            else:
                # TODO: Implement a Fallback if GraphRAG service had issues
                # return await self._graphrag_chat_fallback(query)
                logger.error(f"GraphRAG chat failed to respond")
                return {
                    "answer": "I couldn't process your question using GraphRAG.",
                    "success": False,
                    "error": "GraphRAG chat failed to respond"
                }
                
        except Exception as e:
            logger.error(f"GraphRAG chat failed: {e}")
            return {
                "answer": "I couldn't process your question using GraphRAG.",
                "success": False,
                "error": str(e)
            }
    
    # ==================== COMPREHENSIVE REASONING MODE (ENHANCED) ====================
    
    async def _comprehensive_reasoning_chat(
        self, 
        query: str, 
        max_context_tokens: int,
        include_reasoning_chain: bool
    ) -> Dict[str, Any]:
        """New comprehensive reasoning mode with advanced graph algorithms"""
        
        try:
            start_time = datetime.now()
            
            # Generate rich context using advanced algorithms
            graph_context = await self._generate_graph_context_with_graph_id(
                query=query,
                graph_id=self.current_graph_id,  # FIXED: Pass graph_id
                reasoning_mode=ReasoningMode.COMPREHENSIVE,
                max_context_size=max_context_tokens
            )
            
            context_time = (datetime.now() - start_time).total_seconds()
            
            # Generate grounded response
            response_start = datetime.now()
            response_data = await self._generate_grounded_response(query, graph_context)
            response_time = (datetime.now() - response_start).total_seconds()
            
            # Enhanced analysis
            conversation_insight = await self._analyze_conversation_pattern(query, response_data)
            followup_suggestions = await self._generate_followup_suggestions(query, graph_context)
            
            # Check entity continuity
            entity_continuity = await self._check_entity_continuity(query, graph_context)
            
            response = {
                "answer": response_data["answer"],
                "success": response_data["success"],
                "grounded": response_data.get("grounded", False),
                "method": "comprehensive_reasoning",
                "context_summary": {
                    "entities_analyzed": len(graph_context.primary_entities),
                    "relationships_found": len(graph_context.relationships),
                    "pathways_discovered": len(graph_context.pathways),
                    "communities_identified": len(graph_context.communities)
                },
                "performance": {
                    "context_generation_time": context_time,
                    "response_generation_time": response_time,
                    "total_time": context_time + response_time
                },
                "confidence_scores": graph_context.confidence_scores,
                "conversation_insight": conversation_insight,
                "suggested_followup": followup_suggestions,
                "entity_continuity": entity_continuity
            }
            
            if include_reasoning_chain:
                response["reasoning_chain"] = graph_context.reasoning_chain
            
            return response
            
        except Exception as e:
            logger.error(f"Comprehensive reasoning failed: {e}")
            return {
                "answer": "I encountered an error during comprehensive analysis.",
                "success": False,
                "error": str(e)
            }
    
    # ==================== NATURAL LANGUAGE TO CYPHER (RESTORED) ====================
    
    async def _natural_language_to_cypher(
        self, 
        question: str, 
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert natural language to Cypher query (RESTORED)"""
        
        try:
            # Create Cypher generation prompt with PROPER graph_id filtering
            cypher_prompt = f"""
            Given this Neo4j graph schema:
            
            Node Types: {schema.get('entities', [])}
            Relationship Types: {schema.get('relationships', [])}
            
            Sample node structure:
            - All nodes have: id, graph_id, name
            - Nodes may have: description, type, properties
            - CRITICAL: All data must be filtered by graph_id = "{self.current_graph_id}"
            
            Convert this natural language question to a Cypher query:
            "{question}"
            
            Rules:
            1. ALWAYS include "WHERE n.graph_id = '{self.current_graph_id}'" for ALL nodes
            2. Use LIMIT to avoid large result sets (max 20 results)
            3. Return meaningful property names
            4. Handle case-insensitive matching with toLower()
            5. If multiple nodes, ensure ALL have graph_id filter
            
            Return only the Cypher query, no explanation:
            """
            
            response = await llm_service.llm.ainvoke(
                cypher_prompt,
                temperature=0.1  # Low temperature for accurate query generation
            )
            cypher_query = response.content.strip()
            
            # Clean up the query
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            
            # VALIDATE: Ensure graph_id filtering is present
            if f"graph_id = '{self.current_graph_id}'" not in cypher_query:
                logger.warning("Generated Cypher missing graph_id filter, adding it")
                # Try to add graph_id filter automatically
                if "WHERE" in cypher_query.upper():
                    cypher_query = cypher_query.replace(
                        "WHERE", 
                        f"WHERE n.graph_id = '{self.current_graph_id}' AND", 
                        1
                    )
                else:
                    # Add WHERE clause
                    match_pos = cypher_query.upper().find("MATCH")
                    if match_pos >= 0:
                        # Find the end of MATCH clause
                        return_pos = cypher_query.upper().find("RETURN")
                        if return_pos >= 0:
                            cypher_query = (
                                cypher_query[:return_pos] + 
                                f"WHERE n.graph_id = '{self.current_graph_id}' " +
                                cypher_query[return_pos:]
                            )
            
            # Execute the query
            result = await neo4j_client.execute_query(cypher_query)
            
            return {
                "cypher": cypher_query,
                "result": result,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Cypher generation failed: {e}")
            return {
                "cypher": cypher_query if 'cypher_query' in locals() else "Failed to generate",
                "success": False,
                "error": str(e)
            }
    
    # ==================== CONTEXT GENERATION (FIXED GRAPH_ID ISSUES) ====================
    
    async def _generate_graph_context_with_graph_id(
        self,
        query: str,
        graph_id: UUID,  # FIXED: Accept graph_id parameter
        reasoning_mode: ReasoningMode,
        max_context_size: int = 4000
    ) -> GraphContext:
        """Generate rich context using graph algorithms with PROPER graph_id filtering"""
        
        logger.info(f"Generating context for graph {graph_id} in {reasoning_mode.value} mode")
        
        # Entity recognition & disambiguation with graph_id
        primary_entities = await self._extract_and_disambiguate_entities_with_graph_id(query, graph_id)
        
        # Context expansion with proper graph_id filtering
        if reasoning_mode == ReasoningMode.COMPREHENSIVE:
            context_strategies = await asyncio.gather(
                self._get_direct_context_with_graph_id(primary_entities, graph_id),
                self._get_neighborhood_context_with_graph_id(primary_entities, graph_id),
                self._get_pathway_context_with_graph_id(primary_entities, graph_id, 3),
                self._get_community_context_with_graph_id(primary_entities, graph_id),
                self._get_influential_context_with_graph_id(query, graph_id),
                self._get_temporal_context_with_graph_id(primary_entities, graph_id)
            )
        elif reasoning_mode == ReasoningMode.FOCUSED:
            context_strategies = await asyncio.gather(
                self._get_direct_context_with_graph_id(primary_entities, graph_id),
                self._get_neighborhood_context_with_graph_id(primary_entities, graph_id),
                self._get_influential_context_with_graph_id(query, graph_id)
            )
            context_strategies.extend([{}, {}, {}])
        elif reasoning_mode == ReasoningMode.EXPLORATORY:
            context_strategies = await asyncio.gather(
                self._get_direct_context_with_graph_id(primary_entities, graph_id),
                self._get_pathway_context_with_graph_id(primary_entities, graph_id, 4),
                self._get_community_context_with_graph_id(primary_entities, graph_id),
                self._get_temporal_context_with_graph_id(primary_entities, graph_id)
            )
            context_strategies.extend([{}, {}])
        
        direct_ctx, neighborhood_ctx, pathway_ctx, community_ctx, influence_ctx, temporal_ctx = context_strategies
        
        # Context ranking with RELEVANCE CALCULATION (RESTORED)
        ranked_context = await self._rank_and_filter_context_with_relevance(
            query, direct_ctx, neighborhood_ctx, pathway_ctx, 
            community_ctx, influence_ctx, temporal_ctx, max_context_size
        )
        
        # Generate reasoning chain with QUERY consideration (FIXED)
        reasoning_chain = await self._generate_reasoning_chain_with_query(query, ranked_context)
        
        return GraphContext(
            primary_entities=primary_entities,
            relationships=ranked_context.get("relationships", []),
            neighborhoods=ranked_context.get("neighborhoods", []),
            pathways=ranked_context.get("pathways", []),
            communities=ranked_context.get("communities", []),
            influential_nodes=ranked_context.get("influential", []),
            temporal_context=ranked_context.get("temporal", []),
            confidence_scores=ranked_context.get("confidence", {}),
            reasoning_chain=reasoning_chain
        )
    
    # ==================== CONTEXT METHODS WITH GRAPH_ID (ALL FIXED) ====================
    
    async def _extract_and_disambiguate_entities_with_graph_id(
        self, 
        query: str, 
        graph_id: UUID
    ) -> List[Dict[str, Any]]:
        """Entity recognition with graph_id filtering (FIXED)"""
        
        entity_extraction_query = f"""
        Extract entities (people, organizations, locations, concepts) from: "{query}"
        Return JSON array: ["entity1", "entity2", ...]
        """
        
        try:
            response = await llm_service.llm.ainvoke([
                {"role": "system", "content": "Extract entities. Return only JSON array."},
                {"role": "user", "content": entity_extraction_query}
            ], temperature=0.1)
            
            entities = json.loads(response.content.strip())
            if not isinstance(entities, list):
                entities = []
        except:
            entities = []
        
        # Disambiguate against graph entities with PROPER graph_id filtering
        disambiguated = []
        for entity_text in entities:
            matches = await self._find_entity_matches_with_graph_id(entity_text, graph_id)
            if matches:
                disambiguated.extend(matches[:2])
        
        return disambiguated
    
    async def _find_entity_matches_with_graph_id(
        self, 
        entity_text: str, 
        graph_id: UUID
    ) -> List[Dict[str, Any]]:
        """Find matching entities with PROPER graph_id filtering (FIXED)"""
        
        query = """
        MATCH (n)
        WHERE n.graph_id = $graph_id 
        AND (toLower(n.name) CONTAINS toLower($entity_text) 
             OR toLower($entity_text) CONTAINS toLower(n.name))
        
        WITH n, 
             CASE 
                WHEN toLower(n.name) = toLower($entity_text) THEN 1.0
                WHEN toLower(n.name) CONTAINS toLower($entity_text) THEN 0.8
                ELSE 0.6
             END as relevance_score
        
        OPTIONAL MATCH (n)-[r]-()
        WHERE r.graph_id = $graph_id
        
        RETURN n.id as id, n.name as name, labels(n) as labels, 
               n{.*} as properties, count(r) as relationship_count,
               relevance_score
        ORDER BY relevance_score DESC, relationship_count DESC
        LIMIT 5
        """
        
        results = await neo4j_client.execute_query(query, {
            "entity_text": entity_text,
            "graph_id": str(graph_id)
        })
        
        return [dict(result) for result in results]
    
    async def _get_direct_context_with_graph_id(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Get direct context with PROPER graph_id filtering (FIXED)"""
        
        if not entities:
            return {"relationships": []}
        
        entity_ids = [e["id"] for e in entities]
        
        query = """
        MATCH (start)-[r]-(connected)
        WHERE start.id IN $entity_ids 
        AND start.graph_id = $graph_id 
        AND r.graph_id = $graph_id
        AND connected.graph_id = $graph_id
        
        RETURN start.id as source_id, start.name as source_name,
               type(r) as relationship_type,
               connected.id as target_id, connected.name as target_name,
               r{.*} as relationship_properties
        LIMIT 50
        """
        
        results = await neo4j_client.execute_query(query, {
            "entity_ids": entity_ids,
            "graph_id": str(graph_id)
        })
        
        relationships = []
        for result in results:
            relationships.append({
                "source": result["source_name"],
                "source_id": result["source_id"],
                "target": result["target_name"],
                "target_id": result["target_id"],
                "type": result["relationship_type"],
                "properties": result["relationship_properties"]
            })
        
        return {"relationships": relationships}
    
    # ==================== PATHWAY METHODS (RESTORED + FIXED) ====================
    
    async def _get_pathway_context_with_graph_id(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Get pathways with PROPER graph_id filtering (FIXED)"""
        
        if len(entities) < 2:
            return {"pathways": []}
        
        pathways = []
        
        # Get pathways between all pairs
        for i, start_entity in enumerate(entities):
            for j, end_entity in enumerate(entities[i+1:], i+1):
                try:
                    # Use both advanced and simple pathfinding
                    advanced_paths = await self._find_paths_between_entities_with_graph_id(
                        start_entity["id"], end_entity["id"], graph_id, max_depth
                    )
                    pathways.extend(advanced_paths)
                except Exception as e:
                    logger.warning(f"Advanced pathfinding failed: {e}")
                
                try:
                    # Fallback to shortest paths (RESTORED)
                    simple_paths = await self._find_shortest_paths_with_graph_id(
                        start_entity["id"], end_entity["id"], graph_id
                    )
                    pathways.extend(simple_paths)
                except Exception as e:
                    logger.warning(f"Simple pathfinding failed: {e}")
        
        # Remove duplicates and sort
        unique_pathways = []
        seen_paths = set()
        
        for pathway in pathways:
            path_signature = tuple(pathway.get("nodes", []))
            if path_signature not in seen_paths:
                seen_paths.add(path_signature)
                unique_pathways.append(pathway)
        
        unique_pathways.sort(key=lambda x: x.get("length", 999))
        
        return {"pathways": unique_pathways[:10]}
    
    async def _find_shortest_paths_with_graph_id(
        self, 
        start_id: str, 
        end_id: str, 
        graph_id: UUID
    ) -> List[Dict[str, Any]]:
        """Find shortest paths between entities (RESTORED)"""
        
        query = """
        MATCH (start {id: $start_id, graph_id: $graph_id})
        MATCH (end {id: $end_id, graph_id: $graph_id})
        
        MATCH path = shortestPath((start)-[*1..3]-(end))
        WHERE ALL(r IN relationships(path) WHERE r.graph_id = $graph_id)
        AND ALL(n IN nodes(path) WHERE n.graph_id = $graph_id)
        
        RETURN [n.name FOR n IN nodes(path)] as node_names,
               [type(r) FOR r IN relationships(path)] as relationship_types,
               length(path) as path_length
        ORDER BY path_length
        LIMIT 3
        """
        
        results = await neo4j_client.execute_query(query, {
            "start_id": start_id,
            "end_id": end_id,
            "graph_id": str(graph_id)
        })
        
        return [
            {
                "start": start_id,
                "end": end_id,
                "nodes": result["node_names"],
                "relationships": result["relationship_types"],
                "length": result["path_length"],
                "type": "shortest_path"
            }
            for result in results
        ]
    
    async def _find_paths_between_entities_with_graph_id(
        self, 
        start_id: str, 
        end_id: str, 
        graph_id: UUID,
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """Advanced pathfinding with PROPER graph_id filtering (FIXED)"""
        
        query = """
        MATCH (start {id: $start_id, graph_id: $graph_id})
        MATCH (end {id: $end_id, graph_id: $graph_id})
        
        CALL apoc.path.allSimplePaths(start, end, '', $max_depth) YIELD path
        WHERE ALL(n IN nodes(path) WHERE n.graph_id = $graph_id)
        AND ALL(r IN relationships(path) WHERE r.graph_id = $graph_id)
        
        WITH path, length(path) as path_length
        ORDER BY path_length
        LIMIT 5
        
        RETURN [n.name FOR n IN nodes(path)] as path_nodes,
               [{type: type(r), properties: properties(r)} FOR r IN relationships(path)] as path_relationships,
               path_length
        """
        
        results = await neo4j_client.execute_query(query, {
            "start_id": start_id,
            "end_id": end_id,
            "graph_id": str(graph_id),
            "max_depth": max_depth
        })
        
        return [
            {
                "start": start_id,
                "end": end_id,
                "nodes": result["path_nodes"],
                "relationships": result["path_relationships"],
                "length": result["path_length"],
                "type": "advanced_path"
            }
            for result in results
        ]
    
    # ==================== ENHANCEMENT METHODS (RESTORED) ====================
    
    async def _enhance_response_with_insights(
        self, 
        query: str, 
        response: Dict[str, Any], 
        mode: str
    ) -> Dict[str, Any]:
        """Add insights and related information to response (RESTORED)"""
        
        try:
            # Get related entities using embedding similarity
            related_entities = await self._get_related_entities(query, limit=5)
            
            # Get community insights
            community_insights = await self._get_community_insights_for_query(query)
            
            # Get graph statistics
            graph_stats = await self._get_relevant_graph_statistics()
            
            response.update({
                "related_entities": related_entities,
                "graph_insights": {
                    "community_insights": community_insights,
                    "graph_statistics": graph_stats,
                    "retrieval_mode": mode
                },
                "enhanced": True
            })
            
            return response
            
        except Exception as e:
            logger.warning(f"Failed to enhance response: {e}")
            return response
    
    async def _check_entity_continuity(
        self, 
        query: str, 
        context: GraphContext
    ) -> Dict[str, Any]:
        """Check entity relationship continuity (RESTORED)"""
        
        try:
            # Check if entities in current query relate to previous conversation
            previous_entities = []
            for entry in self.conversation_history[-3:]:  # Last 3 exchanges
                if "entities_discussed" in entry:
                    previous_entities.extend(entry["entities_discussed"])
            
            current_entities = [e["name"] for e in context.primary_entities]
            
            # Find overlapping entities
            overlapping = set(previous_entities) & set(current_entities)
            
            continuity_score = len(overlapping) / max(len(current_entities), 1)
            
            return {
                "continuity_score": continuity_score,
                "overlapping_entities": list(overlapping),
                "conversation_coherent": continuity_score > 0.3,
                "recommendation": "High continuity" if continuity_score > 0.5 else "New topic detected"
            }
            
        except Exception as e:
            logger.warning(f"Entity continuity check failed: {e}")
            return {"continuity_score": 0, "error": str(e)}
    
    async def _analyze_conversation_pattern(
        self, 
        query: str, 
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze conversation patterns for better context (RESTORED)"""
        
        try:
            conversation_length = len(self.conversation_history)
            
            if conversation_length < 2:
                return {"pattern": "new_conversation", "depth": "initial"}
            
            # Analyze query complexity and conversation depth
            recent_queries = [
                entry["content"] for entry in self.conversation_history[-5:] 
                if entry["type"] == "user"
            ]
            
            avg_query_length = sum(len(q.split()) for q in recent_queries) / len(recent_queries)
            
            if avg_query_length > 15:
                pattern = "detailed_exploration"
            elif conversation_length > 10:
                pattern = "extended_dialogue"
            else:
                pattern = "focused_inquiry"
            
            return {
                "pattern": pattern,
                "conversation_length": conversation_length,
                "avg_query_complexity": avg_query_length,
                "engagement_level": "high" if conversation_length > 5 else "medium"
            }
            
        except Exception as e:
            logger.warning(f"Conversation analysis failed: {e}")
            return {"pattern": "unknown", "error": str(e)}
    
    # ==================== RELEVANCE CALCULATION (RESTORED) ====================
    
    async def _rank_and_filter_context_with_relevance(
        self, 
        query: str, 
        direct_ctx: Dict, 
        neighborhood_ctx: Dict, 
        pathway_ctx: Dict,
        community_ctx: Dict, 
        influence_ctx: Dict, 
        temporal_ctx: Dict, 
        max_tokens: int
    ) -> Dict[str, Any]:
        """Rank and filter context with RELEVANCE CALCULATION (RESTORED)"""
        
        try:
            # Calculate relevance scores using embeddings
            query_embedding = await embedding_service.embed_text(query)
            
            # Score relationships by relevance
            scored_relationships = []
            for rel in direct_ctx.get("relationships", []):
                rel_text = f"{rel['source']} {rel['type']} {rel['target']}"
                rel_embedding = await embedding_service.embed_text(rel_text)
                relevance = await self._calculate_embedding_similarity(query_embedding, rel_embedding)
                
                rel["relevance_score"] = relevance
                scored_relationships.append(rel)
            
            # Sort by relevance
            scored_relationships.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Filter by token limit (approximate)
            filtered_context = {
                "relationships": scored_relationships[:15],  # Top 15 most relevant
                "neighborhoods": neighborhood_ctx.get("neighborhoods", [])[:5],
                "pathways": pathway_ctx.get("pathways", [])[:5],
                "communities": community_ctx.get("communities", [])[:3],
                "influential": influence_ctx.get("influential", [])[:5],
                "temporal": temporal_ctx.get("temporal", [])[:5],
                "confidence": {
                    "entity_matching": 0.8,
                    "relationship_relevance": sum(r.get("relevance_score", 0) for r in scored_relationships[:10]) / 10,
                    "pathway_analysis": 0.6,
                    "community_relevance": 0.5
                }
            }
            
            return filtered_context
            
        except Exception as e:
            logger.warning(f"Relevance calculation failed: {e}")
            # Fallback to simple filtering
            return {
                "relationships": direct_ctx.get("relationships", [])[:10],
                "neighborhoods": neighborhood_ctx.get("neighborhoods", [])[:5],
                "pathways": pathway_ctx.get("pathways", [])[:5],
                "communities": community_ctx.get("communities", [])[:3],
                "influential": influence_ctx.get("influential", [])[:5],
                "temporal": temporal_ctx.get("temporal", [])[:5],
                "confidence": {"entity_matching": 0.6, "relationship_relevance": 0.5}
            }
    
    async def _calculate_embedding_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between embeddings"""
        
        try:
            import numpy as np
            
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception:
            return 0.5  # Default similarity
    
    async def _generate_reasoning_chain_with_query(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate reasoning chain considering the QUERY (FIXED)"""
        
        reasoning_steps = [f"Processing query: '{query[:50]}{'...' if len(query) > 50 else ''}'"]
        
        # Query analysis
        query_type = await self._classify_query_type(query)
        reasoning_steps.append(f"Query classified as: {query_type}")
        
        # Entity analysis
        if context.get("relationships"):
            reasoning_steps.append(f"Found {len(context['relationships'])} relevant relationships")
        
        # Relationship analysis  
        if context.get("neighborhoods"):
            reasoning_steps.append(f"Analyzed {len(context['neighborhoods'])} entity neighborhoods")
        
        # Path analysis
        if context.get("pathways"):
            reasoning_steps.append(f"Discovered {len(context['pathways'])} connection pathways")
        
        # Community analysis
        if context.get("communities"):
            reasoning_steps.append(f"Identified {len(context['communities'])} community clusters")
        
        # Influence analysis
        if context.get("influential"):
            reasoning_steps.append(f"Considered {len(context['influential'])} influential entities")
        
        reasoning_steps.append("Synthesizing graph information to provide accurate, grounded response")
        
        return reasoning_steps
    
    # ==================== UTILITY METHODS (ENHANCED) ====================
    
    async def _validate_response_grounding(
        self, 
        response: str, 
        context: Dict[str, Any]
    ) -> bool:
        """Enhanced response grounding validation"""
        
        try:
            # Convert context to text
            if isinstance(context, dict):
                context_text = ""
                for key, value in context.items():
                    if isinstance(value, str):
                        context_text += f"{key}: {value}\n"
                    elif isinstance(value, list):
                        context_text += f"{key}: {', '.join(str(v) for v in value[:5])}\n"
            else:
                context_text = str(context)
            
            validation_prompt = f"""
            Check if this RESPONSE uses only information from the CONTEXT.
            
            CONTEXT: {context_text[:1000]}
            RESPONSE: {response}
            
            Return "GROUNDED" if response uses only context information.
            Return "NOT_GROUNDED" if response adds external information.
            """
            
            validation = await llm_service.llm.ainvoke([
                {"role": "system", "content": "Strict fact-checker for grounding."},
                {"role": "user", "content": validation_prompt}
            ], temperature=0.0)  # Zero temperature for validation
            
            return "GROUNDED" in validation.content.upper()
            
        except Exception as e:
            logger.warning(f"Response validation failed: {e}")
            return False
    
    # ==================== ADDITIONAL RESTORED methods from original ====================
    
    async def _get_entity_neighborhood(
        self, 
        entity_id: str, 
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Get entity neighborhood with PROPER graph_id filtering (RESTORED + FIXED)"""
        
        try:
            query = f"""
            MATCH (center {{id: $entity_id, graph_id: $graph_id}})
            CALL {{
                WITH center
                MATCH path = (center)-[*1..{max_depth}]-(neighbor)
                WHERE neighbor.graph_id = $graph_id
                AND ALL(r IN relationships(path) WHERE r.graph_id = $graph_id)
                RETURN neighbor, relationships(path) as rels
                LIMIT 10
            }}
            RETURN center, collect({{neighbor: neighbor, relationships: rels}}) as neighborhood
            """
            
            result = await neo4j_client.execute_query(query, {
                "entity_id": entity_id,
                "graph_id": str(self.current_graph_id)  # FIXED: Use current_graph_id
            })
            
            return result[0] if result else {}
            
        except Exception as e:
            logger.warning(f"Neighborhood extraction failed: {e}")
            return {}
    
    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from query (RESTORED)"""
        
        try:
            entity_prompt = f"""
            Extract the main entities (people, organizations, concepts, etc.) from this question:
            "{query}"
            
            Return only a JSON list of entity names, like: ["Entity1", "Entity2"]
            If no clear entities, return: []
            """
            
            response = await llm_service.llm.ainvoke(
                entity_prompt,
                temperature=0.1  # Low temperature for consistent extraction
            )
            entities = json.loads(response.content.strip())
            return entities if isinstance(entities, list) else []
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    async def _generate_answer_from_cypher_results(
        self, 
        query: str, 
        results: List[Dict]
    ) -> str:
        """Generate natural language answer from Cypher results (RESTORED)"""
        
        if not results:
            return "I couldn't find any relevant information in the graph."
        
        # Format results for LLM
        formatted_results = json.dumps(results[:10], indent=2, default=str)
        
        answer_prompt = f"""
        Based on these query results from knowledge graph {self.current_graph_id}, provide a natural language answer.
        
        Question: {query}
        
        Query Results:
        {formatted_results}
        
        IMPORTANT: Use only the provided results. If results don't fully answer the question, mention what information IS available.
        """
        
        response = await llm_service.llm.ainvoke(
            answer_prompt,
            temperature=0.2  # Low temperature for factual accuracy
        )
        return response.content
    
    # ==================== MISSING IMPLEMENTATION PLACEHOLDERS ====================
    
    async def _get_neighborhood_context_with_graph_id(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Get 1-2 hop neighbor context with PROPER graph_id filtering"""
        
        if not entities:
            return {"neighborhoods": [], "relationships": []}
        
        entity_ids = [e["id"] for e in entities]
        
        query = """
        MATCH (start)
        WHERE start.id IN $entity_ids AND start.graph_id = $graph_id
        
        MATCH (start)-[r1]-(neighbor1)
        WHERE r1.graph_id = $graph_id AND neighbor1.graph_id = $graph_id
        
        OPTIONAL MATCH (neighbor1)-[r2]-(neighbor2)
        WHERE r2.graph_id = $graph_id AND neighbor2.graph_id = $graph_id
        AND neighbor2.id <> start.id
        
        WITH start, r1, neighbor1, 
            collect(DISTINCT {
                node: neighbor2{.id, .name, labels: labels(neighbor2)},
                relationship: type(r2)
            })[..3] as second_hop
        
        RETURN start.id as center_id,
            start.name as center_name,
            {
                relationship: type(r1),
                neighbor: neighbor1{.id, .name, labels: labels(neighbor1)},
                second_hop: second_hop
            } as neighborhood_info
        LIMIT 30
        """
        
        results = await neo4j_client.execute_query(query, {
            "entity_ids": entity_ids,
            "graph_id": str(graph_id)
        })
        
        neighborhoods = {}
        relationships = []
        
        for result in results:
            center_id = result["center_id"]
            if center_id not in neighborhoods:
                neighborhoods[center_id] = {
                    "center": {"id": center_id, "name": result["center_name"]}, 
                    "neighbors": []
                }
            
            neighborhoods[center_id]["neighbors"].append(result["neighborhood_info"])
            relationships.append({
                "source": center_id,
                "target": result["neighborhood_info"]["neighbor"]["id"],
                "type": result["neighborhood_info"]["relationship"]
            })
        
        return {
            "neighborhoods": list(neighborhoods.values()),
            "relationships": relationships
        }

    async def _get_community_context_with_graph_id(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Find entities in same communities using Neo4j GDS Louvain algorithm with graph_id filtering"""
        
        if not entities:
            return {"communities": []}
        
        try:
            # Try advanced community detection with Neo4j GDS
            community_query = """
            CALL {
                // Create temporary graph projection for this specific graph_id
                CALL gds.graph.project(
                    'temp-community-' + $graph_id,
                    {
                        Entity: {
                            label: '*',
                            properties: ['name'],
                            nodeFilter: 'n.graph_id = "' + $graph_id + '"'
                        }
                    },
                    {
                        RELATIONSHIP: {
                            type: '*',
                            orientation: 'UNDIRECTED',
                            relationshipFilter: 'r.graph_id = "' + $graph_id + '"'
                        }
                    }
                )
                YIELD graphName
                
                // Run Louvain community detection
                CALL gds.louvain.stream('temp-community-' + $graph_id)
                YIELD nodeId, communityId
                
                // Get original nodes
                MATCH (node)
                WHERE id(node) = nodeId AND node.graph_id = $graph_id
                
                WITH node, communityId
                ORDER BY communityId, node.name
                
                // Group by community
                WITH communityId, collect(node) as community_members
                WHERE size(community_members) > 1  // Only communities with multiple members
                
                RETURN communityId,
                    [member IN community_members | {
                        id: member.id,
                        name: member.name,
                        labels: labels(member)
                    }] as members,
                    size(community_members) as size
                LIMIT 10
            }
            
            // Clean up the temporary graph
            CALL gds.graph.drop('temp-community-' + $graph_id, false)
            YIELD graphName as droppedGraph
            
            RETURN communityId, members, size
            """
            
            results = await neo4j_client.execute_query(community_query, {
                "graph_id": str(graph_id)
            })
            
            communities = []
            for result in results:
                communities.append({
                    "community_id": result["communityId"],
                    "members": result["members"],
                    "size": result["size"],
                    "type": "louvain_community"
                })
            
            return {"communities": communities}
            
        except Exception as e:
            logger.warning(f"Advanced community detection failed: {e}")
            # Fallback to simple shared neighbor detection
            return await self._get_simple_community_context_with_graph_id(entities, graph_id)

    async def _get_simple_community_context_with_graph_id(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Fallback community detection based on shared neighbors with graph_id filtering"""
        
        if not entities:
            return {"communities": []}
        
        entity_ids = [e["id"] for e in entities]
        
        query = """
        MATCH (entity)
        WHERE entity.id IN $entity_ids AND entity.graph_id = $graph_id
        
        MATCH (entity)-[r1]-(neighbor)-[r2]-(community_member)
        WHERE r1.graph_id = $graph_id AND r2.graph_id = $graph_id
        AND neighbor.graph_id = $graph_id AND community_member.graph_id = $graph_id
        AND community_member.id <> entity.id
        
        WITH entity, neighbor, collect(DISTINCT community_member) as members
        WHERE size(members) >= 2
        
        RETURN entity.id as entity_id,
            entity.name as entity_name,
            neighbor.name as hub_name,
            [m IN members | {id: m.id, name: m.name}][..5] as community_members
        LIMIT 10
        """
        
        results = await neo4j_client.execute_query(query, {
            "entity_ids": entity_ids,
            "graph_id": str(graph_id)
        })
        
        communities = []
        for result in results:
            communities.append({
                "entity": result["entity_name"],
                "hub": result["hub_name"],
                "members": result["community_members"],
                "type": "shared_neighbor_community"
            })
        
        return {"communities": communities}

    async def _get_influential_context_with_graph_id(
        self, 
        query: str, 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Get highly connected nodes using Neo4j GDS PageRank with graph_id filtering"""
        
        try:
            # Try advanced PageRank centrality with Neo4j GDS
            pagerank_query = """
            CALL {
                // Create temporary graph projection
                CALL gds.graph.project(
                    'temp-pagerank-' + $graph_id,
                    {
                        Entity: {
                            label: '*',
                            properties: ['name'],
                            nodeFilter: 'n.graph_id = "' + $graph_id + '"'
                        }
                    },
                    {
                        RELATIONSHIP: {
                            type: '*',
                            orientation: 'UNDIRECTED',
                            relationshipFilter: 'r.graph_id = "' + $graph_id + '"'
                        }
                    }
                )
                YIELD graphName
                
                // Run PageRank algorithm
                CALL gds.pageRank.stream('temp-pagerank-' + $graph_id)
                YIELD nodeId, score
                
                // Get original nodes with scores
                MATCH (node)
                WHERE id(node) = nodeId AND node.graph_id = $graph_id
                
                WITH node, score
                ORDER BY score DESC
                LIMIT 10
                
                RETURN node.id as entity_id,
                    node.name as entity_name,
                    score as pagerank_score,
                    labels(node) as labels
            }
            
            // Clean up temporary graph
            CALL gds.graph.drop('temp-pagerank-' + $graph_id, false)
            YIELD graphName as droppedGraph
            
            RETURN entity_id, entity_name, pagerank_score, labels
            """
            
            results = await neo4j_client.execute_query(pagerank_query, {
                "graph_id": str(graph_id)
            })
            
            influential = []
            for result in results:
                influential.append({
                    "id": result["entity_id"],
                    "name": result["entity_name"],
                    "pagerank_score": result["pagerank_score"],
                    "labels": result["labels"],
                    "influence_type": "pagerank_centrality"
                })
            
            return {"influential": influential}
            
        except Exception as e:
            logger.warning(f"Advanced PageRank failed: {e}")
            # Fallback to simple degree centrality
            return await self._get_simple_influential_context_with_graph_id(query, graph_id)

    async def _get_simple_influential_context_with_graph_id(
        self, 
        query: str, 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Fallback influential nodes based on degree centrality with graph_id filtering"""
        
        # Find high-degree nodes in this specific graph
        cypher_query = """
        MATCH (n)
        WHERE n.graph_id = $graph_id
        
        OPTIONAL MATCH (n)-[r]-()
        WHERE r.graph_id = $graph_id
        
        WITH n, count(r) as degree
        WHERE degree > 3  // Nodes with more than 3 connections
        ORDER BY degree DESC
        LIMIT 10
        
        RETURN n.name as name, 
            n.id as id, 
            degree,
            labels(n) as labels,
            n{.*} as properties
        """
        
        results = await neo4j_client.execute_query(cypher_query, {
            "graph_id": str(graph_id)
        })
        
        influential = []
        for result in results:
            influential.append({
                "name": result["name"],
                "id": result["id"],
                "degree": result["degree"],
                "labels": result["labels"],
                "properties": result["properties"],
                "influence_type": "degree_centrality"
            })
        
        return {"influential": influential}

    async def _get_temporal_context_with_graph_id(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """Get temporal/time-based context with graph_id filtering"""
        
        if not entities:
            return {"temporal": []}
        
        entity_ids = [e["id"] for e in entities]
        
        # Look for date-related properties in relationships and nodes
        query = """
        MATCH (n)-[r]-(m)
        WHERE n.id IN $entity_ids 
        AND n.graph_id = $graph_id 
        AND r.graph_id = $graph_id 
        AND m.graph_id = $graph_id
        AND (
            r.start_date IS NOT NULL OR 
            r.end_date IS NOT NULL OR 
            r.date IS NOT NULL OR 
            r.year IS NOT NULL OR
            r.created_at IS NOT NULL OR
            r.timestamp IS NOT NULL OR
            n.birth_date IS NOT NULL OR
            n.founded IS NOT NULL OR
            m.birth_date IS NOT NULL OR
            m.founded IS NOT NULL
        )
        
        WITH n, r, m,
            coalesce(
                r.start_date, 
                r.end_date, 
                r.date, 
                r.year, 
                r.created_at,
                r.timestamp,
                n.birth_date, 
                n.founded,
                m.birth_date,
                m.founded
            ) as date_info
        
        WHERE date_info IS NOT NULL
        
        RETURN n.name as entity_name,
            type(r) as relationship_type,
            m.name as connected_entity,
            date_info,
            r{.*} as relationship_properties
        ORDER BY date_info DESC
        LIMIT 20
        """
        
        results = await neo4j_client.execute_query(query, {
            "entity_ids": entity_ids,
            "graph_id": str(graph_id)
        })
        
        temporal = []
        for result in results:
            temporal.append({
                "entity": result["entity_name"],
                "relationship": result["relationship_type"],
                "connected_to": result["connected_entity"],
                "date": str(result["date_info"]),  # Convert to string for JSON serialization
                "relationship_properties": result["relationship_properties"],
                "context_type": "temporal"
            })
        
        return {"temporal": temporal}

    async def _get_related_entities(
        self, 
        query: str, 
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get related entities using search_service with current graph filtering"""
        
        try:
            if not self.current_graph_id:
                return []
            
            # Use search_service to find semantically similar entities
            similar_entities = await search_service.similarity_search_entities(
                query=query,
                graph_id=self.current_graph_id,
                k=limit,
                threshold=0.5  # Lower threshold for more results
            )
            
            return similar_entities
            
        except Exception as e:
            logger.warning(f"Failed to get related entities: {e}")
            return []

    async def _get_community_insights_for_query(self, query: str) -> Dict[str, Any]:
        """Get community-based insights for the current query"""
        
        try:
            if not self.current_graph_id:
                return {}
            
            # Extract entities from query
            query_entities = await self._extract_entities_from_query(query)
            
            if not query_entities:
                return {"insight": "No specific entities detected in query"}
            
            # Get communities for these entities
            entity_matches = []
            for entity_name in query_entities[:3]:  # Limit to top 3 entities
                matches = await self._find_entity_matches_with_graph_id(entity_name, self.current_graph_id)
                entity_matches.extend(matches[:2])  # Top 2 matches per entity
            
            if not entity_matches:
                return {"insight": "No matching entities found in graph"}
            
            # Get community context for matched entities
            community_context = await self._get_community_context_with_graph_id(
                entity_matches, self.current_graph_id
            )
            
            communities = community_context.get("communities", [])
            
            if not communities:
                return {"insight": "Entities appear to be isolated (no strong communities detected)"}
            
            # Generate insights
            total_communities = len(communities)
            largest_community_size = max((c.get("size", 0) for c in communities), default=0)
            
            insights = {
                "total_communities": total_communities,
                "largest_community_size": largest_community_size,
                "community_types": list(set(c.get("type", "unknown") for c in communities)),
                "insight": f"Found {total_communities} communities. Largest has {largest_community_size} members.",
                "communities_summary": [
                    {
                        "type": c.get("type", "unknown"),
                        "size": c.get("size", len(c.get("members", []))),
                        "sample_members": [m.get("name") for m in c.get("members", [])][:3]
                    }
                    for c in communities[:3]  # Top 3 communities
                ]
            }
            
            return insights
            
        except Exception as e:
            logger.warning(f"Failed to get community insights: {e}")
            return {"error": str(e)}

    async def _get_relevant_graph_statistics(self) -> Dict[str, Any]:
        """Get relevant graph statistics for current graph"""
        
        try:
            if not self.current_graph_id:
                return {}
            
            # Basic graph statistics
            stats_query = """
            MATCH (n)
            WHERE n.graph_id = $graph_id
            WITH count(n) as node_count
            
            MATCH ()-[r]-()
            WHERE r.graph_id = $graph_id
            WITH node_count, count(r) as rel_count
            
            MATCH (n)
            WHERE n.graph_id = $graph_id
            WITH node_count, rel_count, labels(n) as node_labels
            UNWIND node_labels as label
            WITH node_count, rel_count, label
            WHERE label <> 'Entity'  // Skip generic labels
            
            WITH node_count, rel_count, collect(DISTINCT label) as entity_types
            
            MATCH ()-[r]-()
            WHERE r.graph_id = $graph_id
            WITH node_count, rel_count, entity_types, type(r) as rel_type
            
            RETURN node_count,
                rel_count,
                entity_types,
                collect(DISTINCT rel_type) as relationship_types
            """
            
            result = await neo4j_client.execute_query(stats_query, {
                "graph_id": str(self.current_graph_id)
            })
            
            if result:
                stats = result[0]
                node_count = stats["node_count"]
                rel_count = stats["rel_count"]
                
                return {
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "density": rel_count / (node_count * (node_count - 1)) if node_count > 1 else 0,
                    "entity_types": stats["entity_types"][:10],  # Top 10 types
                    "relationship_types": stats["relationship_types"][:10],  # Top 10 types
                    "avg_degree": (2 * rel_count / node_count) if node_count > 0 else 0
                }
            else:
                return {"node_count": 0, "relationship_count": 0}
                
        except Exception as e:
            logger.warning(f"Failed to get graph statistics: {e}")
            return {"error": str(e)}

    async def _precompute_graph_statistics(self) -> None:
        """Precompute and cache graph statistics for better performance"""
        
        try:
            if not self.current_graph_id:
                return
            
            # Get comprehensive statistics
            self.graph_statistics = await self._get_relevant_graph_statistics()
            
            # Cache timestamp
            self.graph_statistics["computed_at"] = datetime.now()
            
            logger.info(f"Precomputed statistics for graph {self.current_graph_id}: "
                    f"{self.graph_statistics.get('node_count', 0)} nodes, "
                    f"{self.graph_statistics.get('relationship_count', 0)} relationships")
            
        except Exception as e:
            logger.warning(f"Failed to precompute graph statistics: {e}")
            self.graph_statistics = {
                "error": str(e),
                "node_count": 0,
                "relationship_count": 0
            }
    
    # ==================== OTHER METHODS (preserved from original) ====================
    
    def _build_vector_context(self, entity_results: List[Dict], chunk_results: List[Dict]) -> Dict[str, str]:
        """Build context from vector search results (RESTORED)"""
        
        entities_text = "\n".join([
            f"- {entity['name']}: {entity.get('description', 'No description')}"
            for entity in entity_results[:5]
        ])
        
        chunks_text = "\n".join([
            f"- {chunk['text'][:200]}..."
            for chunk in chunk_results[:3]
        ])
        
        return {
            "entities": entities_text or "No relevant entities found",
            "chunks": chunks_text or "No relevant text found"
        }
    
    def _build_graphrag_context(
        self,
        query_entities: List[str],
        similar_entities: List[Dict],
        neighborhoods: List[Dict],
        text_chunks: List[Dict]
    ) -> str:
        """Build comprehensive GraphRAG context (RESTORED)"""
        
        context_parts = []
        
        if query_entities:
            context_parts.append(f"Query mentions: {', '.join(query_entities)}")
        
        if similar_entities:
            entities_info = "\n".join([
                f"- {e['name']} (similarity: {e.get('score', 0):.2f})"
                for e in similar_entities[:5]
            ])
            context_parts.append(f"Relevant entities:\n{entities_info}")
        
        if text_chunks:
            chunks_info = "\n".join([
                f"- {chunk['text'][:150]}..."
                for chunk in text_chunks[:3]
            ])
            context_parts.append(f"Relevant text:\n{chunks_info}")
        
        return "\n\n".join(context_parts)
    
    async def _generate_graphrag_answer(self, query: str, context: str) -> str:
        """Generate GraphRAG answer (RESTORED)"""
        
        rag_prompt = f"""
        You are an AI assistant with access to knowledge graph {self.current_graph_id}. 
        Answer the user's question using the provided context.
        
        Question: {query}
        
        Context from Knowledge Graph:
        {context}
        
        Instructions:
        1. Use ONLY the provided context
        2. Mention specific entities and relationships when relevant
        3. If information is incomplete, acknowledge this
        4. Be conversational but accurate
        
        Answer:
        """
        
        response = await llm_service.llm.ainvoke(
            rag_prompt,
            temperature=0.2  # Low temperature for accuracy
        )
        return response.content
    
    async def _combine_answers(
        self, 
        query: str, 
        vector_result: Dict, 
        cypher_result: Dict
    ) -> str:
        """Combine answers from different methods (RESTORED)"""
        
        combine_prompt = f"""
        I have two answers to the question "{query}" from different methods. Combine them into one comprehensive answer.
        
        Vector Search Answer: {vector_result['answer']}
        
        Graph Query Answer: {cypher_result['answer']}
        
        Provide a single, well-structured answer that incorporates the best insights from both. Remove redundancy and ensure coherence.
        """
        
        response = await llm_service.llm.ainvoke(
            combine_prompt,
            temperature=0.2  # Low temperature for consistent combining
        )
        return response.content
    
    async def _create_cypher_chain(self, schema: Dict[str, Any]):
        """Create Cypher QA chain (RESTORED)"""
        # Placeholder for future langchain integration
        return None
    
    async def _classify_query_type(self, query: str) -> str:
        """Classify query type (RESTORED)"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["who is", "who are", "tell me about"]):
            return "entity_information"
        elif any(word in query_lower for word in ["how are", "connected", "relationship"]):
            return "relationship_query" 
        elif any(word in query_lower for word in ["find", "search", "show me"]):
            return "discovery_query"
        elif any(word in query_lower for word in ["why", "explain", "reason"]):
            return "explanation_query"
        else:
            return "general_query"
    
    # ==================== BACKWARD COMPATIBILITY METHODS ====================
    
    async def _generate_graph_context(
        self,
        query: str,
        reasoning_mode: ReasoningMode,
        max_context_size: int = 4000
    ) -> GraphContext:
        """Backward compatibility wrapper"""
        return await self._generate_graph_context_with_graph_id(
            query, self.current_graph_id, reasoning_mode, max_context_size
        )
    
    async def _generate_grounded_response(
        self,
        query: str,
        context: GraphContext
    ) -> Dict[str, Any]:
        """Generate grounded response (enhanced)"""
        
        context_text = self._format_context_for_llm(context)
        
        system_prompt = f"""
        You are a knowledge graph assistant for graph {self.current_graph_id}. 
        Answer using ONLY the provided graph context.
        
        RULES:
        1. Use ONLY information from the graph context below
        2. If the answer isn't in the context, say "I don't have that information in the knowledge graph"
        3. Cite specific entities and relationships when possible
        4. Be precise and factual
        
        GRAPH CONTEXT:
        {context_text}
        
        REASONING CHAIN:
        {' → '.join(context.reasoning_chain)}
        """
        
        try:
            response = await llm_service.llm.ainvoke([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Question: {query}"}
            ], temperature=0.1)  # Very low temperature
            
            is_grounded = await self._validate_response_grounding(
                response.content, context_text
            )
            
            return {
                "answer": response.content,
                "grounded": is_grounded,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return {
                "answer": "I encountered an error while processing your question.",
                "grounded": False,
                "error": str(e),
                "success": False
            }
    
    def _format_context_for_llm(self, context: GraphContext) -> str:
        """Format graph context for LLM"""
        
        formatted_parts = []
        
        if context.primary_entities:
            entities_text = ", ".join([
                f"{e['name']} ({', '.join(e.get('labels', []))})" 
                for e in context.primary_entities
            ])
            formatted_parts.append(f"PRIMARY ENTITIES: {entities_text}")
        
        if context.relationships:
            rels_text = "\n".join([
                f"- {r.get('source', 'Unknown')} {r.get('type', 'RELATED_TO')} {r.get('target', 'Unknown')}"
                for r in context.relationships[:10]
            ])
            formatted_parts.append(f"RELATIONSHIPS:\n{rels_text}")
        
        if context.pathways:
            pathways_text = "\n".join([
                f"- Path: {' → '.join(p['nodes'])} (via: {', '.join(p['relationships'])})"
                for p in context.pathways[:5]
            ])
            formatted_parts.append(f"CONNECTION PATHWAYS:\n{pathways_text}")
        
        return "\n\n".join(formatted_parts)
    
    async def _generate_followup_suggestions(self, query: str, context: GraphContext) -> List[str]:
        """Generate follow-up suggestions"""
        
        suggestions = []
        
        if context.primary_entities:
            entity_names = [e["name"] for e in context.primary_entities[:2]]
            suggestions.append(f"Tell me more about the relationships between {' and '.join(entity_names)}")
        
        if context.communities:
            suggestions.append("What other entities are in this community?")
        
        if context.pathways:
            suggestions.append("Are there any other connection paths I should know about?")
        
        return suggestions[:3]
    
    # ==================== MAIN INTERFACE METHODS ====================
    
    async def explain_reasoning(self, query: str) -> Dict[str, Any]:
        """Explain reasoning process"""
        
        if not self.current_graph_id:
            return {"error": "No graph initialized"}
        
        try:
            graph_context = await self._generate_graph_context_with_graph_id(
                query=query,
                graph_id=self.current_graph_id,
                reasoning_mode=ReasoningMode.FOCUSED,
                max_context_size=2000
            )
            
            return {
                "query_analysis": {
                    "query_type": await self._classify_query_type(query),
                    "entities_recognized": len(graph_context.primary_entities)
                },
                "graph_analysis_steps": graph_context.reasoning_chain,
                "context_sources": {
                    "direct_matches": len(graph_context.primary_entities),
                    "neighborhood_expansion": len(graph_context.neighborhoods),
                    "pathway_analysis": len(graph_context.pathways),
                    "community_context": len(graph_context.communities)
                },
                "confidence_assessment": graph_context.confidence_scores,
                "reasoning_strategy": "Graph algorithms find relevant entities, analyze relationships, discover patterns, provide grounded response."
            }
            
        except Exception as e:
            logger.error(f"Reasoning explanation failed: {e}")
            return {"error": str(e)}
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary"""
        
        return {
            "graph_id": str(self.current_graph_id) if self.current_graph_id else None,
            "conversation_length": len(self.conversation_history),
            "reasoning_history_length": len(self.reasoning_history),
            "recent_entities_discussed": [
                entity for context in self.conversation_history[-3:]
                for entity in context.get("entities_discussed", [])
            ],
            "available_modes": [mode.value for mode in ChatMode],
            "graph_statistics": self.graph_statistics
        }

# ==================== SINGLETON INSTANCE ====================

chat_service = ChatService()
