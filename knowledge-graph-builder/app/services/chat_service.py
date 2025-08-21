from typing import Dict, Any, List, Optional, Tuple
from uuid import UUID
from langchain.chains import GraphCypherQAChain
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import json
import asyncio

from app.core.neo4j_client import neo4j_client
from app.services.llm_service import llm_service
from app.services.search_service import search_service
from app.services.embedding_service import embedding_service
from app.core.logging import get_logger

logger = get_logger(__name__)

class ChatService:
    """Advanced chat service with GraphRAG and multiple retrieval modes"""
    
    def __init__(self):
        self.cypher_chain = None
        self.current_graph_id = None
        self.conversation_history = []
    
    async def initialize_chat(
        self, 
        graph_id: UUID, 
        user_id: str,
        provider: str = "openai",
        model: str = "gpt-4o-mini"
    ) -> bool:
        """Initialize chat service for a specific graph"""
        
        try:
            # Initialize LLM if needed
            if not llm_service.is_initialized():
                success = await llm_service.initialize_llm(
                    user_id=user_id,
                    provider=provider,
                    model=model
                )
                if not success:
                    return False
            
            # Initialize embeddings for vector search
            if not embedding_service.is_initialized():
                await embedding_service.initialize_embeddings(
                    provider="openai",
                    user_id=user_id
                )
            
            # Get graph schema for Cypher generation
            schema = await self._get_graph_schema(graph_id)
            
            # Create Cypher QA chain
            self.cypher_chain = await self._create_cypher_chain(schema)
            self.current_graph_id = graph_id
            
            logger.info(f"Chat initialized for graph {graph_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize chat: {e}")
            return False
    
    async def chat_with_graph(
        self,
        query: str,
        mode: str = "graph_vector",
        graph_id: Optional[UUID] = None,
        conversation_id: Optional[str] = None,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """Main chat interface with multiple retrieval modes"""
        
        if graph_id and graph_id != self.current_graph_id:
            logger.warning(f"Graph ID mismatch: expected {self.current_graph_id}, got {graph_id}")
            return {
                "answer": "Error: Chat not initialized for this graph",
                "mode": mode,
                "success": False
            }
        
        try:
            # Add to conversation history
            if include_history:
                self.conversation_history.append({
                    "type": "user",
                    "content": query,
                    "timestamp": "now"
                })
            
            # Route to appropriate retrieval method
            if mode == "vector":
                result = await self._vector_search_chat(query)
            elif mode == "graph":
                result = await self._graph_cypher_chat(query)
            elif mode == "graph_vector":
                result = await self._hybrid_graph_vector_chat(query)
            elif mode == "graphrag":
                result = await self._graphrag_chat(query)
            else:
                raise ValueError(f"Unsupported chat mode: {mode}")
            
            # Add response to history
            if include_history:
                self.conversation_history.append({
                    "type": "assistant", 
                    "content": result["answer"],
                    "timestamp": "now",
                    "metadata": result.get("metadata", {})
                })
            
            result["mode"] = mode
            result["conversation_id"] = conversation_id
            return result
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "mode": mode,
                "success": False,
                "error": str(e)
            }
    
    async def _vector_search_chat(self, query: str) -> Dict[str, Any]:
        """Chat using vector search on entities and chunks"""
        
        try:
            # Search entities
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
            
            # Generate answer using LLM with context
            answer_prompt = f"""
            Based on the following information from a knowledge graph, please answer the user's question.
            
            Question: {query}
            
            Relevant Entities:
            {context['entities']}
            
            Relevant Text:
            {context['chunks']}
            
            Please provide a comprehensive answer based on this information. If the information is insufficient, say so.
            """
            
            response = await llm_service.llm.ainvoke(answer_prompt)
            
            return {
                "answer": response.content,
                "sources": {
                    "entities": entity_results[:3],
                    "chunks": chunk_results[:2]
                },
                "method": "vector_search",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Vector search chat failed: {e}")
            return {
                "answer": "I couldn't find relevant information using vector search.",
                "success": False,
                "error": str(e)
            }
    
    async def _graph_cypher_chat(self, query: str) -> Dict[str, Any]:
        """Chat using Cypher query generation"""
        
        try:
            # Get graph schema
            schema = await self._get_graph_schema(self.current_graph_id)
            
            # Generate Cypher query
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
                "raw_results": results[:5],  # Limit results shown
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
    
    async def _hybrid_graph_vector_chat(self, query: str) -> Dict[str, Any]:
        """Hybrid approach combining graph queries and vector search"""
        
        try:
            # Try both approaches
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
                # Both succeeded - combine answers
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
                # Only vector succeeded
                vector_result["method"] = "hybrid_fallback_vector"
                return vector_result
            
            elif cypher_result["success"]:
                # Only cypher succeeded
                cypher_result["method"] = "hybrid_fallback_cypher"
                return cypher_result
            
            else:
                # Both failed
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
    
    async def _graphrag_chat(self, query: str) -> Dict[str, Any]:
        """Advanced GraphRAG implementation"""
        
        try:
            # Step 1: Entity extraction from query
            query_entities = await self._extract_entities_from_query(query)
            
            # Step 2: Vector search for relevant entities and chunks
            similar_entities = await search_service.similarity_search_entities(
                query=query,
                graph_id=self.current_graph_id,
                k=8,
                threshold=0.6
            )
            
            # Step 3: Get neighborhoods of relevant entities
            neighborhoods = []
            for entity in similar_entities[:5]:
                neighborhood = await self._get_entity_neighborhood(
                    entity["id"], max_depth=2
                )
                neighborhoods.append(neighborhood)
            
            # Step 4: Get relevant text chunks
            text_chunks = []
            try:
                text_chunks = await search_service.similarity_search_chunks(
                    query=query,
                    graph_id=self.current_graph_id,
                    k=5,
                    threshold=0.6
                )
            except Exception:
                pass
            
            # Step 5: Build comprehensive context
            rag_context = self._build_graphrag_context(
                query_entities, similar_entities, neighborhoods, text_chunks
            )
            
            # Step 6: Generate answer with rich context
            answer = await self._generate_graphrag_answer(query, rag_context)
            
            return {
                "answer": answer,
                "context": {
                    "query_entities": query_entities,
                    "similar_entities": similar_entities[:3],
                    "neighborhoods": len(neighborhoods),
                    "text_chunks": len(text_chunks)
                },
                "method": "graphrag",
                "success": True
            }
            
        except Exception as e:
            logger.error(f"GraphRAG chat failed: {e}")
            return {
                "answer": "I couldn't process your question using GraphRAG.",
                "success": False,
                "error": str(e)
            }
    
    async def _natural_language_to_cypher(
        self, 
        question: str, 
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert natural language to Cypher query"""
        
        try:
            # Create Cypher generation prompt
            cypher_prompt = f"""
            Given this Neo4j graph schema:
            
            Node Types: {schema.get('entities', [])}
            Relationship Types: {schema.get('relationships', [])}
            
            Sample node structure:
            - All nodes have: id, graph_id, name
            - Nodes may have: description, type, properties
            - All data is filtered by graph_id = "{self.current_graph_id}"
            
            Convert this natural language question to a Cypher query:
            "{question}"
            
            Rules:
            1. ALWAYS include "WHERE n.graph_id = '{self.current_graph_id}'" for all nodes
            2. Use LIMIT to avoid large result sets (max 20 results)
            3. Return meaningful property names
            4. Handle case-insensitive matching with toLower()
            
            Return only the Cypher query, no explanation:
            """
            
            response = await llm_service.llm.ainvoke(cypher_prompt)
            cypher_query = response.content.strip()
            
            # Clean up the query
            cypher_query = cypher_query.replace("```cypher", "").replace("```", "").strip()
            
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
    
    async def _get_graph_schema(self, graph_id: UUID) -> Dict[str, Any]:
        """Get graph schema for Cypher generation"""
        
        try:
            # Get node labels (entity types)
            labels_query = """
            MATCH (n)
            WHERE n.graph_id = $graph_id
            RETURN DISTINCT labels(n) as labels
            LIMIT 50
            """
            
            # Get relationship types
            rels_query = """
            MATCH ()-[r]->()
            WHERE r.graph_id = $graph_id
            RETURN DISTINCT type(r) as relationship_type
            LIMIT 30
            """
            
            labels_result = await neo4j_client.execute_query(
                labels_query, {"graph_id": str(graph_id)}
            )
            rels_result = await neo4j_client.execute_query(
                rels_query, {"graph_id": str(graph_id)}
            )
            
            # Extract unique entity types
            entity_types = set()
            for record in labels_result:
                for label in record["labels"]:
                    if label not in ["Entity"]:  # Skip generic labels
                        entity_types.add(label)
            
            relationship_types = [r["relationship_type"] for r in rels_result]
            
            return {
                "entities": sorted(list(entity_types)),
                "relationships": sorted(relationship_types)
            }
            
        except Exception as e:
            logger.error(f"Failed to get graph schema: {e}")
            return {"entities": [], "relationships": []}
    
    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entities from user query"""
        
        try:
            entity_prompt = f"""
            Extract the main entities (people, organizations, concepts, etc.) from this question:
            "{query}"
            
            Return only a JSON list of entity names, like: ["Entity1", "Entity2"]
            If no clear entities, return: []
            """
            
            response = await llm_service.llm.ainvoke(entity_prompt)
            entities = json.loads(response.content.strip())
            return entities if isinstance(entities, list) else []
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    async def _get_entity_neighborhood(
        self, 
        entity_id: str, 
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Get the neighborhood of an entity"""
        
        try:
            query = f"""
            MATCH (center {{id: $entity_id, graph_id: $graph_id}})
            CALL {{
                WITH center
                MATCH path = (center)-[*1..{max_depth}]-(neighbor)
                WHERE neighbor.graph_id = $graph_id
                RETURN neighbor, relationships(path) as rels
                LIMIT 10
            }}
            RETURN center, collect({{neighbor: neighbor, relationships: rels}}) as neighborhood
            """
            
            result = await neo4j_client.execute_query(query, {
                "entity_id": entity_id,
                "graph_id": str(self.current_graph_id)
            })
            
            return result[0] if result else {}
            
        except Exception as e:
            logger.warning(f"Neighborhood extraction failed: {e}")
            return {}
    
    def _build_vector_context(
        self, 
        entity_results: List[Dict], 
        chunk_results: List[Dict]
    ) -> Dict[str, str]:
        """Build context from vector search results"""
        
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
        """Build comprehensive GraphRAG context"""
        
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
    
    async def _generate_answer_from_cypher_results(
        self, 
        query: str, 
        results: List[Dict]
    ) -> str:
        """Generate natural language answer from Cypher results"""
        
        if not results:
            return "I couldn't find any relevant information in the graph."
        
        # Format results for LLM
        formatted_results = json.dumps(results[:10], indent=2, default=str)
        
        answer_prompt = f"""
        Based on these query results from a knowledge graph, provide a natural language answer to the question.
        
        Question: {query}
        
        Query Results:
        {formatted_results}
        
        Please provide a clear, concise answer based on this data. If the results don't fully answer the question, mention what information is available.
        """
        
        response = await llm_service.llm.ainvoke(answer_prompt)
        return response.content
    
    async def _generate_graphrag_answer(
        self, 
        query: str, 
        context: str
    ) -> str:
        """Generate answer using GraphRAG context"""
        
        rag_prompt = f"""
        You are an AI assistant with access to a knowledge graph. Answer the user's question using the provided context.
        
        Question: {query}
        
        Context from Knowledge Graph:
        {context}
        
        Instructions:
        1. Provide a comprehensive answer based on the context
        2. Mention specific entities and relationships when relevant
        3. If information is incomplete, acknowledge this
        4. Be conversational but accurate
        
        Answer:
        """
        
        response = await llm_service.llm.ainvoke(rag_prompt)
        return response.content
    
    async def _combine_answers(
        self, 
        query: str, 
        vector_result: Dict, 
        cypher_result: Dict
    ) -> str:
        """Intelligently combine answers from different methods"""
        
        combine_prompt = f"""
        I have two answers to the question "{query}" from different methods. Please combine them into one comprehensive answer.
        
        Vector Search Answer: {vector_result['answer']}
        
        Graph Query Answer: {cypher_result['answer']}
        
        Please provide a single, well-structured answer that incorporates the best insights from both approaches. Remove redundancy and ensure the final answer is coherent.
        """
        
        response = await llm_service.llm.ainvoke(combine_prompt)
        return response.content
    
    async def _create_cypher_chain(self, schema: Dict[str, Any]):
        """Create Cypher QA chain (placeholder for future langchain integration)"""
        
        # For now, we'll use our custom implementation
        # In the future, this could integrate with LangChain's GraphCypherQAChain
        return None

# Global chat service
chat_service = ChatService()
