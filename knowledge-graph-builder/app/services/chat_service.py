import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional

from app.core.neo4j_client import Neo4jClient
from app.core.exceptions import ChatServiceError
from app.models.requests import ChatMode
from app.models.responses import ChatResponse
from app.services.embedding_service import EmbeddingService
from app.utils.llm_clients import LLMClientFactory
from app.config.settings import get_settings

logger = logging.getLogger(__name__)

class ChatService:
    """Service for handling chat interactions with knowledge graph"""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.settings = get_settings()
        self.embedding_service = EmbeddingService()
        self.llm_factory = LLMClientFactory()
    
    async def chat(
        self,
        message: str,
        mode: ChatMode,
        file_names: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> ChatResponse:
        """Process chat message and return response"""
        start_time = time.time()
        
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Get relevant context based on mode
            context = await self._get_context(message, mode, file_names)
            
            # Generate LLM response
            response = await self._generate_response(message, context, mode)
            
            # Store conversation
            await self._store_conversation(session_id, message, response, context)
            
            response_time = time.time() - start_time
            
            return ChatResponse(
                message=response,
                sources=context,
                session_id=session_id,
                response_time=response_time
            )
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise ChatServiceError(f"Chat failed: {e}")
    
    async def _get_context(
        self, 
        message: str, 
        mode: ChatMode, 
        file_names: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get relevant context based on chat mode"""
        
        if mode == ChatMode.VECTOR:
            return await self._get_vector_context(message, file_names)
        
        elif mode == ChatMode.GRAPH_VECTOR:
            vector_context = await self._get_vector_context(message, file_names)
            graph_context = await self._get_graph_context(message, file_names)
            return vector_context + graph_context
        
        elif mode == ChatMode.GRAPH:
            return await self._get_graph_context(message, file_names)
        
        elif mode == ChatMode.FULLTEXT:
            return await self._get_fulltext_context(message, file_names)
        
        elif mode == ChatMode.GRAPH_VECTOR_FULLTEXT:
            vector_context = await self._get_vector_context(message, file_names)
            graph_context = await self._get_graph_context(message, file_names)
            fulltext_context = await self._get_fulltext_context(message, file_names)
            return vector_context + graph_context + fulltext_context
        
        elif mode == ChatMode.ENTITY_VECTOR:
            return await self._get_entity_vector_context(message, file_names)
        
        elif mode == ChatMode.GLOBAL_VECTOR:
            return await self._get_global_vector_context(message, file_names)
        
        return []
    
    async def _get_vector_context(self, message: str, file_names: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Get context using vector similarity search"""
        try:
            # Generate query embedding
            query_embedding = await self.embedding_service.generate_query_embedding(message)
            
            # Build file filter
            file_filter = ""
            params = {"queryEmbedding": query_embedding, "limit": 5}
            
            if file_names:
                file_filter = "MATCH (d:Document) WHERE d.fileName IN $fileNames WITH collect(d) as docs MATCH (doc)-[:HAS_CHUNK]->(chunk) WHERE doc IN docs"
                params["fileNames"] = file_names
            
            query = f"""
            {file_filter}
            CALL db.index.vector.queryNodes('vector', $limit, $queryEmbedding) 
            YIELD node AS chunk, score
            MATCH (d:Document)-[:HAS_CHUNK]->(chunk)
            RETURN chunk.id as chunkId, 
                   chunk.text as text, 
                   d.fileName as fileName,
                   score
            ORDER BY score DESC
            """
            
            result = self.neo4j.execute_query(query, params)
            
            context = []
            for record in result:
                context.append({
                    "type": "chunk",
                    "chunk_id": record["chunkId"],
                    "text": record["text"],
                    "file_name": record["fileName"],
                    "score": record["score"],
                    "source": "vector_search"
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Vector context retrieval failed: {e}")
            return []
    
    async def _get_graph_context(self, message: str, file_names: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Get context using graph traversal"""
        try:
            # Extract entities from message using NER
            entities = await self._extract_entities_from_text(message)
            
            if not entities:
                return []
            
            # Find related entities and relationships
            file_filter = ""
            params = {"entityNames": entities, "limit": 10}
            
            if file_names:
                file_filter = """
                MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk)-[:HAS_ENTITY]->(e)
                WHERE d.fileName IN $fileNames
                WITH collect(DISTINCT e) as docEntities
                MATCH (entity) WHERE entity IN docEntities AND
                """
                params["fileNames"] = file_names
            else:
                file_filter = "MATCH (entity:Entity) WHERE "
            
            query = f"""
            {file_filter}
            (entity.name IN $entityNames OR entity.id IN $entityNames)
            MATCH (entity)-[r]-(related:Entity)
            MATCH (c:Chunk)-[:HAS_ENTITY]->(entity)
            RETURN DISTINCT entity.name as entityName,
                   related.name as relatedEntity,
                   type(r) as relationshipType,
                   c.text as chunkText,
                   c.id as chunkId
            LIMIT $limit
            """
            
            result = self.neo4j.execute_query(query, params)
            
            context = []
            for record in result:
                context.append({
                    "type": "graph",
                    "entity": record["entityName"],
                    "related_entity": record["relatedEntity"],
                    "relationship": record["relationshipType"],
                    "chunk_text": record["chunkText"],
                    "chunk_id": record["chunkId"],
                    "source": "graph_traversal"
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Graph context retrieval failed: {e}")
            return []
    
    async def _get_fulltext_context(self, message: str, file_names: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Get context using fulltext search"""
        try:
            # Build fulltext search query
            search_terms = message.split()[:5]  # Use first 5 words
            
            file_filter = ""
            params = {"searchTerms": search_terms, "limit": 5}
            
            if file_names:
                file_filter = "AND d.fileName IN $fileNames"
                params["fileNames"] = file_names
            
            query = f"""
            CALL db.index.fulltext.queryNodes('chunkText', $searchQuery) 
            YIELD node AS chunk, score
            MATCH (d:Document)-[:HAS_CHUNK]->(chunk)
            WHERE score > 1 {file_filter}
            RETURN chunk.id as chunkId,
                   chunk.text as text,
                   d.fileName as fileName,
                   score
            ORDER BY score DESC
            LIMIT $limit
            """
            
            params["searchQuery"] = " ".join(search_terms)
            result = self.neo4j.execute_query(query, params)
            
            context = []
            for record in result:
                context.append({
                    "type": "chunk",
                    "chunk_id": record["chunkId"],
                    "text": record["text"],
                    "file_name": record["fileName"],
                    "score": record["score"],
                    "source": "fulltext_search"
                })
            
            return context
            
        except Exception as e:
            logger.error(f"Fulltext context retrieval failed: {e}")
            return []
    
    async def _get_entity_vector_context(self, message: str, file_names: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Get context using entity embeddings"""
        # Placeholder - would need entity embeddings implementation
        return []
    
    async def _get_global_vector_context(self, message: str, file_names: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Get context using global document embeddings"""
        # Placeholder - would need document-level embeddings
        return []
    
    async def _extract_entities_from_text(self, text: str) -> List[str]:
        """Extract entities from text using simple NER"""
        try:
            # Simple entity extraction - could be improved with proper NER
            llm = self.llm_factory.get_llm(self.settings.default_llm_model)
            
            prompt = f"""
            Extract the key entities (people, places, organizations, concepts) from this text:
            "{text}"
            
            Return only the entity names, one per line, without explanations.
            """
            
            response = await asyncio.to_thread(llm.predict, prompt)
            entities = [line.strip() for line in response.split('\n') if line.strip()]
            
            return entities[:10]  # Limit to 10 entities
            
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            return []
    
    async def _generate_response(self, message: str, context: List[Dict[str, Any]], mode: ChatMode) -> str:
        """Generate LLM response using context"""
        try:
            llm = self.llm_factory.get_llm(self.settings.default_llm_model)
            
            # Build context string
            context_str = self._format_context(context)
            
            # Create prompt
            prompt = f"""
            You are a helpful assistant that answers questions based on the provided context from a knowledge graph.
            
            Context:
            {context_str}
            
            Question: {message}
            
            Please provide a comprehensive answer based on the context. If the context doesn't contain enough information to answer the question, say so clearly.
            """
            
            response = await asyncio.to_thread(llm.predict, prompt)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I encountered an error while generating a response. Please try again."
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context for LLM prompt"""
        if not context:
            return "No relevant context found."
        
        formatted_parts = []
        
        for item in context:
            if item.get("type") == "chunk":
                formatted_parts.append(f"Text: {item.get('text', '')}")
            elif item.get("type") == "graph":
                formatted_parts.append(
                    f"Entity: {item.get('entity', '')} -> "
                    f"Relationship: {item.get('relationship', '')} -> "
                    f"Related Entity: {item.get('related_entity', '')}"
                )
        
        return "\n\n".join(formatted_parts[:10])  # Limit context size
    
    async def _store_conversation(self, session_id: str, message: str, response: str, context: List[Dict[str, Any]]) -> None:
        """Store conversation in Neo4j"""
        try:
            query = """
            MERGE (s:Session {id: $sessionId})
            CREATE (c:Conversation {
                id: $conversationId,
                message: $message,
                response: $response,
                context: $context,
                timestamp: datetime()
            })
            CREATE (s)-[:HAS_CONVERSATION]->(c)
            """
            
            self.neo4j.execute_write_query(query, {
                "sessionId": session_id,
                "conversationId": str(uuid.uuid4()),
                "message": message,
                "response": response,
                "context": str(context)  # Store as string for simplicity
            })
            
        except Exception as e:
            logger.warning(f"Failed to store conversation: {e}")
    
    async def clear_chat_history(self, session_id: str) -> None:
        """Clear chat history for a session"""
        query = """
        MATCH (s:Session {id: $sessionId})-[:HAS_CONVERSATION]->(c:Conversation)
        DETACH DELETE c
        """
        
        self.neo4j.execute_write_query(query, {"sessionId": session_id})
    
    async def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get chat history for a session"""
        query = """
        MATCH (s:Session {id: $sessionId})-[:HAS_CONVERSATION]->(c:Conversation)
        RETURN c.message as message, 
               c.response as response, 
               c.timestamp as timestamp
        ORDER BY c.timestamp DESC
        LIMIT $limit
        """
        
        result = self.neo4j.execute_query(query, {
            "sessionId": session_id,
            "limit": limit
        })
        
        return result
