#!/usr/bin/env python3
"""
Advanced GraphRAG Conversational AI Agent

A sophisticated chatbot implementation that leverages the complete GraphRAG knowledge graph
pipeline for intelligent, context-aware conversations. Features multi-strategy retrieval,
conversation memory, and advanced reasoning capabilities.

Features:
- Multi-strategy GraphRAG retrieval (Vector, Hybrid, Text2Cypher, Entity-based)
- Conversation memory and context management
- Real-time knowledge graph querying
- Intelligent retrieval strategy selection
- Performance monitoring and logging
- Streaming responses for better UX
- Citation and source tracking
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, AsyncGenerator
from enum import Enum

# Neo4j GraphRAG imports
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.retrievers import (
    VectorRetriever,
    VectorCypherRetriever,
    HybridRetriever,
    Text2CypherRetriever
)

# Import our advanced pipeline
from benchmark import AdvancedGraphRAGPipeline, AdvancedPipelineConfig, LoggingText2CypherRetriever


class QueryType(Enum):
    """Classification of query types for retrieval strategy selection"""
    FACTUAL = "factual"
    RELATIONSHIP = "relationship"
    SEMANTIC = "semantic"
    ANALYTICAL = "analytical"
    CONVERSATIONAL = "conversational"


class RetrievalStrategy(Enum):
    """Available retrieval strategies"""
    VECTOR = "vector"
    HYBRID = "hybrid"
    TEXT2CYPHER = "text2cypher"
    VECTOR_CYPHER = "vector_cypher"
    ENTITY_FOCUSED = "entity_focused"
    RELATIONSHIP_FOCUSED = "relationship_focused"  # NEW
    MULTI_STRATEGY = "multi_strategy"


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sources: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Manages conversation context and memory"""
    messages: List[ChatMessage] = field(default_factory=list)
    entities_mentioned: List[str] = field(default_factory=list)
    topics_discussed: List[str] = field(default_factory=list)
    current_focus: Optional[str] = None
    session_id: str = field(default_factory=lambda: f"session_{int(time.time())}")
    
    def add_message(self, message: ChatMessage):
        """Add a message to conversation history"""
        self.messages.append(message)
        
        # Extract entities and topics from message
        if message.role == "user":
            self._extract_entities_and_topics(message.content)
    
    def _extract_entities_and_topics(self, content: str):
        """Extract entities and topics from user message (simplified)"""
        # This could be enhanced with NER and topic modeling
        words = content.lower().split()
        
        # Simple entity detection (could be improved with NER)
        potential_entities = [word.title() for word in words if word.istitle()]
        self.entities_mentioned.extend(potential_entities)
        
        # Keep only recent entities (last 20)
        self.entities_mentioned = list(set(self.entities_mentioned))[-20:]
    
    def get_conversation_summary(self) -> str:
        """Get a summary of the conversation for context"""
        if not self.messages:
            return "No previous conversation context."
        
        recent_messages = self.messages[-6:]  # Last 3 exchanges
        summary_parts = []
        
        for msg in recent_messages:
            role_prefix = "User" if msg.role == "user" else "Assistant"
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"{role_prefix}: {content_preview}")
        
        context_info = ""
        if self.entities_mentioned:
            context_info += f"\nEntities discussed: {', '.join(self.entities_mentioned[-5:])}"
        if self.current_focus:
            context_info += f"\nCurrent focus: {self.current_focus}"
        
        return "\n".join(summary_parts) + context_info


class QueryClassifier:
    """Classifies user queries to select optimal retrieval strategy"""
    
    def __init__(self):
        self.factual_keywords = {
            "what", "who", "when", "where", "how many", "which", "define", "explain"
        }
        self.relationship_keywords = {
            "relationship", "connected", "related", "works for", "founded by", 
            "partners with", "owns", "leads", "manages", "between"
        }
        self.analytical_keywords = {
            "analyze", "compare", "contrast", "evaluate", "assess", "why", 
            "how", "impact", "effect", "influence", "trends"
        }
    
    def classify_query(self, query: str, context: ConversationContext) -> Tuple[QueryType, RetrievalStrategy]:
        """Classify query and recommend retrieval strategy"""
        query_lower = query.lower()
        
        # Check for relationship queries
        if any(keyword in query_lower for keyword in self.relationship_keywords):
            return QueryType.RELATIONSHIP, RetrievalStrategy.TEXT2CYPHER
        
        # Check for analytical queries
        if any(keyword in query_lower for keyword in self.analytical_keywords):
            return QueryType.ANALYTICAL, RetrievalStrategy.MULTI_STRATEGY
        
        # Check for factual queries
        if any(keyword in query_lower for keyword in self.factual_keywords):
            return QueryType.FACTUAL, RetrievalStrategy.HYBRID
        
        # Check if specific entities are mentioned
        if any(entity.lower() in query_lower for entity in context.entities_mentioned):
            return QueryType.SEMANTIC, RetrievalStrategy.VECTOR_CYPHER
        
        # Enhanced relationship detection
        if any(keyword in query_lower for keyword in self.relationship_keywords):
            # Check if query is specifically about relationship types/patterns
            if any(word in query_lower for word in ["how", "what kind", "type of", "nature of"]):
                return QueryType.RELATIONSHIP, RetrievalStrategy.RELATIONSHIP_FOCUSED
            else:
                return QueryType.RELATIONSHIP, RetrievalStrategy.TEXT2CYPHER
        
        # Default to semantic search
        return QueryType.SEMANTIC, RetrievalStrategy.VECTOR


class GraphRAGChatbot:
    """Advanced GraphRAG-powered conversational AI agent"""
    
    def __init__(self, config: AdvancedPipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Core components
        self.driver = None
        self.llm = None
        self.embedder = None
        
        # Retrievers
        self.vector_retriever = None
        self.hybrid_retriever = None
        self.text2cypher_retriever = None
        self.vector_cypher_retriever = None
        self.entity_retriever = None
        
        # Conversation management
        self.conversations: Dict[str, ConversationContext] = {}
        self.query_classifier = QueryClassifier()
        
        # Performance tracking
        self.query_count = 0
        self.total_response_time = 0.0
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for chatbot"""
        logger = logging.getLogger("graphrag_chatbot")
        
        log_dir = Path("logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'chatbot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        
        return logger
    
    async def initialize(self):
        """Initialize the chatbot with all retrievers"""
        self.logger.info("🤖 Initializing GraphRAG Chatbot...")
        
        try:
            # Initialize Neo4j driver
            self.driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password)
            )
            
            # Test connection
            with self.driver.session(database=self.config.neo4j_database) as session:
                result = session.run("RETURN 1 as test")
                if result.single()["test"] != 1:
                    raise Exception("Neo4j connection test failed")
            
            self.logger.info("✅ Neo4j connection established")
            
            # Initialize LLM
            self.llm = OpenAILLM(
                model_name=self.config.llm_model,
                api_key=self.config.openai_api_key,
                model_params={
                    "temperature": 0.7,  # Slightly higher for conversation
                    "max_tokens": 2000,
                    "response_format": {"type": "text"}  # Text format for chat
                }
            )
            
            # Initialize embedder
            self.embedder = OpenAIEmbeddings(
                model=self.config.embedding_model,
                api_key=self.config.openai_api_key
            )
            
            self.logger.info("✅ LLM and embedder initialized")
            
            # Initialize all retrievers
            await self._initialize_retrievers()
            
            self.logger.info("🎉 GraphRAG Chatbot ready for conversations!")
            
        except Exception as e:
            self.logger.error(f"❌ Chatbot initialization failed: {e}")
            raise
    
    async def _initialize_retrievers(self):
        """Initialize all retrieval strategies"""
        try:
            # 1. Vector Retriever
            self.vector_retriever = VectorRetriever(
                driver=self.driver,
                index_name="text_embeddings_primary",
                embedder=self.embedder,
                return_properties=["text", "chunk_index"]
            )
            
            # 2. Hybrid Retriever
            self.hybrid_retriever = HybridRetriever(
                driver=self.driver,
                vector_index_name="text_embeddings_primary",
                fulltext_index_name="chunk_text_fulltext",
                embedder=self.embedder
            )
            
            # 3. Vector + Cypher Retriever
            self.vector_cypher_retriever = VectorCypherRetriever(
                driver=self.driver,
                index_name="text_embeddings_primary",
                embedder=self.embedder,
                retrieval_query="""
                WITH node AS chunk, score
                MATCH (chunk)<-[:FROM_CHUNK]-(entity:__Entity__)-[r]->(related_entity:__Entity__)
                WHERE r.confidence > 0.5
                RETURN 
                    chunk.text AS context,
                    chunk.chunk_index AS chunk_index,
                    collect(DISTINCT {
                        entity: entity.name,
                        type: labels(entity)[0],
                        relationship: type(r),
                        related_entity: related_entity.name,
                        confidence: r.confidence
                    }) AS knowledge_graph_context,
                    score
                ORDER BY score DESC
                """
            )
            
            # 4. Entity-focused Vector Retriever
            self.entity_retriever = VectorRetriever(
                driver=self.driver,
                index_name="entity_embeddings",
                embedder=self.embedder,
                return_properties=["name", "description", "type"]
            )
            
            # 5. Text2Cypher Retriever with enhanced schema
            await self._initialize_text2cypher()

            # 6. Relationship Embedding Retriever (NEW)
            self.relationship_retriever = VectorRetriever(
                driver=self.driver,
                index_name="relationship_embeddings",  # Assuming you have this index
                embedder=self.embedder,
                return_properties=["description", "source_entity", "target_entity", "type", "confidence"]
            )
            
            # 7. Enhanced Vector Cypher with Relationship Embeddings (NEW)
            self.enhanced_vector_cypher_retriever = VectorCypherRetriever(
                driver=self.driver,
                index_name="relationship_embeddings",
                embedder=self.embedder,
                retrieval_query="""
                WITH node AS rel, score
                MATCH (source)-[r]->(target)
                WHERE id(r) = id(rel)
                RETURN 
                    rel.description AS relationship_context,
                    rel.type AS relationship_type,
                    source.name AS source_entity,
                    target.name AS target_entity,
                    rel.confidence AS confidence,
                    score,
                    rel.embedding_text AS embedding_source
                ORDER BY score DESC
                """
            )
            
            self.logger.info("✅ All retrievers initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Retriever initialization failed: {e}")
            raise
    
    async def _initialize_text2cypher(self):
        """Initialize Text2Cypher with comprehensive schema"""
        try:
            # Get real relationships from the database
            with self.driver.session(database=self.config.neo4j_database) as session:
                result = session.run("""
                    CALL db.relationshipTypes() YIELD relationshipType
                    RETURN collect(relationshipType) as relationships
                """)
                relationships = result.single()["relationships"]
                
                # Get sample entities for better examples
                entity_result = session.run("""
                    MATCH (e:__Entity__)
                    RETURN e.name as name
                    ORDER BY e.name
                    LIMIT 10
                """)
                sample_entities = [record["name"] for record in entity_result]
            
            # Build comprehensive schema
            enhanced_schema = f"""
            # GraphRAG Knowledge Graph Schema
            
            ## Node Types:
            - Document: Source documents with properties: path, metadata
            - Chunk: Text chunks with properties: text, chunk_index, embedding
            - __Entity__: All extracted entities with properties: name, description, type, embedding
            
            ## Relationship Types:
            {', '.join(relationships)}
            
            ## Key Relationships:
            - FROM_DOCUMENT: (Chunk)-[:FROM_DOCUMENT]->(Document)
            - FROM_CHUNK: (__Entity__)-[:FROM_CHUNK]->(Chunk)
            - SAME_AS: (__Entity__)-[:SAME_AS]-(__Entity__)
            
            ## Sample Entities:
            {', '.join(sample_entities[:5])}
            
            ## Important Notes:
            - Use __Entity__ label for all entities regardless of type
            - Relationships have confidence scores (r.confidence property)
            - Always check relationship existence before using in queries
            """
            
            # Enhanced examples with real data
            examples = [
                f"Find all entities -> MATCH (e:__Entity__) RETURN e.name LIMIT 10",
                f"Show relationships for {sample_entities[0] if sample_entities else 'TechNova'} -> MATCH (e1:__Entity__ {{name: '{sample_entities[0] if sample_entities else 'TechNova'}'}})-[r]->(e2:__Entity__) RETURN e1.name, type(r), e2.name, r.confidence",
                f"Get high confidence relationships -> MATCH (e1:__Entity__)-[r]->(e2:__Entity__) WHERE r.confidence > 0.8 RETURN e1.name, type(r), e2.name, r.confidence ORDER BY r.confidence DESC LIMIT 10",
                f"Find entities by type -> MATCH (e:__Entity__) WHERE e.type CONTAINS 'Person' RETURN e.name, e.description LIMIT 10",
                f"Get document context for entity -> MATCH (e:__Entity__)-[:FROM_CHUNK]->(c:Chunk)-[:FROM_DOCUMENT]->(d:Document) WHERE e.name = $entity_name RETURN d.path, c.text"
            ]
            
            # Custom prompt for conversational context
            custom_prompt = """
            You are a GraphRAG knowledge graph query assistant. Generate a Cypher query to answer the user's question.
            
            Available Schema:
            {schema}
            
            Query Examples:
            {examples}
            
            Conversation Context:
            {conversation_context}
            
            User Question: {query_text}
            
            Generate a single, executable Cypher query that answers the question. Rules:
            - Use only relationships and properties that exist in the schema
            - Be syntactically correct
            - Include confidence filters for relationships when relevant (r.confidence > 0.5)
            - Limit results appropriately (use LIMIT clause)
            - Return meaningful data to answer the question
            
            Output ONLY the Cypher query:
            """
            
            text2cypher_retriever = Text2CypherRetriever(
                driver=self.driver,
                llm=self.llm,
                neo4j_schema=enhanced_schema,
                custom_prompt=custom_prompt,
                examples=examples
            )
            
            # Wrap with logging
            self.text2cypher_retriever = LoggingText2CypherRetriever(text2cypher_retriever, self.logger)
            
            self.logger.info("✅ Text2Cypher retriever initialized with real schema")
            
        except Exception as e:
            self.logger.error(f"❌ Text2Cypher initialization failed: {e}")
            raise
    
    async def _relationship_search(self, query: str) -> Dict[str, Any]:
        """Relationship embedding search"""
        try:
            result = self.relationship_retriever.search(query_text=query, top_k=5)
            
            sources = []
            for record in result.records:
                sources.append({
                    "content": f"Relationship: {record.get('source_entity', '')} → {record.get('target_entity', '')}\nType: {record.get('type', '')}\nDescription: {record.get('description', '')}",
                    "relationship_type": record.get('type', ''),
                    "source_entity": record.get('source_entity', ''),
                    "target_entity": record.get('target_entity', ''),
                    "confidence": record.get('confidence', 0.0),
                    "score": record.get("score", 0.0),
                    "source_type": "relationship_embedding_search"
                })
            
            return {"sources": sources, "strategy": "relationship_focused"}
            
        except Exception as e:
            self.logger.error(f"Relationship search failed: {e}")
            return {"sources": [], "error": str(e)}

    
    async def chat(self, user_message: str, session_id: str = None) -> ChatMessage:
        """Main chat interface - process user message and return response"""
        start_time = time.time()
        
        try:
            # Get or create conversation context
            if session_id is None:
                session_id = f"session_{int(time.time())}"
            
            if session_id not in self.conversations:
                self.conversations[session_id] = ConversationContext(session_id=session_id)
            
            context = self.conversations[session_id]
            
            # Add user message to context
            user_msg = ChatMessage(role="user", content=user_message)
            context.add_message(user_msg)
            
            self.logger.info(f"🗣️  User [{session_id}]: {user_message}")
            
            # Classify query and select retrieval strategy
            query_type, retrieval_strategy = self.query_classifier.classify_query(user_message, context)
            
            self.logger.info(f"🎯 Query classified as {query_type.value} -> using {retrieval_strategy.value}")
            
            # Retrieve relevant information
            retrieval_results = await self._retrieve_information(user_message, retrieval_strategy, context)
            
            # Generate response
            response_content = await self._generate_response(
                user_message, retrieval_results, context, query_type
            )
            
            # Create assistant message
            assistant_msg = ChatMessage(
                role="assistant",
                content=response_content,
                sources=retrieval_results.get("sources", []),
                retrieval_info={
                    "strategy": retrieval_strategy.value,
                    "query_type": query_type.value,
                    "results_count": len(retrieval_results.get("sources", [])),
                    "retrieval_time": retrieval_results.get("retrieval_time", 0)
                }
            )
            
            # Add to context
            context.add_message(assistant_msg)
            
            # Update performance metrics
            response_time = time.time() - start_time
            self.query_count += 1
            self.total_response_time += response_time
            
            self.logger.info(f"🤖 Assistant [{session_id}]: {response_content[:100]}... (took {response_time:.2f}s)")
            
            return assistant_msg
            
        except Exception as e:
            self.logger.error(f"❌ Chat processing failed: {e}")
            
            error_msg = ChatMessage(
                role="assistant",
                content=f"I apologize, but I encountered an error while processing your request: {str(e)}. Please try rephrasing your question or ask something else.",
                metadata={"error": str(e)}
            )
            
            if session_id in self.conversations:
                self.conversations[session_id].add_message(error_msg)
            
            return error_msg
    
    async def _retrieve_information(self, query: str, strategy: RetrievalStrategy, context: ConversationContext) -> Dict[str, Any]:
        """Retrieve information using the specified strategy"""
        start_time = time.time()
        
        try:
            if strategy == RetrievalStrategy.VECTOR:
                results = await self._vector_search(query)
            elif strategy == RetrievalStrategy.HYBRID:
                results = await self._hybrid_search(query)
            elif strategy == RetrievalStrategy.TEXT2CYPHER:
                results = await self._text2cypher_search(query, context)
            elif strategy == RetrievalStrategy.VECTOR_CYPHER:
                results = await self._vector_cypher_search(query)
            elif strategy == RetrievalStrategy.ENTITY_FOCUSED:
                results = await self._entity_search(query)
            elif strategy == RetrievalStrategy.RELATIONSHIP_FOCUSED:  # NEW
                results = await self._relationship_search(query)    
            elif strategy == RetrievalStrategy.MULTI_STRATEGY:
                results = await self._multi_strategy_search(query, context)
            else:
                results = await self._vector_search(query)  # Default fallback
            
            retrieval_time = time.time() - start_time
            results["retrieval_time"] = retrieval_time
            
            self.logger.info(f"📊 Retrieved {len(results.get('sources', []))} results in {retrieval_time:.2f}s using {strategy.value}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Retrieval failed for strategy {strategy.value}: {e}")
            return {"sources": [], "retrieval_time": time.time() - start_time, "error": str(e)}
    
    async def _vector_search(self, query: str) -> Dict[str, Any]:
        """Vector similarity search"""
        try:
            result = self.vector_retriever.search(query_text=query, top_k=5)
            
            sources = []
            # Handle different result formats from GraphRAG
            if hasattr(result, 'records') and result.records:
                for record in result.records:
                    sources.append({
                        "content": record.get("text", ""),
                        "chunk_index": record.get("chunk_index", 0),
                        "score": record.get("score", 0.0),
                        "source_type": "vector_search"
                    })
            elif hasattr(result, 'items') and result.items:
                for item in result.items:
                    sources.append({
                        "content": getattr(item, 'content', str(item)),
                        "score": getattr(item, 'score', 0.0),
                        "source_type": "vector_search"
                    })
            else:
                # Fallback: try to convert result to string
                sources.append({
                    "content": str(result),
                    "score": 1.0,
                    "source_type": "vector_search"
                })
            
            return {"sources": sources, "strategy": "vector"}
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            return {"sources": [], "error": str(e)}
    
    async def _hybrid_search(self, query: str) -> Dict[str, Any]:
        """Hybrid vector + fulltext search"""
        try:
            result = self.hybrid_retriever.search(query_text=query, top_k=5)
            
            sources = []
            # Handle different result formats from GraphRAG
            if hasattr(result, 'records') and result.records:
                for record in result.records:
                    sources.append({
                        "content": record.get("text", ""),
                        "chunk_index": record.get("chunk_index", 0),
                        "score": record.get("score", 0.0),
                        "source_type": "hybrid_search"
                    })
            elif hasattr(result, 'items') and result.items:
                for item in result.items:
                    sources.append({
                        "content": getattr(item, 'content', str(item)),
                        "score": getattr(item, 'score', 0.0),
                        "source_type": "hybrid_search"
                    })
            else:
                # Fallback: try to convert result to string
                sources.append({
                    "content": str(result),
                    "score": 1.0,
                    "source_type": "hybrid_search"
                })
            
            return {"sources": sources, "strategy": "hybrid"}
            
        except Exception as e:
            self.logger.error(f"Hybrid search failed: {e}")
            return {"sources": [], "error": str(e)}
    
    async def _text2cypher_search(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Natural language to Cypher search"""
        try:
            # Add conversation context to the query
            conversation_summary = context.get_conversation_summary()
            enhanced_query = f"Context: {conversation_summary}\n\nQuery: {query}" if conversation_summary != "No previous conversation context." else query
            
            result = self.text2cypher_retriever.search(query_text=enhanced_query)
            
            sources = []
            # Handle different result formats from GraphRAG
            if hasattr(result, 'records') and result.records:
                for record in result.records:
                    sources.append({
                        "content": str(record),
                        "source_type": "cypher_query",
                        "generated_cypher": getattr(self.text2cypher_retriever, 'last_generated_cypher', 'Not captured')
                    })
            elif hasattr(result, 'items') and result.items:
                for item in result.items:
                    sources.append({
                        "content": str(item),
                        "source_type": "cypher_query",
                        "generated_cypher": getattr(self.text2cypher_retriever, 'last_generated_cypher', 'Not captured')
                    })
            else:
                # Fallback: try to convert result to string
                sources.append({
                    "content": str(result),
                    "source_type": "cypher_query",
                    "generated_cypher": getattr(self.text2cypher_retriever, 'last_generated_cypher', 'Not captured')
                })
            
            return {
                "sources": sources, 
                "strategy": "text2cypher",
                "generated_cypher": getattr(self.text2cypher_retriever, 'last_generated_cypher', 'Not captured')
            }
            
        except Exception as e:
            self.logger.warning(f"Text2Cypher search failed, falling back to vector search: {e}")
            # Fallback to vector search when Text2Cypher fails
            return await self._vector_search(query)
    
    async def _vector_cypher_search(self, query: str) -> Dict[str, Any]:
        """Vector search with graph context"""
        try:
            result = self.vector_cypher_retriever.search(query_text=query, top_k=5)
            
            sources = []
            for record in result.records:
                sources.append({
                    "content": record.get("context", ""),
                    "chunk_index": record.get("chunk_index", 0),
                    "score": record.get("score", 0.0),
                    "knowledge_graph_context": record.get("knowledge_graph_context", []),
                    "source_type": "vector_cypher_search"
                })
            
            return {"sources": sources, "strategy": "vector_cypher"}
            
        except Exception as e:
            self.logger.error(f"Vector Cypher search failed: {e}")
            return {"sources": [], "error": str(e)}
    
    async def _entity_search(self, query: str) -> Dict[str, Any]:
        """Entity-focused search"""
        try:
            result = self.entity_retriever.search(query_text=query, top_k=5)
            
            sources = []
            for record in result.records:
                sources.append({
                    "content": f"Entity: {record.get('name', '')}\nDescription: {record.get('description', '')}\nType: {record.get('type', '')}",
                    "entity_name": record.get('name', ''),
                    "entity_type": record.get('type', ''),
                    "score": record.get("score", 0.0),
                    "source_type": "entity_search"
                })
            
            return {"sources": sources, "strategy": "entity_focused"}
            
        except Exception as e:
            self.logger.error(f"Entity search failed: {e}")
            return {"sources": [], "error": str(e)}
    
    async def _multi_strategy_search(self, query: str, context: ConversationContext) -> Dict[str, Any]:
        """Combine multiple retrieval strategies"""
        try:
            # Run multiple strategies in parallel
            vector_task = self._vector_search(query)
            text2cypher_task = self._text2cypher_search(query, context)
            hybrid_task = self._hybrid_search(query)
            relationship_task = self._relationship_search(query)  # NEW
            entity_task = self._entity_search(query)  # NEW

            vector_result, cypher_result, rel_result, entity_result = await asyncio.gather(
                vector_task, text2cypher_task, relationship_task, entity_task, return_exceptions=True
            )
            
            # Combine results
            all_sources = []
            
            if isinstance(vector_result, dict) and "sources" in vector_result:
                all_sources.extend(vector_result["sources"][:2])  # Top 2 from vector
            
            if isinstance(cypher_result, dict) and "sources" in cypher_result:
                all_sources.extend(cypher_result["sources"][:2])  # Top 2 from cypher
                
            if isinstance(rel_result, dict) and "sources" in rel_result:
                all_sources.extend(rel_result["sources"][:3])  # Top 3 from relationships
                
            if isinstance(entity_result, dict) and "sources" in entity_result:
                all_sources.extend(entity_result["sources"][:2])  # Top 2 from entities
            
            return {
                "sources": all_sources,
                "strategy": "multi_strategy",
                "individual_results": {
                    "vector": vector_result if isinstance(vector_result, dict) else {"error": str(vector_result)},
                    "cypher": cypher_result if isinstance(cypher_result, dict) else {"error": str(cypher_result)},
                    "relationships": rel_result if isinstance(rel_result, dict) else {"error": str(rel_result)},
                    "entities": entity_result if isinstance(entity_result, dict) else {"error": str(entity_result)}
                }
            }
            
        except Exception as e:
            self.logger.error(f"Multi-strategy search failed: {e}")
            return {"sources": [], "error": str(e)}
    
    async def _generate_response(self, user_query: str, retrieval_results: Dict[str, Any], 
                                context: ConversationContext, query_type: QueryType) -> str:
        """Generate conversational response using LLM"""
        try:
            # Prepare context from retrieval results
            retrieved_content = []
            
            for i, source in enumerate(retrieval_results.get("sources", [])[:5]):  # Top 5 sources
                content = source.get("content", "")
                source_type = source.get("source_type", "unknown")
                
                if content:
                    retrieved_content.append(f"Source {i+1} ({source_type}):\n{content}\n")
            
            # Get conversation history
            conversation_summary = context.get_conversation_summary()
            
            # Build comprehensive prompt
            system_prompt = f"""You are an intelligent AI assistant powered by a GraphRAG knowledge graph. You have access to a comprehensive knowledge base and can provide detailed, accurate answers.

**Your Capabilities:**
- Access to documents, entities, relationships, and structured knowledge
- Ability to understand context and maintain conversation flow
- Expertise in providing detailed explanations with proper citations

**Instructions:**
- Provide accurate, helpful responses based on the retrieved information
- Cite sources when possible (mention "based on the knowledge base" or similar)
- If the information is insufficient, acknowledge limitations honestly
- Maintain conversational tone while being informative
- Connect current query to previous conversation context when relevant

**Query Type:** {query_type.value}
**Retrieval Strategy Used:** {retrieval_results.get('strategy', 'unknown')}

**Conversation Context:**
{conversation_summary}

**Retrieved Information:**
{chr(10).join(retrieved_content) if retrieved_content else "No specific information retrieved from the knowledge base."}

**User Question:** {user_query}

**Your Response:**"""

            # Generate response using LLM
            response = await self.llm.ainvoke(system_prompt)
            
            # Extract text from LLM response
            if hasattr(response, 'content'):
                response_text = response.content
            elif hasattr(response, 'text'):
                response_text = response.text
            else:
                response_text = str(response)
            
            # Add metadata about sources if available
            source_count = len(retrieval_results.get("sources", []))
            if source_count > 0:
                strategy_used = retrieval_results.get("strategy", "knowledge base")
                response_text += f"\n\n*This response is based on {source_count} sources from the {strategy_used} search.*"
            
            return response_text
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return f"I apologize, but I encountered an issue generating a response. The error was: {str(e)}. Please try asking your question differently."
    
    async def stream_chat(self, user_message: str, session_id: str = None) -> AsyncGenerator[str, None]:
        """Stream chat response for real-time UX"""
        try:
            # Get response (could be enhanced to stream from LLM directly)
            response = await self.chat(user_message, session_id)
            
            # Simulate streaming by yielding chunks
            content = response.content
            chunk_size = 50
            
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                yield chunk
                await asyncio.sleep(0.1)  # Small delay for streaming effect
                
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session"""
        if session_id not in self.conversations:
            return []
        
        context = self.conversations[session_id]
        return [
            {
                "role": msg.role,
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat(),
                "sources_count": len(msg.sources),
                "retrieval_strategy": msg.retrieval_info.get("strategy", "unknown")
            }
            for msg in context.messages
        ]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get chatbot performance statistics"""
        avg_response_time = self.total_response_time / self.query_count if self.query_count > 0 else 0
        
        return {
            "total_queries": self.query_count,
            "active_conversations": len(self.conversations),
            "average_response_time": avg_response_time,
            "total_response_time": self.total_response_time
        }
    
    async def cleanup(self):
        """Clean up resources"""
        self.logger.info("🧹 Cleaning up chatbot resources...")
        
        if self.driver:
            self.driver.close()
        
        self.logger.info("✅ Chatbot cleanup completed")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()


# Interactive CLI Interface
class ChatbotCLI:
    """Command-line interface for the GraphRAG chatbot"""
    
    def __init__(self, chatbot: GraphRAGChatbot):
        self.chatbot = chatbot
        self.session_id = f"cli_session_{int(time.time())}"
        
    async def run_interactive_session(self):
        """Run an interactive chat session"""
        print("🤖 GraphRAG Chatbot Interactive Session")
        print("=" * 50)
        print("Type 'quit', 'exit', or 'bye' to end the session")
        print("Type 'history' to see conversation history")
        print("Type 'stats' to see performance statistics")
        print("Type 'help' for more commands")
        print("=" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\n🗣️  You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Thanks for chatting!")
                    break
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                elif user_input.lower() == 'stats':
                    self._show_stats()
                    continue
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'clear':
                    # Clear conversation history
                    if self.session_id in self.chatbot.conversations:
                        del self.chatbot.conversations[self.session_id]
                    print("🧹 Conversation history cleared!")
                    continue
                
                # Process chat message
                print("🤖 Assistant: ", end="", flush=True)
                
                # Stream response for better UX
                full_response = ""
                async for chunk in self.chatbot.stream_chat(user_input, self.session_id):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                
                print()  # New line after streaming
                
            except KeyboardInterrupt:
                print("\n\n👋 Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
    
    def _show_history(self):
        """Show conversation history"""
        history = self.chatbot.get_conversation_history(self.session_id)
        
        if not history:
            print("📝 No conversation history yet.")
            return
        
        print("\n📝 Conversation History:")
        print("-" * 30)
        
        for i, msg in enumerate(history[-10:], 1):  # Show last 10 messages
            role_emoji = "🗣️ " if msg["role"] == "user" else "🤖"
            timestamp = datetime.fromisoformat(msg["timestamp"]).strftime("%H:%M:%S")
            
            content_preview = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
            
            print(f"{i}. [{timestamp}] {role_emoji} {msg['role'].title()}: {content_preview}")
            
            if msg["role"] == "assistant" and msg.get("sources_count", 0) > 0:
                print(f"   └─ Sources: {msg['sources_count']}, Strategy: {msg.get('retrieval_strategy', 'unknown')}")
    
    def _show_stats(self):
        """Show performance statistics"""
        stats = self.chatbot.get_performance_stats()
        
        print("\n📊 Performance Statistics:")
        print("-" * 25)
        print(f"Total Queries: {stats['total_queries']}")
        print(f"Active Conversations: {stats['active_conversations']}")
        print(f"Average Response Time: {stats['average_response_time']:.2f}s")
        print(f"Total Response Time: {stats['total_response_time']:.2f}s")
    
    def _show_help(self):
        """Show help information"""
        print("\n❓ Available Commands:")
        print("-" * 20)
        print("• Type any question to chat with the AI")
        print("• 'history' - Show conversation history")
        print("• 'stats' - Show performance statistics")
        print("• 'clear' - Clear conversation history")
        print("• 'help' - Show this help message")
        print("• 'quit'/'exit'/'bye' - End the session")
        print("\n🎯 Query Types the AI can handle:")
        print("• Factual questions (What is...? Who is...?)")
        print("• Relationship queries (How are X and Y connected?)")
        print("• Semantic search (Find information about...)")
        print("• Analytical questions (Compare, analyze, evaluate...)")


# Example usage and main function
async def main():
    """Main function demonstrating chatbot usage"""
    
    # Configuration - use your existing config
    config = AdvancedPipelineConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        openai_api_key="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",
        llm_model="gpt-4o-mini",  # Using mini for faster responses
        embedding_model="text-embedding-3-large"
    )
    
    # Initialize and run chatbot
    async with GraphRAGChatbot(config) as chatbot:
        print("🚀 Starting GraphRAG Chatbot Demo")
        
        # Option 1: Interactive CLI session
        cli = ChatbotCLI(chatbot)
        await cli.run_interactive_session()
        
        # Option 2: Programmatic chat examples (commented out)
        """
        # Example conversations
        test_queries = [
            "What companies are mentioned in the knowledge base?",
            "Who works for TechNova?",
            "Tell me about the relationships between entities in Austin",
            "What is artificial intelligence according to the documents?",
            "How are different organizations connected?"
        ]
        
        session_id = "demo_session"
        
        for query in test_queries:
            print(f"\n🗣️  User: {query}")
            
            response = await chatbot.chat(query, session_id)
            print(f"🤖 Assistant: {response.content}")
            
            # Show retrieval info
            if response.retrieval_info:
                strategy = response.retrieval_info.get("strategy", "unknown")
                results_count = response.retrieval_info.get("results_count", 0)
                print(f"   └─ Strategy: {strategy}, Sources: {results_count}")
        
        # Show final stats
        stats = chatbot.get_performance_stats()
        print(f"\n📊 Final Stats: {stats}")
        """


if __name__ == "__main__":
    # Run the chatbot
    asyncio.run(main())
