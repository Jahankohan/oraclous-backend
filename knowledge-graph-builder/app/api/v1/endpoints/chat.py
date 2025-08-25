from fastapi import APIRouter, Depends, HTTPException, status, WebSocket, WebSocketDisconnect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update
from typing import List, Optional, Dict, Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from datetime import datetime

from app.api.dependencies import get_current_user_id, get_database
from app.models.graph import KnowledgeGraph
from app.models.chat import ChatSession, ChatMessage
from app.services.chat_service import chat_service  # COMPREHENSIVE SERVICE
from app.services.schema_service import schema_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# ==================== REQUEST/RESPONSE MODELS (ALL MODES) ====================

class ChatRequest(BaseModel):
    """Chat request supporting ALL retrieval modes"""
    message: str = Field(..., min_length=1, description="User's question or message")
    
    # RESTORED: All original retrieval modes
    mode: str = Field("graph_vector", description="vector, graph, graph_vector, graphrag, comprehensive")
    
    # LLM configuration with TEMPERATURE CONTROL for hallucination prevention
    llm_provider: str = Field("openai", description="LLM provider")
    llm_model: str = Field("gpt-4o-mini", description="LLM model")
    temperature: float = Field(0.2, ge=0.0, le=1.0, description="LLM temperature for hallucination control")
    
    # Session management
    session_id: Optional[str] = Field(None, description="Chat session ID for continuity")
    
    # Enhanced parameters
    max_context_tokens: int = Field(4000, description="Maximum context size")
    include_reasoning_chain: bool = Field(True, description="Include reasoning steps")
    include_history: bool = Field(True, description="Include conversation history")

class ChatResponse(BaseModel):
    """Comprehensive chat response model"""
    # Core response
    answer: str
    mode: str
    success: bool
    
    # RESTORED: Original response fields
    sources: Optional[Dict[str, Any]] = None
    cypher: Optional[str] = None
    raw_results: Optional[List[Dict]] = None
    
    # ENHANCED: New response fields
    grounded: Optional[bool] = False
    context_summary: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    
    # Insights and enhancements
    related_entities: Optional[List[Dict]] = []
    graph_insights: Optional[Dict[str, Any]] = {}
    suggested_followup: Optional[List[str]] = []
    entity_continuity: Optional[Dict[str, Any]] = {}
    
    # Transparency
    reasoning_chain: Optional[List[str]] = []
    conversation_insight: Optional[Dict[str, Any]] = {}
    performance: Optional[Dict[str, Any]] = {}
    
    # Session management
    session_id: str
    message_id: str
    timestamp: datetime

class CypherQueryRequest(BaseModel):
    """Request for custom Cypher query execution"""
    query: str = Field(..., min_length=1, description="Cypher query to execute")
    limit: int = Field(100, ge=1, le=1000, description="Result limit")

class ExplainQueryRequest(BaseModel):
    """Request for natural language query explanation"""
    question: str = Field(..., min_length=1, description="Question to explain")
    include_schema: bool = Field(True, description="Include schema information")

# ==================== MAIN CHAT ENDPOINT (ALL MODES RESTORED) ====================

@router.post("/graphs/{graph_id}/chat", response_model=ChatResponse)
async def chat_with_graph(
    graph_id: UUID,
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """
    Chat with a knowledge graph using ALL retrieval modes:
    
    - vector: Pure vector search on entities/chunks
    - graph: Cypher query generation  
    - graph_vector: Hybrid approach (RECOMMENDED)
    - graphrag: Advanced neighborhood analysis
    - comprehensive: New advanced graph reasoning with Neo4j GDS
    """
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == user_id
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        # Get or create chat session
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid4())
            await _create_chat_session(db, session_id, graph_id, user_id)
        
        # Initialize chat service for this graph with TEMPERATURE CONTROL
        chat_initialized = await chat_service.initialize_chat(
            graph_id=graph_id,
            user_id=user_id,
            provider=request.llm_provider,
            model=request.llm_model
        )
        
        if not chat_initialized:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Failed to initialize chat service"
            )
        
        # RESTORED: Process chat with ALL modes supported
        chat_result = await chat_service.chat_with_graph(
            query=request.message,
            mode=request.mode,  # All modes: vector, graph, graph_vector, graphrag, comprehensive
            graph_id=graph_id,
            conversation_id=session_id,
            include_history=request.include_history,
            max_context_tokens=request.max_context_tokens,
            include_reasoning_chain=request.include_reasoning_chain
        )
        
        # Save chat messages
        message_id = str(uuid4())
        await _save_chat_messages(
            db, session_id, request.message, chat_result, message_id
        )
        
        # Update session timestamp
        await _update_session_timestamp(db, session_id)
        
        return ChatResponse(
            # Core response
            answer=chat_result["answer"],
            mode=chat_result.get("mode", request.mode),
            success=chat_result.get("success", True),
            
            # RESTORED: Original fields
            sources=chat_result.get("sources"),
            cypher=chat_result.get("cypher"),
            raw_results=chat_result.get("raw_results"),
            
            # ENHANCED: New fields
            grounded=chat_result.get("grounded", False),
            context_summary=chat_result.get("context_summary"),
            confidence_scores=chat_result.get("confidence_scores"),
            
            # Insights
            related_entities=chat_result.get("related_entities", []),
            graph_insights=chat_result.get("graph_insights", {}),
            suggested_followup=chat_result.get("suggested_followup", []),
            entity_continuity=chat_result.get("entity_continuity", {}),
            
            # Transparency
            reasoning_chain=chat_result.get("reasoning_chain", []),
            conversation_insight=chat_result.get("conversation_insight", {}),
            performance=chat_result.get("performance", {}),
            
            # Session info
            session_id=session_id,
            message_id=message_id,
            timestamp=datetime.utcnow()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )

# ==================== CYPHER QUERY EXECUTION (RESTORED) ====================

@router.post("/graphs/{graph_id}/chat/cypher")
async def execute_cypher_query(
    graph_id: UUID,
    request: CypherQueryRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Execute a custom Cypher query on the graph (RESTORED)"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == user_id
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        # Initialize chat service for schema access
        await chat_service.initialize_chat(graph_id, user_id)
        
        # SECURITY: Validate query is read-only
        query = request.query.strip()
        if not query.upper().startswith(('MATCH', 'RETURN', 'WITH', 'CALL')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only read queries are allowed (MATCH, RETURN, WITH, CALL)"
            )
        
        # CRITICAL: Add graph_id filter if not present
        if f"graph_id = '{graph_id}'" not in query:
            if "WHERE" in query.upper():
                # Insert graph_id filter into existing WHERE clause
                where_pos = query.upper().find("WHERE")
                after_where = query[where_pos + 5:].strip()
                query = (
                    query[:where_pos + 5] + 
                    f" n.graph_id = '{graph_id}' AND " + 
                    after_where
                )
            else:
                # Add WHERE clause before RETURN
                return_pos = query.upper().find("RETURN")
                if return_pos >= 0:
                    query = (
                        query[:return_pos] + 
                        f"WHERE n.graph_id = '{graph_id}' " +
                        query[return_pos:]
                    )
        
        # Add limit if not present
        if "LIMIT" not in query.upper():
            query += f" LIMIT {request.limit}"
        
        # Execute query
        from app.core.neo4j_client import neo4j_client
        result = await neo4j_client.execute_query(query)
        
        return {
            "query": query,
            "results": result,
            "count": len(result),
            "success": True,
            "graph_id": str(graph_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cypher query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )

# ==================== QUERY EXPLANATION (RESTORED) ====================

@router.post("/graphs/{graph_id}/chat/explain")
async def explain_natural_language_query(
    graph_id: UUID,
    request: ExplainQueryRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Explain how a natural language question would be processed (RESTORED)"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == user_id
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        # Initialize chat service
        await chat_service.initialize_chat(graph_id, user_id)
        
        # Get schema using schema_service (RESTORED)
        schema = await schema_service.get_graph_schema(graph_id)
        
        # Extract entities from question (RESTORED)
        query_entities = await chat_service._extract_entities_from_query(request.question)
        
        # Generate Cypher query (RESTORED)
        cypher_result = await chat_service._natural_language_to_cypher(request.question, schema)
        
        # Get explanation of reasoning process
        reasoning_explanation = await chat_service.explain_reasoning(request.question)
        
        response = {
            "question": request.question,
            "extracted_entities": query_entities,
            "generated_cypher": cypher_result.get("cypher"),
            "cypher_valid": cypher_result.get("success", False),
            "reasoning_explanation": reasoning_explanation,
            "processing_modes": [
                {
                    "mode": "vector",
                    "description": "Searches for semantically similar entities and text chunks using embeddings",
                    "best_for": "Finding entities mentioned in documents or similar concepts"
                },
                {
                    "mode": "graph", 
                    "description": "Converts question to Cypher query and searches graph structure",
                    "best_for": "Precise relationship queries and structured data exploration"
                },
                {
                    "mode": "graph_vector",
                    "description": "Combines both vector search and graph queries for comprehensive results",
                    "best_for": "Most questions - provides both precision and coverage"
                },
                {
                    "mode": "graphrag",
                    "description": "Advanced retrieval using entity neighborhoods and multi-hop relationships",
                    "best_for": "Complex questions requiring deep graph analysis"
                },
                {
                    "mode": "comprehensive",
                    "description": "Advanced graph reasoning with community detection and centrality analysis",
                    "best_for": "Analytical questions requiring graph algorithms and insights"
                }
            ]
        }
        
        # Include schema if requested
        if request.include_schema:
            response["available_schema"] = schema
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain query: {str(e)}"
        )

# ==================== REASONING EXPLANATION (ENHANCED) ====================

@router.post("/graphs/{graph_id}/chat/explain-reasoning")
async def explain_reasoning_process(
    graph_id: UUID,
    query: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Explain the reasoning process for a specific query"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == user_id
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        # Initialize chat service
        await chat_service.initialize_chat(graph_id, user_id)
        
        # Get detailed reasoning explanation
        explanation = await chat_service.explain_reasoning(query)
        
        if "error" in explanation:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=explanation["error"]
            )
        
        return {
            "query": query,
            "graph_id": str(graph_id),
            **explanation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reasoning explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain reasoning: {str(e)}"
        )

# ==================== SESSION MANAGEMENT (ENHANCED) ====================

@router.post("/graphs/{graph_id}/chat/sessions")
async def create_chat_session(
    graph_id: UUID,
    name: Optional[str] = None,
    description: Optional[str] = None,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Create a new chat session for a graph"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == user_id
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        # Create session
        session_id = str(uuid4())
        session_name = name or f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = ChatSession(
            id=UUID(session_id),
            graph_id=graph_id,
            user_id=user_id,
            session_name=session_name,
            description=description,
            created_at=datetime.utcnow(),
            last_message_at=datetime.utcnow()
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        return {
            "session_id": str(session.id),
            "graph_id": str(session.graph_id),
            "name": session.session_name,
            "description": session.description,
            "created_at": session.created_at,
            "last_message_at": session.last_message_at,
            "message_count": 0
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat session"
        )

@router.get("/graphs/{graph_id}/chat/sessions")
async def list_chat_sessions(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """List all chat sessions for a graph"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == user_id
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        # Get sessions with message counts
        sessions_query = """
        SELECT s.id, s.graph_id, s.user_id, s.session_name, s.description,
               s.created_at, s.last_message_at, COUNT(m.id) as message_count
        FROM chat_sessions s
        LEFT JOIN chat_messages m ON s.id = m.session_id
        WHERE s.graph_id = :graph_id AND s.user_id = :user_id
        GROUP BY s.id, s.graph_id, s.user_id, s.session_name, s.description, s.created_at, s.last_message_at
        ORDER BY s.last_message_at DESC
        """
        
        result = await db.execute(
            sessions_query,
            {"graph_id": str(graph_id), "user_id": user_id}
        )
        
        sessions = []
        for row in result.fetchall():
            sessions.append({
                "session_id": str(row.id),
                "graph_id": str(row.graph_id),
                "name": row.session_name,
                "description": row.description,
                "created_at": row.created_at,
                "last_message_at": row.last_message_at,
                "message_count": row.message_count
            })
        
        return {
            "sessions": sessions,
            "total_sessions": len(sessions),
            "graph_id": str(graph_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list chat sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list chat sessions"
        )

@router.get("/graphs/{graph_id}/chat/sessions/{session_id}/messages")
async def get_chat_messages(
    graph_id: UUID,
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Get all messages in a chat session"""
    
    try:
        # Verify session ownership
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == UUID(session_id),
                ChatSession.graph_id == graph_id,
                ChatSession.user_id == user_id
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        # Get messages
        result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == UUID(session_id))
            .order_by(ChatMessage.created_at.asc())
        )
        messages = result.scalars().all()
        
        return {
            "session_id": session_id,
            "graph_id": str(graph_id),
            "messages": [
                {
                    "message_id": str(msg.id),
                    "session_id": str(msg.session_id),
                    "role": msg.message_type,
                    "content": msg.content,
                    "metadata": msg.chat_metadata,
                    "created_at": msg.created_at
                }
                for msg in messages
            ],
            "message_count": len(messages)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat messages"
        )

# ==================== CONVERSATION SUMMARY (ENHANCED) ====================

@router.get("/graphs/{graph_id}/chat/summary")
async def get_conversation_summary(
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Get comprehensive conversation summary and statistics"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == user_id
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        # Get conversation summary from chat service
        if chat_service.current_graph_id == graph_id:
            summary = chat_service.get_conversation_summary()
        else:
            summary = {
                "graph_id": str(graph_id),
                "conversation_length": 0,
                "message": "No active conversation for this graph"
            }
        
        # Get database statistics
        sessions_count = await db.execute(
            select(ChatSession).where(ChatSession.graph_id == graph_id)
        )
        total_sessions = len(sessions_count.scalars().all())
        
        messages_count = await db.execute(
            select(ChatMessage)
            .join(ChatSession)
            .where(ChatSession.graph_id == graph_id)
        )
        total_messages = len(messages_count.scalars().all())
        
        summary.update({
            "database_stats": {
                "total_sessions": total_sessions,
                "total_messages": total_messages
            }
        })
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation summary: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get conversation summary"
        )

# ==================== WEBSOCKET STREAMING (ENHANCED) ====================

@router.websocket("/graphs/{graph_id}/chat/stream")
async def chat_websocket(
    websocket: WebSocket,
    graph_id: UUID,
    user_id: str = Depends(get_current_user_id)
):
    """WebSocket endpoint for streaming chat responses with ALL modes"""
    
    await websocket.accept()
    
    try:
        # Initialize chat service for WebSocket session
        success = await chat_service.initialize_chat(graph_id, user_id)
        
        if not success:
            await websocket.send_json({
                "type": "error",
                "error": "Failed to initialize chat for graph"
            })
            await websocket.close()
            return
        
        # Send available modes
        await websocket.send_json({
            "type": "initialization",
            "available_modes": ["vector", "graph", "graph_vector", "graphrag", "comprehensive"],
            "graph_id": str(graph_id),
            "message": "Chat initialized successfully"
        })
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if "message" not in data:
                await websocket.send_json({
                    "type": "error",
                    "error": "Message field is required"
                })
                continue
            
            # Send processing status
            await websocket.send_json({
                "type": "processing",
                "mode": data.get("mode", "graph_vector"),
                "message": f"Processing query in {data.get('mode', 'graph_vector')} mode..."
            })
            
            # Process chat request
            try:
                response_data = await chat_service.chat_with_graph(
                    query=data["message"],
                    mode=data.get("mode", "focused"),  # Use focused for faster streaming
                    max_context_tokens=data.get("max_context_tokens", 3000),
                    include_reasoning_chain=data.get("include_reasoning_chain", False),
                    graph_id=graph_id
                )
                
                # Send response
                await websocket.send_json({
                    "type": "response",
                    "data": {
                        "answer": response_data["answer"],
                        "mode": response_data.get("mode"),
                        "success": response_data.get("success", True),
                        "grounded": response_data.get("grounded", False),
                        "sources": response_data.get("sources"),
                        "cypher": response_data.get("cypher"),
                        "performance": response_data.get("performance"),
                        "suggested_followup": response_data.get("suggested_followup", [])
                    }
                })
                
            except Exception as e:
                logger.error(f"WebSocket chat processing failed: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": f"Chat processing failed: {str(e)}"
                })
        
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({
                "type": "error",
                "error": "WebSocket connection error"
            })
        except:
            pass
        finally:
            await websocket.close()

# ==================== HELPER FUNCTIONS (ENHANCED) ====================

async def _create_chat_session(
    db: AsyncSession, 
    session_id: str, 
    graph_id: UUID, 
    user_id: str
):
    """Create a new chat session"""
    session = ChatSession(
        id=UUID(session_id),
        graph_id=graph_id,
        user_id=user_id,
        session_name=f"Chat {datetime.now().strftime('%H:%M')}",
        created_at=datetime.utcnow(),
        last_message_at=datetime.utcnow()
    )
    db.add(session)
    await db.commit()

async def _save_chat_messages(
    db: AsyncSession,
    session_id: str,
    user_message: str,
    chat_result: dict,
    message_id: str
):
    """Save user message and assistant response with enhanced metadata"""
    
    # Save user message
    user_msg = ChatMessage(
        id=UUID(message_id),
        session_id=UUID(session_id),
        message_type="user",
        content=user_message,
        created_at=datetime.utcnow()
    )
    db.add(user_msg)
    
    # Save assistant response with COMPREHENSIVE metadata
    assistant_metadata = {
        "mode": chat_result.get("mode"),
        "success": chat_result.get("success"),
        "grounded": chat_result.get("grounded", False),
        "confidence_scores": chat_result.get("confidence_scores"),
        "performance": chat_result.get("performance"),
        "sources": chat_result.get("sources"),
        "cypher": chat_result.get("cypher"),
        "context_summary": chat_result.get("context_summary"),
        "entity_continuity": chat_result.get("entity_continuity")
    }
    
    assistant_msg = ChatMessage(
        id=uuid4(),
        session_id=UUID(session_id),
        message_type="assistant",
        content=chat_result["answer"],
        chat_metadata=assistant_metadata,
        created_at=datetime.utcnow()
    )
    db.add(assistant_msg)
    
    await db.commit()

async def _update_session_timestamp(db: AsyncSession, session_id: str):
    """Update session last message timestamp"""
    await db.execute(
        update(ChatSession)
        .where(ChatSession.id == UUID(session_id))
        .values(last_message_at=datetime.utcnow())
    )
    await db.commit()