from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete as sql_delete
from typing import List, Optional
from uuid import UUID, uuid4
from pydantic import BaseModel
from datetime import datetime

from app.api.dependencies import get_current_user_id, get_database
from app.models.graph import KnowledgeGraph
from app.models.chat import ChatSession, ChatMessage
from app.services.chat_service import chat_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

class ChatRequest(BaseModel):
    message: str
    mode: str = "graph_vector"  # vector, graph, graph_vector, graphrag
    session_id: Optional[str] = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

class ChatResponse(BaseModel):
    answer: str
    mode: str
    session_id: str
    message_id: str
    success: bool
    chat_metadata: Optional[dict] = None
    sources: Optional[dict] = None
    cypher: Optional[str] = None
    timestamp: datetime

class ChatSessionCreate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class ChatSessionResponse(BaseModel):
    id: str
    graph_id: str
    name: str
    created_at: datetime
    last_message_at: datetime
    message_count: int

class ChatMessageResponse(BaseModel):
    id: str
    session_id: str
    message_type: str
    content: str
    chat_metadata: Optional[dict] = None
    created_at: datetime

@router.post("/graphs/{graph_id}/chat", response_model=ChatResponse)
async def chat_with_graph(
    graph_id: UUID,
    request: ChatRequest,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Chat with a knowledge graph using various retrieval modes"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == UUID(user_id)
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
        
        # Initialize chat service for this graph
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
        
        # Process the chat request
        chat_result = await chat_service.chat_with_graph(
            query=request.message,
            mode=request.mode,
            graph_id=graph_id,
            conversation_id=session_id
        )
        
        # Save chat messages
        message_id = str(uuid4())
        await _save_chat_messages(
            db, session_id, request.message, chat_result, message_id
        )
        
        # Update session last message time
        await _update_session_timestamp(db, session_id)
        
        return ChatResponse(
            answer=chat_result["answer"],
            mode=chat_result["mode"],
            session_id=session_id,
            message_id=message_id,
            success=chat_result.get("success", True),
            chat_metadata=chat_result.get("chat_metadata"),
            sources=chat_result.get("sources"),
            cypher=chat_result.get("cypher"),
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

@router.post("/graphs/{graph_id}/chat/sessions", response_model=ChatSessionResponse)
async def create_chat_session(
    graph_id: UUID,
    request: ChatSessionCreate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Create a new chat session for a graph"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == UUID(user_id)
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
        session_name = request.name or f"Chat Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        session = ChatSession(
            id=UUID(session_id),
            graph_id=graph_id,
            user_id=UUID(user_id),
            session_name=session_name,
            description=request.description
        )
        
        db.add(session)
        await db.commit()
        await db.refresh(session)
        
        return ChatSessionResponse(
            id=str(session.id),
            graph_id=str(session.graph_id),
            name=session.session_name,
            created_at=session.created_at,
            last_message_at=session.last_message_at,
            message_count=0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create chat session"
        )

@router.get("/graphs/{graph_id}/chat/sessions", response_model=List[ChatSessionResponse])
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
                KnowledgeGraph.user_id == UUID(user_id)
            )
        )
        graph = result.scalar_one_or_none()
        
        if not graph:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Graph not found"
            )
        
        # Get sessions
        sessions_result = await db.execute(
            select(ChatSession).where(
                ChatSession.graph_id == graph_id,
                ChatSession.user_id == UUID(user_id)
            ).order_by(ChatSession.last_message_at.desc())
        )
        sessions = sessions_result.scalars().all()
        
        # Get message counts for each session
        session_responses = []
        for session in sessions:
            message_count_result = await db.execute(
                select(ChatMessage).where(ChatMessage.session_id == session.id)
            )
            message_count = len(message_count_result.scalars().all())
            
            session_responses.append(ChatSessionResponse(
                id=str(session.id),
                graph_id=str(session.graph_id),
                name=session.session_name,
                created_at=session.created_at,
                last_message_at=session.last_message_at,
                message_count=message_count
            ))
        
        return session_responses
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list chat sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list chat sessions"
        )

@router.get("/graphs/{graph_id}/chat/sessions/{session_id}/messages", response_model=List[ChatMessageResponse])
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
                ChatSession.user_id == UUID(user_id)
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
        
        return [
            ChatMessageResponse(
                id=str(msg.id),
                session_id=str(msg.session_id),
                message_type=msg.message_type,
                content=msg.content,
                chat_metadata=msg.chat_metadata,
                created_at=msg.created_at
            )
            for msg in messages
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat messages: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve chat messages"
        )

@router.delete("/graphs/{graph_id}/chat/sessions/{session_id}")
async def delete_chat_session(
    graph_id: UUID,
    session_id: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Delete a chat session and all its messages"""
    
    try:
        # Verify session ownership
        result = await db.execute(
            select(ChatSession).where(
                ChatSession.id == UUID(session_id),
                ChatSession.graph_id == graph_id,
                ChatSession.user_id == UUID(user_id)
            )
        )
        session = result.scalar_one_or_none()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Chat session not found"
            )
        
        # Delete messages first (foreign key constraint)
        await db.execute(
            sql_delete(ChatMessage).where(
                ChatMessage.session_id == UUID(session_id)
            )
        )
        
        # Delete session
        await db.execute(
            sql_delete(ChatSession).where(
                ChatSession.id == UUID(session_id)
            )
        )
        
        await db.commit()
        
        return {"message": "Chat session deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete chat session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete chat session"
        )

@router.post("/graphs/{graph_id}/chat/cypher")
async def execute_cypher_query(
    graph_id: UUID,
    query: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Execute a custom Cypher query on the graph"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == UUID(user_id)
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
        
        # Validate and execute query
        if not query.strip().upper().startswith(('MATCH', 'RETURN', 'WITH', 'CALL')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only read queries are allowed (MATCH, RETURN, WITH, CALL)"
            )
        
        # Add graph_id filter if not present
        if "graph_id" not in query:
            # Simple injection of graph_id filter - could be made more sophisticated
            if "WHERE" in query.upper():
                query = query.replace("WHERE", f"WHERE n.graph_id = '{graph_id}' AND", 1)
            else:
                query += f" WHERE n.graph_id = '{graph_id}'"
        
        # Execute query with limit
        if "LIMIT" not in query.upper():
            query += " LIMIT 100"
        
        from app.core.neo4j_client import neo4j_client
        result = await neo4j_client.execute_query(query)
        
        return {
            "query": query,
            "results": result,
            "count": len(result),
            "success": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cypher query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )

@router.post("/graphs/{graph_id}/chat/explain")
async def explain_natural_language_query(
    graph_id: UUID,
    question: str,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """Explain how a natural language question would be processed"""
    
    try:
        # Verify graph ownership
        result = await db.execute(
            select(KnowledgeGraph).where(
                KnowledgeGraph.id == graph_id,
                KnowledgeGraph.user_id == UUID(user_id)
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
        
        # Get graph schema
        schema = await chat_service._get_graph_schema(graph_id)
        
        # Extract entities from question
        query_entities = await chat_service._extract_entities_from_query(question)
        
        # Generate Cypher query
        cypher_result = await chat_service._natural_language_to_cypher(question, schema)
        
        return {
            "question": question,
            "extracted_entities": query_entities,
            "available_schema": schema,
            "generated_cypher": cypher_result.get("cypher"),
            "cypher_valid": cypher_result.get("success", False),
            "processing_modes": [
                {
                    "mode": "vector",
                    "description": "Searches for semantically similar entities and text chunks using embeddings"
                },
                {
                    "mode": "graph", 
                    "description": "Converts question to Cypher query and searches graph structure"
                },
                {
                    "mode": "graph_vector",
                    "description": "Combines both vector search and graph queries for comprehensive results"
                },
                {
                    "mode": "graphrag",
                    "description": "Advanced retrieval using entity neighborhoods and multi-hop relationships"
                }
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query explanation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to explain query: {str(e)}"
        )

# Helper functions
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
        user_id=UUID(user_id),
        session_name=f"Chat {datetime.now().strftime('%H:%M')}"
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
    """Save user message and assistant response"""
    
    # Save user message
    user_msg = ChatMessage(
        id=UUID(message_id),
        session_id=UUID(session_id),
        message_type="user",
        content=user_message
    )
    db.add(user_msg)
    
    # Save assistant response
    assistant_msg = ChatMessage(
        id=uuid4(),
        session_id=UUID(session_id),
        message_type="assistant",
        content=chat_result["answer"],
        chat_metadata={  # Using chat_metadata instead of metadata
            "mode": chat_result.get("mode"),
            "success": chat_result.get("success"),
            "sources": chat_result.get("sources"),
            "cypher": chat_result.get("cypher")
        }
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