from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Dict, Any
from uuid import UUID, uuid4
from app.api.dependencies import get_current_user_id
from app.services.diffbot_graph_service import diffbot_graph_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

class DiffbotGraphTestRequest(BaseModel):
    text: str
    entities: list = ["Person", "Organization", "Location"]
    relationships: list = ["WORKS_FOR", "LOCATED_IN", "RELATED_TO"]

class DiffbotGraphTestResponse(BaseModel):
    nodes: list
    relationships: list
    source: str
    total_nodes: int
    total_relationships: int

@router.post("/diffbot/test-graph", response_model=DiffbotGraphTestResponse)
async def test_diffbot_graph_extraction(
    request: DiffbotGraphTestRequest,
    user_id: str = Depends(get_current_user_id)
):
    """Test Diffbot Graph Transformer with sample text"""
    
    try:
        # Set schema
        schema = {
            "entities": request.entities,
            "relationships": request.relationships
        }
        
        # Extract graph documents
        graph_docs = await diffbot_graph_service.extract_graph_documents(
            text=request.text,
            user_id=user_id,
            graph_id=uuid4(),  # Dummy UUID for testing
            schema=schema
        )
        
        # Process results
        all_nodes = []
        all_relationships = []
        
        for doc in graph_docs:
            for node in doc.nodes:
                all_nodes.append({
                    "id": node.id,
                    "type": node.type,
                    "properties": node.properties
                })
            
            for rel in doc.relationships:
                all_relationships.append({
                    "type": rel.type,
                    "source": rel.source.id,
                    "target": rel.target.id,
                    "properties": rel.properties
                })
        
        return DiffbotGraphTestResponse(
            nodes=all_nodes,
            relationships=all_relationships,
            source="diffbot_graph_transformer",
            total_nodes=len(all_nodes),
            total_relationships=len(all_relationships)
        )
        
    except Exception as e:
        logger.error(f"Diffbot graph test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diffbot graph extraction failed: {str(e)}"
        )

@router.get("/diffbot/status")
async def diffbot_status(user_id: str = Depends(get_current_user_id)):
    """Check Diffbot Graph Transformer availability"""
    
    try:
        # Try to initialize Diffbot
        success = await diffbot_graph_service.initialize_diffbot(user_id)
        
        return {
            "available": success,
            "status": "ready" if success else "unavailable",
            "message": "Diffbot Graph Transformer ready" if success else "No Diffbot API key or initialization failed",
            "type": "graph_transformer"
        }
        
    except Exception as e:
        logger.error(f"Diffbot status check failed: {e}")
        return {
            "available": False,
            "status": "error",
            "message": f"Status check failed: {str(e)}",
            "type": "graph_transformer"
        }
