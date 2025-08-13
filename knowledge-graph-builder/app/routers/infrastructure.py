from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import json
import asyncio

from app.core.neo4j_client import Neo4jClient, get_neo4j_client
from app.models.requests import Neo4jConnectionRequest
from app.models.responses import ConnectionResponse, BaseResponse, SchemaResponse
from app.config.settings import get_settings
from app.services.graph_service import GraphService

router = APIRouter()

@router.post("/connect", response_model=ConnectionResponse)
async def connect_database(
    request: Neo4jConnectionRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> ConnectionResponse:
    """Connect to Neo4j database and validate connection"""
    try:
        # Test connection with provided credentials
        test_client = Neo4jClient(
            uri=request.uri,
            username=request.username,
            password=request.password,
            database=request.database
        )
        test_client.connect()
        
        # Get database information
        schema = test_client.get_schema()
        vector_dimensions = test_client.check_vector_index_dimensions()
        
        settings = get_settings()
        
        response_data = {
            "message": "Connection Successful",
            "db_vector_dimension": vector_dimensions,
            "application_dimension": 384,  # Default for sentence transformers
            "gds_status": True,  # Placeholder - would need actual GDS check
            "write_access": True,  # Placeholder - would need actual permission check
            "schema": schema
        }
        
        test_client.close()
        
        return ConnectionResponse(
            status="Success",
            data=response_data
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Connection failed: {str(e)}")

@router.get("/schema", response_model=SchemaResponse)
async def get_database_schema(
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> SchemaResponse:
    """Get current database schema"""
    schema = neo4j.get_schema()
    
    return SchemaResponse(
        node_labels=[item['label'] for item in schema.get('node_labels', [])],
        relationship_types=[item['relationshipType'] for item in schema.get('relationship_types', [])],
        properties={
            'nodes': [item['propertyName'] for item in schema.get('property_keys', [])],
            'relationships': []
        }
    )

@router.post("/drop_and_create_vector_index", response_model=BaseResponse)
async def recreate_vector_index(
    index_name: str = "vector",
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Drop and recreate vector index"""
    try:
        # Drop existing index
        neo4j.drop_vector_index(index_name)
        
        # Create new index with correct dimensions
        settings = get_settings()
        dimensions = 384  # Default for sentence transformers
        
        if settings.embedding_model == "text-embedding-ada-002":
            dimensions = 1536
        elif settings.embedding_model == "textembedding-gecko@003":
            dimensions = 768
            
        neo4j.create_vector_index(
            index_name=index_name,
            dimensions=dimensions
        )
        
        return BaseResponse(
            success=True,
            message=f"Vector index '{index_name}' recreated successfully",
            data={"dimensions": dimensions}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to recreate vector index: {str(e)}")

@router.delete("/delete_document_and_entities")
async def delete_documents(
    file_names: list[str],
    delete_entities: bool = False,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Delete documents and optionally their extracted entities"""
    try:
        graph_service = GraphService(neo4j)
        deleted_count = await graph_service.delete_documents(file_names, delete_entities)
        
        return BaseResponse(
            success=True,
            message=f"Successfully deleted {deleted_count} documents",
            data={"deleted_count": deleted_count, "delete_entities": delete_entities}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete documents: {str(e)}")

@router.get("/backend_connection_configuration")
async def get_backend_configuration() -> Dict[str, Any]:
    """Get backend connection configuration"""
    settings = get_settings()
    
    # Check if environment has pre-configured credentials
    has_env_config = all([
        settings.neo4j_uri != "neo4j://localhost:7687",
        settings.neo4j_username,
        settings.neo4j_password
    ])
    
    return {
        "show_login_dialog": not has_env_config,
        "supported_sources": ["local", "s3", "gcs", "youtube", "wiki", "web"],
        "supported_models": ["openai_gpt_4o", "openai_gpt_4o_mini", "gemini_1.5_flash"],
        "chat_modes": ["vector", "graph_vector", "graph", "fulltext"]
    }
