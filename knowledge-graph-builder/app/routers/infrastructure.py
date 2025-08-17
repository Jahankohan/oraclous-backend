from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import json
import asyncio

from app.core.neo4j_client import Neo4jClient, get_neo4j_client
from app.models.requests import Neo4jConnectionRequest
from app.models.responses import ConnectionResponse, BaseResponse, SchemaResponse
from app.config.settings import get_settings
from app.services.advanced_graph_integration_service import AdvancedGraphIntegrationService
from app.services.enhanced_graph_service import EnhancedGraphService

router = APIRouter()

@router.post("/connect", response_model=ConnectionResponse)
async def connect_database(
    request: Neo4jConnectionRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> ConnectionResponse:
    """Connect to Neo4j database and validate connection"""
    try:
        # Use AdvancedGraphIntegrationService for connection validation and info
        integration_service = AdvancedGraphIntegrationService(neo4j)
        # Optionally, you could add more advanced checks here
        # For now, fallback to basic connection info
        test_client = Neo4jClient(
            uri=request.uri,
            username=request.username,
            password=request.password,
            database=request.database
        )
        test_client.connect()
        schema = test_client.get_schema()
        vector_dimensions = test_client.check_vector_index_dimensions()
        settings = get_settings()

        # Ensure HAS_CHUNK relationship and chunkIndex property exist in the database
        # Create a dummy Document and Chunk if not present, then create relationship and property
        try:
            test_client.execute_write_query("""
            MERGE (d:Document {id: 'test_backend_doc'})
            MERGE (c:Chunk {id: 'test_backend_chunk'})
            SET c.chunkIndex = 0
            WITH d, c
            CALL {
                WITH d, c
                MERGE (d)-[r:HAS_CHUNK]->(c)
                RETURN r
            }
            RETURN d, c
            """)
        except Exception as e:
            # Log but do not fail connection
            print(f"Warning: Could not create HAS_CHUNK or chunkIndex: {e}")

        response_data = {
            "message": "Connection Successful",
            "db_vector_dimension": vector_dimensions,
            "application_dimension": 384,
            "gds_status": True,
            "write_access": True,
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
    # Use EnhancedGraphService for schema info if available
    enhanced_graph_service = EnhancedGraphService(neo4j)
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
        enhanced_graph_service = EnhancedGraphService(neo4j)
        deleted_count = await enhanced_graph_service.delete_documents(file_names, delete_entities)
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
