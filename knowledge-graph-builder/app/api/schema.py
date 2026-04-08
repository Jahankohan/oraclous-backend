"""
Schema Management API

API endpoints for managing Neo4j database schemas, providing schema extraction,
caching, and optimization capabilities for the knowledge graph system.
"""
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.services.schema_service import schema_manager
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/schema", tags=["schema"])


# ==================== REQUEST/RESPONSE MODELS ====================

class SchemaRefreshRequest(BaseModel):
    """Request model for schema refresh operations"""
    graph_id: str
    force_refresh: bool = False


class SchemaCacheInfo(BaseModel):
    """Information about cached schema"""
    graph_id: str
    schema_version: str
    last_updated: str
    node_count: int
    relationship_count: int
    age_minutes: float


class SchemaInfo(BaseModel):
    """Detailed schema information"""
    graph_id: str
    schema_version: str
    last_updated: str
    nodes: Dict[str, Any]  # Simplified node info
    relationships: Dict[str, Any]  # Simplified relationship info
    constraints: int
    indexes: int


class Text2CypherSchemaResponse(BaseModel):
    """Response model for Text2Cypher formatted schema"""
    graph_id: str
    schema_version: str
    formatted_schema: str
    last_updated: str


# ==================== API ENDPOINTS ====================

@router.get("/info/{graph_id}", response_model=SchemaInfo)
async def get_schema_info(
    graph_id: str,
    # auth_graph_id: str = Depends(get_graph_id_from_auth)
) -> SchemaInfo:
    """
    Get comprehensive schema information for a graph.
    
    Args:
        graph_id: Graph identifier
        
    Returns:
        Detailed schema information including nodes, relationships, and metadata
    """
    try:
        # TODO: Uncomment when auth is implemented
        # if graph_id != auth_graph_id:
        #     raise HTTPException(status_code=403, detail="Access denied to graph")
        
        schema = await schema_manager.extract_schema(graph_id)
        
        # Simplify node information for API response
        nodes_info: Dict[str, Dict[str, Any]] = {
            label: {
                "sample_count": node.sample_count,
                "property_count": len(node.properties),
                "properties": list(node.properties.keys())[:5]  # Limit to first 5 properties
            }
            for label, node in schema.nodes.items()
        }
        
        # Simplify relationship information
        relationships_info: Dict[str, Dict[str, Any]] = {
            rel_type: {
                "sample_count": rel.sample_count,
                "property_count": len(rel.properties),
                "start_labels": list(rel.start_labels)[:3],  # Limit to first 3 labels
                "end_labels": list(rel.end_labels)[:3]
            }
            for rel_type, rel in schema.relationships.items()
        }
        
        return SchemaInfo(
            graph_id=schema.graph_id,
            schema_version=schema.schema_version,
            last_updated=schema.last_updated.isoformat(),
            nodes=nodes_info,
            relationships=relationships_info,
            constraints=len(schema.constraints),
            indexes=len(schema.indexes)
        )
        
    except Exception as e:
        logger.error(f"Failed to get schema info for graph {graph_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to extract schema: {str(e)}")


@router.get("/text2cypher/{graph_id}", response_model=Text2CypherSchemaResponse)
async def get_text2cypher_schema(
    graph_id: str,
    force_refresh: bool = False,
    # auth_graph_id: str = Depends(get_graph_id_from_auth)
) -> Text2CypherSchemaResponse:
    """
    Get schema formatted for Text2CypherRetriever consumption.
    
    Args:
        graph_id: Graph identifier
        force_refresh: Force schema refresh even if cached
        
    Returns:
        Formatted schema string ready for Text2CypherRetriever
    """
    try:
        # TODO: Uncomment when auth is implemented
        # if graph_id != auth_graph_id:
        #     raise HTTPException(status_code=403, detail="Access denied to graph")
        
        # Extract schema (with optional force refresh)
        schema = await schema_manager.extract_schema(graph_id, force_refresh=force_refresh)
        
        # Format for Text2Cypher
        formatted_schema = schema_manager.format_schema_for_text2cypher(schema)
        
        return Text2CypherSchemaResponse(
            graph_id=schema.graph_id,
            schema_version=schema.schema_version,
            formatted_schema=formatted_schema,
            last_updated=schema.last_updated.isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get Text2Cypher schema for graph {graph_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to format schema: {str(e)}")


@router.post("/refresh", response_model=SchemaInfo)
async def refresh_schema(
    request: SchemaRefreshRequest,
    background_tasks: BackgroundTasks,
    # auth_graph_id: str = Depends(get_graph_id_from_auth)
) -> SchemaInfo:
    """
    Refresh schema cache for a specific graph.
    
    Args:
        request: Schema refresh request with graph_id and options
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        Updated schema information
    """
    try:
        # TODO: Uncomment when auth is implemented
        # if request.graph_id != auth_graph_id:
        #     raise HTTPException(status_code=403, detail="Access denied to graph")
        
        # Force refresh the schema
        schema = await schema_manager.extract_schema(
            request.graph_id, 
            force_refresh=request.force_refresh
        )
        
        logger.info(f"Schema refreshed for graph {request.graph_id}")
        
        # Return simplified schema info (same as get_schema_info)
        nodes_info: Dict[str, Dict[str, Any]] = {
            label: {
                "sample_count": node.sample_count,
                "property_count": len(node.properties),
                "properties": list(node.properties.keys())[:5]
            }
            for label, node in schema.nodes.items()
        }
        
        relationships_info: Dict[str, Dict[str, Any]] = {
            rel_type: {
                "sample_count": rel.sample_count,
                "property_count": len(rel.properties),
                "start_labels": list(rel.start_labels)[:3],
                "end_labels": list(rel.end_labels)[:3]
            }
            for rel_type, rel in schema.relationships.items()
        }
        
        return SchemaInfo(
            graph_id=schema.graph_id,
            schema_version=schema.schema_version,
            last_updated=schema.last_updated.isoformat(),
            nodes=nodes_info,
            relationships=relationships_info,
            constraints=len(schema.constraints),
            indexes=len(schema.indexes)
        )
        
    except Exception as e:
        logger.error(f"Failed to refresh schema for graph {request.graph_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh schema: {str(e)}")


@router.delete("/cache/{graph_id}")
async def clear_schema_cache(
    graph_id: str,
    # auth_graph_id: str = Depends(get_graph_id_from_auth)
) -> Dict[str, str]:
    """
    Clear schema cache for a specific graph.
    
    Args:
        graph_id: Graph identifier
        
    Returns:
        Confirmation message
    """
    try:
        # TODO: Uncomment when auth is implemented
        # if graph_id != auth_graph_id:
        #     raise HTTPException(status_code=403, detail="Access denied to graph")
        
        schema_manager.clear_cache(graph_id)
        
        logger.info(f"Schema cache cleared for graph {graph_id}")
        
        return {
            "message": f"Schema cache cleared for graph {graph_id}",
            "graph_id": graph_id
        }
        
    except Exception as e:
        logger.error(f"Failed to clear schema cache for graph {graph_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.delete("/cache")
async def clear_all_schema_cache() -> Dict[str, str]:
    """
    Clear all schema caches.
    
    Returns:
        Confirmation message
    """
    try:
        schema_manager.clear_cache()
        
        logger.info("All schema caches cleared")
        
        return {
            "message": "All schema caches cleared"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear all schema caches: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear caches: {str(e)}")


@router.get("/cache/status", response_model=Dict[str, SchemaCacheInfo])
async def get_cache_status() -> Dict[str, SchemaCacheInfo]:
    """
    Get status of all cached schemas.
    
    Returns:
        Information about all cached schemas
    """
    try:
        # Get detailed info for each cached schema
        cache_info: Dict[str, SchemaCacheInfo] = {}
        cached_schemas = schema_manager.get_cache_details()
        
        for graph_id, schema in cached_schemas.items():
            from datetime import datetime, timezone
            age_minutes = (datetime.now(timezone.utc) - schema.last_updated).total_seconds() / 60
            
            cache_info[graph_id] = SchemaCacheInfo(
                graph_id=schema.graph_id,
                schema_version=schema.schema_version,
                last_updated=schema.last_updated.isoformat(),
                node_count=len(schema.nodes),
                relationship_count=len(schema.relationships),
                age_minutes=age_minutes
            )
        
        return cache_info
        
    except Exception as e:
        logger.error(f"Failed to get cache status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache status: {str(e)}")


@router.get("/health")
async def schema_health_check() -> Dict[str, Any]:
    """
    Health check for schema management service.
    
    Returns:
        Service health information
    """
    try:
        # Test basic functionality
        cache_stats = schema_manager.get_cache_stats()
        
        return {
            "status": "healthy",
            "service": "schema_manager",
            **cache_stats
        }
        
    except Exception as e:
        logger.error(f"Schema health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "schema_manager",
            "error": str(e)
        }
