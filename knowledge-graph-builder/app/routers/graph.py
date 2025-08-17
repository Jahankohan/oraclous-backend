from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional

from app.core.neo4j_client import Neo4jClient, get_neo4j_client
from app.models.requests import GraphQueryRequest, SchemaRequest, DuplicateNodesRequest
from app.models.responses import GraphVisualization, BaseResponse, SchemaResponse, DuplicateNode
from app.services.advanced_graph_analytic import AdvancedGraphAnalytics
from app.services.enhanced_graph_service import EnhancedGraphService
from app.utils.llm_clients import LLMClientFactory

router = APIRouter()

@router.post("/graph_query", response_model=GraphVisualization)
async def get_graph_visualization(
    request: GraphQueryRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> GraphVisualization:
    """Get graph visualization data"""
    try:
        enhanced_graph_service = EnhancedGraphService(neo4j)
        return await enhanced_graph_service.get_intelligent_graph_visualization(
            file_names=request.file_names,
            limit=request.limit
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph query failed: {str(e)}")

@router.get("/get_neighbours/{node_id}", response_model=GraphVisualization)
async def get_node_neighbors(
    node_id: str,
    depth: int = 1,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> GraphVisualization:
    """Get neighbors of a specific node"""
    try:
        enhanced_graph_service = EnhancedGraphService(neo4j)
        # Assuming get_node_neighbors is available in EnhancedGraphService
        return await enhanced_graph_service.get_node_neighbors(node_id, depth)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get neighbors: {str(e)}")

@router.post("/populate_graph_schema", response_model=SchemaResponse)
async def populate_graph_schema(
    request: SchemaRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> SchemaResponse:
    """Generate schema suggestions from text using LLM"""
    try:
        advanced_analytics = AdvancedGraphAnalytics(neo4j)
        schema_analysis = await advanced_analytics.learn_graph_schema()
        # Adapt to SchemaResponse model
        return SchemaResponse(
            node_labels=[p.source_label for p in schema_analysis["discovered_patterns"]],
            relationship_types=[p.relationship_type for p in schema_analysis["discovered_patterns"]],
            properties={"nodes": [], "relationships": []}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Schema generation failed: {str(e)}")

@router.get("/get_duplicate_nodes_list", response_model=List[DuplicateNode])
async def get_duplicate_nodes(
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> List[DuplicateNode]:
    """Get list of potential duplicate nodes"""
    try:
        advanced_analytics = AdvancedGraphAnalytics(neo4j)
        # Assuming get_duplicate_nodes is implemented in AdvancedGraphAnalytics
        return await advanced_analytics.get_duplicate_nodes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get duplicates: {str(e)}")

@router.post("/merge_duplicate_nodes", response_model=BaseResponse)
async def merge_duplicate_nodes(
    request: DuplicateNodesRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Merge duplicate nodes into target node"""
    try:
        advanced_analytics = AdvancedGraphAnalytics(neo4j)
        # Assuming merge_duplicate_nodes is implemented in AdvancedGraphAnalytics
        await advanced_analytics.merge_duplicate_nodes(request.node_ids, request.target_node_id)
        return BaseResponse(
            success=True,
            message=f"Successfully merged {len(request.node_ids)} duplicate nodes"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to merge duplicates: {str(e)}")

@router.get("/get_unconnected_nodes_list")
async def get_unconnected_nodes(
    neo4j: Neo4jClient = Depends(get_neo4j_client)
):
    """Get list of unconnected entity nodes"""
    try:
        advanced_analytics = AdvancedGraphAnalytics(neo4j)
        # Assuming get_unconnected_nodes is implemented in AdvancedGraphAnalytics
        return await advanced_analytics.get_unconnected_nodes()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get unconnected nodes: {str(e)}")

@router.post("/delete_unconnected_nodes", response_model=BaseResponse)
async def delete_unconnected_nodes(
    node_ids: List[str],
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Delete unconnected entity nodes"""
    try:
        advanced_analytics = AdvancedGraphAnalytics(neo4j)
        # Assuming delete_unconnected_nodes is implemented in AdvancedGraphAnalytics
        deleted_count = await advanced_analytics.delete_unconnected_nodes(node_ids)
        return BaseResponse(
            success=True,
            message=f"Successfully deleted {deleted_count} unconnected nodes"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete unconnected nodes: {str(e)}")
