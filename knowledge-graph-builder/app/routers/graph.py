from fastapi import APIRouter, Depends, HTTPException
from typing import List, Optional

from app.core.neo4j_client import Neo4jClient, get_neo4j_client
from app.models.requests import GraphQueryRequest, SchemaRequest, DuplicateNodesRequest
from app.models.responses import GraphVisualization, BaseResponse, SchemaResponse, DuplicateNode
from app.services.graph_service import GraphService
from app.utils.llm_clients import LLMClientFactory

router = APIRouter()

@router.post("/graph_query", response_model=GraphVisualization)
async def get_graph_visualization(
    request: GraphQueryRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> GraphVisualization:
    """Get graph visualization data"""
    try:
        graph_service = GraphService(neo4j)
        return await graph_service.get_graph_visualization(
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
        graph_service = GraphService(neo4j)
        return await graph_service.get_node_neighbors(node_id, depth)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get neighbors: {str(e)}")

@router.post("/populate_graph_schema", response_model=SchemaResponse)
async def populate_graph_schema(
    request: SchemaRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> SchemaResponse:
    """Generate schema suggestions from text using LLM"""
    try:
        llm_factory = LLMClientFactory()
        llm = llm_factory.get_llm(request.model)
        
        # Create prompt for schema extraction
        schema_prompt = f"""
        Analyze the following text and suggest appropriate node labels and relationship types for a knowledge graph:

        Text: {request.text}

        Provide your response in the following format:
        Node Labels: [list of suggested entity types]
        Relationship Types: [list of suggested relationship types]
        
        Focus on the main entities and their relationships. Keep suggestions concise and relevant.
        """
        
        # Get LLM response
        import asyncio
        response = await asyncio.to_thread(llm.predict, schema_prompt)
        
        # Parse response (simple parsing - could be improved with structured output)
        lines = response.split('\n')
        node_labels = []
        relationship_types = []
        
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith('Node Labels:'):
                current_section = 'nodes'
                labels_text = line.replace('Node Labels:', '').strip()
                if labels_text:
                    node_labels = [label.strip() for label in labels_text.strip('[]').split(',')]
            elif line.startswith('Relationship Types:'):
                current_section = 'relationships'
                rels_text = line.replace('Relationship Types:', '').strip()
                if rels_text:
                    relationship_types = [rel.strip() for rel in rels_text.strip('[]').split(',')]
            elif current_section == 'nodes' and line:
                node_labels.append(line.strip('- '))
            elif current_section == 'relationships' and line:
                relationship_types.append(line.strip('- '))
        
        return SchemaResponse(
            node_labels=node_labels,
            relationship_types=relationship_types,
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
        graph_service = GraphService(neo4j)
        return await graph_service.get_duplicate_nodes()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get duplicates: {str(e)}")

@router.post("/merge_duplicate_nodes", response_model=BaseResponse)
async def merge_duplicate_nodes(
    request: DuplicateNodesRequest,
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Merge duplicate nodes into target node"""
    try:
        graph_service = GraphService(neo4j)
        await graph_service.merge_duplicate_nodes(request.node_ids, request.target_node_id)
        
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
        graph_service = GraphService(neo4j)
        return await graph_service.get_unconnected_nodes()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get unconnected nodes: {str(e)}")

@router.post("/delete_unconnected_nodes", response_model=BaseResponse)
async def delete_unconnected_nodes(
    node_ids: List[str],
    neo4j: Neo4jClient = Depends(get_neo4j_client)
) -> BaseResponse:
    """Delete unconnected entity nodes"""
    try:
        graph_service = GraphService(neo4j)
        deleted_count = await graph_service.delete_unconnected_nodes(node_ids)
        
        return BaseResponse(
            success=True,
            message=f"Successfully deleted {deleted_count} unconnected nodes"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete unconnected nodes: {str(e)}")
