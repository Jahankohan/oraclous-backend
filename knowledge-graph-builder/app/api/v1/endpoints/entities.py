from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional, Dict, Any
from uuid import UUID

from app.api.dependencies import get_current_user_id, get_database
from app.core.neo4j_client import neo4j_client
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/graphs/{graph_id}/entities")
async def list_entities(
    graph_id: UUID,
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
    entity_type: Optional[str] = Query(default=None),
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_database)
):
    """List entities in a knowledge graph"""
    
    # TODO: Verify graph ownership
    
    try:
        # Build Cypher query
        where_clause = "WHERE n.graph_id = $graph_id"
        params = {"graph_id": str(graph_id), "limit": limit, "offset": offset}
        
        if entity_type:
            where_clause += f" AND '{entity_type}' IN labels(n)"
        
        query = f"""
        MATCH (n)
        {where_clause}
        RETURN n, labels(n) as labels
        ORDER BY n.name
        SKIP $offset
        LIMIT $limit
        """
        
        result = await neo4j_client.execute_query(query, params)
        
        entities = []
        for record in result:
            node = record["n"]
            entities.append({
                "id": node.get("id"),
                "labels": record["labels"],
                "properties": dict(node),
                "name": node.get("name", "Unknown")
            })
        
        # Get total count
        count_query = f"""
        MATCH (n)
        {where_clause.replace('SKIP $offset LIMIT $limit', '')}
        RETURN count(n) as total
        """
        count_result = await neo4j_client.execute_query(
            count_query, 
            {k: v for k, v in params.items() if k not in ["limit", "offset"]}
        )
        total = count_result[0]["total"] if count_result else 0
        
        return {
            "entities": entities,
            "total": total,
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error listing entities for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve entities"
        )

@router.get("/graphs/{graph_id}/entities/{entity_id}")
async def get_entity_details(
    graph_id: UUID,
    entity_id: str,
    user_id: str = Depends(get_current_user_id)
):
    """Get detailed information about a specific entity"""
    
    try:
        # Get entity with its relationships
        query = """
        MATCH (n {id: $entity_id, graph_id: $graph_id})
        OPTIONAL MATCH (n)-[r]-(related)
        WHERE related.graph_id = $graph_id
        RETURN n, 
               collect(DISTINCT {
                   relationship: type(r),
                   direction: CASE 
                       WHEN startNode(r) = n THEN 'outgoing'
                       ELSE 'incoming'
                   END,
                   related_entity: {
                       id: related.id,
                       name: related.name,
                       labels: labels(related)
                   },
                   properties: properties(r)
               }) as relationships
        """
        
        result = await neo4j_client.execute_query(query, {
            "entity_id": entity_id,
            "graph_id": str(graph_id)
        })
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Entity not found"
            )
        
        entity_data = result[0]
        entity = entity_data["n"]
        
        return {
            "id": entity.get("id"),
            "labels": entity.labels,
            "properties": dict(entity),
            "relationships": [rel for rel in entity_data["relationships"] if rel["related_entity"]["id"]]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting entity {entity_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve entity details"
        )

# @router.get("/graphs/{graph_id}/schema")
# async def get_graph_schema(
#     graph_id: UUID,
#     user_id: str = Depends(get_current_user_id)
# ):
#     """Get the schema of a knowledge graph"""
    
#     try:
#         # Get node labels (entity types)
#         labels_query = """
#         MATCH (n)
#         WHERE n.graph_id = $graph_id
#         RETURN DISTINCT labels(n) as labels
#         """
        
#         # Get relationship types
#         rels_query = """
#         MATCH ()-[r]->()
#         WHERE r.graph_id = $graph_id
#         RETURN DISTINCT type(r) as relationship_type
#         """
        
#         params = {"graph_id": str(graph_id)}
        
#         labels_result = await neo4j_client.execute_query(labels_query, params)
#         rels_result = await neo4j_client.execute_query(rels_query, params)
        
#         # Extract unique entity types
#         entity_types = set()
#         for record in labels_result:
#             for label in record["labels"]:
#                 entity_types.add(label)
        
#         # Extract relationship types
#         relationship_types = [record["relationship_type"] for record in rels_result]
        
#         return {
#             "entity_types": sorted(list(entity_types)),
#             "relationship_types": sorted(relationship_types),
#             "node_count": len(entity_types),
#             "relationship_count": len(relationship_types)
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting schema for graph {graph_id}: {e}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to retrieve graph schema"
#         )
