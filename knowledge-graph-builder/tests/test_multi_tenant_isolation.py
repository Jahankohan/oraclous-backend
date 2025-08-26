"""
Test multi-tenant isolation to ensure fix works
"""

import pytest
import asyncio
from uuid import uuid4
from app.services.enhanced_graph_service import enhanced_graph_service
from app.core.neo4j_client import neo4j_client

@pytest.mark.asyncio
async def test_same_entity_id_different_graphs():
    """Test that same entity IDs can exist in different graphs"""
    
    # Create two different graphs
    graph_id_1 = uuid4()
    graph_id_2 = uuid4()
    
    # Create entities with same ID in both graphs - should succeed
    entity_data = {
        "id": "test_person_123",
        "name": "John Smith",
        "type": "Person"
    }
    
    # First entity creation
    result_1 = await create_test_entity(entity_data, graph_id_1)
    assert result_1 is not None
    
    # Second entity with same ID but different graph - should also succeed
    result_2 = await create_test_entity(entity_data, graph_id_2)
    assert result_2 is not None
    
    # Verify both entities exist
    entities_graph_1 = await get_entities_for_graph(graph_id_1)
    entities_graph_2 = await get_entities_for_graph(graph_id_2)
    
    assert len(entities_graph_1) == 1
    assert len(entities_graph_2) == 1
    assert entities_graph_1[0]["id"] == entities_graph_2[0]["id"]  # Same ID
    assert entities_graph_1[0]["graph_id"] != entities_graph_2[0]["graph_id"]  # Different graphs

@pytest.mark.asyncio  
async def test_duplicate_entity_same_graph_fails():
    """Test that duplicate entity IDs in same graph still fail"""
    
    graph_id = uuid4()
    
    entity_data = {
        "id": "duplicate_test_123", 
        "name": "Test Entity",
        "type": "Person"
    }
    
    # First creation should succeed
    result_1 = await create_test_entity(entity_data, graph_id)
    assert result_1 is not None
    
    # Second creation with same ID and same graph should fail
    with pytest.raises(Exception):  # Should raise constraint violation
        await create_test_entity(entity_data, graph_id)

async def create_test_entity(entity_data, graph_id):
    """Helper to create test entity"""
    
    query = """
    CREATE (n:Person {
        id: $id,
        name: $name, 
        graph_id: $graph_id
    })
    RETURN n
    """
    
    result = await neo4j_client.execute_write_query(query, {
        "id": entity_data["id"],
        "name": entity_data["name"],
        "graph_id": str(graph_id)
    })
    
    return result[0] if result else None

async def get_entities_for_graph(graph_id):
    """Helper to get entities for specific graph"""
    
    query = """
    MATCH (n {graph_id: $graph_id})
    RETURN n.id as id, n.name as name, n.graph_id as graph_id
    """
    
    return await neo4j_client.execute_query(query, {"graph_id": str(graph_id)})

# Cleanup after tests
@pytest.fixture(autouse=True)
async def cleanup_test_data():
    """Clean up test data after each test"""
    yield
    
    # Remove test entities
    cleanup_query = """
    MATCH (n)
    WHERE n.id STARTS WITH 'test_' OR n.id STARTS WITH 'duplicate_test_'
    DETACH DELETE n
    """
    await neo4j_client.execute_write_query(cleanup_query)
