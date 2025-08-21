import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, patch
import uuid

@pytest.mark.asyncio
async def test_create_graph_unauthorized(test_client: AsyncClient):
    """Test creating graph without authentication"""
    response = await test_client.post(
        "/api/v1/graphs",
        json={
            "name": "Test Graph",
            "description": "A test knowledge graph"
        }
    )
    
    assert response.status_code == 403

@pytest.mark.asyncio
@patch('app.services.auth_service.auth_service.verify_token')
async def test_create_graph_success(mock_verify_token, test_client: AsyncClient, mock_user):
    """Test successful graph creation"""
    mock_verify_token.return_value = mock_user
    
    response = await test_client.post(
        "/api/v1/graphs",
        json={
            "name": "Test Graph",
            "description": "A test knowledge graph"
        },
        headers={"Authorization": "Bearer test-token"}
    )
    
    assert response.status_code == 201
    data = response.json()
    
    assert data["name"] == "Test Graph"
    assert data["description"] == "A test knowledge graph"
    assert data["user_id"] == mock_user["id"]
    assert "id" in data
    assert "created_at" in data

@pytest.mark.asyncio
@patch('app.services.auth_service.auth_service.verify_token')
async def test_list_graphs(mock_verify_token, test_client: AsyncClient, mock_user):
    """Test listing user graphs"""
    mock_verify_token.return_value = mock_user
    
    # First create a graph
    create_response = await test_client.post(
        "/api/v1/graphs",
        json={
            "name": "Test Graph",
            "description": "A test knowledge graph"
        },
        headers={"Authorization": "Bearer test-token"}
    )
    assert create_response.status_code == 201
    
    # Then list graphs
    response = await test_client.get(
        "/api/v1/graphs",
        headers={"Authorization": "Bearer test-token"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == "Test Graph"
