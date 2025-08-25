import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_endpoint(test_client: AsyncClient):
    """Test health check endpoint"""
    response = await test_client.get("/api/v1/health")
    
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "service" in data
    assert "version" in data
    assert "timestamp" in data
    assert "dependencies" in data
    
    assert data["service"] == "knowledge-graph-builder"

@pytest.mark.asyncio
async def test_root_endpoint(test_client: AsyncClient):
    """Test root endpoint"""
    response = await test_client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    assert data["service"] == "knowledge-graph-builder"
    assert "version" in data
    assert "status" in data
    assert data["status"] == "running"
