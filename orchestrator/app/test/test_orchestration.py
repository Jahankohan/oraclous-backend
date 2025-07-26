import pytest
from httpx import AsyncClient
from orchestrator.app.orchestrator import app

@pytest.mark.asyncio
async def test_basic_orchestration():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.post("/run-task", json={
            "task_description": "Extract comments from a GitHub repo and generate Q&A pairs"
        })
        assert response.status_code == 200
        data = response.json()
        assert "result" in data
        assert isinstance(data["result"], dict) or isinstance(data["result"], list)

@pytest.mark.asyncio
async def test_list_mcp():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/mcp")
        assert response.status_code == 200
        mcp_list = response.json()
        assert isinstance(mcp_list, list)
        assert any("github" in m["name"] for m in mcp_list)
