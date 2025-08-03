from fastapi import APIRouter, HTTPException, Request
from uuid import UUID
from typing import List
from app.schemas.tool_schema import ToolUpdate, ToolRead
from app.schemas.mcp_schema import MCPCreate
from app.repositories.tool_repository import ToolRepository
from app.repositories.mcp_repository import MCPRepository
from app.services.tool_service import ToolService

router = APIRouter()


def get_service(request: Request) -> ToolService:
    tool_repository: ToolRepository = request.app.state.tool_repository
    mcp_repository: MCPRepository = request.app.state.mcp_repository
    return ToolService(tool_repository, mcp_repository)


@router.post("/", response_model=ToolRead | None)
async def create_tool(payload: MCPCreate, request: Request):
    service = get_service(request)
    return await service.register_from_mcp_url(payload)


@router.get("/{tool_id}", response_model=ToolRead)
async def get_tool(tool_id: UUID, request: Request):
    service = get_service(request)
    tool = await service.get_tool_by_id(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


@router.delete("/{tool_id}")
async def delete_tool(tool_id: UUID, request: Request):
    service = get_service(request)
    success = await service.delete_tool(tool_id)
    if not success:
        raise HTTPException(status_code=404, detail="Tool not found or delete failed")
    return {"status": "deleted"}


@router.get("/", response_model=List[ToolRead])
async def list_tools(request: Request):
    service = get_service(request)
    return await service.list_all_tools()
