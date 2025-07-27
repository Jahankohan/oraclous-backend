from fastapi import APIRouter, HTTPException, Request
from uuid import UUID
from typing import List
from app.schemas.tool_schema import ToolCreate, ToolUpdate, ToolRead
from app.repositories.tool_repository import ToolRepository
from app.services.tool_service import ToolService

router = APIRouter()


def get_service(request: Request) -> ToolService:
    repository: ToolRepository = request.app.state.tool_repository
    return ToolService(repository)


@router.post("/", response_model=ToolRead)
async def create_tool(payload: ToolCreate, request: Request):
    service = get_service(request)
    return await service.create_tool(payload)


@router.get("/{tool_id}", response_model=ToolRead)
async def get_tool(tool_id: UUID, request: Request):
    service = get_service(request)
    tool = await service.get_tool(tool_id)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found")
    return tool


@router.put("/{tool_id}", response_model=ToolRead)
async def update_tool(tool_id: UUID, payload: ToolUpdate, request: Request):
    service = get_service(request)
    tool = await service.update_tool(tool_id, payload)
    if not tool:
        raise HTTPException(status_code=404, detail="Tool not found or update failed")
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
    return await service.list_tools()
