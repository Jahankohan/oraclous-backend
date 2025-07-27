from app.repositories.tool_repository import ToolRepository
from app.models.tools_model import Tool
from app.schemas.tool_schema import ToolCreate, ToolUpdate
from uuid import UUID
from typing import List, Optional


class ToolService:
    def __init__(self, repository: ToolRepository):
        self.repository = repository

    async def create_tool(self, data: ToolCreate) -> Tool:
        return await self.repository.create_tool(data)

    async def get_tool_by_id(self, tool_id: UUID) -> Optional[Tool]:
        return await self.repository.get_tool(tool_id)

    async def update_tool(self, tool_id: UUID, data: ToolUpdate) -> Optional[Tool]:
        return await self.repository.update_tool(tool_id, data)

    async def delete_tool(self, tool_id: UUID) -> bool:
        return await self.repository.delete_tool(tool_id)

    async def list_all_tools(self) -> List[Tool]:
        return await self.repository.list_all_tools()
