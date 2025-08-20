from sqlalchemy import select
from typing import List
from app.models.tool_definition import ToolDefinitionDB
from sqlalchemy.ext.asyncio import AsyncSession

class ToolDefinitionRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_definitions(self) -> List[ToolDefinitionDB]:
        result = await self.db.execute(select(ToolDefinitionDB))
        return result.scalars().all()
