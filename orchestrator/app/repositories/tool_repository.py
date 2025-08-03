from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from app.models.tools_model import Tool
from app.models.base import Base
from app.schemas.tool_schema import ToolCreate, ToolUpdate
from typing import List, Optional
from uuid import UUID


class ToolRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)
    
    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        # Close the connection to the database engine
        await self.engine.dispose()

    async def create_tool(self, data: ToolCreate) -> Tool:
        async with self.Session() as session:
            async with session.begin():
                tool = Tool(**data.model_dump())
                session.add(tool)
            await session.commit()
            return tool

    async def get_tool(self, tool_id: UUID) -> Optional[Tool]:
        async with self.Session() as session:
            result = await session.execute(select(Tool).where(Tool.id == tool_id))
            return result.scalars().first()
        
    async def get_tool_by_name(self, name: str) -> Optional[Tool]:
        async with self.Session() as session:
            result = await session.execute(select(Tool).where(Tool.name == name))
            return result.scalars().first()

    async def update_tool(self, tool_id: UUID, data: ToolUpdate) -> Optional[Tool]:
        async with self.Session() as session:
            async with session.begin():
                result = await session.execute(select(Tool).where(Tool.id == tool_id))
                tool = result.scalars().first()
                if not tool:
                    return None
                for key, value in data.model_dump(exclude_unset=True).items():
                    setattr(tool, key, value)
            await session.commit()
            return tool
    
    async def search(self, keyword: str, top_k: int = 5) -> List[Tool]:
        async with self.Session() as session:
            stmt = select(Tool)
            result = await session.execute(stmt)
            tools = result.scalars().all()

            keyword = keyword.lower()
            tokens = keyword.split()

            def match_score(tool: Tool) -> int:
                score = 0
                for token in tokens:
                    if token in tool.name.lower():
                        score += 2
                    if token in (tool.description or "").lower():
                        score += 1
                return score

        scored = [(tool, match_score(tool)) for tool in tools]
        filtered = [tool for tool, score in scored if score > 0]
        return sorted(filtered, key=lambda t: match_score(t), reverse=True)[:top_k]

    async def delete_tool(self, tool_id: UUID) -> bool:
        async with self.Session() as session:
            async with session.begin():
                result = await session.execute(select(Tool).where(Tool.id == tool_id))
                tool = result.scalars().first()
                if not tool:
                    return False
                await session.delete(tool)
            await session.commit()
            return True

    async def list_all_tools(self) -> List[Tool]:
        async with self.Session() as session:
            result = await session.execute(select(Tool))
            return result.scalars().all()
