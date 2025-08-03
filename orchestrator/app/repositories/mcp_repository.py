from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from app.models.mcp_model import MCPServer
from app.models.base import Base
from app.schemas.mcp_schema import MCPCreate, MCPCreated
from typing import List, Optional
from uuid import UUID


class MCPRepository:
    def __init__(self, db_url: str):
        self.engine = create_async_engine(db_url, echo=False)
        self.Session = sessionmaker(bind=self.engine, class_=AsyncSession, expire_on_commit=False)
    
    async def create_tables(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    
    async def close(self):
        # Close the connection to the database engine
        await self.engine.dispose()

    async def create_mcp(self, data: MCPCreate) -> MCPServer:
        async with self.Session() as session:
            async with session.begin():
                mcp_server = MCPServer(**data.model_dump())
                session.add(mcp_server)
            await session.commit()
            return mcp_server

    async def get_mcp_server(self, mcp_id: UUID) -> Optional[MCPServer]:
        async with self.Session() as session:
            result = await session.execute(select(MCPServer).where(MCPServer.id == mcp_id))
            return result.scalars().first()
        
    async def get_mcp_by_name(self, name: str) -> Optional[MCPServer]:
        async with self.Session() as session:
            result = await session.execute(select(MCPServer).where(MCPServer.name == name))
            return result.scalars().first()

    async def list_all_mcp_servers(self) -> List[MCPServer]:
        async with self.Session() as session:
            result = await session.execute(select(MCPServer))
            return result.scalars().all()
