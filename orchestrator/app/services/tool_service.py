from app.repositories.tool_repository import ToolRepository
from app.repositories.mcp_repository import MCPRepository
from app.models.tools_model import Tool
from app.schemas.tool_schema import ToolCreate
from app.schemas.mcp_schema import MCPCreate
from fastmcp import Client
from uuid import UUID
from typing import List, Optional, Callable


class ToolService:
    def __init__(self, repository: ToolRepository, mcp_repository: MCPRepository):
        self.repository = repository
        self.mcp_repository = mcp_repository

    async def register_from_mcp_url(self, mcp_create: MCPCreate) -> Tool:
        mcp_server = await self.mcp_repository.get_mcp_by_name(mcp_create.name)
        if not mcp_server:
            mcp_server = await self.mcp_repository.create_mcp(mcp_create)
        try:
            async with Client(mcp_create.url) as client:
                remote_tools = await client.list_tools()
                for t in remote_tools:
                    payload = ToolCreate(
                        mcp_server_id = mcp_server.id,
                        name=t.name,
                        description=t.description,
                        input_schema=t.inputSchema,
                        output_example=t.outputSchema,
                    )
                    await self.repository.create_tool(payload)

                remote_prompts = await client.list_prompts()
                for p in remote_prompts:
                    print("Prompt:", p)
                
                remote_resources = await client.list_resources()
                for r in remote_resources:
                    print("Resources:", r)
                
                remote_resources_template = await client.list_resource_templates()
                print("RR templates:", remote_resources_template)

                remote_prompts_resources = await client.list_prompts_mcp()
                print("RP MCP:", remote_prompts_resources)
        except Exception as e:
            print(f"[ToolService] Failed to register from MCP {mcp_create.url}: {e}")

        return None

    async def search(self, keyword: str, limit: int) -> List[Tool]:
        return await self.repository.search(keyword, limit)
    
    async def get_by_category(self, category: str) -> List[Tool]:
        return await self.repository.get_by_category(category)

    def _make_wrapper(self, tool: Tool) -> Callable:
        @tool(name_or_callable=tool.name, description=tool.description)
        async def wrapped_tool(**kwargs):
            async with Client(tool.url) as client:
                return await client.call_tool(tool.name, kwargs)
        return wrapped_tool

    async def to_langchain_tools(self) -> List[Callable]:
        tools = await self.repository.list_all_tools()
        return [self._make_wrapper(t) for t in tools]
    
    def infer_category(self, name: str) -> str:
        name = name.lower()
        if any(k in name for k in ["csv", "pdf", "scrape"]):
            return "Ingestion"
        elif any(k in name for k in ["augment", "generate", "simulate"]):
            return "Augmentation"
        elif any(k in name for k in ["clean", "normalize", "format"]):
            return "Transformation"
        elif any(k in name for k in ["validate", "check"]):
            return "Validation"
        elif any(k in name for k in ["train", "fine", "lora"]):
            return "Training"
        elif any(k in name for k in ["deploy", "export"]):
            return "Serving"
        return "Utility"

    async def create_tool(self, data: ToolCreate) -> Tool:
        return await self.repository.create_tool(data)

    async def get_tool_by_id(self, tool_id: UUID) -> Optional[Tool]:
        return await self.repository.get_tool(tool_id)

    async def list_all_tools(self) -> List[Tool]:
        return await self.repository.list_all_tools()
