from fastmcp import Client
from pydantic import BaseModel
from typing import Dict, List, Any, Optional, Callable
from langchain_core.tools import tool
import json
import aiofiles
import os


class ToolMetadata(BaseModel):
    name: str
    description: str
    url: str
    input_schema: Optional[dict] = None
    output_example: Optional[dict] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = []

    def validate(self) -> bool:
        if not self.name or not self.description or not self.url:
            return False
        if self.input_schema and self.input_schema.get("type") != "object":
            return False
        return True


class ToolRegistry:
    def __init__(self, servers: List[str], persist_file: str = "tool_registry.json"):
        self.servers = servers
        self._tools: Dict[str, ToolMetadata] = {}
        self.persist_file = persist_file
    
    def register_tool(self, name: str, description: str, url: str, input_schema: Optional[dict] = None, output_example: Optional[dict] = None, tags: Optional[List[str]] = None):
        meta = ToolMetadata(
            name=name,
            description=description,
            url=url,
            input_schema=input_schema,
            output_example=output_example,
            tags=tags or [],
            category=self.infer_category(name)
        )
        if meta.validate():
            self._tools[name] = meta

    async def load_tools_from_mcp(self):
        for url in self.servers:
            try:
                async with Client(url) as client:
                    tools = await client.list_tools()
                    for tool in tools:
                        meta = ToolMetadata(
                            name=tool.name,
                            description=tool.description,
                            url=url,
                            input_schema=tool.inputSchema,
                            output_example=tool.outputSchema,
                            category=self.infer_category(tool.name)
                        )
                        if meta.validate():
                            self._tools[meta.name] = meta
            except Exception as e:
                print(f"[Registry] Failed to load from {url}: {e}")

    async def save_to_disk(self):
        async with aiofiles.open(self.persist_file, mode="w") as f:
            await f.write(json.dumps([t.model_dump() for t in self._tools.values()], indent=2))

    async def load_from_disk(self):
        if not os.path.exists(self.persist_file):
            return
        async with aiofiles.open(self.persist_file, mode="r") as f:
            content = await f.read()
            raw = json.loads(content)
            for entry in raw:
                tool = ToolMetadata(**entry)
                if tool.validate():
                    self._tools[tool.name] = tool
    
    def search(self, keyword: str, top_k: int = 5) -> List[ToolMetadata]:
        keyword = keyword.lower()
        tokens = keyword.split()

        def match_score(tool: ToolMetadata) -> int:
            score = 0
            for token in tokens:
                if token in tool.name.lower():
                    score += 2
                if token in (tool.description or "").lower():
                    score += 1
            return score

        scored = [(tool, match_score(tool)) for tool in self._tools.values()]
        filtered = [tool for tool, score in scored if score > 0]
        return sorted(filtered, key=lambda t: match_score(t), reverse=True)[:top_k]


    def get_by_category(self, category: str) -> List[ToolMetadata]:
        return [t for t in self._tools.values() if t.category and t.category.lower() == category.lower()]

    def get_tool_metadata(self, name: str) -> Optional[ToolMetadata]:
        return self._tools.get(name)

    def all_tools(self) -> List[ToolMetadata]:
        return list(self._tools.values())

    def to_langchain_tools(self) -> List[Callable]:
        tools = []
        for meta in self._tools.values():
            tools.append(self._make_wrapper(meta))
        return tools

    def _make_wrapper(self, meta: ToolMetadata) -> Callable:
        @tool(name_or_callable=meta.name, description=meta.description)
        async def wrapped_tool(**kwargs):
            async with Client(meta.url) as client:
                return await client.call_tool(meta.name, kwargs)
        return wrapped_tool
    

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
