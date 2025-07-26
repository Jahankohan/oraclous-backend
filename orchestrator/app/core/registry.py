from fastmcp import Client
from typing import Dict, List, Any

class MCPRegistry:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.tools: List[Dict[str, Any]] = []

    async def load(self):
        self.tools.clear()

        for url in self.servers:
            try:
                async with Client(url) as client:
                    await client.ping()
                    tools = await client.list_tools()
                    for tool in tools:
                        self.tools.append({
                            "tool": tool,
                            "url": url,
                            "name": tool.name
                        })
                    print(f"Loaded {len(tools)} tools from {url}")
            except Exception as e:
                print(f"Failed to load from {url}: {e}")

    def search(self, keyword: str, top_k: int = 5):
        return sorted(
            self.tools,
            key=lambda t: keyword.lower() in (t["tool"].description or "").lower(),
            reverse=True
        )[:top_k]

    async def call_tool(self, tool_name: str, url: str, args: dict):
        async with Client(url) as client:
            return await client.call_tool(tool_name, args)