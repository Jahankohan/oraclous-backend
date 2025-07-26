from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.registry import MCPRegistry
import json
import os
from app.routes.task import router as task_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Global registry
    servers=json.loads(os.getenv("MCP_SERVERS", ["http://github-mcp:8080/mcp", "http://qa-generator-mcp:8080/mcp", "http://postgres-writer-mcp:8080/mcp"]))

    # Initialize and load the registry
    registry = MCPRegistry(servers)
    await registry.load()
    app.state.registry = registry

    print("MCP Registry initialized with", len(registry.tools), "tools")

    yield


    # If needed: graceful shutdown logic here
    print("Shutting down MCP orchestrator")



app = FastAPI(title="Oraclous Orchestrator", lifespan=lifespan)

app.include_router(task_router, prefix="/tasks")
