from fastapi import FastAPI
from app.core.config import DATABASE_URL, MCP_SERVERS
from contextlib import asynccontextmanager
from app.core.tool_registry import ToolRegistry
import json
import os
from app.repositories.workflow_repository import WorkflowRepository
from app.repositories.tool_repository import ToolRepository
from app.repositories.task_repository import TaskRepository
from app.routes import workflow_routes
from orchestrator.app.routes import tool_routes
from app.routes import task_routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB repository
    repository = WorkflowRepository(db_url=DATABASE_URL)
    await repository.create_tables()
    app.state.repository = repository

    tool_repository = ToolRepository(db_url=DATABASE_URL)
    await tool_repository.create_tables()
    app.state.tool_repository = tool_repository

    task_repository = TaskRepository(db_url=DATABASE_URL)
    await task_repository.create_tables()
    app.state.task_repository = task_repository
    
    servers=MCP_SERVERS

    # Initialize and load the registry
    registry = ToolRegistry(servers=servers, persist_file="tool_registry.json")
    await registry.load_tools_from_mcp()
    app.state.registry = registry
    await registry.save_to_disk()
    print("MCP Registry initialized with", len(registry._tools), "tools")
    
    yield

    # If needed: graceful shutdown logic here
    print("Shutting down MCP orchestrator")
    await app.state.repository.close()
    await app.state.tool_repository.close()
    await app.state.task_repository.close()


app = FastAPI(title="Oraclous Orchestrator", lifespan=lifespan)

app.include_router(task_routes.router, prefix="/tasks", tags=["Tasks"])
app.include_router(workflow_routes.router, prefix="/workflows", tags=["Workflows"])
app.include_router(tool_routes.router, prefix="/tools", tags=["Tools"])

