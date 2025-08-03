from fastapi import FastAPI
from app.core.config import DATABASE_URL
from contextlib import asynccontextmanager
from app.repositories.workflow_repository import WorkflowRepository
from app.repositories.tool_repository import ToolRepository
from app.repositories.task_repository import TaskRepository
from app.repositories.mcp_repository import MCPRepository
from app.routes import workflow_routes
from app.routes import tool_routes
from app.routes import task_routes
from app.routes import auth_routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB repository
    repository = WorkflowRepository(db_url=DATABASE_URL)
    await repository.create_tables()
    app.state.repository = repository

    mcp_repository = MCPRepository(db_url=DATABASE_URL)
    await mcp_repository.create_tables()
    app.state.mcp_repository = mcp_repository

    tool_repository = ToolRepository(db_url=DATABASE_URL)
    await tool_repository.create_tables()
    app.state.tool_repository = tool_repository

    task_repository = TaskRepository(db_url=DATABASE_URL)
    await task_repository.create_tables()
    app.state.task_repository = task_repository
    # Initialize and load the registry
    
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

