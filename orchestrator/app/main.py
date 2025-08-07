from fastapi import FastAPI
from app.core.config import DATABASE_URL
from contextlib import asynccontextmanager
from app.repositories.workflow_repository import WorkflowRepository
from app.repositories.tool_repository import ToolRepository
from app.repositories.task_repository import TaskRepository
from app.repositories.mcp_repository import MCPRepository
from app.core.ingestion_setup import setup_ingestion_system
from app.routes import workflow_routes
from app.routes import tool_routes
from app.routes import task_routes
from app.routes import data_ingestion_routes

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize existing repositories
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
    
    # Initialize data ingestion system
    (data_source_repo, 
     ingestion_job_repo, 
     document_repo, 
     ingestion_registry) = await setup_ingestion_system(DATABASE_URL)
    
    app.state.data_source_repository = data_source_repo
    app.state.ingestion_job_repository = ingestion_job_repo
    app.state.document_repository = document_repo
    app.state.ingestion_registry = ingestion_registry
    
    print("All repositories and ingestion system initialized")
    
    yield

    # Graceful shutdown
    print("Shutting down orchestrator")
    await app.state.repository.close()
    await app.state.tool_repository.close()
    await app.state.task_repository.close()
    await app.state.data_source_repository.close()
    await app.state.ingestion_job_repository.close()
    await app.state.document_repository.close()

app = FastAPI(title="Oraclous Orchestrator", lifespan=lifespan)

# Include existing routes
app.include_router(task_routes.router, prefix="/tasks", tags=["Tasks"])
app.include_router(workflow_routes.router, prefix="/workflows", tags=["Workflows"])
app.include_router(tool_routes.router, prefix="/tools", tags=["Tools"])

# Include new data ingestion routes
app.include_router(data_ingestion_routes.router, tags=["Data Ingestion"])