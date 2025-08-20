from fastapi import APIRouter, Depends, HTTPException, Query
from app.services.instance_manager import InstanceManagerService
from app.repositories.instance_repository import InstanceRepository
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.tool_registry import ToolRegistryService
from app.services.credential_client import CredentialClient
from app.services.async_execution_service import AsyncToolExecutionService
from app.core.database import get_session
from typing import List, Optional
from uuid import UUID


# Dependency to get current user ID
async def get_current_user_id() -> UUID:
    # TODO: Implement proper JWT token validation
    # For now, mock a user ID - replace with actual auth integration
    return UUID("af8531b1-4459-4599-8f70-a438dfca741d")

async def get_instance_service(db: AsyncSession = Depends(get_session)) -> InstanceManagerService:
    instance_repo = InstanceRepository(db)
    tool_registry = ToolRegistryService(db)
    credential_client = CredentialClient()
    
    return InstanceManagerService(
        instance_repo=instance_repo,
        tool_registry=tool_registry,
        credential_client=credential_client
    )

async def get_async_execution_service(
    instance_service: InstanceManagerService = Depends(get_instance_service)
) -> AsyncToolExecutionService:
    credential_client = CredentialClient()
    return AsyncToolExecutionService(
        instance_manager=instance_service,
        credential_client=credential_client
    )

router = APIRouter()

@router.get("/{job_id}", response_model=dict)
async def get_job_info(
    job_id: str,
    user_id: UUID = Depends(get_current_user_id),
    async_service: AsyncToolExecutionService = Depends(get_async_execution_service)
):
    """Get job information by job ID"""
    try:
        progress = await async_service.get_job_progress(job_id)
        return progress
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=dict)
async def list_user_jobs(
    status: Optional[str] = Query(None, description="Filter by job status"),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0),
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """List user's jobs with filtering"""
    try:
        # This would be implemented to query jobs from the repository
        # For now, return empty list
        return {
            "jobs": [],
            "total": 0,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
