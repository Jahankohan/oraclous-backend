from fastapi import APIRouter, Depends, HTTPException, Query, Path
from fastapi.security import OAuth2PasswordBearer
from typing import List, Optional
from uuid import UUID

from app.schemas.tool_instance import (
    ToolInstance, CreateInstanceRequest, UpdateInstanceRequest,
    ConfigureCredentialsRequest, InstanceStatusResponse,
    InstanceListResponse, Execution, CreateExecutionRequest
)
from app.schemas.common import InstanceStatus
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.instance_manager import InstanceManagerService
from app.repositories.instance_repository import InstanceRepository
from app.services.tool_registry import ToolRegistryService
from app.services.credential_client import CredentialClient
from app.core.database import get_session

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
router = APIRouter()


# Dependency to get current user ID
async def get_current_user_id() -> UUID:
    # TODO: Implement proper JWT token validation
    # For now, mock a user ID - replace with actual auth integration
    return UUID("550e8400-e29b-41d4-a716-446655440000")


# Dependency to get instance service
async def get_instance_service(db: AsyncSession = Depends(get_session)) -> InstanceManagerService:
    instance_repo = InstanceRepository(db)
    tool_registry = ToolRegistryService(db)
    credential_client = CredentialClient()
    
    return InstanceManagerService(
        instance_repo=instance_repo,
        tool_registry=tool_registry,
        credential_client=credential_client
    )


@router.post("/", response_model=ToolInstance, status_code=201)
async def create_instance(
    request: CreateInstanceRequest,
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """Create a new tool instance"""
    try:
        instance = await service.create_instance(
            user_id=user_id,
            tool_definition_id=request.tool_definition_id,
            workflow_id=request.workflow_id,
            configuration=request.configuration
        )
        return instance
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create instance: {str(e)}")


@router.get("/{instance_id}", response_model=ToolInstance)
async def get_instance(
    instance_id: str = Path(..., description="Instance ID"),
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """Get a specific tool instance"""
    instance = await service.get_user_instance(instance_id, user_id)
    if not instance:
        raise HTTPException(status_code=404, detail="Instance not found")
    return instance


@router.put("/{instance_id}", response_model=ToolInstance)
async def update_instance(
    instance_id: str,
    request: UpdateInstanceRequest,
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """Update tool instance configuration"""
    try:
        # Update using the repository directly for now
        updated_instance = await service.repo.update_instance(instance_id, user_id, request)
        if not updated_instance:
            raise HTTPException(status_code=404, detail="Instance not found")
        return updated_instance
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update instance: {str(e)}")


@router.post("/{instance_id}/configure-credentials", response_model=InstanceStatusResponse)
async def configure_instance_credentials(
    instance_id: str,
    request: ConfigureCredentialsRequest,
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """Configure credentials for a tool instance"""
    try:
        status_response = await service.configure_credentials(instance_id, user_id, request)
        return status_response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to configure credentials: {str(e)}")


@router.get("/{instance_id}/status", response_model=InstanceStatusResponse)
async def get_instance_status(
    instance_id: str,
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """Get complete status information for an instance"""
    try:
        status_response = await service.get_instance_status(instance_id, user_id)
        return status_response
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get instance status: {str(e)}")


@router.get("/", response_model=InstanceListResponse)
async def list_instances(
    workflow_id: Optional[str] = Query(None, description="Filter by workflow ID"),
    status: Optional[InstanceStatus] = Query(None, description="Filter by status"),
    page: int = Query(0, ge=0, description="Page number"),
    size: int = Query(50, ge=1, le=100, description="Page size"),
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """List user's tool instances with filtering and pagination"""
    try:
        instances, total = await service.list_user_instances(
            user_id=user_id,
            workflow_id=workflow_id,
            status=status,
            page=page,
            size=size
        )
        
        return InstanceListResponse(
            instances=instances,
            total=total,
            page=page,
            size=size
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list instances: {str(e)}")


@router.delete("/{instance_id}", status_code=204)
async def delete_instance(
    instance_id: str,
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """Delete a tool instance"""
    try:
        success = await service.delete_instance(instance_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Instance not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete instance: {str(e)}")


@router.post("/{instance_id}/execute", response_model=Execution, status_code=201)
async def create_execution(
    instance_id: str,
    request: CreateExecutionRequest,
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """Create an execution for a tool instance"""
    try:
        # Update request with instance_id if not provided
        if not request.instance_id:
            request.instance_id = instance_id
        elif request.instance_id != instance_id:
            raise ValueError("Instance ID mismatch between path and body")
        
        execution = await service.create_execution(
            instance_id=instance_id,
            user_id=user_id,
            input_data=request.input_data,
            max_retries=request.max_retries
        )
        return execution
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create execution: {str(e)}")


@router.get("/{instance_id}/executions", response_model=List[Execution])
async def list_instance_executions(
    instance_id: str,
    page: int = Query(0, ge=0, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
    status: Optional[str] = Query(None, description="Filter by execution status"),
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """List executions for a specific instance"""
    try:
        # First verify user owns this instance
        instance = await service.get_user_instance(instance_id, user_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Instance not found")
        
        executions, total = await service.repo.list_executions(
            user_id=user_id,
            instance_id=instance_id,
            status=status,
            page=page,
            size=size
        )
        
        return executions
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list executions: {str(e)}")


# Additional utility endpoints

@router.get("/{instance_id}/validate", response_model=dict)
async def validate_instance_ready(
    instance_id: str,
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """Check if instance is ready for execution"""
    try:
        # Verify user owns this instance
        instance = await service.get_user_instance(instance_id, user_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Instance not found")
        
        is_ready = await service.validate_instance_ready(instance_id)
        
        return {
            "instance_id": instance_id,
            "is_ready": is_ready,
            "status": instance.status.value,
            "message": "Instance is ready for execution" if is_ready else "Instance requires configuration"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate instance: {str(e)}")


@router.get("/workflow/{workflow_id}/instances", response_model=List[ToolInstance])
async def list_workflow_instances(
    workflow_id: str,
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """Get all instances for a specific workflow"""
    try:
        # Filter by user to ensure they only see their own instances
        instances, _ = await service.list_user_instances(
            user_id=user_id,
            workflow_id=workflow_id,
            size=1000  # Large limit to get all instances for the workflow
        )
        
        return instances
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list workflow instances: {str(e)}")


# Credential-related helper endpoints

@router.get("/{instance_id}/available-credentials", response_model=dict)
async def get_available_credentials(
    instance_id: str,
    user_id: UUID = Depends(get_current_user_id),
    service: InstanceManagerService = Depends(get_instance_service)
):
    """Get available credentials/data sources for the instance's tool"""
    try:
        # Get instance and tool definition
        instance = await service.get_user_instance(instance_id, user_id)
        if not instance:
            raise HTTPException(status_code=404, detail="Instance not found")
        
        tool_definition = await service.tool_registry.get_tool(instance.tool_definition_id)
        if not tool_definition:
            raise HTTPException(status_code=404, detail="Tool definition not found")
        
        # Get available data sources from credential client
        available_sources = await service.credential_client.get_available_data_sources(user_id)
        
        # Filter based on tool requirements
        relevant_sources = {}
        for cred_req in tool_definition.credential_requirements:
            if cred_req.type.value == "OAUTH_TOKEN":
                # For OAuth, show available providers
                for provider, sources in available_sources.items():
                    if sources:  # Only show providers with available sources
                        relevant_sources[provider] = sources
        
        return {
            "instance_id": instance_id,
            "tool_name": tool_definition.name,
            "required_credentials": [req.type.value for req in tool_definition.credential_requirements if req.required],
            "optional_credentials": [req.type.value for req in tool_definition.credential_requirements if not req.required],
            "available_data_sources": relevant_sources
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get available credentials: {str(e)}")
