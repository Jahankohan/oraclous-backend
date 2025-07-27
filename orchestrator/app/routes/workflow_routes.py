from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from app.repositories.workflow_repository import WorkflowRepository
from app.schemas.workflow_schema import WorkflowCreate, WorkflowUpdate, WorkflowRead
from app.services.workflow_service import WorkflowService

router = APIRouter()


@router.post("/", response_model=WorkflowRead)
async def create(workflow: WorkflowCreate, request: Request):
    repository: WorkflowRepository = request.app.state.repository
    workflow_service = WorkflowService(repository)
    return await workflow_service.create_workflow(workflow)


@router.get("/{workflow_id}", response_model=WorkflowRead)
async def get(workflow_id: UUID, request: Request):
    repository: WorkflowRepository = request.app.state.repository
    workflow_service = WorkflowService(repository)
    result = await workflow_service.get_workflow_by_id(workflow_id)
    if not result:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return result


@router.put("/{workflow_id}", response_model=WorkflowRead)
async def update(workflow_id: UUID, update_data: WorkflowUpdate, request: Request):
    repository: WorkflowRepository = request.app.state.repository
    workflow_service = WorkflowService(repository)
    result = await workflow_service.update_workflow_by_id(workflow_id, update_data)
    if not result:
        raise HTTPException(status_code=404, detail="Workflow not found or update failed")
    return result


@router.delete("/{workflow_id}")
async def delete(workflow_id: UUID, request: Request):
    repository: WorkflowRepository = request.app.state.repository
    workflow_service = WorkflowService(repository)
    success = await workflow_service.delete_workflow_by_id(workflow_id)
    if not success:
        raise HTTPException(status_code=404, detail="Workflow not found or delete failed")
    return {"status": "deleted"}


@router.get("/", response_model=list[WorkflowRead])
async def list_all(request: Request):
    repository: WorkflowRepository = request.app.state.repository
    workflow_service = WorkflowService(repository)
    return await workflow_service.list_all_workflows()
