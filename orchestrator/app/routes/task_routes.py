from fastapi import APIRouter, HTTPException, Request
from uuid import UUID
from typing import List
from app.services.task_service import TaskService
from app.repositories.task_repository import TaskRepository
from app.schemas.task_schema import TaskCreate, TaskRead, TaskUpdate

router = APIRouter(prefix="/tasks", tags=["Tasks"])

@router.post("/", response_model=TaskRead)
async def create_task(task: TaskCreate, request: Request):
    repo: TaskRepository = request.app.state.task_repository
    service = TaskService(repo)
    return await service.create_task(task)

@router.get("/{task_id}", response_model=TaskRead)
async def get_task(task_id: UUID, request: Request):
    repo: TaskRepository = request.app.state.task_repository
    service = TaskService(repo)
    task = await service.get_task_by_id(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@router.put("/{task_id}", response_model=TaskRead)
async def update_task(task_id: UUID, update_data: TaskUpdate, request: Request):
    repo: TaskRepository = request.app.state.task_repository
    service = TaskService(repo)
    updated = await service.update_task_by_id(task_id, update_data)
    if not updated:
        raise HTTPException(status_code=404, detail="Task not found or update failed")
    return updated

@router.delete("/{task_id}")
async def delete_task(task_id: UUID, request: Request):
    repo: TaskRepository = request.app.state.task_repository
    service = TaskService(repo)
    success = await service.delete_task_by_id(task_id)
    if not success:
        raise HTTPException(status_code=404, detail="Task not found or delete failed")
    return {"status": "deleted"}

@router.get("/", response_model=List[TaskRead])
async def list_tasks(request: Request):
    repo: TaskRepository = request.app.state.task_repository
    service = TaskService(repo)
    return await service.list_all_tasks()
