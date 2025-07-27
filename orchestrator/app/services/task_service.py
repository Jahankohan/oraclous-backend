from app.repositories.task_repository import TaskRepository
from app.models.task_model import Task
from app.schemas.task_schema import TaskCreate, TaskUpdate
from uuid import UUID
from typing import Optional, List

class TaskService:
    def __init__(self, repository: TaskRepository):
        self.repository = repository

    async def create_task(self, data: TaskCreate) -> Task:
        return await self.repository.create_task(data)

    async def get_task_by_id(self, task_id: UUID) -> Optional[Task]:
        return await self.repository.get_task(task_id)

    async def update_task_by_id(self, task_id: UUID, data: TaskUpdate) -> Optional[Task]:
        return await self.repository.update_task(task_id, data)

    async def delete_task_by_id(self, task_id: UUID) -> bool:
        return await self.repository.delete_task(task_id)

    async def list_all_tasks(self) -> List[Task]:
        return await self.repository.list_all_tasks()