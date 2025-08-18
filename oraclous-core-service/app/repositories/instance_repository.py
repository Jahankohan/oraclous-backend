# app/repositories/instance_repository.py
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func, update, delete
from sqlalchemy.orm import selectinload
from uuid import UUID

from app.models.tool_instance import ToolInstanceDB
from app.models.execution import ExecutionDB
from app.models.jobs import JobDB
from app.schemas.tool_instance import (
    ToolInstance, CreateInstanceRequest, UpdateInstanceRequest,
    Execution, CreateExecutionRequest, Job
)
from app.schemas.common import InstanceStatus


class InstanceRepository:
    """Repository for tool instance data operations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_instance(
        self, 
        user_id: UUID, 
        request: CreateInstanceRequest
    ) -> ToolInstance:
        """Create a new tool instance"""
        # Create DB instance
        db_instance = ToolInstanceDB(
            workflow_id=request.workflow_id,
            tool_definition_id=request.tool_definition_id,
            user_id=str(user_id),
            name=request.name,
            description=request.description,
            configuration=request.configuration,
            settings=request.settings,
            status=InstanceStatus.PENDING.value
        )
        
        self.db.add(db_instance)
        await self.db.commit()
        await self.db.refresh(db_instance)
        
        return self._db_instance_to_pydantic(db_instance)
    
    async def get_instance(self, instance_id: str) -> Optional[ToolInstance]:
        """Get instance by ID"""
        query = select(ToolInstanceDB).where(ToolInstanceDB.id == instance_id)
        result = await self.db.execute(query)
        db_instance = result.scalar_one_or_none()
        
        if not db_instance:
            return None
        
        return self._db_instance_to_pydantic(db_instance)
    
    async def get_user_instance(self, instance_id: str, user_id: UUID) -> Optional[ToolInstance]:
        """Get instance by ID, ensuring it belongs to user"""
        query = select(ToolInstanceDB).where(
            and_(
                ToolInstanceDB.id == instance_id,
                ToolInstanceDB.user_id == str(user_id)
            )
        )
        result = await self.db.execute(query)
        db_instance = result.scalar_one_or_none()
        
        if not db_instance:
            return None
        
        return self._db_instance_to_pydantic(db_instance)
    
    async def update_instance(
        self, 
        instance_id: str, 
        user_id: UUID,
        request: UpdateInstanceRequest
    ) -> Optional[ToolInstance]:
        """Update instance"""
        # Build update data
        update_data = {}
        if request.name is not None:
            update_data["name"] = request.name
        if request.description is not None:
            update_data["description"] = request.description
        if request.configuration is not None:
            update_data["configuration"] = request.configuration
        if request.settings is not None:
            update_data["settings"] = request.settings
        
        if not update_data:
            # No updates requested, return current instance
            return await self.get_user_instance(instance_id, user_id)
        
        # Add updated timestamp
        from datetime import datetime
        update_data["updated_at"] = datetime.utcnow()
        
        # Perform update
        query = update(ToolInstanceDB).where(
            and_(
                ToolInstanceDB.id == instance_id,
                ToolInstanceDB.user_id == str(user_id)
            )
        ).values(**update_data)
        
        result = await self.db.execute(query)
        
        if result.rowcount == 0:
            return None
        
        await self.db.commit()
        return await self.get_instance(instance_id)
    
    async def update_instance_status(
        self, 
        instance_id: str, 
        status: InstanceStatus
    ) -> bool:
        """Update instance status"""
        from datetime import datetime
        
        query = update(ToolInstanceDB).where(
            ToolInstanceDB.id == instance_id
        ).values(
            status=status.value,
            updated_at=datetime.utcnow()
        )
        
        result = await self.db.execute(query)
        await self.db.commit()
        
        return result.rowcount > 0
    
    async def configure_credentials(
        self, 
        instance_id: str, 
        user_id: UUID,
        credential_mappings: Dict[str, str]
    ) -> Optional[ToolInstance]:
        """Configure credentials for instance"""
        from datetime import datetime
        
        query = update(ToolInstanceDB).where(
            and_(
                ToolInstanceDB.id == instance_id,
                ToolInstanceDB.user_id == str(user_id)
            )
        ).values(
            credential_mappings=credential_mappings,
            updated_at=datetime.utcnow()
        )
        
        result = await self.db.execute(query)
        
        if result.rowcount == 0:
            return None
        
        await self.db.commit()
        return await self.get_instance(instance_id)
    
    async def list_instances(
        self,
        user_id: Optional[UUID] = None,
        workflow_id: Optional[str] = None,
        status: Optional[InstanceStatus] = None,
        tool_definition_id: Optional[str] = None,
        page: int = 0,
        size: int = 50
    ) -> tuple[List[ToolInstance], int]:
        """List instances with filtering and pagination"""
        # Build query conditions
        conditions = []
        
        if user_id:
            conditions.append(ToolInstanceDB.user_id == str(user_id))
        if workflow_id:
            conditions.append(ToolInstanceDB.workflow_id == workflow_id)
        if status:
            conditions.append(ToolInstanceDB.status == status.value)
        if tool_definition_id:
            conditions.append(ToolInstanceDB.tool_definition_id == tool_definition_id)
        
        # Count query
        count_query = select(func.count(ToolInstanceDB.id))
        if conditions:
            count_query = count_query.where(and_(*conditions))
        
        count_result = await self.db.execute(count_query)
        total = count_result.scalar()
        
        # Data query
        data_query = select(ToolInstanceDB).options(
            selectinload(ToolInstanceDB.tool_definition)
        )
        
        if conditions:
            data_query = data_query.where(and_(*conditions))
        
        data_query = data_query.order_by(ToolInstanceDB.created_at.desc())
        data_query = data_query.offset(page * size).limit(size)
        
        result = await self.db.execute(data_query)
        db_instances = result.scalars().all()
        
        instances = [self._db_instance_to_pydantic(db_inst) for db_inst in db_instances]
        
        return instances, total
    
    async def delete_instance(self, instance_id: str, user_id: UUID) -> bool:
        """Delete instance"""
        query = delete(ToolInstanceDB).where(
            and_(
                ToolInstanceDB.id == instance_id,
                ToolInstanceDB.user_id == str(user_id)
            )
        )
        
        result = await self.db.execute(query)
        await self.db.commit()
        
        return result.rowcount > 0
    
    async def get_instances_by_workflow(self, workflow_id: str) -> List[ToolInstance]:
        """Get all instances for a workflow"""
        query = select(ToolInstanceDB).where(
            ToolInstanceDB.workflow_id == workflow_id
        ).options(selectinload(ToolInstanceDB.tool_definition))
        
        result = await self.db.execute(query)
        db_instances = result.scalars().all()
        
        return [self._db_instance_to_pydantic(db_inst) for db_inst in db_instances]
    
    # ================== EXECUTION OPERATIONS ==================
    
    async def create_execution(
        self, 
        user_id: UUID, 
        request: CreateExecutionRequest
    ) -> Execution:
        """Create a new execution record"""
        # Get instance to validate and get workflow_id
        instance = await self.get_user_instance(request.instance_id, user_id)
        if not instance:
            raise ValueError(f"Instance {request.instance_id} not found for user {user_id}")
        
        # Create execution record
        db_execution = ExecutionDB(
            workflow_id=instance.workflow_id,
            instance_id=request.instance_id,
            user_id=str(user_id),
            status='QUEUED',
            input_data=request.input_data,
            max_retries=request.max_retries
        )
        
        self.db.add(db_execution)
        await self.db.commit()
        await self.db.refresh(db_execution)
        
        return self._db_execution_to_pydantic(db_execution)
    
    async def get_execution(self, execution_id: str) -> Optional[Execution]:
        """Get execution by ID"""
        query = select(ExecutionDB).where(ExecutionDB.id == execution_id)
        result = await self.db.execute(query)
        db_execution = result.scalar_one_or_none()
        
        if not db_execution:
            return None
        
        return self._db_execution_to_pydantic(db_execution)
    
    async def update_execution(
        self, 
        execution_id: str, 
        updates: Dict[str, Any]
    ) -> Optional[Execution]:
        """Update execution record"""
        from datetime import datetime
        updates["updated_at"] = datetime.utcnow()
        
        query = update(ExecutionDB).where(
            ExecutionDB.id == execution_id
        ).values(**updates)
        
        result = await self.db.execute(query)
        
        if result.rowcount == 0:
            return None
        
        await self.db.commit()
        return await self.get_execution(execution_id)
    
    async def list_executions(
        self,
        user_id: Optional[UUID] = None,
        instance_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        page: int = 0,
        size: int = 50
    ) -> tuple[List[Execution], int]:
        """List executions with filtering and pagination"""
        conditions = []
        
        if user_id:
            conditions.append(ExecutionDB.user_id == str(user_id))
        if instance_id:
            conditions.append(ExecutionDB.instance_id == instance_id)
        if workflow_id:
            conditions.append(ExecutionDB.workflow_id == workflow_id)
        if status:
            conditions.append(ExecutionDB.status == status)
        
        # Count query
        count_query = select(func.count(ExecutionDB.id))
        if conditions:
            count_query = count_query.where(and_(*conditions))
        
        count_result = await self.db.execute(count_query)
        total = count_result.scalar()
        
        # Data query
        data_query = select(ExecutionDB)
        if conditions:
            data_query = data_query.where(and_(*conditions))
        
        data_query = data_query.order_by(ExecutionDB.created_at.desc())
        data_query = data_query.offset(page * size).limit(size)
        
        result = await self.db.execute(data_query)
        db_executions = result.scalars().all()
        
        executions = [self._db_execution_to_pydantic(db_exec) for db_exec in db_executions]
        
        return executions, total
    
    # ================== JOB OPERATIONS ==================
    
    async def create_job(self, job: Job) -> Job:
        """Create a new job record"""
        db_job = JobDB(
            id=job.id,
            job_type=job.job_type,
            execution_id=job.execution_id,
            queue_name=job.queue_name,
            priority=job.priority,
            status=job.status,
            job_data=job.job_data,
            retry_count=job.retry_count,
            scheduled_at=job.scheduled_at
        )
        
        self.db.add(db_job)
        await self.db.commit()
        await self.db.refresh(db_job)
        
        return self._db_job_to_pydantic(db_job)
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        query = select(JobDB).where(JobDB.id == job_id)
        result = await self.db.execute(query)
        db_job = result.scalar_one_or_none()
        
        if not db_job:
            return None
        
        return self._db_job_to_pydantic(db_job)
    
    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> Optional[Job]:
        """Update job record"""
        from datetime import datetime
        updates["updated_at"] = datetime.utcnow()
        
        query = update(JobDB).where(JobDB.id == job_id).values(**updates)
        
        result = await self.db.execute(query)
        
        if result.rowcount == 0:
            return None
        
        await self.db.commit()
        return await self.get_job(job_id)
    
    # ================== HELPER METHODS ==================
    
    def _db_instance_to_pydantic(self, db_instance: ToolInstanceDB) -> ToolInstance:
        """Convert DB model to Pydantic model"""
        from decimal import Decimal
        
        return ToolInstance(
            id=db_instance.id,
            workflow_id=db_instance.workflow_id,
            tool_definition_id=db_instance.tool_definition_id,
            user_id=db_instance.user_id,
            name=db_instance.name,
            description=db_instance.description,
            configuration=db_instance.configuration or {},
            settings=db_instance.settings or {},
            credential_mappings=db_instance.credential_mappings or {},
            required_credentials=db_instance.required_credentials or [],
            status=InstanceStatus(db_instance.status),
            last_execution_id=db_instance.last_execution_id,
            execution_count=int(db_instance.execution_count or 0),
            total_credits_consumed=Decimal(str(db_instance.total_credits_consumed or 0)),
            created_at=db_instance.created_at,
            updated_at=db_instance.updated_at
        )
    
    def _db_execution_to_pydantic(self, db_execution: ExecutionDB) -> Execution:
        """Convert DB execution model to Pydantic model"""
        from decimal import Decimal
        
        return Execution(
            id=db_execution.id,
            workflow_id=db_execution.workflow_id,
            instance_id=db_execution.instance_id,
            user_id=db_execution.user_id,
            status=db_execution.status,
            input_data=db_execution.input_data,
            output_data=db_execution.output_data,
            error_message=db_execution.error_message,
            error_type=db_execution.error_type,
            retry_count=int(db_execution.retry_count or 0),
            max_retries=int(db_execution.max_retries or 3),
            credits_consumed=Decimal(str(db_execution.credits_consumed or 0)),
            processing_time_ms=int(db_execution.processing_time_ms) if db_execution.processing_time_ms else None,
            execution_metadata=db_execution.execution_metadata or {},
            created_at=db_execution.created_at,
            queued_at=db_execution.queued_at,
            started_at=db_execution.started_at,
            completed_at=db_execution.completed_at
        )
    
    def _db_job_to_pydantic(self, db_job: JobDB) -> Job:
        """Convert DB job model to Pydantic model"""
        return Job(
            id=db_job.id,
            job_type=db_job.job_type,
            execution_id=db_job.execution_id,
            queue_name=db_job.queue_name,
            priority=int(db_job.priority or 0),
            status=db_job.status,
            worker_id=db_job.worker_id,
            job_data=db_job.job_data,
            result_data=db_job.result_data,
            error_details=db_job.error_details,
            retry_count=int(db_job.retry_count or 0),
            created_at=db_job.created_at,
            scheduled_at=db_job.scheduled_at,
            started_at=db_job.started_at,
            completed_at=db_job.completed_at
        )
