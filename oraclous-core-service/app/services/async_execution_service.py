# app/services/async_execution_service.py
import asyncio
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator
from uuid import UUID, uuid4
from datetime import datetime, timedelta
from decimal import Decimal

from app.services.instance_manager import InstanceManagerService
from app.services.credential_client import CredentialClient
from app.tools.factory import ToolFactory
from app.schemas.tool_instance import (
    ToolInstance, ExecutionContext, ExecutionResult, 
    Execution, CreateExecutionRequest, Job
)
from app.schemas.common import InstanceStatus

logger = logging.getLogger(__name__)


class AsyncToolExecutionService:
    """
    Service for handling asynchronous tool execution with job queue
    Supports long-running tasks, progress tracking, and result streaming
    """
    
    def __init__(
        self,
        instance_manager: InstanceManagerService,
        credential_client: CredentialClient
    ):
        self.instance_manager = instance_manager
        self.credential_client = credential_client
        
        # In-memory job storage (replace with Redis/DB in production)
        self.active_jobs: Dict[str, Dict[str, Any]] = {}
        self.job_results: Dict[str, Dict[str, Any]] = {}
    
    def __init__(
        self,
        instance_manager: InstanceManagerService,
        credential_client: CredentialClient,
        validation_service: Optional['ValidationService'] = None
    ):
        self.instance_manager = instance_manager
        self.credential_client = credential_client
        self.validation_service = validation_service
    
    async def submit_execution_job(
        self,
        instance_id: str,
        user_id: UUID,
        input_data: Dict[str, Any],
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Submit a tool execution as an async job
        Returns job information for tracking
        """
        try:
            # 1. Validate instance
            instance = await self.instance_manager.get_user_instance(instance_id, user_id)
            if not instance:
                raise ValueError("Tool instance not found")
            
            if instance.status != InstanceStatus.READY:
                raise ValueError(f"Tool instance not ready. Status: {instance.status}")
            
            # 2. Create execution record
            execution = await self.instance_manager.create_execution(
                instance_id=instance_id,
                user_id=user_id,
                input_data=input_data,
                max_retries=max_retries
            )
            
            # 3. Create job record
            job_id = str(uuid4())
            job = Job(
                id=job_id,
                job_type="tool_execution",
                execution_id=execution.id,
                queue_name="default",
                priority=0,
                status="QUEUED",
                job_data={
                    "instance_id": instance_id,
                    "user_id": str(user_id),
                    "input_data": input_data,
                    "max_retries": max_retries
                },
                scheduled_at=datetime.utcnow()
            )
            
            # 4. Store job in repository
            await self.instance_manager.repo.create_job(job)
            
            # 5. Add to active jobs for processing
            self.active_jobs[job_id] = {
                "job": job,
                "execution": execution,
                "instance": instance,
                "status": "QUEUED",
                "progress": 0,
                "current_step": None,
                "created_at": datetime.utcnow()
            }
            
            # 6. Start processing (fire and forget)
            asyncio.create_task(self._process_job(job_id))
            
            return {
                "job_id": job_id,
                "execution_id": execution.id,
                "status": "QUEUED",
                "estimated_duration": self._estimate_execution_duration(instance),
                "progress_url": f"/api/v1/jobs/{job_id}/progress",
                "result_url": f"/api/v1/jobs/{job_id}/result"
            }
            
        except Exception as e:
            logger.error(f"Failed to submit execution job: {str(e)}")
            raise
    
    async def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """
        Get current progress of a job
        """
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job_info["status"],
                "progress": job_info["progress"],
                "current_step": job_info["current_step"],
                "started_at": job_info.get("started_at"),
                "estimated_completion": job_info.get("estimated_completion"),
                "error_message": job_info.get("error_message")
            }
        elif job_id in self.job_results:
            result_info = self.job_results[job_id]
            return {
                "job_id": job_id,
                "status": result_info["status"],
                "progress": 100 if result_info["status"] == "COMPLETED" else 0,
                "completed_at": result_info.get("completed_at"),
                "error_message": result_info.get("error_message")
            }
        else:
            raise ValueError(f"Job {job_id} not found")
    
    async def get_job_result(self, job_id: str) -> Dict[str, Any]:
        """
        Get final result of a completed job
        """
        if job_id in self.job_results:
            return self.job_results[job_id]
        elif job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            return {
                "job_id": job_id,
                "status": job_info["status"],
                "message": "Job still processing"
            }
        else:
            raise ValueError(f"Job {job_id} not found")
    
    async def stream_job_progress(self, job_id: str) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream job progress updates (for WebSocket or SSE)
        """
        last_progress = -1
        last_status = None
        
        while True:
            try:
                progress_info = await self.get_job_progress(job_id)
                
                # Yield update if progress or status changed
                if (progress_info["progress"] != last_progress or 
                    progress_info["status"] != last_status):
                    
                    yield progress_info
                    last_progress = progress_info["progress"]
                    last_status = progress_info["status"]
                
                # Break if job is completed or failed
                if progress_info["status"] in ["COMPLETED", "FAILED", "CANCELLED"]:
                    break
                
                # Wait before next check
                await asyncio.sleep(1)
                
            except ValueError:
                # Job not found
                break
            except Exception as e:
                logger.error(f"Error streaming progress for job {job_id}: {str(e)}")
                yield {
                    "job_id": job_id,
                    "status": "ERROR",
                    "error_message": str(e)
                }
                break
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a running job
        """
        if job_id in self.active_jobs:
            job_info = self.active_jobs[job_id]
            if job_info["status"] in ["QUEUED", "RUNNING"]:
                job_info["status"] = "CANCELLED"
                job_info["cancelled_at"] = datetime.utcnow()
                
                # Update execution record
                await self.instance_manager.repo.update_execution(
                    job_info["execution"].id,
                    {
                        "status": "CANCELLED",
                        "completed_at": datetime.utcnow(),
                        "error_message": "Job cancelled by user"
                    }
                )
                
                # Update job record
                await self.instance_manager.repo.update_job(
                    job_id,
                    {
                        "status": "CANCELLED",
                        "completed_at": datetime.utcnow()
                    }
                )
                
                logger.info(f"Job {job_id} cancelled successfully")
                return True
            else:
                logger.warning(f"Cannot cancel job {job_id} in status {job_info['status']}")
                return False
        else:
            logger.warning(f"Job {job_id} not found for cancellation")
            return False
    
    # ================== INTERNAL PROCESSING ==================
    
    async def _process_job(self, job_id: str):
        """
        Internal method to process a job asynchronously
        """
        job_info = self.active_jobs.get(job_id)
        if not job_info:
            logger.error(f"Job {job_id} not found in active jobs")
            return
        
        try:
            job = job_info["job"]
            execution = job_info["execution"]
            instance = job_info["instance"]
            
            # Update status to running
            job_info["status"] = "RUNNING"
            job_info["started_at"] = datetime.utcnow()
            job_info["progress"] = 10
            job_info["current_step"] = "Initializing execution"
            
            await self._update_job_status(job_id, "RUNNING", {"started_at": datetime.utcnow()})
            await self._update_execution_status(execution.id, "RUNNING", {"started_at": datetime.utcnow()})
            
            # Build execution context
            job_info["current_step"] = "Resolving credentials"
            job_info["progress"] = 20
            
            context = await self._build_execution_context(
                instance, execution, UUID(job.job_data["user_id"])
            )
            
            # Execute tool with progress tracking
            job_info["current_step"] = "Executing tool"
            job_info["progress"] = 30
            
            result = await self._execute_with_progress(job_id, instance, job.job_data["input_data"], context)
            
            # Store result
            job_info["progress"] = 100
            job_info["current_step"] = "Completed"
            
            # Move to completed jobs
            self.job_results[job_id] = {
                "job_id": job_id,
                "execution_id": execution.id,
                "status": "COMPLETED" if result.success else "FAILED",
                "result": {
                    "success": result.success,
                    "data": result.data,
                    "error_message": result.error_message,
                    "error_type": result.error_type,
                    "credits_consumed": float(result.credits_consumed),
                    "processing_time_ms": result.processing_time_ms,
                    "metadata": result.metadata
                },
                "completed_at": datetime.utcnow()
            }
            
            # Update records
            await self._update_execution_with_result(execution.id, result)
            await self._update_job_status(
                job_id, 
                "COMPLETED" if result.success else "FAILED",
                {
                    "completed_at": datetime.utcnow(),
                    "result_data": self.job_results[job_id]["result"]
                }
            )
            
            # Clean up active job
            del self.active_jobs[job_id]
            
            logger.info(f"Job {job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed with error: {str(e)}")
            
            # Handle job failure
            job_info["status"] = "FAILED"
            job_info["error_message"] = str(e)
            job_info["completed_at"] = datetime.utcnow()
            
            # Store failed result
            self.job_results[job_id] = {
                "job_id": job_id,
                "execution_id": execution.id,
                "status": "FAILED",
                "error_message": str(e),
                "error_type": type(e).__name__,
                "completed_at": datetime.utcnow()
            }
            
            # Update records
            await self._update_execution_status(
                execution.id, "FAILED", 
                {
                    "completed_at": datetime.utcnow(),
                    "error_message": str(e),
                    "error_type": type(e).__name__
                }
            )
            await self._update_job_status(job_id, "FAILED", {"completed_at": datetime.utcnow()})
            
            # Clean up active job
            del self.active_jobs[job_id]
    
    async def _execute_with_progress(
        self, 
        job_id: str,
        instance: ToolInstance, 
        input_data: Dict[str, Any], 
        context: ExecutionContext
    ) -> ExecutionResult:
        """
        Execute tool with progress updates
        """
        job_info = self.active_jobs.get(job_id)
        
        try:
            # Create executor
            executor = ToolFactory.create_executor(instance.tool_definition_id)
            
            # Update progress
            if job_info:
                job_info["progress"] = 40
                job_info["current_step"] = "Starting tool execution"
            
            # Execute (this would be enhanced to support progress callbacks in real tools)
            result = await ToolFactory.execute_tool(instance, input_data, context)
            
            # Update progress
            if job_info:
                job_info["progress"] = 90
                job_info["current_step"] = "Finalizing results"
            
            return result
            
        except Exception as e:
            logger.error(f"Tool execution failed for job {job_id}: {str(e)}")
            raise
    
    # ================== HELPER METHODS ==================
    
    async def _build_execution_context(
        self,
        instance: ToolInstance,
        execution: Execution,
        user_id: UUID
    ) -> ExecutionContext:
        """Build execution context with resolved credentials"""
        # Same logic as synchronous service
        credentials = {}
        
        if instance.credential_mappings:
            for cred_type, cred_identifier in instance.credential_mappings.items():
                if cred_type == "OAUTH_TOKEN":
                    token_data = await self.credential_client.get_runtime_token(
                        user_id=user_id,
                        provider=cred_identifier
                    )
                    if token_data:
                        credentials[cred_type] = token_data
                else:
                    cred_data = await self.credential_client._get_credential_data(cred_identifier)
                    if cred_data:
                        credentials[cred_type] = cred_data
        
        return ExecutionContext(
            instance_id=instance.id,
            workflow_id=str(instance.workflow_id),
            user_id=str(user_id),
            job_id=execution.id,
            credentials=credentials,
            configuration=instance.configuration,
            settings=instance.settings
        )
    
    async def _update_execution_status(self, execution_id: str, status: str, updates: Dict[str, Any]):
        """Update execution record"""
        updates["status"] = status
        await self.instance_manager.repo.update_execution(execution_id, updates)
    
    async def _update_job_status(self, job_id: str, status: str, updates: Dict[str, Any]):
        """Update job record"""
        updates["status"] = status
        await self.instance_manager.repo.update_job(job_id, updates)
    
    async def _update_execution_with_result(self, execution_id: str, result: ExecutionResult):
        """Update execution with final result"""
        update_data = {
            "status": "SUCCESS" if result.success else "FAILED",
            "completed_at": datetime.utcnow(),
            "credits_consumed": result.credits_consumed,
            "processing_time_ms": result.processing_time_ms
        }
        
        if result.success:
            update_data["output_data"] = result.data
            update_data["execution_metadata"] = result.metadata
        else:
            update_data["error_message"] = result.error_message
            update_data["error_type"] = result.error_type
        
        await self.instance_manager.repo.update_execution(execution_id, update_data)
    
    def _estimate_execution_duration(self, instance: ToolInstance) -> Optional[int]:
        """
        Estimate execution duration based on tool type and history
        Returns estimated seconds
        """
        # This would be enhanced with ML-based estimation based on:
        # - Tool type
        # - Input data size
        # - Historical execution times
        # - Current system load
        
        # Simple heuristic for now
        tool_estimates = {
            "GoogleDriveReader": 30,
            "PostgreSQLReader": 15,
            "MySQLReader": 15,
            "NotionReader": 45
        }
        
        # Get tool name from instance (would need tool definition lookup)
        return tool_estimates.get("default", 60)
