"""
Background Job Service - Clean Interface for API Endpoints
Provides a professional layer between REST endpoints and Celery background jobs
"""

from typing import Any

from app.core.logging import get_logger
from app.services.background_jobs import (
    code_ingest_task,
    ingest_image_task,
    process_ingestion_job,
)

logger = get_logger(__name__)


class BackgroundJobService:
    """
    Professional service layer for managing background jobs
    Abstracts away Celery implementation details from API endpoints
    """

    @staticmethod
    def start_ingestion_job(job_id: str, user_id: str) -> dict[str, Any]:
        """
        Start a data ingestion background job

        Args:
            job_id: Database ID of the ingestion job
            user_id: User who initiated the job

        Returns:
            Job info with task_id and status
        """
        try:
            task = process_ingestion_job.delay(job_id, user_id)

            logger.info(f"Started ingestion job {job_id} with task {task.id}")

            return {
                "task_id": task.id,
                "job_id": job_id,
                "status": "started",
                "message": "Ingestion job started successfully",
            }

        except Exception as e:
            logger.error(f"Failed to start ingestion job {job_id}: {e}")
            return {
                "task_id": None,
                "job_id": job_id,
                "status": "failed",
                "message": f"Failed to start job: {str(e)}",
            }

    @staticmethod
    def start_code_ingest_job(job_id: str, user_id: str) -> dict[str, Any]:
        """
        Start a code repository ingestion background job.

        Args:
            job_id: Database ID of the IngestionJob (source_type='code').
            user_id: User who initiated the job.

        Returns:
            Job info with task_id and status.
        """
        try:
            task = code_ingest_task.delay(job_id, user_id)
            logger.info(f"Started code ingest job {job_id} with task {task.id}")
            return {
                "task_id": task.id,
                "job_id": job_id,
                "status": "started",
                "message": "Code ingestion job started successfully",
            }
        except Exception as e:
            logger.error(f"Failed to start code ingest job {job_id}: {e}")
            return {
                "task_id": None,
                "job_id": job_id,
                "status": "failed",
                "message": f"Failed to start job: {str(e)}",
            }

    @staticmethod
    def start_image_ingestion_job(job_id: str, user_id: str) -> dict[str, Any]:
        """
        Start an image ingestion background job (vision extraction path).

        Args:
            job_id: Database ID of the IngestionJob.
            user_id: User who initiated the job.

        Returns:
            Job info with task_id and status.
        """
        try:
            task = ingest_image_task.delay(job_id, user_id)
            logger.info(f"Started image ingestion job {job_id} with task {task.id}")
            return {
                "task_id": task.id,
                "job_id": job_id,
                "status": "started",
                "message": "Image ingestion job started successfully",
            }
        except Exception as e:
            logger.error(f"Failed to start image ingestion job {job_id}: {e}")
            return {
                "task_id": None,
                "job_id": job_id,
                "status": "failed",
                "message": f"Failed to start job: {str(e)}",
            }


# Create global instance
background_job_service = BackgroundJobService()
