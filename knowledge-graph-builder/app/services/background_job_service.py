"""
Background Job Service - Clean Interface for API Endpoints

Provides a professional layer between REST endpoints and Celery background jobs.
Includes graceful degradation: when the Celery broker (Redis) is unreachable, jobs
are written to the PostgreSQL `job_queue` fallback table with status 'pending'.
"""

from typing import Any

from app.core.errors import KGBError
from app.core.logging import get_logger
from app.services.background_jobs import (
    code_ingest_task,
    ingest_image_task,
    process_ingestion_job,
)

logger = get_logger(__name__)

# Broker-down exceptions — covers both Celery and Kombu connection failures.
try:
    from celery.exceptions import OperationalError as _CeleryOperationalError
    from kombu.exceptions import OperationalError as _KombuOperationalError

    _BROKER_DOWN_EXCEPTIONS: tuple = (_CeleryOperationalError, _KombuOperationalError)
except ImportError:  # pragma: no cover
    _BROKER_DOWN_EXCEPTIONS = (OSError,)


async def _write_fallback_job(
    task_name: str, args: list, kwargs: dict, error: str
) -> None:
    """
    Persist a job to the PostgreSQL fallback queue when the Celery broker is down.

    Uses the async session factory from app.core.database so we don't create a
    separate engine.  Import is deferred to avoid circular imports.
    """
    try:
        from app.core.database import async_session_maker
        from app.models.graph import FallbackJobQueue

        async with async_session_maker() as session:
            fallback = FallbackJobQueue(
                task_name=task_name,
                args=args,
                kwargs=kwargs,
                status="pending",
                error_message=error,
            )
            session.add(fallback)
            await session.commit()
            logger.info(
                f"Celery broker down — job written to fallback queue: "
                f"task={task_name} args={args}"
            )
    except Exception as db_exc:
        logger.error(
            f"Failed to write fallback job to PostgreSQL (task={task_name}): {db_exc}"
        )


class BackgroundJobService:
    """
    Professional service layer for managing background jobs.

    Abstracts away Celery implementation details from API endpoints.
    Provides graceful degradation when the broker is unavailable.
    """

    @staticmethod
    async def start_ingestion_job(job_id: str, user_id: str) -> dict[str, Any]:
        """
        Start a data ingestion background job.

        Args:
            job_id: Database ID of the ingestion job
            user_id: User who initiated the job

        Returns:
            Job info with task_id and status.  When the broker is unavailable
            returns {"status": "queued_to_fallback"} with HTTP-friendly 202 semantics.
        """
        task_name = "app.services.background_jobs.process_ingestion_job"
        try:
            task = process_ingestion_job.delay(job_id, user_id)

            logger.info(f"Started ingestion job {job_id} with task {task.id}")

            return {
                "task_id": task.id,
                "job_id": job_id,
                "status": "started",
                "message": "Ingestion job started successfully",
            }

        except _BROKER_DOWN_EXCEPTIONS as e:
            _code, _msg = KGBError.CELERY_UNAVAILABLE
            logger.error(
                f"Celery broker unavailable [{_code}] — falling back to PostgreSQL queue "
                f"for job {job_id}: {e}"
            )
            await _write_fallback_job(
                task_name=task_name,
                args=[job_id, user_id],
                kwargs={},
                error=str(e),
            )
            return {
                "task_id": None,
                "job_id": job_id,
                "status": "queued_to_fallback",
                "message": (
                    "Background job service temporarily unavailable — "
                    "job queued to fallback store"
                ),
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
    async def start_code_ingest_job(job_id: str, user_id: str) -> dict[str, Any]:
        """
        Start a code repository ingestion background job.

        Args:
            job_id: Database ID of the IngestionJob (source_type='code').
            user_id: User who initiated the job.

        Returns:
            Job info with task_id and status.
        """
        task_name = "app.services.background_jobs.code_ingest_task"
        try:
            task = code_ingest_task.delay(job_id, user_id)
            logger.info(f"Started code ingest job {job_id} with task {task.id}")
            return {
                "task_id": task.id,
                "job_id": job_id,
                "status": "started",
                "message": "Code ingestion job started successfully",
            }
        except _BROKER_DOWN_EXCEPTIONS as e:
            _code, _msg = KGBError.CELERY_UNAVAILABLE
            logger.error(
                f"Celery broker unavailable [{_code}] — falling back to PostgreSQL queue "
                f"for code job {job_id}: {e}"
            )
            await _write_fallback_job(
                task_name=task_name,
                args=[job_id, user_id],
                kwargs={},
                error=str(e),
            )
            return {
                "task_id": None,
                "job_id": job_id,
                "status": "queued_to_fallback",
                "message": (
                    "Background job service temporarily unavailable — "
                    "job queued to fallback store"
                ),
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
    async def start_image_ingestion_job(job_id: str, user_id: str) -> dict[str, Any]:
        """
        Start an image ingestion background job (vision extraction path).

        Args:
            job_id: Database ID of the IngestionJob.
            user_id: User who initiated the job.

        Returns:
            Job info with task_id and status.
        """
        task_name = "app.services.background_jobs.ingest_image_task"
        try:
            task = ingest_image_task.delay(job_id, user_id)
            logger.info(f"Started image ingestion job {job_id} with task {task.id}")
            return {
                "task_id": task.id,
                "job_id": job_id,
                "status": "started",
                "message": "Image ingestion job started successfully",
            }
        except _BROKER_DOWN_EXCEPTIONS as e:
            _code, _msg = KGBError.CELERY_UNAVAILABLE
            logger.error(
                f"Celery broker unavailable [{_code}] — falling back to PostgreSQL queue "
                f"for image job {job_id}: {e}"
            )
            await _write_fallback_job(
                task_name=task_name,
                args=[job_id, user_id],
                kwargs={},
                error=str(e),
            )
            return {
                "task_id": None,
                "job_id": job_id,
                "status": "queued_to_fallback",
                "message": (
                    "Background job service temporarily unavailable — "
                    "job queued to fallback store"
                ),
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
