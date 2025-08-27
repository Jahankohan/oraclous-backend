"""
Task Status Endpoints - Monitor Background Jobs
Professional endpoints for checking job status and results
"""

from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
from app.services.background_job_service import background_job_service
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

@router.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the status of a background task
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Task status and result information
    """
    try:
        task_info = background_job_service.get_task_status(task_id)
        
        if "error" in task_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Task not found or error getting status: {task_info['error']}"
            )
        
        return {
            "task_id": task_id,
            "status": task_info.get("status", "unknown"),
            "result": task_info.get("result"),
            "info": task_info.get("info"),
            "ready": task_info.get("status") in ["SUCCESS", "FAILURE", "REVOKED"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status for {task_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve task status"
        )

@router.post("/admin/jobs/optimize")
async def trigger_system_optimization() -> Dict[str, Any]:
    """
    Trigger system-wide graph optimization
    Restricted to admin users (add proper auth later)
    """
    try:
        job_result = background_job_service.start_graph_optimization()
        
        return {
            "message": job_result["message"],
            "task_id": job_result.get("task_id"),
            "status": job_result["status"],
            "check_status_url": f"/tasks/{job_result.get('task_id')}/status" if job_result.get("task_id") else None
        }
        
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start system optimization"
        )

@router.post("/admin/jobs/cleanup")
async def trigger_data_cleanup() -> Dict[str, Any]:
    """
    Trigger system-wide data cleanup
    Restricted to admin users (add proper auth later)
    """
    try:
        job_result = background_job_service.start_data_cleanup()
        
        return {
            "message": job_result["message"],
            "task_id": job_result.get("task_id"),
            "status": job_result["status"],
            "check_status_url": f"/tasks/{job_result.get('task_id')}/status" if job_result.get("task_id") else None
        }
        
    except Exception as e:
        logger.error(f"Failed to start cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start data cleanup"
        )

@router.post("/graphs/{graph_id}/jobs/reindex")
async def trigger_search_reindex(graph_id: str) -> Dict[str, Any]:
    """
    Trigger search reindexing for a specific graph
    """
    try:
        from uuid import UUID
        graph_uuid = UUID(graph_id)
        
        job_result = background_job_service.start_search_reindex(graph_uuid)
        
        return {
            "message": job_result["message"],
            "task_id": job_result.get("task_id"),
            "graph_id": graph_id,
            "status": job_result["status"],
            "check_status_url": f"/tasks/{job_result.get('task_id')}/status" if job_result.get("task_id") else None
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid graph ID format"
        )
    except Exception as e:
        logger.error(f"Failed to start reindexing for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start search reindexing"
        )

@router.post("/graphs/{graph_id}/jobs/summary")
async def trigger_graph_summary(graph_id: str) -> Dict[str, Any]:
    """
    Trigger graph summary generation
    """
    try:
        from uuid import UUID
        graph_uuid = UUID(graph_id)
        
        job_result = background_job_service.start_graph_summary(graph_uuid)
        
        return {
            "message": job_result["message"],
            "task_id": job_result.get("task_id"),
            "graph_id": graph_id,
            "status": job_result["status"],
            "check_status_url": f"/tasks/{job_result.get('task_id')}/status" if job_result.get("task_id") else None
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid graph ID format"
        )
    except Exception as e:
        logger.error(f"Failed to start summary generation for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start graph summary generation"
        )

@router.post("/graphs/{graph_id}/jobs/export")
async def trigger_graph_export(graph_id: str, export_format: str = "json") -> Dict[str, Any]:
    """
    Trigger graph data export
    """
    try:
        from uuid import UUID
        graph_uuid = UUID(graph_id)
        
        # Validate format
        valid_formats = ["json", "csv", "graphml", "cypher"]
        if export_format not in valid_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid export format. Must be one of: {valid_formats}"
            )
        
        job_result = background_job_service.start_graph_export(graph_uuid, export_format)
        
        return {
            "message": job_result["message"],
            "task_id": job_result.get("task_id"),
            "graph_id": graph_id,
            "format": export_format,
            "status": job_result["status"],
            "check_status_url": f"/tasks/{job_result.get('task_id')}/status" if job_result.get("task_id") else None
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid graph ID format"
        )
    except Exception as e:
        logger.error(f"Failed to start export for graph {graph_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start graph export"
        )
