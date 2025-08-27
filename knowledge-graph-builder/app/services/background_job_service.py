"""
Background Job Service - Clean Interface for API Endpoints
Provides a professional layer between REST endpoints and Celery background jobs
"""

from typing import Dict, Any, Optional
from uuid import UUID
from datetime import datetime

from app.services.background_jobs import (
    process_ingestion_job,
    process_embedding_generation_job,
    optimize_all_graphs,
    cleanup_orphaned_data,
    reindex_graph_search,
    generate_graph_summary
)
from app.core.logging import get_logger

logger = get_logger(__name__)

class BackgroundJobService:
    """
    Professional service layer for managing background jobs
    Abstracts away Celery implementation details from API endpoints
    """
    
    @staticmethod
    def start_ingestion_job(job_id: str, user_id: str) -> Dict[str, Any]:
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
                "message": "Ingestion job started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start ingestion job {job_id}: {e}")
            return {
                "task_id": None,
                "job_id": job_id,
                "status": "failed",
                "message": f"Failed to start job: {str(e)}"
            }
    
    @staticmethod
    def start_embedding_generation(graph_id: UUID, user_id: str) -> Dict[str, Any]:
        """
        Start embedding generation for a graph
        
        Args:
            graph_id: UUID of the knowledge graph
            user_id: User who initiated the job
            
        Returns:
            Job info with task_id and status
        """
        try:
            task = process_embedding_generation_job.delay(str(graph_id), user_id)
            
            logger.info(f"Started embedding generation for graph {graph_id} with task {task.id}")
            
            return {
                "task_id": task.id,
                "graph_id": str(graph_id),
                "status": "started",
                "message": "Embedding generation started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start embedding generation for graph {graph_id}: {e}")
            return {
                "task_id": None,
                "graph_id": str(graph_id),
                "status": "failed",
                "message": f"Failed to start job: {str(e)}"
            }
    
    @staticmethod
    def start_graph_optimization() -> Dict[str, Any]:
        """
        Start system-wide graph optimization (singleton task)
        
        Returns:
            Job info with task_id and status
        """
        try:
            task = optimize_all_graphs.delay()
            
            logger.info(f"Started graph optimization with task {task.id}")
            
            return {
                "task_id": task.id,
                "status": "started",
                "message": "Graph optimization started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start graph optimization: {e}")
            return {
                "task_id": None,
                "status": "failed",
                "message": f"Failed to start optimization: {str(e)}"
            }
    
    @staticmethod
    def start_data_cleanup() -> Dict[str, Any]:
        """
        Start system-wide data cleanup (singleton task)
        
        Returns:
            Job info with task_id and status
        """
        try:
            task = cleanup_orphaned_data.delay()
            
            logger.info(f"Started data cleanup with task {task.id}")
            
            return {
                "task_id": task.id,
                "status": "started",
                "message": "Data cleanup started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start data cleanup: {e}")
            return {
                "task_id": None,
                "status": "failed",
                "message": f"Failed to start cleanup: {str(e)}"
            }
    
    @staticmethod
    def start_search_reindex(graph_id: UUID) -> Dict[str, Any]:
        """
        Start search reindexing for a specific graph
        
        Args:
            graph_id: UUID of the knowledge graph
            
        Returns:
            Job info with task_id and status
        """
        try:
            task = reindex_graph_search.delay(str(graph_id))
            
            logger.info(f"Started search reindexing for graph {graph_id} with task {task.id}")
            
            return {
                "task_id": task.id,
                "graph_id": str(graph_id),
                "status": "started",
                "message": "Search reindexing started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start search reindexing for graph {graph_id}: {e}")
            return {
                "task_id": None,
                "graph_id": str(graph_id),
                "status": "failed",
                "message": f"Failed to start reindexing: {str(e)}"
            }
    
    @staticmethod
    def start_graph_summary(graph_id: UUID) -> Dict[str, Any]:
        """
        Start graph summary generation
        
        Args:
            graph_id: UUID of the knowledge graph
            
        Returns:
            Job info with task_id and status
        """
        try:
            task = generate_graph_summary.delay(str(graph_id))
            
            logger.info(f"Started graph summary generation for graph {graph_id} with task {task.id}")
            
            return {
                "task_id": task.id,
                "graph_id": str(graph_id),
                "status": "started",
                "message": "Graph summary generation started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start graph summary for graph {graph_id}: {e}")
            return {
                "task_id": None,
                "graph_id": str(graph_id),
                "status": "failed",
                "message": f"Failed to start summary generation: {str(e)}"
            }
    
    @staticmethod
    def start_similarity_relationships(graph_id: UUID) -> Dict[str, Any]:
        """
        Start similarity relationship creation for a graph
        
        Args:
            graph_id: UUID of the knowledge graph
            
        Returns:
            Job info with task_id and status
        """
        try:
            # We'll add this task to background_jobs.py
            from app.services.background_jobs import create_similarity_relationships_job
            task = create_similarity_relationships_job.delay(str(graph_id))
            
            logger.info(f"Started similarity relationships creation for graph {graph_id} with task {task.id}")
            
            return {
                "task_id": task.id,
                "graph_id": str(graph_id),
                "status": "started",
                "message": "Similarity relationships creation started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start similarity relationships for graph {graph_id}: {e}")
            return {
                "task_id": None,
                "graph_id": str(graph_id),
                "status": "failed",
                "message": f"Failed to start similarity relationships: {str(e)}"
            }
    
    @staticmethod
    def start_community_detection(graph_id: UUID) -> Dict[str, Any]:
        """
        Start community detection for a graph
        
        Args:
            graph_id: UUID of the knowledge graph
            
        Returns:
            Job info with task_id and status
        """
        try:
            # We'll add this task to background_jobs.py
            from app.services.background_jobs import detect_communities_job
            task = detect_communities_job.delay(str(graph_id))
            
            logger.info(f"Started community detection for graph {graph_id} with task {task.id}")
            
            return {
                "task_id": task.id,
                "graph_id": str(graph_id),
                "status": "started",
                "message": "Community detection started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start community detection for graph {graph_id}: {e}")
            return {
                "task_id": None,
                "graph_id": str(graph_id),
                "status": "failed",
                "message": f"Failed to start community detection: {str(e)}"
            }
    
    @staticmethod
    def start_community_embeddings(graph_id: UUID, user_id: str) -> Dict[str, Any]:
        """
        Start community embeddings generation
        
        Args:
            graph_id: UUID of the knowledge graph
            user_id: User who initiated the job
            
        Returns:
            Job info with task_id and status
        """
        try:
            # We'll add this task to background_jobs.py
            from app.services.background_jobs import update_community_embeddings_job
            task = update_community_embeddings_job.delay(str(graph_id), user_id)
            
            logger.info(f"Started community embeddings for graph {graph_id} with task {task.id}")
            
            return {
                "task_id": task.id,
                "graph_id": str(graph_id),
                "status": "started",
                "message": "Community embeddings generation started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start community embeddings for graph {graph_id}: {e}")
            return {
                "task_id": None,
                "graph_id": str(graph_id),
                "status": "failed",
                "message": f"Failed to start community embeddings: {str(e)}"
            }
    
    @staticmethod
    def refresh_all_communities(user_id: str) -> Dict[str, Any]:
        """
        Refresh communities for all user graphs
        
        Args:
            user_id: User whose communities to refresh
            
        Returns:
            Job info with task_id and status
        """
        try:
            # We'll add this task to background_jobs.py
            from app.services.background_jobs import refresh_all_communities_job
            task = refresh_all_communities_job.delay(user_id)
            
            logger.info(f"Started community refresh for user {user_id} with task {task.id}")
            
            return {
                "task_id": task.id,
                "user_id": user_id,
                "status": "started",
                "message": "Community refresh started successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to start community refresh for user {user_id}: {e}")
            return {
                "task_id": None,
                "user_id": user_id,
                "status": "failed",
                "message": f"Failed to start community refresh: {str(e)}"
            }
    
    @staticmethod
    def get_task_status(task_id: str) -> Dict[str, Any]:
        """
        Get the status of a background task
        
        Args:
            task_id: Celery task ID
            
        Returns:
            Task status information
        """
        try:
            from app.services.background_jobs import celery_app
            
            # Get task result
            result = celery_app.AsyncResult(task_id)
            
            return {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if result.ready() else None,
                "info": result.info if hasattr(result, 'info') else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return {
                "task_id": task_id,
                "status": "unknown",
                "error": str(e)
            }

# Create global instance
background_job_service = BackgroundJobService()
