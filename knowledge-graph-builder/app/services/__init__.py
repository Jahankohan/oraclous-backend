# app/services/__init__.py
"""
Services module for the Knowledge Graph Builder application.

CLEAN IMPORTS: Only services that exist after refactoring
"""

# Core services that remain after refactor
from .analytics_service import analytics_service
from .llm_service import llm_service

# NEW: Refactored services with Neo4j GraphRAG foundation
from .pipeline_service import pipeline_service, get_pipeline_service
from .retriever_service import retrieval_service, get_retrieval_service

# Background job services
from .background_job_service import background_job_service
from .task_executor import AsyncTaskExecutor, TaskConcurrencyManager

__all__ = [
    # Core services (kept)
    'analytics_service',
    'llm_service',
    
    # New Neo4j GraphRAG services
    'pipeline_service',
    'get_pipeline_service',
    'retrieval_service',
    'get_retrieval_service',
    
    # Background job infrastructure
    'background_job_service',
    'AsyncTaskExecutor',
    'TaskConcurrencyManager'
]