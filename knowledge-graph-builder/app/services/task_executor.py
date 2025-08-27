"""
Universal Task Executor for Celery Background Jobs
Handles async function execution with proper event loop isolation
"""

import asyncio
from typing import Any, Callable, Dict, Optional
from app.core.logging import get_logger

logger = get_logger(__name__)

class AsyncTaskExecutor:
    """Universal handler for async tasks in Celery workers"""
    
    @staticmethod
    def run_async_task(async_func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute async function in a clean event loop context
        Safe for all Celery background tasks
        
        Args:
            async_func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Dict containing the result or error information
        """
        loop = None
        try:
            # Always create a fresh event loop in Celery workers
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            logger.info(f"Created new event loop for {async_func.__name__}")
            
            # Run the async function
            result = loop.run_until_complete(async_func(*args, **kwargs))
            return result
            
        except Exception as e:
            logger.error(f"Task execution failed in {async_func.__name__}: {e}")
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
        finally:
            if loop:
                try:
                    # Cancel all pending tasks
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    
                    # Wait for cancellation to complete
                    if pending:
                        loop.run_until_complete(
                            asyncio.gather(*pending, return_exceptions=True)
                        )
                        
                    logger.debug(f"Cleaned up {len(pending)} pending tasks")
                        
                except Exception as cleanup_error:
                    logger.warning(f"Task cleanup warning: {cleanup_error}")
                finally:
                    loop.close()
                    logger.debug("Event loop closed successfully")


class TaskConcurrencyManager:
    """Manages task concurrency and deduplication"""
    
    # Tasks that should not run concurrently (singleton tasks)
    SINGLETON_TASKS = {
        'optimize_all_graphs',
        'cleanup_orphaned_data', 
        'rebuild_search_index',
        'system_maintenance',
        'database_vacuum'
    }
    
    @classmethod
    def should_allow_task(cls, task_name: str, current_task_id: str) -> bool:
        """
        Check if task should be allowed to run based on concurrency rules
        
        Args:
            task_name: Name of the task to check
            current_task_id: ID of the current task
            
        Returns:
            True if task should run, False if it should be skipped
        """
        
        # Extract base task name (remove module path)
        base_task_name = task_name.split('.')[-1]
        
        if base_task_name not in cls.SINGLETON_TASKS:
            return True
        
        # Check for active tasks of the same type
        try:
            from app.services.background_jobs import celery_app
            inspect = celery_app.control.inspect()
            active_tasks = inspect.active()
            
            if active_tasks:
                for worker, tasks in active_tasks.items():
                    for task in tasks:
                        task_base_name = task['name'].split('.')[-1]
                        if (task_base_name == base_task_name and 
                            task['id'] != current_task_id):
                            logger.info(f"Skipping {task_name} - already running as {task['id']}")
                            return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Could not check task concurrency: {e}")
            # If we can't check, allow the task to run
            return True
    
    @classmethod
    def get_task_info(cls, task_name: str) -> Dict[str, Any]:
        """Get information about task concurrency settings"""
        base_name = task_name.split('.')[-1]
        
        return {
            "task_name": base_name,
            "allows_concurrency": base_name not in cls.SINGLETON_TASKS,
            "is_singleton": base_name in cls.SINGLETON_TASKS
        }


# Decorator for easy task wrapping
def async_celery_task(singleton: bool = False):
    """
    Decorator to wrap async functions for Celery execution
    
    Args:
        singleton: If True, only one instance of this task can run at a time
    """
    def decorator(async_func):
        def wrapper(celery_task_self, *args, **kwargs):
            # Add task to singleton list if specified
            if singleton:
                task_name = celery_task_self.name.split('.')[-1]
                TaskConcurrencyManager.SINGLETON_TASKS.add(task_name)
            
            # Check concurrency if this is a bound task
            if hasattr(celery_task_self, 'request'):
                if not TaskConcurrencyManager.should_allow_task(
                    celery_task_self.name, 
                    celery_task_self.request.id
                ):
                    return {
                        'status': 'skipped',
                        'message': f'Task {celery_task_self.name} already running',
                        'task_info': TaskConcurrencyManager.get_task_info(celery_task_self.name)
                    }
            
            # Execute the async function
            return AsyncTaskExecutor.run_async_task(async_func, celery_task_self, *args, **kwargs)
        
        return wrapper
    return decorator
