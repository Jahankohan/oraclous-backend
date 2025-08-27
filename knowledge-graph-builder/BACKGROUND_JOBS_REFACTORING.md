# Background Jobs Refactoring - Implementation Guide

## Overview

This refactoring implements a comprehensive solution for handling async background jobs in Celery workers, eliminating event loop conflicts while maintaining proper resource isolation and concurrency control.

## Key Components

### 1. Universal Task Executor (`task_executor.py`)

**Purpose**: Provides clean event loop isolation for all async tasks in Celery workers.

**Key Features**:
- ✅ Creates fresh event loop for each task
- ✅ Proper resource cleanup with finally blocks
- ✅ Task concurrency management (singleton vs concurrent)
- ✅ Automatic error handling and reporting
- ✅ Decorator support for easy task wrapping

**Usage**:
```python
# Simple usage
return AsyncTaskExecutor.run_async_task(my_async_function, arg1, arg2)

# With decorator
@async_celery_task(singleton=True)
async def my_async_task(task, arg1, arg2):
    # Your async implementation
    pass
```

### 2. Task Database Manager (`task_database.py`)

**Purpose**: Provides isolated database sessions for background tasks.

**Features**:
- ✅ Async and sync session factories
- ✅ Proper connection pooling for tasks
- ✅ Automatic cleanup and resource management
- ✅ Connection recycling and health checks

**Usage**:
```python
# Async usage
async with TaskDatabaseManager.get_async_session() as session:
    result = await session.execute(query)
    await session.commit()

# Sync usage
with TaskDatabaseManager.get_sync_session() as session:
    result = session.execute(query)
    session.commit()
```

### 3. Refactored Background Jobs (`background_jobs.py`)

**Purpose**: Clean implementation of all background tasks using the universal executor.

**Task Categories**:

#### Concurrent Tasks (Multiple instances allowed):
- `process_ingestion_job` - Data ingestion
- `process_embedding_generation_job` - Embedding generation
- `reindex_graph_search` - Search reindexing
- `generate_graph_summary` - Graph summarization
- `export_graph_data` - Data export

#### Singleton Tasks (Only one instance at a time):
- `optimize_all_graphs` - System optimization
- `cleanup_orphaned_data` - Data cleanup
- Database maintenance tasks

### 4. Sync Ingestion Processor (`sync_ingestion_processor.py`)

**Purpose**: Handles data ingestion with proper event loop management.

**Improvements**:
- ✅ Uses universal task executor
- ✅ Better error handling and progress reporting
- ✅ Proper resource cleanup
- ✅ Integration with task database manager

## Task Patterns

### Pattern 1: Concurrent Task (Allows multiple instances)

```python
@celery_app.task(bind=True)
def my_concurrent_task(self, param1: str, param2: str):
    """Task that can run multiple instances concurrently"""
    return AsyncTaskExecutor.run_async_task(
        _my_concurrent_task_async, 
        self, param1, param2
    )

async def _my_concurrent_task_async(task, param1: str, param2: str):
    """Async implementation with progress reporting"""
    try:
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Starting"})
        
        # Your async logic here
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {"status": "success", "result": "data"}
        
    except Exception as e:
        logger.error(f"Task failed: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        # Cleanup resources
        pass
```

### Pattern 2: Singleton Task (Only one instance allowed)

```python
@celery_app.task(bind=True)
def my_singleton_task(self):
    """Task that should not run concurrently"""
    if not TaskConcurrencyManager.should_allow_task('my_singleton_task', self.request.id):
        return {
            'status': 'skipped',
            'message': 'Task already running'
        }
    
    return AsyncTaskExecutor.run_async_task(_my_singleton_task_async, self)

async def _my_singleton_task_async(task):
    """Async implementation for singleton task"""
    # Implementation similar to concurrent task
    pass
```

### Pattern 3: Database-Heavy Task

```python
async def _database_heavy_task_async(task, param1: str):
    """Task that requires database operations"""
    try:
        # Use task database manager for clean sessions
        async with TaskDatabaseManager.get_async_session() as session:
            result = await session.execute(select(Model).where(Model.field == param1))
            data = result.scalars().all()
            
            # Process data
            for item in data:
                # Update item
                await session.commit()
        
        return {"status": "success", "processed": len(data)}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

## Configuration

### Celery Settings (in `background_jobs.py`):

```python
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes
    task_soft_time_limit=25 * 60,  # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)
```

### Task Concurrency Settings:

```python
# In TaskConcurrencyManager.SINGLETON_TASKS
SINGLETON_TASKS = {
    'optimize_all_graphs',
    'cleanup_orphaned_data', 
    'rebuild_search_index',
    'system_maintenance',
    'database_vacuum'
}
```

## Benefits of This Approach

### 1. Event Loop Isolation ✅
- Each task gets a fresh event loop
- No conflicts between concurrent tasks
- Proper cleanup prevents resource leaks

### 2. Resource Management ✅
- Database connections are properly managed
- Neo4j connections are isolated per task
- Automatic cleanup in finally blocks

### 3. Concurrency Control ✅
- Singleton tasks prevent resource conflicts
- Concurrent tasks can run in parallel
- Configurable concurrency rules

### 4. Error Handling ✅
- Comprehensive error reporting
- Graceful degradation on failures
- Proper cleanup even on errors

### 5. Progress Reporting ✅
- Real-time progress updates
- Detailed status messages
- Easy monitoring and debugging

### 6. Maintainability ✅
- Clean separation of concerns
- Consistent patterns across all tasks
- Easy to add new background jobs

## Usage Examples

### Starting Tasks from API:

```python
@router.post("/process-graph")
async def process_graph(graph_id: str):
    # Start ingestion job
    ingestion_task = process_ingestion_job.delay(job_id, user_id)
    
    # Start embedding generation
    embedding_task = process_embedding_generation_job.delay(graph_id, user_id)
    
    return {
        "ingestion_task_id": ingestion_task.id,
        "embedding_task_id": embedding_task.id
    }

@router.post("/admin/optimize")
async def optimize_system():
    # Start optimization (will skip if already running)
    task = optimize_all_graphs.delay()
    return {"task_id": task.id}
```

### Checking Task Status:

```python
@router.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    from app.services.background_jobs import celery_app
    
    task = celery_app.AsyncResult(task_id)
    
    return {
        "task_id": task_id,
        "status": task.status,
        "result": task.result,
        "info": task.info
    }
```

## Migration Guide

To migrate existing background jobs:

1. **Replace manual event loop creation** with `AsyncTaskExecutor.run_async_task()`
2. **Update database sessions** to use `TaskDatabaseManager`
3. **Add concurrency control** for singleton tasks
4. **Implement proper cleanup** in finally blocks
5. **Add progress reporting** for long-running tasks

### Before:
```python
@celery_app.task(bind=True)
def old_task(self, param):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(async_func(param))
    finally:
        loop.close()
```

### After:
```python
@celery_app.task(bind=True)
def new_task(self, param):
    return AsyncTaskExecutor.run_async_task(async_func, self, param)
```

## Monitoring and Debugging

### Enable Detailed Logging:
```python
# In task_executor.py, set more verbose logging
logger.setLevel(logging.DEBUG)
```

### Monitor Task Performance:
```python
# Check active tasks
from app.services.background_jobs import celery_app
inspect = celery_app.control.inspect()
active_tasks = inspect.active()
```

### Check Resource Usage:
```bash
# Monitor database connections
SELECT count(*) FROM pg_stat_activity WHERE application_name LIKE '%celery%';

# Monitor memory usage
ps aux | grep celery
```

## Conclusion

This refactoring provides a robust, scalable solution for background job execution that:

- ✅ Eliminates event loop conflicts
- ✅ Provides proper resource isolation
- ✅ Enables flexible concurrency control
- ✅ Ensures reliable error handling
- ✅ Supports easy monitoring and debugging

The new architecture is production-ready and can handle complex background processing workflows while maintaining system stability and performance.
