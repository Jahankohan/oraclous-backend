# Background Jobs Architecture

## 🎯 Overview

This directory contains a completely refactored background job system that eliminates event loop conflicts and provides consistent execution patterns for all async tasks.

## 📁 File Structure

```
app/services/
├── background_jobs.py          # Main production background jobs
├── task_executor.py            # Universal async task executor
├── task_database.py           # Database session management
├── sync_ingestion_processor.py # Core extraction functions
├── job_templates_library.py   # Templates for new jobs
└── README_BACKGROUND_JOBS.md  # This file
```

## 🚀 Quick Start

### Adding a New Background Job

1. **Concurrent Task** (multiple instances allowed):
```python
@celery_app.task(bind=True)
def my_new_job(self, param1: str):
    return AsyncTaskExecutor.run_async_task(_my_new_job_async, self, param1)

async def _my_new_job_async(task, param1: str):
    try:
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 50, "status": "Processing"})
        
        # Your logic here
        
        return {"status": "success", "result": "data"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
```

2. **Singleton Task** (only one instance):
```python
@celery_app.task(bind=True)
def my_singleton_job(self):
    if not TaskConcurrencyManager.should_allow_task('my_singleton_job', self.request.id):
        return {'status': 'skipped', 'message': 'Already running'}
    
    return AsyncTaskExecutor.run_async_task(_my_singleton_job_async, self)
```

## 🏗️ Architecture Components

### 1. Universal Task Executor (`task_executor.py`)
- **Purpose**: Provides clean event loop isolation
- **Features**: 
  - Fresh event loop per task
  - Automatic cleanup
  - Concurrency management
  - Error handling

### 2. Task Database Manager (`task_database.py`)
- **Purpose**: Isolated database sessions
- **Features**:
  - Async/sync session factories
  - Connection pooling
  - Auto-cleanup

### 3. Job Templates Library (`job_templates_library.py`)
- **Purpose**: Ready-to-use templates
- **Contains**: 9 different job patterns
- **Categories**: Community, Analytics, Maintenance, Notifications

## 🔄 Execution Flow

All background jobs now follow this consistent pattern:

```
1. Celery Task Definition
   ↓
2. Concurrency Check (if singleton)
   ↓
3. AsyncTaskExecutor.run_async_task()
   ↓
4. Fresh Event Loop Creation
   ↓
5. Async Function Execution
   ↓
6. Progress Reporting
   ↓
7. Resource Cleanup
   ↓
8. Return Results
```

## 📊 Current Production Jobs

| Job Name | Type | Purpose |
|----------|------|---------|
| `process_ingestion_job` | Concurrent | Data ingestion |
| `process_embedding_generation_job` | Concurrent | Generate embeddings |
| `optimize_all_graphs` | Singleton | System optimization |
| `cleanup_orphaned_data` | Singleton | Data cleanup |
| `reindex_graph_search` | Concurrent | Search reindexing |
| `generate_graph_summary` | Concurrent | Graph summarization |
| `export_graph_data` | Concurrent | Data export |

## 🛡️ Key Benefits

### ✅ Event Loop Isolation
- Each task gets fresh event loop
- No conflicts between concurrent tasks
- Proper cleanup prevents leaks

### ✅ Resource Management  
- Database connections properly managed
- Neo4j connections isolated per task
- Automatic cleanup in finally blocks

### ✅ Concurrency Control
- Singleton tasks prevent conflicts
- Concurrent tasks run in parallel
- Configurable concurrency rules

### ✅ Progress Reporting
- Real-time progress updates
- Detailed status messages
- Easy monitoring

## 🔧 Configuration

### Celery Settings:
```python
celery_app.conf.update(
    task_time_limit=30 * 60,        # 30 minutes
    task_soft_time_limit=25 * 60,   # 25 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)
```

### Singleton Tasks:
```python
SINGLETON_TASKS = {
    'optimize_all_graphs',
    'cleanup_orphaned_data', 
    'database_maintenance',
    # Add your singleton tasks here
}
```

## 🚦 Usage Examples

### Starting Jobs from API:
```python
@router.post("/process")
async def start_processing(data: dict):
    task = process_ingestion_job.delay(job_id, user_id)
    return {"task_id": task.id}
```

### Checking Status:
```python
@router.get("/tasks/{task_id}")
async def get_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    return {"status": task.status, "result": task.result}
```

## 🐛 Debugging

### Enable Debug Logging:
```python
# In task_executor.py
logger.setLevel(logging.DEBUG)
```

### Monitor Active Tasks:
```python
from app.services.background_jobs import celery_app
inspect = celery_app.control.inspect()
active = inspect.active()
```

## 📝 Migration Notes

### From Old Pattern:
```python
# OLD - Manual event loop
@celery_app.task(bind=True)
def old_task(self, param):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(async_func(param))
    finally:
        loop.close()
```

### To New Pattern:
```python
# NEW - Universal executor
@celery_app.task(bind=True)
def new_task(self, param):
    return AsyncTaskExecutor.run_async_task(async_func, self, param)
```

## 🎯 Best Practices

1. **Always use AsyncTaskExecutor** for async functions
2. **Add progress reporting** for long-running tasks
3. **Use TaskDatabaseManager** for database operations
4. **Implement proper cleanup** in finally blocks
5. **Choose correct concurrency pattern** (singleton vs concurrent)
6. **Follow naming conventions**: `job_name` + `_async` for implementation

## 🔮 Future Enhancements

- Task scheduling and cron-like functionality
- Task dependency management  
- Enhanced monitoring and metrics
- Dead letter queue handling
- Task result storage optimization

---

This architecture provides a robust, scalable foundation for background job processing that eliminates event loop conflicts while maintaining high performance and reliability.
