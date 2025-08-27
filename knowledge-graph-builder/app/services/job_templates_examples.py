"""
Background Job Templates and Examples
=====================================

This file provides ready-to-use templates for common background job patterns.
Each template demonstrates proper usage of the unified task execution framework.

USAGE:
------
1. Copy the relevant template to your use case
2. Modify the async implementation function (_function_name_async)
3. Register the task in your main background_jobs.py file
4. Update the task name in SINGLETON_TASKS if it should run alone

PATTERNS DEMONSTRATED:
---------------------
- Concurrent tasks (multiple instances allowed)
- Singleton tasks (only one instance at a time)
- Progress reporting with task.update_state()
- Proper error handling and cleanup
- Database session management
- Neo4j connection handling

AVAILABLE TEMPLATES:
-------------------
- Community detection and management
- Graph analytics generation
- System maintenance tasks
- Notification processing
- User activity tracking
"""

from app.services.background_jobs import celery_app
from app.services.task_executor import AsyncTaskExecutor, TaskConcurrencyManager
from app.core.logging import get_logger

logger = get_logger(__name__)

# ====== COMMUNITY DETECTION JOBS ======

@celery_app.task(bind=True)
def detect_communities_job(self, graph_id: str):
    """Detect communities in a graph - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_detect_communities_async, self, graph_id)

@celery_app.task(bind=True)
def update_community_persistence_job(self, graph_id: str):
    """Update community persistence scores - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_update_community_persistence_async, self, graph_id)

@celery_app.task(bind=True)
def cleanup_stale_communities_job(self):
    """Clean up stale communities - singleton task"""
    if not TaskConcurrencyManager.should_allow_task('cleanup_stale_communities_job', self.request.id):
        return {
            'status': 'skipped',
            'message': 'Community cleanup already running'
        }
    
    return AsyncTaskExecutor.run_async_task(_cleanup_stale_communities_async, self)

# ====== ANALYTICS JOBS ======

@celery_app.task(bind=True)
def generate_graph_analytics_job(self, graph_id: str):
    """Generate analytics for a graph - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_generate_graph_analytics_async, self, graph_id)

@celery_app.task(bind=True)
def update_user_activity_stats_job(self):
    """Update user activity statistics - singleton task"""
    if not TaskConcurrencyManager.should_allow_task('update_user_activity_stats_job', self.request.id):
        return {
            'status': 'skipped',
            'message': 'User stats update already running'
        }
    
    return AsyncTaskExecutor.run_async_task(_update_user_activity_stats_async, self)

# ====== MAINTENANCE JOBS ======

@celery_app.task(bind=True)
def database_maintenance_job(self):
    """Perform database maintenance - singleton task"""
    if not TaskConcurrencyManager.should_allow_task('database_maintenance_job', self.request.id):
        return {
            'status': 'skipped',
            'message': 'Database maintenance already running'
        }
    
    return AsyncTaskExecutor.run_async_task(_database_maintenance_async, self)

@celery_app.task(bind=True)
def backup_critical_data_job(self):
    """Backup critical data - singleton task"""
    if not TaskConcurrencyManager.should_allow_task('backup_critical_data_job', self.request.id):
        return {
            'status': 'skipped',
            'message': 'Backup already running'
        }
    
    return AsyncTaskExecutor.run_async_task(_backup_critical_data_async, self)

# ====== NOTIFICATION JOBS ======

@celery_app.task(bind=True)
def send_user_notification_job(self, user_id: str, notification_type: str, data: dict):
    """Send notification to user - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_send_user_notification_async, self, user_id, notification_type, data)

@celery_app.task(bind=True)
def process_bulk_notifications_job(self, notification_batch: list):
    """Process bulk notifications - allows concurrency"""
    return AsyncTaskExecutor.run_async_task(_process_bulk_notifications_async, self, notification_batch)

# ====== ASYNC IMPLEMENTATIONS ======

async def _detect_communities_async(task, graph_id: str):
    """Detect communities in a graph"""
    try:
        logger.info(f"Starting community detection for graph {graph_id}")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Initializing community detection"})
        
        # Initialize Neo4j connection
        from app.core.neo4j_client import neo4j_client
        await neo4j_client.connect()
        
        # Run community detection algorithm
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 50, "status": "Running community detection"})
        
        # Implement your community detection logic here
        communities_found = 0
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            'status': 'success',
            'communities_found': communities_found,
            'message': f'Detected {communities_found} communities in graph {graph_id}'
        }
        
    except Exception as e:
        logger.error(f"Community detection failed for graph {graph_id}: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'communities_found': 0
        }
    finally:
        try:
            await neo4j_client.close()
        except Exception as e:
            logger.warning(f"Neo4j cleanup warning: {e}")

async def _update_community_persistence_async(task, graph_id: str):
    """Update community persistence scores"""
    try:
        logger.info(f"Updating community persistence for graph {graph_id}")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Calculating persistence scores"})
        
        # Implement your persistence calculation logic here
        updated_communities = 0
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            'status': 'success',
            'updated_communities': updated_communities,
            'message': f'Updated persistence scores for {updated_communities} communities'
        }
        
    except Exception as e:
        logger.error(f"Community persistence update failed for graph {graph_id}: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'updated_communities': 0
        }

async def _cleanup_stale_communities_async(task):
    """Clean up stale communities across all graphs"""
    try:
        logger.info("Starting stale community cleanup")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Identifying stale communities"})
        
        # Implement your stale community cleanup logic here
        cleaned_communities = 0
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            'status': 'success',
            'cleaned_communities': cleaned_communities,
            'message': f'Cleaned up {cleaned_communities} stale communities'
        }
        
    except Exception as e:
        logger.error(f"Stale community cleanup failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'cleaned_communities': 0
        }

async def _generate_graph_analytics_async(task, graph_id: str):
    """Generate analytics for a graph"""
    try:
        logger.info(f"Generating analytics for graph {graph_id}")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Collecting graph data"})
        
        # Initialize connections
        from app.core.neo4j_client import neo4j_client
        await neo4j_client.connect()
        
        # Generate analytics (implement your logic here)
        analytics_data = {
            "node_count": 0,
            "relationship_count": 0,
            "density": 0.0,
            "clustering_coefficient": 0.0
        }
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            'status': 'success',
            'analytics': analytics_data,
            'message': f'Generated analytics for graph {graph_id}'
        }
        
    except Exception as e:
        logger.error(f"Analytics generation failed for graph {graph_id}: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'analytics': {}
        }
    finally:
        try:
            await neo4j_client.close()
        except Exception as e:
            logger.warning(f"Neo4j cleanup warning: {e}")

async def _update_user_activity_stats_async(task):
    """Update user activity statistics"""
    try:
        logger.info("Updating user activity statistics")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Calculating user stats"})
        
        # Implement your user activity stats logic here
        users_updated = 0
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            'status': 'success',
            'users_updated': users_updated,
            'message': f'Updated activity stats for {users_updated} users'
        }
        
    except Exception as e:
        logger.error(f"User activity stats update failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'users_updated': 0
        }

async def _database_maintenance_async(task):
    """Perform database maintenance"""
    try:
        logger.info("Starting database maintenance")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Running maintenance tasks"})
        
        # Implement your database maintenance logic here
        maintenance_tasks = []
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            'status': 'success',
            'tasks_completed': len(maintenance_tasks),
            'message': f'Completed {len(maintenance_tasks)} maintenance tasks'
        }
        
    except Exception as e:
        logger.error(f"Database maintenance failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'tasks_completed': 0
        }

async def _backup_critical_data_async(task):
    """Backup critical data"""
    try:
        logger.info("Starting critical data backup")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Preparing backup"})
        
        # Implement your backup logic here
        backed_up_items = 0
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            'status': 'success',
            'backed_up_items': backed_up_items,
            'message': f'Backed up {backed_up_items} critical items'
        }
        
    except Exception as e:
        logger.error(f"Critical data backup failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'backed_up_items': 0
        }

async def _send_user_notification_async(task, user_id: str, notification_type: str, data: dict):
    """Send notification to a specific user"""
    try:
        logger.info(f"Sending {notification_type} notification to user {user_id}")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 50, "status": "Sending notification"})
        
        # Implement your notification sending logic here
        notification_sent = True
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            'status': 'success',
            'notification_sent': notification_sent,
            'message': f'Sent {notification_type} notification to user {user_id}'
        }
        
    except Exception as e:
        logger.error(f"Notification sending failed for user {user_id}: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'notification_sent': False
        }

async def _process_bulk_notifications_async(task, notification_batch: list):
    """Process a batch of notifications"""
    try:
        logger.info(f"Processing {len(notification_batch)} bulk notifications")
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 10, "status": "Processing bulk notifications"})
        
        # Implement your bulk notification processing logic here
        processed_count = 0
        
        for i, notification in enumerate(notification_batch):
            # Process each notification
            processed_count += 1
            
            if task and i % 10 == 0:  # Update progress every 10 notifications
                progress = int(10 + (i / len(notification_batch)) * 80)
                task.update_state(
                    state="PROGRESS", 
                    meta={
                        "progress": progress, 
                        "status": f"Processed {processed_count}/{len(notification_batch)} notifications"
                    }
                )
        
        if task:
            task.update_state(state="PROGRESS", meta={"progress": 100, "status": "Completed"})
        
        return {
            'status': 'success',
            'processed_count': processed_count,
            'message': f'Processed {processed_count} bulk notifications'
        }
        
    except Exception as e:
        logger.error(f"Bulk notification processing failed: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'processed_count': processed_count
        }
