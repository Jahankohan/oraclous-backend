# Celery Tasks Integration Plan

## 📋 **Complete API Endpoint Integration**

### **✅ Now Connected: All Celery Tasks Have Endpoints**

| Celery Task | API Endpoint | Purpose | Usage |
|-------------|-------------|---------|-------|
| `_detect_communities_and_cleanup` | `POST /graphs/{graph_id}/optimize/communities` | Sync community creation | Small graphs |
| `create_persistent_communities_task` | `POST /graphs/{graph_id}/communities/create-async` | Async community creation | Large graphs |
| `update_community_embeddings_task` | `POST /graphs/{graph_id}/communities/embeddings` | Generate community embeddings | Any graph |
| `refresh_all_communities_task` | `POST /graphs/optimize/refresh-all` | Refresh all user communities | User maintenance |
| `optimize_all_graphs` | `POST /admin/optimize/all-graphs` | System-wide optimization | Admin/scheduled |

### **📝 API Usage Examples**

#### 1. **Create Communities (Sync - Small Graphs)**
```bash
curl -X POST "http://localhost:8000/api/v1/graphs/{graph_id}/optimize/communities" \
  -H "Authorization: Bearer {token}"
```

**Response**:
```json
{
  "status": "success",
  "communities_detected": 15,
  "message": "Detected 15 communities and cleaned up graph"
}
```

#### 2. **Create Communities (Async - Large Graphs)**
```bash
curl -X POST "http://localhost:8000/api/v1/graphs/{graph_id}/communities/create-async" \
  -H "Authorization: Bearer {token}"
```

**Response**:
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "processing",
  "message": "Community creation started in background",
  "check_status_url": "/graphs/{graph_id}/tasks/{task_id}"
}
```

#### 3. **Generate Community Embeddings**
```bash
curl -X POST "http://localhost:8000/api/v1/graphs/{graph_id}/communities/embeddings" \
  -H "Authorization: Bearer {token}"
```

#### 4. **Refresh All User Communities**
```bash
curl -X POST "http://localhost:8000/api/v1/graphs/optimize/refresh-all" \
  -H "Authorization: Bearer {token}"
```

#### 5. **Check Task Status**
```bash
curl -X GET "http://localhost:8000/api/v1/graphs/{graph_id}/tasks/{task_id}" \
  -H "Authorization: Bearer {token}"
```

**Response**:
```json
{
  "task_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "status": "processing",
  "progress": 75,
  "current": "Creating community embeddings",
  "result": {
    "communities_created": 12,
    "relationships_created": 156,
    "algorithm_used": "louvain"
  }
}
```

### **🔄 Integration Workflow**

#### **For Small Graphs (<1000 nodes)**:
1. Use **sync endpoint**: `/graphs/{graph_id}/optimize/communities`
2. Get immediate response with results
3. Communities are created and ready

#### **For Large Graphs (>1000 nodes)**:
1. Use **async endpoint**: `/graphs/{graph_id}/communities/create-async`  
2. Get task ID for monitoring
3. Poll status with: `/graphs/{graph_id}/tasks/{task_id}`
4. Check completion status

#### **For Production Maintenance**:
1. **User-level refresh**: `/graphs/optimize/refresh-all`
2. **System-wide optimization**: `/admin/optimize/all-graphs` (admin only)

### **🕐 Scheduled Tasks Integration**

#### **Option 1: Cron-style Scheduling**
```python
# In settings or scheduler
CELERY_BEAT_SCHEDULE = {
    'optimize-all-graphs': {
        'task': 'app.services.background_jobs.optimize_all_graphs',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
}
```

#### **Option 2: API-triggered Scheduling**
```bash
# Weekly optimization via admin endpoint
curl -X POST "http://localhost:8000/api/v1/admin/optimize/all-graphs" \
  -H "Authorization: Bearer {admin_token}"
```

### **🏗️ Implementation Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   API Endpoints  │    │  Celery Tasks   │
│                 │    │                  │    │                 │
│ User clicks     │───▶│ POST /communities│───▶│ create_persistent│
│ "Optimize"      │    │ /create-async    │    │ _communities_task│
│                 │    │                  │    │                 │
│ Polls status    │◀───│ GET /tasks/{id}  │◀───│ Task progress   │
│                 │    │                  │    │ updates         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### **🛡️ Security & Authorization**

#### **User Endpoints** (Require Authentication):
- `/graphs/{graph_id}/optimize/communities`
- `/graphs/{graph_id}/communities/create-async`
- `/graphs/{graph_id}/communities/embeddings`
- `/graphs/optimize/refresh-all`

#### **Admin Endpoints** (Require Admin Role):
- `/admin/optimize/all-graphs`

#### **Task Status** (User can only check their own tasks):
- `/graphs/{graph_id}/tasks/{task_id}` - Verifies graph ownership

### **📊 Monitoring & Observability**

#### **Task Status Types**:
- `pending` - Task queued, waiting to start
- `processing` - Task running, includes progress percentage
- `completed` - Task finished successfully
- `failed` - Task failed with error details

#### **Metrics to Track**:
- Community creation time per graph size
- Task failure rates
- Background queue length
- System optimization frequency

### **🚀 Deployment Readiness**

#### **Docker Environment Requirements**:
1. **Celery Worker**: `celery -A app.core.celery worker --loglevel=info`
2. **Celery Beat** (for scheduled tasks): `celery -A app.core.celery beat --loglevel=info`
3. **Redis/RabbitMQ**: Message broker for task queue
4. **API Server**: FastAPI with all endpoints enabled

#### **Test Commands for Docker**:
```bash
# Test community creation
curl -X POST "http://localhost:8000/api/v1/graphs/{graph_id}/optimize/communities"

# Test async task
curl -X POST "http://localhost:8000/api/v1/graphs/{graph_id}/communities/create-async"

# Check task status
curl -X GET "http://localhost:8000/api/v1/graphs/{graph_id}/tasks/{task_id}"
```

### **✅ Status: Ready for Docker Testing**

All Celery tasks now have proper API endpoints and can be tested in the Docker environment. The integration provides both synchronous and asynchronous options for different use cases.
