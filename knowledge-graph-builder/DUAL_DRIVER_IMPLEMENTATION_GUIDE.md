# Neo4j Dual Driver Implementation Guide

**Quick Reference for Development Sessions**

---

## 🎯 **What Was Implemented**

We successfully implemented a **dual driver architecture** to solve Neo4j connection management issues in our distributed FastAPI + Celery system. This eliminates the `@lru_cache` problems and provides proper driver isolation.

---

## 🏗️ **Architecture Overview**

### **Two Driver Types**

1. **AsyncDriver** - For FastAPI endpoints (web requests)
2. **Driver** (sync) - For neo4j_graphrag components (VectorRetriever, Neo4jWriter, etc.)

### **Service Separation**

```
FastAPI Service (knowledge-graph-builder)
├── Uses: neo4j_client.async_driver
├── Pool: 100 connections (shared, long-lived)
└── Purpose: HTTP endpoints, real-time queries

Celery Workers (knowledge-graph-worker)
├── Uses: WorkerNeo4jManager (task-scoped)
├── Pool: 1 connection per task (isolated)
└── Purpose: Background processing, document ingestion
```

---

## 🚀 **How to Use**

### **FastAPI Endpoints**

```python
# ✅ For async operations in FastAPI
from app.core.neo4j_client import neo4j_client

async def some_fastapi_endpoint():
    # Uses async driver automatically
    result = await neo4j_client.execute_query("RETURN 1 as test")
    return result
```

### **GraphRAG Components in FastAPI**

```python
# ✅ For GraphRAG components in FastAPI
from app.core.dependencies import get_neo4j_sync_driver
from app.components.multi_tenant_components import create_multi_tenant_vector_retriever

def get_retriever(graph_id: str = Depends(get_graph_id)):
    # Ensure sync driver is connected
    neo4j_client.connect_sync()

    # Use sync driver for GraphRAG
    retriever = create_multi_tenant_vector_retriever(
        driver=neo4j_client.sync_driver,
        embedder=openai_embedder,
        graph_id=graph_id
    )
    return retriever
```

### **Celery Worker Tasks**

```python
# ✅ For background worker tasks
from app.services.background_jobs import WorkerNeo4jManager

async def some_worker_task():
    # Use task-scoped connections
    async with WorkerNeo4jManager() as neo4j:
        sync_driver = neo4j.get_sync_driver()  # For GraphRAG
        async_driver = neo4j.get_async_driver()  # For async ops

        # Use drivers for task operations
        # Automatic cleanup when exiting context

# OR for sync-only tasks:
def sync_worker_task():
    with WorkerNeo4jManager() as neo4j:
        driver = neo4j.get_sync_driver()
        # Use driver
        # Automatic cleanup
```

---

## 📁 **Key Files Modified**

### **1. Core Neo4j Client**

```
app/core/neo4j_client.py
├── ✅ Dual driver support (async + sync)
├── ✅ Thread-safe connection management
├── ✅ Comprehensive health checks
└── ✅ Backward compatibility methods
```

### **2. FastAPI Dependencies**

```
app/core/dependencies.py
├── ✅ Removed @lru_cache from drivers
├── ✅ Updated health checks
├── ✅ Fixed OpenAI LLM configuration
└── ✅ Maintained backward compatibility
```

### **3. Worker Connection Manager**

```
app/services/background_jobs.py
├── ✅ WorkerNeo4jManager class added
├── ✅ Task-scoped connection patterns
├── ✅ Both sync and async context managers
└── ✅ Automatic cleanup on task completion
```

### **4. Pipeline Service**

```
app/services/pipeline_service.py
├── ✅ Updated to use sync driver for GraphRAG
├── ✅ Proper driver initialization
├── ✅ Enhanced documentation
└── ✅ Dual driver architecture support
```

### **5. Component Updates**

```
app/services/retriever_service.py
app/components/multi_tenant_components.py
├── ✅ Fixed deprecated neo4j_client.driver references
├── ✅ Updated to use neo4j_client.sync_driver
└── ✅ Updated documentation examples
```

---

## 🧪 **Testing the Implementation**

### **Quick Health Check**

```bash
# Test in FastAPI container
docker exec knowledge-graph-builder python -c "
import asyncio
from app.core.neo4j_client import neo4j_client

async def test():
    await neo4j_client.connect_async()
    neo4j_client.connect_sync()
    health = await neo4j_client.health_check()
    print(f'Status: {health[\"status\"]}')
    print(f'Async: {health[\"async_driver\"]}')
    print(f'Sync: {health[\"sync_driver\"]}')

asyncio.run(test())
"
```

### **Test Worker Manager**

```bash
# Test in Worker container
docker exec knowledge-graph-worker python -c "
import asyncio
from app.services.background_jobs import WorkerNeo4jManager

async def test():
    async with WorkerNeo4jManager() as neo4j:
        sync_driver = neo4j.get_sync_driver()
        with sync_driver.session() as session:
            result = session.run('RETURN \"success\" as test')
            print(f'Result: {result.single()[\"test\"]}')

asyncio.run(test())
"
```

---

## ⚠️ **Common Issues and Solutions**

### **Issue: "Neo4jClient object has no attribute 'driver'"**

**Problem**: Old code trying to access deprecated `neo4j_client.driver`

**Solution**: Update to use specific drivers:

```python
# ❌ OLD
driver = neo4j_client.driver

# ✅ NEW
async_driver = neo4j_client.async_driver  # For async ops
sync_driver = neo4j_client.sync_driver    # For GraphRAG
```

### **Issue: GraphRAG components not working**

**Problem**: GraphRAG needs sync drivers, not async

**Solution**: Ensure sync driver is connected:

```python
# ✅ Before using GraphRAG components
neo4j_client.connect_sync()
retriever = VectorRetriever(driver=neo4j_client.sync_driver, ...)
```

### **Issue: Worker connection conflicts**

**Problem**: Workers interfering with FastAPI connections

**Solution**: Use WorkerNeo4jManager for isolation:

```python
# ✅ In worker tasks
async with WorkerNeo4jManager() as neo4j:
    driver = neo4j.get_sync_driver()
    # Use driver - automatically cleaned up
```

---

## 🔧 **Development Guidelines**

### **When to Use Which Driver**

| Use Case                | Driver Type                 | Example                                     |
| ----------------------- | --------------------------- | ------------------------------------------- |
| FastAPI async endpoints | `neo4j_client.async_driver` | `await neo4j_client.execute_query()`        |
| GraphRAG components     | `neo4j_client.sync_driver`  | `VectorRetriever(driver=sync_driver)`       |
| Worker tasks            | `WorkerNeo4jManager`        | `async with WorkerNeo4jManager() as neo4j:` |
| Health checks           | Either                      | `await neo4j_client.health_check()`         |

### **Best Practices**

1. **Always use context managers in workers** to ensure cleanup
2. **Connect sync driver before using GraphRAG** components
3. **Don't cache drivers with @lru_cache** - cache models instead
4. **Use appropriate driver for the context** (FastAPI vs Worker)
5. **Test both drivers** when making changes

---

## 📊 **Connection Pool Configuration**

```python
# Current Settings
FASTAPI_ASYNC_POOL = 100      # Shared long-lived connections
FASTAPI_SYNC_POOL = 50        # For GraphRAG in FastAPI
WORKER_POOL_PER_TASK = 1      # Isolated per task
MAX_CONCURRENT_WORKERS = 10   # Total worker concurrency
```

**Total Neo4j Connections**: ~150-200 (well under Neo4j's 1000 limit)

---

## 🚀 **Deployment Status**

### **✅ What's Working**

- ✅ FastAPI service starts without errors
- ✅ Celery workers start without errors
- ✅ Both async and sync drivers connect successfully
- ✅ Health checks pass for both drivers
- ✅ WorkerNeo4jManager provides proper isolation
- ✅ GraphRAG components work with sync drivers
- ✅ API endpoints respond correctly
- ✅ Background tasks can access Neo4j

### **✅ Verified Components**

- ✅ Neo4j dual driver architecture
- ✅ FastAPI dependencies (no @lru_cache issues)
- ✅ Worker connection isolation
- ✅ Pipeline service integration
- ✅ Retriever service compatibility
- ✅ Multi-tenant components

---

## 🆘 **Emergency Troubleshooting**

### **Service Won't Start**

1. **Check for missing drivers**: Look for `neo4j_client.driver` references
2. **Verify connection strings**: Ensure NEO4J_URI is correct
3. **Check authentication**: Verify NEO4J_USERNAME/PASSWORD
4. **Review logs**: `docker-compose logs knowledge-graph-builder`

### **Worker Tasks Failing**

1. **Use WorkerNeo4jManager**: Don't use global neo4j_client in workers
2. **Check context managers**: Ensure proper cleanup with `async with`
3. **Verify driver types**: GraphRAG needs sync drivers
4. **Check task isolation**: Each task should get fresh connections

### **Performance Issues**

1. **Monitor connection pools**: Check if hitting limits
2. **Review concurrency**: Adjust worker concurrency if needed
3. **Check for leaks**: Ensure all connections are properly closed
4. **Monitor Neo4j**: Check database-side connection usage

---

## 📞 **Quick Help Commands**

```bash
# Check service status
docker-compose ps

# View FastAPI logs
docker-compose logs -f knowledge-graph-builder

# View Worker logs
docker-compose logs -f knowledge-graph-worker

# Test Neo4j connection
curl -s http://localhost:8003/docs

# Restart services
docker-compose restart knowledge-graph-builder knowledge-graph-worker

# Full rebuild if needed
docker-compose up --build -d knowledge-graph-builder knowledge-graph-worker
```

---

## 📝 **Next Steps for Development**

1. **New features should use appropriate drivers** for their context
2. **Test both FastAPI and Worker contexts** when making changes
3. **Follow the WorkerNeo4jManager pattern** for any new worker tasks
4. **Update documentation** when adding new Neo4j operations
5. **Monitor connection usage** in production

---

**🎉 The dual driver architecture is fully implemented and tested!**

For detailed technical information, see `NEO4J_CONNECTION_MANAGEMENT_STRATEGY.md`
