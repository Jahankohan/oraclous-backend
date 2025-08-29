# Knowledge Graph Service Refactoring Instructions for Claude AI

## 🎯 **Project Overview & Goals**

You are helping refactor a large knowledge graph builder service codebase. The goal is to:

1. **Use Neo4j GraphRAG as the foundation** - Leverage their production-ready pipeline and component architecture
2. **Add advanced multi-tenant capabilities** - Preserve existing multi-tenant graph isolation (`graph_id` based)
3. **Integrate Microsoft GraphRAG innovations** - Add community detection, DRIFT search, and global reasoning
4. **Maintain FastAPI best practices** - Clean dependency injection, background jobs, and maintainable code structure
5. **Keep code minimal and focused** - Avoid overcomplication, prefer composition over inheritance

## 🚨 **Critical Guidelines - READ CAREFULLY**

### **Code Style Requirements:**
- ✅ **MINIMAL CODE**: Write only essential code, avoid verbose implementations
- ✅ **SINGLE RESPONSIBILITY**: Each class/function has ONE clear purpose
- ✅ **COMPOSITION OVER INHERITANCE**: Prefer wrapping/delegating to extending classes
- ✅ **NO OVERCOMPLICATED ABSTRACTIONS**: Use simple, direct implementations
- ✅ **FOCUSED IMPLEMENTATIONS**: Small, targeted functions rather than monolithic classes

### **Architecture Requirements:**
- ✅ **Neo4j GraphRAG Foundation**: Use their components, pipelines, and interfaces as base
- ✅ **FastAPI Native Patterns**: Dependency injection, background tasks, proper error handling
- ✅ **Multi-tenant First**: All operations must respect `graph_id` isolation
- ✅ **Component-based Extensions**: Add features as Neo4j-compatible components

### **What NOT to Do:**
- ❌ **Don't create complex inheritance hierarchies**
- ❌ **Don't write verbose, over-engineered abstractions**
- ❌ **Don't duplicate Neo4j GraphRAG functionality**
- ❌ **Don't create unnecessary middleware layers**
- ❌ **Don't write monolithic classes with many responsibilities**

## 📁 **Target Project Structure**

```
app/
├── core/
│   ├── config.py                     # Configuration management
│   ├── dependencies.py               # FastAPI dependency injection
│   └── database.py                   # Neo4j connection management
├── models/
│   ├── graph.py                      # Pydantic models for graph operations
│   ├── tenant.py                     # Multi-tenant models
│   └── pipeline.py                   # Pipeline configuration models
├── components/                       # Neo4j GraphRAG compatible components
│   ├── multi_tenant_retriever.py     # Multi-tenant wrapper component
│   ├── context_engineering.py        # Context optimization component
│   ├── microsoft_community.py        # Microsoft community detection
│   └── drift_search.py              # Microsoft DRIFT search implementation
├── services/
│   ├── graph_service.py              # Core graph operations (thin wrapper)
│   ├── pipeline_service.py           # Pipeline orchestration
│   └── tenant_service.py             # Multi-tenant management
├── api/
│   └── v1/
│       ├── graphs.py                 # Graph management endpoints
│       ├── pipelines.py              # Pipeline execution endpoints
│       └── search.py                 # Search and query endpoints
├── tasks/
│   └── background_jobs.py            # Celery/background task definitions
└── main.py                           # FastAPI application
```

## 🔧 **Implementation Instructions**

### **Step 1: Core Infrastructure (Minimal)**

When implementing core infrastructure, create only essential components:

```python
# Example of GOOD minimal implementation
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorRetriever

class MultiTenantRetriever(VectorRetriever):
    """Simple wrapper that adds graph_id filtering to any Neo4j retriever"""
    
    def __init__(self, base_retriever, graph_id: str):
        self.base_retriever = base_retriever
        self.graph_id = graph_id
    
    def get_search_results(self, query_vector=None, query_text=None, **kwargs):
        # Add graph_id filter to search
        kwargs['filters'] = {**(kwargs.get('filters', {})), 'graph_id': self.graph_id}
        return self.base_retriever.get_search_results(query_vector, query_text, **kwargs)

# Example of BAD overcomplicated implementation - DON'T DO THIS
class AbstractMultiTenantRetrieverFactory(ABC):
    @abstractmethod
    def create_retriever(self, tenant_context: TenantContext) -> BaseRetriever:
        pass

class MultiTenantRetrieverBuilder(AbstractMultiTenantRetrieverFactory):
    def __init__(self, config: RetrieverConfiguration):
        self.config = config
        self.validator = TenantConfigurationValidator()
        self.metrics_collector = MetricsCollectionService()
    # ... 50+ lines of unnecessary complexity
```

### **Step 2: Service Layer (FastAPI Best Practices)**

Keep services thin and focused on orchestration:

```python
# GOOD - Simple, focused service
from fastapi import Depends
from neo4j_graphrag.generation import GraphRAG

class GraphService:
    def __init__(self, neo4j_driver=Depends(get_neo4j_driver)):
        self.driver = neo4j_driver
    
    async def search_graph(self, query: str, graph_id: str) -> dict:
        # Simple delegation to Neo4j GraphRAG with multi-tenant wrapper
        retriever = MultiTenantRetriever(
            VectorRetriever(self.driver, f"embeddings_{graph_id}"),
            graph_id
        )
        rag = GraphRAG(retriever=retriever, llm=get_llm())
        return rag.search(query)

# BAD - Over-engineered service - DON'T DO THIS
class GraphManagementOrchestrationService:
    def __init__(self, 
                 retriever_factory: RetrieverFactory,
                 context_manager: ContextManager,
                 security_validator: SecurityValidator,
                 audit_logger: AuditLogger):
        # ... complex initialization with many dependencies
```

### **Step 3: API Endpoints (Clean FastAPI)**

Keep endpoints simple and delegate to services:

```python
# GOOD - Clean, simple endpoints
from fastapi import APIRouter, Depends, BackgroundTasks

router = APIRouter()

@router.post("/graphs/{graph_id}/search")
async def search_graph(
    graph_id: str,
    query: SearchQuery,
    graph_service: GraphService = Depends(),
    background_tasks: BackgroundTasks = None
):
    # Log search in background if needed
    if background_tasks:
        background_tasks.add_task(log_search, graph_id, query.text)
    
    return await graph_service.search_graph(query.text, graph_id)

# BAD - Overcomplicated endpoint - DON'T DO THIS
@router.post("/graphs/{graph_id}/search")
async def search_graph_with_advanced_processing(
    graph_id: str,
    query: SearchQuery,
    request_context: RequestContext = Depends(get_request_context),
    auth_handler: AuthenticationHandler = Depends(),
    rate_limiter: RateLimiter = Depends(),
    # ... many dependencies
):
    # ... 50+ lines of complex processing logic
```

### **Step 4: Component Integration Pattern**

When adding Microsoft GraphRAG features, use simple composition:

```python
# GOOD - Simple component that extends Neo4j GraphRAG
from neo4j_graphrag.retrievers.base import Retriever

class DRIFTRetriever(Retriever):
    """Microsoft DRIFT search as Neo4j component"""
    
    def __init__(self, base_retriever, community_service, llm):
        self.base_retriever = base_retriever
        self.community_service = community_service
        self.llm = llm
    
    def get_search_results(self, query_vector=None, query_text=None, **kwargs):
        # 1. Community search
        community_results = self.community_service.search_communities(query_text)
        
        # 2. Generate follow-up questions
        follow_ups = self.llm.generate_follow_ups(query_text, community_results)
        
        # 3. Local search on follow-ups
        local_results = []
        for question in follow_ups:
            results = self.base_retriever.get_search_results(query_text=question)
            local_results.extend(results.items)
        
        return RawSearchResult(items=local_results)
```

## 📋 **Specific Refactoring Tasks**

### **Task 1: Replace Current GraphRAG with Neo4j Foundation**
1. Install `neo4j-graphrag` package
2. Replace current `GraphRAGService` with thin wrapper around Neo4j `GraphRAG`
3. Keep existing multi-tenant logic as wrapper component
4. Remove redundant graph construction code - use Neo4j's pipeline

### **Task 2: Implement Multi-Tenant Components**
1. Create `MultiTenantRetriever` wrapper for any Neo4j retriever
2. Add `graph_id` filtering to all Cypher queries automatically
3. Create `MultiTenantKnowledgeGraph` that injects `graph_id` into all nodes/relationships
4. Keep existing tenant isolation logic, just wrap Neo4j components

### **Task 3: Add Microsoft Features as Components**
1. `HierarchicalCommunityComponent` - Extends Neo4j's community detection with Leiden
2. `DRIFTSearchRetriever` - Implements Microsoft's DRIFT methodology as retriever
3. `GlobalReasoningRetriever` - Adds corpus-wide reasoning capabilities
4. All should implement Neo4j GraphRAG interfaces for seamless integration

### **Task 4: FastAPI Integration**
1. Create dependency injection for Neo4j driver, LLM, embeddings
2. Background tasks for heavy operations (community detection, large imports)
3. Simple service layer that orchestrates Neo4j GraphRAG components
4. Clean API endpoints with proper error handling and validation

### **Task 5: Preserve Existing Advanced Features**
1. Keep current `AnalyticsService` functionality as Neo4j-compatible components
2. Preserve background job architecture using Celery
3. Keep existing multi-tenant database patterns
4. Maintain current API structure but simplify implementations

## 🎯 **Implementation Priorities**

### **Phase 1: Foundation (Week 1)**
- [ ] Replace core GraphRAG with Neo4j GraphRAG
- [ ] Create multi-tenant wrapper components
- [ ] Update dependency injection for Neo4j components
- [ ] Ensure existing functionality works

### **Phase 2: Microsoft Features (Week 2)**
- [ ] Add DRIFT search as Neo4j retriever component
- [ ] Implement hierarchical community detection component
- [ ] Add global reasoning capabilities
- [ ] Test integration with multi-tenant system

### **Phase 3: API & Services (Week 3)**
- [ ] Refactor service layer to use Neo4j components
- [ ] Update API endpoints for new component architecture
- [ ] Migrate background jobs to use new pipeline system
- [ ] Add comprehensive error handling

### **Phase 4: Advanced Features (Week 4)**
- [ ] Add context engineering components
- [ ] Implement adaptive query strategies
- [ ] Add performance monitoring and metrics
- [ ] Comprehensive testing and documentation

## ✅ **Quality Checklist**

Before implementing any component, ensure it meets these criteria:

### **Code Quality:**
- [ ] **Single purpose**: Component does one thing well
- [ ] **Minimal complexity**: Under 50 lines for most functions
- [ ] **Clear naming**: Function/class names explain purpose immediately
- [ ] **No unnecessary abstractions**: Direct, simple implementations

### **Architecture Quality:**
- [ ] **Neo4j GraphRAG compatible**: Uses their interfaces correctly
- [ ] **Multi-tenant safe**: All operations respect graph_id isolation
- [ ] **FastAPI native**: Proper dependency injection and async patterns
- [ ] **Composable**: Can be combined with other components easily

### **Integration Quality:**
- [ ] **Drop-in replacement**: Existing code changes minimally
- [ ] **Backward compatible**: Current functionality preserved
- [ ] **Performance maintained**: No significant performance regression
- [ ] **Error handling**: Proper exception handling and logging

## 🚀 **Success Criteria**

The refactoring is successful when:

1. **Codebase is smaller**: Overall lines of code reduced by 30%+
2. **Functionality is preserved**: All existing features work as before
3. **Neo4j GraphRAG integrated**: Using their components as foundation
4. **Microsoft features added**: DRIFT, community detection, global reasoning
5. **Multi-tenancy maintained**: Full graph isolation preserved
6. **FastAPI best practices**: Clean DI, background jobs, proper error handling
7. **Maintainable code**: Simple, focused components that are easy to understand

## 📝 **Code Review Guidelines**

When reviewing implementations, ask:

1. **Is this the simplest solution that works?**
2. **Does it follow Neo4j GraphRAG patterns?**
3. **Is multi-tenant isolation guaranteed?**
4. **Are FastAPI best practices followed?**
5. **Can this be understood in under 2 minutes?**

If any answer is "No", simplify the implementation.

---

**Remember: The goal is a production-ready, maintainable system built on proven foundations (Neo4j GraphRAG) with minimal custom code. Prefer composition, delegation, and simple wrappers over complex custom implementations.**