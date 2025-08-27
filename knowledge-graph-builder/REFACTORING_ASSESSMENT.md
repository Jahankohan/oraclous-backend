# REFACTORING_ASSESSMENT.md

## Current State Analysis

| Service | Current Methods | Violation Issues | Actions Required |
|---------|----------------|------------------|------------------|
| **search_service** | 5 methods | ✅ Mostly correct scope | Minor cleanup |
| **embedding_service** | 4 methods | ✅ Correct scope | No changes needed |
| **vector_service** | 6 methods | ❌ Contains `create_text_chunks()` - GRAPH OPERATION | MOVE graph operations to graph_service |
| **enhanced_graph_service** | 5 methods | ❌ Imports missing `graph_service` | CREATE graph_service, fix imports |
| **graphrag_service** | 10 methods | ✅ Correct scope | No changes needed |
| **graph_service** | N/A | ❌ MISSING ENTIRELY | CREATE with delegated operations |

## Critical Issues Found

### 1. **MISSING graph_service.py**
- enhanced_graph_service imports `from app.services.graph_service import graph_service`
- File doesn't exist → Import error
- **ACTION**: Create graph_service.py with proper Neo4j operations

### 2. **vector_service.py Violations**
```python
# WRONG - Graph operations in vector service:
async def create_text_chunks(self, graph_id, text_chunks) -> int:
    # Creates DocumentChunk nodes in Neo4j - BELONGS IN GRAPH_SERVICE
    
async def add_entity_embedding(self, entity_id, embedding, graph_id):
    # Updates entity nodes - BELONGS IN GRAPH_SERVICE
```

### 3. **Missing Core Methods**
- vector_service missing: `store_node_embedding()`, `store_chunk_embedding()`
- graph_service missing: `store_graph_documents()`, `create_chunk_node()`, `create_chunk_entity_relationship()`

## Action Plan

### Phase 1: Create Missing graph_service.py
- Create core graph service with proper Neo4j operations
- Add delegated methods: `store_graph_documents()`, `create_chunk_node()`, `create_chunk_entity_relationship()`

### Phase 2: Fix vector_service.py Violations  
- MOVE `create_text_chunks()` → graph_service
- MOVE `add_entity_embedding()` → graph_service  
- ADD proper `store_node_embedding()`, `store_chunk_embedding()` methods
- KEEP only vector indexing operations

### Phase 3: Fix enhanced_graph_service.py
- Fix import error (graph_service will exist)
- Ensure proper delegation patterns

### Phase 4: Update All Callers
- Find files calling moved methods
- Update to use correct service references

## Service Responsibilities (TARGET STATE)

| Service | ONLY Responsible For | Key Methods |
|---------|---------------------|-------------|
| **search_service** | Search coordination, ranking | `similarity_search_entities()`, `hybrid_search()` |
| **embedding_service** | Text→vector conversion | `embed_text()`, `embed_documents()` |  
| **vector_service** | Vector storage, indexing | `store_node_embedding()`, `create_vector_indexes()` |
| **graph_service** | Direct Neo4j operations | `store_graph_documents()`, `create_chunk_node()` |
| **enhanced_graph_service** | Enrichment coordination | `store_complete_graph()`, `enrich_and_store_entities()` |
| **graphrag_service** | End-to-end RAG pipeline | `graph_augmented_retrieval()` |

## Files That Need Updates

### vector_service.py Callers:
```bash
app/services/background_jobs.py:                chunks_stored = await vector_service.create_text_chunks(
app/services/enhanced_graph_service.py:                    await vector_service.store_node_embedding(
app/services/enhanced_graph_service.py:                await vector_service.store_chunk_embedding(
```

### Expected Changes:
- `background_jobs.py`: Change `vector_service.create_text_chunks()` → `graph_service.create_text_chunks()`
- All `vector_service.store_*_embedding()` calls: Keep (but implement proper methods)
