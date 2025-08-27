# MIGRATION_GUIDE.md

## Services Refactoring Migration Guide

### **For External API Consumers**

If you're consuming the knowledge graph builder APIs, **NO CHANGES** are required. All external API endpoints remain the same.

### **For Internal Service Consumers**

If you have code that directly imports and uses the services, update these imports:

#### **Method Moves**

| Old Call | New Call | Reason |
|----------|----------|---------|
| `vector_service.create_text_chunks()` | `graph_service.create_text_chunks()` | Graph operation moved to graph service |
| `vector_service.add_entity_embedding()` | `graph_service.update_entity_embedding()` | Graph update moved to graph service |

#### **New Service Available**

```python
# NEW: Import graph_service for direct Neo4j operations
from app.services.graph_service import graph_service

# Available methods:
await graph_service.store_graph_documents(graph_id, documents)
await graph_service.create_chunk_node(chunk_id, text, graph_id)
await graph_service.create_chunk_entity_relationship(chunk_id, entity_id, rel_type, graph_id)
```

#### **Enhanced Methods**

```python
# NEW: Proper vector storage methods in vector_service
await vector_service.store_node_embedding(node_id, embedding, graph_id)
await vector_service.store_chunk_embedding(chunk_id, embedding, graph_id, text)
await vector_service.similarity_search(query_embedding, graph_id, k=5)
```

### **For Developers Adding New Features**

#### **Service Selection Guide**

| Need to... | Use Service | Example Method |
|------------|-------------|----------------|
| Convert text to vectors | `embedding_service` | `embed_text()` |
| Store vectors in indexes | `vector_service` | `store_node_embedding()` |
| Create Neo4j nodes/relationships | `graph_service` | `create_chunk_node()` |
| Search for similar entities | `search_service` | `similarity_search_entities()` |
| Coordinate full graph workflows | `enhanced_graph_service` | `store_complete_graph()` |
| Run end-to-end RAG queries | `graphrag_service` | `graph_augmented_retrieval()` |

#### **Dependency Guidelines**

```python
# ✅ GOOD - Clear delegation:
class MyService:
    async def my_workflow(self):
        embedding = await embedding_service.embed_text(text)
        await vector_service.store_node_embedding(node_id, embedding, graph_id)
        
# ❌ BAD - Don't bypass service boundaries:
class MyService:
    async def my_workflow(self):
        # Don't call Neo4j directly if graph_service has the method
        await neo4j_client.execute_query(...)  # Use graph_service instead
```

### **Breaking Changes**

**None.** This refactoring maintains full backward compatibility for all public APIs.
