# Chat Feature Implementation Specification

## Overview

Implementation of a streaming chat endpoint that provides conversational AI interface to knowledge graphs using neo4j_graphrag components with multi-tenant support.

## Core Requirements

### 1. Data Architecture Changes

#### Neo4j Graph Node Structure

#### Neo4j-Only Architecture

- **Complete PostgreSQL migration**: Remove knowledge_graphs table entirely
- **Graph nodes in Neo4j**: All graph metadata stored in Neo4j Graph nodes
- **Sessions in Neo4j**: Chat sessions as Neo4j Session nodes connected to Graph
- **Message History**: Use neo4j_graphrag's `Neo4jMessageHistory` with native Session nodes
- **Single database**: Neo4j as the single source of truth for all graph and chat data
- **Multi-tenant isolation**: Graph nodes with unique graph_id properties

#### Relationship Structure

```cypher
// Graph as root node (replaces PostgreSQL knowledge_graphs table completely)
(graph:Graph {
    graph_id: "uuid",
    name: "Graph Name",
    description: "Graph description",
    user_id: "uuid",
    created_at: datetime(),
    updated_at: datetime()
})

// Documents connected to graph
(document:Document)-[:BELONGS_TO]->(graph)

// Existing chunk/entity structure remains unchanged
(chunk:Chunk)-[:FROM_DOCUMENT]->(document)
(entity:__Entity__)-[:FROM_CHUNK]->(chunk)

// Chat sessions in Neo4j (no PostgreSQL)
(session:Session {
    session_id: "uuid",
    session_name: "Chat Session Name",
    created_at: datetime(),
    last_message_at: datetime()
})-[:USES_GRAPH]->(graph)

// Message history using neo4j_graphrag's Neo4jMessageHistory
// This leverages the existing Session node structure from neo4j_graphrag
(message:Message {
    role: "user|assistant",
    content: "message text",
    created_at: datetime()
})-[:IN_SESSION]->(session)
```

### 2. Chat Service Components

#### Core Services Required:

1. **GraphChatService** - Main chat orchestration
2. **RetrieverFactory** - Dynamic retriever creation
3. **StreamingResponseHandler** - Sentence-level streaming
4. **SessionManager** - Session lifecycle management

#### Single Responsibility Compliance:

- `GraphChatService`: Orchestrates chat flow only
- `RetrieverFactory`: Creates and configures retrievers only
- `StreamingResponseHandler`: Handles response streaming only
- `SessionManager`: Manages session CRUD only

### 3. Retrieval Strategy

#### Manual Retrieval Selection (Phase 1)

Client specifies retrieval type via API parameter:

- `vector`: Semantic similarity search
- `graph`: Graph traversal patterns
- `hybrid`: Combined vector + full-text with ranking
- `text2cypher`: Natural language to Cypher
- `vector_cypher`: Vector search + custom Cypher

#### Retriever Implementation

- Use native neo4j_graphrag retrievers (VectorRetriever, HybridRetriever, etc.)
- No imports from clean/ directory
- Follow neo4j_graphrag component patterns

### 4. Streaming Implementation

#### Streaming Strategy: Server-Sent Events (SSE)

- **Granularity**: Sentence-level streaming
- **Real-time feedback** during retrieval and generation
- **Error handling** with graceful fallbacks

#### Response Flow:

1. **Retrieval phase**: Stream retrieval progress
2. **Generation phase**: Stream LLM response sentence by sentence
3. **Completion**: Final metadata and sources

### 5. API Endpoint Structure

```python
POST /api/v1/graphs/{graph_id}/chat/sessions/{session_id}/stream
Content-Type: application/json

{
    "message": "user query",
    "retrieval_type": "hybrid",  # manual selection
    "retrieval_config": {
        "top_k": 5,
        "alpha": 0.5
    }
}

Response: text/event-stream (SSE)
```

## Implementation Phases

### Phase 1: Foundation (Priority 1) - Step by Step Testing

**Step 1.1: Graph Node Migration**

- Remove PostgreSQL knowledge_graphs table
- Create Graph node creation in Neo4j
- Test: Create graph, verify node structure

**Step 1.2: Document-Graph Connection**

- Modify document processing to connect to Graph node
- Update pipeline to use `(Document)-[:BELONGS_TO]->(Graph)` relationship
- Test: Process document, verify graph connection

**Step 1.3: Session Management**

- Implement Session node creation in Neo4j
- Connect sessions to Graph nodes via `[:USES_GRAPH]` relationship
- Test: Create session, verify graph connection

**Step 1.4: Basic Message History**

- Integrate neo4j_graphrag's `Neo4jMessageHistory` with Session nodes
- Test: Send message, verify storage and retrieval

**Step 1.5: Simple Chat Service**

- Create basic GraphChatService (non-streaming)
- Implement single hybrid retrieval
- Test: Send query, get response

### Phase 2: Streaming (Priority 2)

1. **SSE implementation**: Sentence-level streaming
2. **Progress notifications**: Retrieval and generation progress
3. **Error handling**: Graceful fallbacks and error streaming

### Phase 3: Multi-Retrieval (Priority 3)

1. **Retriever factory**: All 5 retrieval types
2. **Dynamic configuration**: Per-request retrieval settings
3. **Result ranking**: Confidence scoring and source attribution

### Phase 4: Advanced Features (Future)

1. **Context management**: Conversation summarization
2. **Query enhancement**: Intent detection and query optimization
3. **Analytics**: Usage tracking and performance metrics

## Architecture Questions

### Location Decisions:

1. **GraphChatService**: `app/services/chat_service.py`
2. **RetrieverFactory**: `app/services/retriever_factory.py`
3. **StreamingHandler**: `app/services/streaming_service.py`
4. **Chat API**: `app/api/v1/chat.py`

### Required Components:

- **Is Graph node migration required?** Yes - centralizes graph metadata in Neo4j
- **Is retriever factory required?** Yes - supports multiple retrieval strategies
- **Is streaming required?** Yes - core requirement for user experience
- **Is session management required?** Yes - conversational context needed

### Maintainability:

- Follow neo4j_graphrag patterns for easy component reuse
- Use dependency injection for service composition
- Implement clear interfaces for future extensibility
- No complex inheritance hierarchies

### Extensibility:

- Factory pattern for easy retriever addition
- Plugin architecture for future retrieval strategies
- Clear separation of concerns for independent development
- Standard neo4j_graphrag component interfaces

## Technical Constraints

1. **No imports from clean/ directory** - Use only neo4j_graphrag components
2. **No backward compatibility** - Clean implementation without legacy support
3. **DRY principles** - Reuse existing neo4j_graphrag patterns
4. **Single responsibility** - Each component has one clear purpose
5. **Multi-tenant isolation** - All operations scoped to graph_id

## Success Criteria

1. **Functional**: Chat responses with accurate, grounded information
2. **Performance**: Sentence-level streaming with <500ms first response
3. **Reliability**: Graceful error handling and fallback responses
4. **Maintainable**: Clear, testable, and extensible code structure
5. **Compliant**: Follows neo4j_graphrag patterns and best practices

## Next Steps - Phase 1 Implementation

### Immediate Action Plan:

**Step 1.1: Graph Node Migration (First Step)**

1. Create Graph node model/schema in Neo4j
2. Remove PostgreSQL knowledge_graphs table dependency
3. Update existing services to use Neo4j Graph nodes
4. **Manual Test**: Create a graph node and verify structure

**Step 1.2: Document-Graph Relationship**

1. Modify document processing pipeline
2. Add `[:BELONGS_TO]` relationship creation
3. **Manual Test**: Process a document and verify graph connection

**Ready to Start**: Should we begin with Step 1.1 - Graph Node Migration?

This approach ensures we can test each component individually before moving to the next step.

---

_This specification follows the principle of starting simple and building incrementally, ensuring each phase is stable before moving to the next._
