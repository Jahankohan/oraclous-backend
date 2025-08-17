# Neo4j LLM Graph Builder - High-Level Architecture Overview

## Overview

This project is a modular FastAPI backend for building, analyzing, and interacting with knowledge graphs using Neo4j and Large Language Models (LLMs). It supports advanced document processing, graph analytics, chat interfaces, and real-time progress updates.

---

## Main Architectural Layers

### 1. **API Layer**
- **Routers** (`app/routers/`): FastAPI endpoints for chat, documents, graph management, and infrastructure.
- **Request/Response Models** (`app/models/`): Pydantic models for validation and serialization.

### 2. **Service Layer**
- **Document Services** (`app/services/document_service.py`, `document_service_async.py`, `document_processor.py`): Document ingestion, scanning, and processing.
- **Extraction Service** (`app/services/extraction_service.py`): Knowledge graph extraction from documents.
- **Graph Services** (`app/services/graph_service.py`, `enhanced_graph_service.py`, `advanced_graph_analytic.py`, `advanced_graph_integration_service.py`): Graph querying, analytics, visualization, schema management, and integration.
- **Chat Services** (`app/services/chat_service.py`, `enhanced_chat_service.py`): Chatbot logic, retrieval strategies, and context management.
- **Embedding Service** (`app/services/embedding_service.py`): Embedding generation and similarity calculations.
- **Entity Resolution** (`app/services/entity_resolution.py`): Duplicate detection and node merging.
- **Multi-modal Processing** (`app/services/multi_modal_processing.py`): Support for various document types and sources.

### 3. **Core Layer**
- **Neo4j Connection** (`app/core/neo4j_pool.py`, `neo4j_client.py`): Async connection pooling and database client.
- **Dependency Injection** (`app/core/dependencies.py`): Centralized service and database dependency management.
- **Exception Handling** (`app/core/exceptions.py`): Custom error types and global error handling.

### 4. **Utility Layer**
- **LLM Client Factory** (`app/utils/llm_clients.py`): Abstraction for multiple LLM providers (OpenAI, Anthropic, Gemini, etc.).
- **File Handlers** (`app/utils/file_handlers.py`): File upload and storage utilities.

---

## Key Features

- **Async/Await Support**: Non-blocking operations throughout.
- **Connection Pooling**: Efficient Neo4j resource management.
- **Streaming Responses**: Real-time progress via Server-Sent Events.
- **Advanced Graph Analytics**: Centrality, community detection, schema learning.
- **Flexible Document Sources**: Local, S3, GCS, YouTube, Wikipedia, web.
- **Chat Interface**: Multiple retrieval modes, session management.
- **LLM Integration**: Pluggable support for various providers.
- **Error Handling & Validation**: Centralized, type-safe, and robust.

---

## Extensibility

- **Add new LLM providers** via `app/utils/llm_clients.py`
- **Support new document sources** in `app/services/document_service.py`
- **Implement new chat modes** in `app/services/chat_service.py`
- **Extend graph analytics** in `app/services/advanced_graph_analytic.py`

---

## Deployment

- **Docker**: Multi-stage build and health checks.
- **Environment Variables**: Centralized config via `.env` files.
- **Monitoring**: Health endpoints, structured logging, metrics-ready.

---

## References

- [README.md](README.md)
- [improvement-plan.md](improvement-plan.md)
- [IMPROVEMENTS_SUMMARY.md](IMPROVEMENTS_SUMMARY.md)

```// filepath: /Users/reza/workspace/Oraclous/oraclous-data-studio/knowledge-graph-builder/PROJECT_OVERVIEW.md

# Neo4j LLM Graph Builder - High-Level Architecture Overview

## Overview

This project is a modular FastAPI backend for building, analyzing, and interacting with knowledge graphs using Neo4j and Large Language Models (LLMs). It supports advanced document processing, graph analytics, chat interfaces, and real-time progress updates.

---

## Main Architectural Layers

### 1. **API Layer**
- **Routers** (`app/routers/`): FastAPI endpoints for chat, documents, graph management, and infrastructure.
- **Request/Response Models** (`app/models/`): Pydantic models for validation and serialization.

### 2. **Service Layer**
- **Document Services** (`app/services/document_service.py`, `document_service_async.py`, `document_processor.py`): Document ingestion, scanning, and processing.
- **Extraction Service** (`app/services/extraction_service.py`): Knowledge graph extraction from documents.
- **Graph Services** (`app/services/graph_service.py`, `enhanced_graph_service.py`, `advanced_graph_analytic.py`, `advanced_graph_integration_service.py`): Graph querying, analytics, visualization, schema management, and integration.
- **Chat Services** (`app/services/chat_service.py`, `enhanced_chat_service.py`): Chatbot logic, retrieval strategies, and context management.
- **Embedding Service** (`app/services/embedding_service.py`): Embedding generation and similarity calculations.
- **Entity Resolution** (`app/services/entity_resolution.py`): Duplicate detection and node merging.
- **Multi-modal Processing** (`app/services/multi_modal_processing.py`): Support for various document types and sources.

### 3. **Core Layer**
- **Neo4j Connection** (`app/core/neo4j_pool.py`, `neo4j_client.py`): Async connection pooling and database client.
- **Dependency Injection** (`app/core/dependencies.py`): Centralized service and database dependency management.
- **Exception Handling** (`app/core/exceptions.py`): Custom error types and global error handling.

### 4. **Utility Layer**
- **LLM Client Factory** (`app/utils/llm_clients.py`): Abstraction for multiple LLM providers (OpenAI, Anthropic, Gemini, etc.).
- **File Handlers** (`app/utils/file_handlers.py`): File upload and storage utilities.

---

## Key Features

- **Async/Await Support**: Non-blocking operations throughout.
- **Connection Pooling**: Efficient Neo4j resource management.
- **Streaming Responses**: Real-time progress via Server-Sent Events.
- **Advanced Graph Analytics**: Centrality, community detection, schema learning.
- **Flexible Document Sources**: Local, S3, GCS, YouTube, Wikipedia, web.
- **Chat Interface**: Multiple retrieval modes, session management.
- **LLM Integration**: Pluggable support for various providers.
- **Error Handling & Validation**: Centralized, type-safe, and robust.

---

## Extensibility

- **Add new LLM providers** via `app/utils/llm_clients.py`
- **Support new document sources** in `app/services/document_service.py`
- **Implement new chat modes** in `app/services/chat_service.py`
- **Extend graph analytics** in `app/services/advanced_graph_analytic.py`

---

##