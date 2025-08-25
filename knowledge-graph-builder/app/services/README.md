# Services Directory

## 1. Overview

This directory contains the main business logic for the knowledge graph builder. Services here orchestrate graph operations, chat flows, embeddings, entity extraction, schema management, and integrations with external APIs and databases. They are invoked by API endpoints and interact with models, core utilities, and external systems.

## 2. Modules & Services

### advanced_graph_context.py
- **Functionality:** Extracts advanced context from the graph for analytics and reasoning, e.g., subgraph extraction, context-aware recommendations.
- **Connections:** Used by analytics and chat services for context-aware responses. Relies on `graph_service.py` and models.
- **Risks:** May tightly couple to graph schema; changes in graph structure can break context extraction.

### auth_service.py
- **Functionality:** Handles authentication logic, user/session management, token validation.
- **Connections:** Integrates with credential services and API routers. Relies on models and possibly external auth providers.
- **Risks:** Tight coupling to credential storage; changes in credential schema may require updates.
- **External APIs:** If using OAuth, refer to [OAuth 2.0 RFC](https://datatracker.ietf.org/doc/html/rfc6749).

### background_jobs.py
- **Functionality:** Manages background tasks (async jobs, scheduled tasks, e.g., ingestion, notifications).
- **Connections:** Used for long-running operations, possibly via Celery or FastAPI background tasks. May interact with database and external APIs.
- **Risks:** Async/sync mismatches if jobs are called synchronously from endpoints. Ensure proper error handling and job status tracking.
- **External APIs:** [Celery Documentation](https://docs.celeryq.dev/en/stable/), [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)

### chat_service.py
- **Functionality:** Implements chat logic, context retrieval, LLM interaction, multi-turn dialog management.
- **Connections:** Called by chat endpoints; depends on `embedding_service.py`, `llm_service.py`, and `graph_service.py`.
- **Risks:** Async/sync mismatches if LLM calls are blocking. Tight coupling to embedding and graph services.
- **External APIs:** [OpenAI API](https://platform.openai.com/docs/api-reference)

### credential_service.py
- **Functionality:** Manages credentials for external services (APIs, databases).
- **Connections:** Used by auth and integration services. Relies on secure storage and retrieval.
- **Risks:** Tight coupling to credential storage backend. Security risk if not properly encrypted.

### diffbot_graph_service.py
- **Functionality:** Integrates with Diffbot for external graph enrichment and data extraction.
- **Connections:** Used by ingestion and graph update flows. Relies on external Diffbot API.
- **Risks:** External API changes can break integration. Rate limits and error handling required.
- **External APIs:** [Diffbot API Docs](https://docs.diffbot.com/docs)

### embedding_service.py
- **Functionality:** Generates vector embeddings for documents, nodes, and queries using ML models or external APIs.
- **Connections:** Used by chat, document, and graph services for semantic search. May call external embedding APIs.
- **Risks:** Async/sync mismatches if embedding generation is slow. Tight coupling to embedding model version.
- **External APIs:** [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings)

### enhanced_chat_service.py
- **Functionality:** Extends chat logic with multi-turn, context-aware, and advanced retrieval features.
- **Connections:** Used by enhanced chat endpoints. Depends on chat, embedding, and graph services.
- **Risks:** Complex inter-service dependencies; risk of circular calls.

### enhanced_graph_service.py
- **Functionality:** Advanced graph operations (schema learning, visualization, enhanced querying).
- **Connections:** Used by graph router and analytics endpoints. Relies on `graph_service.py` and models.
- **Risks:** Tight coupling to graph schema and database structure.

### entity_extractor.py
- **Functionality:** Extracts entities from documents using NLP/LLMs.
- **Connections:** Used by document and extraction services. May call external NLP APIs.
- **Risks:** Async/sync mismatches if extraction is slow. Model drift risk.
- **External APIs:** [spaCy](https://spacy.io/), [OpenAI API](https://platform.openai.com/docs/api-reference)

### graph_service.py
- **Functionality:** Core graph CRUD/query logic, node/edge management, traversal, and analytics.
- **Connections:** Called by most graph endpoints; interacts with Neo4j via `core/neo4j_client.py`.
- **Risks:** **Tight coupling** to Neo4j database; schema changes require code updates. Risk of sync/async mismatches if DB calls are blocking.
- **External APIs:** [Neo4j Python Driver](https://neo4j.com/docs/api/python-driver/current/)

### graphrag_service.py
- **Functionality:** Retrieval-augmented generation (RAG) for graph-based queries, combining search and LLMs.
- **Connections:** Used by chat and analytics flows. Depends on graph, embedding, and LLM services.
- **Risks:** Async/sync mismatches if LLM or search is slow. Complex error handling required.

### llm_service.py
- **Functionality:** Handles LLM interaction (prompting, response parsing, streaming).
- **Connections:** Used by chat, extraction, and analytics services. Calls external LLM APIs.
- **Risks:** Async/sync mismatches if LLM calls are blocking. API changes can break integration.
- **External APIs:** [OpenAI API](https://platform.openai.com/docs/api-reference)

### schema_service.py
- **Functionality:** Manages schema inference, validation, and evolution for graph data.
- **Connections:** Used by graph and analytics services. Relies on models and graph service.
- **Risks:** Tight coupling to graph schema; changes require updates across services.

### search_service.py
- **Functionality:** Search utilities (fulltext, semantic, graph-based search).
- **Connections:** Used by chat and graph endpoints. Depends on embedding and graph services.
- **Risks:** Async/sync mismatches if search is slow or blocking.

### sync_ingestion_processor.py
- **Functionality:** Handles synchronous ingestion of external data sources into the graph.
- **Connections:** Used by document and integration services. Relies on graph and entity extraction services.
- **Risks:** Blocking calls can slow down API responses. Tight coupling to ingestion pipeline.

### vector_service.py
- **Functionality:** Manages vector operations (similarity search, clustering, indexing).
- **Connections:** Used by embedding and graph services. May call external vector DBs.
- **Risks:** Async/sync mismatches if vector DB is slow. Tight coupling to vector DB schema.
- **External APIs:** [Pinecone](https://docs.pinecone.io/docs/overview), [FAISS](https://github.com/facebookresearch/faiss)

### __init__.py
- **Functionality:** Package initializer.

### __pycache__/
- **Functionality:** Python bytecode cache; not relevant for documentation.

## 3. Interconnections

- **Services depend on models** (from `app/models`) for data structures.
- **Most services interact with Neo4j** via `core/neo4j_client.py`.
- **API endpoints** (in `app/api/v1/endpoints`) call these services to fulfill requests.
- **Embedding, LLM, and vector services** are used by chat, extraction, and analytics flows.
- **Schema and graph services** are central to most graph operations.
- **Background jobs** may call any service asynchronously.

## 4. Example Flow

```
api/v1/endpoints/chat.py -> chat_service.py -> embedding_service.py -> llm_service.py
api/v1/endpoints/graph.py -> graph_service.py -> schema_service.py -> enhanced_graph_service.py
api/v1/endpoints/extract.py -> entity_extractor.py -> graph_service.py
```

## 5. Notes

- **Tight Coupling:**
  - `graph_service.py` and `enhanced_graph_service.py` are tightly coupled to Neo4j and graph schema. Changes in DB require code updates.
  - `credential_service.py` and `auth_service.py` are tightly coupled to credential storage.
  - `embedding_service.py`, `vector_service.py`, and `llm_service.py` are coupled to external APIs and model versions.

- **Async/Sync Mismatches:**
  - Background jobs and external API calls should be async; blocking calls can degrade performance.
  - Ensure endpoints do not block on slow external services.

- **External API Documentation:**
  - [OpenAI API](https://platform.openai.com/docs/api-reference)
  - [Diffbot API](https://docs.diffbot.com/docs)
  - [Neo4j Python Driver](https://neo4j.com/docs/api/python-driver/current/)
  - [Pinecone](https://docs.pinecone.io/docs/overview)
  - [FAISS](https://github.com/facebookresearch/faiss)
  - [spaCy](https://spacy.io/)
  - [Celery](https://docs.celeryq.dev/en/stable/)
  - [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)

- **Risks:**
  - External API changes, rate limits, and outages can break integrations.
  - Security risks if credentials are not properly managed.
  - Model drift and schema evolution require regular updates.

- **Assumptions:**
  - Neo4j and external APIs are available and properly configured.
  - Services are called via API endpoints or background jobs.
