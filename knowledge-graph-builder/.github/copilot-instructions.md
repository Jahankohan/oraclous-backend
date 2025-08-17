
# Copilot Instructions for knowledge-graph-builder

## Architecture Overview

- **Microservice** for building, querying, and analyzing a knowledge graph using FastAPI.
- **Entry point:** `app/main.py`
- **Layered structure:**
  - **Routers** (`app/routers/`): API endpoints for graph, documents, chat, infrastructure.
  - **Services** (`app/services/`): Business logic, orchestration, analytics, embedding, multi-modal processing.
  - **Core** (`app/core/`): Neo4j DB connection, pooling, exceptions, dependencies.
  - **Models** (`app/models/`): Pydantic schemas for requests/responses.
  - **Config** (`app/config/`): Loads environment variables and settings.
  - **Utils** (`app/utils/`): Helper functions (LLM clients, file handlers).

## Key Modules & Responsibilities

- `core/neo4j_client.py`: Manages Neo4j sessions and queries.
- `core/neo4j_pool.py`: Handles Neo4j connection pooling.
- `routers/graph.py`, `routers/documents.py`: Main API endpoints.
- `services/document_service.py`, `document_service_async.py`: Document node processing (sync/async).
- `services/embedding_service.py`: Generates embeddings for graph nodes.
- `services/multi_modal_processing.py`: Integrates cross-modal data (text, images, etc.).
- `models/requests.py`, `models/responses.py`: Pydantic schemas for API validation.

## Data Flow & Integration

- Request: Router → Service → Core Neo4j client → Neo4j DB → Response model.
- All graph operations use `neo4j_client.py` and `neo4j_pool.py`.
- Embedding/multi-modal: External models/services via `embedding_service.py`, `multi_modal_processing.py`.
- LLMs and file handling: Utilities in `utils/llm_clients.py`, `utils/file_handlers.py`.

## Developer Workflows

- **Run locally:**  
  `uvicorn app.main:app --reload`
- **Test:**  
  `pytest`
- **Coverage:**  
  `pytest --cov=app`
- **Debug:**  
  Use FastAPI docs at `/docs` after starting the service.
- **Docker Compose:**  
  `docker-compose up -d` (see README for details)

## Project Conventions

- Service modules use `_service.py` suffix.
- Routers import request/response models from `models/`.
- Cypher queries are written in service modules, not routers.
- Exceptions from `core/exceptions.py` are mapped to HTTP responses.
- Async services require async/await usage and proper DB drivers.
- Environment/config loaded via `config/settings.py`.

## Common Gotchas

- **Neo4j must be running** before starting the service.
- Required environment variables:
  - `NEO4J_URI`
  - `NEO4J_USER`
  - `NEO4J_PASSWORD`
- External embedding/multi-modal services may require additional env vars or API keys.
- Schema changes in Neo4j may require updates in Pydantic models.
