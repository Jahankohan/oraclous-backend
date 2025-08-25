# Knowledge Graph Builder Service

A FastAPI-based microservice for building, querying, and analyzing knowledge graphs from unstructured data, fully integrated into the Oraclous ecosystem.

---

## 🎯 Project Purpose

Transform unstructured data into rich, queryable knowledge graphs using LLMs, advanced entity extraction, and graph analytics. Expose APIs for graph management, ingestion, chat, search, and analytics, supporting natural language queries and orchestrator workflows.

---

## 🏗️ Architecture Overview

- **API Layer:** FastAPI endpoints, organized by domain and version ([see API README](app/api/README.md), [v1 README](app/api/v1/README.md))
- **Service Layer:** Business logic for graph ops, chat, embeddings, extraction, schema, analytics ([see Services README](app/services/README.md))
- **Model Layer:** Data structures for graph nodes, edges, chat, jobs ([see Models README](app/models/README.md))
- **Core Layer:** Configuration, logging, database clients ([see Core README](app/core/README.md))
- **Schema Layer:** Pydantic schemas for validation ([see Schemas README](app/schemas/README.md))
- **Design Docs:** Architecture, implementation plan, backlog, and development rules ([see design-docs/](design-docs/))

### Key Integration Points

- **Auth Service:** User authentication & OAuth
- **Credential Broker:** LLM/Neo4j credential management
- **Oraclous Core Service:** Tool registration, workflow orchestration
- **Neo4j:** Graph storage, vector indexes, analytics
- **PostgreSQL:** Metadata, jobs, chat sessions

---

## 📋 Implementation Roadmap

See [knowledge-graph-builder-implementation-integration.md](design-docs/knowledge-graph-builder-implementation-integration.md) for full checkpoint breakdown.

- **Checkpoint 1:** Service foundation, Neo4j connectivity, health/auth endpoints
- **Checkpoint 2:** Entity/relationship extraction, Diffbot integration, schema management
- **Checkpoint 3:** Embeddings, vector search, hybrid search
- **Checkpoint 4:** LLM chat, GraphRAG, text-to-Cypher, chat history
- **Checkpoint 5:** Orchestrator integration, tool registration, credits
- **Checkpoint 6:** Advanced analytics, optimization, monitoring

---

## 🔗 API Endpoints

See [API README](app/api/README.md) and [v1 README](app/api/v1/README.md) for details.

- `/api/v1/graphs/` - Graph CRUD
- `/api/v1/graphs/{id}/ingest` - Data ingestion
- `/api/v1/graphs/{id}/entities` - Entity management
- `/api/v1/graphs/{id}/relationships` - Relationship management
- `/api/v1/graphs/{id}/search` - Search (keyword, semantic, hybrid)
- `/api/v1/graphs/{id}/chat` - Chat with graph (LLM, GraphRAG)
- `/api/v1/graphs/{id}/analytics/` - Metrics, community detection, centrality
- `/api/v1/health` - Health check
- `/api/v1/metrics` - Prometheus metrics

---

## 📊 Database & Indexes

- **PostgreSQL:** Metadata tables for graphs, jobs, chat sessions/messages
- **Neo4j:** Nodes (Entity, Chunk, Document), vector indexes for embeddings, full-text indexes, constraints

See implementation plan for schema details.

---

## 🔧 Configuration

Environment variables (see `.env.example`):

- `NEO4J_URI`, `POSTGRES_URL`, `REDIS_URL`
- `AUTH_SERVICE_URL`, `CREDENTIAL_BROKER_URL`
- `INTERNAL_SERVICE_KEY`, `JWT_SECRET_KEY`
- `LOG_LEVEL`, `ENABLE_METRICS`

---

## 🧪 Testing

- Unit and integration tests in `tests/`
- Run with `pytest` or `test.sh`
- Coverage reports: `pytest --cov=app --cov-report=html`

---

## 📈 Monitoring & Health

- Health endpoint: `/api/v1/health`
- Logs: `logs/` directory
- Metrics: `/api/v1/metrics` (Prometheus format)

---

## 🤝 Oraclous Ecosystem Integration

- Auth via `auth-service`
- Credentials via `credential-broker-service`
- Tool registration and workflow via `oraclous-core-service`
- All services communicate over `app-network` (Docker Compose)

---

## 🛠️ Development Rules & Refactoring

**Golden Rule:** Enhance existing services in place—never duplicate functionality.  
See [development-architecture-guide.md](design-docs/development-architecture-guide.md) for strict rules, enhancement patterns, and refactoring checklists.

- One service file per major functionality
- All enhancements done in place
- Endpoints connect to actual implementations
- No orphaned or unused service files
- Clear method organization within services

---

## 📋 Backlog & Next Steps

See [ultimate-graph-plans.md](design-docs/ultimate-graph-plans.md) for prioritized backlog, sprint plan, and risk mitigation.

- Critical: Chat hallucination prevention, enhanced graph modeling
- Medium: Multi-modal extraction, user context system
- Nice-to-have: Real-time progress, streaming, analytics dashboard
- Production: Monitoring, optimization, orchestrator integration

---

## 🚀 Deployment

- Dockerfile and docker-compose integration
- Healthcheck and auto-reload scripts
- Production configuration and monitoring

---

## 📚 References

- [app/core/README.md](app/core/README.md)
- [app/models/README.md](app/models/README.md)
- [app/schemas/README.md](app/schemas/README.md)
- [app/services/README.md](app/services/README.md)
- [app/api/README.md](app/api/README.md)
- [app/api/v1/README.md](app/api/v1/README.md)
- [design-docs/development-architecture-guide.md](design-docs/development-architecture-guide.md)
- [design-docs/knowledge-graph-builder-implementation-integration.md](design-docs/knowledge-graph-builder-implementation-integration.md)
- [design-docs/ultimate-graph-plans.md](design-docs/ultimate-graph-plans.md)

---

## 🎯 Success Criteria

- All checkpoints completed and integrated
- No duplicate services or endpoints
- All endpoints have working implementations
- Performance, scalability, and monitoring in place
- User documentation and API docs complete

---

## 👥 Contribution & Contact

Open issues or pull requests for questions, improvements, or bug reports.  
Follow the architecture and enhancement guidelines for all