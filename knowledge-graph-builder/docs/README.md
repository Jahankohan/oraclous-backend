# Knowledge Graph Builder — Documentation

## Getting Started

| Guide | Description |
|---|---|
| [Quickstart](getting-started/quickstart.md) | Zero to first chat query in 10 minutes |

## API Reference

| Reference | Description |
|---|---|
| [Graph Management](api-reference/graphs.md) | Create, update, and manage knowledge graphs; ingest documents; set extraction instructions |
| [Chat](api-reference/chat.md) | Chat with a graph, stream responses, and explore retriever modes |
| [Schema](api-reference/schema.md) | Inspect and manage the Neo4j schema cache |

## Service Info

- **Base URL:** `http://localhost:8003/api/v1`
- **Authentication:** Bearer token via `Authorization: Bearer <token>` header
- **Interactive docs:** Available at `http://localhost:8003/docs` when `LOG_LEVEL=DEBUG`
- **Health check:** `GET /api/v1/health` (no auth required)
