# API Reference: Chat

Base URL: `http://localhost:8003/api/v1`

All endpoints require `Authorization: Bearer <token>` header.

---

## Chat With a Knowledge Graph

Sends a natural language query to a knowledge graph and returns an answer grounded in the graph data. Hallucination is actively prevented — if the graph does not contain sufficient context, the response says so explicitly and sets `is_grounded: false`.

```
POST /chat
```

**Request Body**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | Yes | — | Natural language question (1–10,000 characters) |
| `graph_id` | string | Yes | — | Knowledge graph UUID |
| `mode` | string | No | `enhanced` | Chat mode (see [Modes](#chat-modes)) |
| `retriever_type` | string | No | — | Explicit retriever type — overrides `mode` |
| `retriever_config` | object | No | — | Retriever-specific configuration |
| `return_context` | boolean | No | `false` | Include full retrieval context in response |
| `include_sources` | boolean | No | `true` | Include source node information |
| `include_cypher` | boolean | No | `false` | Include generated Cypher (for `natural` mode) |
| `examples` | string | No | `""` | Few-shot examples for Text2Cypher mode |
| `conversation_id` | string | No | — | Session ID to track conversation history |

**Retriever Config Object**

| Field | Type | Default | Description |
|---|---|---|---|
| `top_k` | integer | `5` | Number of results to retrieve (1–100) |
| `effective_search_ratio` | integer | `1` | Search breadth multiplier (1–10) |
| `index_name` | string | auto | Vector index name |
| `fulltext_index_name` | string | auto | Full-text index name (hybrid modes) |
| `ranker` | string | `naive` | Ranking algorithm: `naive` or `linear` |
| `alpha` | float | — | Weight for `linear` ranker (0.0–1.0, required when `ranker=linear`) |
| `retrieval_query` | string | — | Custom Cypher retrieval query |
| `neo4j_schema` | string | — | Schema description for Text2Cypher mode |

```bash
curl -X POST http://localhost:8003/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What companies did TechNova acquire?",
    "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "mode": "enhanced",
    "include_sources": true
  }'
```

```python
import httpx

response = httpx.post(
    "http://localhost:8003/api/v1/chat",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "query": "What companies did TechNova acquire?",
        "graph_id": graph_id,
        "mode": "enhanced",
        "include_sources": True
    }
)
result = response.json()
print(result["answer"])
```

**Response** `200 OK`

```json
{
  "answer": "TechNova Corporation acquired DataStream Inc., a data pipeline startup based in Austin, in 2023.",
  "query": "What companies did TechNova acquire?",
  "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "success": true,
  "mode": "enhanced",
  "retriever_type": "vector_cypher",
  "is_grounded": true,
  "confidence": 0.91,
  "sources": [
    {
      "node_id": "4:abc123",
      "node_labels": ["Company", "__Entity__"],
      "relevance_score": 0.94,
      "content": "TechNova acquired DataStream Inc. in 2023.",
      "entities": ["TechNova Corporation", "DataStream Inc."],
      "properties": {
        "name": "DataStream Inc.",
        "description": "Data pipeline startup based in Austin"
      }
    }
  ],
  "context": null,
  "metadata": {
    "model": "gpt-4o",
    "include_cypher": false,
    "return_context": false
  },
  "timestamp": "2026-04-07T10:05:00Z",
  "conversation_id": null
}
```

**Response Fields**

| Field | Type | Description |
|---|---|---|
| `answer` | string | The generated answer |
| `is_grounded` | boolean | `true` if the answer is based on graph data; `false` if the graph lacked sufficient context |
| `confidence` | float | Retrieval confidence score [0.0–1.0] |
| `retriever_type` | string | Which retriever was actually used |
| `sources` | array | Source nodes used to generate the answer (when `include_sources: true`) |
| `context` | object | Full retrieval context (when `return_context: true`) |

**Errors**

| Status | Detail | Cause |
|---|---|---|
| `403` | Access denied | Graph belongs to another user or doesn't exist |
| `422` | Validation Error | Request body failed validation |
| `500` | Chat processing failed | LLM or retriever error |

---

### Example: Hybrid Plus Mode

For complex analytical questions requiring broad search coverage:

```bash
curl -X POST http://localhost:8003/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find all executives who joined companies after 2020",
    "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "mode": "hybrid_plus",
    "retriever_config": {
      "top_k": 10,
      "ranker": "linear",
      "alpha": 0.7
    },
    "return_context": true
  }'
```

> **Note:** `hybrid` and `hybrid_plus` modes require full-text indexes on the graph. If not present, the service will return an error suggesting you use `enhanced` mode instead.

---

### Example: Natural Language to Cypher

For precise graph queries where you need exact counts or specific relationship traversal:

```bash
curl -X POST http://localhost:8003/api/v1/chat \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How many companies were acquired between 2020 and 2024?",
    "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "mode": "natural",
    "include_cypher": true
  }'
```

---

## Stream Chat (Server-Sent Events)

Streams the chat response in real-time using Server-Sent Events (SSE). Use this for interactive UIs where you want to display the answer word-by-word as it's generated.

```
POST /chat/stream
```

**Request body** is identical to `POST /chat`.

```bash
curl -X POST http://localhost:8003/api/v1/chat/stream \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "query": "Tell me about TechNova Corporation",
    "graph_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "mode": "enhanced"
  }'
```

```python
import httpx

with httpx.stream(
    "POST",
    "http://localhost:8003/api/v1/chat/stream",
    headers={
        "Authorization": f"Bearer {token}",
        "Accept": "text/event-stream"
    },
    json={
        "query": "Tell me about TechNova Corporation",
        "graph_id": graph_id,
        "mode": "enhanced"
    }
) as response:
    for line in response.iter_lines():
        if line.startswith("data: "):
            import json
            event = json.loads(line[6:])
            if event["type"] == "answer_chunk":
                print(event["text"], end="", flush=True)
            elif event["type"] == "done":
                print()  # newline
                print(f"Grounded: {event['is_grounded']}, Confidence: {event['confidence']}")
```

**Response** `200 OK` — `text/event-stream`

Events are emitted in this order:

```
data: {"type": "source", "node_id": "4:abc123", "node_labels": ["Company"], "relevance_score": 0.94, "content": "TechNova was founded..."}

data: {"type": "source", "node_id": "4:def456", "node_labels": ["Person"], "relevance_score": 0.88, "content": "Dr. Sarah Chen, CTO..."}

data: {"type": "answer_chunk", "text": "TechNova "}

data: {"type": "answer_chunk", "text": "Corporation "}

data: {"type": "answer_chunk", "text": "is a technology company..."}

data: {"type": "done", "confidence": 0.91, "is_grounded": true, "retriever_used": "vector_cypher"}
```

**Event Types**

| Type | Timing | Fields |
|---|---|---|
| `source` | Before answer | `node_id`, `node_labels`, `relevance_score`, `content` |
| `answer_chunk` | During answer | `text` (word-level chunk) |
| `done` | After answer | `confidence`, `is_grounded`, `retriever_used` |
| `error` | On failure only | `message` |

---

## Get Available Chat Modes

Returns all available chat modes with descriptions and capabilities.

```
GET /modes
```

```bash
curl http://localhost:8003/api/v1/modes \
  -H "Authorization: Bearer $TOKEN"
```

**Response** `200 OK`

```json
{
  "modes": [
    {
      "mode": "simple",
      "name": "Simple Search",
      "description": "Fast vector similarity search for quick facts and direct questions",
      "retriever_type": "vector",
      "use_cases": [
        "Quick factual questions",
        "Simple entity lookups",
        "Fast responses"
      ],
      "requires_fulltext_index": false
    },
    {
      "mode": "enhanced",
      "name": "Enhanced Search",
      "description": "Vector similarity search combined with graph traversal for comprehensive context",
      "retriever_type": "vector_cypher",
      "use_cases": [
        "Complex questions requiring context",
        "Entity relationships exploration",
        "Comprehensive analysis"
      ],
      "requires_fulltext_index": false
    },
    {
      "mode": "hybrid",
      "name": "Hybrid Search",
      "description": "Combines vector similarity and full-text search for broader coverage",
      "retriever_type": "hybrid",
      "use_cases": [
        "Text and semantic search",
        "Keyword-based queries",
        "Broader result coverage"
      ],
      "requires_fulltext_index": true
    },
    {
      "mode": "hybrid_plus",
      "name": "Hybrid Plus",
      "description": "Hybrid search with graph traversal for maximum context and coverage",
      "retriever_type": "hybrid_cypher",
      "use_cases": [
        "Complex analytical questions",
        "Research and exploration",
        "Maximum context retrieval"
      ],
      "requires_fulltext_index": true
    },
    {
      "mode": "natural",
      "name": "Natural Query",
      "description": "Natural language to Cypher translation for precise graph queries",
      "retriever_type": "text2cypher",
      "use_cases": [
        "Complex graph analytics",
        "Specific relationship queries",
        "Custom analysis requirements"
      ],
      "requires_fulltext_index": false
    }
  ],
  "default_mode": "enhanced",
  "graph_capabilities": {
    "has_fulltext_indexes": true,
    "has_vector_indexes": true,
    "supports_cypher": true
  }
}
```

---

## Chat Modes Reference

### Choosing the Right Mode

| Mode | Speed | Context Depth | Full-Text Required | Best For |
|---|---|---|---|---|
| `simple` | Fastest | Low | No | Simple factual lookups, high-volume queries |
| `enhanced` | Fast | High | No | Most use cases — start here |
| `hybrid` | Medium | Medium | Yes | Keyword-heavy or mixed queries |
| `hybrid_plus` | Slower | Very High | Yes | Research, complex analysis |
| `natural` | Slow | Precise | No | Exact counts, specific graph traversals |

### Mode to Retriever Mapping

| Mode | Underlying Retriever | Description |
|---|---|---|
| `simple` | `vector` | Pure vector similarity on `__Entity__` index |
| `enhanced` | `vector_cypher` | Vector search + Cypher traversal for related nodes |
| `hybrid` | `hybrid` | Vector + full-text indexes, configurable `ranker` |
| `hybrid_plus` | `hybrid_cypher` | Hybrid search + Cypher traversal |
| `natural` | `text2cypher` | LLM-generated Cypher query from natural language |

You can bypass the mode system entirely by setting `retriever_type` directly in the request, which lets you combine any retriever with custom `retriever_config` parameters.
