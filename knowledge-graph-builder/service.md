# Service Index and In-Depth Analysis

This document provides a detailed index and analysis of each service in the `/app/services/` directory, including their main classes/functions, responsibilities, and integration points.

---

## 1. `advanced_graph_analytic.py`
- **Purpose:** Implements advanced graph analytics (centrality, community detection, clustering, etc.)
- **Key Functions/Classes:**
  - Graph metric calculators (e.g., PageRank, betweenness)
  - Community detection algorithms
- **Integration:** Used by graph routers and analytics endpoints to provide insights on graph structure.

## 2. `advanced_graph_integration_service.py`
- **Purpose:** Integrates external graph data sources and merges them into the Neo4j graph.
- **Key Functions/Classes:**
  - Data harmonization and schema mapping utilities
  - Merge and update logic for external graphs
- **Integration:** Called by ingestion and graph routers when importing or synchronizing external graphs.

## 3. `chat_service.py`
- **Purpose:** Core chat logic, context management, and LLM interaction for user queries.
- **Key Functions/Classes:**
  - Session management
  - Query routing and retrieval strategies
  - LLM prompt construction and response parsing
- **Integration:** Used by chat router for conversational endpoints.

## 4. `document_processor.py`
- **Purpose:** Processes raw documents, extracts text and metadata.
- **Key Functions/Classes:**
  - File parsing (PDF, DOCX, HTML, etc.)
  - Metadata extraction
- **Integration:** Used by document ingestion and extraction services.

## 5. `document_service.py`
- **Purpose:** Manages document ingestion, storage, and basic processing.
- **Key Functions/Classes:**
  - Upload, list, retrieve documents
  - Source management (local, S3, GCS, web)
- **Integration:** Called by document routers and extraction services.

## 6. `document_service_async.py`
- **Purpose:** Asynchronous document ingestion and processing for scalability.
- **Key Functions/Classes:**
  - Async versions of upload, scan, and process
- **Integration:** Used for non-blocking document operations in routers/services.

## 7. `embedding_service.py`
- **Purpose:** Generates vector embeddings for documents, nodes, and queries.
- **Key Functions/Classes:**
  - Embedding generation (OpenAI, HuggingFace, etc.)
  - Similarity search utilities
- **Integration:** Used by chat, document, and graph services for semantic search and retrieval.

## 8. `enhanced_chat_service.py`
- **Purpose:** Extends chat service with multi-turn, context-aware, and advanced retrieval features.
- **Key Functions/Classes:**
  - Conversation history management
  - Advanced retrieval and reranking
- **Integration:** Used by chat router for enhanced conversational endpoints.

## 9. `enhanced_graph_service.py`
- **Purpose:** Advanced graph operations (schema learning, visualization, enhanced querying).
- **Key Functions/Classes:**
  - Schema inference
  - Visualization utilities
  - Advanced query builders
- **Integration:** Used by graph router and analytics endpoints.

## 10. `entity_resolution.py`
- **Purpose:** Detects and merges duplicate entities in the graph.
- **Key Functions/Classes:**
  - Entity matching algorithms
  - Merge and deduplication logic
- **Integration:** Used during graph ingestion and update operations.

## 11. `extraction_service.py`
- **Purpose:** Extracts structured knowledge (entities, relationships) from documents using NLP/LLMs.
- **Key Functions/Classes:**
  - Entity and relation extraction pipelines
  - Integration with LLMs and NLP models
- **Integration:** Used by document and graph services to populate the knowledge graph.

## 12. `graph_service.py`
- **Purpose:** Core graph operations (CRUD for nodes/relationships, querying, updating).
- **Key Functions/Classes:**
  - Node and relationship management
  - Query builders and executors
- **Integration:** Central service for all graph-related routers and endpoints.

## 13. `multi_modal_processing.py`
- **Purpose:** Processes multi-modal data (text, images, audio, video) for knowledge graph enrichment.
- **Key Functions/Classes:**
  - Multi-modal data extraction pipelines
  - Integration with external APIs/models
- **Integration:** Used by document and extraction services for richer graph construction.

---

# Index Table
| Service File                      | Main Responsibility                        | Key Classes/Functions                | Integration Points                  |
|-----------------------------------|--------------------------------------------|--------------------------------------|-------------------------------------|
| advanced_graph_analytic.py        | Graph analytics, metrics, clustering       | Metric calculators, community algos  | Graph routers, analytics endpoints  |
| advanced_graph_integration_service.py | External graph integration, harmonization | Schema mapping, merge logic          | Ingestion, graph routers            |
| chat_service.py                   | Chat logic, LLM interaction                | Session mgmt, retrieval, LLM prompts | Chat router                         |
| document_processor.py             | Document parsing, metadata extraction      | File parsers, metadata extractors    | Document ingestion, extraction      |
| document_service.py               | Document ingestion/storage                 | Upload, list, retrieve               | Document routers, extraction        |
| document_service_async.py         | Async document operations                  | Async upload/scan/process            | Async routers/services              |
| embedding_service.py              | Embedding generation, similarity search    | Embedding generators, search utils   | Chat, document, graph services      |
| enhanced_chat_service.py          | Advanced chat features                     | History mgmt, advanced retrieval     | Chat router                         |
| enhanced_graph_service.py         | Advanced graph ops, schema, visualization  | Schema inference, visualization      | Graph router, analytics             |
| entity_resolution.py              | Entity deduplication/merging               | Matching, merge logic                | Ingestion, graph update             |
| extraction_service.py             | Knowledge extraction from docs             | Extraction pipelines, LLM/NLP        | Document/graph services             |
| graph_service.py                  | Core graph CRUD/query                      | Node/rel mgmt, query builders        | All graph endpoints                 |
| multi_modal_processing.py         | Multi-modal data processing                | Extraction pipelines, API integration| Document/extraction services        |
