### **Project: Universal Graph Builder Service (Middle Stream ETL for Neo4j)**

**Objective:** Create a FastAPI service that accepts various raw data types, processes them into graph-ready objects (nodes and relationships), and returns these structures via API, decoupled from the final storage step.

---

### **Milestone 0: Project Foundation & Setup**

**Goal:** Establish the core project structure, environment, and basic API skeleton.

**Deliverables:**
1.  Project repository initialized with a standard Python structure.
2.  `requirements.txt` or `pyproject.toml` with initial dependencies.
3.  A basic, running FastAPI application with a simple health check endpoint.
4.  Dockerfile for containerization (highly recommended).
5.  Configuration management setup (e.g., for Neo4j connection details, even if just used for testing).

**Techniques & Libraries:**
*   **FastAPI:** Web framework.
*   **Pydantic:** For data validation and settings management.
*   **Python-dotenv:** For loading environment variables.
*   **Docker:** For containerization.
*   **UV:** Modern Python package manager (optional but recommended for speed).

**Code Snippet (app/main.py):**
```python
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI(title="Graph Builder Service")

class HealthCheckResponse(BaseModel):
    status: str
    message: str

@app.get("/", response_model=HealthCheckResponse)
async def health_check():
    return HealthCheckResponse(status="OK", message="Graph Builder Service is operational.")

# Run with: uvicorn app.main:app --reload
```

---

### **Milestone 1: Data Ingestion & Type Detection API**

**Goal:** Design robust API endpoints to accept different data types and automatically detect their nature.

**Deliverables:**
1.  FastAPI endpoints for each major data type:
    *   `POST /ingest/text`
    *   `POST /ingest/document` (for PDF/DOCX)
    *   `POST /ingest/relational` (accepts a connection string or data dump)
    *   `POST /ingest/web` (accepts raw HTML or a URL to crawl)
    *   `POST /ingest/qa`
2.  Request models (Pydantic schemas) for each endpoint.
3.  A utility function to detect MIME type or data structure upon ingestion.

**Techniques & Libraries:**
*   **FastAPI:** For defining endpoints with correct `Form` and `File` parameters.
*   **python-magic:** For accurate MIME type detection of uploaded files.

**Code Snippet (app/routers/ingest.py):**
```python
from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel, AnyUrl
import magic

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

class TextPayload(BaseModel):
    raw_text: str

class UrlPayload(BaseModel):
    url: AnyUrl

@router.post("/text")
async def ingest_text(payload: TextPayload):
    """Ingest raw text string."""
    return {"message": "Text ingested", "data_type": "text"}

@router.post("/document")
async def ingest_document(file: UploadFile = File(...)):
    """Ingest a document file (PDF, DOCX)."""
    # Detect file type
    content = await file.read()
    file_type = magic.from_buffer(content, mime=True)
    await file.seek(0)  # Reset file pointer for later processing
    return {"message": f"Document {file.filename} ingested", "detected_type": file_type}

@router.post("/web")
async def ingest_web(payload: UrlPayload):
    """Ingest content from a web URL."""
    return {"message": "URL received for processing", "url": str(payload.url)}
```

---

### **Milestone 2: Core Data Processing Modules**

**Goal:** Build the heart of the service - modular processors for each data type. Each processor should take raw input and return a standardized graph-object output.

**Deliverables:**
1.  A standardized Pydantic model for a `GraphObject` (e.g., `Node`, `Relationship`).
2.  A dedicated Python module for each processor:
    *   `processors/text_processor.py`
    *   `processors/document_processor.py`
    *   `processors/relational_processor.py`
    *   `processors/web_processor.py`
    *   `processors/qa_processor.py`
3.  Each module must have a main function `process(data) -> GraphObjectCollection`.

**Techniques & Libraries:**
*   **spaCy:** For high-quality NER and dependency parsing on text/HTML/QA data.
*   **pdfplumber / PyMuPDF:** For extracting text from PDFs with layout information.
*   **python-docx:** For extracting text from DOCX files.
*   **BeautifulSoup / Selectolax:** For fast and efficient HTML parsing.
*   **Pandas:** For handling relational data (CSVs, dataframes).
*   **SQLAlchemy:** For connecting to and introspecting SQL databases (optional, advanced).

**Code Snippet (app/models/graph.py):**
```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional, Dict, Any

class NodeLabels(str, Enum):
    PERSON = "Person"
    ORG = "Organization"
    # ... define all your node types

class Node(BaseModel):
    id: str = Field(..., description="A unique identifier for the node")
    labels: List[NodeLabels] = Field(..., description="The labels of the node")
    properties: Dict[str, Any] = Field(default_factory=dict, description="The properties of the node")

class RelationshipType(str, Enum):
    WORKS_AT = "WORKS_AT"
    # ... define all your relationship types

class Relationship(BaseModel):
    source_id: str = Field(..., description="The ID of the source node")
    target_id: str = Field(..., description="The ID of the target node")
    type: RelationshipType = Field(..., description="The type of the relationship")
    properties: Dict[str, Any] = Field(default_factory=dict, description="The properties of the relationship")

class GraphObjectCollection(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
```

**Code Snippet (app/processors/text_processor.py):**
```python
import spacy
from .graph import GraphObjectCollection, Node, NodeLabels, Relationship, RelationshipType

nlp = spacy.load("en_core_web_sm") # Load model once

def process_text(raw_text: str) -> GraphObjectCollection:
    """Process raw text into graph objects using spaCy NER."""
    collection = GraphObjectCollection()
    doc = nlp(raw_text)

    seen_entities = {}  # To deduplicate nodes by text

    for ent in doc.ents:
        # Simple deduplication
        if ent.text not in seen_entities:
            label = map_spacy_label_to_ours(ent.label_)  # Map 'ORG' -> NodeLabels.ORG
            node_id = f"{label}_{len(seen_entities)}"
            new_node = Node(id=node_id, labels=[label], properties={"name": ent.text})
            collection.nodes.append(new_node)
            seen_entities[ent.text] = node_id

    # Add logic for relationship extraction here (more advanced)
    # This could involve dependency parsing or a separate model
    return collection

def map_spacy_label_to_ours(label: str) -> NodeLabels:
    mapping = {"ORG": NodeLabels.ORG, "PERSON": NodeLabels.PERSON, "GPE": NodeLabels.LOCATION}
    return mapping.get(label, NodeLabels.ENTITY) # Default fallback
```

---

### **Milestone 3: Service Integration & Orchestration**

**Goal:** Connect the ingestion API endpoints to the correct processing modules and return the graph objects.

**Deliverables:**
1.  Updated router functions that call the appropriate processor.
2.  Robust error handling and logging throughout the ingestion and processing pipeline.
3.  The API now returns a `GraphObjectCollection` as the response.

**Techniques & Libraries:**
*   **FastAPI Dependency Injection:** For managing processor dependencies (e.g., shared NLP model).
*   **Logging:** Python's built-in `logging` module.
*   **Async Processing:** For long-running tasks, consider using `BackgroundTasks` or a message queue (e.g., **Celery** + **Redis**) for Milestone 5.

**Code Snippet (app/routers/ingest.py - updated):**
```python
from app.processors.text_processor import process_text
from app.processors.document_processor import process_document_file
from app.models.graph import GraphObjectCollection

# ... other imports and endpoints ...

@router.post("/text", response_model=GraphObjectCollection)
async def ingest_text(payload: TextPayload) -> GraphObjectCollection:
    """Ingest raw text string and return graph objects."""
    try:
        graph_data = process_text(payload.raw_text)
        return graph_data
    except Exception as e:
        logger.error(f"Text processing failed: {e}")
        raise HTTPException(status_code=500, detail="Text processing failed")

@router.post("/document", response_model=GraphObjectCollection)
async def ingest_document(file: UploadFile = File(...)) -> GraphObjectCollection:
    """Ingest a document and return graph objects."""
    file_type = magic.from_buffer(await file.read(1024), mime=True)
    await file.seek(0)

    if file_type == "application/pdf":
        graph_data = process_pdf_file(await file.read())
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        graph_data = process_docx_file(await file.read())
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_type}")
    return graph_data
```

---

### **Milestone 4: MCP Server Implementation (Optional but Recommended)**

**Goal:** Implement a Model Context Protocol server to make the Graph Builder Service available to AI coding assistants (like Claude Code).

**Deliverables:**
1.  A new MCP server file (`mcp_server.py`) using the `mcp` SDK.
2.  Implementation of `tools` that mirror the API endpoints (e.g., `ingest_text_to_graph`).
3.  The MCP server connects to the FastAPI service internally or calls the processors directly.

**Techniques & Libraries:**
*   **MCP SDK:** `anthropic-mcp` (or the official SDK when available).
*   **SSE Client/Server:** For communication.

**Code Snippet (mcp_server.py - simplified):**
```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
import httpx

# Initialize MCP Server
app = Server("graph-builder")

@app.tool()
async def ingest_text_to_graph(raw_text: str) -> str:
    """
    Ingests raw text and converts it into a graph structure of nodes and relationships.
    Returns a JSON string ready for import into a graph database.
    """
    # Call the internal processor directly
    from app.processors.text_processor import process_text
    graph_data = process_text(raw_text)
    return graph_data.json() # Return as JSON string

    # Alternatively, call the HTTP API of your own service
    # async with httpx.AsyncClient() as client:
    #     response = await client.post("http://localhost:8000/ingest/text", json={"raw_text": raw_text})
    #     return response.text

if __name__ == "__main__":
    # Run the server using stdin/stdout
    stdio_server(app).run()
```

---

### **Milestone 5: Advanced Features & Production Readiness**

**Goal:** Harden the service for production use.

**Deliverables:**
1.  **Authentication & Authorization:** Add API key security to FastAPI endpoints.
2.  **Rate Limiting:** Implement limits to prevent abuse.
3.  **Async Processing:** For long-running jobs (e.g., processing a large book), implement a job queue system. Return a job ID immediately and provide a `/status/{job_id}` endpoint.
4.  **Enhanced Logging & Monitoring:** Integrate with **Prometheus** and **Grafana**.
5.  **Configuration:** Full environment-based configuration for all external services.
6.  **Testing:** A comprehensive test suite (pytest) for critical processors and the API.
7.  **Docker Compose:** File to spin up the service alongside Redis for the job queue.

**Techniques & Libraries:**
*   **FastAPI Middleware:** for auth and rate limiting.
*   **Celery:** For distributed task queues with Redis/RabbitMQ as a broker.
*   **Prometheus FastAPI Instrumentator:** for metrics.
*   **pytest:** for testing.
*   **Docker Compose:** for orchestration.

This roadmap provides a clear, phased approach for a coding agent to build a powerful and scalable graph builder service. Each milestone has a focused goal and recommends the best modern Python libraries to achieve it.