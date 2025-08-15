# Neo4j LLM Graph Builder - Clean FastAPI Backend

A restructured and cleaner implementation of the Neo4j LLM Graph Builder backend using FastAPI.

Original Implementation: https://github.com/neo4j-labs/llm-graph-builder

## Features

- **Clean Architecture**: Well-organized code structure with clear separation of concerns
- **Modern FastAPI**: Latest FastAPI with async/await support and automatic OpenAPI docs
- **Multiple LLM Support**: OpenAI, Gemini, Anthropic, and more
- **Flexible Document Sources**: Local files, S3, GCS, YouTube, Wikipedia, web pages
- **Advanced Graph Processing**: Entity extraction, relationship mapping, post-processing
- **Real-time Processing**: Server-sent events for live progress updates  
- **Chat Interface**: Multiple retrieval modes (vector, graph, hybrid)
- **Graph Management**: Duplicate detection, schema suggestions, visualization

## Quick Start

### Using Docker Compose (Recommended)

1. Clone the repository and set up environment:
```bash
git clone <repository-url>
cd neo4j-llm-graph-builder-backend
cp .env.example .env
# Edit .env with your API keys and configuration
```

2. Start services:
```bash
docker-compose up -d
```

3. Access the application:
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs  
- Neo4j Browser: http://localhost:7474

### Manual Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Start Neo4j database (version 5.23+ with APOC plugin)

3. Set up environment variables:
```bash
export NEO4J_URI=neo4j://localhost:7687
export NEO4J_USERNAME=neo4j
export NEO4J_PASSWORD=your_password
export OPENAI_API_KEY=your_openai_key
```

4. Run the application:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI application setup
│   ├── config/
│   │   └── settings.py      # Configuration and settings
│   ├── core/
│   │   ├── neo4j_client.py  # Neo4j database client
│   │   └── exceptions.py    # Custom exceptions
│   ├── models/
│   │   ├── requests.py      # Pydantic request models
│   │   └── responses.py     # Pydantic response models
│   ├── services/
│   │   ├── document_service.py    # Document handling
│   │   ├── extraction_service.py  # Graph extraction
│   │   ├── graph_service.py       # Graph operations
│   │   ├── embedding_service.py   # Embeddings generation
│   │   └── chat_service.py        # Chat functionality
│   ├── routers/
│   │   ├── infrastructure.py # Database and system endpoints
│   │   ├── documents.py      # Document processing endpoints
│   │   ├── graph.py          # Graph management endpoints
│   │   └── chat.py           # Chat endpoints
│   └── utils/
│       ├── llm_clients.py    # LLM client factory
│       └── file_handlers.py  # File upload/storage utilities
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## API Endpoints

### Infrastructure
- `POST /api/v1/connect` - Connect to Neo4j database
- `GET /api/v1/schema` - Get database schema
- `POST /api/v1/drop_and_create_vector_index` - Recreate vector index
- `DELETE /api/v1/delete_document_and_entities` - Delete documents

### Document Processing  
- `POST /api/v1/upload` - Upload files
- `POST /api/v1/url/scan` - Scan URL sources
- `POST /api/v1/extract` - Extract knowledge graph
- `GET /api/v1/sources_list` - List all documents
- `POST /api/v1/post_processing` - Run post-processing

### Graph Management
- `POST /api/v1/graph_query` - Get graph visualization
- `GET /api/v1/get_neighbours/{node_id}` - Get node neighbors  
- `POST /api/v1/populate_graph_schema` - Generate schema suggestions
- `GET /api/v1/get_duplicate_nodes_list` - Find duplicate nodes
- `POST /api/v1/merge_duplicate_nodes` - Merge duplicates

### Chat Interface
- `POST /api/v1/chat_bot` - Chat with knowledge graph
- `POST /api/v1/clear_chat_bot` - Clear chat history
- `GET /api/v1/chat_history/{session_id}` - Get chat history

## Configuration

Key environment variables:

```bash
# Neo4j Database
NEO4J_URI=neo4j://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# LLM API Keys  
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GEMINI_API_KEY=your_key

# Processing Settings
MAX_TOKEN_CHUNK_SIZE=10000
DUPLICATE_SCORE_VALUE=0.97
KNN_MIN_SCORE=0.94

# Storage (Optional)
BUCKET=your_gcs_bucket
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

## Usage Examples

### 1. Connect to Database
```python
import requests

response = requests.post("http://localhost:8000/api/v1/connect", data={
    "uri": "neo4j://localhost:7687",
    "username": "neo4j", 
    "password": "password",
    "database": "neo4j"
})
print(response.json())
```

### 2. Upload and Process Documents
```python
# Upload files
files = [("files", open("document.pdf", "rb"))]
response = requests.post("http://localhost:8000/api/v1/upload", files=files)

# Extract knowledge graph
response = requests.post("http://localhost:8000/api/v1/extract", json={
    "file_names": ["document.pdf"],
    "model": "gpt-4o-mini",
    "enable_schema": true
})
```

### 3. Chat with Knowledge Graph
```python
response = requests.post("http://localhost:8000/api/v1/chat_bot", json={
    "message": "What are the main topics in the documents?",
    "mode": "graph_vector",
    "file_names": ["document.pdf"]
})
print(response.json()["message"])
```

### 4. Get Graph Visualization
```python
response = requests.post("http://localhost:8000/api/v1/graph_query", json={
    "file_names": ["document.pdf"],
    "limit": 50
})
graph_data = response.json()
```

## Key Improvements Over Original

### Architecture
- **Modular Design**: Clear separation between routers, services, and models
- **Dependency Injection**: Proper FastAPI dependency management
- **Type Safety**: Full Pydantic model validation
- **Error Handling**: Centralized exception handling with proper HTTP status codes

### Code Quality
- **Async/Await**: Proper async implementation throughout
- **Logging**: Structured logging with proper levels
- **Documentation**: Comprehensive docstrings and OpenAPI specs
- **Testing Ready**: Structure supports easy unit and integration testing

### Performance
- **Connection Pooling**: Efficient Neo4j connection management
- **Batch Processing**: Optimized chunk and embedding processing
- **Streaming Responses**: Real-time progress updates via Server-Sent Events
- **Caching**: LLM client caching and reuse

### Maintainability
- **Configuration Management**: Centralized settings with environment variable support
- **Service Layer**: Business logic separated from API endpoints
- **Consistent Patterns**: Uniform error handling, response formats, and naming
- **Extension Points**: Easy to add new LLM providers, document sources, or chat modes

## Development

### Running in Development Mode
```bash
# Install development dependencies
pip install -r requirements.txt

# Run with auto-reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests (when implemented)
pytest tests/
```

### Adding New Features

#### New LLM Provider
1. Add provider enum to `app/config/settings.py`
2. Implement client in `app/utils/llm_clients.py` 
3. Add API key configuration to settings

#### New Document Source
1. Add source enum to `app/models/requests.py`
2. Implement loader in `app/services/document_service.py`
3. Add scanning logic for the source type

#### New Chat Mode
1. Add mode enum to `app/models/requests.py`
2. Implement retriever in `app/services/chat_service.py`
3. Add any required Neo4j queries or processing

### Database Schema

The application creates the following node types:

```cypher
// Core document structure
(:Document)-[:HAS_CHUNK]->(:Chunk)
(:Document)-[:FIRST_CHUNK]->(:Chunk)
(:Chunk)-[:NEXT_CHUNK]->(:Chunk)

// Extracted entities and relationships  
(:Chunk)-[:HAS_ENTITY]->(:Entity)
(:Entity)-[various relationship types]->(:Entity)

// Similarities and communities
(:Chunk)-[:SIMILAR_TO]->(:Chunk)
(:Entity)-[:SIMILAR_ENTITY]->(:Entity)

// Chat sessions
(:Session)-[:HAS_CONVERSATION]->(:Conversation)
```

### Monitoring and Observability

The application includes:
- Health check endpoint at `/health`
- Structured logging with correlation IDs
- Processing progress tracking
- Performance metrics (response times)
- Error tracking and reporting

### Security Considerations

- Input validation via Pydantic models
- SQL injection prevention through parameterized queries
- API key management through environment variables
- CORS configuration for cross-origin requests
- Rate limiting support (configurable)

### Deployment

#### Production Deployment
```bash
# Build and push Docker image
docker build -t neo4j-llm-backend .
docker push your-registry/neo4j-llm-backend

# Deploy with orchestration platform
kubectl apply -f k8s/
# or
docker-compose -f docker-compose.prod.yml up -d
```

#### Environment-Specific Configuration
- Development: `.env.dev`  
- Staging: `.env.staging`
- Production: `.env.prod`

#### Health Checks and Monitoring
```bash
# Health check
curl http://localhost:8000/health

# Metrics endpoint (if implemented)
curl http://localhost:8000/metrics

# API documentation
curl http://localhost:8000/docs
```

## Troubleshooting

### Common Issues

1. **Neo4j Connection Failed**
   - Check Neo4j is running and accessible
   - Verify URI, username, and password
   - Ensure APOC plugin is installed

2. **LLM API Errors**
   - Verify API keys are set correctly
   - Check API rate limits and quotas
   - Ensure model names are correct

3. **Memory Issues**
   - Reduce chunk size or batch size
   - Monitor memory usage during processing
   - Consider processing files individually

4. **Vector Index Issues**
   - Drop and recreate vector index
   - Check embedding dimensions match
   - Verify Neo4j version supports vector indexes

### Performance Tuning

- Adjust `MAX_TOKEN_CHUNK_SIZE` based on your LLM
- Tune `NUMBER_OF_CHUNKS_TO_COMBINE` for batch processing
- Configure appropriate `KNN_MIN_SCORE` for similarity
- Use appropriate embedding model for your use case

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with proper tests
4. Submit a pull request

## License

This project maintains the same license as the original Neo4j LLM Graph Builder.

## Support

For issues and questions:
- Check the troubleshooting section above
- Review the original project documentation
- Open an issue in the repository
- Check Neo4j community forums for database-related questions
