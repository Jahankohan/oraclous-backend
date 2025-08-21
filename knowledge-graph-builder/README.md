# Knowledge Graph Builder Service

A FastAPI-based microservice for building and querying knowledge graphs from unstructured data, integrated with the Oraclous ecosystem.

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker & Docker Compose
- Neo4j database
- PostgreSQL database

### Setup

1. **Clone and navigate to the service directory**
   ```bash
   cd knowledge-graph-builder
   ```

2. **Run the setup script**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Update environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your actual configuration
   ```

4. **Start with Docker Compose** (recommended)
   ```bash
   docker-compose up -d
   ```

   Or **run locally**:
   ```bash
   chmod +x scripts/dev.sh
   ./scripts/dev.sh
   ```

### Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload --port 8003

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 📊 Database Setup

### PostgreSQL Migration

```bash
# Create initial migration
python create_initial_migration.py

# Apply migrations
alembic upgrade head

# Create new migration after model changes
alembic revision --autogenerate -m "Description of changes"
```

### Neo4j Setup

The service automatically connects to Neo4j and creates necessary constraints and indexes.

## 🔗 API Endpoints

### Health Check
- `GET /api/v1/health` - Service health status

### Graph Management
- `POST /api/v1/graphs` - Create new graph
- `GET /api/v1/graphs` - List user graphs
- `GET /api/v1/graphs/{id}` - Get graph details
- `PUT /api/v1/graphs/{id}` - Update graph
- `DELETE /api/v1/graphs/{id}` - Delete graph

### Data Ingestion
- `POST /api/v1/graphs/{id}/ingest` - Ingest data into graph
- `GET /api/v1/graphs/{id}/jobs` - List ingestion jobs

## 🏗️ Architecture

```
knowledge-graph-builder/
├── app/
│   ├── api/v1/endpoints/     # API route handlers
│   ├── core/                 # Configuration, database, logging
│   ├── models/               # SQLAlchemy models
│   ├── schemas/              # Pydantic schemas
│   ├── services/             # Business logic services
│   └── main.py              # FastAPI application
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
├── alembic/                 # Database migrations
└── requirements.txt         # Python dependencies
```

## 🔧 Configuration

Key environment variables:

```bash
# Service
SERVICE_NAME=knowledge-graph-builder
SERVICE_URL=http://localhost:8003

# Databases
NEO4J_URI=bolt://neo4j:7687
POSTGRES_URL=postgresql+asyncpg://user:pass@host/db

# External Services
AUTH_SERVICE_URL=http://auth-service:8000
CREDENTIAL_BROKER_URL=http://credential-broker:8002

# Security
INTERNAL_SERVICE_KEY=your-internal-key
JWT_SECRET_KEY=your-jwt-secret
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_graphs.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

## 📈 Monitoring

- Health endpoint: `/api/v1/health`
- Logs are written to `logs/` directory
- Metrics available when `ENABLE_METRICS=true`

## 🤝 Integration with Oraclous Ecosystem

This service integrates with:
- **Auth Service**: User authentication and authorization
- **Credential Broker**: OAuth token management
- **Core Service**: Tool registration and orchestration

## 📝 Development Notes

- This implements **Checkpoint 1** of the Knowledge Graph Builder integration
- Next steps: Entity extraction (Checkpoint 2)
- Uses async/await throughout for optimal performance
- Follows FastAPI best practices and patterns
- Comprehensive error handling and logging
