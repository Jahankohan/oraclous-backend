# Testing Guide

This directory contains comprehensive tests for the Knowledge Graph Builder service with Docker support.

## 🏗️ Test Structure

```
tests/
├── conftest.py              # Test configuration and fixtures
├── unit/                    # Unit tests (isolated, fast)
│   ├── test_schema_service.py
│   └── test_retriever_factory.py
├── integration/             # Integration tests (with dependencies)
│   ├── test_chat_api.py
│   └── test_schema_api.py
└── fixtures/                # Test data and utilities
```

## 🚀 Running Tests

### Quick Start

Run all tests in Docker:

```bash
./test.sh
```

### Test Types

Run specific test types:

```bash
# Unit tests only (fast, no external dependencies)
./test.sh --type unit

# Integration tests (with Neo4j, APIs)
./test.sh --type integration

# Schema management tests
./test.sh --type schema

# API tests
./test.sh --type api

# Verbose output
./test.sh --type unit --verbose
```

### Manual Docker Testing

If you prefer manual control:

```bash
# Start services
docker-compose up -d

# Wait for services to be ready (Neo4j takes ~30 seconds)

# Run tests
docker-compose exec knowledge-graph-builder python -m pytest tests/ -v

# Run specific test file
docker-compose exec knowledge-graph-builder python -m pytest tests/unit/test_schema_service.py -v

# Run with coverage
docker-compose exec knowledge-graph-builder python -m pytest tests/ --cov=app --cov-report=term-missing

# Cleanup
docker-compose down -v
```

## 🧪 Test Categories

### Unit Tests (`-m unit`)

- **test_schema_service.py**: Schema extraction, caching, formatting
- **test_retriever_factory.py**: All retriever types creation and configuration

### Integration Tests (`-m integration`)

- **test_chat_api.py**: Complete chat API with all retriever modes
- **test_schema_api.py**: Schema management API endpoints

### Test Markers

- `@pytest.mark.unit`: Fast isolated tests
- `@pytest.mark.integration`: Tests requiring services
- `@pytest.mark.api`: API endpoint tests
- `@pytest.mark.schema`: Schema management tests
- `@pytest.mark.docker`: Docker-specific tests
- `@pytest.mark.slow`: Long-running tests

## 🔧 Test Configuration

### Environment Variables

Tests use these environment variables (automatically set in Docker):

```bash
TEST_NEO4J_URI=neo4j://neo4j:7687
TEST_NEO4J_USERNAME=neo4j
TEST_NEO4J_PASSWORD=password
TEST_OPENAI_API_KEY=test-key-for-mocking
```

### Pytest Configuration

See `pytest.ini` for complete configuration:

- Async test support
- Marker definitions
- Output formatting
- Coverage settings

## 🏃‍♂️ Test Examples

### Running Specific Tests

```bash
# Test schema service caching
docker-compose exec knowledge-graph-builder python -m pytest tests/unit/test_schema_service.py::TestNeo4jSchemaManager::test_extract_schema_with_cache -v

# Test all chat API functionality
docker-compose exec knowledge-graph-builder python -m pytest tests/integration/test_chat_api.py -v

# Test schema API with error handling
docker-compose exec knowledge-graph-builder python -m pytest tests/integration/test_schema_api.py::TestSchemaAPIIntegration::test_schema_api_error_handling -v
```

### Test Data

Tests use:

- **Mock data**: For unit tests (fast, isolated)
- **Sample Neo4j data**: For integration tests (realistic)
- **Fixtures**: Reusable test components in `conftest.py`

## 🐳 Docker Integration

### Service Dependencies

Tests require these Docker services:

- **neo4j**: Graph database for schema extraction
- **postgres**: Metadata storage
- **redis**: Caching layer
- **knowledge-graph-builder**: Main service

### Test Database

- Tests use isolated test data with `graph_id="test-graph-12345"`
- Automatic cleanup before and after tests
- No interference with development data

## 📊 Coverage

Generate coverage reports:

```bash
# Basic coverage
./test.sh --type all

# Detailed HTML coverage report
docker-compose exec knowledge-graph-builder python -m pytest tests/ --cov=app --cov-report=html
```

Coverage includes:

- Schema service functionality
- Retriever factory patterns
- API endpoint behaviors
- Error handling paths

## 🔍 Debugging Tests

### Verbose Output

```bash
./test.sh --verbose
```

### Test Specific Components

```bash
# Debug schema extraction
docker-compose exec knowledge-graph-builder python -m pytest tests/unit/test_schema_service.py::TestNeo4jSchemaManager::test_extract_node_schemas -v -s

# Debug API responses
docker-compose exec knowledge-graph-builder python -m pytest tests/integration/test_chat_api.py::TestChatAPIIntegration::test_chat_response_format -v -s
```

### Service Logs

```bash
# View service logs
docker-compose logs knowledge-graph-builder
docker-compose logs neo4j
```

## 🚨 Troubleshooting

### Common Issues

**Neo4j not ready:**

```bash
# Wait longer for Neo4j initialization
docker-compose logs neo4j
# Look for "Started" message
```

**Import errors:**

```bash
# Ensure you're in the knowledge-graph-builder directory
cd knowledge-graph-builder
./test.sh
```

**Permission errors:**

```bash
# Make test script executable
chmod +x test.sh
```

**Port conflicts:**

```bash
# Check if ports are in use
docker-compose down -v
./test.sh
```

### Test Development

Adding new tests:

1. Create test file in appropriate directory (`unit/` or `integration/`)
2. Add appropriate markers (`@pytest.mark.unit`, etc.)
3. Use fixtures from `conftest.py`
4. Follow naming convention: `test_*.py`

## 📋 Test Checklist

Before committing:

- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] New features have test coverage
- [ ] Error cases are tested
- [ ] Documentation is updated

## 🎯 Performance

- **Unit tests**: ~30 seconds
- **Integration tests**: ~2-3 minutes (includes service startup)
- **Full test suite**: ~3-5 minutes

Optimize by running unit tests first during development, then integration tests before commits.
