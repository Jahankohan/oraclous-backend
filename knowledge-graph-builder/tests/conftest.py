"""
Test configuration and shared fixtures for Knowledge Graph Builder tests.

This module provides common test fixtures, utilities, and configuration
for testing the knowledge graph builder service in Docker environment.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

try:
    import neo4j
    from neo4j import AsyncDriver

    _NEO4J_AVAILABLE = True
except ImportError:
    _NEO4J_AVAILABLE = False

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# Import application components lazily so unit tests can run without Neo4j.
# NOTE: `app.main` is intentionally NOT imported at module level here.
# Importing it at collection time causes SQLAlchemy to partially register
# models (including knowledge_graphs), and if the import chain then fails
# (e.g., due to missing services or reserved attribute names), Python removes
# the partially-loaded app.models.graph from sys.modules.  Any test module
# that subsequently imports the same module re-executes graph.py, triggering
# a double-registration error.  Deferring the import inside each fixture that
# actually needs `app` prevents this race entirely.
try:
    from app.core.config import settings
    from app.core.neo4j_client import neo4j_client
    from app.schemas.retriever_schemas import (
        HybridRetrieverConfig,
        RetrieverType,
        Text2CypherRetrieverConfig,
        VectorRetrieverConfig,
    )
    from app.services.retriever_factory import retriever_factory
    from app.services.schema_service import (
        GraphSchema,
        NodeSchema,
        RelationshipSchema,
        schema_manager,
    )

    _APP_AVAILABLE = True
except Exception:
    settings = None
    neo4j_client = None
    schema_manager = None
    GraphSchema = NodeSchema = RelationshipSchema = None
    retriever_factory = None
    RetrieverType = VectorRetrieverConfig = Text2CypherRetrieverConfig = (
        HybridRetrieverConfig
    ) = None
    _APP_AVAILABLE = False


# ==================== TEST CONFIGURATION ====================

# Test environment settings
TEST_NEO4J_URI = os.getenv("TEST_NEO4J_URI", "neo4j://neo4j:7687")
TEST_NEO4J_USERNAME = os.getenv("TEST_NEO4J_USERNAME", "neo4j")
TEST_NEO4J_PASSWORD = os.getenv("TEST_NEO4J_PASSWORD", "password")
TEST_GRAPH_ID = "test-graph-12345"
TEST_OPENAI_API_KEY = os.getenv("TEST_OPENAI_API_KEY", "test-key-for-mocking")


# ==================== PYTEST FIXTURES ====================


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI application."""
    from app.main import app

    return TestClient(app)


@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI application."""
    from app.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        yield client


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("app.core.config.settings") as mock:
        mock.NEO4J_URI = TEST_NEO4J_URI
        mock.NEO4J_USERNAME = TEST_NEO4J_USERNAME
        mock.NEO4J_PASSWORD = TEST_NEO4J_PASSWORD
        mock.NEO4J_DATABASE = "neo4j"
        mock.OPENAI_API_KEY = TEST_OPENAI_API_KEY
        mock.SERVICE_NAME = "test-knowledge-graph-builder"
        yield mock


# ==================== NEO4J TEST FIXTURES ====================


@pytest_asyncio.fixture
async def neo4j_test_driver():
    """Create a test Neo4j driver for integration tests."""
    driver = neo4j.AsyncGraphDatabase.driver(
        TEST_NEO4J_URI, auth=(TEST_NEO4J_USERNAME, TEST_NEO4J_PASSWORD)
    )

    try:
        # Test connection
        await driver.verify_connectivity()
        yield driver
    finally:
        await driver.close()


@pytest_asyncio.fixture
async def clean_test_graph(neo4j_test_driver: AsyncDriver):
    """Clean test graph before and after test."""
    # Clean before test
    await _clean_test_data(neo4j_test_driver)

    yield

    # Clean after test
    await _clean_test_data(neo4j_test_driver)


async def _clean_test_data(driver: AsyncDriver):
    """Helper to clean test data from Neo4j."""
    async with driver.session() as session:
        # Delete all test data
        await session.run(f"MATCH (n {{graph_id: '{TEST_GRAPH_ID}'}}) DETACH DELETE n")
        await session.run(f"MATCH ()-[r {{graph_id: '{TEST_GRAPH_ID}'}}]->() DELETE r")


@pytest_asyncio.fixture
async def sample_test_data(neo4j_test_driver: AsyncDriver, clean_test_graph: None):
    """Create sample test data in Neo4j."""
    async with neo4j_test_driver.session() as session:
        # Create sample entities
        await session.run(
            """
            CREATE (c:Company {name: 'TechNova Corp', type: 'Technology', graph_id: $graph_id})
            CREATE (p:Person {name: 'John Doe', role: 'CEO', graph_id: $graph_id})
            CREATE (l:Location {name: 'San Francisco', type: 'City', graph_id: $graph_id})
            CREATE (d:Document {title: 'Company Overview', path: '/docs/overview.pdf', graph_id: $graph_id})
            CREATE (ch:Chunk {text: 'TechNova Corp is a leading technology company...', graph_id: $graph_id})

            CREATE (p)-[:CEO_OF]->(c)
            CREATE (c)-[:LOCATED_IN]->(l)
            CREATE (ch)-[:FROM_DOCUMENT]->(d)
            CREATE (c)-[:FROM_CHUNK]->(ch)
        """,
            graph_id=TEST_GRAPH_ID,
        )

    yield
    # Cleanup handled by clean_test_graph


# ==================== SCHEMA SERVICE FIXTURES ====================


@pytest.fixture
def mock_schema_manager():
    """Mock schema manager for unit tests."""
    manager = MagicMock()

    # Mock schema data
    mock_schema = GraphSchema(
        graph_id=TEST_GRAPH_ID,
        nodes={
            "Company": NodeSchema(
                label="Company",
                properties={"name": "string", "type": "string", "graph_id": "string"},
                sample_count=1,
                indexes=[],
            ),
            "Person": NodeSchema(
                label="Person",
                properties={"name": "string", "role": "string", "graph_id": "string"},
                sample_count=1,
                indexes=[],
            ),
        },
        relationships={
            "CEO_OF": RelationshipSchema(
                type="CEO_OF",
                properties={"graph_id": "string"},
                start_labels={"Person"},
                end_labels={"Company"},
                sample_count=1,
            )
        },
        constraints=[],
        indexes=[],
        last_updated=datetime.now(UTC),
        schema_version="test_v1",
    )

    manager.extract_schema = AsyncMock(return_value=mock_schema)
    manager.format_schema_for_text2cypher = MagicMock(return_value="Mock schema string")
    manager.get_cache_details = MagicMock(return_value={TEST_GRAPH_ID: mock_schema})
    manager.clear_cache = MagicMock()

    return manager


@pytest.fixture
def sample_node_schemas() -> dict[str, NodeSchema]:
    """Sample node schemas for testing."""
    return {
        "Company": NodeSchema(
            label="Company",
            properties={
                "name": "string",
                "type": "string",
                "founded": "integer",
                "graph_id": "string",
            },
            sample_count=5,
            indexes=["name"],
        ),
        "Person": NodeSchema(
            label="Person",
            properties={
                "name": "string",
                "role": "string",
                "age": "integer",
                "graph_id": "string",
            },
            sample_count=3,
            indexes=["name", "role"],
        ),
    }


@pytest.fixture
def sample_relationship_schemas() -> dict[str, RelationshipSchema]:
    """Sample relationship schemas for testing."""
    return {
        "CEO_OF": RelationshipSchema(
            type="CEO_OF",
            properties={"since": "date", "graph_id": "string"},
            start_labels={"Person"},
            end_labels={"Company"},
            sample_count=2,
        ),
        "WORKS_AT": RelationshipSchema(
            type="WORKS_AT",
            properties={"department": "string", "graph_id": "string"},
            start_labels={"Person"},
            end_labels={"Company"},
            sample_count=5,
        ),
    }


# ==================== RETRIEVER FIXTURES ====================


@pytest.fixture
def mock_retriever_factory():
    """Mock retriever factory for unit tests."""
    factory = MagicMock()

    # Mock retrievers
    mock_retriever = MagicMock()
    mock_retriever.search = AsyncMock(
        return_value=[
            {"text": "Test result 1", "score": 0.9},
            {"text": "Test result 2", "score": 0.8},
        ]
    )

    factory.create_vector_retriever = AsyncMock(return_value=mock_retriever)
    factory.create_text2cypher_retriever = AsyncMock(return_value=mock_retriever)
    factory.create_hybrid_retriever = AsyncMock(return_value=mock_retriever)
    factory.create_retriever = AsyncMock(return_value=mock_retriever)

    return factory


@pytest.fixture
def sample_retriever_configs() -> dict[
    str,
    VectorRetrieverConfig | Text2CypherRetrieverConfig | HybridRetrieverConfig,
]:
    """Sample retriever configurations for testing."""
    return {
        "vector": VectorRetrieverConfig(index_name="test-vector-index"),
        "text2cypher": Text2CypherRetrieverConfig(
            neo4j_schema="Mock schema",
            llm_params={"model": "gpt-4o", "temperature": 0.1},
        ),
        "hybrid": HybridRetrieverConfig(
            vector_index_name="test-vector-index",
            fulltext_index_name="test-fulltext-index",
        ),
    }


# ==================== API TEST FIXTURES ====================


@pytest.fixture
def sample_chat_request():
    """Sample chat request for API testing."""
    return {
        "query": "Tell me about TechNova Corporation",
        "graph_id": TEST_GRAPH_ID,
        "mode": "simple",
        "return_context": True,
        "max_results": 5,
    }


@pytest.fixture
def sample_schema_refresh_request():
    """Sample schema refresh request."""
    return {"graph_id": TEST_GRAPH_ID, "force_refresh": True}


# ==================== MOCK LLM FIXTURES ====================


@pytest.fixture
def mock_openai_llm():
    """Mock OpenAI LLM for testing."""
    with patch("neo4j_graphrag.llm.OpenAILLM") as mock:
        llm_instance = MagicMock()
        llm_instance.invoke = AsyncMock(return_value="Mock LLM response")
        mock.return_value = llm_instance
        yield llm_instance


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings for testing."""
    with patch("neo4j_graphrag.embeddings.OpenAIEmbeddings") as mock:
        embeddings_instance = MagicMock()
        embeddings_instance.embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embeddings_instance.embed_documents = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock.return_value = embeddings_instance
        yield embeddings_instance


# ==================== UTILITY FIXTURES ====================


@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent testing."""
    fixed_time = datetime(2025, 9, 4, 12, 0, 0, tzinfo=UTC)
    with patch("app.services.schema_service.datetime") as mock_dt:
        mock_dt.now.return_value = fixed_time
        mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
        yield fixed_time


@pytest.fixture
def test_graph_id():
    """Test graph ID for consistent testing."""
    return TEST_GRAPH_ID


# ==================== HELPER FUNCTIONS ====================


def assert_valid_schema_response(response_data: dict[str, Any]):
    """Assert that a schema response has the expected structure."""
    required_fields = [
        "graph_id",
        "schema_version",
        "last_updated",
        "nodes",
        "relationships",
    ]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"

    assert isinstance(response_data["nodes"], dict)
    assert isinstance(response_data["relationships"], dict)


def assert_valid_chat_response(response_data: dict[str, Any]):
    """Assert that a chat response has the expected structure."""
    required_fields = ["response", "graph_id", "retriever_type"]
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"


async def wait_for_neo4j_ready(driver, max_retries: int = 30):
    """Wait for Neo4j to be ready for testing."""
    for i in range(max_retries):
        try:
            await driver.verify_connectivity()
            return True
        except Exception:
            if i == max_retries - 1:
                raise
            await asyncio.sleep(1)
    return False
