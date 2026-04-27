"""
Regression tests for ORA-103: vector search embed_query call contract.

Verifies that the vector search service:
1. Calls embed_query with the user's query_text argument (not a modified value)
2. Passes the resulting vector to the Neo4j query (not None or empty)
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from app.services.retriever_service import RetrievalService

GRAPH_ID = uuid.UUID("12345678-1234-5678-1234-567812345678")
QUERY_TEXT = "what are the key concepts in this knowledge graph?"
EXPECTED_VECTOR = [0.1, 0.2, 0.3, 0.4, 0.5]


class _FakeVectorRetriever:
    """
    Test double for neo4j_graphrag VectorRetriever.

    Faithfully simulates the critical behaviour of the real class:
      - calls embedder.embed_query(query_text) when query_text is provided
      - passes the resulting vector to driver.execute_query as query_vector

    This lets unit tests assert on embed_query + Neo4j call without requiring
    a live Neo4j connection or a real neo4j.Driver instance.
    """

    VERIFY_NEO4J_VERSION = False

    def __init__(
        self, driver, index_name, embedder=None, return_properties=None, **kwargs
    ):
        self.driver = driver
        self.index_name = index_name
        self.embedder = embedder
        self.return_properties = return_properties or []

    def get_search_results(self, query_text=None, query_vector=None, top_k=5, **kwargs):
        if query_text is not None and self.embedder is not None:
            query_vector = self.embedder.embed_query(query_text)
        self.driver.execute_query(
            "CALL db.index.vector.queryNodes($index, $k, $vector) YIELD node, score",
            {"query_vector": query_vector, "index": self.index_name, "k": top_k},
        )
        return MagicMock(items=[])


class TestVectorSearchRegressionORA103:
    """Regression tests for ORA-103: vector search embed_query contract."""

    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = EXPECTED_VECTOR
        return embedder

    @pytest.fixture
    def mock_driver(self):
        return MagicMock()

    @pytest.mark.unit
    async def test_vector_search_calls_embed_query_with_query_text(
        self, mock_embedder, mock_driver
    ):
        """Verify similarity_search_entities calls embed_query with the query_text argument.

        Regression for ORA-103: ensures embed_query receives the user's actual query
        text, not an empty string, None, or any pre-transformed value.
        """
        with (
            patch(
                "app.components.multi_tenant_components.VectorRetriever",
                _FakeVectorRetriever,
            ),
            patch(
                "neo4j_graphrag.retrievers.base.get_version",
                return_value=((5, 23, 0), False, False),
            ),
        ):
            service = RetrievalService(driver=mock_driver, embedder=mock_embedder)
            await service.similarity_search_entities(QUERY_TEXT, GRAPH_ID)

        mock_embedder.embed_query.assert_called_once_with(QUERY_TEXT)

    @pytest.mark.unit
    async def test_vector_search_passes_real_vector_to_neo4j(
        self, mock_embedder, mock_driver
    ):
        """Verify the embedded vector is passed to the Neo4j query, not None or empty.

        Regression for ORA-103: ensures the vector returned by embed_query is actually
        forwarded to the Neo4j vector index query rather than being discarded.
        """
        with (
            patch(
                "app.components.multi_tenant_components.VectorRetriever",
                _FakeVectorRetriever,
            ),
            patch(
                "neo4j_graphrag.retrievers.base.get_version",
                return_value=((5, 23, 0), False, False),
            ),
        ):
            service = RetrievalService(driver=mock_driver, embedder=mock_embedder)
            await service.similarity_search_entities(QUERY_TEXT, GRAPH_ID)

        assert mock_driver.execute_query.called, "No Neo4j execute_query call was made"

        # The EXPECTED_VECTOR must appear as query_vector in at least one call
        vector_found = any(
            isinstance(call_args.args[1], dict)
            and call_args.args[1].get("query_vector") == EXPECTED_VECTOR
            for call_args in mock_driver.execute_query.call_args_list
        )
        assert vector_found, (
            f"Expected query_vector={EXPECTED_VECTOR} was not passed to Neo4j. "
            f"Actual execute_query calls: {mock_driver.execute_query.call_args_list}"
        )
