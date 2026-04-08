"""
Tests for RetrieverFactory service.

Tests the factory for creating and managing all types of Neo4j GraphRAG retrievers.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from app.schemas.retriever_schemas import (
    HybridRetrieverConfig,
    RetrieverConfig,
    RetrieverType,
    Text2CypherRetrieverConfig,
    VectorRetrieverConfig,
)
from app.services.retriever_factory import RetrieverFactory


class TestRetrieverFactory:
    """Test cases for RetrieverFactory."""

    @pytest.fixture
    def factory(self):
        """Create RetrieverFactory instance for testing."""
        return RetrieverFactory()

    @pytest.fixture
    def mock_neo4j_client(self):
        """Mock Neo4j client."""
        with patch("app.services.retriever_factory.neo4j_client") as mock:
            mock.sync_driver = MagicMock()
            mock.async_driver = MagicMock()
            mock.connect_async = AsyncMock()
            mock.connect_sync = MagicMock()
            yield mock

    @pytest.fixture
    def sample_vector_config(self):
        """Sample vector retriever configuration."""
        return VectorRetrieverConfig(
            index_name="chunk_vector_index", return_properties=["text", "page", "url"]
        )

    @pytest.fixture
    def sample_text2cypher_config(self):
        """Sample Text2Cypher retriever configuration."""
        return Text2CypherRetrieverConfig(
            neo4j_schema=None,
            examples=["Example query"],
            custom_prompt="Custom prompt",
            llm_params={"model": "gpt-4o", "temperature": 0.1},
        )

    @pytest.fixture
    def sample_hybrid_config(self):
        """Sample hybrid retriever configuration."""
        return HybridRetrieverConfig(
            vector_index_name="chunk_vector_index",
            fulltext_index_name="chunk_fulltext_index",
            return_properties=["text", "page", "url"],
        )

    @pytest.mark.unit
    @pytest.mark.retriever
    def test_get_llm_with_default_params(self, factory):
        """Test LLM creation with default parameters."""
        with patch("app.services.retriever_factory.OpenAILLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm

            llm = factory._get_llm()

            assert llm is mock_llm
            mock_llm_class.assert_called_once()

            call_kwargs = mock_llm_class.call_args[1]
            assert call_kwargs["model_name"] == "gpt-4o"
            assert call_kwargs["model_params"]["temperature"] == 0.1
            assert call_kwargs["model_params"]["max_tokens"] == 3000

    @pytest.mark.unit
    @pytest.mark.retriever
    def test_get_llm_with_custom_params(self, factory):
        """Test LLM creation with custom parameters."""
        with patch("app.services.retriever_factory.OpenAILLM") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm_class.return_value = mock_llm

            llm = factory._get_llm(
                model="gpt-3.5-turbo", temperature=0.7, max_tokens=2000
            )

            assert llm is mock_llm
            mock_llm_class.assert_called_once()

            call_kwargs = mock_llm_class.call_args[1]
            assert call_kwargs["model_name"] == "gpt-3.5-turbo"
            assert call_kwargs["model_params"]["temperature"] == 0.7
            assert call_kwargs["model_params"]["max_tokens"] == 2000

    @pytest.mark.unit
    @pytest.mark.retriever
    def test_get_embedder_with_default_params(self, factory):
        """Test embedder creation with default parameters."""
        with patch(
            "app.services.retriever_factory.OpenAIEmbeddings"
        ) as mock_embeddings_class:
            mock_embeddings = MagicMock()
            mock_embeddings_class.return_value = mock_embeddings

            embeddings = factory._get_embedder()

            assert embeddings is mock_embeddings
            mock_embeddings_class.assert_called_once()

            call_kwargs = mock_embeddings_class.call_args[1]
            assert call_kwargs["model"] == "text-embedding-3-large"

    @pytest.mark.unit
    @pytest.mark.retriever
    def test_get_embedder_with_custom_params(self, factory):
        """Test embedder creation with custom parameters."""
        with patch(
            "app.services.retriever_factory.OpenAIEmbeddings"
        ) as mock_embeddings_class:
            mock_embeddings = MagicMock()
            mock_embeddings_class.return_value = mock_embeddings

            embeddings = factory._get_embedder(model="text-embedding-3-small")

            assert embeddings is mock_embeddings
            mock_embeddings_class.assert_called_once()

            call_kwargs = mock_embeddings_class.call_args[1]
            assert call_kwargs["model"] == "text-embedding-3-small"

    @pytest.mark.unit
    @pytest.mark.retriever
    async def test_ensure_connections(self, factory, mock_neo4j_client):
        """Test connection initialization."""
        await factory._ensure_connections()

        mock_neo4j_client.connect_async.assert_called_once()
        mock_neo4j_client.connect_sync.assert_called_once()
        assert factory._initialized is True

    @pytest.mark.unit
    @pytest.mark.retriever
    async def test_create_vector_retriever(
        self, factory, mock_neo4j_client, sample_vector_config
    ):
        """Test vector retriever creation."""
        with patch(
            "app.services.retriever_factory.VectorRetriever"
        ) as mock_retriever_class:
            mock_retriever = MagicMock()
            mock_retriever_class.return_value = mock_retriever

            with patch.object(factory, "_get_embedder") as mock_get_embedder:
                mock_embedder = MagicMock()
                mock_get_embedder.return_value = mock_embedder

                retriever = await factory.create_vector_retriever(
                    sample_vector_config, "test-graph"
                )

                assert retriever is mock_retriever
                mock_get_embedder.assert_called_once()
                mock_retriever_class.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.retriever
    async def test_create_text2cypher_retriever(
        self, factory, mock_neo4j_client, sample_text2cypher_config
    ):
        """Test Text2Cypher retriever creation."""
        with patch(
            "app.services.retriever_factory.Text2CypherRetriever"
        ) as mock_retriever_class:
            mock_retriever = MagicMock()
            mock_retriever_class.return_value = mock_retriever

            with patch.object(factory, "_get_llm") as mock_get_llm:
                mock_llm = MagicMock()
                mock_get_llm.return_value = mock_llm

                with patch.object(
                    factory, "_generate_schema_for_graph"
                ) as mock_generate_schema:
                    mock_generate_schema.return_value = "Sample schema"

                    retriever = await factory.create_text2cypher_retriever(
                        sample_text2cypher_config, "test-graph"
                    )

                    assert retriever is mock_retriever
                    mock_get_llm.assert_called_once_with(
                        model="gpt-4o", temperature=0.1, max_tokens=3000
                    )
                    mock_retriever_class.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.retriever
    async def test_create_hybrid_retriever(
        self, factory, mock_neo4j_client, sample_hybrid_config
    ):
        """Test hybrid retriever creation."""
        with patch(
            "app.services.retriever_factory.HybridRetriever"
        ) as mock_retriever_class:
            mock_retriever = MagicMock()
            mock_retriever_class.return_value = mock_retriever

            with patch.object(factory, "_get_embedder") as mock_get_embedder:
                mock_embedder = MagicMock()
                mock_get_embedder.return_value = mock_embedder

                retriever = await factory.create_hybrid_retriever(
                    sample_hybrid_config, "test-graph"
                )

                assert retriever is mock_retriever
                mock_get_embedder.assert_called_once()
                mock_retriever_class.assert_called_once()

    @pytest.mark.unit
    @pytest.mark.retriever
    async def test_create_retriever_unified_interface(
        self, factory, mock_neo4j_client, sample_vector_config
    ):
        """Test unified retriever creation interface."""
        config = RetrieverConfig(type=RetrieverType.VECTOR, config=sample_vector_config)

        with patch.object(factory, "create_vector_retriever") as mock_create_vector:
            mock_retriever = MagicMock()
            mock_create_vector.return_value = mock_retriever

            retriever = await factory.create_retriever(config, "test-graph")

            assert retriever is mock_retriever
            mock_create_vector.assert_called_once_with(
                sample_vector_config, "test-graph"
            )

    @pytest.mark.unit
    @pytest.mark.retriever
    async def test_create_unsupported_retriever_type(self, factory):
        """Test error handling for unsupported retriever types."""
        # RetrieverType is a Pydantic enum — passing an invalid value raises ValidationError
        with pytest.raises(ValidationError):
            RetrieverConfig(type="INVALID_TYPE", config={})  # type: ignore

    @pytest.mark.unit
    @pytest.mark.retriever
    async def test_neo4j_connection_error_handling(self, factory):
        """Test error handling when Neo4j connection fails."""
        with patch("app.services.retriever_factory.neo4j_client") as mock_client:
            mock_client.sync_driver = None
            mock_client.connect_async = AsyncMock()
            mock_client.connect_sync = MagicMock()

            config = VectorRetrieverConfig(
                index_name="test_index", return_properties=["text"]
            )

            with pytest.raises(
                ConnectionError,
                match="Failed to establish Neo4j sync driver connection for GraphRAG",
            ):
                await factory.create_vector_retriever(config, "test-graph")
