import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from uuid import uuid4
from app.services.embedding_service import embedding_service
from app.services.vector_service import vector_service
from app.services.enhanced_graph_service import enhanced_graph_service
from app.services.search_service import search_service

@pytest.mark.asyncio
async def test_embedding_service_initialization():
    """Test embedding service initialization with OpenAI"""
    
    with patch('app.services.credential_service.credential_service.get_openai_token') as mock_creds:
        mock_creds.return_value = "test-api-key"
        
        with patch('langchain_openai.OpenAIEmbeddings') as mock_embeddings:
            mock_instance = MagicMock()
            mock_embeddings.return_value = mock_instance
            
            success = await embedding_service.initialize_embeddings(
                provider="openai",
                model="text-embedding-3-small",
                user_id="test-user"
            )
            
            assert success is True
            assert embedding_service.provider == "openai"
            assert embedding_service.dimension == 512
            assert embedding_service.is_initialized() is True

@pytest.mark.asyncio
async def test_embedding_text_generation():
    """Test embedding generation for text"""
    
    # Setup mock embedding service
    embedding_service.embeddings = AsyncMock()
    embedding_service.embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    
    text = "John works at OpenAI"
    embedding = await embedding_service.embed_text(text)
    
    assert embedding == [0.1, 0.2, 0.3]
    embedding_service.embeddings.aembed_query.assert_called_once_with(text)

@pytest.mark.asyncio
async def test_vector_service_index_creation():
    """Test vector index creation in Neo4j"""
    
    with patch('app.core.neo4j_client.neo4j_client.execute_write_query') as mock_query:
        mock_query.return_value = None
        
        await vector_service.create_vector_indexes(dimension=512)
        
        # Should have called multiple queries for index creation
        assert mock_query.call_count >= 2
        
        # Check that vector index queries were called
        call_args = [call[0][0] for call in mock_query.call_args_list]
        assert any("CREATE VECTOR INDEX entity_embeddings" in query for query in call_args)
        assert any("CREATE VECTOR INDEX chunk_embeddings" in query for query in call_args)

@pytest.mark.asyncio
async def test_enhanced_graph_service_with_embeddings():
    """Test storing graph documents with embeddings"""
    
    # Mock dependencies
    with patch('app.services.embedding_service.embedding_service.initialize_embeddings') as mock_init:
        mock_init.return_value = True
        embedding_service.dimension = 512
        
        with patch('app.services.vector_service.vector_service.create_vector_indexes') as mock_indexes:
            mock_indexes.return_value = None
            
            with patch('app.core.neo4j_client.neo4j_client.execute_write_query') as mock_query:
                mock_query.return_value = None
                
                # Create mock graph document
                from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
                
                node1 = Node(id="1", type="Person", properties={"name": "John"})
                node2 = Node(id="2", type="Organization", properties={"name": "OpenAI"})
                rel = Relationship(source=node1, target=node2, type="WORKS_FOR")
                
                graph_doc = GraphDocument(nodes=[node1, node2], relationships=[rel])
                
                # Test storing with embeddings
                entities_count, relationships_count = await enhanced_graph_service.store_graph_documents_with_embeddings(
                    graph_id=uuid4(),
                    graph_documents=[graph_doc],
                    user_id="test-user",
                    generate_embeddings=True
                )
                
                assert entities_count == 2
                assert relationships_count == 1
                mock_init.assert_called_once()
                mock_indexes.assert_called_once()

@pytest.mark.asyncio
async def test_similarity_search_entities():
    """Test semantic similarity search on entities"""
    
    # Mock embedding service
    embedding_service.embeddings = AsyncMock()
    embedding_service.embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    
    # Mock Neo4j query result
    mock_result = [
        {
            "id": "entity-1",
            "name": "John Doe",
            "type": "Person",
            "labels": ["Person"],
            "score": 0.95,
            "properties": {"name": "John Doe", "title": "Engineer"}
        },
        {
            "id": "entity-2", 
            "name": "Jane Smith",
            "type": "Person",
            "labels": ["Person"],
            "score": 0.87,
            "properties": {"name": "Jane Smith", "title": "Manager"}
        }
    ]
    
    with patch('app.core.neo4j_client.neo4j_client.execute_query') as mock_query:
        mock_query.return_value = mock_result
        
        with patch('app.services.embedding_service.embedding_service.is_initialized') as mock_init:
            mock_init.return_value = True
            
            results = await search_service.similarity_search_entities(
                query="software engineer",
                graph_id=uuid4(),
                k=5,
                threshold=0.7
            )
            
            assert len(results) == 2
            assert results[0]["name"] == "John Doe"
            assert results[0]["score"] == 0.95
            embedding_service.embeddings.aembed_query.assert_called_once_with("software engineer")

@pytest.mark.asyncio
async def test_hybrid_search():
    """Test hybrid search combining semantic and keyword search"""
    
    # Mock semantic results
    semantic_results = [
        {"id": "1", "name": "John", "score": 0.9},
        {"id": "2", "name": "Jane", "score": 0.8}
    ]
    
    # Mock keyword results
    keyword_results = [
        {"id": "1", "name": "John", "score": 0.7},
        {"id": "3", "name": "Mike", "score": 0.6}
    ]
    
    with patch.object(search_service, 'similarity_search_entities') as mock_semantic:
        mock_semantic.return_value = semantic_results
        
        with patch.object(search_service, 'fulltext_search_entities') as mock_keyword:
            mock_keyword.return_value = keyword_results
            
            results = await search_service.hybrid_search(
                query="engineer",
                graph_id=uuid4(),
                k=5,
                semantic_weight=0.7,
                keyword_weight=0.3
            )
            
            # Should return combined and reranked results
            assert len(results) >= 2
            assert all("combined_score" in result for result in results)
            
            # John should have highest combined score (appears in both)
            john_result = next(r for r in results if r["id"] == "1")
            assert john_result["combined_score"] > 0.8

@pytest.mark.asyncio 
async def test_embedding_stats():
    """Test getting embedding statistics for a graph"""
    
    mock_stats = [
        {
            "total_nodes": 100,
            "nodes_with_embeddings": 80,
            "nodes_without_embeddings": 20
        }
    ]
    
    mock_chunk_stats = [
        {
            "total_chunks": 50,
            "chunks_with_embeddings": 45
        }
    ]
    
    with patch('app.core.neo4j_client.neo4j_client.execute_query') as mock_query:
        mock_query.side_effect = [mock_stats, mock_chunk_stats]
        
        stats = await enhanced_graph_service.get_embedding_stats(uuid4())
        
        assert stats["total_nodes"] == 100
        assert stats["nodes_with_embeddings"] == 80
        assert stats["nodes_without_embeddings"] == 20
        assert stats["total_chunks"] == 50
        assert stats["chunks_with_embeddings"] == 45
        assert stats["embedding_coverage"] == 80.0

@pytest.mark.asyncio
async def test_generate_embeddings_for_existing_nodes():
    """Test generating embeddings for existing nodes without them"""
    
    # Mock nodes without embeddings
    mock_nodes = [
        {"id": "1", "name": "John Doe", "description": "Software Engineer"},
        {"id": "2", "name": "OpenAI", "description": "AI Research Company"}
    ]
    
    # Mock embedding generation
    embedding_service.embeddings = AsyncMock()
    embedding_service.embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    
    with patch('app.core.neo4j_client.neo4j_client.execute_query') as mock_query:
        # First call returns nodes, second call returns empty (no more nodes)
        mock_query.side_effect = [mock_nodes, []]
        
        with patch('app.services.vector_service.vector_service.add_entity_embedding') as mock_add:
            mock_add.return_value = None
            
            with patch('app.services.embedding_service.embedding_service.is_initialized') as mock_init:
                mock_init.return_value = True
                
                nodes_processed = await enhanced_graph_service.generate_embeddings_for_existing_nodes(
                    graph_id=uuid4(),
                    user_id="test-user",
                    batch_size=50
                )
                
                assert nodes_processed == 2
                assert embedding_service.embeddings.aembed_query.call_count == 2
                assert mock_add.call_count == 2

@pytest.mark.asyncio
async def test_text_chunk_creation_with_embeddings():
    """Test creating text chunks with embeddings"""
    
    from langchain.schema import Document
    
    # Mock document
    source_doc = Document(
        page_content="This is a long text that should be chunked. " * 20,
        metadata={"source": "test_document"}
    )
    
    # Mock embedding generation
    embedding_service.embeddings = AsyncMock()
    embedding_service.embeddings.aembed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    
    with patch('app.services.vector_service.vector_service.create_text_chunks') as mock_chunks:
        mock_chunks.return_value = 5  # 5 chunks created
        
        with patch('app.services.embedding_service.embedding_service.is_initialized') as mock_init:
            mock_init.return_value = True
            
            await enhanced_graph_service._create_text_chunks_with_embeddings(
                source_document=source_doc,
                graph_id=uuid4(),
                generate_embeddings=True
            )
            
            # Should have created chunks with embeddings
            mock_chunks.assert_called_once()
            chunks_arg = mock_chunks.call_args[0][1]  # Second argument is chunks list
            
            assert len(chunks_arg) > 0
            assert all("embedding" in chunk for chunk in chunks_arg)
            assert all("text" in chunk for chunk in chunks_arg)
            assert all("id" in chunk for chunk in chunks_arg)
