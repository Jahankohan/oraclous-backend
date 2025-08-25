import pytest
from unittest.mock import AsyncMock, patch
from app.services.entity_extractor import entity_extractor
from app.services.llm_service import llm_service
import uuid

@pytest.mark.asyncio
@patch('app.services.llm_service.llm_service.initialize_llm')
@patch('app.services.llm_service.llm_service.graph_transformer')
async def test_entity_extraction(mock_transformer, mock_initialize):
    """Test entity extraction from text"""
    
    # Mock LLM initialization
    mock_initialize.return_value = True
    llm_service.llm = AsyncMock()
    llm_service.graph_transformer = AsyncMock()
    
    # Mock graph transformer response
    mock_graph_doc = AsyncMock()
    mock_graph_doc.nodes = [AsyncMock()]
    mock_graph_doc.relationships = [AsyncMock()]
    mock_transformer.aconvert_to_graph_documents.return_value = [mock_graph_doc]
    
    # Test extraction
    text = "John works at OpenAI. OpenAI is located in San Francisco."
    user_id = str(uuid.uuid4())
    graph_id = uuid.uuid4()
    
    result = await entity_extractor.extract_entities_from_text(
        text=text,
        user_id=user_id,
        graph_id=graph_id,
        schema={"entities": ["Person", "Organization"], "relationships": ["WORKS_AT"]}
    )
    
    assert len(result) > 0
    mock_initialize.assert_called_once()

@pytest.mark.asyncio
async def test_schema_learning():
    """Test schema learning from text"""
    
    with patch('app.services.llm_service.llm_service.llm') as mock_llm:
        mock_response = AsyncMock()
        mock_response.content = '{"entities": ["Person", "Organization"], "relationships": ["WORKS_AT"]}'
        mock_llm.ainvoke.return_value = mock_response
        
        result = await entity_extractor.learn_schema_from_text(
            "John works at OpenAI",
            "user123"
        )
        
        assert "entities" in result
        assert "relationships" in result
        assert "Person" in result["entities"]
