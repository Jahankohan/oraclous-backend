#!/usr/bin/env python3
"""
Test to verify LLMEntityRelationExtractor configuration with create_lexical_graph=True
"""
import asyncio
import logging

from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig, DocumentInfo, TextChunk, TextChunks

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_extractor_configuration():
    """Test that the LLMEntityRelationExtractor can be configured correctly"""
    try:
        # Mock LLM for testing
        class MockLLM:
            async def ainvoke(self, prompt):
                class MockResponse:
                    content = '{"nodes": [{"id": "einstein", "label": "Person", "properties": {"name": "Albert Einstein"}}], "relationships": []}'
                return MockResponse()
        
        # Test 1: Create extractor with create_lexical_graph=True
        logger.info("Creating LLMEntityRelationExtractor with create_lexical_graph=True...")
        extractor = LLMEntityRelationExtractor(
            llm=MockLLM(),
            create_lexical_graph=True,
            on_error="IGNORE"
        )
        logger.info("✅ Extractor created successfully!")
        
        # Test 2: Test the run method with lexical_graph_config
        logger.info("Testing extractor.run() with lexical_graph_config...")
        
        chunks = TextChunks(chunks=[
            TextChunk(text="Albert Einstein was a physicist", index=0, chunk_id="chunk_0")
        ])
        
        document_info = DocumentInfo(path="test_document.txt")
        lexical_graph_config = LexicalGraphConfig()
        
        result = await extractor.run(
            chunks=chunks,
            document_info=document_info,
            lexical_graph_config=lexical_graph_config
        )
        
        logger.info("✅ Extractor ran successfully!")
        logger.info(f"Result type: {type(result)}")
        logger.info(f"Total nodes: {len(result.nodes)}")
        logger.info(f"Total relationships: {len(result.relationships)}")
        
        # Check node types
        node_labels = [node.label for node in result.nodes]
        logger.info(f"Node labels: {node_labels}")
        
        # Check relationship types  
        rel_types = [rel.type for rel in result.relationships]
        logger.info(f"Relationship types: {rel_types}")
        
        # Check for FROM_CHUNK relationships specifically
        from_chunk_rels = [rel for rel in result.relationships if rel.type == "FROM_CHUNK"]
        logger.info(f"FROM_CHUNK relationships: {len(from_chunk_rels)}")
        
        if from_chunk_rels:
            logger.info("✅ SUCCESS: FROM_CHUNK relationships are being created!")
            for rel in from_chunk_rels:
                logger.info(f"  {rel.start_node_id} -[FROM_CHUNK]-> {rel.end_node_id}")
        else:
            logger.warning("⚠️  No FROM_CHUNK relationships found")
            
        # Check for Document and Chunk nodes
        has_document = any(node.label == "Document" for node in result.nodes)
        has_chunk = any(node.label == "Chunk" for node in result.nodes)
        has_entity = any(node.label == "Person" for node in result.nodes)
        
        logger.info(f"Has Document node: {has_document}")
        logger.info(f"Has Chunk node: {has_chunk}")
        logger.info(f"Has Entity node: {has_entity}")
        
        if has_document and has_chunk and has_entity:
            logger.info("✅ All expected node types are present!")
        else:
            logger.warning("⚠️  Some expected node types are missing")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_extractor_configuration())
