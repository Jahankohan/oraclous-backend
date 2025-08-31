#!/usr/bin/env python3
"""
Minimal test to verify entity-chunk connections work correctly
"""
import asyncio
import logging
import os
from pathlib import Path

from neo4j import GraphDatabase
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig, DocumentInfo
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_entity_chunk_connections():
    """Test that entities are properly connected to chunks"""
    try:
        # Simple test text
        text = """
        Albert Einstein was a German-born theoretical physicist widely held to be one of the greatest scientists. 
        He worked at Princeton University and developed the theory of relativity.
        """
        
        # Create components
        splitter = FixedSizeSplitter(chunk_size=100, chunk_overlap=20)
        
        # Mock embedder (to avoid API calls for this test)
        class MockEmbedder:
            async def embed_query(self, text: str):
                return [0.1] * 10  # Mock embedding
        
        embedder = TextChunkEmbedder(embedder=MockEmbedder())
        
        # Mock LLM for testing
        class MockLLM:
            async def ainvoke(self, prompt):
                class MockResponse:
                    content = '{"nodes": [{"id": "einstein", "label": "Person", "properties": {"name": "Albert Einstein"}}], "relationships": []}'
                return MockResponse()
        
        # Entity extractor with lexical graph enabled
        extractor = LLMEntityRelationExtractor(
            llm=MockLLM(),
            create_lexical_graph=True,
            on_error="IGNORE"
        )
        
        # Create pipeline
        pipeline = Pipeline()
        pipeline.add_component(splitter, "splitter")
        pipeline.add_component(embedder, "embedder")
        pipeline.add_component(extractor, "extractor")
        
        # Connect components
        pipeline.connect("splitter", "embedder", input_config={"text_chunks": "splitter"})
        pipeline.connect("embedder", "extractor", input_config={"chunks": "embedder"})
        
        # Run pipeline
        logger.info("Running pipeline...")
        result = await pipeline.run({
            "splitter": {"text": text},
            "extractor": {
                "document_info": DocumentInfo(path="test_document.txt"),
                "lexical_graph_config": LexicalGraphConfig()
            }
        })
        
        # Check the result
        graph = result["extractor"]
        logger.info(f"Pipeline completed successfully!")
        logger.info(f"Total nodes: {len(graph.nodes)}")
        logger.info(f"Total relationships: {len(graph.relationships)}")
        
        # Print nodes
        for node in graph.nodes:
            logger.info(f"Node: {node.label} - {node.properties}")
        
        # Print relationships
        for rel in graph.relationships:
            logger.info(f"Relationship: {rel.start_node_id} -[{rel.type}]-> {rel.end_node_id}")
        
        # Check for FROM_CHUNK relationships
        from_chunk_rels = [rel for rel in graph.relationships if rel.type == "FROM_CHUNK"]
        logger.info(f"FROM_CHUNK relationships found: {len(from_chunk_rels)}")
        
        if from_chunk_rels:
            logger.info("✅ SUCCESS: FROM_CHUNK relationships are being created!")
            for rel in from_chunk_rels:
                logger.info(f"  Entity -> Chunk: {rel.start_node_id} -> {rel.end_node_id}")
        else:
            logger.warning("❌ ISSUE: No FROM_CHUNK relationships found")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_entity_chunk_connections())
