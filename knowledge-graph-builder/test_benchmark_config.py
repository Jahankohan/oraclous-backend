#!/usr/bin/env python3
"""
Test benchmark.py configuration with actual database but mock LLM
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_benchmark_configuration():
    """Test the corrected benchmark configuration with mock LLM and real database"""
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', os.getenv('NEO4J_PASSWORD', 'password')))
        
        # Clear database
        with driver.session() as session:
            session.run('MATCH (n) DETACH DELETE n')
            logger.info("Database cleared")
        
        # Test text
        text = """
        Albert Einstein was a German-born theoretical physicist widely held to be one of the greatest scientists. 
        He worked at Princeton University and developed the theory of relativity.
        Marie Curie was a physicist and chemist who conducted pioneering research on radioactivity.
        """
        
        # Mock LLM that returns multiple entities
        class MockLLM:
            async def ainvoke(self, prompt):
                class MockResponse:
                    content = '''{"nodes": [
                        {"id": "einstein", "label": "Person", "properties": {"name": "Albert Einstein"}},
                        {"id": "curie", "label": "Person", "properties": {"name": "Marie Curie"}},
                        {"id": "princeton", "label": "Organization", "properties": {"name": "Princeton University"}}
                    ], "relationships": [
                        {"start_node_id": "einstein", "end_node_id": "princeton", "type": "WORKED_AT"}
                    ]}'''
                return MockResponse()
        
        # Mock embedder
        class MockEmbedder:
            def embed_query(self, text: str):
                return [0.1] * 10  # Mock embedding
        
        # Create components using the same configuration as benchmark.py
        splitter = FixedSizeSplitter(chunk_size=300, chunk_overlap=50)
        embedder = TextChunkEmbedder(embedder=MockEmbedder())
        
        # This is the key: extractor with create_lexical_graph=True
        extractor = LLMEntityRelationExtractor(
            llm=MockLLM(),
            create_lexical_graph=True,  # This enables FROM_CHUNK relationships
            max_concurrency=1,
            on_error="IGNORE"
        )
        
        writer = Neo4jWriter(driver=driver, batch_size=100)
        
        # Create pipeline exactly like in benchmark.py
        pipeline = Pipeline()
        pipeline.add_component(splitter, "splitter")
        pipeline.add_component(embedder, "embedder")
        pipeline.add_component(extractor, "extractor")
        pipeline.add_component(writer, "writer")
        
        # Connect components exactly like in benchmark.py
        pipeline.connect("splitter", "embedder", input_config={"text_chunks": "splitter"})
        pipeline.connect("embedder", "extractor", input_config={"chunks": "embedder"})
        pipeline.connect("extractor", "writer", input_config={"graph": "extractor"})
        
        # Run pipeline exactly like in benchmark.py
        logger.info("Running corrected pipeline...")
        result = await pipeline.run({
            "splitter": {"text": text},
            "extractor": {
                "document_info": DocumentInfo(path="test_document.txt"),
                "lexical_graph_config": LexicalGraphConfig()
            }
        })
        
        logger.info("Pipeline completed! Checking database...")
        
        # Check what was written to database
        with driver.session() as session:
            # Check all nodes
            nodes_result = session.run("MATCH (n) RETURN n.name as name, labels(n) as labels")
            logger.info("Nodes in database:")
            for record in nodes_result:
                logger.info(f"  {record['labels']}: {record['name']}")
                
            # Check all relationships
            rels_result = session.run("MATCH (a)-[r]->(b) RETURN type(r) as type, a.name as from_name, b.name as to_name, labels(a) as from_labels, labels(b) as to_labels")
            logger.info("Relationships in database:")
            for record in rels_result:
                logger.info(f"  {record['from_labels']}:{record['from_name']} -[{record['type']}]-> {record['to_labels']}:{record['to_name']}")
            
            # Specific check for entity-chunk connections
            entity_chunk_query = """
            MATCH (entity)-[:FROM_CHUNK]->(chunk:Chunk)
            RETURN entity.name as entity_name, labels(entity) as entity_labels, chunk.text[0..50] as chunk_preview
            """
            entity_chunk_result = session.run(entity_chunk_query)
            logger.info("Entity-Chunk connections:")
            count = 0
            for record in entity_chunk_result:
                count += 1
                logger.info(f"  {record['entity_labels']}:{record['entity_name']} -> Chunk: {record['chunk_preview']}...")
            
            if count > 0:
                logger.info(f"✅ SUCCESS: Found {count} entity-chunk connections!")
            else:
                logger.warning("❌ ISSUE: No entity-chunk connections found")
                
            # Check complete traceability chain
            traceability_query = """
            MATCH (doc:Document)-[:FROM_DOCUMENT]->(chunk:Chunk)<-[:FROM_CHUNK]-(entity)
            RETURN doc.path as document, chunk.text[0..30] as chunk_preview, entity.name as entity_name, labels(entity) as entity_labels
            """
            traceability_result = session.run(traceability_query)
            logger.info("Complete traceability chain (Document -> Chunk -> Entity):")
            trace_count = 0
            for record in traceability_result:
                trace_count += 1
                logger.info(f"  {record['document']} -> Chunk: {record['chunk_preview']}... -> {record['entity_labels']}:{record['entity_name']}")
            
            if trace_count > 0:
                logger.info(f"✅ COMPLETE SUCCESS: Found {trace_count} complete traceability chains!")
            else:
                logger.warning("❌ TRACEABILITY ISSUE: No complete chains found")
        
        driver.close()
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_benchmark_configuration())
