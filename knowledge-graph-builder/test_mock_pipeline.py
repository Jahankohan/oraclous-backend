#!/usr/bin/env python3
"""
Mock GraphRAG Performance Test - Test the pipeline with fake LLM responses
This demonstrates the corrected entity-chunk connections without needing OpenAI API
"""
import logging
import time
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline.pipeline import Pipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.text_chunk_embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.embedder import Embedder
from neo4j_graphrag.llm import LLMInterface
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MockLLM(LLMInterface):
    """Mock LLM that returns realistic entity extraction results for TechNova document"""
    
    def invoke(self, input: str) -> Any:
        """Mock entity extraction based on input text"""
        if "entity_relation_extractor" in str(input).lower() or "extract" in str(input).lower():
            # Return realistic entities for TechNova content
            if "TechNova" in input or "Sarah Chen" in input or "MedAI" in input:
                return {
                    "nodes": [
                        {"id": "1", "label": "Person", "properties": {"name": "Dr. Sarah Chen"}},
                        {"id": "2", "label": "Person", "properties": {"name": "Michael Rodriguez"}},
                        {"id": "3", "label": "Organization", "properties": {"name": "TechNova Corporation"}},
                        {"id": "4", "label": "Organization", "properties": {"name": "MIT"}},
                        {"id": "5", "label": "Organization", "properties": {"name": "Google"}},
                        {"id": "6", "label": "Product", "properties": {"name": "MedAI Platform"}},
                        {"id": "7", "label": "Location", "properties": {"name": "Austin"}},
                        {"id": "8", "label": "Location", "properties": {"name": "Texas"}},
                    ],
                    "relationships": [
                        {"startNodeId": "1", "endNodeId": "3", "type": "FOUNDED"},
                        {"startNodeId": "2", "endNodeId": "3", "type": "FOUNDED"},
                        {"startNodeId": "1", "endNodeId": "3", "type": "CEO_OF"},
                        {"startNodeId": "2", "endNodeId": "3", "type": "CTO_OF"},
                        {"startNodeId": "1", "endNodeId": "4", "type": "GRADUATED_FROM"},
                        {"startNodeId": "2", "endNodeId": "5", "type": "WORKED_AT"},
                        {"startNodeId": "3", "endNodeId": "6", "type": "DEVELOPED"},
                        {"startNodeId": "3", "endNodeId": "7", "type": "LOCATED_IN"},
                    ]
                }
            elif "Emily Watson" in input or "Johns Hopkins" in input:
                return {
                    "nodes": [
                        {"id": "9", "label": "Person", "properties": {"name": "Dr. Emily Watson"}},
                        {"id": "10", "label": "Organization", "properties": {"name": "Johns Hopkins Hospital"}},
                        {"id": "11", "label": "Role", "properties": {"name": "Chief Medical Officer"}},
                    ],
                    "relationships": [
                        {"startNodeId": "9", "endNodeId": "10", "type": "WORKED_AT"},
                        {"startNodeId": "9", "endNodeId": "3", "type": "WORKS_FOR"},
                        {"startNodeId": "9", "endNodeId": "11", "type": "HAS_ROLE"},
                    ]
                }
            elif "Alex Thompson" in input or "Amazon" in input:
                return {
                    "nodes": [
                        {"id": "12", "label": "Person", "properties": {"name": "Alex Thompson"}},
                        {"id": "13", "label": "Organization", "properties": {"name": "Amazon Web Services"}},
                        {"id": "14", "label": "Role", "properties": {"name": "Vice President of Engineering"}},
                    ],
                    "relationships": [
                        {"startNodeId": "12", "endNodeId": "13", "type": "WORKED_AT"},
                        {"startNodeId": "12", "endNodeId": "3", "type": "WORKS_FOR"},
                        {"startNodeId": "12", "endNodeId": "14", "type": "HAS_ROLE"},
                    ]
                }
        
        # Default fallback
        return {
            "nodes": [
                {"id": "default", "label": "Concept", "properties": {"name": "Technology"}}
            ],
            "relationships": []
        }

class MockEmbedder(Embedder):
    """Mock embedder that returns fake but consistent embeddings"""
    
    def embed_query(self, text: str) -> list[float]:
        """Return a fake embedding based on text hash"""
        # Create a simple hash-based embedding for consistency
        hash_val = hash(text) % 1000000
        return [float((hash_val + i) % 1000) / 1000.0 for i in range(1536)]

def test_mock_pipeline():
    """Test the GraphRAG pipeline with mock components"""
    print("🚀 Testing GraphRAG Pipeline with Mock LLM")
    print("=" * 60)
    
    # Read the document
    doc_path = Path("document.txt")
    if not doc_path.exists():
        print("❌ document.txt not found!")
        return
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"📄 Document loaded: {len(text)} characters")
    
    # Initialize Neo4j connection
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
    
    # Clear existing data for clean test
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("🧹 Cleared existing data")
    
    try:
        # Initialize components with corrected configuration
        mock_llm = MockLLM()
        mock_embedder = MockEmbedder()
        
        # Fixed size splitter
        splitter = FixedSizeSplitter(chunk_size=1200, chunk_overlap=200)
        
        # Text chunk embedder with mock embedder
        embedder = TextChunkEmbedder(embedder=mock_embedder)
        
        # CORRECTED: LLMEntityRelationExtractor with create_lexical_graph=True
        extractor = LLMEntityRelationExtractor(
            llm=mock_llm,
            create_lexical_graph=True,  # This is the KEY FIX for FROM_CHUNK relationships!
            lexical_graph_config={
                "id_column_name": "uid",
                "chunk_column_name": "text", 
                "chunk_id_column_name": "uid",
                "document_id_column_name": "document_id",
                "communities_column_name": "communities"
            }
        )
        
        # Neo4j writer
        writer = Neo4jWriter(driver=driver, neo4j_database="neo4j")
        
        # Create pipeline with corrected configuration
        pipeline = Pipeline(
            driver=driver,
            neo4j_database="neo4j"
        )
        
        # Add components in correct order
        pipeline.add_component(splitter, "splitter")
        pipeline.add_component(embedder, "embedder") 
        pipeline.add_component(extractor, "extractor")
        pipeline.add_component(writer, "writer")
        
        # Connect components
        pipeline.connect("splitter", "embedder", input_config={"text_chunks": "chunks"})
        pipeline.connect("embedder", "extractor", input_config={"text_chunks": "text_chunks"})
        pipeline.connect("extractor", "writer", input_config={"graph": "graph"})
        
        print("🔧 Pipeline configured with corrected entity-chunk connections")
        
        # Process the document
        start_time = time.time()
        
        print("⚡ Running pipeline...")
        result = pipeline.run({"text": text})
        
        processing_time = time.time() - start_time
        print(f"✅ Pipeline completed in {processing_time:.2f} seconds")
        
        # Analyze results
        with driver.session() as session:
            # Count nodes and relationships
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            # Check for FROM_CHUNK relationships specifically
            from_chunk_count = session.run(
                "MATCH ()-[r:FROM_CHUNK]->() RETURN count(r) as count"
            ).single()["count"]
            
            # Check for complete traceability chains
            trace_chains = session.run("""
                MATCH (entity)-[:FROM_CHUNK]->(chunk)-[:FROM_DOCUMENT]->(doc)
                RETURN count(*) as count
            """).single()["count"]
            
            # Get sample entities
            entities = session.run("""
                MATCH (n) WHERE n.name IS NOT NULL
                RETURN labels(n)[0] as label, n.name as name
                ORDER BY n.name LIMIT 10
            """).data()
            
            # Get sample chunks
            chunks = session.run("""
                MATCH (c:Chunk)
                RETURN c.index as index, c.text[..100] + "..." as preview
                ORDER BY c.index LIMIT 5
            """).data()
        
        # Print results
        print("\n📊 Processing Results:")
        print(f"Total Nodes Created: {node_count}")
        print(f"Total Relationships Created: {rel_count}")
        print(f"FROM_CHUNK Relationships: {from_chunk_count} ✅" if from_chunk_count > 0 else "FROM_CHUNK Relationships: 0 ❌")
        print(f"Complete Traceability Chains: {trace_chains} ✅" if trace_chains > 0 else "Complete Traceability Chains: 0 ❌")
        
        if entities:
            print("\n👥 Sample Entities Created:")
            for entity in entities:
                print(f"  {entity['label']}: {entity['name']}")
        
        if chunks:
            print("\n📝 Sample Chunks Created:")
            for chunk in chunks:
                print(f"  Chunk {chunk['index']}: {chunk['preview']}")
        
        # Final assessment
        print("\n🎯 Entity-Chunk Connection Test:")
        if from_chunk_count > 0 and trace_chains > 0:
            print("✅ SUCCESS: Entity-chunk connections are working correctly!")
            print("✅ Complete traceability: Entity -> Chunk -> Document chains established")
        elif node_count > 0:
            print("⚠️  Partial success: Entities created but connections may need verification")
        else:
            print("❌ FAILED: No entities or relationships created")
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    test_mock_pipeline()
