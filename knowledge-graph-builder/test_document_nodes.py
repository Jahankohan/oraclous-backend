#!/usr/bin/env python3
"""
Test script to verify Document node creation with the enhanced pipeline
"""

import asyncio
import os
from benchmark import AdvancedGraphRAGPipeline, AdvancedPipelineConfig
from neo4j import GraphDatabase

async def test_document_nodes():
    """Test that Document nodes are created properly"""
    
    # Clean the database first
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', ''))
    
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("🧹 Database cleaned")
    
    driver.close()
    
    # Create configuration
    config = AdvancedPipelineConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j", 
        neo4j_password="",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        chunk_size=500,
        chunk_overlap=50,
        enable_entity_resolution=False,  # Disable for simplicity
        max_concurrency=1,  # Keep it simple
        batch_size=100
    )
    
    # Initialize pipeline
    pipeline = AdvancedGraphRAGPipeline(config)
    await pipeline.initialize()
    
    try:
        # Process the test document
        print("📄 Processing test document...")
        result = await pipeline.process_document("./document.txt")
        print(f"✅ Processing result: {result.get('success', False)}")
        
        # Check what was created in the database
        driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', ''))
        
        with driver.session() as session:
            print("\n=== Database Analysis ===")
            
            # Count all nodes by type
            result = session.run('MATCH (n) RETURN labels(n) as labels, count(n) as count ORDER BY count DESC')
            total_nodes = 0
            for record in result:
                labels = record['labels']
                count = record['count']
                total_nodes += count
                print(f'{labels}: {count}')
            print(f'Total nodes: {total_nodes}')
            
            # Check Document nodes specifically
            result = session.run('MATCH (d:Document) RETURN d.path as path, count(d) as count')
            for record in result:
                print(f"Document node found: path={record['path']}, count={record['count']}")
            
            # Check chunk to document relationships
            result = session.run('MATCH (c:Chunk)-[r:FROM_DOCUMENT]->(d:Document) RETURN count(r) as count')
            chunk_doc_rels = result.single()['count']
            print(f"Chunk->Document relationships: {chunk_doc_rels}")
            
            # Check total relationships
            result = session.run('MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC')
            total_rels = 0
            for record in result:
                rel_type = record['rel_type']
                count = record['count']
                total_rels += count
                print(f'{rel_type}: {count}')
            print(f'Total relationships: {total_rels}')
        
        driver.close()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        await pipeline.cleanup()

if __name__ == "__main__":
    asyncio.run(test_document_nodes())
