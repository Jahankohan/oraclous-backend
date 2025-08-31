#!/usr/bin/env python3
"""
Simple test for document node creation without LLM dependency
"""

import asyncio
from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.types import DocumentInfo

async def test_document_nodes_simple():
    """Simple test for Document node creation"""
    
    # Clean the database first
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("🧹 Database cleaned")
    
    # Create simple pipeline for lexical graph only
    pipeline = Pipeline()
    
    # Add components
    text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=50)
    lexical_graph_builder = LexicalGraphBuilder()
    kg_writer = Neo4jWriter(driver=driver)
    
    pipeline.add_component(text_splitter, "splitter")
    pipeline.add_component(lexical_graph_builder, "lexical_graph_builder")
    pipeline.add_component(kg_writer, "writer")
    
    # Connect components
    pipeline.connect("splitter", "lexical_graph_builder", input_config={"text_chunks": "splitter"})
    pipeline.connect("lexical_graph_builder", "writer", input_config={
        "graph": "lexical_graph_builder.graph",
        "lexical_graph_config": "lexical_graph_builder.config"
    })
    
    # Read the test document
    with open("./document.txt", "r") as f:
        text = f.read()
    
    # Create document info
    document_info = DocumentInfo(path="./document.txt")
    
    try:
        print("📄 Running lexical graph pipeline...")
        
        # Run the pipeline with document info
        result = await pipeline.run({
            "splitter": {"text": text},
            "lexical_graph_builder": {"document_info": document_info}
        })
        
        print("✅ Pipeline completed")
        
        # Check what was created in the database
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
            result = session.run('MATCH (d:Document) RETURN d.path as path')
            doc_count = 0
            for record in result:
                doc_count += 1
                print(f"✅ Document node found: path={record['path']}")
            
            if doc_count == 0:
                print("❌ No Document nodes found")
            
            # Check chunk to document relationships
            result = session.run('MATCH (c:Chunk)-[r:FROM_DOCUMENT]->(d:Document) RETURN count(r) as count')
            chunk_doc_rels = result.single()['count']
            print(f"Chunk->Document relationships: {chunk_doc_rels}")
            
            # Check all relationships
            result = session.run('MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC')
            total_rels = 0
            for record in result:
                rel_type = record['rel_type']
                count = record['count']
                total_rels += count
                print(f'{rel_type}: {count}')
            print(f'Total relationships: {total_rels}')
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise
    finally:
        driver.close()

if __name__ == "__main__":
    asyncio.run(test_document_nodes_simple())
