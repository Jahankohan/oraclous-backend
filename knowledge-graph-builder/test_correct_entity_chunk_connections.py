#!/usr/bin/env python3
"""
Test the correct way to connect entities to chunks based on neo4j-graphrag source code
"""

import asyncio
import os
from neo4j import GraphDatabase
from neo4j_graphrag.experimental.pipeline import Pipeline
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
from neo4j_graphrag.experimental.components.lexical_graph import LexicalGraphBuilder
from neo4j_graphrag.experimental.components.entity_relation_extractor import LLMEntityRelationExtractor
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter
from neo4j_graphrag.experimental.components.types import DocumentInfo, LexicalGraphConfig
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings import OpenAIEmbeddings

async def test_correct_entity_chunk_connections():
    """Test the CORRECT way to connect entities to chunks"""
    
    print("🔍 Testing CORRECT Entity-Chunk Connections")
    print("=" * 50)
    
    # Clean the database first
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', ''))
    
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("🧹 Database cleaned")
    
    # Initialize LLM and embeddings
    llm = OpenAILLM(
        model_name="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Add components
    text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=50)
    chunk_embedder = TextChunkEmbedder(embedder=embedder)
    lexical_graph_builder = LexicalGraphBuilder()
    
    # CRITICAL: The LLMEntityRelationExtractor should have create_lexical_graph=True
    # AND get lexical_graph_config to connect entities to existing chunks
    entity_extractor = LLMEntityRelationExtractor(
        llm=llm,
        create_lexical_graph=True,  # This is important - let it create the full graph including chunk connections
        max_concurrency=1
    )
    
    kg_writer = Neo4jWriter(driver=driver)
    
    # Add components to pipeline
    pipeline.add_component(text_splitter, "splitter")
    pipeline.add_component(chunk_embedder, "embedder")
    pipeline.add_component(entity_extractor, "extractor")
    pipeline.add_component(kg_writer, "writer")
    
    # SIMPLIFIED CONNECTION - just connect the essential components
    print("\n📋 Setting up SIMPLIFIED pipeline connections...")
    
    # 1. Text splitting -> embedding
    pipeline.connect("splitter", "embedder", input_config={"text_chunks": "splitter"})
    print("✅ Connected splitter -> embedder")
    
    # 2. Embedder -> Entity Extractor (this should create complete graph with chunks AND entities)
    pipeline.connect("embedder", "extractor", input_config={"chunks": "embedder"})
    print("✅ Connected embedder -> extractor")
    
    # 3. Entity extractor -> writer
    pipeline.connect("extractor", "writer", input_config={"graph": "extractor"})
    print("✅ Connected extractor -> writer")
    
    # Read test document
    with open("./document.txt", "r") as f:
        text = f.read()
    
    print(f"\n📄 Processing text ({len(text)} characters)...")
    
    # Create document info
    document_info = DocumentInfo(path="./document.txt")
    lexical_graph_config = LexicalGraphConfig()
    
    try:
        print("\n🚀 Running simplified pipeline...")
        
        result = await pipeline.run({
            "splitter": {"text": text},
            "extractor": {
                "document_info": document_info,
                "lexical_graph_config": lexical_graph_config
            }
        })
        
        print("✅ Pipeline completed successfully!")
        
        # Detailed database analysis
        print("\n🔍 Detailed Database Analysis")
        print("=" * 40)
        
        with driver.session() as session:
            # 1. Count all nodes by type
            print("\n1️⃣ Node counts by type:")
            result = session.run('MATCH (n) RETURN labels(n) as labels, count(n) as count ORDER BY count DESC')
            total_nodes = 0
            for record in result:
                labels = record['labels']
                count = record['count']
                total_nodes += count
                print(f"   {labels}: {count}")
            print(f"   📊 Total nodes: {total_nodes}")
            
            # 2. Check FROM_CHUNK relationships specifically
            print("\n2️⃣ FROM_CHUNK relationships (Entity -> Chunk):")
            result = session.run('''
                MATCH (e:__Entity__)-[r:FROM_CHUNK]->(c:Chunk)
                RETURN elementId(e) as entity_id, e.name as entity_name, 
                       elementId(c) as chunk_id, c.index as chunk_index
                ORDER BY chunk_index, entity_name
                LIMIT 20
            ''')
            from_chunk_count = 0
            for record in result:
                from_chunk_count += 1
                print(f"   🔗 {record['entity_name']} -> Chunk {record['chunk_index']}")
            
            print(f"\n   📊 Total FROM_CHUNK relationships: {from_chunk_count}")
            
            # 3. Critical validation: Complete traceability chain
            print("\n3️⃣ Complete traceability validation:")
            result = session.run('''
                MATCH (d:Document)<-[:FROM_DOCUMENT]-(c:Chunk)<-[:FROM_CHUNK]-(e:__Entity__)
                RETURN d.path as doc_path, c.index as chunk_index, e.name as entity_name
                ORDER BY chunk_index, entity_name
                LIMIT 10
            ''')
            traceability_count = 0
            for record in result:
                traceability_count += 1
                print(f"   ✅ {record['entity_name']} -> Chunk {record['chunk_index']} -> {record['doc_path']}")
            
            if traceability_count > 0:
                print(f"\n🎉 SUCCESS: {traceability_count} entities have complete traceability!")
            else:
                print("\n❌ FAILURE: No complete traceability chains found!")
                
                # Debug: Check what relationships exist
                print("\n🔍 All relationship types:")
                result = session.run('MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC')
                for record in result:
                    rel_type = record['rel_type']
                    count = record['count']
                    print(f"   🔗 {rel_type}: {count}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    asyncio.run(test_correct_entity_chunk_connections())
