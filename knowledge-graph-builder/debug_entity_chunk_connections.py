#!/usr/bin/env python3
"""
Debug script to test entity-chunk connections in detail
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

async def debug_entity_chunk_connections():
    """Debug entity-chunk connections with detailed logging"""
    
    print("🔍 Debugging Entity-Chunk Connections")
    print("=" * 50)
    
    # Clean the database first
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
        print("🧹 Database cleaned")
    
    # Initialize LLM and embeddings
    llm = OpenAILLM(
        model_name="gpt-4o",
        api_key=""
    )
    
    embedder = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=""
    )
    
    # Create pipeline
    pipeline = Pipeline()
    
    # Add components
    text_splitter = FixedSizeSplitter(chunk_size=500, chunk_overlap=50)
    chunk_embedder = TextChunkEmbedder(embedder=embedder)
    lexical_graph_builder = LexicalGraphBuilder()
    
    # Entity extractor configured to NOT create lexical graph
    entity_extractor = LLMEntityRelationExtractor(
        llm=llm,
        create_lexical_graph=False,  # This is key - don't create duplicates
        max_concurrency=1
    )
    
    lexical_writer = Neo4jWriter(driver=driver)
    entity_writer = Neo4jWriter(driver=driver)
    
    # Add components to pipeline
    pipeline.add_component(text_splitter, "splitter")
    pipeline.add_component(chunk_embedder, "embedder")
    pipeline.add_component(lexical_graph_builder, "lexical_graph_builder")
    pipeline.add_component(entity_extractor, "extractor")
    pipeline.add_component(lexical_writer, "lexical_writer")
    pipeline.add_component(entity_writer, "entity_writer")
    
    # Connect components step by step
    print("\n📋 Setting up pipeline connections...")
    
    # 1. Text splitting -> embedding
    pipeline.connect("splitter", "embedder", input_config={"text_chunks": "splitter"})
    print("✅ Connected splitter -> embedder")
    
    # 2. Embedding -> lexical graph building (creates Document and Chunk nodes)
    pipeline.connect("embedder", "lexical_graph_builder", input_config={"text_chunks": "embedder"})
    print("✅ Connected embedder -> lexical_graph_builder")
    
    # 3. Lexical graph -> writer (writes Document and Chunk nodes first)
    pipeline.connect("lexical_graph_builder", "lexical_writer", input_config={
        "graph": "lexical_graph_builder.graph",
        "lexical_graph_config": "lexical_graph_builder.config"
    })
    print("✅ Connected lexical_graph_builder -> lexical_writer")
    
    # 4. CRITICAL: Entity extractor needs chunks from embedder to create FROM_CHUNK relationships
    pipeline.connect("embedder", "extractor", input_config={"chunks": "embedder"})
    print("✅ Connected embedder -> extractor (for FROM_CHUNK relationships)")
    
    # 5. Ensure lexical graph is written before entities (dependency)
    pipeline.connect("lexical_writer", "extractor", {})
    print("✅ Connected lexical_writer -> extractor (dependency)")
    
    # 6. Entity extraction -> entity writer
    pipeline.connect("extractor", "entity_writer", input_config={"graph": "extractor"})
    print("✅ Connected extractor -> entity_writer")
    
    # Read test document
    with open("./document.txt", "r") as f:
        text = f.read()
    
    print(f"\n📄 Processing text ({len(text)} characters)...")
    
    # Create document info
    document_info = DocumentInfo(path="./document.txt")
    lexical_graph_config = LexicalGraphConfig()
    
    try:
        print("\n🚀 Running pipeline...")
        
        result = await pipeline.run({
            "splitter": {"text": text},
            "lexical_graph_builder": {"document_info": document_info},
            "extractor": {
                "lexical_graph_config": lexical_graph_config,
                "document_info": document_info
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
            
            # 2. Check Document nodes
            print("\n2️⃣ Document nodes:")
            result = session.run('MATCH (d:Document) RETURN d.path as path, elementId(d) as id')
            doc_count = 0
            for record in result:
                doc_count += 1
                print(f"   📄 Document: {record['path']} (ID: {record['id']})")
            
            # 3. Check Chunk nodes and their connections
            print("\n3️⃣ Chunk nodes and Document connections:")
            result = session.run('''
                MATCH (c:Chunk)
                OPTIONAL MATCH (c)-[r:FROM_DOCUMENT]->(d:Document)
                RETURN elementId(c) as chunk_id, c.index as chunk_index, elementId(d) as doc_id, d.path as doc_path
                ORDER BY c.index
            ''')
            chunk_count = 0
            for record in result:
                chunk_count += 1
                chunk_id = record['chunk_id']
                chunk_index = record['chunk_index']
                doc_id = record['doc_id']
                doc_path = record['doc_path']
                connection = "✅ Connected" if doc_id else "❌ Not connected"
                print(f"   📝 Chunk {chunk_index} (ID: {chunk_id}) -> {connection} to {doc_path}")
            
            # 4. Check Entity nodes
            print("\n4️⃣ Entity nodes (sample):")
            result = session.run('MATCH (e:__Entity__) RETURN elementId(e) as id, labels(e) as labels, e.name as name LIMIT 10')
            entity_count = 0
            for record in result:
                entity_count += 1
                print(f"   👤 Entity: {record['name']} ({record['labels']}) ID: {record['id']}")
            
            # 5. CRITICAL: Check FROM_CHUNK relationships
            print("\n5️⃣ FROM_CHUNK relationships (Entity -> Chunk):")
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
            
            # 6. Check all relationship types
            print("\n6️⃣ All relationship types:")
            result = session.run('MATCH ()-[r]->() RETURN type(r) as rel_type, count(r) as count ORDER BY count DESC')
            total_rels = 0
            for record in result:
                rel_type = record['rel_type']
                count = record['count']
                total_rels += count
                print(f"   🔗 {rel_type}: {count}")
            print(f"   📊 Total relationships: {total_rels}")
            
            # 7. Critical validation: Complete traceability chain
            print("\n7️⃣ Complete traceability validation:")
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
                
                # Debug: Check what's missing
                print("\n🔍 Debugging missing connections:")
                
                # Check if entities exist
                entity_result = session.run('MATCH (e:__Entity__) RETURN count(e) as count')
                entity_total = entity_result.single()['count']
                print(f"   Entities found: {entity_total}")
                
                # Check if chunks exist
                chunk_result = session.run('MATCH (c:Chunk) RETURN count(c) as count')
                chunk_total = chunk_result.single()['count']
                print(f"   Chunks found: {chunk_total}")
                
                # Check if FROM_CHUNK relationships exist
                from_chunk_result = session.run('MATCH ()-[r:FROM_CHUNK]->() RETURN count(r) as count')
                from_chunk_total = from_chunk_result.single()['count']
                print(f"   FROM_CHUNK relationships: {from_chunk_total}")
                
                if from_chunk_total == 0:
                    print("   🔍 Issue: No FROM_CHUNK relationships found!")
                    print("   💡 This suggests the entity extractor is not connecting entities to chunks")
    
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        driver.close()

if __name__ == "__main__":
    asyncio.run(debug_entity_chunk_connections())
