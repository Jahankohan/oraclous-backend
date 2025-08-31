#!/usr/bin/env python3
"""
Quick GraphRAG Query Test Runner

A simplified script to test specific types of queries without running the full test suite.
Useful for quick testing and debugging of individual query types.
"""

import asyncio
import json
from pathlib import Path
from benchmark import AdvancedGraphRAGPipeline, AdvancedPipelineConfig
from test_graphrag_queries import GraphRAGQueryTester

async def quick_test_chunk_search():
    """Quick test of chunk search functionality"""
    print("🔍 Quick Test: Chunk Search")
    print("-" * 30)
    
    config = AdvancedPipelineConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j", 
        neo4j_password="",
        openai_api_key=""
    )
    
    tester = GraphRAGQueryTester(config)
    
    try:
        await tester.initialize()
        
        # Test a single query
        query = input("Enter your search query: ")
        
        # Search chunks
        result = tester.retrievers['vector'].search(query_text=query, top_k=3)
        
        print(f"\n📄 Found {len(result.items)} relevant chunks:")
        for i, item in enumerate(result.items, 1):
            print(f"\n{i}. Content: {item.content[:200]}...")
            if item.metadata:
                print(f"   Metadata: {item.metadata}")
        
    finally:
        await tester.cleanup()

async def quick_test_entity_search():
    """Quick test of entity search functionality"""
    print("👤 Quick Test: Entity Search")
    print("-" * 30)
    
    config = AdvancedPipelineConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="", 
        openai_api_key=""
    )
    
    tester = GraphRAGQueryTester(config)
    
    try:
        await tester.initialize()
        
        # Test a single query
        query = input("Enter your entity search query: ")
        
        # Search entities
        result = tester.retrievers['entity'].search(query_text=query, top_k=5)
        
        print(f"\n🏢 Found {len(result.items)} relevant entities:")
        for i, item in enumerate(result.items, 1):
            print(f"{i}. {item.content}")
        
    finally:
        await tester.cleanup()

async def quick_test_graph_traversal():
    """Quick test of graph traversal functionality"""
    print("🔗 Quick Test: Graph Traversal")
    print("-" * 30)
    
    config = AdvancedPipelineConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="",
        openai_api_key=""
    )
    
    tester = GraphRAGQueryTester(config)
    
    try:
        await tester.initialize()
        
        # Test a single query
        query = input("Enter your relationship search query: ")
        
        # Search with graph context
        result = tester.retrievers['vector_cypher'].search(query_text=query, top_k=3)
        
        print(f"\n🕸️ Found {len(result.items)} contexts with relationships:")
        for i, item in enumerate(result.items, 1):
            print(f"\n{i}. Context: {item.content}")
            if item.metadata:
                print(f"   Metadata: {json.dumps(item.metadata, indent=2)}")
        
    finally:
        await tester.cleanup()

async def quick_test_database_stats():
    """Quick test to check database statistics"""
    print("📊 Quick Test: Database Statistics")
    print("-" * 30)
    
    config = AdvancedPipelineConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password=""
    )
    
    from neo4j import GraphDatabase
    
    driver = GraphDatabase.driver(
        config.neo4j_uri,
        auth=(config.neo4j_user, config.neo4j_password)
    )
    
    try:
        with driver.session(database=config.neo4j_database) as session:
            # Get basic stats
            result = session.run("""
                MATCH (d:Document) 
                OPTIONAL MATCH (d)-[:FROM_DOCUMENT]->(c:Chunk)
                OPTIONAL MATCH (c)<-[:FROM_CHUNK]-(e:__Entity__)
                OPTIONAL MATCH (e)-[r]->(re:__Entity__)
                RETURN 
                    count(DISTINCT d) as documents,
                    count(DISTINCT c) as chunks, 
                    count(DISTINCT e) as entities,
                    count(DISTINCT r) as relationships
            """).single()
            
            print(f"📚 Documents: {result['documents']}")
            print(f"📄 Chunks: {result['chunks']}")
            print(f"🏷️ Entities: {result['entities']}")
            print(f"🔗 Relationships: {result['relationships']}")
            
            # Check indexes
            indexes = session.run("SHOW INDEXES").data()
            vector_indexes = [idx for idx in indexes if idx.get('type') == 'VECTOR']
            fulltext_indexes = [idx for idx in indexes if idx.get('type') == 'FULLTEXT']
            
            print(f"🔢 Vector Indexes: {len(vector_indexes)}")
            print(f"📝 Fulltext Indexes: {len(fulltext_indexes)}")
            
            if vector_indexes:
                print("\nVector Indexes:")
                for idx in vector_indexes:
                    print(f"  - {idx['name']} ({idx.get('labelsOrTypes', 'N/A')})")
            
            if fulltext_indexes:
                print("\nFulltext Indexes:")
                for idx in fulltext_indexes:
                    print(f"  - {idx['name']} ({idx.get('labelsOrTypes', 'N/A')})")
    
    finally:
        driver.close()

def show_menu():
    """Show the test menu"""
    print("\n" + "="*50)
    print("🧪 GraphRAG Quick Test Menu")
    print("="*50)
    print("1. Test Chunk Search (Vector Similarity)")
    print("2. Test Entity Search")
    print("3. Test Graph Traversal (Relationships)")
    print("4. Check Database Statistics")
    print("5. Run Full Test Suite")
    print("6. Exit")
    print("-"*50)

async def main():
    """Main menu for quick testing"""
    
    while True:
        show_menu()
        choice = input("Select an option (1-6): ").strip()
        
        try:
            if choice == "1":
                await quick_test_chunk_search()
            elif choice == "2":
                await quick_test_entity_search()
            elif choice == "3":
                await quick_test_graph_traversal()
            elif choice == "4":
                await quick_test_database_stats()
            elif choice == "5":
                print("🚀 Running Full Test Suite...")
                from test_graphrag_queries import main as run_full_tests
                await run_full_tests()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please select 1-6.")
                
        except KeyboardInterrupt:
            print("\n⏹️ Test interrupted by user")
        except Exception as e:
            print(f"❌ Error running test: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    asyncio.run(main())
