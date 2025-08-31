#!/usr/bin/env python3
"""Simple debug script for Text2CypherRetriever."""

import os
import logging
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm.openai_llm import OpenAILLM

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Hardcoded values - no environment bullshit
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = ""
    OPENAI_API_KEY = ""
    
    # Set OpenAI key
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    
    print(f"Connecting to Neo4j at {NEO4J_URI}")
    
    # Create driver
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    # Test connection
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 as test")
            print(f"✅ Connected to Neo4j: {result.single()['test']}")
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return
    
    # Check database content
    with driver.session() as session:
        entity_count = session.run("MATCH (e:__Entity__) RETURN count(e) as count").single()['count']
        print(f"📊 Entities in database: {entity_count}")
        
        if entity_count == 0:
            print("❌ No entities found - Text2CypherRetriever won't work without data")
            return
    
    # Create LLM
    print("Creating OpenAI LLM...")
    llm = OpenAILLM(model_name="gpt-4o-mini")
    
    # Create Text2CypherRetriever
    print("Creating Text2CypherRetriever...")
    try:
        retriever = Text2CypherRetriever(
            driver=driver,
            llm=llm,
            neo4j_schema=None  # Auto-discover
        )
        print("✅ Text2CypherRetriever created")
    except Exception as e:
        print(f"❌ Failed to create Text2CypherRetriever: {e}")
        return
    
    # Test query
    test_query = "Show me all people who work at TechNova Corporation"
    print(f"\nTesting query: '{test_query}'")
    
    try:
        result = retriever.get_search_results(test_query)
        print(f"✅ Query executed")
        print(f"Result type: {type(result)}")
        print(f"Result: {result}")
        
        # Try to find generated Cypher
        cypher = None
        if hasattr(result, 'cypher_query'):
            cypher = result.cypher_query
        elif hasattr(retriever, 'last_query'):
            cypher = retriever.last_query
        
        if cypher:
            print(f"🔍 Generated Cypher: {cypher}")
        else:
            print("⚠️ No Cypher found")
            
    except Exception as e:
        print(f"❌ Query failed: {e}")
        if "Unexpected end of input" in str(e):
            print("🔍 Empty query error - LLM returned empty string")
    
    driver.close()
    print("Done.")

if __name__ == "__main__":
    main()
