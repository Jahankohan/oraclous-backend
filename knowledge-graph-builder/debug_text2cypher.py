#!/usr/bin/env python3

import asyncio
import os
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import Text2CypherRetriever
from neo4j_graphrag.llm.openai_llm import OpenAILLM

async def test_text2cypher():
    """Test Text2CypherRetriever to see what Cypher it generates"""
    
    # Neo4j connection
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", ""),
        database="neo4j"
    )
    
    # Set OpenAI API key (hardcoded for testing)
    os.environ["OPENAI_API_KEY"] = ""
    
    # OpenAI LLM (using environment variable)
    llm = OpenAILLM(
        model_name="gpt-4o-mini",
        model_params={
            "temperature": 0.1,
            "max_tokens": 1000,
            "response_format": {"type": "text"}  # Ensure text output, not JSON
        }
    )
    
    # Simple schema
    schema = """
    Node Types:
    - __Entity__: Properties: name, description, type
    - Chunk: Properties: text, chunk_index
    - Document: Properties: path
    
    Relationships:
    - FROM_CHUNK: Entity -> Chunk
    - FROM_DOCUMENT: Chunk -> Document
    - RELATED_TO: Entity -> Entity
    """
    
    # Simple examples
    examples = [
        "Find all entities -> MATCH (e:__Entity__) RETURN e.name LIMIT 10",
        "Show entity relationships -> MATCH (e1:__Entity__)-[r]->(e2:__Entity__) RETURN e1.name, type(r), e2.name LIMIT 5"
    ]
    
    # Simple prompt - no JSON keywords
    prompt = """
    Convert this question to a Cypher query using the provided schema.
    
    Schema: {schema}
    
    Examples: {examples}
    
    Question: {query_text}
    
    Return only the Cypher query, nothing else:
    """
    
    # Create retriever
    retriever = Text2CypherRetriever(
        driver=driver,
        llm=llm,
        neo4j_schema=schema,
        custom_prompt=prompt,
        examples=examples
    )
    
    # Test queries
    test_queries = [
        "Show me all entities",
        "Find entities named TechNova",
        "What entities are related to healthcare?"
    ]
    
    print("🔍 Testing Text2CypherRetriever...")
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        try:
            # Get the generated Cypher (before execution)
            # We'll need to access the internal method to see the raw generation
            
            # Try to get results first
            results = await retriever.get_search_results(query=query, top_k=5)
            print(f"✅ Results: {len(results) if results else 0} items")
            if results:
                print(f"   Sample: {results[0]}")
            else:
                print("   No results returned")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            # Print the actual error details
            if "Unexpected end of input" in str(e):
                print("   This indicates empty string was generated as Cypher")
            elif "Invalid input" in str(e):
                print("   This indicates malformed Cypher was generated")
    
    # Close driver
    driver.close()

if __name__ == "__main__":
    asyncio.run(test_text2cypher())
