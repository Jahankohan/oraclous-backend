#!/usr/bin/env python3

import os
from neo4j_graphrag.llm.openai_llm import OpenAILLM

# Hardcode API key for testing (updated)
os.environ["OPENAI_API_KEY"] = "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"

def test_prompt_generation():
    """Test what the LLM generates with our prompt"""
    
    llm = OpenAILLM(
        model_name="gpt-4o-mini",
        model_params={
            "temperature": 0.1,
            "max_tokens": 1000,
        }
    )
    
    # Schema (same as in benchmark.py)
    schema = """
    # Knowledge Graph Schema for GraphRAG
    
    ## Core Node Types:
    - Document: Represents source documents
      Properties: path (string), metadata (map)
    
    - Chunk: Text chunks from documents  
      Properties: text (string), chunk_index (integer), source (string), embedding (vector)
    
    - __Entity__: Extracted entities
      Properties: name (string), description (string), type (string), embedding (vector)
    
    - Person: Human entities
      Properties: name (string), occupation (string), nationality (string)
    
    - Organization: Organizational entities
      Properties: name (string), industry (string), headquarters (string)
    
    - Concept: Abstract concepts
      Properties: name (string), description (string), domain (string)
    
    ## Key Relationships:
    - FROM_DOCUMENT: Links chunks to their source documents
    - FROM_CHUNK: Links entities to the chunks they were extracted from
    - WORKS_FOR: Person to Organization employment
    - RELATED_TO: General entity relationships
    - SAME_AS: Entity resolution links
    
    ## Vector Indexes Available:
    - text_embeddings_primary: For semantic search on chunks
    - entity_embeddings: For semantic search on entities
    
    ## Fulltext Indexes Available:  
    - chunk_text_fulltext: For keyword search on chunk text
    - entity_text_fulltext: For keyword search on entity names/descriptions
    """
    
    # Examples (same as in benchmark.py)
    examples = [
        "Find all people working for organizations -> MATCH (p:Person)-[:WORKS_FOR]->(o:Organization) RETURN p.name, o.name",
        "Show relationships between entities in specific document -> MATCH (d:Document {path: $document_path})-[:FROM_DOCUMENT]->(c:Chunk)<-[:FROM_CHUNK]-(e1)-[r]->(e2) RETURN e1.name, type(r), e2.name",
        "Find entities similar to concept -> MATCH (e:__Entity__) WHERE e.name CONTAINS $term RETURN e.name, e.description LIMIT 10",
        "Get document context for entity -> MATCH (e:__Entity__ {name: $entity_name})-[:FROM_CHUNK]->(c:Chunk)-[:FROM_DOCUMENT]->(d:Document) RETURN d.path, c.text"
    ]
    
    # Test the exact prompt from benchmark.py
    prompt_template = """
    Generate a Cypher query for a Neo4j graph database based on this natural language request.

    Available Schema:
    {schema}

    Query Examples:
    {examples}

    User Question:
    {query_text}

    Generate a single Cypher query that answers the user's question. The query must:
    - Use only the nodes, relationships, and properties shown in the schema
    - Start with MATCH, CREATE, MERGE, or other valid Cypher keywords
    - Be syntactically correct and executable
    - Return relevant data to answer the question

    Output ONLY the raw Cypher query without any JSON formatting, code blocks, or explanations. Example:
    MATCH (n:Entity) RETURN n LIMIT 10
    """
    
    # Test queries from the failing test
    test_queries = [
        "Show me all people who work at TechNova Corporation",
        "Find companies founded in Austin, Texas", 
        "What are the most mentioned medical concepts?",
        "Show documents and their chunks about healthcare",
        "Find entities related to artificial intelligence or machine learning"
    ]
    
    print("🔍 Testing Cypher Generation with OpenAI LLM...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: {query}")
        print("-" * 40)
        
        # Format the prompt
        formatted_prompt = prompt_template.format(
            schema=schema,
            examples="\n".join(examples),
            query_text=query
        )
        
        try:
            # Generate response
            response = llm.invoke(formatted_prompt)
            
            print(f"Generated Cypher:")
            print(f"```cypher")
            print(response.content.strip())
            print(f"```")
            
            # Check if it's empty or problematic
            if not response.content.strip():
                print("❌ EMPTY RESPONSE!")
            elif response.content.strip().startswith("{"):
                print("⚠️  JSON FORMAT DETECTED (should be raw Cypher)")
            elif not any(keyword in response.content.upper() for keyword in ["MATCH", "CREATE", "MERGE", "CALL", "RETURN"]):
                print("⚠️  NO CYPHER KEYWORDS DETECTED")
            else:
                print("✅ Looks like valid Cypher")
            
        except Exception as e:
            print(f"❌ Error generating Cypher: {e}")

if __name__ == "__main__":
    test_prompt_generation()
