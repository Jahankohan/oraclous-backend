#!/usr/bin/env python3
"""
Example Queries for GraphRAG Testing

This file contains example queries organized by type to help test different
aspects of your GraphRAG knowledge graph system.
"""

# 1. CHUNK/TEXT SIMILARITY QUERIES
# These queries test semantic search over document chunks
CHUNK_QUERIES = [
    "What is artificial intelligence?",
    "Tell me about machine learning algorithms",
    "How does neural network training work?", 
    "What are the benefits of cloud computing?",
    "Explain software development methodologies",
    "What is data science?",
    "How do databases work?",
    "What are microservices?",
    "Explain API design principles",
    "What is blockchain technology?"
]

# 2. ENTITY-FOCUSED QUERIES  
# These queries test entity retrieval and entity-centric search
ENTITY_QUERIES = [
    "Find companies in technology sector",
    "Show me people in artificial intelligence",
    "What organizations are mentioned?", 
    "Tell me about key concepts",
    "Find entities related to software development",
    "Who are the important people mentioned?",
    "What technology companies exist?",
    "Find academic institutions",
    "Show me government organizations",
    "What products or services are discussed?"
]

# 3. RELATIONSHIP/GRAPH TRAVERSAL QUERIES
# These queries test graph-aware context retrieval
RELATIONSHIP_QUERIES = [
    "Show connections between technology companies",
    "Find relationships in artificial intelligence",
    "What are the connections between entities?",
    "Show entity relationships with high confidence", 
    "Find related concepts and their connections",
    "How are companies connected to each other?",
    "What partnerships exist between organizations?",
    "Show employee relationships",
    "Find collaborations between entities",
    "What supply chain relationships exist?"
]

# 4. HYBRID SEARCH QUERIES
# These queries combine vector similarity with keyword matching
HYBRID_QUERIES = [
    "machine learning algorithms",
    "artificial intelligence companies", 
    "software development best practices",
    "cloud computing architecture",
    "data science methodology",
    "cybersecurity frameworks",
    "mobile app development",
    "enterprise software solutions",
    "startup funding strategies",
    "digital transformation initiatives"
]

# 5. NATURAL LANGUAGE TO CYPHER QUERIES
# These queries test the text-to-Cypher conversion capability
TEXT2CYPHER_QUERIES = [
    "Show me all organizations mentioned in the documents",
    "Find people who work for technology companies",
    "What are the most mentioned entities?",
    "Show documents and their chunks",
    "Find entities that appear in multiple chunks",
    "List all relationships between entities",
    "Show me entities with the most connections",
    "Find documents that mention both AI and companies",
    "What types of entities are in the graph?",
    "Show the structure of the knowledge graph"
]

# 6. DIRECT CYPHER ANALYTICS QUERIES
# These are direct Cypher queries for advanced analytics
ANALYTICS_QUERIES = {
    "basic_stats": {
        "description": "Basic graph statistics",
        "query": """
        MATCH (d:Document) 
        OPTIONAL MATCH (d)-[:FROM_DOCUMENT]->(c:Chunk)
        OPTIONAL MATCH (c)<-[:FROM_CHUNK]-(e:__Entity__)
        OPTIONAL MATCH (e)-[r]->(re:__Entity__)
        RETURN 
            count(DISTINCT d) as documents,
            count(DISTINCT c) as chunks, 
            count(DISTINCT e) as entities,
            count(DISTINCT r) as relationships
        """
    },
    
    "top_entities": {
        "description": "Most frequently mentioned entities",
        "query": """
        MATCH (e:__Entity__)-[:FROM_CHUNK]->(c:Chunk)
        RETURN e.name as entity_name, 
               count(c) as mentions,
               collect(DISTINCT labels(e)[0]) as entity_types
        ORDER BY mentions DESC
        LIMIT 10
        """
    },
    
    "entity_relationships": {
        "description": "Entities with most relationships",
        "query": """
        MATCH (e:__Entity__)-[r]->(re:__Entity__)
        RETURN e.name as entity,
               count(r) as relationship_count,
               collect(DISTINCT type(r)) as relationship_types,
               collect(DISTINCT re.name)[0..5] as sample_related_entities
        ORDER BY relationship_count DESC
        LIMIT 10
        """
    },
    
    "document_coverage": {
        "description": "Entity coverage per document",
        "query": """
        MATCH (d:Document)-[:FROM_DOCUMENT]->(c:Chunk)<-[:FROM_CHUNK]-(e:__Entity__)
        RETURN d.path as document,
               count(DISTINCT c) as chunks,
               count(DISTINCT e) as unique_entities,
               count(e) as total_entity_mentions,
               round(count(e) * 1.0 / count(DISTINCT c), 2) as entities_per_chunk
        ORDER BY unique_entities DESC
        """
    },
    
    "entity_resolution": {
        "description": "Entity resolution analysis",
        "query": """
        MATCH (e1:__Entity__)-[:SAME_AS]-(e2:__Entity__)
        RETURN e1.name as entity_name,
               count(*) as resolution_links,
               collect(DISTINCT elementId(e2))[0..3] as sample_resolved_ids
        ORDER BY resolution_links DESC
        LIMIT 10
        """
    },
    
    "chunk_analysis": {
        "description": "Chunk analysis with entity density",
        "query": """
        MATCH (c:Chunk)
        OPTIONAL MATCH (c)<-[:FROM_CHUNK]-(e:__Entity__)
        RETURN c.chunk_index as chunk_index,
               c.source as source,
               length(c.text) as text_length,
               count(e) as entity_count,
               round(count(e) * 1000.0 / length(c.text), 2) as entities_per_1000_chars
        ORDER BY entity_count DESC
        LIMIT 15
        """
    },
    
    "relationship_types": {
        "description": "Analysis of relationship types",
        "query": """
        MATCH (e1:__Entity__)-[r]->(e2:__Entity__)
        RETURN type(r) as relationship_type,
               count(*) as frequency,
               avg(r.confidence) as avg_confidence,
               collect(DISTINCT e1.name)[0..3] as sample_source_entities,
               collect(DISTINCT e2.name)[0..3] as sample_target_entities
        ORDER BY frequency DESC
        """
    },
    
    "entity_types_distribution": {
        "description": "Distribution of entity types",
        "query": """
        MATCH (e:__Entity__)
        RETURN labels(e)[0] as entity_type,
               count(*) as count,
               collect(DISTINCT e.name)[0..5] as sample_entities
        ORDER BY count DESC
        """
    },
    
    "embedding_coverage": {
        "description": "Check embedding coverage",
        "query": """
        MATCH (c:Chunk)
        WITH count(c) as total_chunks
        MATCH (c:Chunk) WHERE c.embedding IS NOT NULL
        WITH total_chunks, count(c) as chunks_with_embeddings
        MATCH (e:__Entity__)
        WITH total_chunks, chunks_with_embeddings, count(e) as total_entities
        MATCH (e:__Entity__) WHERE e.embedding IS NOT NULL
        RETURN total_chunks,
               chunks_with_embeddings,
               round(100.0 * chunks_with_embeddings / total_chunks, 1) as chunk_embedding_percentage,
               total_entities,
               count(e) as entities_with_embeddings,
               round(100.0 * count(e) / total_entities, 1) as entity_embedding_percentage
        """
    },
    
    "document_entities_network": {
        "description": "Document-entity network analysis",
        "query": """
        MATCH (d1:Document)-[:FROM_DOCUMENT]->(c1:Chunk)<-[:FROM_CHUNK]-(e:__Entity__)-[:FROM_CHUNK]->(c2:Chunk)<-[:FROM_DOCUMENT]-(d2:Document)
        WHERE d1 <> d2
        RETURN d1.path as document1,
               d2.path as document2,
               count(DISTINCT e) as shared_entities,
               collect(DISTINCT e.name)[0..5] as sample_shared_entities
        ORDER BY shared_entities DESC
        LIMIT 10
        """
    }
}

# 7. COMPLEX ANALYTICAL QUERIES
# These test complex scenarios combining multiple concepts
COMPLEX_QUERIES = [
    "Find technology companies that work with artificial intelligence and their key people",
    "Show me relationships between organizations in the same industry",
    "What are the main topics discussed across all documents?",
    "Find entities that bridge different document contexts",
    "Analyze the network of relationships in the technology sector",
    "Show me the most influential entities based on their connections",
    "Find clusters of related concepts in the knowledge graph",
    "What are the key themes that connect different documents?",
    "Identify potential partnerships based on entity relationships",
    "Show me the evolution of topics across different document sources"
]

# 8. PERFORMANCE TEST QUERIES
# These are designed to test system performance and scalability
PERFORMANCE_QUERIES = [
    "artificial intelligence machine learning neural networks deep learning",
    "technology companies software development programming languages",
    "data science analytics visualization business intelligence", 
    "cloud computing distributed systems microservices architecture",
    "cybersecurity privacy encryption blockchain digital transformation"
]

# Helper functions to get query sets
def get_queries_by_type(query_type: str):
    """Get queries by type"""
    query_sets = {
        'chunk': CHUNK_QUERIES,
        'entity': ENTITY_QUERIES,
        'relationship': RELATIONSHIP_QUERIES,
        'hybrid': HYBRID_QUERIES,
        'text2cypher': TEXT2CYPHER_QUERIES,
        'complex': COMPLEX_QUERIES,
        'performance': PERFORMANCE_QUERIES
    }
    return query_sets.get(query_type, [])

def get_analytics_query(query_name: str):
    """Get specific analytics query"""
    return ANALYTICS_QUERIES.get(query_name)

def list_query_types():
    """List available query types"""
    return ['chunk', 'entity', 'relationship', 'hybrid', 'text2cypher', 'complex', 'performance']

def list_analytics_queries():
    """List available analytics queries"""
    return list(ANALYTICS_QUERIES.keys())

if __name__ == "__main__":
    print("GraphRAG Example Queries")
    print("=" * 40)
    print("\nAvailable Query Types:")
    for qtype in list_query_types():
        count = len(get_queries_by_type(qtype))
        print(f"  - {qtype}: {count} queries")
    
    print(f"\nAvailable Analytics Queries:")
    for qname in list_analytics_queries():
        desc = ANALYTICS_QUERIES[qname]['description']
        print(f"  - {qname}: {desc}")
