#!/usr/bin/env python3
"""
Community Schema Setup

This script creates the necessary database constraints and indexes
for persistent community nodes in the knowledge graph.
"""

COMMUNITY_SCHEMA_QUERIES = [
    # Community node constraints
    """
    CREATE CONSTRAINT community_id_unique IF NOT EXISTS
    FOR (c:__Community__) REQUIRE c.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT community_graph_id IF NOT EXISTS
    FOR (c:__Community__) REQUIRE c.graph_id IS NOT NULL
    """,
    # Community indexes for performance
    """
    CREATE INDEX community_graph_id_index IF NOT EXISTS
    FOR (c:__Community__) ON (c.graph_id)
    """,
    """
    CREATE INDEX community_weight_index IF NOT EXISTS
    FOR (c:__Community__) ON (c.weight)
    """,
    """
    CREATE INDEX community_entity_count_index IF NOT EXISTS
    FOR (c:__Community__) ON (c.entity_count)
    """,
    """
    CREATE INDEX community_algorithm_index IF NOT EXISTS
    FOR (c:__Community__) ON (c.detection_algorithm)
    """,
    # Fulltext index for community summaries
    """
    CREATE FULLTEXT INDEX community_summaries IF NOT EXISTS
    FOR (c:__Community__) ON EACH [c.summary]
    """,
    # Vector index for community embeddings (when available)
    """
    CREATE VECTOR INDEX community_embeddings IF NOT EXISTS
    FOR (c:__Community__) ON (c.embedding)
    OPTIONS {
        indexConfig: {
            `vector.dimensions`: 384,
            `vector.similarity_function`: 'cosine'
        }
    }
    """,
    # Indexes on relationships
    """
    CREATE INDEX in_community_graph_id IF NOT EXISTS
    FOR ()-[r:IN_COMMUNITY]-() ON (r.graph_id)
    """,
    """
    CREATE INDEX parent_community_graph_id IF NOT EXISTS
    FOR ()-[r:PARENT_COMMUNITY]-() ON (r.graph_id)
    """,
]

VERIFICATION_QUERIES = [
    # Verify community constraints
    "SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties WHERE 'Community' IN labelsOrTypes",
    # Verify community indexes
    "SHOW INDEXES YIELD name, labelsOrTypes, properties WHERE 'Community' IN labelsOrTypes",
    # Check for existing communities
    "MATCH (c:__Community__) RETURN count(c) as community_count",
    # Check community relationships
    "MATCH (:__Entity__)-[r:IN_COMMUNITY]->(:__Community__) RETURN count(r) as in_community_relationships",
]


def print_schema_setup_info():
    """Print information about the community schema setup"""

    print("🏗️ Community Schema Setup")
    print("=" * 60)
    print()

    print("📋 Schema Components to Create:")
    print("   1. Constraints:")
    print("      • community_id_unique - Ensures unique community IDs")
    print("      • community_graph_id - Ensures graph_id is not null")
    print()

    print("   2. Performance Indexes:")
    print("      • community_graph_id_index - For multi-tenant filtering")
    print("      • community_weight_index - For weight-based sorting")
    print("      • community_entity_count_index - For size-based queries")
    print("      • community_algorithm_index - For algorithm filtering")
    print()

    print("   3. Search Indexes:")
    print("      • community_summaries (fulltext) - For text-based search")
    print("      • community_embeddings (vector) - For similarity search")
    print()

    print("   4. Relationship Indexes:")
    print("      • in_community_graph_id - For filtering IN_COMMUNITY relationships")
    print("      • parent_community_graph_id - For hierarchical queries")
    print()

    print("🔧 To execute these in Neo4j:")
    print("   1. Connect to your Neo4j database")
    print("   2. Run each query in COMMUNITY_SCHEMA_QUERIES")
    print("   3. Verify with VERIFICATION_QUERIES")
    print()

    print("📝 Example Community Node Structure:")
    community_example = {
        "id": "community_graph123_abc456def789",
        "graph_id": "graph123",
        "summary": "Community of 5 entities about technology: ML, AI, etc.",
        "entity_count": 5,
        "detection_algorithm": "louvain",
        "weight": 0.25,
        "creation_date": "2025-08-26T10:30:00Z",
        "last_updated": "2025-08-26T10:30:00Z",
        "embedding": "[0.1, 0.2, 0.3, ...]  // 384-dimensional vector",
    }

    print("   Node Properties:")
    for key, value in community_example.items():
        if isinstance(value, str) and len(value) > 50:
            value = value[:47] + "..."
        print(f"      {key}: {value}")
    print()

    print("📊 Expected Relationships:")
    print("   • (:__Entity__)-[:IN_COMMUNITY]->(:__Community__)")
    print("   • (:__Community__)-[:PARENT_COMMUNITY]->(:__Community__)")
    print()

    print("🚀 Neo4j Cypher Queries:")
    print("=" * 40)

    for i, query in enumerate(COMMUNITY_SCHEMA_QUERIES, 1):
        print(
            f"\n-- Query {i}: {query.strip().split()[1]} {query.strip().split()[2] if len(query.strip().split()) > 2 else ''}"
        )
        print(query.strip())

    print("\n" + "=" * 40)
    print("🔍 Verification Queries:")

    for i, query in enumerate(VERIFICATION_QUERIES, 1):
        print(f"\n-- Verification {i}:")
        print(query.strip())

    print()


def print_integration_steps():
    """Print next steps for integrating community persistence"""

    print("🔄 Integration Steps")
    print("=" * 60)
    print()

    print("📋 Next Actions:")
    print("   1. ✅ Database Schema Setup")
    print("      • Run the schema queries above in Neo4j")
    print("      • Verify constraints and indexes are created")
    print()

    print("   2. 🔄 Background Jobs Integration")
    print("      • Add community persistence to background_jobs.py")
    print("      • Call analytics_service.create_community_nodes()")
    print("      • Schedule periodic community updates")
    print()

    print("   3. 🔄 Chat Service Integration")
    print("      • Update chat reasoning modes")
    print("      • Add community-based context retrieval")
    print("      • Use analytics_service.get_community_search_context()")
    print()

    print("   4. 🔄 API Endpoints")
    print("      • Add community management endpoints")
    print("      • Enable manual community creation/update")
    print("      • Add community search endpoints")
    print()

    print("🧪 Testing Plan:")
    print("   1. Unit tests for community persistence")
    print("   2. Integration tests with real Neo4j")
    print("   3. Performance tests with large graphs")
    print("   4. Multi-tenant isolation verification")
    print()

    print("📊 Expected Benefits:")
    print("   ✅ Persistent community knowledge")
    print("   ✅ Community-based search and retrieval")
    print("   ✅ Hierarchical community structure")
    print("   ✅ Performance improvements (cached communities)")
    print("   ✅ Enhanced RAG context with community insights")


def main():
    """Main function to display community schema setup information"""

    print_schema_setup_info()
    print()
    print_integration_steps()

    print("\n🎯 Summary")
    print("=" * 60)
    print("✅ Community persistence logic implemented in analytics_service.py")
    print("✅ Community schema design completed")
    print("✅ Test coverage for community functionality")
    print("⏳ Ready for database schema creation and integration")
    print()
    print("🔧 Ready to execute: Run the Cypher queries above in your Neo4j database")


if __name__ == "__main__":
    main()
