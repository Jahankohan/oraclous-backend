#!/usr/bin/env python3
"""
Simple Entity Resolution for GraphRAG Pipeline
Creates SAME_AS relationships between duplicate entities
"""

from neo4j import GraphDatabase
from collections import defaultdict


def resolve_duplicate_entities():
    """
    Simple entity resolution: Create SAME_AS relationships between exact name matches
    """
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', ''))
    
    print("🔄 Starting Simple Entity Resolution...")
    
    with driver.session() as session:
        # Find entities with identical names in different chunks
        result = session.run("""
            MATCH (e1:__Entity__)-[:FROM_CHUNK]->(c1:Chunk)
            MATCH (e2:__Entity__)-[:FROM_CHUNK]->(c2:Chunk)
            WHERE e1.name = e2.name 
            AND c1.index <> c2.index
            AND elementId(e1) < elementId(e2)  // Avoid duplicates
            AND NOT (e1)-[:SAME_AS]-(e2)  // Don't create if already exists
            RETURN e1.name as entity_name,
                   elementId(e1) as e1_id,
                   elementId(e2) as e2_id,
                   c1.index as chunk1,
                   c2.index as chunk2
        """)
        
        matches = list(result)
        print(f"📊 Found {len(matches)} entity pairs to link")
        
        # Create SAME_AS relationships
        links_created = 0
        for record in matches:
            try:
                session.run("""
                    MATCH (e1) WHERE elementId(e1) = $e1_id
                    MATCH (e2) WHERE elementId(e2) = $e2_id
                    MERGE (e1)-[:SAME_AS {created_by: 'entity_resolution'}]-(e2)
                """, e1_id=record['e1_id'], e2_id=record['e2_id'])
                
                links_created += 1
                print(f"   ✅ Linked '{record['entity_name']}' between Chunk {record['chunk1']} ↔ Chunk {record['chunk2']}")
                
            except Exception as e:
                print(f"   ❌ Failed to link {record['entity_name']}: {e}")
        
        print(f"\n🎉 Entity Resolution Complete!")
        print(f"📊 Created {links_created} SAME_AS relationships")
        
        # Verify the results
        print("\n🔍 Verification - Cross-chunk entity connectivity:")
        verification = session.run("""
            MATCH (e1:__Entity__)-[:SAME_AS]-(e2:__Entity__)
            MATCH (e1)-[:FROM_CHUNK]->(c1:Chunk)
            MATCH (e2)-[:FROM_CHUNK]->(c2:Chunk)
            RETURN e1.name as entity_name,
                   c1.index as chunk1,
                   c2.index as chunk2
            ORDER BY entity_name
        """)
        
        for record in verification:
            print(f"   🔗 '{record['entity_name']}': Chunk {record['chunk1']} ↔ Chunk {record['chunk2']}")
    
    driver.close()


def analyze_entity_connectivity():
    """
    Analyze how entity resolution improved graph connectivity
    """
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', ''))
    
    print("\n📈 Analyzing Entity Connectivity Improvements...")
    
    with driver.session() as session:
        # Count total SAME_AS relationships
        same_as_count = session.run("""
            MATCH ()-[:SAME_AS]-()
            RETURN count(*) / 2 as same_as_relationships
        """).single()['same_as_relationships']
        
        # Find entities that can now be traversed across chunks
        traversal_paths = session.run("""
            MATCH path = (c1:Chunk)<-[:FROM_CHUNK]-(e1:__Entity__)-[:SAME_AS]-(e2:__Entity__)-[:FROM_CHUNK]->(c2:Chunk)
            WHERE c1.index <> c2.index
            RETURN e1.name as entity_name,
                   c1.index as chunk1,
                   c2.index as chunk2,
                   length(path) as path_length
            ORDER BY entity_name
            LIMIT 10
        """)
        
        print(f"🔗 SAME_AS relationships created: {same_as_count}")
        print("🚶 Sample cross-chunk entity traversal paths:")
        
        for record in traversal_paths:
            print(f"   📍 '{record['entity_name']}': Chunk {record['chunk1']} ↔ Chunk {record['chunk2']} (path length: {record['path_length']})")
        
        # Check for potential semantic relationships across chunks
        semantic_opportunities = session.run("""
            MATCH (e1:__Entity__)-[:SAME_AS]-(e2:__Entity__)
            MATCH (e1)-[r1]->(related1)
            MATCH (e2)-[r2]->(related2)
            WHERE type(r1) = type(r2)
            AND related1.name <> related2.name
            RETURN e1.name as entity_name,
                   type(r1) as relationship_type,
                   related1.name as related_entity1,
                   related2.name as related_entity2
            LIMIT 5
        """)
        
        print("\n🧠 Potential cross-chunk semantic relationships discovered:")
        for record in semantic_opportunities:
            print(f"   🔍 '{record['entity_name']}' {record['relationship_type']} '{record['related_entity1']}' AND '{record['related_entity2']}'")
    
    driver.close()


if __name__ == "__main__":
    resolve_duplicate_entities()
    analyze_entity_connectivity()
