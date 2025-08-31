#!/usr/bin/env python3
"""
Analyze duplicate entities in the knowledge graph
"""

from neo4j import GraphDatabase
from collections import defaultdict

def analyze_duplicate_entities():
    """Analyze and report duplicate entities"""
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    
    print("🔍 Analyzing Duplicate Entities")
    print("=" * 50)
    
    with driver.session() as session:
        # Find entities with same names
        result = session.run("""
            MATCH (e:__Entity__)
            WHERE e.name IS NOT NULL
            RETURN e.name as name, 
                   labels(e) as labels,
                   elementId(e) as id,
                   e.type as type
            ORDER BY e.name
        """)
        
        entities_by_name = defaultdict(list)
        
        for record in result:
            name = record['name']
            labels = record['labels']
            entity_id = record['id']
            entity_type = record['type']
            
            entities_by_name[name].append({
                'id': entity_id,
                'labels': labels,
                'type': entity_type
            })
        
        print("\n📊 Entity Name Analysis:")
        print(f"Total unique entity names: {len(entities_by_name)}")
        
        # Find duplicates
        duplicates = {name: entities for name, entities in entities_by_name.items() if len(entities) > 1}
        
        print(f"Duplicate entity names: {len(duplicates)}")
        print("\n🔄 Duplicate Entities Found:")
        
        for name, entities in duplicates.items():
            print(f"\n📝 Entity: '{name}' ({len(entities)} instances)")
            for i, entity in enumerate(entities, 1):
                print(f"   {i}. ID: {entity['id'][:20]}... | Labels: {entity['labels']} | Type: {entity['type']}")
                
                # Check which chunks each entity is connected to
                chunk_result = session.run("""
                    MATCH (e)-[:FROM_CHUNK]->(c:Chunk)
                    WHERE elementId(e) = $entity_id
                    RETURN c.index as chunk_index, substring(c.text, 0, 100) + '...' as chunk_preview
                """, entity_id=entity['id'])
                
                for chunk_record in chunk_result:
                    chunk_idx = chunk_record['chunk_index']
                    chunk_preview = chunk_record['chunk_preview']
                    print(f"      → Connected to Chunk {chunk_idx}: {chunk_preview}")
        
        # Analyze connectivity impact
        print("\n🔗 Connectivity Impact Analysis:")
        
        # Check if duplicates prevent cross-chunk entity relationships
        cross_chunk_missing = session.run("""
            // Find entities with same names in different chunks that should be connected
            MATCH (e1:__Entity__)-[:FROM_CHUNK]->(c1:Chunk)
            MATCH (e2:__Entity__)-[:FROM_CHUNK]->(c2:Chunk)
            WHERE e1.name = e2.name 
            AND c1.index <> c2.index
            AND NOT (e1)-[:SAME_AS]-(e2)  // Should be connected but aren't
            RETURN e1.name as entity_name,
                   c1.index as chunk1,
                   c2.index as chunk2,
                   elementId(e1) as e1_id,
                   elementId(e2) as e2_id
            LIMIT 10
        """)
        
        missing_connections = list(cross_chunk_missing)
        if missing_connections:
            print(f"❌ Found {len(missing_connections)} entities that should be connected across chunks:")
            for record in missing_connections:
                print(f"   • '{record['entity_name']}' in Chunk {record['chunk1']} vs Chunk {record['chunk2']}")
        else:
            print("✅ No obvious missing cross-chunk entity connections detected")
        
        # Analyze potential entity resolution strategies
        print("\n🛠️  Entity Resolution Strategy Recommendations:")
        print("1. **Name-based matching**: Merge entities with identical names")
        print("2. **Fuzzy matching**: Handle variations like 'TechNova' vs 'TechNova Corporation'")
        print("3. **Type-aware merging**: Only merge entities with compatible types")
        print("4. **Cross-chunk linking**: Create SAME_AS relationships between duplicate entities")
        
    driver.close()

if __name__ == "__main__":
    analyze_duplicate_entities()
