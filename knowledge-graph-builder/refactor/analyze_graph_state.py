#!/usr/bin/env python3
"""
Graph State Analyzer - Check what's currently in the Neo4j database
Run this to see the current state of your knowledge graph without needing OpenAI
"""

import logging

from neo4j import GraphDatabase

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphStateAnalyzer:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password=""):
        """Initialize the Neo4j driver"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the driver"""
        self.driver.close()

    def get_database_stats(self):
        """Get basic database statistics"""
        with self.driver.session() as session:
            # Get total counts first
            total_nodes = session.run("MATCH (n) RETURN count(n) as count").single()[
                "count"
            ]
            total_rels = session.run(
                "MATCH ()-[r]->() RETURN count(r) as count"
            ).single()["count"]

            # Get node counts by label - simplified approach
            node_counts = []
            try:
                labels = session.run("CALL db.labels()").data()
                for label_record in labels:
                    label = label_record["label"]
                    count = session.run(
                        f"MATCH (n:`{label}`) RETURN count(n) as count"
                    ).single()["count"]
                    node_counts.append({"label": label, "count": count})
                node_counts.sort(key=lambda x: x["count"], reverse=True)
            except Exception as e:
                logger.warning(f"Could not get node counts by label: {e}")

            # Get relationship counts by type - simplified approach
            rel_counts = []
            try:
                rel_types = session.run("CALL db.relationshipTypes()").data()
                for rel_record in rel_types:
                    rel_type = rel_record["relationshipType"]
                    count = session.run(
                        f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) as count"
                    ).single()["count"]
                    rel_counts.append({"relationshipType": rel_type, "count": count})
                rel_counts.sort(key=lambda x: x["count"], reverse=True)
            except Exception as e:
                logger.warning(f"Could not get relationship counts by type: {e}")

            return {
                "total_nodes": total_nodes,
                "total_relationships": total_rels,
                "node_counts_by_label": node_counts,
                "relationship_counts_by_type": rel_counts,
            }

    def check_entity_chunk_connections(self):
        """Check for entity-chunk connections (FROM_CHUNK relationships)"""
        with self.driver.session() as session:
            # Check for FROM_CHUNK relationships
            from_chunk_query = """
                MATCH (entity)-[r:FROM_CHUNK]->(chunk)
                RETURN
                    labels(entity) as entity_labels,
                    entity.name as entity_name,
                    chunk.text[..100] + "..." as chunk_preview,
                    chunk.index as chunk_index
                ORDER BY chunk.index, entity.name
                LIMIT 20
            """
            from_chunk_results = session.run(from_chunk_query).data()

            # Check for FROM_DOCUMENT relationships
            from_doc_query = """
                MATCH (chunk)-[r:FROM_DOCUMENT]->(doc)
                RETURN
                    chunk.index as chunk_index,
                    chunk.text[..100] + "..." as chunk_preview,
                    doc.id as document_id
                ORDER BY chunk.index
                LIMIT 10
            """
            from_doc_results = session.run(from_doc_query).data()

            # Check complete traceability chains
            trace_query = """
                MATCH (entity)-[:FROM_CHUNK]->(chunk)-[:FROM_DOCUMENT]->(doc)
                RETURN
                    labels(entity) as entity_labels,
                    entity.name as entity_name,
                    chunk.index as chunk_index,
                    doc.id as document_id
                ORDER BY chunk.index, entity.name
                LIMIT 20
            """
            trace_results = session.run(trace_query).data()

            return {
                "from_chunk_connections": from_chunk_results,
                "from_document_connections": from_doc_results,
                "complete_traceability_chains": trace_results,
            }

    def get_sample_entities(self):
        """Get sample entities and their properties"""
        with self.driver.session() as session:
            entities_query = """
                MATCH (n)
                WHERE n.name IS NOT NULL
                RETURN
                    labels(n) as labels,
                    n.name as name,
                    keys(n) as properties
                ORDER BY n.name
                LIMIT 15
            """
            return session.run(entities_query).data()

    def get_schema_info(self):
        """Get schema information"""
        with self.driver.session() as session:
            try:
                schema = session.run("CALL db.schema.visualization()").single()
                return {
                    "nodes": schema["nodes"] if schema else [],
                    "relationships": schema["relationships"] if schema else [],
                }
            except Exception as e:
                logger.warning(f"Could not get schema visualization: {e}")
                return {"nodes": [], "relationships": []}


def main():
    print("🔍 Analyzing Neo4j Knowledge Graph State")
    print("=" * 50)

    analyzer = GraphStateAnalyzer()

    try:
        # 1. Get basic database statistics
        print("\n📊 Database Statistics:")
        stats = analyzer.get_database_stats()
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Relationships: {stats['total_relationships']}")

        if stats["node_counts_by_label"]:
            print("\nNode Counts by Label:")
            for item in stats["node_counts_by_label"]:
                print(f"  {item['label']}: {item['count']}")

        if stats["relationship_counts_by_type"]:
            print("\nRelationship Counts by Type:")
            for item in stats["relationship_counts_by_type"]:
                print(f"  {item['relationshipType']}: {item['count']}")

        # 2. Check entity-chunk connections
        print("\n🔗 Entity-Chunk Connection Analysis:")
        connections = analyzer.check_entity_chunk_connections()

        print(
            f"FROM_CHUNK relationships found: {len(connections['from_chunk_connections'])}"
        )
        if connections["from_chunk_connections"]:
            print("\nSample FROM_CHUNK connections:")
            for conn in connections["from_chunk_connections"][:5]:
                print(
                    f"  {conn['entity_labels'][0] if conn['entity_labels'] else 'Unknown'}: {conn['entity_name']} -> Chunk {conn['chunk_index']}"
                )

        print(
            f"\nFROM_DOCUMENT relationships found: {len(connections['from_document_connections'])}"
        )
        if connections["from_document_connections"]:
            print("\nSample FROM_DOCUMENT connections:")
            for conn in connections["from_document_connections"][:5]:
                print(
                    f"  Chunk {conn['chunk_index']} -> Document {conn['document_id']}"
                )

        print(
            f"\nComplete traceability chains: {len(connections['complete_traceability_chains'])}"
        )
        if connections["complete_traceability_chains"]:
            print("\nSample complete chains (Entity -> Chunk -> Document):")
            for chain in connections["complete_traceability_chains"][:5]:
                entity_label = (
                    chain["entity_labels"][0] if chain["entity_labels"] else "Unknown"
                )
                print(
                    f"  {entity_label}: {chain['entity_name']} -> Chunk {chain['chunk_index']} -> Doc {chain['document_id']}"
                )

        # 3. Get sample entities
        print("\n👥 Sample Entities:")
        entities = analyzer.get_sample_entities()
        if entities:
            for entity in entities[:10]:
                labels_str = (
                    ":".join(entity["labels"]) if entity["labels"] else "Unknown"
                )
                print(f"  {labels_str}: {entity['name']}")
        else:
            print("  No entities found with names")

        # 4. Schema info
        print("\n🏗️  Schema Information:")
        schema = analyzer.get_schema_info()
        if schema["nodes"]:
            print("Node Types:", [node.get("labels", []) for node in schema["nodes"]])
        if schema["relationships"]:
            print(
                "Relationship Types:",
                [rel.get("type") for rel in schema["relationships"]],
            )

        # 5. Final assessment
        print("\n✅ Assessment:")
        if stats["total_nodes"] == 0:
            print(
                "❌ No data found in database - pipeline hasn't processed any documents yet"
            )
        elif len(connections["from_chunk_connections"]) == 0:
            print(
                "⚠️  Entities found but no FROM_CHUNK relationships - connection issue detected"
            )
        elif len(connections["complete_traceability_chains"]) > 0:
            print(
                "✅ Complete traceability chains found - entity-chunk-document connections working!"
            )
        else:
            print("⚠️  Partial data found - investigate further")

    except Exception as e:
        print(f"❌ Error analyzing graph: {e}")
    finally:
        analyzer.close()


if __name__ == "__main__":
    main()
