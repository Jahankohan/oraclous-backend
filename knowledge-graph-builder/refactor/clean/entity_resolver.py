"""Multi-algorithm entity resolution - exact same functionality as original."""

import time
from typing import Any

from neo4j_graphrag.experimental.components.resolver import (
    FuzzyMatchResolver,
    SinglePropertyExactMatchResolver,
    SpaCySemanticMatchResolver,
)


class SimpleEntityResolver:
    """Simple entity resolution that creates SAME_AS relationships between duplicate entities - exact copy from original"""

    def __init__(self, neo4j_driver):
        self.driver = neo4j_driver

    async def run(self) -> dict[str, Any]:
        """Run simple entity resolution"""
        print("🔄 Running Simple Entity Resolution...")

        with self.driver.session() as session:
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

            # Create SAME_AS relationships
            links_created = 0
            for record in matches:
                try:
                    session.run(
                        """
                        MATCH (e1) WHERE elementId(e1) = $e1_id
                        MATCH (e2) WHERE elementId(e2) = $e2_id
                        MERGE (e1)-[:SAME_AS {created_by: 'entity_resolution'}]-(e2)
                    """,
                        e1_id=record["e1_id"],
                        e2_id=record["e2_id"],
                    )
                    links_created += 1
                    print(
                        f"   ✅ Linked '{record['entity_name']}' between Chunk {record['chunk1']} ↔ Chunk {record['chunk2']}"
                    )

                except Exception as e:
                    print(f"   ❌ Failed to link {record['entity_name']}: {e}")

            print(
                f"🎉 Entity Resolution Complete! Created {links_created} SAME_AS relationships"
            )

            return {
                "entities_resolved": links_created,
                "method": "simple_same_as_linking",
            }


class MultiAlgorithmEntityResolver:
    """Advanced entity resolution using multiple algorithms"""

    def __init__(self, driver, config):
        self.driver = driver
        self.config = config
        import logging

        self.logger = logging.getLogger(__name__)

        # Initialize resolvers
        self.resolvers = []

        if config.enable_entity_resolution:
            # Exact match resolver
            self.resolvers.append(
                SinglePropertyExactMatchResolver(driver=driver, resolve_property="name")
            )

            # Semantic similarity resolver
            self.resolvers.append(
                SpaCySemanticMatchResolver(
                    driver=driver,
                    similarity_threshold=config.similarity_threshold,
                    resolve_properties=["name", "description"],
                )
            )

            # Fuzzy match resolver
            self.resolvers.append(
                FuzzyMatchResolver(
                    driver=driver, similarity_threshold=config.fuzzy_threshold
                )
            )

    async def resolve_entities(self) -> dict[str, Any]:
        """Run multi-algorithm entity resolution"""
        if not self.config.enable_entity_resolution:
            return {"entities_resolved": 0, "resolution_methods": []}

        resolution_results = []
        total_resolved = 0

        for i, resolver in enumerate(self.resolvers):
            try:
                self.logger.info(
                    f"Running entity resolution with algorithm {i + 1}/{len(self.resolvers)}"
                )
                start_time = time.time()

                result = await resolver.run()
                duration = time.time() - start_time

                resolved_count = (
                    getattr(result, "entities_resolved", 0)
                    if hasattr(result, "entities_resolved")
                    else 0
                )
                total_resolved += resolved_count

                resolution_results.append(
                    {
                        "algorithm": resolver.__class__.__name__,
                        "entities_resolved": resolved_count,
                        "duration_seconds": duration,
                    }
                )

                self.logger.info(
                    f"Algorithm {resolver.__class__.__name__} resolved {resolved_count} entities in {duration:.2f}s"
                )

            except Exception as e:
                self.logger.error(
                    f"Entity resolution failed for {resolver.__class__.__name__}: {e}"
                )
                resolution_results.append(
                    {
                        "algorithm": resolver.__class__.__name__,
                        "entities_resolved": 0,
                        "duration_seconds": 0,
                        "error": str(e),
                    }
                )

        return {
            "total_entities_resolved": total_resolved,
            "resolution_methods": resolution_results,
        }
