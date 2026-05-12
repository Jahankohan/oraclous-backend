#!/usr/bin/env python3
"""
Entity Resolution Component for GraphRAG Pipeline
Handles entity deduplication and cross-chunk entity linking
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass

from neo4j import GraphDatabase


@dataclass
class EntityMatch:
    """Represents a potential entity match for resolution"""

    entity1_id: str
    entity2_id: str
    entity1_name: str
    entity2_name: str
    similarity_score: float
    match_type: str  # "exact", "fuzzy", "semantic"
    entity1_chunk: int
    entity2_chunk: int


class EntityResolver:
    """
    Resolves duplicate entities across chunks and creates unified entity relationships
    """

    def __init__(self, neo4j_driver, similarity_threshold: float = 0.85):
        self.driver = neo4j_driver
        self.similarity_threshold = similarity_threshold

    def resolve_entities(self) -> dict[str, any]:
        """
        Main entity resolution process
        Returns statistics about resolution performed
        """
        print("🔄 Starting Entity Resolution Process...")

        # Step 1: Find potential duplicate entities
        duplicates = self._find_duplicate_entities()

        # Step 2: Score and match entities
        matches = self._score_entity_matches(duplicates)

        # Step 3: Merge high-confidence matches
        merge_stats = self._merge_entities(matches)

        # Step 4: Create cross-chunk relationships for remaining similar entities
        linking_stats = self._create_cross_chunk_links(matches)

        return {
            "duplicates_found": len(duplicates),
            "matches_scored": len(matches),
            "entities_merged": merge_stats["merged"],
            "links_created": linking_stats["links_created"],
            "processing_time": merge_stats.get("processing_time", 0),
        }

    def _find_duplicate_entities(self) -> dict[str, list[dict]]:
        """Find entities with same or similar names"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (e:__Entity__)
                WHERE e.name IS NOT NULL
                RETURN elementId(e) as id,
                       e.name as name,
                       labels(e) as labels,
                       e.type as type,
                       [(e)-[:FROM_CHUNK]->(c:Chunk) | c.index][0] as chunk_index
                ORDER BY e.name
            """)

            entities_by_name = defaultdict(list)

            for record in result:
                name = record["name"].strip().lower()  # Normalize for comparison
                entities_by_name[name].append(
                    {
                        "id": record["id"],
                        "original_name": record["name"],
                        "normalized_name": name,
                        "labels": record["labels"],
                        "type": record["type"],
                        "chunk_index": record["chunk_index"],
                    }
                )

            # Filter to only groups with duplicates
            duplicates = {
                name: entities
                for name, entities in entities_by_name.items()
                if len(entities) > 1
            }

            print(f"📊 Found {len(duplicates)} entity names with duplicates")
            return duplicates

    def _score_entity_matches(
        self, duplicates: dict[str, list[dict]]
    ) -> list[EntityMatch]:
        """Score potential entity matches"""
        matches = []

        for name, entities in duplicates.items():
            # For exact name matches, create high-confidence matches
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    entity1, entity2 = entities[i], entities[j]

                    # Exact name match
                    if entity1["original_name"] == entity2["original_name"]:
                        similarity = 1.0
                        match_type = "exact"
                    else:
                        # Simple fuzzy matching (can be enhanced)
                        similarity = self._calculate_name_similarity(
                            entity1["original_name"], entity2["original_name"]
                        )
                        match_type = "fuzzy"

                    matches.append(
                        EntityMatch(
                            entity1_id=entity1["id"],
                            entity2_id=entity2["id"],
                            entity1_name=entity1["original_name"],
                            entity2_name=entity2["original_name"],
                            similarity_score=similarity,
                            match_type=match_type,
                            entity1_chunk=entity1["chunk_index"],
                            entity2_chunk=entity2["chunk_index"],
                        )
                    )

        print(f"🎯 Scored {len(matches)} potential entity matches")
        return matches

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Simple name similarity calculation (can be enhanced with ML)"""
        # Jaccard similarity for now
        set1 = set(name1.lower().split())
        set2 = set(name2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _merge_entities(self, matches: list[EntityMatch]) -> dict[str, any]:
        """Merge high-confidence entity matches"""
        high_confidence_matches = [
            m for m in matches if m.similarity_score >= self.similarity_threshold
        ]

        print(f"🔗 Merging {len(high_confidence_matches)} high-confidence matches")

        merged_count = 0

        with self.driver.session() as session:
            for match in high_confidence_matches:
                try:
                    # Merge entities by redirecting all relationships from entity2 to entity1
                    session.run(
                        """
                        MATCH (e1) WHERE elementId(e1) = $entity1_id
                        MATCH (e2) WHERE elementId(e2) = $entity2_id

                        // Copy all incoming relationships from e2 to e1
                        MATCH (n)-[r]->(e2)
                        WHERE type(r) <> 'FROM_CHUNK'  // Don't duplicate chunk connections
                        WITH n, e1, e2, r, type(r) as rel_type, properties(r) as rel_props
                        CALL apoc.create.relationship(n, rel_type, rel_props, e1) YIELD rel as new_rel

                        // Copy all outgoing relationships from e2 to e1
                        MATCH (e2)-[r]->(n)
                        WHERE type(r) <> 'FROM_CHUNK'  // Don't duplicate chunk connections
                        WITH e1, e2, n, r, type(r) as rel_type, properties(r) as rel_props
                        CALL apoc.create.relationship(e1, rel_type, rel_props, n) YIELD rel as new_rel2

                        // Create SAME_AS relationship
                        MERGE (e1)-[:SAME_AS]-(e2)

                        // Mark e2 as merged
                        SET e2.merged_into = elementId(e1)
                        SET e2.resolution_status = 'merged'

                        RETURN count(*) as merged
                    """,
                        entity1_id=match.entity1_id,
                        entity2_id=match.entity2_id,
                    )

                    merged_count += 1
                    print(
                        f"   ✅ Merged '{match.entity2_name}' (Chunk {match.entity2_chunk}) → '{match.entity1_name}' (Chunk {match.entity1_chunk})"
                    )

                except Exception as e:
                    print(
                        f"   ❌ Failed to merge {match.entity1_name} ↔ {match.entity2_name}: {e}"
                    )

        return {"merged": merged_count}

    def _create_cross_chunk_links(self, matches: list[EntityMatch]) -> dict[str, any]:
        """Create SAME_AS links for lower-confidence matches"""
        medium_confidence_matches = [
            m for m in matches if 0.7 <= m.similarity_score < self.similarity_threshold
        ]

        print(
            f"🔗 Creating cross-chunk links for {len(medium_confidence_matches)} medium-confidence matches"
        )

        links_created = 0

        with self.driver.session() as session:
            for match in medium_confidence_matches:
                try:
                    session.run(
                        """
                        MATCH (e1) WHERE elementId(e1) = $entity1_id
                        MATCH (e2) WHERE elementId(e2) = $entity2_id
                        MERGE (e1)-[r:SIMILAR_TO]-(e2)
                        SET r.similarity_score = $similarity
                        SET r.match_type = $match_type
                    """,
                        entity1_id=match.entity1_id,
                        entity2_id=match.entity2_id,
                        similarity=match.similarity_score,
                        match_type=match.match_type,
                    )

                    links_created += 1
                    print(
                        f"   🔗 Linked '{match.entity1_name}' ↔ '{match.entity2_name}' (similarity: {match.similarity_score:.2f})"
                    )

                except Exception as e:
                    print(
                        f"   ❌ Failed to link {match.entity1_name} ↔ {match.entity2_name}: {e}"
                    )

        return {"links_created": links_created}


async def main():
    """Test the entity resolver"""
    driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", ""))

    resolver = EntityResolver(driver)
    stats = resolver.resolve_entities()

    print("\n🎉 Entity Resolution Complete!")
    print("📊 Statistics:")
    print(f"   - Duplicates found: {stats['duplicates_found']}")
    print(f"   - Matches scored: {stats['matches_scored']}")
    print(f"   - Entities merged: {stats['entities_merged']}")
    print(f"   - Links created: {stats['links_created']}")

    driver.close()


if __name__ == "__main__":
    asyncio.run(main())
