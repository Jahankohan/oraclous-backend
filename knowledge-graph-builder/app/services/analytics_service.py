"""
Graph Analytics Service

This service provides advanced graph analytics capabilities including:
- Community detection using Neo4j GDS Louvain algorithm
- Centrality analysis (PageRank and degree centrality)
- Neighborhood analysis for entity relationships
- Pathway discovery between entities
- Temporal context analysis
- Graph statistics and metrics

All methods are multi-tenant safe with proper graph_id filtering.
"""

import re
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime
import hashlib
import asyncio

# Only alphanumeric + underscore are safe to interpolate as Neo4j labels into GDS subquery strings.
# GDS executes subquery strings internally — Cypher parameters cannot be used inside them.
_SAFE_LABEL_RE = re.compile(r'^[A-Za-z0-9_]+$')

from sqlalchemy import text, create_engine
from sqlalchemy.pool import NullPool

from app.core.neo4j_client import neo4j_client
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class GraphAnalyticsService:
    """
    Dedicated service for graph analytics and algorithm execution.
    Extracted from ChatService for better separation of concerns.
    """
    
    def __init__(self):
        self.cached_statistics = {}
        
    # ==================== COMMUNITY DETECTION ====================
    
    async def get_community_context(
        self,
        entities: List[Dict[str, Any]],
        graph_id: UUID,
    ) -> Dict[str, Any]:
        """
        Return community context from persisted __Community__ nodes (post-Leiden).

        Falls back to simple shared-neighbor detection when no active communities exist.
        """
        if not entities:
            return {"communities": []}

        entity_ids = [e["id"] for e in entities if e.get("id")]

        try:
            query = """
            MATCH (entity:__Entity__)-[r:IN_COMMUNITY {graph_id: $graph_id, level: 1}]->(community:__Community__)
            WHERE entity.id IN $entity_ids
              AND community.graph_id = $graph_id
              AND community.status = 'active'
            WITH community, count(entity) AS member_hits
            WHERE member_hits >= 2
            RETURN community.id AS community_id,
                   community.summary AS summary,
                   community.level AS level,
                   community.entity_count AS entity_count,
                   community.status AS status,
                   member_hits
            ORDER BY member_hits DESC, community.entity_count ASC
            LIMIT 3
            """
            results = await neo4j_client.execute_query(query, {
                "entity_ids": entity_ids,
                "graph_id": str(graph_id),
            })

            if results:
                communities = [
                    {
                        "community_id": r["community_id"],
                        "summary": r["summary"],
                        "level": r["level"],
                        "entity_count": r["entity_count"],
                        "member_hits": r["member_hits"],
                        "type": "leiden_community",
                    }
                    for r in results
                ]
                return {"communities": communities}

        except Exception as exc:
            logger.warning(f"Persisted community lookup failed: {exc}")

        # Fallback when no active communities
        return await self.get_simple_community_context(entities, graph_id)

    # ==================== NEW: COMMUNITY MANAGEMENT METHODS ====================

    async def detect_communities_async(
        self,
        graph_id: UUID,
        levels: int = 3,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """
        Queue a Celery community detection job and return the task ID.

        Args:
            graph_id: Target graph
            levels: Number of hierarchy levels (1-5)
            force_rebuild: Run even if status == 'active'

        Returns:
            Dict with job_id, graph_id, status
        """
        from app.tasks.community_tasks import detect_communities_task

        level_indices = list(range(levels))
        resolutions = [0.5, 1.0, 2.0, 3.0, 4.0][:levels]

        result = detect_communities_task.apply_async(
            args=[str(graph_id)],
            kwargs={
                "levels": level_indices,
                "resolutions": resolutions,
                "force_rebuild": force_rebuild,
            },
            countdown=0,
        )

        return {
            "job_id": result.id,
            "graph_id": str(graph_id),
            "status": "queued",
        }

    async def get_community_status(self, graph_id: UUID) -> Dict[str, Any]:
        """Return current community detection status for a graph."""
        # Count communities per level
        level_query = """
        MATCH (c:__Community__ {graph_id: $graph_id})
        RETURN c.level AS level, count(c) AS cnt, c.status AS status
        ORDER BY level
        """
        level_results = await neo4j_client.execute_query(level_query, {"graph_id": str(graph_id)})

        communities_by_level: Dict[str, int] = {}
        detected_status = "not_detected"
        for r in level_results:
            communities_by_level[str(r["level"])] = r["cnt"]
            if r["status"] == "active":
                detected_status = "active"
            elif r["status"] == "rebuilding":
                detected_status = "rebuilding"
            elif detected_status == "not_detected" and r["status"] == "stale":
                detected_status = "stale"

        # Entity counts from Postgres
        entity_count_at_detection = 0
        last_detected_at = None
        current_entity_count = 0
        try:
            pg_engine = create_engine(
                settings.POSTGRES_URL.replace("+asyncpg", ""),
                poolclass=NullPool,
            )
            with pg_engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT communities_detected_at, entity_count_at_detection, "
                        "entity_delta_since_detection, communities_status "
                        "FROM knowledge_graphs WHERE id = :gid"
                    ),
                    {"gid": str(graph_id)},
                ).fetchone()
                if row:
                    last_detected_at = row[0]
                    entity_count_at_detection = row[1] or 0
                    delta = row[2] or 0
                    current_entity_count = entity_count_at_detection + delta
                    if row[3]:
                        detected_status = row[3]
            pg_engine.dispose()
        except Exception as exc:
            logger.warning(f"Postgres community status lookup failed: {exc}")

        staleness_pct = 0.0
        if entity_count_at_detection > 0:
            staleness_pct = (current_entity_count - entity_count_at_detection) / entity_count_at_detection

        return {
            "status": detected_status,
            "last_detected_at": last_detected_at.isoformat() if last_detected_at else None,
            "communities_by_level": communities_by_level,
            "entity_count_at_detection": entity_count_at_detection,
            "current_entity_count": current_entity_count,
            "staleness_pct": round(staleness_pct, 4),
        }

    async def get_communities_list(
        self,
        graph_id: UUID,
        level: Optional[int] = None,
        min_size: int = 2,
        limit: int = 50,
        offset: int = 0,
        include_summary: bool = True,
    ) -> Dict[str, Any]:
        """Return paginated list of communities for a graph."""
        where_clauses = ["c.graph_id = $graph_id", "c.entity_count >= $min_size"]
        params: Dict[str, Any] = {
            "graph_id": str(graph_id),
            "min_size": min_size,
            "limit": limit,
            "offset": offset,
        }
        if level is not None:
            where_clauses.append("c.level = $level")
            params["level"] = level

        where = " AND ".join(where_clauses)
        fields = "c.id AS community_id, c.level AS level, c.entity_count AS entity_count, c.weight AS weight, c.parent_id AS parent_id, c.status AS status"
        if include_summary:
            fields += ", c.summary AS summary"

        count_query = f"MATCH (c:__Community__) WHERE {where} RETURN count(c) AS total"
        list_query = f"""
        MATCH (c:__Community__) WHERE {where}
        RETURN {fields}
        ORDER BY c.level, c.entity_count DESC
        SKIP $offset LIMIT $limit
        """

        total_results = await neo4j_client.execute_query(count_query, params)
        total = total_results[0]["total"] if total_results else 0

        list_results = await neo4j_client.execute_query(list_query, params)

        communities = []
        for r in list_results:
            item = {
                "community_id": r["community_id"],
                "level": r["level"],
                "entity_count": r["entity_count"],
                "weight": r["weight"],
                "parent_id": r["parent_id"],
                "status": r["status"],
            }
            if include_summary:
                item["summary"] = r.get("summary")
            communities.append(item)

        # Detection status
        status_info = await self.get_community_status(graph_id)

        return {
            "communities": communities,
            "total": total,
            "detection_status": status_info["status"],
            "last_detected_at": status_info["last_detected_at"],
        }

    async def get_community_detail(
        self, graph_id: UUID, community_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return full community detail with members and parent/child links."""
        community_query = """
        MATCH (c:__Community__ {id: $community_id, graph_id: $graph_id})
        RETURN c.id AS community_id, c.level AS level, c.summary AS summary,
               c.entity_count AS entity_count, c.algorithm AS algorithm,
               c.parent_id AS parent_id, c.created_at AS created_at,
               c.last_updated AS last_updated, c.status AS status
        """
        result = await neo4j_client.execute_query(community_query, {
            "community_id": community_id,
            "graph_id": str(graph_id),
        })
        if not result:
            return None

        r = result[0]

        # Members
        members_query = """
        MATCH (e:__Entity__)-[:IN_COMMUNITY {graph_id: $graph_id, level: $level}]->(c:__Community__ {id: $cid, graph_id: $graph_id})
        RETURN e.id AS entity_id, e.name AS entity_name, labels(e) AS entity_labels
        LIMIT 100
        """
        members_result = await neo4j_client.execute_query(members_query, {
            "graph_id": str(graph_id),
            "level": r["level"],
            "cid": community_id,
        })
        members = [
            {
                "entity_id": m["entity_id"],
                "entity_name": m["entity_name"],
                "entity_type": next(
                    (lbl for lbl in (m["entity_labels"] or []) if lbl != "__Entity__"),
                    "Entity",
                ),
            }
            for m in members_result
        ]

        # Parent community
        parent_community = None
        if r.get("parent_id"):
            parent_result = await neo4j_client.execute_query(
                "MATCH (p:__Community__ {id: $pid, graph_id: $gid}) RETURN p.id AS community_id, p.summary AS summary",
                {"pid": r["parent_id"], "gid": str(graph_id)},
            )
            if parent_result:
                parent_community = {
                    "community_id": parent_result[0]["community_id"],
                    "summary": parent_result[0]["summary"],
                }

        # Child communities
        child_query = """
        MATCH (child:__Community__ {graph_id: $graph_id})-[:PARENT_COMMUNITY]->(parent:__Community__ {id: $cid, graph_id: $graph_id})
        RETURN child.id AS community_id, child.summary AS summary, child.entity_count AS entity_count
        LIMIT 20
        """
        child_results = await neo4j_client.execute_query(child_query, {
            "graph_id": str(graph_id),
            "cid": community_id,
        })
        child_communities = [
            {"community_id": c["community_id"], "summary": c["summary"], "entity_count": c["entity_count"]}
            for c in child_results
        ]

        return {
            "community_id": r["community_id"],
            "level": r["level"],
            "summary": r["summary"],
            "entity_count": r["entity_count"],
            "algorithm": r["algorithm"],
            "status": r["status"],
            "parent_community": parent_community,
            "child_communities": child_communities,
            "members": members,
            "created_at": r["created_at"],
            "last_updated": r["last_updated"],
        }

    async def get_simple_community_context(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """
        Fallback community detection based on shared neighbors with graph_id filtering.
        
        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys
            graph_id: UUID of the specific graph to analyze
            
        Returns:
            Dictionary containing simple community information
        """
        if not entities:
            return {"communities": []}
        
        entity_ids = [e["id"] for e in entities]
        
        query = """
        MATCH (entity)
        WHERE entity.id IN $entity_ids AND entity.graph_id = $graph_id
        
        MATCH (entity)-[r1]-(neighbor)-[r2]-(community_member)
        WHERE r1.graph_id = $graph_id AND r2.graph_id = $graph_id
        AND neighbor.graph_id = $graph_id AND community_member.graph_id = $graph_id
        AND community_member.id <> entity.id
        
        WITH entity, neighbor, collect(DISTINCT community_member) as members
        WHERE size(members) >= 2
        
        RETURN entity.id as entity_id,
            entity.name as entity_name,
            neighbor.name as hub_name,
            [m IN members | {id: m.id, name: m.name}][..5] as community_members
        LIMIT 10
        """
        
        results = await neo4j_client.execute_query(query, {
            "entity_ids": entity_ids,
            "graph_id": str(graph_id)
        })
        
        communities = []
        for result in results:
            communities.append({
                "entity": result["entity_name"],
                "hub": result["hub_name"],
                "members": result["community_members"],
                "type": "shared_neighbor_community"
            })
        
        return {"communities": communities}

    # ==================== COMMUNITY PERSISTENCE ====================
    
    async def create_community_nodes(self, graph_id: UUID) -> Dict[str, Any]:
        """
        Create persistent community nodes from detected communities.
        
        This method:
        1. Runs community detection on all entities
        2. Creates __Community__ nodes for each detected community
        3. Creates IN_COMMUNITY relationships between entities and communities
        4. Generates community summaries and embeddings
        
        Args:
            graph_id: UUID of the specific graph to analyze
            
        Returns:
            Dictionary containing creation results and statistics
        """
        try:
            logger.info(f"Creating persistent community nodes for graph {graph_id}")
            
            # First, check what entities exist and what labels they have
            check_entities_query = """
            MATCH (n)
            WHERE n.graph_id = $graph_id
            RETURN DISTINCT labels(n) as node_labels, count(n) as count
            ORDER BY count DESC
            """
            
            entity_check = await neo4j_client.execute_query(check_entities_query, {
                "graph_id": str(graph_id)
            })
            
            if not entity_check:
                return {"communities_created": 0, "relationships_created": 0, "message": "No entities found for this graph"}
            
            # Find the correct entity label (could be __Entity__, Entity, or something else)
            entity_label = None
            for result in entity_check:
                labels = result["node_labels"]
                if "__Entity__" in labels:
                    entity_label = "__Entity__"
                    break
                elif "Entity" in labels:
                    entity_label = "Entity"
                    break
                elif any(label not in ["__Community__", "Chunk"] for label in labels):
                    # Use the first non-community, non-chunk label
                    entity_label = next(label for label in labels if label not in ["__Community__", "Chunk"])
                    break
            
            if not entity_label:
                logger.warning(f"No suitable entity label found for graph {graph_id}")
                return await self._create_simple_communities(graph_id)

            # Validate label before interpolating into GDS subquery string.
            # GDS executes subquery strings internally so $params cannot be used inside them;
            # graph_id is a UUID (safe by type); entity_label must match [A-Za-z0-9_] only.
            if not _SAFE_LABEL_RE.match(entity_label):
                logger.error(f"Unsafe entity label '{entity_label}' rejected for graph {graph_id}")
                return await self._create_simple_communities(graph_id)

            logger.info(f"Using entity label: {entity_label}")
            
            # Generate unique graph projection name
            graph_name = f"temp_persist_{str(graph_id).replace('-', '_')}"
            
            # Step 1: Create graph projection with correct syntax and parameter handling
            projection_query = """
            CALL gds.graph.project.cypher(
                $graph_name,
                $node_query,
                $relationship_query
            )
            YIELD graphName
            RETURN graphName
            """
            
            # GDS subquery strings are executed internally by the GDS library and cannot accept
            # outer Cypher parameters — interpolation is unavoidable here.
            # Safety: entity_label validated against _SAFE_LABEL_RE above;
            # graph_id is a UUID type whose str() is always "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx".
            graph_id_str = str(graph_id)
            node_query = f"MATCH (n:{entity_label}) WHERE n.graph_id = '{graph_id_str}' RETURN id(n) AS id"
            relationship_query = (
                f"MATCH (a:{entity_label})-[r]->(b:{entity_label}) "
                f"WHERE a.graph_id = '{graph_id_str}' AND b.graph_id = '{graph_id_str}' "
                f"RETURN id(a) AS source, id(b) AS target"
            )
            
            try:
                await neo4j_client.execute_query(projection_query, {
                    "graph_name": graph_name,
                    "node_query": node_query,
                    "relationship_query": relationship_query
                })
            except Exception as projection_error:
                logger.warning(f"GDS projection failed: {projection_error}")
                return await self._create_simple_communities(graph_id)
            
            # Step 2: Run Louvain community detection
            community_detection_query = """
            CALL gds.louvain.stream($graph_name)
            YIELD nodeId, communityId
            
            // Get original nodes with their community assignments
            MATCH (node)
            WHERE id(node) = nodeId AND node.graph_id = $graph_id
            
            RETURN node.id as entity_id,
                node.name as entity_name,
                communityId,
                labels(node) as entity_labels
            """
            
            try:
                detection_results = await neo4j_client.execute_query(community_detection_query, {
                    "graph_name": graph_name,
                    "graph_id": str(graph_id)
                })
            except Exception as detection_error:
                logger.warning(f"Community detection failed: {detection_error}")
                # Clean up and fallback
                try:
                    cleanup_query = "CALL gds.graph.drop($graph_name, false) YIELD graphName"
                    await neo4j_client.execute_query(cleanup_query, {"graph_name": graph_name})
                except:
                    pass
                return await self._create_simple_communities(graph_id)
            
            # Step 3: Clean up the temporary graph
            try:
                cleanup_query = "CALL gds.graph.drop($graph_name, false) YIELD graphName"
                await neo4j_client.execute_query(cleanup_query, {"graph_name": graph_name})
            except Exception as cleanup_error:
                logger.warning(f"Graph cleanup failed: {cleanup_error}")
            
            if not detection_results:
                return {"communities_created": 0, "relationships_created": 0, "message": "No entities found for community detection"}
            
            # Step 4: Group entities by community
            communities_map = {}
            for result in detection_results:
                community_id = result["communityId"]
                if community_id not in communities_map:
                    communities_map[community_id] = []
                
                communities_map[community_id].append({
                    "entity_id": result["entity_id"],
                    "entity_name": result["entity_name"],
                    "entity_labels": result["entity_labels"]
                })
            
            # Step 5: Create community nodes and relationships
            communities_created = 0
            relationships_created = 0
            
            for community_id, members in communities_map.items():
                if len(members) < 2:  # Skip single-entity communities
                    continue
                
                # Generate unique community ID
                community_uuid = self._generate_community_id(graph_id, community_id, members)
                
                # Generate community summary
                community_summary = self._generate_community_summary(members)
                
                # Create community node
                create_community_query = """
                MERGE (community:__Community__ {
                    id: $community_id,
                    graph_id: $graph_id
                })
                SET community.summary = $summary,
                    community.entity_count = $entity_count,
                    community.detection_algorithm = $algorithm,
                    community.weight = $weight,
                    community.creation_date = datetime(),
                    community.last_updated = datetime()
                RETURN community
                """
                
                await neo4j_client.execute_query(create_community_query, {
                    "community_id": community_uuid,
                    "graph_id": str(graph_id),
                    "summary": community_summary,
                    "entity_count": len(members),
                    "algorithm": "louvain",
                    "weight": len(members) / len(detection_results)  # Relative size as weight
                })
                
                communities_created += 1
                
                # Create IN_COMMUNITY relationships
                for member in members:
                    # Use flexible entity matching since we might not know the exact label
                    relationship_query = """
                    MATCH (entity)
                    WHERE entity.id = $entity_id AND entity.graph_id = $graph_id
                    MATCH (community:__Community__ {id: $community_id, graph_id: $graph_id})
                    MERGE (entity)-[:IN_COMMUNITY]->(community)
                    """
                    
                    await neo4j_client.execute_query(relationship_query, {
                        "entity_id": member["entity_id"],
                        "community_id": community_uuid,
                        "graph_id": str(graph_id)
                    })
                    
                    relationships_created += 1
            
            logger.info(f"Created {communities_created} communities and {relationships_created} relationships for graph {graph_id}")
            
            return {
                "communities_created": communities_created,
                "relationships_created": relationships_created,
                "total_entities_processed": len(detection_results),
                "graph_id": str(graph_id),
                "algorithm_used": "louvain",
                "entity_label_used": entity_label
            }
            
        except Exception as e:
            logger.error(f"Failed to create community nodes: {e}")
            # Fallback: create communities using simple method
            return await self._create_simple_communities(graph_id)

    async def _create_simple_communities(self, graph_id: UUID) -> Dict[str, Any]:
        """
        Fallback method to create communities using simple clustering based on shared neighbors.
        
        Args:
            graph_id: UUID of the specific graph to analyze
            
        Returns:
            Dictionary containing creation results
        """
        try:
            logger.info(f"Creating simple communities for graph {graph_id}")
            
            # Find entities with shared neighbors using more flexible matching
            shared_neighbors_query = """
            MATCH (a)-[r1]-(common)-[r2]-(b)
            WHERE a.graph_id = $graph_id 
            AND b.graph_id = $graph_id 
            AND common.graph_id = $graph_id
            AND a.id < b.id  // Avoid duplicates
            AND a.id IS NOT NULL
            AND b.id IS NOT NULL
            
            WITH a, b, count(DISTINCT common) as shared_count
            WHERE shared_count >= 2  // At least 2 shared neighbors
            
            RETURN a.id as entity_a_id, 
                coalesce(a.name, a.id) as entity_a_name,
                b.id as entity_b_id, 
                coalesce(b.name, b.id) as entity_b_name,
                shared_count
            ORDER BY shared_count DESC
            LIMIT 50
            """
            
            results = await neo4j_client.execute_query(shared_neighbors_query, {
                "graph_id": str(graph_id)
            })
            
            if not results:
                return {"communities_created": 0, "relationships_created": 0, "message": "No shared neighbor communities found"}
            
            # Group entities into simple communities
            communities_created = 0
            relationships_created = 0
            
            for i, result in enumerate(results[:10]):  # Limit to 10 communities
                community_uuid = f"simple_community_{graph_id}_{i}"
                
                # Create community node
                create_community_query = """
                MERGE (community:__Community__ {
                    id: $community_id,
                    graph_id: $graph_id
                })
                SET community.summary = $summary,
                    community.entity_count = 2,
                    community.detection_algorithm = 'shared_neighbors',
                    community.weight = $weight,
                    community.creation_date = datetime(),
                    community.last_updated = datetime()
                """
                
                summary = f"Community of {result['entity_a_name']} and {result['entity_b_name']} (shared {result['shared_count']} connections)"
                
                await neo4j_client.execute_query(create_community_query, {
                    "community_id": community_uuid,
                    "graph_id": str(graph_id),
                    "summary": summary,
                    "weight": result["shared_count"] / 10.0  # Normalize weight
                })
                
                communities_created += 1
                
                # Create relationships for both entities using flexible matching
                for entity_id in [result["entity_a_id"], result["entity_b_id"]]:
                    relationship_query = """
                    MATCH (entity)
                    WHERE entity.id = $entity_id AND entity.graph_id = $graph_id
                    MATCH (community:__Community__ {id: $community_id, graph_id: $graph_id})
                    MERGE (entity)-[:IN_COMMUNITY]->(community)
                    """
                    
                    await neo4j_client.execute_query(relationship_query, {
                        "entity_id": entity_id,
                        "community_id": community_uuid,
                        "graph_id": str(graph_id)
                    })
                    
                    relationships_created += 1
            
            logger.info(f"Created {communities_created} simple communities and {relationships_created} relationships for graph {graph_id}")
            
            return {
                "communities_created": communities_created,
                "relationships_created": relationships_created,
                "total_entities_processed": len(results) * 2,
                "graph_id": str(graph_id),
                "algorithm_used": "shared_neighbors"
            }
            
        except Exception as e:
            logger.error(f"Failed to create simple communities: {e}")
            return {
                "communities_created": 0,
                "relationships_created": 0,
                "error": str(e),
                "graph_id": str(graph_id)
            }
    
    def _generate_community_id(self, graph_id: UUID, community_id: int, members: List[Dict]) -> str:
        """
        Generate a unique, deterministic community ID based on members.
        
        Args:
            graph_id: UUID of the graph
            community_id: Original community ID from algorithm
            members: List of community members
            
        Returns:
            Unique community ID string
        """
        # Sort members by ID for consistency
        sorted_ids = sorted([member["entity_id"] for member in members])
        
        # Create hash from graph_id + community_id + member IDs
        content = f"{graph_id}_{community_id}_{'_'.join(sorted_ids)}"
        community_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        
        return f"community_{graph_id}_{community_hash}"

    def _generate_community_summary(self, members: List[Dict]) -> str:
        """
        Generate a human-readable summary for a community.
        
        Args:
            members: List of community members with entity info
            
        Returns:
            Community summary string
        """
        entity_names = [member["entity_name"] for member in members]
        
        if len(entity_names) <= 3:
            names_str = ", ".join(entity_names)
        else:
            names_str = f"{', '.join(entity_names[:3])} and {len(entity_names) - 3} others"
        
        # Determine primary entity types
        all_labels = []
        for member in members:
            all_labels.extend(member.get("entity_labels", []))
        
        # Count label frequency (excluding __Entity__)
        label_counts = {}
        for label in all_labels:
            if label != "__Entity__":
                label_counts[label] = label_counts.get(label, 0) + 1
        
        if label_counts:
            primary_type = max(label_counts, key=label_counts.get)
            summary = f"Community of {len(members)} entities primarily about {primary_type.lower()}: {names_str}"
        else:
            summary = f"Community of {len(members)} related entities: {names_str}"
        
        return summary

    async def get_community_search_context(
        self, 
        query_text: str,
        graph_id: UUID,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search communities by text similarity and return context for RAG.
        
        Args:
            query_text: Text query to search communities
            graph_id: UUID of the specific graph
            top_k: Number of top communities to return
            
        Returns:
            Dictionary containing community search results
        """
        try:
            # For now, search by text matching in community summaries
            # TODO: Implement embedding-based search when embeddings are ready
            search_query = """
            MATCH (community:__Community__)
            WHERE community.graph_id = $graph_id
            AND (
                community.summary CONTAINS $query_text 
                OR any(word IN split(toLower($query_text), ' ') 
                       WHERE community.summary CONTAINS word)
            )
            
            // Get community members
            MATCH (entity:__Entity__)-[:IN_COMMUNITY]->(community)
            WHERE entity.graph_id = $graph_id
            
            RETURN community.id as community_id,
                   community.summary as summary,
                   community.entity_count as entity_count,
                   community.weight as weight,
                   community.detection_algorithm as algorithm,
                   collect({id: entity.id, name: entity.name}) as members
            ORDER BY community.weight DESC, community.entity_count DESC
            LIMIT $top_k
            """
            
            results = await neo4j_client.execute_query(search_query, {
                "query_text": query_text.lower(),
                "graph_id": str(graph_id),
                "top_k": top_k
            })
            
            communities = []
            for result in results:
                communities.append({
                    "community_id": result["community_id"],
                    "summary": result["summary"],
                    "entity_count": result["entity_count"],
                    "weight": result["weight"],
                    "algorithm": result["algorithm"],
                    "members": result["members"]
                })
            
            return {
                "communities": communities,
                "search_type": "text_matching",
                "query": query_text,
                "graph_id": str(graph_id)
            }
            
        except Exception as e:
            logger.error(f"Community search failed: {e}")
            return {"communities": [], "error": str(e), "graph_id": str(graph_id)}

    # ==================== CENTRALITY ANALYSIS ====================
    
    async def get_influential_context(
        self, 
        query: str, 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """
        Get highly connected nodes using Neo4j GDS PageRank with graph_id filtering.
        
        Args:
            query: User query for context (used for logging/debugging)
            graph_id: UUID of the specific graph to analyze
            
        Returns:
            Dictionary containing influential nodes information
        """
        try:
            # Try advanced PageRank centrality with Neo4j GDS
            pagerank_query = """
            CALL {
                // Create temporary graph projection
                CALL gds.graph.project.cypher(
                    'temp-pagerank-' + $graph_id,
                    'MATCH (n) WHERE n.graph_id = "' + $graph_id + '" RETURN id(n) AS id, n.name AS name',
                    'MATCH (a)-[r]-(b) WHERE a.graph_id = "' + $graph_id + '" AND b.graph_id = "' + $graph_id + '" AND r.graph_id = "' + $graph_id + '" RETURN id(a) AS source, id(b) AS target'
                )
                YIELD graphName
                
                // Run PageRank algorithm
                CALL gds.pageRank.stream('temp-pagerank-' + $graph_id)
                YIELD nodeId, score
                
                // Get original nodes with scores
                MATCH (node)
                WHERE id(node) = nodeId AND node.graph_id = $graph_id
                
                WITH node, score
                ORDER BY score DESC
                LIMIT 10
                
                RETURN node.id as entity_id,
                    node.name as entity_name,
                    score as pagerank_score,
                    labels(node) as labels
            }
            
            // Clean up temporary graph
            CALL gds.graph.drop('temp-pagerank-' + $graph_id, false)
            YIELD graphName as droppedGraph
            
            RETURN entity_id, entity_name, pagerank_score, labels
            """
            
            results = await neo4j_client.execute_query(pagerank_query, {
                "graph_id": str(graph_id)
            })
            
            influential = []
            for result in results:
                influential.append({
                    "id": result["entity_id"],
                    "name": result["entity_name"],
                    "pagerank_score": result["pagerank_score"],
                    "labels": result["labels"],
                    "influence_type": "pagerank_centrality"
                })
            
            return {"influential": influential}
            
        except Exception as e:
            logger.warning(f"Advanced PageRank failed: {e}")
            # Fallback to simple degree centrality
            return await self.get_simple_influential_context(query, graph_id)

    async def get_simple_influential_context(
        self, 
        query: str, 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """
        Fallback influential nodes based on degree centrality with graph_id filtering.
        
        Args:
            query: User query for context (used for logging/debugging)
            graph_id: UUID of the specific graph to analyze
            
        Returns:
            Dictionary containing influential nodes based on degree centrality
        """
        # Find high-degree nodes in this specific graph
        cypher_query = """
        MATCH (n)
        WHERE n.graph_id = $graph_id
        
        OPTIONAL MATCH (n)-[r]-()
        WHERE r.graph_id = $graph_id
        
        WITH n, count(r) as degree
        WHERE degree > 3  // Nodes with more than 3 connections
        ORDER BY degree DESC
        LIMIT 10
        
        RETURN n.name as name, 
            n.id as id, 
            degree,
            labels(n) as labels,
            n{.*} as properties
        """
        
        results = await neo4j_client.execute_query(cypher_query, {
            "graph_id": str(graph_id)
        })
        
        influential = []
        for result in results:
            influential.append({
                "name": result["name"],
                "id": result["id"],
                "degree": result["degree"],
                "labels": result["labels"],
                "properties": result["properties"],
                "influence_type": "degree_centrality"
            })
        
        return {"influential": influential}

    # ==================== NEIGHBORHOOD ANALYSIS ====================
    
    async def get_neighborhood_context(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """
        Get 1-2 hop neighbor context with proper graph_id filtering.
        
        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys
            graph_id: UUID of the specific graph to analyze
            
        Returns:
            Dictionary containing neighborhood information
        """
        if not entities:
            return {"neighborhoods": [], "relationships": []}
        
        entity_ids = [e["id"] for e in entities]
        
        query = """
        MATCH (start)
        WHERE start.id IN $entity_ids AND start.graph_id = $graph_id
        
        MATCH (start)-[r1]-(neighbor1)
        WHERE r1.graph_id = $graph_id AND neighbor1.graph_id = $graph_id
        
        OPTIONAL MATCH (neighbor1)-[r2]-(neighbor2)
        WHERE r2.graph_id = $graph_id AND neighbor2.graph_id = $graph_id
        AND neighbor2.id <> start.id
        
        WITH start, r1, neighbor1, 
            collect(DISTINCT {
                node: neighbor2{.id, .name, labels: labels(neighbor2)},
                relationship: type(r2)
            })[..3] as second_hop
        
        RETURN start.id as center_id,
            start.name as center_name,
            {
                relationship: type(r1),
                neighbor: neighbor1{.id, .name, labels: labels(neighbor1)},
                second_hop: second_hop
            } as neighborhood_info
        LIMIT 30
        """
        
        results = await neo4j_client.execute_query(query, {
            "entity_ids": entity_ids,
            "graph_id": str(graph_id)
        })
        
        neighborhoods = {}
        relationships = []
        
        for result in results:
            center_id = result["center_id"]
            if center_id not in neighborhoods:
                neighborhoods[center_id] = {
                    "center": {"id": center_id, "name": result["center_name"]}, 
                    "neighbors": []
                }
            
            neighborhoods[center_id]["neighbors"].append(result["neighborhood_info"])
            relationships.append({
                "source": center_id,
                "target": result["neighborhood_info"]["neighbor"]["id"],
                "type": result["neighborhood_info"]["relationship"]
            })
        
        return {
            "neighborhoods": list(neighborhoods.values()),
            "relationships": relationships
        }

    # ==================== PATHWAY ANALYSIS ====================
    
    async def get_pathway_context(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Get pathways between entities with proper graph_id filtering.
        
        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys
            graph_id: UUID of the specific graph to analyze
            max_depth: Maximum depth for path discovery (default: 3)
            
        Returns:
            Dictionary containing pathway information
        """
        if len(entities) < 2:
            return {"pathways": []}
        
        pathways = []
        
        # Get pathways between all pairs
        for i, start_entity in enumerate(entities):
            for j, end_entity in enumerate(entities[i+1:], i+1):
                paths = await self.find_paths_between_entities(
                    start_entity["id"], 
                    end_entity["id"], 
                    graph_id, 
                    max_depth
                )
                pathways.extend(paths)
        
        # Remove duplicates and sort
        unique_pathways = []
        seen_paths = set()
        
        for pathway in pathways:
            path_signature = tuple(pathway.get("nodes", []))
            if path_signature not in seen_paths:
                seen_paths.add(path_signature)
                unique_pathways.append(pathway)
        
        unique_pathways.sort(key=lambda x: x.get("length", 999))
        
        return {"pathways": unique_pathways[:10]}
    
    async def find_shortest_paths(
        self, 
        start_id: str, 
        end_id: str, 
        graph_id: UUID
    ) -> List[Dict[str, Any]]:
        """
        Find shortest paths between entities with proper graph_id filtering.
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            graph_id: UUID of the specific graph to analyze
            
        Returns:
            List of shortest path dictionaries
        """
        query = """
        MATCH (start {id: $start_id, graph_id: $graph_id})
        MATCH (end {id: $end_id, graph_id: $graph_id})
        
        MATCH path = shortestPath((start)-[*1..3]-(end))
        WHERE ALL(r IN relationships(path) WHERE r.graph_id = $graph_id)
        AND ALL(n IN nodes(path) WHERE n.graph_id = $graph_id)
        
        RETURN [n.name FOR n IN nodes(path)] as node_names,
               [type(r) FOR r IN relationships(path)] as relationship_types,
               length(path) as path_length
        ORDER BY path_length
        LIMIT 3
        """
        
        results = await neo4j_client.execute_query(query, {
            "start_id": start_id,
            "end_id": end_id,
            "graph_id": str(graph_id)
        })
        
        return [
            {
                "start": start_id,
                "end": end_id,
                "nodes": result["node_names"],
                "relationships": result["relationship_types"],
                "length": result["path_length"],
                "type": "shortest_path"
            }
            for result in results
        ]
    
    async def find_paths_between_entities(
        self, 
        start_id: str, 
        end_id: str, 
        graph_id: UUID,
        max_depth: int
    ) -> List[Dict[str, Any]]:
        """
        Advanced pathfinding with proper graph_id filtering.
        
        Args:
            start_id: Starting entity ID
            end_id: Target entity ID
            graph_id: UUID of the specific graph to analyze
            max_depth: Maximum depth for path discovery
            
        Returns:
            List of path dictionaries
        """
        query = """
        MATCH (start {id: $start_id, graph_id: $graph_id})
        MATCH (end {id: $end_id, graph_id: $graph_id})
        
        CALL apoc.path.allSimplePaths(start, end, '', $max_depth) YIELD path
        WHERE ALL(n IN nodes(path) WHERE n.graph_id = $graph_id)
        AND ALL(r IN relationships(path) WHERE r.graph_id = $graph_id)
        
        WITH path, length(path) as path_length
        ORDER BY path_length
        LIMIT 5
        
        RETURN [n.name FOR n IN nodes(path)] as path_nodes,
               [{type: type(r), properties: properties(r)} FOR r IN relationships(path)] as path_relationships,
               path_length
        """
        
        results = await neo4j_client.execute_query(query, {
            "start_id": start_id,
            "end_id": end_id,
            "graph_id": str(graph_id),
            "max_depth": max_depth
        })
        
        return [
            {
                "start": start_id,
                "end": end_id,
                "nodes": result["path_nodes"],
                "relationships": result["path_relationships"],
                "length": result["path_length"],
                "type": "advanced_path"
            }
            for result in results
        ]

    # ==================== TEMPORAL ANALYSIS ====================
    
    async def get_temporal_context(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID
    ) -> Dict[str, Any]:
        """
        Get temporal/time-based context with graph_id filtering.
        
        Args:
            entities: List of entity dictionaries with 'id' and 'name' keys
            graph_id: UUID of the specific graph to analyze
            
        Returns:
            Dictionary containing temporal context information
        """
        if not entities:
            return {"temporal": []}
        
        entity_ids = [e["id"] for e in entities]
        
        # Look for date-related properties in relationships and nodes
        query = """
        MATCH (n)-[r]-(m)
        WHERE n.id IN $entity_ids 
        AND n.graph_id = $graph_id 
        AND r.graph_id = $graph_id 
        AND m.graph_id = $graph_id
        AND (
            r.start_date IS NOT NULL OR 
            r.end_date IS NOT NULL OR 
            r.date IS NOT NULL OR 
            r.year IS NOT NULL OR
            r.created_at IS NOT NULL OR
            r.timestamp IS NOT NULL OR
            n.birth_date IS NOT NULL OR
            n.founded IS NOT NULL OR
            m.birth_date IS NOT NULL OR
            m.founded IS NOT NULL
        )
        
        WITH n, r, m,
            coalesce(
                r.start_date, 
                r.end_date, 
                r.date, 
                r.year, 
                r.created_at,
                r.timestamp,
                n.birth_date, 
                n.founded,
                m.birth_date,
                m.founded
            ) as date_info
        
        WHERE date_info IS NOT NULL
        
        RETURN n.name as entity_name,
            type(r) as relationship_type,
            m.name as connected_entity,
            date_info,
            r{.*} as relationship_properties
        ORDER BY date_info DESC
        LIMIT 20
        """
        
        results = await neo4j_client.execute_query(query, {
            "entity_ids": entity_ids,
            "graph_id": str(graph_id)
        })
        
        temporal = []
        for result in results:
            temporal.append({
                "entity": result["entity_name"],
                "relationship": result["relationship_type"],
                "connected_to": result["connected_entity"],
                "date": str(result["date_info"]),  # Convert to string for JSON serialization
                "relationship_properties": result["relationship_properties"],
                "context_type": "temporal"
            })
        
        return {"temporal": temporal}

    # ==================== GRAPH STATISTICS ====================
    
    async def get_graph_statistics(self, graph_id: UUID) -> Dict[str, Any]:
        """
        Get comprehensive graph statistics for the specified graph.
        
        Args:
            graph_id: UUID of the specific graph to analyze
            
        Returns:
            Dictionary containing graph statistics
        """
        try:
            # Basic graph statistics
            stats_query = """
            MATCH (n)
            WHERE n.graph_id = $graph_id
            WITH count(n) as node_count
            
            MATCH ()-[r]-()
            WHERE r.graph_id = $graph_id
            WITH node_count, count(r) as rel_count
            
            MATCH (n)
            WHERE n.graph_id = $graph_id
            WITH node_count, rel_count, labels(n) as node_labels
            UNWIND node_labels as label
            WITH node_count, rel_count, label
            WHERE label <> 'Entity'  // Skip generic labels
            
            WITH node_count, rel_count, collect(DISTINCT label) as entity_types
            
            MATCH ()-[r]-()
            WHERE r.graph_id = $graph_id
            WITH node_count, rel_count, entity_types, type(r) as rel_type
            
            RETURN node_count,
                rel_count,
                entity_types,
                collect(DISTINCT rel_type) as relationship_types
            """
            
            result = await neo4j_client.execute_query(stats_query, {
                "graph_id": str(graph_id)
            })
            
            if result:
                stats = result[0]
                node_count = stats["node_count"]
                rel_count = stats["rel_count"]
                
                return {
                    "node_count": node_count,
                    "relationship_count": rel_count,
                    "density": rel_count / (node_count * (node_count - 1)) if node_count > 1 else 0,
                    "entity_types": stats["entity_types"][:10],  # Top 10 types
                    "relationship_types": stats["relationship_types"][:10],  # Top 10 types
                    "avg_degree": (2 * rel_count / node_count) if node_count > 0 else 0,
                    "computed_at": datetime.now().isoformat()
                }
            else:
                return {"node_count": 0, "relationship_count": 0}
                
        except Exception as e:
            logger.warning(f"Failed to get graph statistics: {e}")
            return {"error": str(e)}

    async def precompute_and_cache_statistics(self, graph_id: UUID) -> None:
        """
        Precompute and cache graph statistics for better performance.
        
        Args:
            graph_id: UUID of the specific graph to analyze
        """
        try:
            # Get comprehensive statistics
            stats = await self.get_graph_statistics(graph_id)
            
            # Cache with timestamp
            self.cached_statistics[str(graph_id)] = {
                **stats,
                "cached_at": datetime.now()
            }
            
            logger.info(f"Precomputed statistics for graph {graph_id}: "
                    f"{stats.get('node_count', 0)} nodes, "
                    f"{stats.get('relationship_count', 0)} relationships")
            
        except Exception as e:
            logger.warning(f"Failed to precompute graph statistics: {e}")
            self.cached_statistics[str(graph_id)] = {
                "error": str(e),
                "node_count": 0,
                "relationship_count": 0,
                "cached_at": datetime.now()
            }

    def get_cached_statistics(self, graph_id: UUID) -> Optional[Dict[str, Any]]:
        """
        Get cached statistics for a graph if available.
        
        Args:
            graph_id: UUID of the specific graph
            
        Returns:
            Cached statistics dictionary or None if not cached
        """
        return self.cached_statistics.get(str(graph_id))

    # ==================== COMPREHENSIVE ANALYSIS ====================
    
    async def comprehensive_graph_analysis(
        self, 
        entities: List[Dict[str, Any]], 
        graph_id: UUID,
        include_communities: bool = True,
        include_influence: bool = True,
        include_pathways: bool = True,
        include_temporal: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive graph analysis combining multiple analytics methods.
        
        Args:
            entities: List of entity dictionaries to analyze
            graph_id: UUID of the specific graph to analyze
            include_communities: Whether to include community detection
            include_influence: Whether to include influence analysis  
            include_pathways: Whether to include pathway analysis
            include_temporal: Whether to include temporal analysis
            
        Returns:
            Dictionary containing comprehensive analysis results
        """
        analysis_results = {
            "graph_id": str(graph_id),
            "entities_analyzed": len(entities),
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        try:
            # Always include neighborhood analysis
            neighborhood_ctx = await self.get_neighborhood_context(entities, graph_id)
            analysis_results["neighborhoods"] = neighborhood_ctx
            
            if include_communities:
                community_ctx = await self.get_community_context(entities, graph_id)
                analysis_results["communities"] = community_ctx
            
            if include_influence:
                influence_ctx = await self.get_influential_context("", graph_id)
                analysis_results["influential"] = influence_ctx
            
            if include_pathways and len(entities) >= 2:
                pathway_ctx = await self.get_pathway_context(entities, graph_id)
                analysis_results["pathways"] = pathway_ctx
            
            if include_temporal:
                temporal_ctx = await self.get_temporal_context(entities, graph_id)
                analysis_results["temporal"] = temporal_ctx
            
            # Include graph statistics
            stats = await self.get_graph_statistics(graph_id)
            analysis_results["statistics"] = stats
            
            analysis_results["success"] = True
            
        except Exception as e:
            logger.error(f"Comprehensive analysis failed: {e}")
            analysis_results["error"] = str(e)
            analysis_results["success"] = False
        
        return analysis_results


# Create global instance
analytics_service = GraphAnalyticsService()
