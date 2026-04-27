"""
Neo4j Graph Node Service

This service manages Graph nodes in Neo4j, replacing PostgreSQL knowledge_graphs table.
Each Graph node represents a knowledge graph with metadata and tenant isolation.

Graph Node Schema:
(graph:Graph {
    graph_id: "uuid",
    name: "Graph Name",
    description: "Optional description",
    user_id: "user_uuid",
    created_at: datetime(),
    updated_at: datetime(),
    node_count: 0,
    relationship_count: 0,
    status: "active"
})
"""

from typing import Any

from neo4j import Driver
from neo4j.exceptions import Neo4jError


class GraphNodeService:
    """Service for managing Graph nodes in Neo4j"""

    def __init__(self, driver: Driver):
        """
        Initialize GraphNodeService with Neo4j driver

        Args:
            driver: Neo4j driver instance
        """
        self.driver = driver

    # Temporal indexes created once per graph at creation time.
    # Neo4j 5.x syntax: relationship property indexes use wildcard type (-[r]-).
    _TEMPORAL_INDEX_STATEMENTS = [
        # Composite index for point-in-time entity queries
        "CREATE INDEX entity_temporal_idx IF NOT EXISTS "
        "FOR (e:__Entity__) ON (e.graph_id, e.valid_from, e.valid_to)",
        # Composite index for transaction-time lookups
        "CREATE INDEX entity_transaction_time_idx IF NOT EXISTS "
        "FOR (e:__Entity__) ON (e.graph_id, e.transaction_time)",
        # Relationship temporal index (Neo4j 5.x wildcard syntax)
        "CREATE INDEX rel_temporal_idx IF NOT EXISTS "
        "FOR ()-[r]-() ON (r.graph_id, r.valid_from, r.valid_to)",
        # Standalone relationship indexes for traversal queries that filter by
        # valid_from / valid_to without r.graph_id in the WHERE clause
        # (e.g. multihop enrichment in chat_service). The composite rel_temporal_idx
        # requires graph_id as the leading key and is not used by the planner in
        # those traversal patterns.
        "CREATE INDEX rel_valid_from_idx IF NOT EXISTS "
        "FOR ()-[r]-() ON (r.valid_from)",
        "CREATE INDEX rel_valid_to_idx IF NOT EXISTS " "FOR ()-[r]-() ON (r.valid_to)",
        # __Contradiction__ label added at schema init (not ad hoc per CTO review)
        "CREATE INDEX contradiction_graph_idx IF NOT EXISTS "
        "FOR (c:__Contradiction__) ON (c.graph_id, c.detected_at)",
        # Federation indexes
        "CREATE INDEX graph_federatable_idx IF NOT EXISTS "
        "FOR (g:Graph) ON (g.federatable)",
        "CREATE INDEX graph_federation_group_idx IF NOT EXISTS "
        "FOR (g:Graph) ON (g.federation_group)",
        # Cross-graph entity deduplication — name+type lookup
        "CREATE INDEX entity_name_type_idx IF NOT EXISTS "
        "FOR (e:__Entity__) ON (e.name, e.type)",
    ]

    def create_graph(
        self, graph_id: str, name: str, user_id: str, description: str | None = None
    ) -> dict[str, Any]:
        """Create a new Graph node in Neo4j and ensure temporal indexes exist."""

        query = """
        CREATE (g:Graph {
            graph_id: $graph_id,
            name: $name,
            description: $description,
            user_id: $user_id,
            created_at: datetime(),
            updated_at: datetime(),
            node_count: 0,
            relationship_count: 0,
            status: 'active',
            federatable: false,
            federation_group: null
        })
        RETURN g
        """

        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")

            with self.driver.session() as session:
                result = session.run(
                    query,
                    {
                        "graph_id": graph_id,
                        "name": name,
                        "description": description or "",
                        "user_id": user_id,
                    },
                )

                record = result.single()
                if not record:
                    raise ValueError("Failed to create Graph node")

                graph_data = dict(record["g"])

            # Ensure temporal indexes exist (idempotent — IF NOT EXISTS)
            self._ensure_temporal_indexes()

            return {
                "graph_id": graph_data["graph_id"],
                "name": graph_data["name"],
                "description": graph_data["description"],
                "user_id": graph_data["user_id"],
                "created_at": graph_data["created_at"].isoformat(),
                "updated_at": graph_data["updated_at"].isoformat(),
                "node_count": graph_data["node_count"],
                "relationship_count": graph_data["relationship_count"],
                "status": graph_data["status"],
                "federatable": graph_data.get("federatable", False),
                "federation_group": graph_data.get("federation_group"),
            }

        except Neo4jError as e:
            raise Exception(f"Failed to create graph: {str(e)}") from None
        except Exception as e:
            raise Exception(f"Unexpected error creating graph: {str(e)}") from None

    def _ensure_temporal_indexes(self) -> None:
        """Create temporal indexes if they don't exist. Idempotent."""
        with self.driver.session() as session:
            for stmt in self._TEMPORAL_INDEX_STATEMENTS:
                try:
                    session.run(stmt)
                except Exception:
                    # Index creation errors are non-fatal; the graph node was already created
                    pass

    def get_graph(self, graph_id: str) -> dict[str, Any] | None:
        """
        Retrieve a Graph node by graph_id.

        Args:
            graph_id: Graph identifier

        Returns:
            Graph metadata dict or None if not found
        """
        query = """
        MATCH (g:Graph {graph_id: $graph_id})
        RETURN g {
            .graph_id,
            .name,
            .description,
            .user_id,
            .created_at,
            .updated_at,
            .node_count,
            .relationship_count,
            .status,
            .federatable,
            .federation_group
        } as graph
        """

        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")

            with self.driver.session() as session:
                result = session.run(query, {"graph_id": graph_id})
                record = result.single()

                if record:
                    data = dict(record["graph"])
                    data.setdefault("federatable", False)
                    data.setdefault("federation_group", None)
                    return data
                return None

        except Neo4jError as e:
            raise Exception(f"Failed to retrieve Graph node {graph_id}: {e}") from None

    def list_user_graphs(self, user_id: str) -> list[dict[str, Any]]:
        """
        List all Graph nodes for a specific user.

        Args:
            user_id: User identifier

        Returns:
            List of graph metadata dicts
        """
        query = """
        MATCH (g:Graph {user_id: $user_id})
        RETURN g {
            .graph_id,
            .name,
            .description,
            .user_id,
            .created_at,
            .updated_at,
            .node_count,
            .relationship_count,
            .status,
            .federatable,
            .federation_group
        } as graph
        ORDER BY g.created_at DESC
        """

        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")

            with self.driver.session() as session:
                result = session.run(query, {"user_id": user_id})

                graphs: list[dict[str, Any]] = []
                for record in result:
                    data = dict(record["graph"])
                    data.setdefault("federatable", False)
                    data.setdefault("federation_group", None)
                    graphs.append(data)

                return graphs

        except Neo4jError as e:
            raise Exception(f"Failed to list user graphs: {e}") from None

    def update_graph(
        self,
        graph_id: str,
        user_id: str,
        name: str | None = None,
        description: str | None = None,
        node_count: int | None = None,
        relationship_count: int | None = None,
        status: str | None = None,
        federatable: bool | None = None,
        federation_group: str | None = None,
    ) -> dict[str, Any] | None:
        """Update Graph node metadata."""
        params: dict[str, Any] = {"graph_id": graph_id, "user_id": user_id}

        set_parts = ["g.updated_at = datetime()"]
        if name is not None:
            set_parts.append("g.name = $name")
            params["name"] = name
        if description is not None:
            set_parts.append("g.description = $description")
            params["description"] = description
        if node_count is not None:
            set_parts.append("g.node_count = $node_count")
            params["node_count"] = node_count
        if relationship_count is not None:
            set_parts.append("g.relationship_count = $relationship_count")
            params["relationship_count"] = relationship_count
        if status is not None:
            set_parts.append("g.status = $status")
            params["status"] = status
        if federatable is not None:
            set_parts.append("g.federatable = $federatable")
            params["federatable"] = federatable
        if federation_group is not None:
            set_parts.append("g.federation_group = $federation_group")
            params["federation_group"] = federation_group

        set_clause = ", ".join(set_parts)

        if federatable is not None:
            # Sync shadow node atomically so federation_service reads the updated flag.
            # MERGE ensures the shadow node is created if it does not yet exist.
            # Only SET shadow.federatable — never overwrite owner_user_id, graph_name, etc.
            query = f"""
        MATCH (g:Graph {{graph_id: $graph_id, user_id: $user_id}})
        MERGE (shadow:Graph {{graph_id: $graph_id, namespace: "__system__"}})
        SET {set_clause}, shadow.federatable = $federatable
        RETURN g {{
            .graph_id,
            .name,
            .description,
            .user_id,
            .created_at,
            .updated_at,
            .node_count,
            .relationship_count,
            .status,
            .federatable,
            .federation_group
        }} as graph
        """
        else:
            query = f"""
        MATCH (g:Graph {{graph_id: $graph_id, user_id: $user_id}})
        SET {set_clause}
        RETURN g {{
            .graph_id,
            .name,
            .description,
            .user_id,
            .created_at,
            .updated_at,
            .node_count,
            .relationship_count,
            .status,
            .federatable,
            .federation_group
        }} as graph
        """

        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")

            with self.driver.session() as session:
                result = session.run(query, params)
                record = result.single()

                if record:
                    data = dict(record["graph"])
                    data.setdefault("federatable", False)
                    data.setdefault("federation_group", None)
                    return data
                return None

        except Neo4jError as e:
            raise Exception(f"Failed to update graph: {e}") from None

    def list_federatable_graphs(
        self, user_id: str, graph_ids: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return graphs owned by user_id that have federatable=true.

        If graph_ids is provided, only return graphs in that list (ownership + federatable check).
        Used by FederationService to validate and filter federation requests.
        """
        if graph_ids is not None:
            query = """
            MATCH (g:Graph {user_id: $user_id, federatable: true})
            WHERE g.graph_id IN $graph_ids
            RETURN g {
                .graph_id,
                .name,
                .user_id,
                .federatable,
                .federation_group
            } as graph
            """
            params: dict[str, Any] = {"user_id": user_id, "graph_ids": graph_ids}
        else:
            query = """
            MATCH (g:Graph {user_id: $user_id, federatable: true})
            RETURN g {
                .graph_id,
                .name,
                .user_id,
                .federatable,
                .federation_group
            } as graph
            ORDER BY g.name ASC
            """
            params = {"user_id": user_id}

        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")

            with self.driver.session() as session:
                result = session.run(query, params)
                return [dict(record["graph"]) for record in result]

        except Neo4jError as e:
            raise Exception(f"Failed to list federatable graphs: {e}") from None

    def delete_graph(self, graph_id: str, user_id: str) -> bool:
        """
        Delete a Graph node and all its relationships.

        Args:
            graph_id: Graph identifier
            user_id: User identifier (for tenant isolation)

        Returns:
            True if deleted, False if not found
        """
        query = """
        MATCH (g:Graph {graph_id: $graph_id, user_id: $user_id})
        DETACH DELETE g
        RETURN count(g) as deleted_count
        """

        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")

            with self.driver.session() as session:
                result = session.run(query, {"graph_id": graph_id, "user_id": user_id})

                record = result.single()
                if record:
                    return record["deleted_count"] > 0
                return False

        except Neo4jError as e:
            raise Exception(f"Failed to delete graph: {e}") from None

    def migrate_relationship_properties(self, graph_id: str) -> dict[str, Any]:
        """
        Run 3-phase migration to move contextual properties from entity nodes
        to their corresponding relationships, per the ORA-4 spec.

        Phase 1 — Transfer: copy banned props from nodes onto existing matching edges.
        Phase 2 — Orphan detection: find nodes with banned props but no edge to receive them.
        Phase 3 — Cleanup: remove banned props from all nodes in this graph.

        Returns:
            dict with counts: transferred, orphans_detected, cleaned
        """
        phase1_job_title = """
        MATCH (p:__Entity__ {graph_id: $graph_id})
        WHERE p.job_title IS NOT NULL
        MATCH (p)-[r:WORKS_FOR]->()
        SET r.position = COALESCE(r.position, p.job_title),
            r.migrated_from_node = true,
            r.migration_date = datetime()
        RETURN count(r) AS transferred
        """

        phase1_proficiency = """
        MATCH (s:__Entity__ {type: "Skill", graph_id: $graph_id})
        WHERE s.proficiency IS NOT NULL
        MATCH (p)-[r:HAS_SKILL]->(s)
        SET r.proficiency = COALESCE(r.proficiency, s.proficiency),
            r.migrated_from_node = true,
            r.migration_date = datetime()
        RETURN count(r) AS transferred
        """

        phase2_orphans = """
        MATCH (p:__Entity__ {graph_id: $graph_id})
        WHERE p.job_title IS NOT NULL AND NOT (p)-[:WORKS_FOR]->()
        RETURN p.entity_id AS entity_id,
               p.name AS name,
               p.type AS type,
               keys(p) AS props
        LIMIT 500
        """

        phase3_cleanup = """
        MATCH (p:__Entity__ {graph_id: $graph_id})
        WHERE p.job_title IS NOT NULL
           OR p.position IS NOT NULL
           OR p.role IS NOT NULL
           OR p.seniority IS NOT NULL
           OR p.proficiency IS NOT NULL
           OR p.ownership_pct IS NOT NULL
           OR p.allocation IS NOT NULL
           OR p.start_date_of_employment IS NOT NULL
           OR p.end_date_of_employment IS NOT NULL
        REMOVE p.job_title, p.position, p.role, p.seniority, p.proficiency,
               p.ownership_pct, p.allocation,
               p.start_date_of_employment, p.end_date_of_employment
        RETURN count(p) AS cleaned
        """

        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")

            with self.driver.session() as session:
                # Phase 1a: transfer job_title to WORKS_FOR.position
                r1a = session.run(phase1_job_title, {"graph_id": graph_id}).single()
                transferred_job_title = r1a["transferred"] if r1a else 0

                # Phase 1b: transfer proficiency to HAS_SKILL.proficiency
                r1b = session.run(phase1_proficiency, {"graph_id": graph_id}).single()
                transferred_proficiency = r1b["transferred"] if r1b else 0

                # Phase 2: detect orphans
                orphan_records = session.run(phase2_orphans, {"graph_id": graph_id})
                orphans = [
                    {"entity_id": r["entity_id"], "name": r["name"], "type": r["type"]}
                    for r in orphan_records
                ]

                # Phase 3: cleanup all banned props
                r3 = session.run(phase3_cleanup, {"graph_id": graph_id}).single()
                cleaned = r3["cleaned"] if r3 else 0

            return {
                "graph_id": graph_id,
                "transferred_job_title": transferred_job_title,
                "transferred_proficiency": transferred_proficiency,
                "transferred_total": transferred_job_title + transferred_proficiency,
                "orphans_detected": len(orphans),
                "orphans": orphans,
                "nodes_cleaned": cleaned,
            }

        except Neo4jError as e:
            raise Exception(f"Migration failed for graph {graph_id}: {e}") from None

    def graph_exists(self, graph_id: str, user_id: str) -> bool:
        """
        Check if a Graph node exists for the given user.

        Args:
            graph_id: Graph identifier
            user_id: User identifier

        Returns:
            True if graph exists, False otherwise
        """
        query = """
        MATCH (g:Graph {graph_id: $graph_id, user_id: $user_id})
        RETURN count(g) > 0 as exists
        """

        try:
            if not self.driver:
                raise ValueError("Neo4j driver not initialized")

            with self.driver.session() as session:
                result = session.run(query, {"graph_id": graph_id, "user_id": user_id})

                record = result.single()
                if record:
                    return record["exists"]
                return False

        except Neo4jError as e:
            raise Exception(f"Failed to check graph existence: {e}") from None
