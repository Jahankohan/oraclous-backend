"""
ReBAC (Relationship-Based Access Control) service.

Phase A — Graph Access Guard (original): simple CAN_ACCESS model for
backward-compat with existing tenants. All existing graph owners are
auto-granted admin level via sync_existing_data(), which runs once on startup.

Phase B — Full ORA-48 model: Role/Permission/SubGraph nodes with
HAS_ROLE / HAS_PERMISSION / INHERITS_FROM / APPLIES_TO / IN_SUBGRAPH
relationships. Adds fine-grained permission checking, role management, and
subgraph-scoped access control.

Permission model:
  Phase A (legacy):
    (User {graph_id:"__system__"})-[:CAN_ACCESS {level}]->(Graph {namespace:"__system__"})
  Phase B (ORA-48):
    (User)-[:HAS_ROLE {graph_id}]->(Role {graph_id})-[:HAS_PERMISSION]->(Permission)
    (Role)-[:INHERITS_FROM]->(Role)  [max depth 5]
    (Role)-[:APPLIES_TO]->(SubGraph) [scoped access]

check_graph_permission() tries Phase B first (HAS_ROLE traversal), then falls
back to Phase A (CAN_ACCESS) so existing tenants are never locked out.

Multi-tenancy: EVERY query includes graph_id filter — Architecture Rule #4.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from neo4j import AsyncDriver
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ── Phase A ────────────────────────────────────────────────────────────────
_ACCEPTABLE_LEVELS: dict[str, list[str]] = {
    "read": ["read", "write", "admin"],
    "write": ["write", "admin"],
    "admin": ["admin"],
}

_PERM_CACHE_TTL = 60  # seconds

# ── Phase B constants ──────────────────────────────────────────────────────

# Built-in role names — order matters: each inherits from the one after it.
_SYSTEM_ROLES = ["owner", "admin", "editor", "viewer", "restricted_viewer"]

# System permissions seeded once globally (idempotent MERGE).
_SYSTEM_PERMISSIONS = [
    {"name": "graph:read", "resource_type": "graph", "action": "read"},
    {"name": "graph:write", "resource_type": "graph", "action": "write"},
    {"name": "graph:delete", "resource_type": "graph", "action": "delete"},
    {"name": "graph:manage_access", "resource_type": "graph", "action": "manage"},
    {"name": "entity:read", "resource_type": "entity", "action": "read"},
    {"name": "entity:write", "resource_type": "entity", "action": "write"},
    {"name": "entity:delete", "resource_type": "entity", "action": "delete"},
    {"name": "chunk:read", "resource_type": "chunk", "action": "read"},
    {"name": "document:read", "resource_type": "document", "action": "read"},
    {"name": "document:write", "resource_type": "document", "action": "write"},
    {"name": "session:read", "resource_type": "session", "action": "read"},
    {"name": "session:write", "resource_type": "session", "action": "write"},
    {"name": "pii:read", "resource_type": "entity", "action": "read"},
]

# Permissions granted to each built-in role.
_ROLE_PERMISSIONS: dict[str, list[str]] = {
    "owner": [p["name"] for p in _SYSTEM_PERMISSIONS],  # all
    "admin": [
        "graph:read",
        "graph:write",
        "graph:manage_access",
        "entity:read",
        "entity:write",
        "entity:delete",
        "chunk:read",
        "document:read",
        "document:write",
        "session:read",
        "session:write",
        "pii:read",
    ],
    "editor": [
        "graph:read",
        "graph:write",
        "entity:read",
        "entity:write",
        "chunk:read",
        "document:read",
        "document:write",
        "session:read",
        "session:write",
    ],
    "viewer": [
        "graph:read",
        "entity:read",
        "chunk:read",
        "document:read",
        "session:read",
        "pii:read",
    ],
    "restricted_viewer": [
        "graph:read",
        "entity:read",
        "chunk:read",
        "document:read",
        "session:read",
    ],
}

# Level → minimum role name for Phase A→B bridging.
_LEVEL_TO_PERM: dict[str, str] = {
    "read": "graph:read",
    "write": "graph:write",
    "admin": "graph:manage_access",
}


class ReBACService:
    """
    ReBAC permission engine backed by Neo4j graph traversal and Redis cache.

    Phase A (legacy) 3-path resolution:
      1. Direct user CAN_ACCESS grant
      2. Team CAN_ACCESS grant  (user MEMBER_OF team → team CAN_ACCESS graph)
      3. Org ownership          (user BELONGS_TO {role:owner} org → org OWNS graph)

    Phase B (ORA-48) resolution:
      (User)-[:HAS_ROLE {graph_id}]->(Role)-[:HAS_PERMISSION|INHERITS_FROM*0..5]->...->Permission
      Falls back to Phase A if no HAS_ROLE data exists for the graph.
    """

    def __init__(self) -> None:
        self._redis: object | None = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis

            self._redis = await aioredis.from_url(
                settings.REDIS_URL, decode_responses=True
            )
        return self._redis

    # ── SCHEMA INITIALIZATION ─────────────────────────────────────────────

    async def initialize_schema(self, driver: AsyncDriver) -> None:
        """Phase A indexes — safe to call on every startup (IF NOT EXISTS)."""
        index_queries = [
            "CREATE INDEX user_id_idx IF NOT EXISTS FOR (u:User) ON (u.user_id)",
            "CREATE INDEX org_id_idx IF NOT EXISTS FOR (o:Organization) ON (o.org_id)",
            "CREATE INDEX team_id_idx IF NOT EXISTS FOR (t:Team) ON (t.team_id)",
            "CREATE INDEX rebac_graph_id_idx IF NOT EXISTS FOR (g:Graph) ON (g.graph_id)",
            "CREATE INDEX api_key_hash_idx IF NOT EXISTS FOR (k:ApiKey) ON (k.key_hash)",
        ]
        async with driver.session() as session:
            for q in index_queries:
                await session.run(q)
        logger.info("ReBAC Phase A Neo4j schema indexes created/verified")

    async def initialize_schema_full(self, driver: AsyncDriver) -> None:
        """
        Phase B indexes and constraints for Role, Permission, SubGraph nodes.
        Runs after initialize_schema(). Safe to call multiple times.
        """
        queries = [
            # ── Uniqueness constraints ────────────────────────────────────
            "CREATE CONSTRAINT role_id_unique IF NOT EXISTS FOR (r:Role) REQUIRE r.role_id IS UNIQUE",
            "CREATE CONSTRAINT permission_id_unique IF NOT EXISTS FOR (p:Permission) REQUIRE p.permission_id IS UNIQUE",
            "CREATE CONSTRAINT subgraph_id_unique IF NOT EXISTS FOR (sg:SubGraph) REQUIRE sg.subgraph_id IS UNIQUE",
            # ── Lookup indexes ────────────────────────────────────────────
            "CREATE INDEX role_graph_id IF NOT EXISTS FOR (r:Role) ON (r.graph_id)",
            "CREATE INDEX role_name_graph IF NOT EXISTS FOR (r:Role) ON (r.graph_id, r.name)",
            "CREATE INDEX subgraph_graph_id IF NOT EXISTS FOR (sg:SubGraph) ON (sg.graph_id)",
            "CREATE INDEX permission_name IF NOT EXISTS FOR (p:Permission) ON (p.name)",
            "CREATE INDEX user_email IF NOT EXISTS FOR (u:User) ON (u.email)",
            # ── Relationship property index ───────────────────────────────
            "CREATE INDEX has_role_graph_id IF NOT EXISTS FOR ()-[hr:HAS_ROLE]-() ON (hr.graph_id)",
        ]
        async with driver.session() as session:
            for q in queries:
                await session.run(q)
        logger.info("ReBAC Phase B Neo4j schema indexes/constraints created/verified")

    # ── SYSTEM PERMISSIONS SEED ───────────────────────────────────────────

    async def seed_system_permissions(self, driver: AsyncDriver) -> None:
        """
        Idempotent MERGE of all system Permission nodes. Run once on startup
        after initialize_schema_full(). Safe to re-run.
        """
        async with driver.session() as session:
            for perm in _SYSTEM_PERMISSIONS:
                await session.run(
                    """
                    MERGE (p:Permission:__System__ {name: $name})
                    ON CREATE SET
                        p.permission_id = $pid,
                        p.resource_type = $resource_type,
                        p.action        = $action,
                        p.description   = $name,
                        p.is_system     = true
                    """,
                    {
                        "name": perm["name"],
                        "pid": str(uuid4()),
                        "resource_type": perm["resource_type"],
                        "action": perm["action"],
                    },
                )
        logger.info(
            f"ReBAC: {len(_SYSTEM_PERMISSIONS)} system permissions seeded/verified"
        )

    # ── SYNC SCRIPTS ──────────────────────────────────────────────────────

    async def sync_existing_data(self, driver: AsyncDriver, db: AsyncSession) -> None:
        """
        One-shot Phase A migration: sync PostgreSQL knowledge_graphs → Neo4j.
        Creates User + Graph nodes and CAN_ACCESS {level:"admin"} edges for
        all existing graph owners. Idempotent — uses MERGE, safe to re-run.
        """
        from sqlalchemy import select

        from app.models.graph import KnowledgeGraph

        result = await db.execute(select(KnowledgeGraph))
        graphs = result.scalars().all()

        if not graphs:
            logger.info("ReBAC sync: no existing graphs found")
            return

        now = datetime.now(UTC).isoformat()
        async with driver.session() as session:
            for kg in graphs:
                await session.run(
                    """
                    MERGE (u:User:__Platform__ {user_id: $user_id, graph_id: "__system__"})
                    ON CREATE SET u.created_at = $now, u.status = "active"
                    MERGE (g:Graph:__Rebac__ {graph_id: $graph_id, namespace: "__system__"})
                    ON CREATE SET
                        g.name = $name,
                        g.owner_user_id = $user_id,
                        g.created_at = $now,
                        g.status = $status
                    MERGE (u)-[r:CAN_ACCESS]->(g)
                    ON CREATE SET r.level = "admin", r.granted_by = $user_id, r.granted_at = $now
                    """,
                    {
                        "user_id": str(kg.user_id),
                        "graph_id": str(kg.id),
                        "name": kg.name,
                        "status": kg.status or "active",
                        "now": now,
                    },
                )

        logger.info(f"ReBAC Phase A sync complete: {len(graphs)} graphs processed")

    # ── PERMISSION CHECK ──────────────────────────────────────────────────

    async def check_graph_permission(
        self,
        driver: AsyncDriver,
        user_id: str,
        graph_id: str,
        required_level: str,
    ) -> bool:
        """
        Return True if user_id has at least required_level access to graph_id.

        Resolution order:
          1. Redis cache (TTL=60s)
          2. Phase B HAS_ROLE traversal (ORA-48 model)
          3. Phase A CAN_ACCESS traversal (legacy fallback)
        """
        if not graph_id:
            raise ValueError("graph_id is required for permission check")

        acceptable = _ACCEPTABLE_LEVELS.get(required_level, ["admin"])
        required_perm = _LEVEL_TO_PERM.get(required_level, "graph:manage_access")
        cache_key = f"perm:{user_id}:{graph_id}:{required_level}"

        # 1. Redis cache
        try:
            redis = await self._get_redis()
            cached = await redis.get(cache_key)
            if cached is not None:
                return cached == "1"
        except Exception as exc:
            logger.warning(f"Redis permission cache read error: {exc}")

        authorized = False

        # 2. Phase B — HAS_ROLE traversal (ORA-48 model)
        try:
            async with driver.session() as session:
                result = await session.run(
                    """
                    MATCH (u:User {user_id: $user_id})
                      -[hr:HAS_ROLE]->(r:Role {graph_id: $graph_id})
                    WHERE hr.graph_id = $graph_id
                      AND hr.is_active = true
                      AND (hr.expires_at IS NULL OR hr.expires_at > datetime())

                    OPTIONAL MATCH (r)-[:HAS_PERMISSION|INHERITS_FROM*0..5]->
                                   (:Role)-[:HAS_PERMISSION]->(p1:Permission {name: $perm})

                    OPTIONAL MATCH (r)-[:HAS_PERMISSION]->(p2:Permission {name: $perm})

                    WITH count(p1) + count(p2) AS perm_count
                    RETURN perm_count > 0 AS authorized
                    """,
                    {
                        "user_id": user_id,
                        "graph_id": graph_id,
                        "perm": required_perm,
                    },
                )
                record = await result.single()
                if record is not None:
                    authorized = bool(record["authorized"])
                    # cache and return — Phase B data is authoritative
                    try:
                        redis = await self._get_redis()
                        await redis.set(
                            cache_key, "1" if authorized else "0", ex=_PERM_CACHE_TTL
                        )
                    except Exception:
                        pass
                    # Only return if Phase B nodes actually exist for this graph
                    # (count check: if there are any roles for this graph, trust Phase B)
                    role_check = await session.run(
                        "MATCH (r:Role {graph_id: $graph_id}) RETURN count(r) AS cnt LIMIT 1",
                        {"graph_id": graph_id},
                    )
                    role_record = await role_check.single()
                    if role_record and role_record["cnt"] > 0:
                        return authorized
        except Exception as exc:
            logger.error(f"ReBAC Phase B permission check error: {exc}")

        # 3. Phase A — CAN_ACCESS fallback
        query = """
        WITH $user_id AS uid, $graph_id AS gid, $acceptable AS ok

        OPTIONAL MATCH (u:User {user_id: uid, graph_id: "__system__"})
          -[r1:CAN_ACCESS]->
          (g1:Graph {graph_id: gid, namespace: "__system__"})
        WHERE r1.level IN ok
          AND (r1.expires_at IS NULL OR r1.expires_at > datetime())

        OPTIONAL MATCH (u2:User {user_id: uid, graph_id: "__system__"})
          -[:MEMBER_OF]->(:Team {graph_id: "__system__"})
          -[r2:CAN_ACCESS]->
          (g2:Graph {graph_id: gid, namespace: "__system__"})
        WHERE r2.level IN ok
          AND (r2.expires_at IS NULL OR r2.expires_at > datetime())

        OPTIONAL MATCH (u3:User {user_id: uid, graph_id: "__system__"})
          -[:BELONGS_TO {role: "owner"}]->(:Organization {graph_id: "__system__"})
          -[:OWNS]->
          (g3:Graph {graph_id: gid, namespace: "__system__"})

        RETURN (u IS NOT NULL OR u2 IS NOT NULL OR u3 IS NOT NULL) AS authorized
        """
        try:
            async with driver.session() as session:
                result = await session.run(
                    query,
                    {
                        "user_id": user_id,
                        "graph_id": graph_id,
                        "acceptable": acceptable,
                    },
                )
                record = await result.single()
                authorized = bool(record and record["authorized"])
        except Exception as exc:
            logger.error(f"ReBAC Phase A permission check Neo4j error: {exc}")
            return False

        try:
            redis = await self._get_redis()
            await redis.set(cache_key, "1" if authorized else "0", ex=_PERM_CACHE_TTL)
        except Exception:
            pass

        return authorized

    async def invalidate_permission_cache(self, user_id: str, graph_id: str) -> None:
        """Invalidate all cached permission entries for a user+graph pair."""
        try:
            redis = await self._get_redis()
            for level in ("read", "write", "admin"):
                await redis.delete(f"perm:{user_id}:{graph_id}:{level}")
        except Exception as exc:
            logger.warning(f"Redis permission cache invalidation error: {exc}")

    # ── GRAPH LIFECYCLE — Phase A ─────────────────────────────────────────

    async def register_new_graph(
        self,
        driver: AsyncDriver,
        user_id: str,
        graph_id: str,
        name: str,
    ) -> None:
        """
        Phase A: register a newly created graph in the legacy CAN_ACCESS model.
        Creates Graph node + CAN_ACCESS {level:"admin"} edge for the creator.
        Also bootstraps Phase B roles so fine-grained permissions are immediately
        available.
        """
        now = datetime.now(UTC).isoformat()
        async with driver.session() as session:
            await session.run(
                """
                MERGE (u:User:__Platform__ {user_id: $user_id, graph_id: "__system__"})
                ON CREATE SET u.created_at = $now, u.status = "active"
                MERGE (g:Graph:__Rebac__ {graph_id: $graph_id, namespace: "__system__"})
                ON CREATE SET
                    g.name = $name,
                    g.owner_user_id = $user_id,
                    g.created_at = $now,
                    g.status = "active"
                MERGE (u)-[r:CAN_ACCESS]->(g)
                ON CREATE SET r.level = "admin", r.granted_by = $user_id, r.granted_at = $now
                """,
                {"user_id": user_id, "graph_id": graph_id, "name": name, "now": now},
            )
        await self.invalidate_permission_cache(user_id, graph_id)

        # Bootstrap Phase B roles for new graph
        try:
            await self.bootstrap_graph_roles(driver, graph_id, owner_user_id=user_id)
        except Exception as exc:
            logger.warning(f"Phase B bootstrap failed for graph {graph_id}: {exc}")

    # ── PHASE B — ROLE BOOTSTRAP ──────────────────────────────────────────

    async def bootstrap_graph_roles(
        self,
        driver: AsyncDriver,
        graph_id: str,
        owner_user_id: str,
    ) -> None:
        """
        Create the 5 built-in Role nodes for a graph, wire HAS_PERMISSION edges
        to system Permission nodes, set up INHERITS_FROM chain, and grant
        owner role to owner_user_id.

        Idempotent — uses MERGE. Safe to re-run.
        Architecture Rule #4: all queries include graph_id.
        """
        now = datetime.now(UTC).isoformat()
        role_descriptions = {
            "owner": "Full control including deletion and role management",
            "admin": "Read/write + manage members, cannot delete graph",
            "editor": "Read + write entities/relationships, cannot manage access",
            "viewer": "Read-only on all nodes/edges",
            "restricted_viewer": "Read-only on non-PII nodes only",
        }

        async with driver.session() as session:
            # Create/merge role nodes
            role_ids: dict[str, str] = {}
            for role_name in _SYSTEM_ROLES:
                role_id = str(uuid4())
                rec = await session.run(
                    """
                    MERGE (r:Role:__System__ {graph_id: $graph_id, name: $name})
                    ON CREATE SET
                        r.role_id       = $role_id,
                        r.description   = $description,
                        r.is_system_role = true,
                        r.created_at    = $now,
                        r.created_by    = 'system'
                    ON MATCH SET r.role_id = CASE WHEN r.role_id IS NULL THEN $role_id ELSE r.role_id END
                    RETURN r.role_id AS role_id
                    """,
                    {
                        "graph_id": graph_id,
                        "name": role_name,
                        "role_id": role_id,
                        "description": role_descriptions[role_name],
                        "now": now,
                    },
                )
                record = await rec.single()
                role_ids[role_name] = record["role_id"] if record else role_id

            # Wire HAS_PERMISSION edges from each role to its permissions
            for role_name, perms in _ROLE_PERMISSIONS.items():
                for perm_name in perms:
                    await session.run(
                        """
                        MATCH (r:Role {graph_id: $graph_id, name: $role_name})
                        MATCH (p:Permission {name: $perm_name})
                        MERGE (r)-[hp:HAS_PERMISSION]->(p)
                        ON CREATE SET hp.graph_id = $graph_id, hp.granted_at = $now
                        """,
                        {
                            "graph_id": graph_id,
                            "role_name": role_name,
                            "perm_name": perm_name,
                            "now": now,
                        },
                    )

            # INHERITS_FROM chain: owner→admin→editor→viewer
            inheritance = [
                ("owner", "admin"),
                ("admin", "editor"),
                ("editor", "viewer"),
                ("owner", "restricted_viewer"),
                ("admin", "restricted_viewer"),
            ]
            for parent, child in inheritance:
                await session.run(
                    """
                    MATCH (parent:Role {graph_id: $graph_id, name: $parent})
                    MATCH (child:Role {graph_id: $graph_id, name: $child})
                    MERGE (parent)-[i:INHERITS_FROM]->(child)
                    ON CREATE SET i.graph_id = $graph_id, i.created_at = $now
                    """,
                    {
                        "graph_id": graph_id,
                        "parent": parent,
                        "child": child,
                        "now": now,
                    },
                )

            # Grant owner role to creating user
            await session.run(
                """
                MERGE (u:User:__Platform__ {user_id: $user_id})
                ON CREATE SET u.created_at = $now, u.is_service_account = false
                WITH u
                MATCH (r:Role:__System__ {graph_id: $graph_id, name: 'owner'})
                MERGE (u)-[hr:HAS_ROLE {graph_id: $graph_id}]->(r)
                ON CREATE SET
                    hr.granted_at  = $now,
                    hr.granted_by  = 'system',
                    hr.expires_at  = null,
                    hr.is_active   = true
                ON MATCH SET hr.is_active = true
                """,
                {"user_id": owner_user_id, "graph_id": graph_id, "now": now},
            )

        await self.invalidate_permission_cache(owner_user_id, graph_id)
        logger.info(f"Phase B bootstrap complete for graph {graph_id}")

    # ── PHASE B — ROLE MANAGEMENT ─────────────────────────────────────────

    async def grant_role(
        self,
        driver: AsyncDriver,
        graph_id: str,
        target_user_id: str,
        role_name: str,
        granted_by: str,
        expires_at: str | None = None,
        email: str | None = None,
    ) -> None:
        """
        Grant target_user_id the named role on graph_id.
        Creates the User node if it doesn't exist.
        Invalidates permission cache for the user.
        Architecture Rule #4: graph_id is required.
        """
        if not graph_id:
            raise ValueError("graph_id is required for grant_role")

        now = datetime.now(UTC).isoformat()
        async with driver.session() as session:
            await session.run(
                """
                MERGE (u:User:__Platform__ {user_id: $user_id})
                ON CREATE SET
                    u.created_at       = $now,
                    u.is_service_account = false,
                    u.email            = $email
                ON MATCH SET u.email = CASE WHEN $email IS NOT NULL THEN $email ELSE u.email END
                WITH u
                MATCH (r:Role {graph_id: $graph_id, name: $role_name})
                MERGE (u)-[hr:HAS_ROLE {graph_id: $graph_id}]->(r)
                ON CREATE SET
                    hr.granted_at = $now,
                    hr.granted_by = $granted_by,
                    hr.expires_at = $expires_at,
                    hr.is_active  = true
                ON MATCH SET
                    hr.granted_at = $now,
                    hr.granted_by = $granted_by,
                    hr.expires_at = $expires_at,
                    hr.is_active  = true
                """,
                {
                    "user_id": target_user_id,
                    "graph_id": graph_id,
                    "role_name": role_name,
                    "granted_by": granted_by,
                    "expires_at": expires_at,
                    "email": email,
                    "now": now,
                },
            )
        await self.invalidate_permission_cache(target_user_id, graph_id)

    async def revoke_role(
        self,
        driver: AsyncDriver,
        graph_id: str,
        target_user_id: str,
        role_name: str,
    ) -> int:
        """
        Soft-revoke target_user_id's named role on graph_id (sets is_active=false).
        Returns the count of revoked edges (0 = no matching edge found).
        Architecture Rule #4: graph_id required.
        """
        if not graph_id:
            raise ValueError("graph_id is required for revoke_role")

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (u:User {user_id: $user_id})
                  -[hr:HAS_ROLE {graph_id: $graph_id}]->
                  (r:Role {graph_id: $graph_id, name: $role_name})
                SET hr.is_active = false
                RETURN count(hr) AS revoked_count
                """,
                {
                    "user_id": target_user_id,
                    "graph_id": graph_id,
                    "role_name": role_name,
                },
            )
            record = await result.single()
            count = record["revoked_count"] if record else 0

        await self.invalidate_permission_cache(target_user_id, graph_id)
        return count

    # ── PHASE B — MEMBER QUERIES ──────────────────────────────────────────

    async def list_graph_members(
        self, driver: AsyncDriver, graph_id: str
    ) -> list[dict]:
        """
        Return all users with active HAS_ROLE grants on graph_id.
        Architecture Rule #4: filtered by graph_id.
        """
        if not graph_id:
            raise ValueError("graph_id is required for list_graph_members")

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (u:User)-[hr:HAS_ROLE {graph_id: $graph_id}]->(r:Role {graph_id: $graph_id})
                WHERE hr.is_active = true
                  AND (hr.expires_at IS NULL OR hr.expires_at > datetime())
                RETURN
                    u.user_id   AS user_id,
                    u.email     AS email,
                    r.name      AS role,
                    hr.granted_at AS granted_at,
                    hr.expires_at AS expires_at
                ORDER BY r.name, u.user_id
                """,
                {"graph_id": graph_id},
            )
            return [
                {
                    "user_id": r["user_id"],
                    "email": r["email"],
                    "role": r["role"],
                    "granted_at": str(r["granted_at"]) if r["granted_at"] else None,
                    "expires_at": str(r["expires_at"]) if r["expires_at"] else None,
                }
                async for r in result
            ]

    async def get_user_access_filter(
        self, driver: AsyncDriver, user_id: str, graph_id: str
    ) -> dict:
        """
        Return the access filter for a user on a graph.
        Used by the retrieval layer to restrict entity queries.

        Returns:
            {
              "has_global_read": bool,
              "allowed_subgraph_ids": list[str],  # empty = global access
            }
        Architecture Rule #4: all queries filtered by graph_id.
        """
        if not graph_id:
            raise ValueError("graph_id is required for get_user_access_filter")

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (u:User {user_id: $user_id})
                  -[hr:HAS_ROLE {graph_id: $graph_id}]->
                  (r:Role {graph_id: $graph_id})
                WHERE hr.is_active = true
                  AND (hr.expires_at IS NULL OR hr.expires_at > datetime())

                OPTIONAL MATCH (r)-[:APPLIES_TO]->(sg:SubGraph {graph_id: $graph_id})

                OPTIONAL MATCH (r)-[:HAS_PERMISSION|INHERITS_FROM*0..5]->
                               (:Role)-[:HAS_PERMISSION]->(p:Permission {name: 'graph:read'})
                OPTIONAL MATCH (r)-[:HAS_PERMISSION]->(p2:Permission {name: 'graph:read'})

                WITH
                    count(p) + count(p2) > 0 AS has_global_read,
                    collect(DISTINCT sg.subgraph_id) AS allowed_subgraph_ids

                RETURN has_global_read, allowed_subgraph_ids
                """,
                {"user_id": user_id, "graph_id": graph_id},
            )
            record = await result.single()
            if record is None:
                return {"has_global_read": False, "allowed_subgraph_ids": []}
            return {
                "has_global_read": bool(record["has_global_read"]),
                "allowed_subgraph_ids": [
                    s for s in record["allowed_subgraph_ids"] if s
                ],
            }

    # ── PHASE B — SUBGRAPH MANAGEMENT ────────────────────────────────────

    async def create_subgraph(
        self,
        driver: AsyncDriver,
        graph_id: str,
        name: str,
        description: str | None = None,
        created_by: str | None = None,
    ) -> dict:
        """
        Create a named SubGraph partition within graph_id.
        Architecture Rule #4: SubGraph.graph_id is always set.
        """
        if not graph_id:
            raise ValueError("graph_id is required for create_subgraph")

        subgraph_id = str(uuid4())
        now = datetime.now(UTC).isoformat()
        async with driver.session() as session:
            result = await session.run(
                """
                MERGE (sg:SubGraph:__Platform__ {graph_id: $graph_id, name: $name})
                ON CREATE SET
                    sg.subgraph_id  = $subgraph_id,
                    sg.description  = $description,
                    sg.created_at   = $now,
                    sg.created_by   = $created_by
                RETURN
                    sg.subgraph_id AS subgraph_id,
                    sg.name        AS name,
                    sg.description AS description,
                    sg.created_at  AS created_at
                """,
                {
                    "graph_id": graph_id,
                    "name": name,
                    "subgraph_id": subgraph_id,
                    "description": description,
                    "created_by": created_by,
                    "now": now,
                },
            )
            record = await result.single()
            return {
                "subgraph_id": record["subgraph_id"],
                "graph_id": graph_id,
                "name": record["name"],
                "description": record["description"],
                "created_at": (
                    str(record["created_at"]) if record["created_at"] else now
                ),
            }

    async def list_subgraphs(self, driver: AsyncDriver, graph_id: str) -> list[dict]:
        """
        List all SubGraph partitions for graph_id.
        Architecture Rule #4: filtered by graph_id.
        """
        if not graph_id:
            raise ValueError("graph_id is required for list_subgraphs")

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (sg:SubGraph {graph_id: $graph_id})
                RETURN
                    sg.subgraph_id AS subgraph_id,
                    sg.name        AS name,
                    sg.description AS description,
                    sg.created_at  AS created_at
                ORDER BY sg.name
                """,
                {"graph_id": graph_id},
            )
            return [
                {
                    "subgraph_id": r["subgraph_id"],
                    "graph_id": graph_id,
                    "name": r["name"],
                    "description": r["description"],
                    "created_at": str(r["created_at"]) if r["created_at"] else None,
                }
                async for r in result
            ]


rebac_service = ReBACService()
