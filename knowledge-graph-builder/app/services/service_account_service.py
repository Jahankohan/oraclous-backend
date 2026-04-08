"""Agent Service Account Service.

Manages AgentServiceAccount principals in Neo4j and coordinates API key
lifecycle with the auth-service (via internal endpoints).

Architecture:
  - AgentServiceAccount node + CAN_ACCESS / HAS_SERVICE_ACCOUNT edges → Neo4j
  - API key hash storage → auth-service PostgreSQL (via internal HTTP call)
  - JWT issuance → auth-service /service-token endpoint (public)

Multi-tenancy: every Cypher query includes tenant_id + graph_id filters.
"""
import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from neo4j import AsyncDriver

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Level hierarchy: each level implies all levels below it
_LEVEL_HIERARCHY: dict[str, list[str]] = {
    "admin":  ["admin", "writer", "reader"],
    "writer": ["writer", "reader"],
    "reader": ["reader"],
}

# Explicit allowlist for update_service_account — property names MUST NOT be
# interpolated into Cypher strings. This constant is the single source of truth
# for which fields are updateable; any key outside this set is rejected.
_ALLOWED_SA_UPDATE_FIELDS: frozenset[str] = frozenset({"name", "description"})


def _validate_sa_update_fields(updates: dict) -> None:
    """Raise ValueError if *updates* contains any key outside _ALLOWED_SA_UPDATE_FIELDS.

    Structural guard against Cypher property-name injection. Call this before
    building any Cypher that touches SA node properties.
    """
    unknown = set(updates) - _ALLOWED_SA_UPDATE_FIELDS
    if unknown:
        raise ValueError(f"Disallowed SA update fields: {unknown!r}")


class ServiceAccountService:
    """CRUD + permission engine for AgentServiceAccount principals."""

    # ── Schema initialization ──────────────────────────────────────────────

    async def initialize_schema(self, driver: AsyncDriver) -> None:
        """Create Neo4j constraints and indexes for AgentServiceAccount nodes.

        Idempotent — uses IF NOT EXISTS, safe to call on every startup.
        """
        queries = [
            # Uniqueness constraint
            "CREATE CONSTRAINT agent_sa_id_unique IF NOT EXISTS "
            "FOR (sa:AgentServiceAccount) REQUIRE sa.service_account_id IS UNIQUE",
            # Lookup indexes
            "CREATE INDEX agent_sa_tenant IF NOT EXISTS "
            "FOR (sa:AgentServiceAccount) ON (sa.tenant_id)",
            "CREATE INDEX agent_sa_status IF NOT EXISTS "
            "FOR (sa:AgentServiceAccount) ON (sa.status)",
        ]
        async with driver.session() as session:
            for q in queries:
                await session.run(q)
        logger.info("AgentServiceAccount Neo4j schema initialized")

    # ── SA lifecycle ───────────────────────────────────────────────────────

    async def create_service_account(
        self,
        driver: AsyncDriver,
        tenant_id: str,
        graph_id: str,
        created_by_user_id: str,
        name: str,
        description: str = "",
        level: str = "reader",
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create AgentServiceAccount in Neo4j + generate API key via auth-service.

        Returns full SA metadata including the raw api_key (shown only once).
        """
        sa_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        # 1. Create AgentServiceAccount node + ownership + default CAN_ACCESS edge
        async with driver.session() as session:
            await session.run(
                """
                MATCH (org:Organization {org_id: $tenant_id})
                MERGE (sa:AgentServiceAccount {service_account_id: $sa_id})
                ON CREATE SET
                    sa.tenant_id         = $tenant_id,
                    sa.name              = $name,
                    sa.description       = $description,
                    sa.created_by_user_id = $created_by,
                    sa.home_graph_id     = $graph_id,
                    sa.status            = 'active',
                    sa.created_at        = datetime($now),
                    sa.last_used_at      = null,
                    sa.key_prefix        = ''
                MERGE (org)-[:HAS_SERVICE_ACCOUNT]->(sa)
                WITH sa
                MATCH (g:Graph {graph_id: $graph_id})
                MERGE (sa)-[r:CAN_ACCESS]->(g)
                ON CREATE SET
                    r.level       = $level,
                    r.granted_by  = $created_by,
                    r.granted_at  = datetime($now),
                    r.expires_at  = CASE WHEN $expires_at IS NOT NULL
                                         THEN datetime($expires_at) ELSE null END,
                    r.source      = 'default'
                """,
                {
                    "sa_id":      sa_id,
                    "tenant_id":  tenant_id,
                    "name":       name,
                    "description": description,
                    "created_by": created_by_user_id,
                    "graph_id":   graph_id,
                    "level":      level,
                    "now":        now,
                    "expires_at": expires_at,
                },
            )

        # 2. Generate API key via auth-service internal endpoint
        key_data = await self._create_auth_key(
            sa_id=sa_id,
            tenant_id=tenant_id,
            home_graph_id=graph_id,
            created_by_user_id=created_by_user_id,
            expires_at=expires_at,
        )

        # 3. Store key_prefix on the Neo4j node
        async with driver.session() as session:
            await session.run(
                """
                MATCH (sa:AgentServiceAccount {service_account_id: $sa_id, tenant_id: $tenant_id})
                SET sa.key_prefix = $key_prefix
                """,
                {"sa_id": sa_id, "tenant_id": tenant_id, "key_prefix": key_data["key_prefix"]},
            )

        return {
            "service_account_id": sa_id,
            "name":               name,
            "description":        description,
            "home_graph_id":      graph_id,
            "tenant_id":          tenant_id,
            "status":             "active",
            "key_prefix":         key_data["key_prefix"],
            "api_key":            key_data["api_key"],  # ← shown ONCE
            "created_at":         now,
        }

    async def list_service_accounts(
        self, driver: AsyncDriver, graph_id: str, tenant_id: str
    ) -> List[Dict[str, Any]]:
        """List all active service accounts for a graph (admin-only endpoint)."""
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (sa:AgentServiceAccount {tenant_id: $tenant_id})
                      -[r:CAN_ACCESS]->(:Graph {graph_id: $graph_id})
                RETURN sa.service_account_id AS service_account_id,
                       sa.name               AS name,
                       sa.description        AS description,
                       sa.home_graph_id      AS home_graph_id,
                       sa.status             AS status,
                       sa.key_prefix         AS key_prefix,
                       toString(sa.created_at) AS created_at,
                       toString(sa.last_used_at) AS last_used_at,
                       r.level               AS access_level
                ORDER BY sa.created_at DESC
                """,
                {"graph_id": graph_id, "tenant_id": tenant_id},
            )
            return await result.data()

    async def get_service_account(
        self, driver: AsyncDriver, sa_id: str, tenant_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get a single service account (tenant-isolated)."""
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (sa:AgentServiceAccount {service_account_id: $sa_id, tenant_id: $tenant_id})
                RETURN sa.service_account_id AS service_account_id,
                       sa.name               AS name,
                       sa.description        AS description,
                       sa.home_graph_id      AS home_graph_id,
                       sa.status             AS status,
                       sa.key_prefix         AS key_prefix,
                       toString(sa.created_at) AS created_at,
                       toString(sa.last_used_at) AS last_used_at
                """,
                {"sa_id": sa_id, "tenant_id": tenant_id},
            )
            record = await result.single()
        return dict(record) if record else None

    async def update_service_account(
        self,
        driver: AsyncDriver,
        sa_id: str,
        tenant_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update name and/or description (tenant-isolated)."""
        if name is None and description is None:
            return await self.get_service_account(driver, sa_id, tenant_id)

        # Allowlist guard — rejects any field outside {name, description}.
        # Cypher uses explicit CASE WHEN with hardcoded property names; no dynamic
        # property name construction ever appears below.
        _validate_sa_update_fields(
            {k: v for k, v in {"name": name, "description": description}.items() if v is not None}
        )

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (sa:AgentServiceAccount {service_account_id: $sa_id, tenant_id: $tenant_id})
                SET sa.name = CASE WHEN $name IS NOT NULL THEN $name ELSE sa.name END,
                    sa.description = CASE WHEN $description IS NOT NULL THEN $description ELSE sa.description END
                RETURN sa.service_account_id AS service_account_id,
                       sa.name               AS name,
                       sa.description        AS description,
                       sa.status             AS status,
                       sa.key_prefix         AS key_prefix,
                       toString(sa.created_at) AS created_at
                """,
                {"sa_id": sa_id, "tenant_id": tenant_id, "name": name, "description": description},
            )
            record = await result.single()
        return dict(record) if record else None

    async def revoke_service_account(
        self, driver: AsyncDriver, sa_id: str, tenant_id: str
    ) -> bool:
        """Soft-revoke SA: set status=revoked in Neo4j + revoke all auth keys."""
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (sa:AgentServiceAccount {service_account_id: $sa_id, tenant_id: $tenant_id})
                WHERE sa.status = 'active'
                SET sa.status = 'revoked', sa.revoked_at = datetime()
                RETURN sa.service_account_id AS sa_id
                """,
                {"sa_id": sa_id, "tenant_id": tenant_id},
            )
            record = await result.single()

        if not record:
            return False

        # Revoke all API keys in auth-service
        await self._revoke_auth_keys(sa_id)
        return True

    async def rotate_key(
        self,
        driver: AsyncDriver,
        sa_id: str,
        tenant_id: str,
        created_by_user_id: str,
    ) -> Dict[str, Any]:
        """Rotate the API key: revoke old, create new, update key_prefix in Neo4j."""
        # Verify SA exists and is active
        sa = await self.get_service_account(driver, sa_id, tenant_id)
        if not sa:
            raise ValueError("Service account not found")
        if sa["status"] != "active":
            raise ValueError("Cannot rotate key for a revoked service account")

        # Revoke existing keys in auth-service
        await self._revoke_auth_keys(sa_id)

        # Create new key
        key_data = await self._create_auth_key(
            sa_id=sa_id,
            tenant_id=tenant_id,
            home_graph_id=sa["home_graph_id"],
            created_by_user_id=created_by_user_id,
        )

        # Update key_prefix in Neo4j
        async with driver.session() as session:
            await session.run(
                """
                MATCH (sa:AgentServiceAccount {service_account_id: $sa_id, tenant_id: $tenant_id})
                SET sa.key_prefix = $key_prefix
                """,
                {"sa_id": sa_id, "tenant_id": tenant_id, "key_prefix": key_data["key_prefix"]},
            )

        return {
            "service_account_id": sa_id,
            "key_prefix": key_data["key_prefix"],
            "api_key": key_data["api_key"],  # ← shown ONCE
            "rotated_at": datetime.now(timezone.utc).isoformat(),
        }

    # ── Graph grants ───────────────────────────────────────────────────────

    async def add_graph_grant(
        self,
        driver: AsyncDriver,
        sa_id: str,
        tenant_id: str,
        graph_id: str,
        level: str,
        granted_by_user_id: str,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Grant a service account access to an additional graph.

        Requires: SA and Graph must both belong to the same tenant.
        """
        now = datetime.now(timezone.utc).isoformat()
        grant_id = str(uuid.uuid4())

        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (sa:AgentServiceAccount {service_account_id: $sa_id, tenant_id: $tenant_id,
                                               status: 'active'})
                MATCH (g:Graph {graph_id: $graph_id})
                MATCH (org:Organization {org_id: $tenant_id})-[:OWNS]->(g)
                MERGE (sa)-[r:CAN_ACCESS]->(g)
                ON CREATE SET
                    r.grant_id    = $grant_id,
                    r.level       = $level,
                    r.granted_by  = $granted_by,
                    r.granted_at  = datetime($now),
                    r.expires_at  = CASE WHEN $expires_at IS NOT NULL
                                         THEN datetime($expires_at) ELSE null END,
                    r.source      = 'explicit'
                ON MATCH SET
                    r.level       = $level,
                    r.granted_by  = $granted_by,
                    r.granted_at  = datetime($now),
                    r.expires_at  = CASE WHEN $expires_at IS NOT NULL
                                         THEN datetime($expires_at) ELSE null END
                RETURN r.grant_id AS grant_id, r.level AS level,
                       r.granted_by AS granted_by, toString(r.granted_at) AS granted_at,
                       toString(r.expires_at) AS expires_at,
                       $graph_id AS graph_id
                """,
                {
                    "sa_id":      sa_id,
                    "tenant_id":  tenant_id,
                    "graph_id":   graph_id,
                    "level":      level,
                    "granted_by": granted_by_user_id,
                    "now":        now,
                    "expires_at": expires_at,
                    "grant_id":   grant_id,
                },
            )
            record = await result.single()

        if not record:
            raise ValueError(
                "Grant failed — SA not found, revoked, or graph belongs to a different tenant"
            )
        return dict(record)

    async def list_graph_grants(
        self, driver: AsyncDriver, sa_id: str, tenant_id: str
    ) -> List[Dict[str, Any]]:
        """List all CAN_ACCESS grants for a service account."""
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (sa:AgentServiceAccount {service_account_id: $sa_id, tenant_id: $tenant_id})
                      -[r:CAN_ACCESS]->(g:Graph)
                RETURN g.graph_id    AS graph_id,
                       g.name        AS graph_name,
                       r.level       AS level,
                       r.source      AS source,
                       r.granted_by  AS granted_by,
                       toString(r.granted_at)  AS granted_at,
                       toString(r.expires_at)  AS expires_at
                ORDER BY r.granted_at DESC
                """,
                {"sa_id": sa_id, "tenant_id": tenant_id},
            )
            return await result.data()

    async def delete_graph_grant(
        self, driver: AsyncDriver, sa_id: str, tenant_id: str, graph_id: str
    ) -> bool:
        """Remove a CAN_ACCESS edge (revoke graph access)."""
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (sa:AgentServiceAccount {service_account_id: $sa_id, tenant_id: $tenant_id})
                      -[r:CAN_ACCESS]->(g:Graph {graph_id: $graph_id})
                DELETE r
                RETURN count(r) AS deleted_count
                """,
                {"sa_id": sa_id, "tenant_id": tenant_id, "graph_id": graph_id},
            )
            record = await result.single()
        return bool(record and record["deleted_count"] > 0)

    # ── Permission check ───────────────────────────────────────────────────

    async def check_sa_graph_permission(
        self,
        driver: AsyncDriver,
        sa_id: str,
        tenant_id: str,
        graph_id: str,
        required_level: str,
    ) -> bool:
        """Return True if the SA has at least required_level on graph_id.

        Enforces:
        1. SA must be active
        2. SA and graph must share the same tenant (tenant isolation)
        3. CAN_ACCESS edge must exist with adequate level
        4. Grant must not be expired
        """
        permitted_levels = _LEVEL_HIERARCHY.get(required_level, [required_level])
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (sa:AgentServiceAccount {service_account_id: $sa_id, status: 'active'})
                MATCH (org:Organization {org_id: $tenant_id})-[:HAS_SERVICE_ACCOUNT]->(sa)
                MATCH (org)-[:OWNS]->(g:Graph {graph_id: $graph_id})
                MATCH (sa)-[r:CAN_ACCESS]->(g)
                WHERE r.level IN $permitted_levels
                  AND (r.expires_at IS NULL OR r.expires_at > datetime())
                RETURN sa.service_account_id AS sa_id
                LIMIT 1
                """,
                {
                    "sa_id":           sa_id,
                    "tenant_id":       tenant_id,
                    "graph_id":        graph_id,
                    "permitted_levels": permitted_levels,
                },
            )
            record = await result.single()
        return record is not None

    async def get_sa_accessible_graphs(
        self,
        driver: AsyncDriver,
        sa_id: str,
        tenant_id: str,
        requested_graph_ids: List[str],
    ) -> List[str]:
        """Return intersection of requested_graph_ids that SA can access.

        Used by federation middleware to resolve cross-graph access for SA principals.
        """
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (sa:AgentServiceAccount {service_account_id: $sa_id,
                                               tenant_id: $tenant_id,
                                               status: 'active'})
                      -[r:CAN_ACCESS]->(g:Graph)
                WHERE g.graph_id IN $graph_ids
                  AND (r.expires_at IS NULL OR r.expires_at > datetime())
                RETURN g.graph_id AS graph_id
                """,
                {"sa_id": sa_id, "tenant_id": tenant_id, "graph_ids": requested_graph_ids},
            )
            rows = await result.data()
        return [row["graph_id"] for row in rows]

    # ── Internal helpers (auth-service calls) ─────────────────────────────

    async def _create_auth_key(
        self,
        sa_id: str,
        tenant_id: str,
        home_graph_id: str,
        created_by_user_id: str,
        expires_at: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call auth-service internal endpoint to generate + store API key."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{settings.AUTH_SERVICE_URL}/internal/service-account-keys",
                json={
                    "service_account_id": sa_id,
                    "created_by_user_id": created_by_user_id,
                    "tenant_id": tenant_id,
                    "home_graph_id": home_graph_id,
                    "expires_at": expires_at,
                },
                headers={"X-Internal-Key": settings.INTERNAL_SERVICE_KEY},
            )
        if response.status_code != 200:
            logger.error("Auth-service key creation failed: %s %s", response.status_code, response.text)
            raise RuntimeError("Failed to create API key — auth-service error")
        return response.json()

    async def _revoke_auth_keys(self, sa_id: str) -> None:
        """Call auth-service to revoke all keys for this SA."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.delete(
                    f"{settings.AUTH_SERVICE_URL}/internal/service-account-keys/{sa_id}",
                    headers={"X-Internal-Key": settings.INTERNAL_SERVICE_KEY},
                )
            if response.status_code not in (200, 204):
                logger.warning("Auth-service key revocation returned %s", response.status_code)
        except Exception as exc:
            logger.error("Failed to revoke auth keys for SA %s: %s", sa_id, exc)


# Singleton instance
service_account_service = ServiceAccountService()
