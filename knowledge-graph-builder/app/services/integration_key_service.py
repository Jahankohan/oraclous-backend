"""Integration key management for published agents (STORY-022)."""

import hashlib
import secrets
import time
import uuid

from app.core.logging import get_logger

logger = get_logger(__name__)


class IntegrationKeyService:
    def __init__(self, driver):
        self._driver = driver

    # ── Key utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def generate_key() -> str:
        """Generate a 32-byte URL-safe key prefixed with 'oak-'."""
        return "oak-" + secrets.token_urlsafe(32)

    @staticmethod
    def hash_key(key: str) -> str:
        """SHA-256 hash of the key for storage. 256-bit entropy makes brute-force infeasible."""
        return hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def verify_key(key: str, key_hash: str) -> bool:
        """Constant-time comparison to resist timing attacks."""
        candidate = hashlib.sha256(key.encode()).hexdigest()
        return secrets.compare_digest(candidate, key_hash)

    # ── CRUD ──────────────────────────────────────────────────────────────────

    async def publish_agent(
        self,
        agent_id: str,
        graph_id: str,
        org_id: str,
        user_id: str,
        slug: str,
        cors_origins: list[str],
        rate_limit_rpm: int = 60,
        egress_url: str | None = None,
    ) -> tuple[str, str]:
        """
        Create a :PublishedAgent node. Returns (plaintext_key, slug).
        Raises ValueError if the slug is already taken by an active published agent.
        """
        key = self.generate_key()
        key_hash = self.hash_key(key)
        key_last4 = key[-4:]
        now = int(time.time())

        result = await self._driver.execute_query(
            """
            MATCH (a:Agent {agent_id: $agent_id, graph_id: $graph_id})
            OPTIONAL MATCH (existing:PublishedAgent {slug: $slug})
              WHERE existing.unpublished_at IS NULL
            WITH a, existing
            WHERE a IS NOT NULL AND existing IS NULL
            CREATE (p:PublishedAgent {
                agent_id:       $agent_id,
                graph_id:       $graph_id,
                org_id:         $org_id,
                slug:           $slug,
                cors_origins:   $cors_origins,
                rate_limit_rpm: $rate_limit_rpm,
                egress_url:     $egress_url,
                key_hash:       $key_hash,
                key_last4:      $key_last4,
                published_at:   $now,
                unpublished_at: null,
                created_by:     $user_id
            })
            RETURN p.slug AS slug
            """,
            {
                "agent_id": agent_id,
                "graph_id": graph_id,
                "org_id": org_id,
                "slug": slug,
                "cors_origins": cors_origins,
                "rate_limit_rpm": rate_limit_rpm,
                "egress_url": egress_url,
                "key_hash": key_hash,
                "key_last4": key_last4,
                "now": now,
                "user_id": user_id,
            },
        )
        if not result.records:
            # Check why — slug conflict vs agent not found
            check = await self._driver.execute_query(
                "MATCH (p:PublishedAgent {slug: $slug}) WHERE p.unpublished_at IS NULL RETURN p",
                {"slug": slug},
            )
            if check.records:
                raise ValueError(f"Slug '{slug}' is already taken by an active published agent")
            raise ValueError(f"Agent {agent_id} not found in graph {graph_id}")
        return key, slug

    async def unpublish_agent(self, agent_id: str, graph_id: str) -> bool:
        """Set unpublished_at. Returns True if found and was active."""
        now = int(time.time())
        result = await self._driver.execute_query(
            """
            MATCH (p:PublishedAgent {agent_id: $agent_id, graph_id: $graph_id})
            WHERE p.unpublished_at IS NULL
            SET p.unpublished_at = $now
            RETURN p.slug AS slug
            """,
            {"agent_id": agent_id, "graph_id": graph_id, "now": now},
        )
        return len(result.records) > 0

    async def rotate_key(self, agent_id: str, graph_id: str) -> str:
        """Generate and store a new key. Returns the new plaintext key."""
        new_key = self.generate_key()
        new_hash = self.hash_key(new_key)
        new_last4 = new_key[-4:]

        result = await self._driver.execute_query(
            """
            MATCH (p:PublishedAgent {agent_id: $agent_id, graph_id: $graph_id})
            WHERE p.unpublished_at IS NULL
            SET p.key_hash = $key_hash, p.key_last4 = $key_last4
            RETURN p.slug AS slug
            """,
            {"agent_id": agent_id, "graph_id": graph_id, "key_hash": new_hash, "key_last4": new_last4},
        )
        if not result.records:
            raise ValueError(f"No active published agent {agent_id} in graph {graph_id}")
        return new_key

    async def get_published(self, slug: str) -> dict | None:
        """Fetch active :PublishedAgent by slug. Returns None if not found or unpublished."""
        result = await self._driver.execute_query(
            """
            MATCH (p:PublishedAgent {slug: $slug})
            WHERE p.unpublished_at IS NULL
            RETURN p {.*} AS p
            """,
            {"slug": slug},
        )
        if not result.records:
            return None
        return dict(result.records[0]["p"])

    async def get_published_by_agent(self, agent_id: str, graph_id: str) -> dict | None:
        """Fetch active :PublishedAgent by agent_id + graph_id."""
        result = await self._driver.execute_query(
            """
            MATCH (p:PublishedAgent {agent_id: $agent_id, graph_id: $graph_id})
            WHERE p.unpublished_at IS NULL
            RETURN p {.*} AS p
            """,
            {"agent_id": agent_id, "graph_id": graph_id},
        )
        if not result.records:
            return None
        return dict(result.records[0]["p"])

    async def validate_key(self, slug: str, api_key: str) -> dict | None:
        """Verify key against stored hash. Returns the :PublishedAgent dict on success."""
        published = await self.get_published(slug)
        if not published:
            return None
        if not self.verify_key(api_key, published["key_hash"]):
            return None
        return published
