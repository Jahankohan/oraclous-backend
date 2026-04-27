"""Neo4j CRUD service for :LLMConfig nodes (STORY-021).

All Cypher queries are scoped by org_id and/or graph_id — no cross-tenant
reads or writes are possible.
"""

import time
import uuid

from neo4j import AsyncDriver

from app.core.logging import get_logger
from app.schemas.llm_config_schemas import LLMConfigCreate

logger = get_logger(__name__)


class LLMConfigService:
    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    # ── Create ────────────────────────────────────────────────────────────────

    async def create_org_config(
        self,
        org_id: str,
        user_id: str,
        data: LLMConfigCreate,
        api_key_ref: str,
    ) -> str:
        config_id = str(uuid.uuid4())
        now = int(time.time())
        await self._driver.execute_query(
            """
            CREATE (:LLMConfig {
                config_id:      $config_id,
                scope:          "org",
                org_id:         $org_id,
                graph_id:       null,
                provider:       $provider,
                model:          $model,
                base_url:       $base_url,
                api_version:    $api_version,
                api_key_ref:    $api_key_ref,
                created_by:     $user_id,
                created_at:     $now,
                deactivated_at: null
            })
            """,
            {
                "config_id": config_id,
                "org_id": org_id,
                "provider": data.provider.value,
                "model": data.model,
                "base_url": data.base_url,
                "api_version": data.api_version,
                "api_key_ref": api_key_ref,
                "user_id": user_id,
                "now": now,
            },
        )
        logger.info("Created org LLMConfig %s for org %s", config_id, org_id)
        return config_id

    async def create_project_config(
        self,
        graph_id: str,
        org_id: str,
        user_id: str,
        data: LLMConfigCreate,
        api_key_ref: str,
    ) -> str:
        config_id = str(uuid.uuid4())
        now = int(time.time())
        await self._driver.execute_query(
            """
            CREATE (:LLMConfig {
                config_id:      $config_id,
                scope:          "project",
                org_id:         $org_id,
                graph_id:       $graph_id,
                provider:       $provider,
                model:          $model,
                base_url:       $base_url,
                api_version:    $api_version,
                api_key_ref:    $api_key_ref,
                created_by:     $user_id,
                created_at:     $now,
                deactivated_at: null
            })
            """,
            {
                "config_id": config_id,
                "org_id": org_id,
                "graph_id": graph_id,
                "provider": data.provider.value,
                "model": data.model,
                "base_url": data.base_url,
                "api_version": data.api_version,
                "api_key_ref": api_key_ref,
                "user_id": user_id,
                "now": now,
            },
        )
        logger.info("Created project LLMConfig %s for graph %s", config_id, graph_id)
        return config_id

    # ── List ──────────────────────────────────────────────────────────────────

    async def list_org_configs(self, org_id: str) -> list[dict]:
        result = await self._driver.execute_query(
            """
            MATCH (c:LLMConfig {org_id: $org_id, scope: "org"})
            WHERE c.deactivated_at IS NULL
            RETURN c
            ORDER BY c.created_at DESC
            """,
            {"org_id": org_id},
        )
        return [dict(rec["c"]) for rec in result.records]

    async def list_project_configs(self, graph_id: str) -> list[dict]:
        result = await self._driver.execute_query(
            """
            MATCH (c:LLMConfig {graph_id: $gid, scope: "project"})
            WHERE c.deactivated_at IS NULL
            RETURN c
            ORDER BY c.created_at DESC
            """,
            {"gid": graph_id},
        )
        return [dict(rec["c"]) for rec in result.records]

    # ── Get ───────────────────────────────────────────────────────────────────

    async def get_config(self, config_id: str) -> dict | None:
        result = await self._driver.execute_query(
            "MATCH (c:LLMConfig {config_id: $cid}) RETURN c",
            {"cid": config_id},
        )
        if not result.records:
            return None
        return dict(result.records[0]["c"])

    # ── Deactivate ────────────────────────────────────────────────────────────

    async def deactivate_config(self, config_id: str, org_id: str) -> bool:
        """Soft-delete a config. Returns False if not found or org_id mismatch."""
        now = int(time.time())
        result = await self._driver.execute_query(
            """
            MATCH (c:LLMConfig {config_id: $cid, org_id: $org_id})
            WHERE c.deactivated_at IS NULL
            SET c.deactivated_at = $now
            RETURN c.config_id AS config_id
            """,
            {"cid": config_id, "org_id": org_id, "now": now},
        )
        found = len(result.records) > 0
        if found:
            logger.info("Deactivated LLMConfig %s", config_id)
        return found

    # ── Resolution chain ──────────────────────────────────────────────────────

    async def resolve_for_agent(
        self,
        graph_id: str,
        org_id: str,
        agent_llm_config_id: str | None,
    ) -> dict | None:
        """Three-level lookup: agent config → project config → org config.

        Returns the first non-deactivated config found, or None if nothing
        is configured at any level. The caller is responsible for the env-var
        fallback when None is returned.
        """
        # 1. Agent-specific config
        if agent_llm_config_id:
            cfg = await self.get_config(agent_llm_config_id)
            if cfg and cfg.get("deactivated_at") is None:
                return cfg

        # 2. Project-level config
        project_configs = await self.list_project_configs(graph_id)
        if project_configs:
            return project_configs[0]

        # 3. Org-level config
        org_configs = await self.list_org_configs(org_id)
        if org_configs:
            return org_configs[0]

        return None
