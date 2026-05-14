"""Neo4j persistence layer for Graph-Native Agent definitions (STORY-020)."""

import json
import time
from typing import Any
from uuid import uuid4

from neo4j import AsyncDriver

from app.core.logging import get_logger
from app.schemas.agent_schemas import AgentCreate, AgentUpdate, RetrieverConfig

logger = get_logger(__name__)

_AGENT_PROPS = """
    agent_id:        $agent_id,
    graph_id:        $graph_id,
    name:            $name,
    description:     $description,
    system_prompt:   $system_prompt,
    reasoning_mode:  $reasoning_mode,
    retriever_strategy:   $retriever_strategy,
    retriever_hop_depth:  $retriever_hop_depth,
    retriever_max_results: $retriever_max_results,
    tools:           $tools,
    llm_config_id:   $llm_config_id,
    created_by:      $created_by,
    created_at:      $created_at,
    deactivated_at:  null
"""


def _row_to_dict(rec: dict) -> dict[str, Any]:
    a = rec["a"]
    props = dict(a)
    tools = props.get("tools", "[]")
    if isinstance(tools, str):
        tools = json.loads(tools)
    retriever = RetrieverConfig(
        strategy=props.get("retriever_strategy", "hybrid"),
        hop_depth=props.get("retriever_hop_depth", 2),
        max_results=props.get("retriever_max_results", 20),
    )
    return {
        "agent_id": props["agent_id"],
        "graph_id": props["graph_id"],
        "name": props["name"],
        "description": props.get("description", ""),
        "system_prompt": props.get("system_prompt", ""),
        "reasoning_mode": props.get("reasoning_mode", "direct"),
        "retriever": retriever,
        "tools": tools,
        "llm_config_id": props.get("llm_config_id"),
        "created_by": props.get("created_by", ""),
        "created_at": str(props.get("created_at", "")),
        "deactivated_at": (
            str(props["deactivated_at"]) if props.get("deactivated_at") else None
        ),
    }


class AgentService:
    def __init__(self, driver: AsyncDriver):
        self._driver = driver

    async def create_agent(self, graph_id: str, user_id: str, data: AgentCreate) -> str:
        agent_id = str(uuid4())
        now = int(time.time())
        await self._driver.execute_query(
            """
            CREATE (a:Agent:__Platform__ {
                agent_id:              $agent_id,
                graph_id:              $graph_id,
                name:                  $name,
                description:           $description,
                system_prompt:         $system_prompt,
                reasoning_mode:        $reasoning_mode,
                retriever_strategy:    $retriever_strategy,
                retriever_hop_depth:   $retriever_hop_depth,
                retriever_max_results: $retriever_max_results,
                tools:                 $tools,
                llm_config_id:         $llm_config_id,
                created_by:            $created_by,
                created_at:            $created_at,
                deactivated_at:        null
            })
            """,
            {
                "agent_id": agent_id,
                "graph_id": graph_id,
                "name": data.name,
                "description": data.description,
                "system_prompt": data.system_prompt,
                "reasoning_mode": data.reasoning_mode,
                "retriever_strategy": data.retriever.strategy,
                "retriever_hop_depth": data.retriever.hop_depth,
                "retriever_max_results": data.retriever.max_results,
                "tools": json.dumps(data.tools),
                "llm_config_id": data.llm_config_id,
                "created_by": user_id,
                "created_at": now,
            },
        )
        logger.info("Created Agent %s for graph %s", agent_id, graph_id)
        return agent_id

    async def list_agents(self, graph_id: str) -> list[dict[str, Any]]:
        result = await self._driver.execute_query(
            """
            MATCH (a:Agent:__Platform__ {graph_id: $graph_id})
            WHERE a.deactivated_at IS NULL
            RETURN a
            ORDER BY a.created_at DESC
            """,
            {"graph_id": graph_id},
        )
        return [_row_to_dict(dict(rec)) for rec in result.records]

    async def get_agent(self, graph_id: str, agent_id: str) -> dict[str, Any] | None:
        result = await self._driver.execute_query(
            """
            MATCH (a:Agent:__Platform__ {graph_id: $graph_id, agent_id: $agent_id})
            WHERE a.deactivated_at IS NULL
            RETURN a
            """,
            {"graph_id": graph_id, "agent_id": agent_id},
        )
        if not result.records:
            return None
        return _row_to_dict(dict(result.records[0]))

    async def update_agent(
        self, graph_id: str, agent_id: str, data: AgentUpdate
    ) -> dict[str, Any] | None:
        existing = await self.get_agent(graph_id, agent_id)
        if not existing:
            return None

        set_clauses: list[str] = []
        params: dict[str, Any] = {"graph_id": graph_id, "agent_id": agent_id}

        if data.name is not None:
            set_clauses.append("a.name = $name")
            params["name"] = data.name
        if data.description is not None:
            set_clauses.append("a.description = $description")
            params["description"] = data.description
        if data.system_prompt is not None:
            set_clauses.append("a.system_prompt = $system_prompt")
            params["system_prompt"] = data.system_prompt
        if data.reasoning_mode is not None:
            set_clauses.append("a.reasoning_mode = $reasoning_mode")
            params["reasoning_mode"] = data.reasoning_mode
        if data.retriever is not None:
            set_clauses += [
                "a.retriever_strategy = $retriever_strategy",
                "a.retriever_hop_depth = $retriever_hop_depth",
                "a.retriever_max_results = $retriever_max_results",
            ]
            params["retriever_strategy"] = data.retriever.strategy
            params["retriever_hop_depth"] = data.retriever.hop_depth
            params["retriever_max_results"] = data.retriever.max_results
        if data.tools is not None:
            set_clauses.append("a.tools = $tools")
            params["tools"] = json.dumps(data.tools)
        if "llm_config_id" in data.model_fields_set:
            set_clauses.append("a.llm_config_id = $llm_config_id")
            params["llm_config_id"] = data.llm_config_id

        if not set_clauses:
            return existing

        query = f"""
        MATCH (a:Agent:__Platform__ {{graph_id: $graph_id, agent_id: $agent_id}})
        WHERE a.deactivated_at IS NULL
        SET {", ".join(set_clauses)}
        RETURN a
        """
        result = await self._driver.execute_query(query, params)
        if not result.records:
            return None
        return _row_to_dict(dict(result.records[0]))

    async def deactivate_agent(self, graph_id: str, agent_id: str) -> bool:
        now = int(time.time())
        result = await self._driver.execute_query(
            """
            MATCH (a:Agent:__Platform__ {graph_id: $graph_id, agent_id: $agent_id})
            WHERE a.deactivated_at IS NULL
            SET a.deactivated_at = $now
            RETURN a.agent_id AS agent_id
            """,
            {"graph_id": graph_id, "agent_id": agent_id, "now": now},
        )
        return bool(result.records)
