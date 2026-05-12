"""Agent execution engine — four reasoning modes (TASK-034 / STORY-020).

AgentExecutor is instantiated per chat request. It loads the agent definition
from Neo4j, resolves the LLM client via the three-level LLMConfig chain
(agent → project → org → env-var fallback), and dispatches to the correct
reasoning mode. Every execution produces a ProvenancePayload that the caller
returns alongside the agent's response.

Reasoning modes
---------------
direct          Single retrieval pass → generate (≤ 2 LLM calls)
research        ReAct loop: retrieve → think → retrieve → generate (≤ 5 iterations)
analytical      CoT reasoning section prepended before the final answer
conversational  Full session history injected; warm tone system prompt suffix
"""

import json
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from neo4j import AsyncDriver
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.agent_schemas import AgentChatResponse, NodeResult, PathResult
from app.services.agent_service import AgentService
from app.services.agent_tools import AgentToolkit, ToolNotPermittedError
from app.services.credential_broker_client import (
    CredentialBrokerClient,
    CredentialBrokerError,
)
from app.services.llm_client_factory import LLMClientFactory
from app.services.llm_config_service import LLMConfigService
from app.services.provenance import ProvenanceCollector

logger = get_logger(__name__)

# Process-scoped in-memory session store (conversational mode).
# key: session_id, value: list of {"role": ..., "content": ...} dicts
_sessions: dict[str, list[dict]] = {}

_REACT_SYSTEM = """You are a graph-native research agent. To answer the question you \
may call tools one at a time. After each tool result you will decide whether to call \
another tool or produce a final answer.

Respond ONLY with valid JSON — no prose, no markdown fences:
  Call a tool : {{"action": "<tool_name>", "args": {{...}}}}
  Final answer: {{"action": "answer", "text": "<your response>"}}

Available tools: {tools}
"""

_CONVERSATIONAL_SUFFIX = (
    "\n\nYou are having a friendly, ongoing conversation. "
    "Reference earlier turns naturally where relevant."
)


def _format_nodes(nodes: list[NodeResult]) -> str:
    if not nodes:
        return "(no results)"
    lines = []
    for n in nodes:
        qn = f" ({n.qualified_name})" if n.qualified_name else ""
        lines.append(f"- [{n.id}] {n.label}{qn}")
    return "\n".join(lines)


def _extract_cited_nodes(response: str, nodes: list[dict]) -> list[str]:
    """Return IDs of nodes whose id string appears literally in the response."""
    return [n["id"] for n in nodes if n["id"] in response]


class AgentExecutor:
    """Runs a single agent's reasoning loop for one chat turn."""

    def __init__(
        self,
        agent_def: dict[str, Any],
        toolkit: AgentToolkit,
        llm: AsyncOpenAI,
        model: str,
    ) -> None:
        self._agent = agent_def
        self._toolkit = toolkit
        self._llm = llm
        self._model = model

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def from_neo4j(
        cls,
        driver: AsyncDriver,
        graph_id: str,
        agent_id: str,
    ) -> "AgentExecutor":
        svc = AgentService(driver)
        agent_def = await svc.get_agent(graph_id, agent_id)
        if not agent_def:
            raise ValueError(f"Agent '{agent_id}' not found in graph '{graph_id}'")

        # Resolve LLM config: agent → project → org → env-var fallback
        org_id = agent_def.get("org_id") or ""
        svc_config = LLMConfigService(driver)
        resolved = await svc_config.resolve_for_agent(
            graph_id=graph_id,
            org_id=org_id,
            agent_llm_config_id=agent_def.get("llm_config_id"),
        )

        if resolved:
            try:
                broker = CredentialBrokerClient(settings.CREDENTIAL_BROKER_URL)
                api_key = await broker.retrieve_api_key(resolved["api_key_ref"])
            except CredentialBrokerError as exc:
                raise RuntimeError(f"Could not retrieve LLM API key: {exc}") from exc
            config_dict = {**resolved, "api_key": api_key}
            llm = LLMClientFactory.build(config_dict)
            model = resolved["model"]
        else:
            api_key = settings.LLM_API_KEY or settings.OPENAI_API_KEY
            if not api_key:
                raise RuntimeError(
                    "LLM not configured. Set an LLM config at org or project level."
                )
            model = settings.LLM_MODEL
            llm = AsyncOpenAI(api_key=api_key)

        toolkit = AgentToolkit(driver, agent_def["tools"])
        return cls(agent_def, toolkit, llm, model)

    # ── Public entry point ────────────────────────────────────────────────────

    async def run(self, message: str, session_id: str | None) -> AgentChatResponse:
        prov = ProvenanceCollector()
        mode = self._agent.get("reasoning_mode", "direct")
        returned_session_id = session_id

        if mode == "direct":
            response = await self._direct(message, prov)
        elif mode == "research":
            response = await self._research(message, prov)
        elif mode == "analytical":
            response = await self._analytical(message, prov)
        elif mode == "conversational":
            response, returned_session_id = await self._conversational(
                message, session_id, prov
            )
        else:
            response = await self._direct(message, prov)

        prov.nodes_used_in_response = _extract_cited_nodes(response, prov._nodes)
        return AgentChatResponse(
            response=response,
            session_id=returned_session_id,
            provenance=prov.to_payload(),
        )

    # ── LLM helpers ───────────────────────────────────────────────────────────

    async def _call_llm(
        self, messages: list[dict], system_prompt: str | None = None
    ) -> str:
        sp = system_prompt or self._agent.get(
            "system_prompt", "You are a helpful assistant."
        )

        try:
            from anthropic import AsyncAnthropic

            _is_anthropic = isinstance(self._llm, AsyncAnthropic)
        except ImportError:
            _is_anthropic = False

        if _is_anthropic:
            resp = await self._llm.messages.create(
                model=self._model,
                max_tokens=4096,
                system=sp,
                messages=messages,
            )
            return resp.content[0].text or ""
        else:
            all_msgs = [{"role": "system", "content": sp}] + messages
            resp = await self._llm.chat.completions.create(
                model=self._model,
                messages=all_msgs,
                temperature=0.7,
            )
            return resp.choices[0].message.content or ""

    async def _call_llm_stream(
        self, messages: list[dict], system_prompt: str | None = None
    ) -> AsyncGenerator[str, None]:
        """Yield LLM response tokens one at a time.

        Phase 1: direct mode only. For Anthropic uses the streaming context manager;
        for OpenAI-compatible uses stream=True. Non-direct modes use run() instead.
        """
        sp = system_prompt or self._agent.get(
            "system_prompt", "You are a helpful assistant."
        )

        try:
            from anthropic import AsyncAnthropic

            _is_anthropic = isinstance(self._llm, AsyncAnthropic)
        except ImportError:
            _is_anthropic = False

        if _is_anthropic:
            async with self._llm.messages.stream(
                model=self._model,
                max_tokens=4096,
                system=sp,
                messages=messages,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        else:
            all_msgs = [{"role": "system", "content": sp}] + messages
            async with await self._llm.chat.completions.create(
                model=self._model,
                messages=all_msgs,
                temperature=0.7,
                stream=True,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content if chunk.choices else None
                    if delta:
                        yield delta

    async def run_stream(
        self, message: str, session_id: str | None
    ) -> AsyncGenerator[tuple[str | None, Any], None]:
        """Streaming variant of run(). Yields (token, None) pairs, then (None, provenance).

        Only direct mode streams token-by-token. Other modes yield the full response
        as a single token then the provenance payload.
        """
        prov = ProvenanceCollector()
        mode = self._agent.get("reasoning_mode", "direct")

        if mode == "direct":
            context = ""
            if "graph_search" in self._agent.get("tools", []):
                nodes = await self._dispatch("graph_search", {"query": message}, prov)
                context = _format_nodes(nodes)

            user_content = (
                f"Context:\n{context}\n\nQuestion: {message}" if context else message
            )
            sp = self._agent.get("system_prompt", "You are a helpful assistant.")

            full_response = ""
            async for token in self._call_llm_stream(
                [{"role": "user", "content": user_content}], system_prompt=sp
            ):
                full_response += token
                yield token, None

            prov.nodes_used_in_response = _extract_cited_nodes(
                full_response, prov._nodes
            )
            yield None, prov.to_payload()
        else:
            # Non-direct modes: collect full response then yield once
            chat_resp = await self.run(message, session_id)
            yield chat_resp.response, None
            yield None, chat_resp.provenance

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    async def _dispatch(
        self, name: str, args: dict, prov: ProvenanceCollector
    ) -> list[NodeResult]:
        graph_id = self._agent["graph_id"]
        method = getattr(self._toolkit, name, None)
        if method is None:
            raise ToolNotPermittedError(name)

        result = await method(graph_id, **args)

        if isinstance(result, PathResult):
            nodes = result.nodes
        elif result is None:
            nodes = []
        else:
            nodes = result

        prov.record_tool(name, nodes)
        return nodes

    # ── Reasoning modes ───────────────────────────────────────────────────────

    async def _direct(self, message: str, prov: ProvenanceCollector) -> str:
        context = ""
        if "graph_search" in self._agent.get("tools", []):
            nodes = await self._dispatch("graph_search", {"query": message}, prov)
            context = _format_nodes(nodes)

        user_content = (
            f"Context:\n{context}\n\nQuestion: {message}" if context else message
        )
        return await self._call_llm([{"role": "user", "content": user_content}])

    async def _research(self, message: str, prov: ProvenanceCollector) -> str:
        tools_list = ", ".join(self._agent.get("tools", []))
        system = _REACT_SYSTEM.format(tools=tools_list)
        messages: list[dict] = [{"role": "user", "content": message}]

        for _ in range(5):
            raw = await self._call_llm(messages, system_prompt=system)
            messages.append({"role": "assistant", "content": raw})

            try:
                decision = json.loads(raw.strip())
            except json.JSONDecodeError:
                # Unparseable response — treat as final answer
                return raw

            action = decision.get("action", "answer")
            if action == "answer":
                return decision.get("text", raw)

            # Tool call
            args = decision.get("args", {})
            try:
                nodes = await self._dispatch(action, args, prov)
            except (ToolNotPermittedError, TypeError) as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool error: {exc}. Try a different tool.",
                    }
                )
                continue

            tool_result = _format_nodes(nodes)
            messages.append(
                {"role": "user", "content": f"Tool '{action}' result:\n{tool_result}"}
            )

        # Iteration cap reached — generate final answer with accumulated context
        messages.append(
            {
                "role": "user",
                "content": "Please provide your final answer based on the information gathered so far.",
            }
        )
        return await self._call_llm(messages, system_prompt=system)

    async def _analytical(self, message: str, prov: ProvenanceCollector) -> str:
        context = ""
        if "graph_search" in self._agent.get("tools", []):
            nodes = await self._dispatch("graph_search", {"query": message}, prov)
            context = _format_nodes(nodes)

        user_content = (
            (
                f"Context:\n{context}\n\nQuestion: {message}\n\n"
                "Think step by step before giving your final answer.\n\n"
                "Reasoning:\n"
            )
            if context
            else (
                f"{message}\n\nThink step by step before giving your final answer.\n\nReasoning:\n"
            )
        )
        return await self._call_llm([{"role": "user", "content": user_content}])

    async def _conversational(
        self, message: str, session_id: str | None, prov: ProvenanceCollector
    ) -> tuple[str, str]:
        if session_id is None:
            session_id = str(uuid4())

        history = list(_sessions.get(session_id, []))

        context = ""
        if "graph_search" in self._agent.get("tools", []):
            nodes = await self._dispatch("graph_search", {"query": message}, prov)
            context = _format_nodes(nodes)

        user_content = (
            f"Context:\n{context}\n\nQuestion: {message}" if context else message
        )

        sp = self._agent.get("system_prompt", "You are a helpful assistant.")
        system = sp + _CONVERSATIONAL_SUFFIX
        messages = history + [{"role": "user", "content": user_content}]
        response = await self._call_llm(messages, system_prompt=system)

        _sessions[session_id] = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]

        return response, session_id
