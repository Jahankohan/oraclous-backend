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

OUTPUT FORMAT — STRICTLY ONE JSON OBJECT, NOTHING ELSE:
  - To call a tool, emit ONLY: {{"action": "<tool_name>", "args": {{...}}}}
  - To produce the final answer, emit ONLY: {{"action": "answer", "text": "<your response>"}}

Hard rules about output:
  - No prose before or after the JSON.
  - No markdown code fences (```).
  - No <function_calls> tags, no <invoke> tags, no XML of any kind.
  - No comments inside the JSON.
  - Exactly one JSON object per turn.

The tool will be executed by the system. You will see its result in the next turn \
as a user message labeled "Tool 'X' result:". Then decide your next move with \
another single JSON object.

Available tools: {tools}
"""


import re as _re

_FUNCTION_CALLS_BLOCK = _re.compile(
    r"<function_calls>\s*(\[.+?\]|\{.+?\})\s*</function_calls>", _re.DOTALL
)
_FENCED_JSON = _re.compile(r"```(?:json)?\s*(\{.+?\})\s*```", _re.DOTALL)


def _scan_json_object(s: str, start: int) -> str | None:
    """Return the substring of s starting at s[start] (must be '{') that
    spans the matching closing '}' with string-aware bracket counting.
    Returns None if no balanced object is found."""
    if start >= len(s) or s[start] != "{":
        return None
    depth = 0
    i = start
    in_str = False
    str_quote = ""
    escape = False
    while i < len(s):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == str_quote:
                in_str = False
        else:
            if ch in ('"', "'"):
                in_str = True
                str_quote = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start : i + 1]
        i += 1
    return None


def _find_first_json_object(s: str) -> str | None:
    """Find the first balanced JSON object substring in s, handling nested
    braces inside strings (e.g. Cypher queries with `{prop: value}` literals)."""
    pos = 0
    while True:
        idx = s.find("{", pos)
        if idx == -1:
            return None
        candidate = _scan_json_object(s, idx)
        if candidate is not None:
            return candidate
        pos = idx + 1


def _normalize_decision(obj: dict) -> dict:
    """Map Claude's native tool-use shape onto the ReAct shape we dispatch on.

    Claude sometimes emits {"type": "tool", "id": "<tool_name>", "args": {...}}
    or {"name": "<tool_name>", "input": {...}} instead of {"action": ..., "args": ...}.
    """
    if "action" in obj:
        return obj
    if obj.get("type") == "tool" and obj.get("id"):
        return {"action": obj["id"], "args": obj.get("args") or obj.get("input") or {}}
    if "name" in obj and ("input" in obj or "args" in obj):
        return {
            "action": obj["name"],
            "args": obj.get("args") or obj.get("input") or {},
        }
    return obj


def _parse_react_decision(raw: str) -> dict | None:
    """Robustly parse the agent's decision JSON.

    Handles five shapes:
      1. Pure JSON object.
      2. Claude tool-use wrapper: <function_calls>[{...}]</function_calls>.
      3. Markdown-fenced JSON: ```json {...} ```.
      4. JSON object embedded in prose (Cypher-safe bracket counting).
      5. Normalizes Anthropic-style tool-use fields ({"type":"tool","id":...}
         or {"name":...,"input":...}) to the expected {"action":...,"args":...}.
    Returns None if no JSON object can be recovered.
    """
    s = raw.strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return _normalize_decision(obj)
        if isinstance(obj, list) and obj and isinstance(obj[0], dict):
            return _normalize_decision(obj[0])
    except json.JSONDecodeError:
        pass
    m = _FUNCTION_CALLS_BLOCK.search(s)
    if m:
        try:
            inner = json.loads(m.group(1))
            if isinstance(inner, list) and inner and isinstance(inner[0], dict):
                return _normalize_decision(inner[0])
            if isinstance(inner, dict):
                return _normalize_decision(inner)
        except json.JSONDecodeError:
            pass
    m = _FENCED_JSON.search(s)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return _normalize_decision(obj)
        except json.JSONDecodeError:
            pass
    candidate = _find_first_json_object(s)
    if candidate:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                normalized = _normalize_decision(obj)
                if "action" in normalized or "text" in normalized:
                    return normalized
        except json.JSONDecodeError:
            pass
    return None


_CONVERSATIONAL_SUFFIX = (
    "\n\nYou are having a friendly, ongoing conversation. "
    "Reference earlier turns naturally where relevant."
)


def _format_nodes(nodes: list[NodeResult]) -> str:
    if not nodes:
        return "(no results)"
    lines = []
    for n in nodes:
        # Chunks have no id/label/name; the substance lives in properties['text'].
        # Surface that text so the agent's LLM has grounded content to cite.
        props = getattr(n, "properties", {}) or {}
        chunk_text = props.get("text") or props.get("content") or ""
        identifier = n.id or props.get("element_id") or ""
        label = (
            n.label or (props.get("nodeLabels") or [""])[0]
            if isinstance(props.get("nodeLabels"), list)
            else (n.label or "")
        )
        qn = f" ({n.qualified_name})" if n.qualified_name else ""
        header = f"- [{identifier}] {label}{qn}".rstrip()
        if chunk_text:
            # Limit to 1500 chars per chunk to keep prompt within budget for top_k=10
            preview = chunk_text[:1500].strip()
            lines.append(f"{header}\n  {preview}")
        else:
            lines.append(header)
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

        # Embedder for tools that need vector search (graph_search). Uses the
        # same OPENAI_BASE_URL/OPENAI_API_KEY env as the ingest pipeline so it
        # routes through whatever the deployment has configured (OpenAI, LM
        # Studio, OpenRouter, etc.).
        from neo4j_graphrag.embeddings import OpenAIEmbeddings as _ToolEmbedder

        embedder = _ToolEmbedder(
            api_key=settings.OPENAI_API_KEY,
            model=settings.EMBEDDING_MODEL or "text-embedding-3-large",
        )
        toolkit = AgentToolkit(driver, agent_def["tools"], embedder=embedder)
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
        react_block = _REACT_SYSTEM.format(tools=tools_list)
        # Prepend the agent's own system_prompt (grounding rules, refusal phrase,
        # citation requirement, etc.) so research-mode answers still follow the
        # configured persona instead of pure ReAct defaults.
        configured = (self._agent.get("system_prompt") or "").strip()
        system = f"{configured}\n\n{react_block}" if configured else react_block
        messages: list[dict] = [{"role": "user", "content": message}]

        for _ in range(5):
            raw = await self._call_llm(messages, system_prompt=system)
            messages.append({"role": "assistant", "content": raw})

            decision = _parse_react_decision(raw)
            if decision is None:
                # Unparseable response — treat as final answer
                return raw

            action = decision.get("action", "answer")
            if action == "answer":
                return decision.get("text", raw)

            # Tool call
            args = decision.get("args", {})
            try:
                nodes = await self._dispatch(action, args, prov)
            except (ToolNotPermittedError, TypeError, ValueError) as exc:
                messages.append(
                    {
                        "role": "user",
                        "content": f"Tool error: {exc}. Try a different tool or fix the args.",
                    }
                )
                continue
            except Exception as exc:
                # Backend/DB errors (e.g. CypherSyntaxError from a malformed
                # cypher_query) — feed back to the agent so it can retry with
                # corrected args rather than crashing the whole request.
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool '{action}' raised an error: {type(exc).__name__}: {exc}. "
                            f"Fix the arguments and try again."
                        ),
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
