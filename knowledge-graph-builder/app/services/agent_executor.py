"""Agent execution engine — four reasoning modes (TASK-034 / STORY-020, STORY-5).

AgentExecutor is instantiated per chat request. It loads the agent definition
from Neo4j, resolves the LLM client via the three-level LLMConfig chain
(agent → project → org → env-var fallback), and dispatches to the correct
reasoning mode. Every execution produces a ProvenancePayload that the caller
returns alongside the agent's response.

Reasoning modes
---------------
direct          Single retrieval pass → generate (≤ 2 LLM calls)
research        Native tool-use loop: model emits tool_calls, we dispatch
                them, return tool results, model may call more or answer.
                Up to 5 iterations.
analytical      Same loop as research with a structured-reasoning system
                prompt nudging the model to lay out its analysis.
conversational  Full session history injected; warm-tone system prompt suffix

STORY-5 — Real tool-use protocol
-------------------------------
Previously ``_research`` prompted the model to emit JSON describing tool
calls and parsed the raw text with regex-based parsers. Claude (and most
modern models) are trained on a native tool-use protocol where the SDK
returns structured ``tool_calls`` and accepts ``role: tool`` results.
This module now uses that protocol directly via both the Anthropic
native SDK and the OpenAI-compatible SDK (which OpenRouter and Azure
OpenAI also serve).

Tool schemas live in ``app/services/agent_tool_schemas.py`` and are
formatted per-provider; the executor never hand-writes them inline.
"""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from neo4j import AsyncDriver
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.agent_schemas import AgentChatResponse, NodeResult, PathResult
from app.services.agent_service import AgentService
from app.services.agent_tool_schemas import tool_schemas_for
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


# Max iterations of the tool-use loop. Each iteration = one LLM call +
# zero or more dispatched tools. Cap protects against pathological loops.
_MAX_TOOL_ITERATIONS = 5

# System-prompt suffix that nudges analytical mode to lay out its reasoning
# before the final answer. Combined with the agent's configured system
# prompt and the tool-use protocol.
_ANALYTICAL_SUFFIX = (
    "\n\nReasoning style: think step by step. Use the available tools to "
    "gather evidence, then lay out your analysis explicitly in the final "
    "answer (e.g. 'Step 1:', 'Step 2:', then 'Conclusion:')."
)

_CONVERSATIONAL_SUFFIX = (
    "\n\nYou are having a friendly, ongoing conversation. "
    "Reference earlier turns naturally where relevant."
)


# ── Tool-use protocol types ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _ToolCall:
    """One tool invocation the LLM wants the executor to dispatch."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _LLMResponse:
    """A single LLM completion. Either text-only or includes tool_calls.

    When tool_calls is non-empty, ``text`` may still hold reasoning the
    model emitted alongside the calls. Callers that detect tool_calls
    should dispatch them and feed results back via a new completion.
    """

    text: str
    tool_calls: list[_ToolCall] = field(default_factory=list)
    # Raw assistant-message payload to echo back in the next turn. SDK-
    # specific (Anthropic uses content blocks, OpenAI uses str+tool_calls)
    # so we keep it opaque here and reconstruct in ``_append_assistant``.
    raw_assistant_payload: Any = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _node_to_jsonable(n: NodeResult) -> dict[str, Any]:
    """Project a NodeResult down to a small JSON-safe dict for tool
    results. Drops the embedding (huge) and keeps id, label,
    qualified_name, and a trimmed properties bag including the text/
    content excerpt if present."""
    props = dict(getattr(n, "properties", {}) or {})
    # Don't ship embeddings or huge blobs back to the LLM
    props.pop("embedding", None)
    text = props.get("text") or props.get("content")
    if isinstance(text, str) and len(text) > 1500:
        props["text"] = text[:1500] + "…"
    return {
        "id": n.id,
        "label": n.label,
        "qualified_name": n.qualified_name,
        "properties": props,
    }


def _format_tool_result_json(nodes: list[NodeResult]) -> str:
    """Serialize tool output to JSON the LLM can read.

    Empty-result cases are explicit so the model doesn't second-guess
    whether the tool ran. Truncates to top 20 nodes per call (the agent
    can ask for more via max_results on the next call if needed).
    """
    if not nodes:
        return json.dumps({"results": [], "count": 0})
    capped = nodes[:20]
    return json.dumps(
        {
            "results": [_node_to_jsonable(n) for n in capped],
            "count": len(capped),
            "truncated": len(nodes) > len(capped),
        },
        default=str,
    )


def _extract_cited_nodes(response: str, nodes: list[dict]) -> list[str]:
    """Return IDs of nodes whose id string appears literally in the response."""
    return [n["id"] for n in nodes if n["id"] in response]


def _is_anthropic_client(llm: Any) -> bool:
    """True if ``llm`` is an AsyncAnthropic instance.

    Done lazily so test mocks that don't import the anthropic SDK still
    work — and so the executor doesn't crash if anthropic isn't installed
    in a deployment that only uses OpenRouter / OpenAI.
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        return False
    return isinstance(llm, AsyncAnthropic)


# ── Executor ─────────────────────────────────────────────────────────────────


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
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
    ) -> _LLMResponse:
        """Single LLM completion.

        - ``tools`` is the per-provider-formatted tool list (from
          ``tool_schemas_for``). When provided, the response may contain
          ``tool_calls`` — the executor dispatches each and feeds back
          the result on the next call.
        - When ``tools`` is None, behaves as a pure text completion.

        Two SDK paths share the same _LLMResponse shape.
        """
        sp = system_prompt or self._agent.get(
            "system_prompt", "You are a helpful assistant."
        )

        if _is_anthropic_client(self._llm):
            return await self._call_anthropic(messages, sp, tools)
        return await self._call_openai_compatible(messages, sp, tools)

    async def _call_anthropic(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None,
    ) -> _LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        resp = await self._llm.messages.create(**kwargs)

        text_parts: list[str] = []
        tool_calls: list[_ToolCall] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif btype == "tool_use":
                tool_calls.append(
                    _ToolCall(
                        id=getattr(block, "id", "") or "",
                        name=getattr(block, "name", "") or "",
                        args=dict(getattr(block, "input", {}) or {}),
                    )
                )
        return _LLMResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            raw_assistant_payload=resp.content,
        )

    async def _call_openai_compatible(
        self,
        messages: list[dict],
        system_prompt: str,
        tools: list[dict] | None,
    ) -> _LLMResponse:
        all_msgs = [{"role": "system", "content": system_prompt}] + messages
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": all_msgs,
            "temperature": 0.7,
        }
        if tools:
            kwargs["tools"] = tools
        resp = await self._llm.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        text = msg.content or ""
        tool_calls: list[_ToolCall] = []
        for raw_tc in getattr(msg, "tool_calls", None) or []:
            fn = getattr(raw_tc, "function", None)
            fn_name = getattr(fn, "name", "") if fn else ""
            fn_args_raw = getattr(fn, "arguments", "") if fn else ""
            try:
                fn_args = json.loads(fn_args_raw) if fn_args_raw else {}
            except (json.JSONDecodeError, TypeError):
                fn_args = {}
            tool_calls.append(
                _ToolCall(
                    id=getattr(raw_tc, "id", "") or "",
                    name=fn_name,
                    args=fn_args,
                )
            )
        return _LLMResponse(text=text, tool_calls=tool_calls, raw_assistant_payload=msg)

    async def _call_llm_stream(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield LLM response tokens one at a time (no tool use).

        Used for the final-answer streaming pass after a tool-use loop
        has concluded, and for direct mode where there are no tool calls.
        """
        sp = system_prompt or self._agent.get(
            "system_prompt", "You are a helpful assistant."
        )

        if _is_anthropic_client(self._llm):
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

    # ── Tool-use loop primitives ──────────────────────────────────────────────

    def _provider_format(self) -> str:
        return "anthropic" if _is_anthropic_client(self._llm) else "openai"

    def _append_assistant_turn(self, messages: list[dict], resp: _LLMResponse) -> None:
        """Append the assistant turn to the running messages list in the
        shape the provider expects on the next call."""
        if _is_anthropic_client(self._llm):
            # Anthropic wants the content blocks back verbatim
            messages.append(
                {
                    "role": "assistant",
                    "content": resp.raw_assistant_payload,
                }
            )
        else:
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": resp.text or None,
            }
            if resp.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.args),
                        },
                    }
                    for tc in resp.tool_calls
                ]
            messages.append(entry)

    def _append_tool_result(
        self,
        messages: list[dict],
        tool_call: _ToolCall,
        content: str,
        is_error: bool = False,
    ) -> None:
        """Append a tool-result turn in the provider's expected shape."""
        if _is_anthropic_client(self._llm):
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": content,
                            "is_error": is_error,
                        }
                    ],
                }
            )
        else:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content,
                }
            )

    async def _tool_use_loop(
        self,
        message: str,
        prov: ProvenanceCollector,
        system_prompt: str,
    ) -> tuple[str, list[dict]]:
        """Drive the native tool-use loop.

        Returns ``(final_text, messages)`` once the model produces a turn
        with no tool_calls, or when the iteration cap is hit (in which
        case ``final_text`` is whatever text was last produced — caller
        may want to re-prompt for a definitive answer).
        """
        tools = tool_schemas_for(
            self._agent.get("tools", []) or [], self._provider_format()
        )
        messages: list[dict] = [{"role": "user", "content": message}]

        last_text = ""
        for _ in range(_MAX_TOOL_ITERATIONS):
            resp = await self._call_llm(
                messages, system_prompt=system_prompt, tools=tools or None
            )
            last_text = resp.text

            if not resp.tool_calls:
                return resp.text, messages

            self._append_assistant_turn(messages, resp)

            for tc in resp.tool_calls:
                try:
                    nodes = await self._dispatch(tc.name, tc.args, prov, tc.id)
                    self._append_tool_result(
                        messages, tc, _format_tool_result_json(nodes), is_error=False
                    )
                except ToolNotPermittedError as exc:
                    self._append_tool_result(
                        messages,
                        tc,
                        json.dumps({"error": "tool_not_permitted", "detail": str(exc)}),
                        is_error=True,
                    )
                except (TypeError, ValueError) as exc:
                    self._append_tool_result(
                        messages,
                        tc,
                        json.dumps({"error": "bad_arguments", "detail": str(exc)}),
                        is_error=True,
                    )
                except Exception as exc:
                    # Backend/DB errors (e.g. CypherSyntaxError from a malformed
                    # cypher_query) — feed back so the agent can retry.
                    self._append_tool_result(
                        messages,
                        tc,
                        json.dumps(
                            {
                                "error": type(exc).__name__,
                                "detail": str(exc),
                            }
                        ),
                        is_error=True,
                    )
        return last_text, messages

    # ── run_stream ────────────────────────────────────────────────────────────

    async def run_stream(
        self, message: str, session_id: str | None
    ) -> AsyncGenerator[tuple[str | None, Any], None]:
        """Streaming variant of run(). Yields (token, None) pairs, then
        (None, provenance).

        - direct: token-by-token from a single LLM call after optional graph_search.
        - research: runs the tool-use loop non-streamed; once the model
          produces a turn with no tool_calls, re-runs that final turn
          streamed so the user sees tokens flow.
        - analytical / conversational: non-streamed (fall back to run()).
        """
        prov = ProvenanceCollector()
        mode = self._agent.get("reasoning_mode", "direct")

        if mode == "direct":
            context = ""
            if "graph_search" in self._agent.get("tools", []):
                nodes = await self._dispatch("graph_search", {"query": message}, prov)
                context = _format_tool_result_json(nodes)

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
            return

        if mode == "research":
            system = (self._agent.get("system_prompt") or "").strip() or (
                "You are a helpful research agent."
            )
            # Drive the tool-use loop first to gather evidence + decide a
            # final answer. The model's last (no-tool_calls) turn is what
            # we want to stream — re-issue it with streaming.
            _final_text, messages = await self._tool_use_loop(message, prov, system)

            # The loop already accumulated all messages; the last call
            # produced a no-tool_calls response which we want streamed.
            # Re-call without tools to get a streaming completion of the
            # final answer.
            full_response = ""
            async for token in self._call_llm_stream(messages, system_prompt=system):
                full_response += token
                yield token, None

            prov.nodes_used_in_response = _extract_cited_nodes(
                full_response, prov._nodes
            )
            yield None, prov.to_payload()
            return

        # analytical / conversational / unknown — non-streamed fallback
        chat_resp = await self.run(message, session_id)
        yield chat_resp.response, None
        yield None, chat_resp.provenance

    # ── Tool dispatch ─────────────────────────────────────────────────────────

    async def _dispatch(
        self,
        name: str,
        args: dict,
        prov: ProvenanceCollector,
        tool_call_id: str | None = None,
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

        prov.record_tool(name, nodes, tool_call_id=tool_call_id)
        return nodes

    # ── Reasoning modes ───────────────────────────────────────────────────────

    async def _direct(self, message: str, prov: ProvenanceCollector) -> str:
        context = ""
        if "graph_search" in self._agent.get("tools", []):
            nodes = await self._dispatch("graph_search", {"query": message}, prov)
            context = _format_tool_result_json(nodes)

        user_content = (
            f"Context:\n{context}\n\nQuestion: {message}" if context else message
        )
        resp = await self._call_llm([{"role": "user", "content": user_content}])
        return resp.text

    async def _research(self, message: str, prov: ProvenanceCollector) -> str:
        """Native tool-use loop. The model decides which tools to call
        and when to stop, using each tool result as input to its next
        decision."""
        system = (self._agent.get("system_prompt") or "").strip() or (
            "You are a helpful research agent."
        )
        final_text, _messages = await self._tool_use_loop(message, prov, system)
        return final_text or "(no final answer produced)"

    async def _analytical(self, message: str, prov: ProvenanceCollector) -> str:
        """Tool-use loop with a structured-reasoning system-prompt suffix.

        Same mechanics as research; the suffix nudges the model to lay
        out its analysis explicitly in the final answer.
        """
        configured = (self._agent.get("system_prompt") or "").strip() or (
            "You are a careful analytical assistant."
        )
        system = configured + _ANALYTICAL_SUFFIX
        final_text, _messages = await self._tool_use_loop(message, prov, system)
        return final_text or "(no final answer produced)"

    async def _conversational(
        self, message: str, session_id: str | None, prov: ProvenanceCollector
    ) -> tuple[str, str]:
        if session_id is None:
            session_id = str(uuid4())

        history = list(_sessions.get(session_id, []))

        context = ""
        if "graph_search" in self._agent.get("tools", []):
            nodes = await self._dispatch("graph_search", {"query": message}, prov)
            context = _format_tool_result_json(nodes)

        user_content = (
            f"Context:\n{context}\n\nQuestion: {message}" if context else message
        )

        sp = self._agent.get("system_prompt", "You are a helpful assistant.")
        system = sp + _CONVERSATIONAL_SUFFIX
        messages = history + [{"role": "user", "content": user_content}]
        resp = await self._call_llm(messages, system_prompt=system)

        _sessions[session_id] = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": resp.text},
        ]

        return resp.text, session_id
