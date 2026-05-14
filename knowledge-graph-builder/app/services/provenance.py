"""Provenance tracking for agent execution (TASK-034 / STORY-020, STORY-5).

ProvenanceCollector is passed through all reasoning mode methods and records
every tool call, node returned, and query executed during a single agent run.
"""

from uuid import uuid4

from app.schemas.agent_schemas import NodeResult, ProvenancePayload


class ProvenanceCollector:
    """Accumulates execution evidence for a single agent chat turn."""

    def __init__(self) -> None:
        self._nodes: list[dict] = []
        self._queries: list[str] = []
        self._tools: list[str] = []
        # One entry per dispatched tool call. Each is {"id", "name", "node_count"}.
        # ``id`` mirrors the LLM-provided tool_call_id when the native tool-use
        # protocol is in use; falls back to a synthetic id otherwise.
        self._tool_calls: list[dict] = []
        self.nodes_used_in_response: list[str] = []

    def record_tool(
        self,
        tool_name: str,
        nodes: list[NodeResult],
        tool_call_id: str | None = None,
    ) -> None:
        self._tools.append(tool_name)
        self._tool_calls.append(
            {
                "id": tool_call_id or f"synthetic_{uuid4().hex[:12]}",
                "name": tool_name,
                "node_count": len(nodes),
            }
        )
        for n in nodes:
            self._nodes.append({"id": n.id, "label": n.label})

    def record_query(self, cypher: str) -> None:
        self._queries.append(cypher)

    def to_payload(self) -> ProvenancePayload:
        return ProvenancePayload(
            nodes=list(self._nodes),
            edges=[],
            queries_executed=list(self._queries),
            nodes_used_in_response=list(self.nodes_used_in_response),
            total_nodes_traversed=len(self._nodes),
            reasoning_steps=len(self._tools),
            tools_called=list(self._tools),
            tool_calls=list(self._tool_calls),
        )
