from typing import List, Dict, Any
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import Runnable
from langchain_core.messages import BaseMessage


class BlockConfig:
    def __init__(self, block_id: str, block_type: str, config: dict):
        self.id = block_id
        self.type = block_type
        self.config = config


class WorkflowCompiler:
    def __init__(self, blocks: List[Dict[str, Any]], edges: List[Dict[str, str]], tool_registry):
        self.blocks = [BlockConfig(b['id'], b['type'], b.get('config', {})) for b in blocks]
        self.edges = edges
        self.tool_registry = tool_registry
        self.block_map: Dict[str, Runnable] = {}
        self.graph = StateGraph(MessagesState)

    def compile(self) -> Runnable:
        self._initialize_blocks()
        self._add_nodes_and_edges()
        return self.graph.compile()

    def _initialize_blocks(self):
        for block in self.blocks:
            if block.type == "agent":
                self.block_map[block.id] = self._make_agent_node(block)
            elif block.type in self.tool_registry:
                self.block_map[block.id] = self._make_tool_node(block)
            else:
                raise ValueError(f"Unsupported block type: {block.type}")

    def _make_agent_node(self, block: BlockConfig) -> Runnable:
        llm = block.config.get("llm")
        tools = block.config.get("tools", [])
        bound_agent = llm.bind_tools(tools)

        async def run_agent(state: MessagesState):
            response = await bound_agent.ainvoke(state["messages"])
            return {"messages": [response]}

        return run_agent

    def _make_tool_node(self, block: BlockConfig) -> Runnable:
        tool = self.tool_registry[block.type]

        async def run_tool(state: MessagesState):
            result = await tool.ainvoke(state)
            return {"messages": result.get("messages", [])}

        return run_tool

    def _add_nodes_and_edges(self):
        for block_id, node in self.block_map.items():
            self.graph.add_node(block_id, node)

        # Automatically connect entry and end
        entry = self._find_entry_block()
        self.graph.set_entry_point(entry)

        for edge in self.edges:
            source = edge["from"]
            target = edge["to"]
            if target.lower() == "end":
                self.graph.add_edge(source, END)
            else:
                self.graph.add_edge(source, target)

    def _find_entry_block(self) -> str:
        targets = {e['to'] for e in self.edges}
        for block in self.blocks:
            if block.id not in targets:
                return block.id
        raise ValueError("No entry block found")
