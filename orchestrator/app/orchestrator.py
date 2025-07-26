from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import os
import httpx

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Oraclous Orchestrator")

# Load MCP catalog on startup
CATALOG_PATH = os.getenv("MCP_CATALOG_PATH", "./catalog/mcp-index.json")

class MCPMetadata(BaseModel):
    name: str
    endpoint: str
    inputs: List[str]
    outputs: List[str]

class OrchestratorRequest(BaseModel):
    task_description: str

mcp_catalog: Dict[str, MCPMetadata] = {}

@app.on_event("startup")
async def load_catalog():
    global mcp_catalog
    with open(CATALOG_PATH, "r") as f:
        raw = json.load(f)
        for item in raw:
            mcp_catalog[item["name"]] = MCPMetadata(**item)
    print("Loaded MCP Catalog:", list(mcp_catalog.keys()))

@app.get("/mcp")
async def list_mcp():
    return list(mcp_catalog.values())

# Step 1: Define tool functions with decorator
@tool
async def github_ingest(repo_url: str) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(mcp_catalog["github-mcp"].endpoint, json={"repo_url": repo_url})
    return resp.json()["comments"]

@tool
async def qa_generator(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(mcp_catalog["qa-generator-mcp"].endpoint, json={"documents": documents})
    return resp.json()["qa_pairs"]

@tool
async def postgres_writer(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    async with httpx.AsyncClient() as client:
        resp = await client.post(mcp_catalog["postgres-writer-mcp"].endpoint, json={"records": records})
    return resp.json()


@app.post("/run-task")
async def run_task(req: OrchestratorRequest):
    # Initialize LLM and bind tools
    llm = init_chat_model(model="gpt-4", temperature=0.3)
    tools = [github_ingest, qa_generator, postgres_writer]
    model_with_tools = llm.bind_tools(tools)

    tool_node = ToolNode(tools)
    
    def should_continue(state: MessagesState):
        return "tools" if tools_condition(state) == "tools" else END

    async def call_model(state: MessagesState):
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Build workflow graph
    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, ["tools", END])
    graph.add_edge("tools", "agent")
    compiled = graph.compile()

    result = compiled.invoke({"messages": [{"role": "user", "content": req.task_description}]})

    return {"result": result["messages"][-1].content}