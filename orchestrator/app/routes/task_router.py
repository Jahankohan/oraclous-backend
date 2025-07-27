from fastapi import APIRouter, HTTPException, Request
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolMessage
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.chat_models import init_chat_model
from app.schemas.task_schema import TaskRequest

router = APIRouter()


@router.post("/run")
async def run_task(req: TaskRequest, request: Request):
    registry = request.app.state.registry

    initial_state = {
            "messages": [
                {"role": "system", "content": (
                "You are a task automation agent. Use available tools to complete the user's request "
                "efficiently. Stop when the task is complete or no further tool is needed."
            )},
                {"role": "user", "content": req.task_description}],
            "used_tools": [],
            "tool_counter": 0,
    }

    if registry is None or not registry._tools:
        raise HTTPException(status_code=500, detail="MCP registry not initialized.")

    # Step 1: Search tools based on user prompt
    matches = registry.search(req.task_description, top_k=5)
    if not matches:
        raise HTTPException(status_code=404, detail="No matching tools found.")

    # Step 2: Wrap discovered tools into LangChain @tool
    tool_funcs = [registry._make_wrapper(meta) for meta in matches]

    # Step 3: Create LLM agent bound with tools
    llm = init_chat_model(
        model="gpt-4",
        temperature=0.3
    )

    agent_with_tools = llm.bind_tools(tool_funcs)
    tool_node = ToolNode(tool_funcs)

    # Step 4: LangGraph loop planner
    async def tools_with_counter(state: MessagesState):
        state["tool_counter"] = state.get("tool_counter", 0) + 1
        print(f"Tool call #{state['tool_counter']}")
    
        tool_result = await tool_node.ainvoke(state)

        print("Tool Result:", tool_result)

        return {
            **state,
            **tool_result
        }

    
    async def call_model(state: MessagesState):
        response = await agent_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}
        
    def should_continue(state: MessagesState):
        if state.get("tool_counter", 0) >= 10:
            print("Max tool invocations reached. Stopping.")
            return END
        # Find the latest ToolMessage if present
        last_tool_msg = next(
            (msg for msg in reversed(state["messages"])
            if isinstance(msg, ToolMessage) and getattr(msg, "status", None) == "error"),
            None
        )
        if last_tool_msg:
            print("Detected tool error message. Stopping.")
            return "reflect"
        return "tools" if tools_condition(state) == "tools" else "reflect"
    
    async def reflect_on_completion(state: MessagesState):
        llm = init_chat_model(model="gpt-4", temperature=0.5)
        messages = state["messages"]
        final_reflection = await llm.ainvoke([
            {"role": "system", "content": (
                "You are a meta-cognition assistant. Reflect on whether the task described was achievable with "
                "the tools used. Suggest if any other tools, inputs, or clarifications would improve results."
            )},
            *messages
        ])
        return {"messages": messages + [final_reflection]}

    # Step 5: Define and compile LangGraph
    graph = StateGraph(MessagesState)


    # Core steps
    graph.add_node("agent", call_model)
    graph.add_node("tools", tools_with_counter)
    graph.add_node("reflect", reflect_on_completion)

    # Edges
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "reflect": END
    })
    graph.add_edge("tools", "agent")
    graph.add_edge("reflect", END)

    workflow = graph.compile(debug=True)

    # Step 6: Execute workflow
    result = await workflow.ainvoke(initial_state)

    ai_messages = [m.content for m in result["messages"] if isinstance(m, AIMessage)]

    return {
        "result": ai_messages[-2] if len(ai_messages) >= 2 else "",
        "reflection": ai_messages[-1] if ai_messages else "No reflection."
    }
