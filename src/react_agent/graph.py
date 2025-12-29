"""
graph.py

Defines a data analyst ReAct agent that combines:
- A reasoning LLM (call_model)
- Database exploration tools routed via ToolRouter
- Cyclical execution flow between model and tools
- Fully compatible with LangGraph Platform
"""

from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.model_manager import ModelManager
from react_agent.context import ToolRouter


# -----------------------------
# Node: call_model
# -----------------------------
async def call_model(state: State, runtime: Runtime[Context]):
    """
    Executes the reasoning model to decide the next action.
    This model can generate tool calls that will be handled by ToolRouter.
    """
    # Use the reasoning model explicitly
    model_name = runtime.context.reasoning_model

    # Load the model
    model_manager = ModelManager(
        config_path=Path(__file__).parent / "config" / "models.yaml"
    )
    model = await model_manager.get_backend(model_name)

    # Construct the prompt from system prompt + conversation history
    system_prompt = runtime.context.system_prompt
    messages = [system_prompt] + [
        msg.content for msg in state.messages if hasattr(msg, "content")
    ]
    prompt = "\n\n".join(messages)

    # Generate model response
    output = model(
        prompt,
        max_tokens=512,
        stop=["</s>"]
    )

    response_text = output["choices"][0]["text"]

    return {
        "messages": state.messages + [
            AIMessage(content=response_text)
        ]
    }


# -----------------------------
# Node: tools (ToolRouter)
# -----------------------------
async def route_tool(state: State, runtime: Runtime[Context]):
    """
    Executes tool calls from the last AI message using the correct model.
    ToolRouter selects the model for each tool based on context.tool_models.
    """
    tool_router = ToolRouter(runtime)
    last_msg = state.messages[-1]

    if not last_msg.tool_calls:
        # No tools requested, return messages unchanged
        return {"messages": state.messages}

    results = []

    # Execute each tool call using ToolRouter
    for tool_call in last_msg.tool_calls:
        tool_name = tool_call.name
        tool_args = tool_call.args
        output_text = await tool_router.execute_tool(tool_name, **tool_args)
        results.append(output_text)

    # Append tool outputs to conversation
    return {"messages": state.messages + [AIMessage(content="\n".join(results))]}


# -----------------------------
# Conditional routing function
# -----------------------------
def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """
    Determines the next node based on the last AI message.

    - If the AI did not request any tools → end the flow (__end__)
    - If the AI requested tools → route to the tools node
    """
    last_message = state.messages[-1]

    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, got {type(last_message).__name__}"
        )

    return "tools" if last_message.tool_calls else "__end__"


# -----------------------------
# Graph construction
# -----------------------------
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Main node: reasoning LLM
builder.add_node("call_model", call_model)

# Tools node: routes tool calls to the correct model
builder.add_node("tools", route_tool)

# Entry point: start with the reasoning model
builder.add_edge("__start__", "call_model")

# Conditional routing after reasoning model
builder.add_conditional_edges("call_model", route_model_output)

# After tools execute, return to reasoning model for interpretation
builder.add_edge("tools", "call_model")

# Compile the graph
graph = builder.compile(name="DataExplorer Agent")
