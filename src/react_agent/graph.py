"""
graph.py

Defines a custom ReAct agent that combines:
- An LLM for reasoning and conversation (call_model)
- External tools, such as PostgreSQL (query_postgres)
- Cyclical execution flow between model and tools

Fully compatible with LangGraph Platform.
"""

import asyncio
import torch
from transformers import pipeline
from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.tools import TOOLS
from react_agent.utils import load_chat_model
#from langchain.chat_models import ChatOpenAI
from react_agent.models import get_agent_model_async
from react_agent.model_manager import ModelManager

async def call_model(
    state: State, runtime: Runtime[Context]
) -> Dict[str, List[AIMessage]]:

    model_backend = await ModelManager.get_backend(runtime.context.model)

    system_message = runtime.context.system_prompt.format(
        system_time=datetime.now(tz=UTC).isoformat()
    )

    prompt_parts = [system_message] + [m.content for m in state.messages]
    prompt = "\n".join(prompt_parts)

    model_response = await model_backend.generate(
        prompt=prompt,
        tools=TOOLS,
    )

    ai_message = AIMessage(
        id="llm_output",
        content=model_response.content,
        tool_calls=model_response.tool_calls,
    )

    return {"messages": [ai_message]}



# -----------------------------
# Graph construction
# -----------------------------
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Main node: the LLM
builder.add_node(call_model)

# Tools node (PostgreSQL or other external tools)
builder.add_node("tools", ToolNode(TOOLS))

# Entry point of the graph
builder.add_edge("__start__", "call_model")


# -----------------------------
# Conditional routing function
# -----------------------------
def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """
    Determines the next node based on the model output.

    - If the model does not request tools, ends the flow (__end__)
    - If the model wants to use tools, routes to the tools node
    """
    last_message = state.messages[-1]
    if not isinstance(last_message, AIMessage):
        raise ValueError(
            f"Expected AIMessage in output edges, but got {type(last_message).__name__}"
        )

    # No tool calls → finish
    if not last_message.tool_calls:
        return "__end__"

    # Tool calls present → execute tools node
    return "tools"


# -----------------------------
# Configure edges
# -----------------------------
# Conditional edge: decides whether to go to __end__ or tools after call_model
builder.add_conditional_edges(
    "call_model",
    route_model_output,
)

# Normal edge: after tools, return to call_model
builder.add_edge("tools", "call_model")


graph = builder.compile(name="ReAct Agent")
