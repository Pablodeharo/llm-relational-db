# graph.py

from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.model_manager import ModelManager
from react_agent.tool_router import ToolRouter


async def call_model(state: State, runtime: Runtime[Context]):
    model_name = runtime.context.reasoning_model

    model_manager = ModelManager(
        config_path=Path(__file__).parent / "config" / "models.yaml"
    )
    model = await model_manager.get_backend(model_name)

    system_prompt = runtime.context.system_prompt
    messages = [system_prompt] + [
        msg.content for msg in state.messages if hasattr(msg, "content")
    ]

    prompt = "\n\n".join(messages)

    output = model(
        prompt,
        max_tokens=512,
        stop=["</s>"]
    )

    return {
        "messages": state.messages + [AIMessage(content=output["choices"][0]["text"])]
    }


async def route_tool(state: State, runtime: Runtime[Context]):
    tool_router = ToolRouter(runtime)
    last_msg = state.messages[-1]

    if not last_msg.tool_calls:
        return {"messages": state.messages}

    outputs = []

    for call in last_msg.tool_calls:
        result = await tool_router.execute_tool(
            tool_name=call.name,
            **call.args
        )
        outputs.append(result)

    return {
        "messages": state.messages + [AIMessage(content="\n".join(outputs))]
    }

def route_model_output(state: State) -> Literal["__end__", "tools"]:
    last = state.messages[-1]
    return "tools" if last.tool_calls else "__end__"

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("call_model", call_model)
builder.add_node("tools", route_tool)

builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")

graph = builder.compile(name="DataExplorer Agent")
