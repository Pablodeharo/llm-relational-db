from pathlib import Path
from typing import Literal
import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import State, InputState, IntentMemory
from react_agent.model_manager import ModelManager
from react_agent.backends.base import GenerationConfig
from react_agent.tool_router import ToolRouter
from react_agent.sql_prompts import SQLCODER_PROMPT
from react_agent.schema_builder import build_schema_memory
from react_agent.utils import get_message_text

async def init_schema(state: State, runtime: Runtime[Context]):
    """
    Load database schema once and store it in structured SchemaMemory.
    """
    if state.schema is None:
        tool_router = ToolRouter(runtime)
        raw_schema = await tool_router.execute_tool("explore_database")
        state.schema = build_schema_memory(raw_schema)

    return {}


def infer_intent(state: State):
    """
    Infer user intent deterministically from the last user message.
    NO model calls. NO tools.
    """
    text = get_message_text(state.messages[-1]).lower()

    schema_keywords = [
        "tabla", "tablas", "columnas", "columns",
        "schema", "estructura",
        "how many tables", "cuantas tablas",
        "lista de tablas", "what tables",
    ]

    sql_keywords = [
        "count", "sum", "avg", "average", "total",
        "where", "group by", "order by",
        "cuantos", "cuántos", "mayor que", "menor que",
    ]

    analysis_keywords = [
        "explain", "describe", "explica",
        "analiza", "analysis", "por qué", "why",
    ]

    if any(k in text for k in schema_keywords):
        state.intent = IntentMemory(type="schema", confidence=0.9)
    elif any(k in text for k in sql_keywords):
        state.intent = IntentMemory(type="sql", confidence=0.9)
    elif any(k in text for k in analysis_keywords):
        state.intent = IntentMemory(type="analysis", confidence=0.7)
    else:
        state.intent = IntentMemory(type="unknown", confidence=0.3)

    return {}


async def answer_from_schema(state: State, runtime: Runtime[Context]):
    """
    Answer questions that can be resolved using SchemaMemory only.
    """
    schema = state.schema
    question = get_message_text(state.messages[-1]).lower()

    if not schema or not schema.loaded:
        return {
            "messages": state.messages + [
                AIMessage(content="Database schema has not been loaded yet.")
            ]
        }

    if "cuantas tablas" in question or "how many tables" in question:
        return {
            "messages": state.messages + [
                AIMessage(
                    content=f"The database contains {schema.table_count} public tables."
                )
            ]
        }

    for table_name, table in schema.tables.items():
        if table_name.lower() in question:
            cols = ", ".join(table.columns)
            return {
                "messages": state.messages + [
                    AIMessage(
                        content=f"The table `{table_name}` has the following columns: {cols}"
                    )
                ]
            }

    table_names = ", ".join(schema.tables.keys())
    return {
        "messages": state.messages + [
            AIMessage(
                content=f"Available tables are: {table_names}"
            )
        ]
    }

async def call_model(state: State, runtime: Runtime[Context]):
    """
    General reasoning / explanation node.
    """
    model_manager = ModelManager(
        config_path=Path(__file__).parent / "config" / "models.yaml"
    )
    backend = await model_manager.get_backend(runtime.context.reasoning_model)

    conversation = []
    for msg in state.messages:
        role = "User"
        if isinstance(msg, AIMessage):
            role = "Assistant"
        elif isinstance(msg, ToolMessage):
            role = "Tool Result"
        conversation.append(f"{role}: {msg.content}")

    prompt = runtime.context.system_prompt + "\n\n" + "\n".join(conversation)

    response = await backend.generate(
        prompt=prompt,
        config=GenerationConfig(max_tokens=512, temperature=0.7)
    )

    return {
        "messages": state.messages + [AIMessage(content=response.content)]
    }


async def generate_sql(sql_request: str, schema_memory) -> str:
    """
    Generate SELECT-only SQL using SQLCoder and compact schema.
    """
    model_manager = ModelManager(
        config_path=Path(__file__).parent / "config" / "models.yaml"
    )
    sql_model = await model_manager.get_backend("sqlcoder")

    schema_json = {
        "tables": {
            name: table.columns
            for name, table in schema_memory.tables.items()
        }
    }

    prompt = SQLCODER_PROMPT.format(
        schema=json.dumps(schema_json),
        task=sql_request
    )

    response = await sql_model.generate(
        prompt=prompt,
        config=GenerationConfig(max_tokens=256, temperature=0.1)
    )

    return response.content.strip()


async def run_sql(state: State, runtime: Runtime[Context]):
    """
    Generate SQL via SQLCoder and execute it.
    """
    tool_router = ToolRouter(runtime)
    sql = await generate_sql(
        get_message_text(state.messages[-1]),
        state.schema
    )

    result = await tool_router.execute_tool(
        tool_name="query_postgres",
        sql=sql
    )

    return {
        "messages": state.messages + [
            ToolMessage(
                name="query_postgres",
                content=str(result),
                tool_call_id="sql"
            )
        ]
    }

def route_by_intent(state: State) -> Literal[
    "schema_answer", "sql", "call_model"
]:
    intent = state.intent.type

    if intent == "schema":
        return "schema_answer"
    if intent == "sql":
        return "sql"
    return "call_model"

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("init_schema", init_schema)
builder.add_node("infer_intent", infer_intent)
builder.add_node("schema_answer", answer_from_schema)
builder.add_node("call_model", call_model)
builder.add_node("sql", run_sql)

builder.add_edge("__start__", "init_schema")
builder.add_edge("init_schema", "infer_intent")
builder.add_conditional_edges("infer_intent", route_by_intent)

builder.add_edge("schema_answer", "__end__")
builder.add_edge("sql", "call_model")
builder.add_edge("call_model", "__end__")

graph = builder.compile(name="DataExplorerAgent")