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
    if state.db_schema is None:
        tool_router = ToolRouter(runtime)
        raw_schema = await tool_router.execute_tool("explore_database")
        schema_memory = build_schema_memory(raw_schema)
        
        return {"db_schema": schema_memory}
    
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
        intent_obj = IntentMemory(type="schema")
    elif any(k in text for k in sql_keywords):
        intent_obj = IntentMemory(type="sql")
    elif any(k in text for k in analysis_keywords):
        intent_obj = IntentMemory(type="analysis",)
    else:
        intent_obj = IntentMemory(type="unknown")

    return {"intent": intent_obj}


async def answer_from_schema(state: State, runtime: Runtime[Context]):
    """
    Answer questions that can be resolved using SchemaMemory only.
    """
    schema = state.db_schema
    question = get_message_text(state.messages[-1]).lower()

    if schema is None or not schema.loaded:
        return {
            "messages": [
                AIMessage(content="Database schema has not been loaded yet.")
            ]
        }

    if "cuantas tablas" in question or "how many tables" in question:
        return {
            "messages": [
                AIMessage(
                    content=f"La base de datos contiene {schema.table_count} tablas públicas."
                )
            ]
        }

    for table_name, table in schema.tables.items():
        if table_name.lower() in question:
            cols = ", ".join(table.columns)
            return {
                "messages": [
                    AIMessage(
                        content=f"La tabla `{table_name}` tiene las siguientes columnas: {cols}"
                    )
                ]
            }
    # Fallback all tables
    table_names = ", ".join(schema.tables.keys())
    return {
        "messages": [
            AIMessage(
                content=f"Las tablas disponibles son: {table_names}"
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
        "messages": [AIMessage(content=response.content)]
    }

async def generate_sql(sql_request: str, schema_memory) -> str:
    """
    Generate SELECT-only SQL using SQLCoder and compact schema with relationships.
    """
    model_manager = ModelManager(
        config_path=Path(__file__).parent / "config" / "models.yaml"
    )
    sql_model = await model_manager.get_backend("sqlcoder")

    # Incluir tablas
    schema_json = {
        "tables": {
            name: table.columns
            for name, table in schema_memory.tables.items()
        }
    }
    
    # Incluir relaciones
    relationships_json = [
        {
            "from": f"{rel.from_table}.{rel.from_column}",
            "to": f"{rel.to_table}.{rel.to_column}"
        }
        for rel in schema_memory.relationships
    ]

    prompt = SQLCODER_PROMPT.format(
        schema=json.dumps(schema_json, indent=2),
        relationships=json.dumps(relationships_json, indent=2),
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
        state.db_schema
    )

    result = await tool_router.execute_tool(
        tool_name="query_postgres",
        sql=sql
    )

    return {
        "messages": [
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

    if state.intent is None:
        return "call_model"
    
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