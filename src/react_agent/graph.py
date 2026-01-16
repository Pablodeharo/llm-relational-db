from pathlib import Path
from typing import Literal
import json

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.model_manager import ModelManager
from react_agent.backends.base import GenerationConfig
from react_agent.tool_router import ToolRouter

from react_agent.sql_prompts import SQLCODER_PROMPT
from react_agent.utils import detect_action, is_schema_question


# =========================
# CALL MODEL
# =========================
async def call_model(state: State, runtime: Runtime[Context]):
    """LLM principal: razonamiento + redacción."""
    model_manager = ModelManager(config_path=Path(__file__).parent / "config" / "models.yaml")
    backend = await model_manager.get_backend(runtime.context.reasoning_model)

    system_prompt = runtime.context.system_prompt

    conversation_parts = []
    for msg in state.messages:
        if hasattr(msg, "content") and msg.content:
            if isinstance(msg, AIMessage):
                conversation_parts.append(f"Assistant: {msg.content}")
            elif isinstance(msg, ToolMessage):
                conversation_parts.append(f"Tool Result: {msg.content}")
            else:
                conversation_parts.append(f"User: {msg.content}")

    prompt = f"{system_prompt}\n\n" + "\n\n".join(conversation_parts)

    response = await backend.generate(
        prompt=prompt,
        config=GenerationConfig(
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["</s>", "User:", "Tool Result:"]
        )
    )

    return {
        "messages": state.messages + [
            AIMessage(
                content=response.content,
                tool_calls=response.tool_calls or []
            )
        ]
    }


# =========================
# INIT SCHEMA
# =========================
async def init_schema(state: State, runtime: Runtime[Context]):
    """Explora la base de datos si el schema no está cargado."""
    if runtime.context.db_schema is None:
        tool_router = ToolRouter(runtime)
        raw_schema = await tool_router.execute_tool("explore_database")
        # Parsear JSON si devuelve string
        if isinstance(raw_schema, str):
            schema = json.loads(raw_schema)
        else:
            schema = raw_schema
        runtime.context.db_schema = schema

    return {"messages": state.messages}


# =========================
# ANSWER FROM SCHEMA
# =========================
async def answer_from_schema(state: State, runtime: Runtime[Context]):
    """Responde directamente usando el schema en memoria (sin SQL)."""
    schema = runtime.context.db_schema
    question = state.messages[-1].content.lower()

    if not schema:
        return {
            "messages": state.messages + [
                AIMessage(content="Aún no he explorado la base de datos.")
            ]
        }

    # Número de tablas
    if "cuantas tablas" in question or "how many tables" in question:
        return {
            "messages": state.messages + [
                AIMessage(content=f"La base de datos tiene {schema.get('table_count', 0)} tablas.")
            ]
        }

    # Columnas de tabla específica
    for table_name, columns in schema.get("tables", {}).items():
        if table_name.lower() in question:
            cols_str = ", ".join(columns)
            return {
                "messages": state.messages + [
                    AIMessage(content=f"Las columnas de {table_name} son: {cols_str}")
                ]
            }

    # Listar nombres de tablas si se pregunta
    if "nombres de las tablas" in question or "what tables" in question:
        table_names = ", ".join(schema.get("tables", {}).keys())
        return {
            "messages": state.messages + [
                AIMessage(content=f"Las tablas disponibles son: {table_names}")
            ]
        }

    return {
        "messages": state.messages + [
            AIMessage(content="No pude responder eso usando únicamente el schema.")
        ]
    }


# =========================
# GENERATE SQL
# =========================
async def generate_sql(sql_request: str, schema: dict) -> str:
    """Genera SQL SELECT usando SQLCoder y el schema JSON."""
    model_manager = ModelManager(config_path=Path(__file__).parent / "config" / "models.yaml")
    sql_model = await model_manager.get_backend("sqlcoder")

    prompt = SQLCODER_PROMPT.format(
        schema=json.dumps(schema),
        task=sql_request
    )

    response = await sql_model.generate(
        prompt=prompt,
        config=GenerationConfig(
            max_tokens=256,
            temperature=0.1,
            top_p=0.95,
            stop_sequences=[";", "\n\n"]
        )
    )

    return response.content.strip()


# =========================
# RUN SQL
# =========================
async def run_sql(state: State, runtime: Runtime[Context]):
    """Genera SQL con SQLCoder y ejecuta query_postgres."""
    last_msg = state.messages[-1]
    sql_request = last_msg.content

    tool_router = ToolRouter(runtime)

    # Asegurar schema
    if runtime.context.db_schema is None:
        raw_schema = await tool_router.execute_tool("explore_database")
        if isinstance(raw_schema, str):
            schema = json.loads(raw_schema)
        else:
            schema = raw_schema
        runtime.context.db_schema = schema
    else:
        schema = runtime.context.db_schema

    sql = await generate_sql(sql_request, schema)

    result = await tool_router.execute_tool("query_postgres", sql=sql)

    return {
        "messages": state.messages + [
            ToolMessage(
                content=str(result),
                name="query_postgres",
                tool_call_id="sqlcoder"
            )
        ]
    }


# =========================
# ROUTE TOOL
# =========================
async def route_tool(state: State, runtime: Runtime[Context]):
    """Ejecuta herramientas solicitadas explícitamente por el LLM."""
    tool_router = ToolRouter(runtime)
    last_msg = state.messages[-1]

    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {"messages": state.messages}

    tool_messages = []

    for call in last_msg.tool_calls:
        result = await tool_router.execute_tool(
            tool_name=call.name,
            **call.arguments
        )

        if call.name == "explore_database":
            if isinstance(result, str):
                runtime.context.db_schema = json.loads(result)
            else:
                runtime.context.db_schema = result

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=call.id, name=call.name)
        )

    return {"messages": state.messages + tool_messages}


# =========================
# ROUTING FUNCTIONS
# =========================
def route_start(state: State) -> Literal["schema_answer", "call_model"]:
    """Al inicio, si es pregunta sobre schema, ir a schema_answer."""
    last_msg = state.messages[-1]
    if is_schema_question(last_msg.content):
        return "schema_answer"
    return "call_model"


def route_model_output(state: State) -> Literal["__end__", "schema_answer", "tools", "sql"]:
    """Decidir siguiente paso tras la respuesta del LLM."""
    last = state.messages[-1]

    if not isinstance(last, AIMessage):
        return "__end__"

    # Preguntas respondibles por schema
    if is_schema_question(last.content):
        return "schema_answer"

    # Tools explícitos
    if last.tool_calls:
        return "tools"

    # SQL por defecto
    return "sql"


# =========================
# BUILD GRAPH
# =========================
builder = StateGraph(State, input_schema=InputState, context_schema=Context)

builder.add_node("init_schema", init_schema)
builder.add_node("schema_answer", answer_from_schema)
builder.add_node("call_model", call_model)
builder.add_node("sql", run_sql)
builder.add_node("tools", route_tool)

builder.add_edge("__start__", "init_schema")
builder.add_conditional_edges("init_schema", route_start)

builder.add_edge("schema_answer", "__end__")
builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("sql", "call_model")

graph = builder.compile(name="DataExplorer Agent")
