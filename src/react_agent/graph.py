from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

from react_agent.context import Context
from react_agent.state import InputState, State
from react_agent.model_manager import ModelManager
from react_agent.backends.base import GenerationConfig
from react_agent.tool_router import ToolRouter
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver










async def call_model(state: State, runtime: Runtime[Context]):
    """
    Llama al modelo LLM y genera una respuesta.
    
    Args:
        state: Estado actual con historial de mensajes
        runtime: Runtime con contexto (model name, prompts, etc)
        
    Returns:
        Estado actualizado con AIMessage del modelo
    """
    model_name = runtime.context.reasoning_model
    
    # Cargar backend del modelo
    model_manager = ModelManager(
        config_path=Path(__file__).parent / "config" / "models.yaml"
    )
    backend = await model_manager.get_backend(model_name)
    
    # Construir prompt desde el system prompt y mensajes
    system_prompt = runtime.context.system_prompt
    
    # Extraer contenido de todos los mensajes
    conversation_parts = []
    for msg in state.messages:
        if hasattr(msg, "content") and msg.content:
            # Identificar tipo de mensaje para formato
            if isinstance(msg, AIMessage):
                conversation_parts.append(f"Assistant: {msg.content}")
            elif isinstance(msg, ToolMessage):
                conversation_parts.append(f"Tool Result: {msg.content}")
            else:  # HumanMessage
                conversation_parts.append(f"User: {msg.content}")
    
    # Construir prompt completo
    prompt = f"{system_prompt}\n\n" + "\n\n".join(conversation_parts)
    
    # Generar respuesta
    response = await backend.generate(
        prompt=prompt,
        config=GenerationConfig(
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            stop_sequences=["</s>", "User:", "Tool Result:"]
        )
    )
    
    # Crear AIMessage con contenido y posibles tool_calls
    ai_message = AIMessage(
        content=response.content,
        tool_calls=response.tool_calls if response.tool_calls else []
    )
    
    return {
        "messages": state.messages + [ai_message]
    }


async def route_tool(state: State, runtime: Runtime[Context]):
    """
    Ejecuta las herramientas llamadas por el modelo.
    
    Args:
        state: Estado con mensajes (último debe ser AIMessage con tool_calls)
        runtime: Runtime con contexto
        
    Returns:
        Estado actualizado con ToolMessage por cada herramienta ejecutada
    """
    tool_router = ToolRouter(runtime)
    last_msg = state.messages[-1]
    
    # Verificar que es AIMessage con tool_calls
    if not isinstance(last_msg, AIMessage):
        print("⚠️ route_tool called but last message is not AIMessage")
        return {"messages": state.messages}
    
    if not last_msg.tool_calls:
        print("⚠️ route_tool called but no tool_calls found")
        return {"messages": state.messages}
    
    # Ejecutar cada tool call
    tool_messages = []
    
    for call in last_msg.tool_calls:
        try:
            print(f"🔧 Executing tool: {call.name}")
            
            # Ejecutar herramienta
            result = await tool_router.execute_tool(
                tool_name=call.name,
                **call.arguments  # ToolCall.arguments es un Dict
            )
            
            # Crear ToolMessage con el resultado
            tool_msg = ToolMessage(
                content=str(result),
                tool_call_id=call.id,
                name=call.name
            )
            tool_messages.append(tool_msg)
            
            print(f"✓ Tool '{call.name}' executed successfully")
            
        except Exception as e:
            print(f"❌ Error executing tool '{call.name}': {e}")
            
            # Crear ToolMessage con el error
            error_msg = ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=call.id,
                name=call.name
            )
            tool_messages.append(error_msg)
    
    return {
        "messages": state.messages + tool_messages
    }


def route_model_output(state: State) -> Literal["__end__", "tools"]:
    """
    Decide el siguiente paso basándose en la salida del modelo.
    
    Si el modelo llamó herramientas → "tools"
    Si no → "__end__" (terminar)
    
    Args:
        state: Estado actual con mensajes
        
    Returns:
        Nombre del siguiente nodo ("tools" o "__end__")
    """
    messages = state.messages
    
    if not messages:
        print("⚠️ No messages in state, ending")
        return "__end__"
    
    last = messages[-1]
    
    # Solo AIMessage puede tener tool_calls
    if not isinstance(last, AIMessage):
        print(f"ℹ️ Last message is {type(last).__name__}, not AIMessage, ending")
        return "__end__"
    
    # Verificar si tiene tool_calls
    has_tool_calls = (
        hasattr(last, "tool_calls") and
        last.tool_calls is not None and
        len(last.tool_calls) > 0
    )
    
    if has_tool_calls:
        print(f"🔀 Model called {len(last.tool_calls)} tool(s), routing to tools")
        return "tools"
    else:
        print("🏁 No tool calls, ending conversation")
        return "__end__"


# ═══════════════════════════════════════════════════════════════
# Construcción del grafo
# ═══════════════════════════════════════════════════════════════

builder = StateGraph(State, input_schema=InputState, context_schema=Context)

# Nodos
builder.add_node("call_model", call_model)
builder.add_node("tools", route_tool)

# Edges
builder.add_edge("__start__", "call_model")
builder.add_conditional_edges("call_model", route_model_output)
builder.add_edge("tools", "call_model")  # Después de tools, vuelve al modelo

# Compilar
graph = builder.compile(name="DataExplorer Agent")