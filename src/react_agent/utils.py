from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from react_agent.models import get_agent_model_async

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message."""
    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    provider, model = fully_specified_name.split("/", maxsplit=1)
    return init_chat_model(model, model_provider=provider)

import re

def detect_action(text: str) -> str:
    t = text.lower()

    # 1️⃣ Respuesta directa desde schema
    if any(x in t for x in [
        "cuantas tablas",
        "how many tables",
        "lista de tablas",
        "what tables",
        "nombres de las tablas",
    ]):
        return "schema_answer"

    # 2️⃣ Explorar schema
    if "explore" in t or "schema" in t:
        return "explore_schema"

    # 3️⃣ SQL explícito
    if "select" in t or "consulta" in t or "custom_sql" in t:
        return "custom_sql"

    return "simple_tool"

def is_schema_question(text: str) -> bool:
    """
    Detecta preguntas que pueden responderse SOLO con el schema,
    sin ejecutar SQL.
    """
    text = text.lower()

    keywords = [
        "cuantas tablas",
        "how many tables",
        "nombres de las tablas",
        "what tables",
        "lista de tablas",
        "columnas",
        "columns",
        "schema",
        "estructura"
    ]

    return any(k in text for k in keywords)