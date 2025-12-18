from sqlalchemy import text
from langchain_core.tools import tool
from react_agent.database import SessionLocal


def _is_safe_query(sql: str) -> bool:
    """Prevent destructive SQL queries."""
    forbidden = ["insert", "update", "delete", "drop", "alter"]
    sql_lower = sql.lower().strip()
    return sql_lower.startswith("select") and not any(word in sql_lower for word in forbidden)


@tool
def query_postgres(sql: str) -> str:
    """
    Execute a read-only SQL query against PostgreSQL.
    Only SELECT statements are allowed.
    """
    if not _is_safe_query(sql):
        return "Query rejected: only SELECT statements are allowed."

    db = SessionLocal()
    try:
        result = db.execute(text(sql))
        rows = result.fetchall()
        return str(rows)
    except Exception as e:
        return f"Query error: {e}"
    finally:
        db.close()


# REQUIRED: LangGraph imports this symbol
TOOLS = [query_postgres]