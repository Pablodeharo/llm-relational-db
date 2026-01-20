
"""
tools.py

Generic database exploration tools using SQLAlchemy SessionLocal.

"""

from sqlalchemy import text
from langchain_core.tools import tool
from react_agent.database import SessionLocal
from typing import Annotated
import json

def _is_safe_query(sql: str) -> bool:
    """Prevent destructive SQL queries."""
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "grant", "revoke"]
    sql_lower = sql.lower().strip()
    return (sql_lower.startswith("select") or sql_lower.startswith("with")) and \
           not any(word in sql_lower for word in forbidden)

@tool
def explore_database() -> str:
    """
    Discover the database schema (tables, columns, and relationships).
    Returns a compact JSON suitable for reasoning by the agent.
    """
    db = SessionLocal()
    try:
        tables_dict = {}
        relationships = []

        # Obtener tablas
        tables_result = db.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """))

        tables = [row[0] for row in tables_result]

        # Obtener columnas por tabla
        for table_name in tables:
            columns_result = db.execute(text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position;
            """), {"table_name": table_name})

            tables_dict[table_name] = [row[0] for row in columns_result]

        # Obtener relaciones (foreign keys)
        fk_result = db.execute(text("""
            SELECT
                tc.table_name AS from_table,
                kcu.column_name AS from_column,
                ccu.table_name AS to_table,
                ccu.column_name AS to_column,
                tc.constraint_name
            FROM information_schema.table_constraints AS tc
            JOIN information_schema.key_column_usage AS kcu
                ON tc.constraint_name = kcu.constraint_name
                AND tc.table_schema = kcu.table_schema
            JOIN information_schema.constraint_column_usage AS ccu
                ON ccu.constraint_name = tc.constraint_name
                AND ccu.table_schema = tc.table_schema
            WHERE tc.constraint_type = 'FOREIGN KEY'
                AND tc.table_schema = 'public'
            ORDER BY tc.table_name, kcu.column_name;
        """))

        for row in fk_result:
            relationships.append({
                "from_table": row[0],
                "from_column": row[1],
                "to_table": row[2],
                "to_column": row[3],
                "constraint_name": row[4]
            })

        schema = {
            "table_count": len(tables_dict),
            "tables": tables_dict,
            "relationships": relationships
        }

        return json.dumps(schema, indent=2)

    finally:
        db.close()

@tool
def query_postgres(sql: str) -> str:
    """
    Execute a read-only SQL query against PostgreSQL.
    Only SELECT statements are allowed.
    
    Args:
        sql: The SQL query (must be SELECT only)
    
    Returns:
        Query results formatted as a readable table
    """
    if not _is_safe_query(sql):
        return "❌ Query rejected: only SELECT statements are allowed."
    
    db = SessionLocal()
    try:
        result = db.execute(text(sql))
        rows = result.fetchall()
        columns = result.keys()
        
        if not rows:
            return "✅ Query executed successfully but returned 0 rows."
        
        # Format results
        result_lines = [
            f"✅ Query returned {len(rows)} row(s):\n",
            "="*70
        ]
        
        # Limit display to 50 rows
        display_rows = rows[:50]
        
        for i, row in enumerate(display_rows, 1):
            result_lines.append(f"\nRow {i}:")
            for col_name, value in zip(columns, row):
                # Format value
                if value is None:
                    display_val = "NULL"
                elif isinstance(value, (int, float)):
                    display_val = f"{value:,}" if isinstance(value, int) else f"{value:.4f}"
                else:
                    display_val = str(value)[:200]
                
                result_lines.append(f"  {col_name:25} = {display_val}")
        
        if len(rows) > 50:
            result_lines.append(f"\n... and {len(rows) - 50} more rows (showing first 50)")
        
        result_lines.append("\n" + "="*70)
        return "\n".join(result_lines)
    
    except Exception as e:
        return f"❌ Query error: {e}\n\nQuery attempted:\n{sql}"
    
    finally:
        db.close()

@tool
def suggest_interesting_queries() -> str:
    """
    Automatically suggest interesting analyses based on the actual database structure.
    Scans tables and proposes relevant questions the user might want to explore.
    
    Use this when:
    - User asks "What can I analyze?"
    - User seems stuck or unsure what to explore
    - After showing schema to spark ideas
    
    Returns specific, actionable query suggestions tailored to this database.
    """
    db = SessionLocal()
    try:
        suggestions = ["💡 Based on your database structure, here are some interesting analyses:\n"]
        
        # Get all tables
        tables_query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        tables_result = db.execute(tables_query)
        tables = [row[0] for row in tables_result]
        
        for table_name in tables[:3]:  # Analyze top 3 tables
            suggestions.append(f"\n📋 For table '{table_name}':")
            
            # Get columns by type
            columns_query = text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position;
            """)
            columns_result = db.execute(columns_query, {"table_name": table_name})
            columns = columns_result.fetchall()
            
            numeric_cols = [c[0] for c in columns 
                          if c[1] in ['integer', 'numeric', 'real', 'double precision', 'bigint', 'smallint']]
            
            text_cols = [c[0] for c in columns 
                        if c[1] in ['character varying', 'text', 'varchar', 'char']]
            
            date_cols = [c[0] for c in columns 
                        if 'timestamp' in c[1] or 'date' in c[1]]
            
            # Numeric suggestions
            if numeric_cols:
                suggestions.append(f"  📊 Numeric analysis:")
                for col in numeric_cols[:2]:
                    suggestions.append(f"     • Distribution and outliers in '{col}'")
                    suggestions.append(f"     • Average/median '{col}' by category")
            
            # Categorical suggestions  
            if text_cols:
                suggestions.append(f"  🏷️  Categorical analysis:")
                for col in text_cols[:2]:
                    suggestions.append(f"     • Count and frequency of '{col}' values")
                    suggestions.append(f"     • Most/least common '{col}'")
            
            # Time-based suggestions
            if date_cols:
                suggestions.append(f"  📅 Time-series analysis:")
                for col in date_cols[:1]:
                    suggestions.append(f"     • Trends over time using '{col}'")
                    suggestions.append(f"     • Patterns by day/month/year")
            
            # Relationship suggestions
            if len(numeric_cols) >= 2:
                suggestions.append(f"  🔗 Relationships:")
                suggestions.append(f"     • Correlation between '{numeric_cols[0]}' and '{numeric_cols[1]}'")
            
            if len(text_cols) >= 1 and len(numeric_cols) >= 1:
                suggestions.append(f"  📈 Segmentation:")
                suggestions.append(f"     • Compare '{numeric_cols[0]}' across different '{text_cols[0]}'")
        
        suggestions.extend([
            "\n" + "="*60,
            "💬 Just ask me any of these in plain language!",
            "   Example: 'Show me the distribution of [column]'",
            "           'What are the most common [category]?'",
            "           'Analyze trends over time'"
        ])
        
        return "\n".join(suggestions)
    
    except Exception as e:
        return f"❌ Error generating suggestions: {str(e)}"
    
    finally:
        db.close()


# REQUIRED
TOOLS = [
    explore_database,
    query_postgres,
    suggest_interesting_queries
]

