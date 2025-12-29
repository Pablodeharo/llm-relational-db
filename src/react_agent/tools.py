
"""
tools.py

Generic database exploration tools using SQLAlchemy SessionLocal.

"""

from sqlalchemy import text
from langchain_core.tools import tool
from react_agent.database import SessionLocal
from typing import Annotated

def _is_safe_query(sql: str) -> bool:
    """Prevent destructive SQL queries."""
    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate", "grant", "revoke"]
    sql_lower = sql.lower().strip()
    return (sql_lower.startswith("select") or sql_lower.startswith("with")) and \
           not any(word in sql_lower for word in forbidden)

@tool
def explore_database() -> str:
    """
    Automatically discover the complete database structure.
    
    Returns:
    - All table names
    - Column names and types for each table
    - Row counts
    - Primary keys
    
    Use this as your FIRST action when the user asks about their data.
    This tool works with ANY PostgreSQL database.
    """
    db = SessionLocal()
    try:
        result_lines = ["Database Schema Explorer\n"]
        
        # Get all tables
        tables_query = text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        tables_result = db.execute(tables_query)
        tables = [row[0] for row in tables_result]
        
        if not tables:
            return "No tables found in the database."
        
        result_lines.append(f"Found {len(tables)} table(s):\n")
        
        total_rows = 0
        
        for table_name in tables:
            # Get row count
            count_query = text(f"SELECT COUNT(*) FROM {table_name};")
            row_count = db.execute(count_query).scalar()
            total_rows += row_count
            
            result_lines.append(f"\n{'='*60}")
            result_lines.append(f"📋 Table: {table_name}")
            result_lines.append(f"   Rows: {row_count:,}")
            
            # Get columns
            columns_query = text("""
                SELECT 
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position;
            """)
            columns_result = db.execute(columns_query, {"table_name": table_name})
            columns = columns_result.fetchall()
            
            result_lines.append(f"   Columns ({len(columns)}):")
            for col in columns:
                col_name, data_type, max_length, is_nullable, col_default = col
                
                col_type = data_type
                if max_length:
                    col_type += f"({max_length})"
                
                nullable = "NULL" if is_nullable == 'YES' else "NOT NULL"
                default = f" DEFAULT {col_default}" if col_default else ""
                
                result_lines.append(f"      • {col_name}: {col_type} {nullable}{default}")
            
            # Get primary key
            pk_query = text("""
                SELECT column_name
                FROM information_schema.key_column_usage
                WHERE table_name = :table_name 
                AND constraint_name LIKE '%_pkey'
                ORDER BY ordinal_position;
            """)
            pk_result = db.execute(pk_query, {"table_name": table_name})
            pkeys = [row[0] for row in pk_result]
            
            if pkeys:
                pk_cols = ", ".join(pkeys)
                result_lines.append(f"   🔑 Primary Key: {pk_cols}")
        
        result_lines.append(f"\n{'='*60}")
        result_lines.append(f"\n📈 Total records in database: {total_rows:,}")
        result_lines.append(f"✅ Schema exploration complete!")
        
        return "\n".join(result_lines)
    
    except Exception as e:
        return f"❌ Error exploring database: {str(e)}"
    
    finally:
        db.close()


@tool
def peek_table(
    table_name: Annotated[str, "Name of the table to preview"],
    limit: Annotated[int, "Number of sample rows"] = 3
) -> str:
    """
    Get a quick preview of actual data from any table.
    Shows sample rows to understand what the data looks like.
    
    Args:
        table_name: Name of the table (case-sensitive)
        limit: Number of rows to show (default: 3, max: 10)
    
    Perfect for: "Show me some data", "What's in table X?", "Give me examples"
    """
    if limit > 10:
        limit = 10
    
    db = SessionLocal()
    try:
        # Validate table exists
        check_query = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name = :table_name
            );
        """)
        table_exists = db.execute(check_query, {"table_name": table_name}).scalar()
        
        if not table_exists:
            # Get available tables
            tables_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables_result = db.execute(tables_query)
            available_tables = [row[0] for row in tables_result]
            return f"❌ Table '{table_name}' not found.\n\n📋 Available tables: {', '.join(available_tables)}"
        
        # Get sample data
        sample_query = text(f"SELECT * FROM {table_name} LIMIT :limit;")
        result = db.execute(sample_query, {"limit": limit})
        rows = result.fetchall()
        columns = result.keys()
        
        if not rows:
            return f"📭 Table '{table_name}' exists but is empty (0 rows)."
        
        # Format output
        result_lines = [f"👀 Preview of '{table_name}' ({len(rows)} sample row(s)):\n"]
        
        for i, row in enumerate(rows, 1):
            result_lines.append(f"{'─'*50}")
            result_lines.append(f"Row {i}:")
            for col_name, value in zip(columns, row):
                # Format value nicely
                if value is None:
                    display_val = "NULL"
                elif isinstance(value, (int, float)):
                    display_val = f"{value:,}" if isinstance(value, int) else f"{value:.2f}"
                else:
                    display_val = str(value)[:100]  # Truncate long strings
                
                result_lines.append(f"  {col_name:25} = {display_val}")
        
        result_lines.append(f"{'─'*50}")
        return "\n".join(result_lines)
    
    except Exception as e:
        return f"❌ Error reading table: {str(e)}"
    
    finally:
        db.close()

@tool
def analyze_column_stats(
    table_name: Annotated[str, "Name of the table"],
    column_name: Annotated[str, "Name of the column to analyze"]
) -> str:
    """
    Get comprehensive statistics for any column.
    Automatically adapts analysis based on data type.
    
    For numeric columns: min, max, avg, median, distribution
    For text columns: unique values, most common, value distribution
    For dates: range, distribution over time
    
    Args:
        table_name: Name of the table
        column_name: Name of the column
    
    Perfect for: "Analyze column X", "Tell me about field Y", "Statistics for Z"
    """
    db = SessionLocal()
    try:
        # Validate and get column info
        col_query = text("""
            SELECT data_type, udt_name
            FROM information_schema.columns
            WHERE table_name = :table_name AND column_name = :column_name;
        """)
        col_result = db.execute(col_query, {
            "table_name": table_name,
            "column_name": column_name
        })
        col_info = col_result.fetchone()
        
        if not col_info:
            return f"❌ Column '{column_name}' not found in table '{table_name}'."
        
        data_type = col_info[0]
        result_lines = [
            f"📊 Analysis of column '{column_name}' in '{table_name}'",
            f"Data type: {data_type}\n"
        ]
        
        # Basic stats
        basic_stats_query = text(f"""
            SELECT 
                COUNT(*) as total_rows,
                COUNT({column_name}) as non_null,
                COUNT(*) - COUNT({column_name}) as null_count,
                COUNT(DISTINCT {column_name}) as distinct_values
            FROM {table_name};
        """)
        stats = db.execute(basic_stats_query).fetchone()
        
        uniqueness_pct = (stats[3] / stats[1] * 100) if stats[1] > 0 else 0
        
        result_lines.extend([
            "📈 Basic Statistics:",
            f"  Total rows: {stats[0]:,}",
            f"  Non-null values: {stats[1]:,}",
            f"  Null values: {stats[2]:,} ({stats[2]/stats[0]*100:.1f}%)",
            f"  Distinct values: {stats[3]:,}",
            f"  Uniqueness: {uniqueness_pct:.1f}%\n"
        ])
        
        # Type-specific analysis
        if data_type in ['integer', 'bigint', 'smallint', 'numeric', 'real', 'double precision', 'decimal']:
            # Numeric analysis
            num_stats_query = text(f"""
                SELECT 
                    MIN({column_name}) as min_val,
                    MAX({column_name}) as max_val,
                    AVG({column_name}) as avg_val,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column_name}) as median_val,
                    STDDEV({column_name}) as stddev_val
                FROM {table_name}
                WHERE {column_name} IS NOT NULL;
            """)
            num_stats = db.execute(num_stats_query).fetchone()
            
            result_lines.extend([
                "🔢 Numeric Statistics:",
                f"  Minimum: {num_stats[0]}",
                f"  Maximum: {num_stats[1]}",
                f"  Average: {num_stats[2]:.2f}",
                f"  Median: {num_stats[3]:.2f}"
            ])
            if num_stats[4]:
                result_lines.append(f"  Std Dev: {num_stats[4]:.2f}\n")
        
        elif 'timestamp' in data_type or 'date' in data_type:
            # Date/time analysis
            date_stats_query = text(f"""
                SELECT 
                    MIN({column_name}) as earliest,
                    MAX({column_name}) as latest
                FROM {table_name}
                WHERE {column_name} IS NOT NULL;
            """)
            date_stats = db.execute(date_stats_query).fetchone()
            
            result_lines.extend([
                "📅 Date/Time Statistics:",
                f"  Earliest: {date_stats[0]}",
                f"  Latest: {date_stats[1]}\n"
            ])
        
        # Value distribution (for reasonable cardinality)
        if stats[3] <= 50:  # distinct_values
            top_values_query = text(f"""
                SELECT {column_name}, COUNT(*) as count,
                       (COUNT(*)::float / :non_null * 100) as percentage
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                GROUP BY {column_name}
                ORDER BY count DESC
                LIMIT 15;
            """)
            top_values = db.execute(top_values_query, {"non_null": stats[1]}).fetchall()
            
            result_lines.append("📊 Value Distribution (Top 15):")
            for val in top_values:
                bar_length = int(val[2] / 2)
                bar = "█" * bar_length
                result_lines.append(f"  {str(val[0])[:30]:30} {val[1]:>6,} ({val[2]:5.1f}%) {bar}")
        
        elif stats[3] > 50:
            # For high cardinality, show top 10 only
            top_10_query = text(f"""
                SELECT {column_name}, COUNT(*) as count
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                GROUP BY {column_name}
                ORDER BY count DESC
                LIMIT 10;
            """)
            top_values = db.execute(top_10_query).fetchall()
            
            result_lines.append(f"🔝 Top 10 Most Frequent Values (out of {stats[3]:,} unique):")
            for i, val in enumerate(top_values, 1):
                result_lines.append(f"  {i:2}. {str(val[0])[:40]:40} → {val[1]:,} times")
        
        return "\n".join(result_lines)
    
    except Exception as e:
        return f"❌ Error analyzing column: {str(e)}"
    
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
    peek_table,
    analyze_column_stats,
    query_postgres,
    suggest_interesting_queries
]
