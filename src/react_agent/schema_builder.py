import json
from react_agent.state import SchemaMemory, TableSchema, TableRelationship

def build_schema_memory(raw_schema: str | dict) -> SchemaMemory:
    """
    Convert JSON returned from explore_database()
    into a structured SchemaMemory with relationships.
    
    :param raw_schema: JSON string or dict with tables and relationships
    :type raw_schema: str | dict
    :return: Structured schema memory
    :rtype: SchemaMemory
    """
    if isinstance(raw_schema, str):
        raw_schema = json.loads(raw_schema)

    tables = {}
    for table_name, columns in raw_schema["tables"].items():
        tables[table_name] = TableSchema(
            name=table_name,
            columns=columns,
            column_count=len(columns)
        )

    # Procesar relaciones
    relationships = []
    for rel in raw_schema.get("relationships", []):
        relationships.append(TableRelationship(
            from_table=rel["from_table"],
            from_column=rel["from_column"],
            to_table=rel["to_table"],
            to_column=rel["to_column"],
            constraint_name=rel["constraint_name"]
        ))

    return SchemaMemory(
        loaded=True,
        table_count=raw_schema.get("table_count", len(tables)),
        public_only=True,
        tables=tables,
        relationships=relationships
    )