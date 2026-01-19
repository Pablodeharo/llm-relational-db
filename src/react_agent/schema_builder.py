import json
from react_agent.state import SchemaMemory, TableSchema

def build_schema_memory(raw_schema: str | dict) -> SchemaMemory:
    """
    Convert JSON returned from explore_database()
    in a structered SchemaMemory 
    
    :param raw_schema: Description
    :type raw_schema: str | dict
    :return: Description
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

    return SchemaMemory(
        loaded=True,
        table_count=raw_schema.get("table_count", len(tables)),
        public_only=True,
        tables=tables
    )