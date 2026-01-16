from react_agent.tools import TOOLS

class ToolRouter:
    def __init__(self, runtime):
        self.runtime = runtime
        self.tools = {tool.name: tool for tool in TOOLS}

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not found")

        tool_fn = self.tools[tool_name]

        # Ejecuta directamente el código Python
        result = tool_fn.invoke(kwargs)

        return result
