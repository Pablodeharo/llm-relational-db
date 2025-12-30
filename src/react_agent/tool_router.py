# react_agent/tool_router.py

from pathlib import Path
from react_agent.model_manager import ModelManager


class ToolRouter:
    """
    Routes tool calls to the appropriate model backend.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self.model_manager = ModelManager(
            config_path=Path(__file__).parent / "config" / "models.yaml"
        )
        self._loaded_models = {}

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        Execute a tool using the model assigned in context.tool_models
        """
        model_name = self.runtime.context.tool_models.get(tool_name)

        if not model_name:
            raise ValueError(f"No model configured for tool '{tool_name}'")

        if model_name not in self._loaded_models:
            self._loaded_models[model_name] = await self.model_manager.get_backend(model_name)

        model = self._loaded_models[model_name]

        prompt = self._build_prompt(tool_name, kwargs)

        output = model(
            prompt,
            max_tokens=512,
            stop=["</s>"]
        )

        return output["choices"][0]["text"]

    def _build_prompt(self, tool_name: str, args: dict) -> str:
        """
        Simple prompt constructor for tools.
        Can be extended later for tool-specific formatting.
        """
        return f"Tool: {tool_name}\nArguments:\n{args}\n\nExecute the tool and return the result."
