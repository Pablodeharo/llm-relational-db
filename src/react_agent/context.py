# context.py
"""
Runtime context for the data analyst agent.
Defines reasoning model, tool-specific models, and system prompt.
"""

from dataclasses import dataclass, field
from react_agent.prompts import get_prompt

@dataclass(kw_only=True)
class Context:
    """
    Context configuration for the ReAct agent.
    
    Attributes:
        reasoning_model: The main LLM for reasoning (decides which tools to call)
        tool_models: Mapping of tool types to specific models
        system_prompt: System prompt that defines agent behavior
    """
    
    reasoning_model: str = "mistral"  # Main reasoning LLM
    
    tool_models: dict = field(default_factory=lambda: {
        "query_postgres": "sqlcoder",
        "peek_table": "sqlcoder",
        "analyze_column_stats": "sqlcoder",
        "explore_database": "sqlcoder",
        "suggest_interesting_queries": "sqlcoder",
        # future tools: "web_search": "tavily"
    })
    
    system_prompt: str = field(
        default_factory=lambda: get_prompt("analyst")
    )
    
    def __post_init__(self):
        if not self.reasoning_model:
            raise ValueError("Reasoning model cannot be empty")
        
        if not self.tool_models:
            raise ValueError("Tool models cannot be empty")
        
        if not self.system_prompt:
            raise ValueError("System prompt cannot be empty")
