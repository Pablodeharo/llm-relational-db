"""
context.py

Defines configurable parameters for the agent.
Provides dynamic configuration via environment variables or direct initialization.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Annotated

from . import prompts


@dataclass(kw_only=True)
class Context:
    """
    Context container for the agent.
    
    Holds configuration for:
    - System prompt
    - Model selection
    - Tool parameters (e.g., search results limit)
    """

    system_prompt: str = field(
        default=prompts.SYSTEM_PROMPT,
        metadata={
            "description": (
                "The system prompt used for the agent. "
                "Controls the behavior, style, and context for the model responses."
            )
        },
    )

    model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        metadata={
            "description": (
                "The language model used for the agent's main interactions. "
                "Should follow the format 'provider/model-name' or a local path."
            )
        },
    )

    max_search_results: int = field(
        default=10,
        metadata={
            "description": (
                "Maximum number of search results returned for each search query "
                "(used by tools such as web search or RAG pipelines)."
            )
        },
    )

    def __post_init__(self) -> None:
        """
        Automatically override fields with environment variables if they exist.

        Environment variable names are the uppercased field names, e.g.:
        - SYSTEM_PROMPT
        - MODEL
        - MAX_SEARCH_RESULTS
        """
        for f in fields(self):
            if not f.init:
                continue
            current_value = getattr(self, f.name)
            # Override default with env var if available
            env_value = os.environ.get(f.name.upper())
            if env_value is not None:
                # Convert to int if the field type is int
                if f.type == int:
                    try:
                        env_value = int(env_value)
                    except ValueError:
                        pass
                setattr(self, f.name, env_value)
