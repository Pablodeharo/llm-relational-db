"""
backends/base.py

Abstract base class for all model backends.
Defines the interface that TransformersBackend, LlamaCppBackend, etc. must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: List[str] = None
    stream: bool = True
    
    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


@dataclass
class ToolCall:
    """Represents a tool call parsed from model output"""
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelResponse:
    """Standardized response from any backend"""
    content: str
    tool_calls: List[ToolCall]
    usage: Dict[str, int]  # tokens_prompt, tokens_completion, total
    latency_ms: float
    backend_name: str
    raw_response: Optional[Any] = None


@dataclass
class ModelInfo:
    """Information about loaded model"""
    name: str
    backend: str
    device: str
    memory_usage_mb: float
    quantization: Optional[str] = None
    context_length: int = 4096


class ModelBackend(ABC):
    """
    Abstract base class for model backends.
    
    Each backend (Transformers, LlamaCpp, VLLM) implements this interface
    to provide a unified API for the ModelManager.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize backend with configuration.
        
        Args:
            model_config: Dictionary with model configuration from YAML
        """
        self.model_config = model_config
        self.model_name = model_config.get("name", "unknown")
        self.is_loaded = False
        self._model = None
        self._tokenizer = None
    
    @abstractmethod
    async def load(self) -> None:
        """
        Load the model into memory.
        Must be implemented by each backend.
        
        Should:
        - Load model and tokenizer
        - Apply quantization if specified
        - Set self.is_loaded = True
        - Raise exception if loading fails
        """
        pass
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        tools: Optional[List[Dict]] = None
    ) -> ModelResponse:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            config: Generation parameters
            tools: Available tools for tool calling (optional)
            
        Returns:
            ModelResponse with generated text and parsed tool calls
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """
        Unload model from memory.
        Free GPU/CPU resources.
        """
        pass
    
    @abstractmethod
    def get_info(self) -> ModelInfo:
        """
        Get information about the loaded model.
        
        Returns:
            ModelInfo with details about model state
        """
        pass
    
    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """
        Check if this backend/model supports tool calling.
        
        Returns:
            True if tool calling is supported
        """
        pass
    
    def _parse_tool_calls(self, text: str, tools: Optional[List[Dict]] = None) -> List[ToolCall]:
        """
        Parse tool calls from generated text.
        Default implementation - can be overridden.
        
        Looks for patterns like:
        - JSON: {"tool": "query_postgres", "args": {...}}
        - ReAct: Action: query_postgres\nAction Input: {...}
        
        Args:
            text: Generated text
            tools: Available tools
            
        Returns:
            List of parsed ToolCall objects
        """
        import json
        import re
        
        tool_calls = []
        
        # Try JSON format first
        try:
            # Look for JSON blocks
            json_pattern = r'```json\s*(\{.*?\})\s*```'
            matches = re.finditer(json_pattern, text, re.DOTALL)
            
            for i, match in enumerate(matches):
                json_str = match.group(1)
                data = json.loads(json_str)
                
                if "tool" in data or "function" in data:
                    tool_calls.append(ToolCall(
                        id=f"call_{i}",
                        name=data.get("tool") or data.get("function"),
                        arguments=data.get("args") or data.get("arguments", {})
                    ))
        except Exception:
            pass
        
        # Try ReAct format
        if not tool_calls:
            action_pattern = r'Action:\s*(\w+)\s*Action Input:\s*(\{.*?\})'
            matches = re.finditer(action_pattern, text, re.DOTALL)
            
            for i, match in enumerate(matches):
                tool_name = match.group(1)
                try:
                    args = json.loads(match.group(2))
                    tool_calls.append(ToolCall(
                        id=f"call_{i}",
                        name=tool_name,
                        arguments=args
                    ))
                except Exception:
                    pass
        
        return tool_calls
    
    def _count_tokens(self, text: str) -> int:
        """
        Estimate token count.
        Default implementation - should be overridden with actual tokenizer.
        
        Args:
            text: Text to count
            
        Returns:
            Approximate token count
        """
        # Rough approximation: 1 token ≈ 4 characters
        return len(text) // 4
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, loaded={self.is_loaded})"