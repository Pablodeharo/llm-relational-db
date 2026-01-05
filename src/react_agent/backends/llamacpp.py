"""
backends/llamacpp.py

Llama.cpp backend optimized for CPU inference with GGUF models.
Extremely efficient for quantized models (Q4, Q5, Q8).
"""

import asyncio
import time
import os
from typing import Dict, List, Optional, Any

try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    print("llama-cpp-python not installed. Install: pip install llama-cpp-python")

from .base import (
    ModelBackend,
    GenerationConfig,
    ModelResponse,
    ModelInfo,
    ToolCall
)


class LlamaCppBackend(ModelBackend):
    """
    Backend using llama.cpp for GGUF models.
    Optimized for CPU inference with low memory usage.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        
        if not LLAMACPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is required for LlamaCppBackend. "
                "Install: pip install llama-cpp-python"
            )
        
        self.model_path = model_config.get("model_path")
        if not self.model_path:
            raise ValueError(f"'model_path' not specified for {self.model_name}")
            
        self.n_ctx = model_config.get("context_length", 4096)
        self.n_threads = model_config.get("n_threads", os.cpu_count() or 4)
        self.n_gpu_layers = model_config.get("n_gpu_layers", 0)
        self.device = "cpu" if self.n_gpu_layers == 0 else "cuda"
        
    async def load(self) -> None:
        """Load GGUF model with llama.cpp"""
        if self.is_loaded:
            print(f"Model {self.model_name} already loaded")
            return
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"GGUF model not found at: {self.model_path}\n"
                f"Download it or check the path in config."
            )
        
        print(f"  Loading {self.model_name} with llama.cpp...")
        print(f"  Path: {self.model_path}")
        print(f"  Context: {self.n_ctx} tokens")
        print(f"  Threads: {self.n_threads}")
        print(f"  GPU Layers: {self.n_gpu_layers}")
        
        def _load():
            model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
                n_batch=512,
                use_mlock=True,
            )
            return model
        
        # Load in thread
        self._model = await asyncio.to_thread(_load)
        self.is_loaded = True
        
        print(f"✓ {self.model_name} ready ({self.device})")
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        tools: Optional[List[Dict]] = None
    ) -> ModelResponse:
        """Generate text using llama.cpp"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded. Call load() first.")
        
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        def _generate():
            # Prepare stop sequences
            stop = config.stop_sequences if config.stop_sequences else []
            
            # Generate
            response = self._model(
                prompt,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                stop=stop,
                echo=False,  # Don't repeat the prompt
            )
            
            generated_text = response['choices'][0]['text']
            
            # Token usage
            usage = response.get('usage', {})
            prompt_tokens = usage.get('prompt_tokens', self._count_tokens(prompt))
            completion_tokens = usage.get('completion_tokens', self._count_tokens(generated_text))
            
            return generated_text, prompt_tokens, completion_tokens
        
        # Run in thread
        generated_text, prompt_tokens, completion_tokens = await asyncio.to_thread(_generate)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Parse tool calls if tools provided
        tool_calls = []
        if tools:
            tool_calls = self._parse_tool_calls(generated_text, tools)
        
        return ModelResponse(
            content=generated_text,
            tool_calls=tool_calls,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            },
            latency_ms=latency_ms,
            backend_name="llamacpp",
            raw_response=None
        )
    
    def unload(self) -> None:
        """Free memory"""
        if self._model is not None:
            # llama.cpp handles cleanup automatically
            del self._model
            self._model = None
        
        self.is_loaded = False
        print(f"🗑️  {self.model_name} unloaded")
    
    def get_info(self) -> ModelInfo:
        """Get model information"""
        memory_mb = 0.0
        quantization = "unknown"
        
        if self.model_path:
            # Try to extract quantization from filename
            filename = os.path.basename(self.model_path).lower()
            if "q4" in filename:
                quantization = "Q4"
            elif "q5" in filename:
                quantization = "Q5"
            elif "q8" in filename:
                quantization = "Q8"
            
            # Estimate memory from file size
            try:
                file_size_mb = os.path.getsize(self.model_path) / (1024 ** 2)
                # Add ~500MB for context and overhead
                memory_mb = file_size_mb + 500
            except Exception:
                pass
        
        return ModelInfo(
            name=self.model_name,
            backend="llamacpp",
            device=self.device,
            memory_usage_mb=memory_mb,
            quantization=quantization,
            context_length=self.n_ctx
        )
    
    def supports_tool_calling(self) -> bool:
        """
        llama.cpp supports tool calling through prompt engineering.
        All models can technically do it with proper prompts.
        """
        return True
    
    def _count_tokens(self, text: str) -> int:
        """Estimate tokens (llama.cpp doesn't expose tokenizer easily)"""
        # More accurate than base: 1 token ≈ 3.5 chars for most models
        return int(len(text) / 3.5)