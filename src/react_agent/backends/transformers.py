"""
backends/transformers.py

HuggingFace Transformers backend with support for:
- GPU acceleration (CUDA)
- Quantization (4-bit, 8-bit via BitsAndBytes)
- CPU fallback
- Tool calling
"""

import asyncio
import time
import torch
from typing import Dict, List, Optional, Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from .base import (
    ModelBackend,
    GenerationConfig,
    ModelResponse,
    ModelInfo,
    ToolCall
)

class TransformersBackend(ModelBackend):
    """
    Backend using HuggingFace Transformers.
    Optimized for GPU with quantization support.
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.device = "cpu"
        self.quantization = model_config.get("quantization")
        self.repo_id = model_config.get("repo")
        self._pipeline = None
        
    async def load(self) -> None:
        """Load model with transformers + optional quantization"""
        if self.is_loaded:
            print(f"Model {self.model_name} already loaded")
            return
        
        print(f"Loading {self.model_name} with Transformers...")
        
        def _load():
            # Detect device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.repo_id,
                trust_remote_code=True
            )
            
            # Configure quantization
            model_kwargs = {
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
            }
            
            if self.quantization == "4bit" and self.device == "cuda":
                print("  🔧 Applying 4-bit quantization (NF4)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model_kwargs["quantization_config"] = quantization_config
                model_kwargs["device_map"] = "auto"
                
            elif self.quantization == "8bit" and self.device == "cuda":
                print("  🔧 Applying 8-bit quantization")
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
                
            else:
                # No quantization
                if self.device == "cuda":
                    model_kwargs["torch_dtype"] = torch.float16
                    model_kwargs["device_map"] = "auto"
                else:
                    model_kwargs["torch_dtype"] = torch.float32
            
            # Load model
            self._model = AutoModelForCausalLM.from_pretrained(
                self.repo_id,
                **model_kwargs
            )
            self.context_lenght = getattr(
                self._model.config,
                "Max_position_embeddings",
                4096
            )
            
            # Move to device if not using device_map
            if "device_map" not in model_kwargs:
                self._model.to(self.device)
            
            print(f"  ✓ Model loaded on {self.device}")
            
            return self._tokenizer, self._model
        
        # Load in thread to avoid blocking
        self._tokenizer, self._model = await asyncio.to_thread(_load)
        self.is_loaded = True
        
        print(f"✓ {self.model_name} ready ({self.device})")
    
    async def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        tools: Optional[List[Dict]] = None
    ) -> ModelResponse:
        """Generate text using the loaded model"""
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} not loaded. Call load() first.")
        
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        def _generate():
            # Tokenize
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.context_length,
                context_length=self.context_length
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = inputs.to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    top_k=config.top_k,
                    do_sample=config.temperature > 0,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            
            # Decode only new tokens
            input_length = inputs.input_ids.shape[1]
            generated_tokens = outputs[0][input_length:]
            generated_text = self._tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            )
            
            # Token counts
            prompt_tokens = inputs.input_ids.shape[1]
            completion_tokens = len(generated_tokens)
            
            return generated_text, prompt_tokens, completion_tokens
        
        # Run generation in thread
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
            backend_name="transformers",
            raw_response=None
        )
    
    def unload(self) -> None:
        """Free GPU/CPU memory"""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_loaded = False
        print(f"{self.model_name} unloaded")
    
    def get_info(self) -> ModelInfo:
        """Get model information"""
        memory_mb = 0.0
        
        if self._model is not None:
            if hasattr(self._model, "get_memory_footprint"):
                memory_mb = self._model.get_memory_footprint() / (1024 ** 2)
            elif torch.cuda.is_available():
                memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        
        return ModelInfo(
            name=self.model_name,
            backend="transformers",
            device=self.device,
            memory_usage_mb=memory_mb,
            quantization=self.quantization,
            context_length=self.model_config.get("context_length", 4096)
        )
    
    def supports_tool_calling(self) -> bool:
        """
        Check if model supports tool calling.
        Can be enhanced with model-specific checks.
        """
        # Basic check: some models have "tool" or "function" in name
        model_name_lower = self.model_name.lower()
        return any(keyword in model_name_lower for keyword in ["tool", "function", "agent"])
    
    def _count_tokens(self, text: str) -> int:
        """Accurate token count using tokenizer"""
        if self._tokenizer:
            return len(self._tokenizer.encode(text))
        return super()._count_tokens(text)