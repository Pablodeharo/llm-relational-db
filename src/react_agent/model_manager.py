"""
model_manager.py

Central orchestrator for managing multiple model backends.
Handles model loading, caching, auto-selection, and health checks.
"""

import os
import yaml
import torch
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path

from react_agent.backends.base import ModelBackend, GenerationConfig, ModelResponse
from react_agent.backends.transformers import TransformersBackend
from react_agent.backends.llamacpp import LlamaCppBackend


class HardwareDetector:
    """Detects available hardware capabilities"""
    
    @staticmethod
    def get_info() -> Dict[str, Any]:
        """Detect system hardware"""
        info = {
            "cpu_cores": os.cpu_count() or 4,
            "has_cuda": torch.cuda.is_available(),
            "cuda_device_count": 0,
            "cuda_device_name": None,
            "total_vram_mb": 0,
            "available_vram_mb": 0,
        }
        
        if info["has_cuda"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            if info["cuda_device_count"] > 0:
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
                info["total_vram_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                info["available_vram_mb"] = (
                    torch.cuda.get_device_properties(0).total_memory - 
                    torch.cuda.memory_allocated(0)
                ) / (1024**2)
        
        return info
    
    @staticmethod
    def can_use_gpu(required_vram_mb: float = 4000) -> bool:
        """Check if GPU is available with enough VRAM"""
        if not torch.cuda.is_available():
            return False
        
        info = HardwareDetector.get_info()
        return info["available_vram_mb"] >= required_vram_mb


class ModelManager:
    """
    Central manager for all model backends.
    
    Features:
    - Load/unload models dynamically
    - Cache multiple models
    - Auto-select best backend for hardware
    - Health monitoring
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        """
        Initialize ModelManager.
        
        Args:
            config_path: Path to models configuration YAML
        """
        self.config_path = config_path
        self.models_config = self._load_config()
        self.loaded_models: Dict[str, ModelBackend] = {}
        self.hardware_info = HardwareDetector.get_info()
        
        print("🖥️  Hardware detected:")
        print(f"  CPU Cores: {self.hardware_info['cpu_cores']}")
        print(f"  CUDA Available: {self.hardware_info['has_cuda']}")
        if self.hardware_info['has_cuda']:
            print(f"  GPU: {self.hardware_info['cuda_device_name']}")
            print(f"  VRAM: {self.hardware_info['available_vram_mb']:.0f} MB available")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load models configuration from YAML"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            print(f"⚠️  Config not found: {self.config_path}")
            print("   Using default configuration")
            return self._get_default_config()
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Loaded config from {self.config_path}")
        return config
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration if YAML not found"""
        return {
            "models": {
                "default": {
                    "name": "default",
                    "repo": "meta-llama/Llama-3.2-1B-Instruct",
                    "backends": ["transformers"],
                    "default_backend": "transformers",
                    "quantization": "4bit",
                    "context_length": 4096,
                    "memory_required": {
                        "transformers-4bit": 2048,
                        "transformers-8bit": 3072,
                        "llamacpp": 2048
                    }
                }
            }
        }
    
    def _select_backend(self, model_config: Dict[str, Any]) -> str:
        """
        Auto-select best backend based on hardware and model config.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            Backend name ("transformers" or "llamacpp")
        """
        available_backends = model_config.get("backends", [])
        normalized_backends = [
            b.split("-", 1)[0] for b in available_backends
        ]
        
        default_backend = model_config.get("default_backend", "transformers")
        memory_required = model_config.get("memory_required", {})
        
        # If GPU available and transformers is supported
        if self.hardware_info["has_cuda"] and "transformers" in available_backends:
            # Check if we have enough VRAM for quantized model
            quantization = model_config.get("quantization", "4bit")
            key = f"transformers-{quantization}"
            required_vram = memory_required.get(key, 4000)
            
            if self.hardware_info["available_vram_mb"] >= required_vram:
                print(f"  ✓ Selected backend: transformers (GPU, {quantization})")
                return "transformers"
        
        # Fallback to llamacpp for CPU
        if "llamacpp" in available_backends:
            gguf_path = model_config.get("gguf_path")
            if gguf_path and os.path.exists(gguf_path):
                print(f"  ✓ Selected backend: llamacpp (CPU)")
                return "llamacpp"
            else:
                print(f"  ⚠️  GGUF not found: {gguf_path}")
        
        # Default fallback
        print(f"  ✓ Selected backend: {default_backend} (default)")
        return default_backend
    
    async def load_model(
        self,
        model_name: str,
        backend: Optional[str] = None,
        force_reload: bool = False
    ) -> ModelBackend:
        """
        Load a model by name.
        
        Args:
            model_name: Name from config (e.g., "sqlcoder")
            backend: Force specific backend (optional)
            force_reload: Reload even if already loaded
            
        Returns:
            Loaded ModelBackend instance
        """
        # Check cache
        if model_name in self.loaded_models and not force_reload:
            print(f"♻️  Using cached model: {model_name}")
            return self.loaded_models[model_name]
        
        # Get model config
        if model_name not in self.models_config.get("models", {}):
            raise ValueError(f"Model '{model_name}' not found in config")
        
        model_config = self.models_config["models"][model_name].copy()
        model_config["name"] = model_name
        
        # Select backend
        if backend is None:
            backend = self._select_backend(model_config)
        
        print(f"🚀 Loading model: {model_name} (backend: {backend})")
        
        # Create backend instance
        if backend == "transformers":
            model_backend = TransformersBackend(model_config)
        elif backend == "llamacpp":
            model_backend = LlamaCppBackend(model_config)
        else:
            raise ValueError(f"Unknown backend: {backend}")
        
        # Load model
        await model_backend.load()
        
        # Cache it
        self.loaded_models[model_name] = model_backend
        
        return model_backend
    
    async def generate(
        self,
        model_name: str,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        tools: Optional[List[Dict]] = None
    ) -> ModelResponse:
        """
        Generate text using specified model.
        
        Args:
            model_name: Model to use
            prompt: Input prompt
            config: Generation config
            tools: Available tools
            
        Returns:
            ModelResponse
        """
        # Load model if not cached
        if model_name not in self.loaded_models:
            await self.load_model(model_name)
        
        backend = self.loaded_models[model_name]
        return await backend.generate(prompt, config, tools)
    
    def unload_model(self, model_name: str) -> None:
        """Unload a model from memory"""
        if model_name in self.loaded_models:
            self.loaded_models[model_name].unload()
            del self.loaded_models[model_name]
            print(f"🗑️  Unloaded: {model_name}")
        else:
            print(f"⚠️  Model not loaded: {model_name}")
    
    def unload_all(self) -> None:
        """Unload all models"""
        model_names = list(self.loaded_models.keys())
        for name in model_names:
            self.unload_model(name)
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get info about a loaded model"""
        if model_name in self.loaded_models:
            info = self.loaded_models[model_name].get_info()
            return {
                "name": info.name,
                "backend": info.backend,
                "device": info.device,
                "memory_mb": info.memory_usage_mb,
                "quantization": info.quantization,
                "context_length": info.context_length
            }
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all loaded models.
        
        Returns:
            Dict with system and model status
        """
        return {
            "hardware": self.hardware_info,
            "loaded_models": [
                self.get_model_info(name) 
                for name in self.loaded_models.keys()
            ],
            "total_models": len(self.loaded_models),
            "available_models": list(self.models_config.get("models", {}).keys())
        }
    
    async def warmup(self, model_name: str) -> None:
        """
        Warm up a model with a test generation.
        Useful to compile CUDA kernels and cache.
        
        Args:
            model_name: Model to warm up
        """
        print(f"🔥 Warming up {model_name}...")
        
        test_prompt = "SELECT"
        config = GenerationConfig(max_tokens=10, temperature=0.1)
        
        await self.generate(model_name, test_prompt, config)
        
        print(f"✓ {model_name} warmed up")


# ====================
# Singleton instance
# ====================
_manager_instance: Optional[ModelManager] = None


def get_model_manager(config_path: str = "config/models.yaml") -> ModelManager:
    """
    Get singleton ModelManager instance.
    
    Args:
        config_path: Path to config (only used on first call)
        
    Returns:
        ModelManager instance
    """
    global _manager_instance
    
    if _manager_instance is None:
        _manager_instance = ModelManager(config_path)
    
    return _manager_instance