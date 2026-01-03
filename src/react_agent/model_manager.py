"""
model_manager.py

Central manager for loading and caching model backends.
"""

from pathlib import Path
from typing import Dict, Optional
import yaml

from .backends.base import ModelBackend
from .backends.llamacpp import LlamaCppBackend
from .backends.transformers import TransformersBackend


class ModelManager:
    """
    Manages model backends with caching and lifecycle.
    
    Responsibilities:
    - Load backends based on YAML config
    - Cache loaded models to avoid reloading
    - Route to correct backend (llamacpp, transformers, etc.)
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to models.yaml configuration
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Cache: model_name -> backend instance
        self._models: Dict[str, ModelBackend] = {}
        
        print(f"ModelManager initialized with {len(self.config['models'])} models")
    
    async def get_backend(self, model_name: str) -> ModelBackend:
        """
        Get or create a model backend.
        
        Args:
            model_name: Name from models.yaml
            
        Returns:
            Loaded ModelBackend instance
            
        Raises:
            ValueError: If model not in config or unsupported type
        """
        # Return cached if already loaded
        if model_name in self._models:
            backend = self._models[model_name]
            if backend.is_loaded:
                print(f"Using cached {model_name}")
                return backend
        
        # Get config
        model_cfg = self.config["models"].get(model_name)
        if not model_cfg:
            available = ", ".join(self.config["models"].keys())
            raise ValueError(
                f"Model '{model_name}' not found in config.\n"
                f"Available models: {available}"
            )
        
        # Create backend based on type
        backend_type = model_cfg.get("type", "").lower()
        
        if backend_type == "llamacpp":
            backend = LlamaCppBackend(model_cfg)
        elif backend_type == "transformers":
            backend = TransformersBackend(model_cfg)
        else:
            raise ValueError(
                f"Unsupported backend type: '{backend_type}'\n"
                f"Supported: llamacpp, transformers"
            )
        
        # Load the model
        await backend.load()
        
        # Cache it
        self._models[model_name] = backend
        
        return backend
    
    def list_models(self) -> Dict[str, Dict]:
        """
        List all available models from config.
        
        Returns:
            Dict of model_name -> config
        """
        return self.config["models"]
    
    def get_loaded_models(self) -> Dict[str, ModelBackend]:
        """
        Get all currently loaded backends.
        
        Returns:
            Dict of model_name -> backend
        """
        return {
            name: backend 
            for name, backend in self._models.items() 
            if backend.is_loaded
        }
    
    async def unload(self, model_name: str) -> None:
        """
        Unload a specific model from memory.
        
        Args:
            model_name: Name of model to unload
        """
        if model_name in self._models:
            self._models[model_name].unload()
            del self._models[model_name]
            print(f"{model_name} unloaded and removed from cache")
    
    async def unload_all(self) -> None:
        """Unload all models from memory."""
        for name in list(self._models.keys()):
            await self.unload(name)
        print("All models unloaded")
    
    def get_hardware_config(self) -> Dict:
        """Get hardware configuration if present in config."""
        return self.config.get("hardware", {})