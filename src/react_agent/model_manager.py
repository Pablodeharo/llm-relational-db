from pathlib import Path
import yaml
from llama_cpp import Llama


class ModelManager:
    """
    Central manager responsible for loading and caching LLM backends.

    This class ensures that:
    - Models are loaded only once.
    - The same model instance is reused across the application.
    - Backend-specific logic is isolated from the rest of the system.
    """

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        """
        Carga el archivo YAML con la configuracion de modelos.
        Inicializa un diccionario privado _models que actuara como cache de modelos cargados.
        Esto evita cargar un modelos ya iniciado
        """
        self._models = {}

    async def get_backend(self, model_name: str):
        """
        Return a model backend instance by name.

        If the model has already been loaded, the cached instance is returned.
        Otherwise, it is created based on the configuration file.
        """

        if model_name in self._models:
            return self._models[model_name]

        model_cfg = self.config["models"].get(model_name)
        if not model_cfg:
            raise ValueError(f"Model '{model_name}' not found in config")

        if model_cfg["type"] == "llama_cpp":
            model = Llama(
                model_path=model_cfg["model_path"],
                n_ctx=model_cfg["context_length"],
                n_gpu_layers=model_cfg["n_gpu_layers"],
                n_threads=model_cfg["n_threads"],
                temperature=model_cfg["temperature"],
                top_p=model_cfg["top_p"],
            )
        else:
            raise ValueError(f"Unsupported model type: {model_cfg['type']}")

        self._models[model_name] = model
        return model
