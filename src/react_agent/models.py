"""
models.py

Async model loader for the agent.
Supports dynamic model selection via context.model,
GPU acceleration (RTX 3050), and CPU fallback.
"""

import os
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM


DEFAULT_MODEL_AGENT = "bigcode/starcoderbase"


# Async state for caching
_tokenizer_agent = None
_model_agent = None
_model_ready = asyncio.Event()
_model_loading = False

# Offload folder (if GPU memory is insufficient)
OFFLOAD_DIR = os.environ.get("HF_OFFLOAD_DIR", "./offload")
os.makedirs(OFFLOAD_DIR, exist_ok=True)


async def get_agent_model_async(model_name: str = None):
    """
    Load tokenizer and model asynchronously.
    - model_name: Optional, overrides the default model
    """
    global _tokenizer_agent, _model_agent, _model_loading, _model_ready

    model_name = model_name or DEFAULT_MODEL_AGENT

    # Return cached model if already loaded
    if _model_agent is not None:
        return _tokenizer_agent, _model_agent

    # Wait if another coroutine is loading the model
    if _model_loading:
        await _model_ready.wait()
        return _tokenizer_agent, _model_agent

    _model_loading = True

    try:
        def _load_model():
            global _tokenizer_agent, _model_agent

            # Double check in thread
            if _model_agent is not None:
                return _tokenizer_agent, _model_agent

            try:
                # Load tokenizer
                _tokenizer_agent = AutoTokenizer.from_pretrained(model_name, use_fast=False)

                # Load model on GPU if available
                device_map = "auto" if torch.cuda.is_available() else None
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                _model_agent = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map=None,
                    dtype=torch.float32,
                    torch_dtype=dtype,
                    offload_folder=OFFLOAD_DIR if device_map else None
                )
                print(f"Model '{model_name}' loaded successfully on {'GPU' if device_map else 'CPU'}.")

            except Exception as e:
                # Fallback to CPU float32 if GPU load fails
                print(f"[GPU load failed, fallback to CPU] {e}")
                _tokenizer_agent = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                _model_agent = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map=None
                )
                print(f"Model '{model_name}' loaded on CPU (float32).")

            return _tokenizer_agent, _model_agent

        # Load model in separate thread
        tokenizer, model = await asyncio.to_thread(_load_model)

    finally:
        _model_ready.set()
        _model_loading = False

    return tokenizer, model


def get_agent_model(model_name: str = None):
    """
    Synchronous version for compatibility.
    """
    global _tokenizer_agent, _model_agent
    if _model_agent is None:
        loop = asyncio.get_event_loop()
        tokenizer, model = loop.run_until_complete(get_agent_model_async(model_name))
        return tokenizer, model
    return _tokenizer_agent, _model_agent
