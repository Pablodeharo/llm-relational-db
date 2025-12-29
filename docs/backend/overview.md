# Backend Architecture Overview

The backend layer is responsible for loading, managing, and executing large language models
in a unified and extensible way.

It abstracts away model-specific details (Transformers, llama.cpp, etc.)
and exposes a common interface used by the rest of the system.

## Core Principles

- Backend-agnostic model execution
- Explicit resource management (CPU / GPU / memory)
- Pluggable model backends
- Unified generation and tool-calling interface

## Main Components

### ModelBackend (Abstract Base Class)

All backends must implement the `ModelBackend` interface, which defines:

- How models are loaded and unloaded
- How text generation is performed
- How tool calls are parsed
- How model metadata is exposed

This allows higher-level components to remain agnostic to the underlying inference engine.

### Implementations

Currently supported backends:

- **TransformersBackend** – Hugging Face Transformers (GPU / CPU, quantization)
- **LlamaCppBackend** – llama.cpp GGUF models (CPU-first, low memory)

Additional backends can be added by implementing the same interface.
