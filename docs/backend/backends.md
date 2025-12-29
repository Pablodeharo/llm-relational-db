# Backend Implementations

This document describes the supported model backends.

## LlamaCppBackend

Optimized for running GGUF models locally using `llama.cpp`.

### Features
- Extremely efficient CPU inference
- Optional GPU offloading
- Low memory footprint
- Ideal for local inference and edge devices

### Supported features
- Quantized models (Q4, Q5, Q8)
- Tool calling via prompt parsing
- Custom context length

---

## TransformersBackend

Uses HuggingFace Transformers for GPU-accelerated inference.

### Features
- CUDA acceleration
- 4-bit / 8-bit quantization via BitsAndBytes
- Large model support
- Flexible tokenizer handling

### When to use
- GPU available
- Larger models (7B+)
- Advanced fine-tuning workflows
