# Backend Configuration

Models are configured using YAML files.

## Example

```yaml
models:
  mistral:
    type: llama_cpp
    model_path: /models/mistral-7b.Q4_K_M.gguf
    context_length: 8192
    n_gpu_layers: 35
    n_threads: 8
    temperature: 0.7
    top_p: 0.9
