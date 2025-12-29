# Model Manager

The `ModelManager` is responsible for orchestrating model loading and reuse.

It acts as a central registry that:
- Loads models on demand
- Prevents duplicate loading
- Provides access to active backend instances

## Responsibilities

- Read model configuration from YAML
- Instantiate the correct backend class
- Cache loaded models
- Return existing instances when requested again

## Why a Model Manager?

Without a central manager:
- Multiple models could be loaded redundantly
- GPU memory would be wasted
- Lifecycle control would be fragmented

The `ModelManager` ensures:
- One model instance per configuration
- Clean separation between configuration and execution
