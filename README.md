# LLM + Relational Database ReAct Agent

[![LangGraph](https://img.shields.io/badge/Built_with-LangGraph-00324d.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/Powered_by-LangChain-1c3c3c.svg)](https://github.com/langchain-ai/langchain)
[![Python](https://img.shields.io/badge/Python-Backend_AI-blue.svg)](https://www.python.org/)

This repository implements a **ReAct-style AI agent** using **LangGraph**, designed to reason over user queries and interact safely with a **relational PostgreSQL database**.

The project follows a **backend-first, production-minded architecture**, focusing on agentic reasoning, tool-based execution, and structured data access.

---

## Overview

The agent is built around the **ReAct (Reason + Act)** paradigm:

- The LLM reasons about the user request
- Decides whether an external action is required
- Executes tools (e.g. database queries)
- Observes results and continues reasoning
- Produces a final answer

The execution flow is implemented as a **cyclical state machine** using **LangGraph**, making it extensible and easy to integrate into backend services.

---

## What it does

The ReAct agent:

1. Accepts a natural language **user query**
2. Uses an LLM to **reason** about the request
3. Determines whether database access is needed
4. Executes a **safe SQL query** via a tool
5. Observes results and iterates if necessary
6. Returns a final, user-friendly answer

By default, the agent is configured with a **PostgreSQL read-only tool**, but it can be extended with additional tools or data sources.

---

## Model Backend Architecture

The agent uses a **pluggable backend system** to load and run language models in a flexible and hardware-aware way.

All model execution is abstracted behind a common interface, allowing the agent to switch seamlessly between **CPU and GPU inference**, different runtimes, and multiple model formats.

### Supported Backends

The backend system currently supports:

#### 🔹 Transformers Backend
- Based on **HuggingFace Transformers**
- Supports **CPU and CUDA GPU**
- Optional **4-bit / 8-bit quantization** via BitsAndBytes
- Automatic device detection
- Suitable for large models and GPU acceleration

#### 🔹 llama.cpp Backend
- Based on **llama.cpp** (`llama-cpp-python`)
- Optimized for **CPU inference**
- Supports **GGUF quantized models** (Q4, Q5, Q8)
- Low memory footprint and fast startup
- Ideal for local and resource-constrained environments

### Unified Backend Interface

All backends implement a shared abstract interface:

- `load()` – load model into memory
- `generate()` – generate text and optional tool calls
- `unload()` – free system resources
- `get_info()` – runtime model metadata
- `supports_tool_calling()` – tool-call capability check

This allows the agent to remain **backend-agnostic**, while the `ModelManager` decides which backend to use based on configuration and hardware availability.

### Tool Calling Support

Backend outputs are normalized into a standard response format:

- Generated content
- Parsed tool calls (JSON or ReAct format)
- Token usage statistics
- Latency metrics

This ensures consistent agent behavior regardless of the underlying model runtime.

### Production-Oriented Design

The backend layer is designed with:
- Clear separation of concerns
- Asynchronous model loading and generation
- Safe resource cleanup
- Extensibility for future backends (e.g. vLLM)

This makes the agent suitable for **local development, research, and backend deployment**.