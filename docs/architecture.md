# DataExplorer Agent Architecture

This document describes the architecture of the **DataExplorer Agent**, a ReAct agent for data analysis with multiple models, tool routing, and PostgreSQL database integration.

---

## 1. Overview

The agent is built using **LangGraph Platform** and a **custom React Agent** structure:

- **Reasoning model (`mistral`)**: Decides what action to take and whether to use tools.
- **SQL model (`sqlcoder`)**: Executes queries safely on PostgreSQL databases.
- **ToolRouter**: Routes tool calls to the correct model based on configuration.
- **Tools**: Python functions for database exploration and analysis.
- **StateGraph**: Orchestrates the conversation flow between LLM and tools.

---

## 2. Core Components

### 2.1 Models

Defined in `models.yaml`:

| Name      | Type       | Purpose                    |
|----------|------------|---------------------------|
| mistral  | llama_cpp  | Reasoning & conversation  |
| sqlcoder | llama_cpp  | SQL query execution       |

**Configurable parameters**: `context_length`, `n_gpu_layers`, `n_threads`, `temperature`, `top_p`.

---

### 2.2 Context

Defined in `context.py`:

- **reasoning_model**: Model used for reasoning and deciding actions.
- **tool_models**: Mapping of tools to the model that executes them (e.g., `sqlcoder` for SQL tools, `mistral` for reasoning-based tools).
- **system_prompt**: Defines agent behavior using the prompt from `prompts.py`.

---

### 2.3 Tools

Defined in `tools.py`:

- **Database tools**: `explore_database`, `peek_table`, `analyze_column_stats`.
- **Query execution**: `query_postgres`.
- **Suggestions**: `suggest_interesting_queries`.
- Each tool is executed via **ToolRouter**, which selects the model (reasoning or SQL) based on context.

---

### 2.4 ToolRouter

Defined in `tool_router.py`:

- Receives tool calls from the reasoning model.
- Looks up which model should execute the tool.
- Returns results to the conversation, appended as an AIMessage.
- Supports asynchronous execution of multiple tool calls.

---

### 2.5 Graph Flow

Defined in `graph.py`:

- **Nodes**:
  1. `call_model`: Runs the reasoning model.
  2. `tools`: Executes tools using ToolRouter.
- **Edges**:
  - Start: `__start__ → call_model`
  - Conditional: `call_model → tools` if tool calls exist, else `__end__`.
  - Loop: `tools → call_model` to allow reasoning on tool outputs.
- **Routing logic**:
  - `route_model_output(state)`: Checks last AIMessage for tool calls.

---