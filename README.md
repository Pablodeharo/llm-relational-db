# LLM + Relational Database ReAct Agent

[![LangGraph](https://img.shields.io/badge/Built_with-LangGraph-00324d.svg)](https://github.com/langchain-ai/langgraph)
[![LangChain](https://img.shields.io/badge/Powered_by-LangChain-1c3c3c.svg)](https://github.com/langchain-ai/langchain)
[![Python](https://img.shields.io/badge/Python-Backend_AI-blue.svg)](https://www.python.org/)

This repository implements a **ReAct-style AI agent** using **LangGraph**, designed to reason over user queries and interact safely with a **PostgreSQL relational database**.

The project follows a **backend-first, production-minded architecture**, with clear separation between reasoning, tool execution, and database access.

---

## Overview

The agent is built around the **ReAct (Reason + Act)** paradigm:

- The **reasoning model** (`mistral`) interprets user queries and decides the next action.
- Determines whether an external action (tool) is required.
- Executes the corresponding **tool** via the **ToolRouter**.
- Observes results and continues reasoning.
- Produces a final, user-friendly response.

The execution flow is implemented as a **cyclical state machine** using **LangGraph**, making it extensible and backend-friendly.

---

## Agent Workflow

1. User sends a **natural language query**.
2. The **reasoning model** decides:
   - Answer directly, or
   - Call a tool (e.g., SQL query, database exploration, or other tools)
3. **ToolRouter** selects the appropriate model for the tool:
   - SQL tools → `sqlcoder`
   - Other tools (or reasoning-heavy tasks) → `mistral`
4. Tool executes and returns results.
5. Reasoning model interprets results and decides next steps.
6. Conversation continues until a **final answer** is produced.

This design ensures **context-aware reasoning**, **safe execution**, and **iterative exploration** of data.

---

## Models and Backend

The agent uses a **pluggable model backend system**:

| Model      | Role                  | Type       |
|-----------|----------------------|------------|
| mistral   | Reasoning & conversation | llama_cpp |
| sqlcoder  | SQL query execution     | llama_cpp |

### llama.cpp Backend

- Optimized for CPU inference.
- Supports **GGUF quantized models** (Q4, Q5, Q8).
- Low memory footprint, fast startup.
- Handles both reasoning and SQL execution tasks.
- Integrated via `ModelManager` for dynamic loading and caching.

### ModelManager

- Loads models based on configuration (`models.yaml`).
- Provides unified interface for **prompt generation**, **text completion**, and **tool call handling**.
- Caches models for efficiency.
- Supports **async execution**, allowing reasoning and tools to run concurrently if needed.

---

## Tool System

### ToolRouter

- Routes tool calls from the reasoning model to the appropriate backend.
- Supports multiple tools and models:
  - Database queries → `sqlcoder`
  - Reasoning / advanced tools → `mistral`
- Returns results as AIMessages back to the reasoning model.
- Fully asynchronous to avoid blocking the conversation flow.

### Available Tools

Defined in `tools.py`:

- **Database exploration**: `explore_database()`, `peek_table()`, `analyze_column_stats()`
- **SQL query execution**: `query_postgres()`
- **Intelligent suggestions**: `suggest_interesting_queries()`

Tools are designed for **read-only safe operations** and support PostgreSQL.

---

## Graph Architecture (LangGraph)

The conversation flow is implemented as a **StateGraph**:

- **Nodes**:
  - `call_model`: Executes the reasoning model (`mistral`)
  - `tools`: Executes tools via ToolRouter
- **Edges**:
  - `__start__ → call_model`
  - Conditional routing based on tool calls:
    - `call_model → tools` if tool calls exist
    - `call_model → __end__` if no tool calls
  - `tools → call_model` for iterative reasoning
- **Routing function**: `route_model_output(state)` checks for tool calls in the last AIMessage.

This allows the agent to **loop naturally between reasoning and tool execution**.

---

## Backend Stack

- **Python 3.11**
- **LangGraph**: State machine orchestration for nodes and edges
- **LangChain Core**: AIMessage objects and messaging
- **SQLAlchemy**: PostgreSQL database interaction
- **llama_cpp**: Reasoning and SQL model backend
- **Asyncio**: Non-blocking model calls and tool execution

---

## Example Flow

User: "Show me the top customers in my database."

1. **Reasoning model (`mistral`)** decides to call `query_postgres()`.
2. **ToolRouter** selects `sqlcoder` to execute the query.
3. SQL query executed safely on PostgreSQL.
4. Results returned to `mistral` for interpretation.
5. Agent responds with formatted, user-friendly output and suggested next steps.

---

## Key Features

- **Safe database operations**: Only SELECT queries allowed.
- **Iterative reasoning**: Loops between reasoning and tools for context-aware answers.
- **Pluggable models**: Easily add new reasoning or tool models.
- **ToolRouter abstraction**: Dynamically routes tool calls to the correct model.
- **Async execution**: Efficient handling of multiple requests and tool calls.
- **Extensible architecture**: Add new tools, backends, or models with minimal changes.

---

This agent is designed for **research, development, and backend deployment**, providing a strong foundation for **AI-assisted data exploration**.

