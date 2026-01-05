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

Setup

pip install -r requirements_demo.txt

---


Configuration

Configure your database connection in environment variables:

DATABASE_URL=postgresql://user:password@localhost:5432/dbname

DB_USER=youruser
DB_PASSWORD=yourpassword
DB_NAME=databasename
DB_HOST=localhost
DB_PORT=5432
LANGSMITH_API_KEY=yourapikeyhere

Configure models in models.yaml:

- Model paths

- Backend type

- Context length

- Quantization

---

Running the Agent

Once installed and configured, the agent can be started using LangGraph's development runtime.

Using LangGraph Dev

Ensure you have your environment variables configured (including database access and model paths), then run:

langgraph dev

This will start the LangGraph development server and execute the agent graph locally, allowing you to:

- Inspect graph execution step by step

- Observe reasoning ↔ tool interactions

- Debug state transitions and tool calls

Using LangSmith (Optional)

For observability and tracing, you can enable LangSmith:

1. Add your API key to .env:

LANGSMITH_API_KEY=your_api_key_here

2. Run the agent as usual

LangSmith will automatically capture:

- Model inputs and outputs

- Tool calls and results

- Graph execution traces

This is especially useful for debugging, evaluation, and iterative improvement of the agent.

