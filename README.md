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