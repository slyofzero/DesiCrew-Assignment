# Agent Operating Guidelines & Development Principles

## 1. Review-Driven Development (RDD)
- **Atomic, Incremental Steps**: Break down the development cycle into small, self-contained, atomic steps.
- **Minimal, Focused Edits**: Make only the minimal edits required for each specific step without sacrificing the quality or robustness of the end product.
- **Reviewability**: Avoid broad, sweeping changes across multiple disconnected files in a single step. Every change must be easily and quickly reviewable by the user within a short time.
- **Explicit Checkpoints**: Before moving to the next component or file, summarize what was created or changed, why it was implemented that way, and verify that it compiles/runs cleanly.

## 2. Architecture & Tech Stack Rules
- **Backend**: Python 3.12 (managed via `uv`), FastAPI, LangChain / LangGraph, OpenPyXL, Pandas.
- **Frontend**: Next.js (App Router, React, Tailwind CSS, Lucide icons) with 3 dedicated routes:
  - Route 1 (`/excel-agent`): Question 1 - Excel Data Intelligence Agent with Code Execution, Search, Sheet Management, and Self-Reflection.
  - Route 2 (`/support-assistant`): Question 2 - Document-Aware Support Assistant with Multi-Turn Memory & Anti-Repetition.
  - Route 3 (`/idp-pipeline`): Question 3 - Intelligent Document Processing Pipeline with per-field confidence scoring and human review triage.
- **LLM / Vision Provider**: AI Pipe (`https://aipipe.org/openrouter/v1`) using `AI_PIPE_TOKEN` from `.env`.

## 3. Project 1 Specific Guidelines
- **Core Tools**: `execute_python` (sandboxed code runner), `search_definitions` (web/definition lookup), `create_sheet` (write/export new sheets), `edit_sheet` (modify/add columns to existing sheets), `read_metadata` (schema inspection).
- **Agentic Paradigms**:
  - **Self-Reflection & Self-Repair**: Capture runtime tracebacks and schema errors to iteratively fix generated code before returning answers.
  - **Memory**: Working memory for intermediate findings, schema/data dictionary memory, and multi-turn conversational context.
