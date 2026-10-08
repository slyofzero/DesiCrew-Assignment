# DesiCrew Intelligence Suite - Backend API

FastAPI backend service orchestrating:
1. **Excel Data Intelligence Agent** (`excel_agent/`) - LangGraph agent with sandboxed Python code execution and self-reflection.
2. **Support Assistant** (`support_assistant/`) - RAG agent with local FastEmbed vector store and document exploration tools.
3. **Intelligent Document Processing** (`idp_pipeline/`) - RapidOCR ONNX extraction, field validation, and human triage review.

## Quick Start

```bash
# 1. Environment setup
cp .env.example .env
# Edit .env with your AI_TOKEN

# 2. Install dependencies & run with uv
uv sync
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Refer to the root [README.md](../README.md) for full project architecture, environment variable specifications, and end-to-end setup instructions.
