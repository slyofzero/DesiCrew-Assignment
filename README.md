# DesiCrew Autonomous AI Intelligence & IDP Suite

A unified enterprise-grade AI system integrating **stateful multi-turn agents**, **vector-retrieval document synthesis**, and **intelligent document processing (IDP)** with human-in-the-loop review triage.

Built with **FastAPI**, **LangChain / LangGraph**, **Python 3.12**, and **Next.js 16 (React 19 + Tailwind CSS)**.

---

## Architecture Overview

```
                                 ┌────────────────────────────────────────────────┐
                                 │         Next.js Web Frontend (Port 3000)        │
                                 │  /excel-agent | /support-assistant | /idp-pipeline │
                                 └───────────────────────┬────────────────────────┘
                                                         │ HTTP REST / Uploads
                                                         ▼
                                 ┌────────────────────────────────────────────────┐
                                 │          FastAPI Central Hub (Port 8000)        │
                                 └───────┬─────────────────┬─────────────────┬────┘
                                         │                 │                 │
                  ┌──────────────────────▼──┐    ┌─────────▼────────┐   ┌───▼──────────────────────┐
                  │   Excel Data Agent      │    │ Support Assistant│   │ Intelligent Doc Pipeline │
                  │  (LangGraph Graph)      │    │   (RAG / Vector) │   │     (RapidOCR + HITL)    │
                  └──────────┬──────────────┘    └─────────┬────────┘   └───┬──────────────────────┘
                             │                             │                │
            ┌────────────────┴─────────────────────────────┴────────────────┴────────────────┐
            ▼                                              ▼                                 ▼
   External LLM Services                          Local Embedded Models             External Search
   • AI Pipe / OpenRouter Gateway                 • RapidOCR (ONNX Runtime)         • DuckDuckGo API
   • Models: GPT-4o, GPT-4o-mini, Nova-Lite       • FastEmbed (bge-small-en-v1.5)   • Wikipedia API
```

---

## Applications & Modules

### 1. Excel Data Intelligence Agent (`/excel-agent`)
* **LangGraph Stateful Loop**: Uses a dynamic `Reasoner -> Act -> Critic` agent loop with runtime self-reflection.
* **Sandboxed Code Execution**: Executes Python code on DataFrames in an isolated namespace, intercepting Tracebacks and feeding runtime schema hints back into the reflection engine to self-repair queries.
* **Workbook Tools**: Schema metadata inspection, cell range fetching, statistical summaries (mean, median, mode, quantiles), web definition lookup, sheet creation, and calculated column insertion via `openpyxl`.
* **State Persistence**: Multi-turn chat sessions with full agent trajectory logs (Reasoning, Actions, Critic feedback) saved to IndexedDB with localStorage fallback.

### 2. Notes Summarization & Intelligence Assistant (`/support-assistant`)
* **Multi-Format Document Parsing**: Recursively indexes `.md`, `.txt`, `.pdf` (via PyMuPDF), and `.ipynb` files.
* **Local Vector Retrieval**: Uses `fastembed` with quantized `BAAI/bge-small-en-v1.5` embeddings, persisting normalized vectors locally (`embeddings.npz` and `chunks_metadata.json`).
* **Anti-Repetition & Grounding**: Sliding window conversational history, section citation linking, and anti-repetition guardrails.
* **Exploration Tools**: Agent can dynamically browse directory trees, search documents by name pattern, read specific line ranges, and perform vector searches.

### 3. Intelligent Document Processing (IDP) Pipeline (`/idp-pipeline`)
* **OCR & Multi-Modal Parsing**: High-accuracy local text extraction on images and PDFs powered by `rapidocr-onnxruntime` (PP-OCRv4 ONNX model).
* **Document Classification**: Hybrid rule-based keyword matching and zero-shot LLM classification for Indian identity & financial documents (Aadhaar, PAN Card, Driving License, Passport, Bank Statement, Invoice/Bill, Salary Slip, Cheque).
* **Confidence & Format Validation**: Per-field confidence combining OCR character probabilities, regex pattern validation, and algorithmic checksums (e.g. Verhoeff algorithm for Aadhaar).
* **Human-in-the-Loop (HITL) Triage**: Flagging low-confidence fields (< 0.85), side-by-side document image preview with bounding box zoom, in-place correction, accept/reject decisions, and audit trail export (`idp_triage_audit.json`).

---

## External Services & Dependencies

| Service / Dependency | Purpose | Authentication / Configuration | Notes |
| :--- | :--- | :--- | :--- |
| **AI Pipe / OpenRouter** | Primary LLM Provider for reasoning, planning, reflection, and extraction | `AI_TOKEN` in `backend/.env`<br>`AI_BASE_URL` (default: `https://aipipe.org/openrouter/v1`) | Used across all 3 modules (`openai/gpt-4o`, `openai/gpt-4o-mini`, `amazon/nova-lite-v1`). |
| **DuckDuckGo Instant Answer API** | Business formula & metric definition lookup | None (Public HTTP) | Invoked by `search_definitions` tool in the Excel Agent. |
| **Wikipedia Search API** | Fallback knowledge lookup | None (Public HTTP) | Invoked if DuckDuckGo returns no abstract. |
| **RapidOCR (ONNX Runtime)** | Local document OCR engine | None (Runs locally on CPU/ONNX) | ONNX model weights auto-download on first execution. |
| **FastEmbed (`bge-small-en-v1.5`)** | Local embedding engine for Support Assistant | None (Runs locally on CPU/ONNX) | Generates 384-dimensional dense vectors cached in `backend/data/support_kb/`. |

---

## Environment Variables (`.env`)

The backend requires a `.env` file located inside the `backend/` directory. An example file is provided at `backend/.env.example`.

### Backend Configuration (`backend/.env`)

Create `backend/.env` with the following variables:

```env
# AI Pipe / OpenRouter Gateway
AI_TOKEN="your_aipipe_token_here"
AI_BASE_URL="https://aipipe.org/openrouter/v1"

# Support Assistant Model Selection
SUPPORT_MODEL="openai/gpt-4o-mini"

# Optional: Path to custom knowledge base documents directory
# Defaults to backend/data if omitted
# RAG_CHATBOT_DATA_DIR="C:/path/to/your/notes"
```

### Environment Variable Reference

* `AI_TOKEN` (**Required**): Your AI Pipe or OpenRouter API bearer token.
* `AI_BASE_URL` (**Optional**): Base URL for the OpenAI-compatible endpoint. Defaults to `https://aipipe.org/openrouter/v1`.
* `SUPPORT_MODEL` (**Optional**): Model used by the Support Assistant. Defaults to `openai/gpt-4o-mini`.
* `RAG_CHATBOT_DATA_DIR` (**Optional**): Absolute or relative path to a local directory containing documents to index into the Support Assistant vector store. Defaults to `backend/data`.

---

## Prerequisites

Before running the application, make sure you have the following installed:

* **Python 3.12+**: Managed via [`uv`](https://docs.astral.sh/uv/) (recommended) or standard `python -m venv`.
* **Node.js 18+** (Node.js 20+ recommended).
* **pnpm** (recommended, `npm i -g pnpm`) or `npm`.

---

## Step-by-Step Installation & Running Guide

### 1. Clone the Repository

```bash
git clone <repository-url>
cd DesiCrew
```

---

### 2. Configure Environment Variables

Copy the example environment configuration into `backend/.env`:

```bash
# On Linux / macOS / Git Bash
cp backend/.env.example backend/.env

# On Windows PowerShell
Copy-Item backend/.env.example backend/.env
```

Open `backend/.env` in your editor and insert your valid `AI_TOKEN`.

---

### 3. Start the Backend (FastAPI)

#### Option A: Using `uv` (Recommended)

From the project root:

```bash
cd backend
uv sync
uv run uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Option B: Using standard Python `venv`

```bash
cd backend
python -m venv .venv

# Activate virtual environment:
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows Command Prompt:
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate

pip install -e .
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The FastAPI backend will start at:
* **API Server**: [http://localhost:8000](http://localhost:8000)
* **Interactive OpenAPI Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

---

### 4. Start the Frontend (Next.js)

Open a new terminal window:

```bash
cd frontend
pnpm install
pnpm dev
```

*(Alternatively, use `npm install && npm run dev`)*

The Next.js application will start at:
* **Frontend Application**: [http://localhost:3000](http://localhost:3000)

### Frontend Routes:
* **Home Portal**: [http://localhost:3000/](http://localhost:3000/)
* **Excel Data Intelligence Agent**: [http://localhost:3000/excel-agent](http://localhost:3000/excel-agent)
* **Notes Summarization Assistant**: [http://localhost:3000/support-assistant](http://localhost:3000/support-assistant)
* **IDP Pipeline & Triage Dashboard**: [http://localhost:3000/idp-pipeline](http://localhost:3000/idp-pipeline)

---

## Running Automated Tests

Run the test suite using `uv run pytest` from the `backend/` directory:

```bash
cd backend
uv run pytest tests/test_idp_pipeline.py
uv run pytest tests/test_support_assistant.py
```

---

## Directory Structure

```
DesiCrew/
├── README.md                      # Unified project documentation
├── backend/
│   ├── .env                       # Environment variables (private)
│   ├── .env.example               # Environment template
│   ├── pyproject.toml             # Python dependencies and metadata
│   ├── main.py                    # Central FastAPI application & router mounting
│   ├── data/                      # Sample datasets, uploads & vector index cache
│   │   ├── data.xlsx              # Default Excel dataset for analysis
│   │   ├── idp_uploads/           # Uploaded documents and crop previews
│   │   ├── support_kb/            # Serialized vector embeddings (embeddings.npz)
│   │   └── idp_triage_audit.json  # Human audit persistence log
│   ├── excel_agent/               # Question 1: LangGraph Excel Intelligence Agent
│   │   ├── agent.py               # StateGraph orchestration & reflection loop
│   │   ├── prompts.py             # Reasoner, Act, and Critic prompt templates
│   │   ├── server.py              # FastAPI endpoints for chat, uploads, and metadata
│   │   └── tools/                 # Execution, inspection, and calculator tools
│   ├── support_assistant/         # Question 2: Notes Summarization & RAG
│   │   ├── agent.py               # Conversational agent with anti-repetition memory
│   │   ├── server.py              # Knowledge base endpoints and chat routing
│   │   ├── tools.py               # Filesystem exploration and doc readers
│   │   └── vector_store.py        # FastEmbed embedding generator and search index
│   ├── idp_pipeline/              # Question 3: Intelligent Document Processing
│   │   ├── classifier.py          # Document type classifier (rules + LLM)
│   │   ├── extractor.py           # RapidOCR extraction & field parsing
│   │   ├── router.py              # Upload, sample preview, and triage endpoints
│   │   ├── schemas.py             # Pydantic schemas & DocumentType definitions
│   │   └── validators.py          # Checksum & regex validators (Verhoeff, PAN, etc.)
│   └── tests/                     # Pytest suite
└── frontend/
    ├── package.json               # Next.js scripts and dependencies
    ├── app/
    │   ├── page.tsx               # Main landing portal
    │   ├── excel-agent/           # UI for Excel Agent (Chat, Trajectory, Data viewer)
    │   ├── support-assistant/     # UI for Support Assistant (Chat, KB stats, Re-index)
    │   └── idp-pipeline/          # UI for IDP Pipeline (Upload, Triage, Confidence audit)
    └── lib/
        └── db.ts                  # Client-side IndexedDB session storage
```

---

## Troubleshooting & FAQ

1. **`AI_TOKEN is not set` Error**:
   * Verify that `backend/.env` exists and contains a valid token for `AI_TOKEN`.
   * Ensure `AI_BASE_URL` is accessible from your network.

2. **CORS or Connection Refused from Frontend**:
   * Make sure the FastAPI backend is running on `http://localhost:8000`.
   * Check that port 8000 is not blocked by another process.

3. **First-Time Model Downloads**:
   * On initial run, `rapidocr_onnxruntime` and `fastembed` will download lightweight ONNX model weights (`~15-70 MB`). Ensure you have an active internet connection for the first run.

4. **Vector Store Reindexing**:
   * You can trigger a complete re-index of the knowledge base at any time from the Support Assistant UI via the **Re-index Knowledge Base** button or by calling `POST /api/support-assistant/reindex`.
