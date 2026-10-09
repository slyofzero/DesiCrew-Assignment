import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Ensure environment variables are loaded before any subrouters or agents are initialized
backend_env = Path(__file__).resolve().parent / ".env"
root_env = Path(__file__).resolve().parent.parent / ".env"
if backend_env.exists():
    load_dotenv(backend_env)
if root_env.exists():
    load_dotenv(root_env)
load_dotenv(find_dotenv())

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_idp_pipeline.router import router as agentic_idp_router
from excel_agent.server import router as excel_agent_router
from idp_pipeline.router import router as heuristic_idp_router
from support_assistant.server import router as support_assistant_router

# Initialize central FastAPI application
app = FastAPI(
    title="DesiCrew AI Intelligence Backend",
    description="Unified backend API powering the Excel Intelligence Agent, Notes Intelligence & Summarizer, and IDP Pipeline.",
    version="0.1.0",
)

# Enable CORS for Next.js frontend (default port 3000) and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

idp_router = agentic_idp_router if os.getenv("IDP_ROUTER_TYPE") == "agentic" else heuristic_idp_router

# Mount modular routers
app.include_router(excel_agent_router, prefix="/api")
app.include_router(support_assistant_router, prefix="/api")
app.include_router(idp_router, prefix="/api")

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "DesiCrew AI Intelligence Backend",
        "version": "0.1.0",
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
