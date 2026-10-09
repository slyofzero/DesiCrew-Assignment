from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from excel_agent.server import router as excel_agent_router
from idp_pipeline.router import router as heuristic_idp_router
from support_assistant.server import router as support_assistant_router
from agentic_idp_pipeline.router import router as agentic_idp_router

import os
from dotenv import load_dotenv

load_dotenv()

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
