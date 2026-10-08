import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

try:
    from support_assistant.agent import SupportAssistantAgent
    from support_assistant.vector_store import vector_store
except ImportError:
    from agent import SupportAssistantAgent
    from vector_store import vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/support-assistant", tags=["Support Assistant"])

agent = SupportAssistantAgent()


class HistoryTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query about the technical knowledge base")
    history: list[HistoryTurn] = Field(default_factory=list, description="Prior conversation turns for multi-turn context")


class ChatResponse(BaseModel):
    answer: str
    thoughts: list[str] = Field(default_factory=list)
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)


@router.get("/stats")
def get_knowledge_base_stats():
    """Return indexing status, chunks count, and document overview."""
    try:
        return vector_store.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reindex")
def trigger_reindex():
    """Force reindexing of all documents in the knowledge base."""
    try:
        res = vector_store.build_index(force=True)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
def list_documents():
    """List all indexed documents and files."""
    try:
        sources = sorted(list({c.source for c in vector_store.chunks}))
        return {
            "total_documents": len(sources),
            "documents": sources,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    """Execute CoT + ReAct conversation turn with vector search augmentation and anti-repetition memory."""
    try:
        history_dicts = [{"role": h.role, "content": h.content} for h in request.history]
        result = agent.run(query=request.query, history=history_dicts)
        return ChatResponse(
            answer=result["answer"],
            thoughts=result["thoughts"],
            trajectory=result["trajectory"],
            retrieved_chunks=result["retrieved_chunks"],
        )
    except Exception as e:
        logger.error(f"Chat execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent reasoning failed: {e!s}")
