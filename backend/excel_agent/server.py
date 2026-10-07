import json
import shutil
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

try:
    from excel_agent.agent import ExcelAgent
    from excel_agent.tools import load_clean_sheet, read_metadata, resolve_file_path
except ImportError:
    from agent import ExcelAgent
    from tools import load_clean_sheet, read_metadata, resolve_file_path

# Dedicated router for Excel Agent endpoints
router = APIRouter(prefix="/excel-agent", tags=["Excel Agent"])

# Default data path
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DATASET = DATA_DIR / "data.xlsx"

# Shared agent instance
agent = ExcelAgent()


class HistoryTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    query: str = Field(..., description="User query about the Excel dataset")
    file_path: str | None = Field(default=None, description="Optional custom file path to Excel workbook")
    history: list[HistoryTurn] = Field(default_factory=list, description="Prior conversation turns for context")


class ChatResponse(BaseModel):
    answer: str
    thoughts: list[str] = Field(default_factory=list)
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    tool_trace: list[dict[str, Any]]
    reflection_notes: list[str]
    retry_count: int


@router.get("/metadata")
def get_metadata(file_path: str | None = Query(default=None)):
    print(file_path)
    """Fetch sheet names, columns, shapes, and sample rows for the dataset."""
    target_path = file_path or str(DEFAULT_DATASET)
    meta = read_metadata.invoke({"file_path": target_path})
    if isinstance(meta, dict) and "error" in meta:
        raise HTTPException(status_code=404, detail=str(meta["error"]))
    return meta


@router.get("/data")
def get_sheet_data(
    sheet_name: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    file_path: str | None = Query(default=None),
):
    """Fetch raw tabular rows from a specific sheet for data-grid rendering."""
    target_path = file_path or str(DEFAULT_DATASET)
    try:
        resolved = resolve_file_path(target_path)
        df = load_clean_sheet(str(resolved), sheet_name=sheet_name)
        return {
            "sheet_name": sheet_name or "Default",
            "total_rows": len(df),
            "columns": list(df.columns),
            "records": df.head(limit).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a new Excel workbook to the backend data directory."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = file.filename or "uploaded_dataset.xlsx"
    destination = DATA_DIR / filename
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "status": "success",
            "filename": filename,
            "saved_path": str(destination),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {e!s}")


@router.post("/chat", response_model=ChatResponse)
def chat_with_excel_agent(request: ChatRequest):
    """Send a query to the Excel Agent and receive structured answers with tool execution trace."""
    target_path = request.file_path or str(DEFAULT_DATASET)

    print(request.query)

    try:
        chat_history: list[BaseMessage] = []
        for turn in request.history:
            if turn.role == "user" and turn.content:
                chat_history.append(HumanMessage(content=turn.content))
            elif turn.role == "assistant" and turn.content:
                chat_history.append(AIMessage(content=turn.content))

        reply = agent.run(target_path, request.query, chat_history=chat_history)

        # Extract structured tool trace from trajectory messages
        trace: list[dict[str, Any]] = []
        messages = reply.get("messages", [])

        # Match tool calls with their corresponding ToolMessage outputs
        tool_call_map: dict[str, dict[str, Any]] = {}

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    tc_id = str(tc.get("id") or tc["name"])
                    entry = {
                        "tool": tc["name"],
                        "args": tc.get("args", {}),
                        "output": None,
                    }
                    tool_call_map[tc_id] = entry
                    trace.append(entry)
            elif isinstance(msg, ToolMessage):
                tc_id = str(msg.tool_call_id)
                if tc_id in tool_call_map:
                    tool_call_map[tc_id]["output"] = str(msg.content)
                else:
                    trace.append({
                        "tool": msg.name,
                        "args": {},
                        "output": str(msg.content),
                    })

        return ChatResponse(
            answer=reply.get("answer", ""),
            thoughts=reply.get("thoughts", []),
            trajectory=reply.get("trajectory", []),
            tool_trace=trace,
            reflection_notes=reply.get("reflection_notes", []),
            retry_count=reply.get("retry_count", 0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {e!s}")


@router.post("/chat/stream")
async def chat_stream_with_excel_agent(request: ChatRequest):
    """Stream agent reasoning, actions, tool traces, reflections, and final answer in real-time via SSE."""
    target_path = request.file_path or str(DEFAULT_DATASET)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Yield initial status event
            yield f"data: {json.dumps({'type': 'start', 'query': request.query})}\n\n"

            chat_history: list[BaseMessage] = []
            for turn in request.history:
                if turn.role == "user" and turn.content:
                    chat_history.append(HumanMessage(content=turn.content))
                elif turn.role == "assistant" and turn.content:
                    chat_history.append(AIMessage(content=turn.content))

            for step in agent.stream(target_path, request.query, chat_history=chat_history):
                for node_name, output in step.items():
                    if node_name == "reason":
                        traj_step = output.get("trajectory", [{}])[-1]
                        yield f"data: {json.dumps({'type': 'reason', 'thought': output.get('current_thought'), 'is_aligned': traj_step.get('is_aligned', True), 'output_as_table': traj_step.get('output_as_table', False), 'decision': output.get('current_decision'), 'instruction': output.get('tool_instruction')})}\n\n"

                    elif node_name == "act":
                        action_calls = []
                        if "messages" in output:
                            for m in output["messages"]:
                                if isinstance(m, AIMessage) and m.tool_calls:
                                    for tc in m.tool_calls:
                                        action_calls.append({
                                            "tool": tc["name"],
                                            "args": tc.get("args", {}),
                                            "id": str(tc.get("id", "")),
                                        })
                        yield f"data: {json.dumps({'type': 'act', 'calls': action_calls})}\n\n"

                    elif node_name == "tools":
                        tool_outputs = []
                        if "messages" in output:
                            for m in output["messages"]:
                                if isinstance(m, ToolMessage):
                                    tool_outputs.append({
                                        "tool": m.name,
                                        "id": str(m.tool_call_id),
                                        "output": str(m.content),
                                    })
                        yield f"data: {json.dumps({'type': 'tool_trace', 'outputs': tool_outputs})}\n\n"

                    elif node_name == "reflect":
                        notes = output.get("reflection_notes", [])
                        latest_note = notes[-1] if notes else ""
                        yield f"data: {json.dumps({'type': 'reflect', 'critique': latest_note, 'is_satisfied': output.get('is_satisfied', True)})}\n\n"

                    elif node_name == "synthesize":
                        final_ans = output.get("final_answer", "")
                        traj_step = output.get("trajectory", [{}])[-1]
                        yield f"data: {json.dumps({'type': 'final_output', 'answer': final_ans, 'output_as_table': traj_step.get('output_as_table', False)})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

