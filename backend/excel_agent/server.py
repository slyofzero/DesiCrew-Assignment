from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from langchain_core.messages import AIMessage, ToolMessage
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


# Request / Response Models
class ChatRequest(BaseModel):
    query: str = Field(..., description="User query about the Excel dataset")
    file_path: Optional[str] = Field(default=None, description="Optional custom file path to Excel workbook")


class ChatResponse(BaseModel):
    answer: str
    thoughts: List[str] = Field(default_factory=list)
    tool_trace: List[Dict[str, Any]]
    reflection_notes: List[str]
    retry_count: int


@router.get("/metadata")
def get_metadata(file_path: Optional[str] = Query(default=None)):
    print(file_path)
    """Fetch sheet names, columns, shapes, and sample rows for the dataset."""
    target_path = file_path or str(DEFAULT_DATASET)
    meta = read_metadata.invoke({"file_path": target_path})
    if isinstance(meta, dict) and "error" in meta:
        raise HTTPException(status_code=404, detail=str(meta["error"]))
    return meta


@router.get("/data")
def get_sheet_data(
    sheet_name: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    file_path: Optional[str] = Query(default=None),
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
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@router.post("/chat", response_model=ChatResponse)
def chat_with_excel_agent(request: ChatRequest):
    """Send a query to the Excel Agent and receive structured answers with tool execution trace."""
    target_path = request.file_path or str(DEFAULT_DATASET)

    print(request.query)

    try:
        reply = agent.run(target_path, request.query)

        # Extract structured tool trace from trajectory messages
        trace: List[Dict[str, Any]] = []
        messages = reply.get("messages", [])

        # Match tool calls with their corresponding ToolMessage outputs
        tool_call_map: Dict[str, Dict[str, Any]] = {}

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
            tool_trace=trace,
            reflection_notes=reply.get("reflection_notes", []),
            retry_count=reply.get("retry_count", 0),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
