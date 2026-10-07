import ast
import contextlib
import io
import json
from pathlib import Path
import re
import traceback
from typing import Any, Dict, List, Optional
import urllib.parse
import urllib.request

import numpy as np
import openpyxl
import pandas as pd
from langchain_core.tools import tool


def clean_column_name(col: Any) -> str:
    """Normalize messy Excel column headers (e.g. remove embedded newlines)."""
    return " ".join(str(col).replace("/", " / ").split())


def resolve_file_path(file_path: str) -> Path:
    """Resolve file path against multiple fallback candidates (CWD, module parent, data dir)."""
    p = Path(file_path)
    if p.exists():
        return p
    for cand in [
        p.resolve(),
        (Path.cwd() / p).resolve(),
        (Path(__file__).resolve().parent / p).resolve(),
        (Path(__file__).resolve().parent.parent / p).resolve(),
        (Path.cwd() / "data" / p.name).resolve(),
        (Path(__file__).resolve().parent.parent / "data" / p.name).resolve(),
    ]:
        if cand.exists():
            return cand
    return p


def load_clean_sheet(
    file_path: str,
    sheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load an Excel sheet and automatically detect the header row and clean column names."""
    path = resolve_file_path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Excel file not found at: {path}")

    # Read raw data to find the actual header row
    raw_read = pd.read_excel(path, sheet_name=sheet_name or 0, header=None)
    if not isinstance(raw_read, pd.DataFrame):
        raise ValueError("Expected a single DataFrame from Excel file.")
    df_raw: pd.DataFrame = raw_read

    # Detect header row: first row where at least 3 cells are non-empty strings
    header_idx = 0
    for idx, row in df_raw.iterrows():
        non_empty = [c for c in row if pd.notna(c) and str(c).strip()]
        if len(non_empty) >= 3 and any("ID" in str(c) or "Product" in str(c) for c in non_empty):
            header_idx = int(str(idx))
            break

    # Re-read with proper header row
    data_read = pd.read_excel(path, sheet_name=sheet_name or 0, skiprows=header_idx)
    if not isinstance(data_read, pd.DataFrame):
        raise ValueError("Expected a single DataFrame from Excel file.")
    df: pd.DataFrame = data_read

    # Drop entirely empty columns
    df = df.dropna(how="all", axis=1)
    # Clean column headers
    df.columns = [clean_column_name(c) for c in df.columns]
    # Drop rows that are entirely NaN
    df = df.dropna(how="all").reset_index(drop=True)
    return df


@tool
def read_metadata(file_path: str) -> Dict[str, Any]:
    """Inspect the Excel workbook and return sheet names, column names, shapes, and sample rows.

    Args:
        file_path: Path to the Excel file.
    """
    path = resolve_file_path(file_path)

    if not path.exists():
        return {"error": f"File not found at: {path}"}

    wb = openpyxl.load_workbook(path, read_only=True)
    sheet_names = wb.sheetnames
    wb.close()

    sheets_info: Dict[str, Any] = {}
    for s_name in sheet_names:
        try:
            df = load_clean_sheet(str(path), sheet_name=s_name)
            sheets_info[s_name] = {
                "rows": int(len(df)),
                "columns": list(df.columns),
                "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
                "sample_rows": df.head(3).to_dict(orient="records"),
            }
        except Exception as e:
            sheets_info[s_name] = {"error": str(e)}

    return {
        "file_name": path.name,
        "sheets": sheet_names,
        "details": sheets_info,
    }


@tool
def read_data(
    file_path: str,
    sheet_name: Optional[str] = None,
    columns: Optional[List[str]] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Read records from an Excel sheet with optional column filtering and row limit.

    Args:
        file_path: Path to the Excel file.
        sheet_name: Name of the sheet to read. Defaults to the first sheet.
        columns: Optional list of column names to select.
        limit: Maximum number of rows to return (default 10).
    """
    try:
        df = load_clean_sheet(file_path, sheet_name=sheet_name)
        if columns:
            available = [c for c in columns if c in df.columns]
            if not available:
                return {
                    "error": f"None of the requested columns {columns} exist. Available columns: {list(df.columns)}"
                }
            df = df[available]

        records = df.head(limit).to_dict(orient="records")
        return {
            "total_rows": int(len(df)),
            "returned_rows": len(records),
            "columns": list(df.columns),
            "data": records,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def create_sheet(
    file_path: str,
    sheet_name: str,
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Create a new sheet in the Excel workbook with provided records and save changes.

    Args:
        file_path: Target Excel file path.
        sheet_name: Name for the new worksheet.
        records: List of dictionaries representing the rows to insert.
    """
    path = resolve_file_path(file_path)
    if not path.exists():
        return {"error": f"File not found at: {path}"}
    if not records:
        return {"error": "Cannot create sheet with empty records list."}

    try:
        new_df = pd.DataFrame(records)
        with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            new_df.to_excel(writer, sheet_name=sheet_name, index=False)

        return {
            "status": "success",
            "message": f"Sheet '{sheet_name}' created successfully with {len(new_df)} rows.",
            "columns": list(new_df.columns),
            "file": str(path),
        }
    except Exception as e:
        return {"error": f"Failed to create sheet: {str(e)}"}


@tool
def edit_sheet(
    file_path: str,
    sheet_name: str,
    column_name: str,
    values: List[Any],
) -> Dict[str, Any]:
    """Add or update a column in an existing sheet in the Excel workbook.

    Args:
        file_path: Excel file path.
        sheet_name: Target sheet name.
        column_name: The name of the column to add or update.
        values: List of values for the column matching the row count.
    """
    try:
        df = load_clean_sheet(file_path, sheet_name=sheet_name)
        if len(values) != len(df):
            return {
                "error": f"Length of values ({len(values)}) does not match sheet row count ({len(df)})."
            }

        df[column_name] = values
        path = resolve_file_path(file_path)

        with pd.ExcelWriter(path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        return {
            "status": "success",
            "message": f"Column '{column_name}' updated in sheet '{sheet_name}'.",
            "row_count": len(df),
            "columns": list(df.columns),
        }
    except Exception as e:
        return {"error": f"Failed to edit sheet: {str(e)}"}


@tool
def execute_python(
    code: str,
    file_path: Optional[str] = None,
    sheet_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute Python code against the dataset.

    The execution environment contains:
      - `pd`: pandas library
      - `np`: numpy library
      - `df`: Cleaned pandas DataFrame loaded from file_path (if provided)

    Captured outputs (print statements or return values) and any errors/tracebacks are returned.

    Args:
        code: Python code string to execute.
        file_path: Optional path to the Excel file to pre-populate `df`.
        sheet_name: Optional sheet name to load into `df`.
    """
    stdout_buf = io.StringIO()
    df_obj: Optional[pd.DataFrame] = None

    if file_path:
        try:
            df_obj = load_clean_sheet(file_path, sheet_name=sheet_name)
        except Exception as e:
            return {
                "status": "error",
                "error_type": "DataLoadError",
                "error_message": f"Failed to load dataset: {str(e)}",
            }

    local_env: Dict[str, Any] = {
        "pd": pd,
        "np": np,
        "df": df_obj,
    }

    try:
        with contextlib.redirect_stdout(stdout_buf):
            try:
                parsed = ast.parse(code.strip())
                if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                    last_expr = parsed.body.pop()
                    if parsed.body:
                        exec(compile(parsed, "<agent_code>", "exec"), local_env)
                    val = eval(compile(ast.Expression(last_expr.value), "<agent_code>", "eval"), local_env)
                    if val is not None:
                        print(val)
                else:
                    exec(compile(parsed, "<agent_code>", "exec"), local_env)
            except Exception:
                # Fallback to standard exec
                exec(code, local_env)

        captured = stdout_buf.getvalue().strip()
        return {
            "status": "success",
            "stdout": captured,
            "has_output": bool(captured),
        }

    except Exception as e:
        # Return structured error details for agent self-reflection & repair
        available_cols = list(df_obj.columns) if df_obj is not None else []
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "available_columns": available_cols,
        }


@tool
def search_definitions(query: str) -> Dict[str, Any]:
    """Look up definitions, formulas, or business metrics context on the web.

    Useful for financial/inventory concepts (e.g. GMROI, Days of Inventory on Hand, Reorder Point).

    Args:
        query: The business term or formula to search for.
    """
    clean_query = query.strip()
    encoded = urllib.parse.quote(clean_query)

    # 1. Try DuckDuckGo Instant Answer API
    try:
        url_ddg = f"https://api.duckduckgo.com/?q={encoded}&format=json"
        req = urllib.request.Request(url_ddg, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            abstract = data.get("AbstractText") or data.get("Abstract")
            if abstract:
                return {
                    "query": clean_query,
                    "source": "DuckDuckGo Instant Answer",
                    "title": data.get("Heading", clean_query),
                    "summary": abstract,
                }
            # Check related topics
            for topic in data.get("RelatedTopics", []):
                if isinstance(topic, dict) and topic.get("Text"):
                    return {
                        "query": clean_query,
                        "source": "DuckDuckGo Related Topics",
                        "title": clean_query,
                        "summary": topic.get("Text"),
                    }
    except Exception:
        pass

    # 2. Try Wikipedia Search API
    try:
        url_wiki = (
            f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded}&utf8=&format=json"
        )
        req_w = urllib.request.Request(url_wiki, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req_w, timeout=5) as resp:
            data_w = json.loads(resp.read().decode("utf-8"))
            search_items = data_w.get("query", {}).get("search", [])
            if search_items:
                top = search_items[0]
                # Strip raw HTML tags from Wikipedia snippet
                clean_snippet = re.sub(r"<[^<]+?>", "", top.get("snippet", ""))
                return {
                    "query": clean_query,
                    "source": "Wikipedia",
                    "title": top.get("title", clean_query),
                    "summary": clean_snippet,
                }
    except Exception:
        pass

    return {
        "query": clean_query,
        "source": "None",
        "title": clean_query,
        "summary": f"No online definition found for '{clean_query}'.",
    }
