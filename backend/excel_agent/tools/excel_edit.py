import ast
import contextlib
import io
import json
import re
import traceback
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from .excel_read import load_clean_sheet, resolve_file_path


@tool
def create_sheet(
    file_path: str,
    sheet_name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
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
        return {"error": f"Failed to create sheet: {e!s}"}


@tool
def edit_sheet(
    file_path: str,
    sheet_name: str,
    column_name: str,
    values: list[Any],
) -> dict[str, Any]:
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
        return {"error": f"Failed to edit sheet: {e!s}"}


@tool
def execute_python(
    code: str,
    file_path: str | None = None,
    sheet_name: str | None = None,
) -> dict[str, Any]:
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
    df_obj: pd.DataFrame | None = None

    if file_path:
        try:
            df_obj = load_clean_sheet(file_path, sheet_name=sheet_name)
        except Exception as e:
            return {
                "status": "error",
                "error_type": "DataLoadError",
                "error_message": f"Failed to load dataset: {e!s}",
            }

    local_env: dict[str, Any] = {
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
                elif parsed.body and isinstance(parsed.body[-1], ast.Assign):
                    last_assign = parsed.body[-1]
                    exec(compile(parsed, "<agent_code>", "exec"), local_env)
                    if last_assign.targets and isinstance(last_assign.targets[0], ast.Name):
                        var_name = last_assign.targets[0].id
                        if var_name in local_env:
                            print(local_env[var_name])
                else:
                    exec(compile(parsed, "<agent_code>", "exec"), local_env)
            except Exception:
                exec(code, local_env)

        captured = stdout_buf.getvalue().strip()
        return {
            "status": "success",
            "stdout": captured,
            "has_output": bool(captured),
        }

    except Exception as e:
        available_cols = list(df_obj.columns) if df_obj is not None else []
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(),
            "available_columns": available_cols,
        }


@tool
def search_definitions(query: str) -> dict[str, Any]:
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
