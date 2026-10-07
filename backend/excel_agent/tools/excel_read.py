from pathlib import Path
from typing import Any

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
    sheet_name: str | None = None,
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
def read_metadata(file_path: str) -> dict[str, Any]:
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

    sheets_info: dict[str, Any] = {}
    for s_name in sheet_names:
        try:
            df = load_clean_sheet(str(path), sheet_name=s_name)
            sheets_info[s_name] = {
                "rows": len(df),
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
def get_cells_by_indices(
    file_path: str,
    row_indices: list[int] | None = None,
    columns: list[str | int] | None = None,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    """Explicitly access cells based upon specific row indices and/or column indices/names.

    Either row_indices, columns, or both must be provided.
    If both are omitted, raises an error.

    Args:
        file_path: Path to the Excel file.
        row_indices: Optional list of integer row indices (0-indexed, e.g. [16, 17, 18, 45]).
        columns: Optional list of column names (e.g. ['Product Name', 'Cost']) or 0-indexed column integer indices (e.g. [1, 4]).
        sheet_name: Optional sheet name (defaults to first sheet).
    """
    if row_indices is None and columns is None:
        return {
            "error": "Invalid request: At least one of 'row_indices' or 'columns' must be provided. Both cannot be missing."
        }

    try:
        df = load_clean_sheet(file_path, sheet_name=sheet_name)

        # Resolve columns (names or integer indices)
        resolved_cols: list[str] = list(df.columns)
        if columns is not None:
            resolved_cols = []
            for col in columns:
                if isinstance(col, int):
                    if 0 <= col < len(df.columns):
                        resolved_cols.append(df.columns[col])
                    else:
                        return {"error": f"Column index {col} is out of bounds (0 to {len(df.columns) - 1})."}
                elif isinstance(col, str):
                    if col in df.columns:
                        resolved_cols.append(col)
                    else:
                        return {"error": f"Column '{col}' not found. Available columns: {list(df.columns)}"}
                else:
                    return {"error": f"Invalid column identifier: {col}"}

        # Resolve rows
        if row_indices is not None:
            invalid_rows = [r for r in row_indices if not (0 <= r < len(df))]
            if invalid_rows:
                return {
                    "error": f"Row indices {invalid_rows} are out of bounds (dataset has {len(df)} rows, 0 to {len(df) - 1})."
                }
            sub_df = df.iloc[row_indices][resolved_cols]
        else:
            sub_df = df[resolved_cols]

        return {
            "requested_rows": row_indices if row_indices is not None else "all",
            "requested_columns": resolved_cols,
            "total_cells_returned": int(sub_df.size),
            "rows_returned": len(sub_df),
            "records": sub_df.to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": f"Failed to get cells by indices: {e!s}"}


@tool
def get_cell_range(
    file_path: str,
    start_row: int | None = None,
    end_row: int | None = None,
    start_col: str | int | None = None,
    end_col: str | int | None = None,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    """Access a rectangular block / slice of cells based on start and end bounds for rows and/or columns.

    Either row bounds (start_row/end_row), column bounds (start_col/end_col), or both must be provided.
    If all row and column bounds are omitted, raises an error.

    Pagination Example:
        This tool can be used to paginate through large datasets in windowed chunks:
        - Page 1: `start_row=0`, `end_row=19` (first 20 rows)
        - Page 2: `start_row=20`, `end_row=39` (next 20 rows)
        - Page 3: `start_row=40`, `end_row=59`
        You can also restrict to specific columns during pagination, e.g.
        `start_row=0, end_row=19, start_col='Product Name', end_col='Number of Units Sold'`.

    Args:
        file_path: Path to the Excel file.
        start_row: Optional 0-indexed start row (inclusive).
        end_row: Optional 0-indexed end row (inclusive).
        start_col: Optional start column name or 0-indexed column integer (inclusive).
        end_col: Optional end column name or 0-indexed column integer (inclusive).
        sheet_name: Optional sheet name (defaults to first sheet).
    """
    has_row_bounds = start_row is not None or end_row is not None
    has_col_bounds = start_col is not None or end_col is not None

    if not has_row_bounds and not has_col_bounds:
        return {
            "error": "Invalid request: At least row bounds ('start_row'/'end_row') or column bounds ('start_col'/'end_col') must be provided. Both cannot be missing."
        }

    try:
        df = load_clean_sheet(file_path, sheet_name=sheet_name)
        total_rows = len(df)
        total_cols = len(df.columns)

        # 1. Resolve row range
        r_start = 0 if start_row is None else max(0, start_row)
        r_end = total_rows - 1 if end_row is None else min(total_rows - 1, end_row)
        if r_start > r_end:
            return {"error": f"start_row ({r_start}) cannot be greater than end_row ({r_end})."}

        # 2. Resolve column range
        col_list = list(df.columns)

        def resolve_col_idx(c_val: str | int | None, default: int) -> int:
            if c_val is None:
                return default
            if isinstance(c_val, int):
                return max(0, min(total_cols - 1, c_val))
            if isinstance(c_val, str):
                if c_val in col_list:
                    return col_list.index(c_val)
                raise ValueError(f"Column '{c_val}' not found in sheet. Available: {col_list}")
            return default

        try:
            c_start = resolve_col_idx(start_col, 0)
            c_end = resolve_col_idx(end_col, total_cols - 1)
        except ValueError as ve:
            return {"error": str(ve)}

        if c_start > c_end:
            return {"error": f"start_col ({start_col}) index cannot be greater than end_col ({end_col}) index."}

        # Slice DataFrame (loc/iloc)
        sub_df = df.iloc[r_start : r_end + 1, c_start : c_end + 1]

        return {
            "row_range": {"start_row": r_start, "end_row": r_end, "count": len(sub_df)},
            "col_range": {
                "start_col": col_list[c_start],
                "end_col": col_list[c_end],
                "columns": col_list[c_start : c_end + 1],
            },
            "total_cells_returned": int(sub_df.size),
            "records": sub_df.to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": f"Failed to get cell range: {e!s}"}


@tool
def filter_rows_by_string(
    file_path: str,
    column_name: str,
    search_value: str,
    sheet_name: str | None = None,
    case_sensitive: bool = False,
    exact_match: bool = False,
) -> dict[str, Any]:
    """Filter row indices and matched records based upon a string match for a specific column.

    Useful to find exact row positions, indices, and product subsets matching keywords (e.g. 'gaming', 'laptop').

    Args:
        file_path: Path to the Excel file.
        column_name: Name of the column to search in (e.g., 'Product Name').
        search_value: The text substring or exact string to look for.
        sheet_name: Optional sheet name (defaults to first sheet).
        case_sensitive: Whether match should be case-sensitive (default False).
        exact_match: If True, matches full string; if False, matches substring (default False).
    """
    try:
        df = load_clean_sheet(file_path, sheet_name=sheet_name)
        if column_name not in df.columns:
            return {
                "error": f"Column '{column_name}' not found. Available columns: {list(df.columns)}"
            }

        series = df[column_name].astype(str)
        if exact_match:
            if case_sensitive:
                mask = series == search_value
            else:
                mask = series.str.lower() == search_value.lower()
        else:
            mask = series.str.contains(search_value, case=case_sensitive, na=False)
            # If literal phrase returned 0 matches and search_value has multiple words, try matching key tokens
            if not mask.any() and " " in search_value:
                stopwords = {"items", "item", "product", "products", "goods", "all", "the", "a", "an"}
                tokens = [t.strip() for t in search_value.split() if t.strip().lower() not in stopwords]
                if tokens:
                    token_regex = "|".join(tokens)
                    mask = series.str.contains(token_regex, case=case_sensitive, na=False)

        matched_indices = df.index[mask].tolist()
        matched_df = df.loc[matched_indices]

        return {
            "search_column": column_name,
            "search_value": search_value,
            "match_count": len(matched_indices),
            "matched_row_indices": matched_indices,
            "sample_matched_records": matched_df.head(15).to_dict(orient="records"),
        }
    except Exception as e:
        return {"error": f"Failed to filter rows: {e!s}"}


@tool
def get_unique_values(
    file_path: str,
    column_name: str,
    sheet_name: str | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    """Retrieve distinct / unique values and their frequencies from a specific column.

    Useful for category exploration, distinct item inspection, and verifying available choices.

    Args:
        file_path: Path to the Excel file.
        column_name: Name of the column to extract distinct values from.
        sheet_name: Optional sheet name (defaults to first sheet).
        limit: Maximum number of distinct values to return (default 50).
    """
    try:
        df = load_clean_sheet(file_path, sheet_name=sheet_name)
        if column_name not in df.columns:
            return {
                "error": f"Column '{column_name}' not found. Available columns: {list(df.columns)}"
            }

        series = df[column_name].dropna()
        counts = series.value_counts()
        total_unique = len(counts)
        unique_list = counts.index[:limit].tolist()

        return {
            "column_name": column_name,
            "total_rows_evaluated": len(df),
            "total_unique_values": total_unique,
            "returned_count": len(unique_list),
            "unique_values": unique_list,
            "value_counts": {str(k): int(v) for k, v in counts.head(limit).items()},
        }
    except Exception as e:
        return {"error": f"Failed to get unique values: {e!s}"}
