"""Statistical analysis tools for the Excel Data Intelligence Agent."""

from typing import Any

import pandas as pd
from langchain_core.tools import tool

from .excel_read import load_clean_sheet


def _get_numeric_series(
    file_path: str,
    column_name: str,
    sheet_name: str | None = None,
) -> tuple[pd.Series | None, dict[str, Any] | None]:
    """Helper to load and validate numeric data from an Excel column."""
    try:
        df = load_clean_sheet(file_path, sheet_name=sheet_name)
        if column_name not in df.columns:
            return None, {
                "error": f"Column '{column_name}' not found. Available columns: {list(df.columns)}"
            }

        numeric_series = pd.to_numeric(df[column_name], errors="coerce").dropna()
        if numeric_series.empty:
            return None, {
                "error": f"Column '{column_name}' contains no valid numeric values for statistical calculation."
            }
        return numeric_series, None
    except Exception as e:
        return None, {"error": f"Failed to load column data: {e!s}"}


@tool
def calculate_mean(
    file_path: str,
    column_name: str,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    """Calculate the arithmetic mean (average) of a numerical column in the Excel sheet.

    Args:
        file_path: Path to the Excel file.
        column_name: Name of the numerical column to average.
        sheet_name: Optional sheet name (defaults to first sheet).
    """
    series, err = _get_numeric_series(file_path, column_name, sheet_name)
    if err:
        return err

    assert series is not None
    val = float(series.mean())
    return {
        "metric": "mean",
        "column_name": column_name,
        "count": int(len(series)),
        "mean": round(val, 4),
    }


@tool
def calculate_median(
    file_path: str,
    column_name: str,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    """Calculate the median (50th percentile) of a numerical column in the Excel sheet.

    Args:
        file_path: Path to the Excel file.
        column_name: Name of the numerical column.
        sheet_name: Optional sheet name (defaults to first sheet).
    """
    series, err = _get_numeric_series(file_path, column_name, sheet_name)
    if err:
        return err

    assert series is not None
    val = float(series.median())
    return {
        "metric": "median",
        "column_name": column_name,
        "count": int(len(series)),
        "median": round(val, 4),
    }


@tool
def calculate_mode(
    file_path: str,
    column_name: str,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    """Calculate the mode (most frequently occurring value(s)) of a column in the Excel sheet.

    Works for both numerical and categorical/text columns.

    Args:
        file_path: Path to the Excel file.
        column_name: Name of the column.
        sheet_name: Optional sheet name (defaults to first sheet).
    """
    try:
        df = load_clean_sheet(file_path, sheet_name=sheet_name)
        if column_name not in df.columns:
            return {
                "error": f"Column '{column_name}' not found. Available columns: {list(df.columns)}"
            }

        series = df[column_name].dropna()
        if series.empty:
            return {"error": f"Column '{column_name}' contains no non-empty values."}

        modes = series.mode().tolist()
        counts = series.value_counts()
        frequency = int(counts.iloc[0]) if not counts.empty else 0

        return {
            "metric": "mode",
            "column_name": column_name,
            "total_evaluated": int(len(series)),
            "modes": modes[:5],
            "frequency_of_mode": frequency,
        }
    except Exception as e:
        return {"error": f"Failed to calculate mode: {e!s}"}


@tool
def calculate_quantiles(
    file_path: str,
    column_name: str,
    quantiles: list[float] | None = None,
    sheet_name: str | None = None,
) -> dict[str, Any]:
    """Calculate quantiles / percentiles for a numerical column.

    Defaults to quartiles [0.25, 0.50, 0.75] if quantiles list is not provided.

    Args:
        file_path: Path to the Excel file.
        column_name: Name of the numerical column.
        quantiles: Optional list of float quantiles between 0.0 and 1.0 (e.g. [0.25, 0.5, 0.75]).
        sheet_name: Optional sheet name (defaults to first sheet).
    """
    series, err = _get_numeric_series(file_path, column_name, sheet_name)
    if err:
        return err

    assert series is not None
    q_list = quantiles or [0.25, 0.50, 0.75]
    for q in q_list:
        if not (0.0 <= q <= 1.0):
            return {"error": f"Invalid quantile value {q}. Quantiles must be between 0.0 and 1.0."}

    results: dict[str, float] = {}
    for q in q_list:
        pct_label = f"p{int(q * 100)}" if q * 100 == int(q * 100) else f"q_{q}"
        results[pct_label] = round(float(series.quantile(q)), 4)

    return {
        "metric": "quantiles",
        "column_name": column_name,
        "count": int(len(series)),
        "quantiles_evaluated": q_list,
        "values": results,
    }
