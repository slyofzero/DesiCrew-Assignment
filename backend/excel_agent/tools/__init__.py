"""Tools package for the Excel Data Intelligence Agent.

Organized into:
  - excel_read.py: Schema inspection & cell/range getters (read_metadata, get_cells_by_indices, get_cell_range, filter_rows_by_string).
  - excel_edit.py: Modification & execution tools (create_sheet, edit_sheet, execute_python, search_definitions).
  - calculator.py: Mathematical calculation tools.
"""

from .calculator import (
    add,
    calculate,
    divide,
    multiply,
    subtract,
)
from .excel_edit import (
    create_sheet,
    edit_sheet,
    execute_python,
    search_definitions,
)
from .excel_read import (
    clean_column_name,
    filter_rows_by_string,
    get_cell_range,
    get_cells_by_indices,
    get_unique_values,
    load_clean_sheet,
    read_metadata,
    resolve_file_path,
)
from .statistics import (
    calculate_mean,
    calculate_median,
    calculate_mode,
    calculate_quantiles,
)

ALL_TOOLS = [
    read_metadata,
    get_unique_values,
    get_cells_by_indices,
    get_cell_range,
    filter_rows_by_string,
    calculate_mean,
    calculate_median,
    calculate_mode,
    calculate_quantiles,
    execute_python,
    search_definitions,
    create_sheet,
    edit_sheet,
    calculate,
    add,
    subtract,
    multiply,
    divide,
]

__all__ = [
    "ALL_TOOLS",
    "add",
    "calculate",
    "calculate_mean",
    "calculate_median",
    "calculate_mode",
    "calculate_quantiles",
    "clean_column_name",
    "create_sheet",
    "divide",
    "edit_sheet",
    "execute_python",
    "filter_rows_by_string",
    "get_cell_range",
    "get_cells_by_indices",
    "get_unique_values",
    "load_clean_sheet",
    "multiply",
    "read_metadata",
    "resolve_file_path",
    "search_definitions",
    "subtract",
]

