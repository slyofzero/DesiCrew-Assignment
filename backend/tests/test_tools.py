from pathlib import Path

from excel_agent.tools import execute_python, search_definitions

# Path to the dataset
fp = str(Path(__file__).resolve().parent.parent / "data" / "data.xlsx")

# 1. Test execute_python on real data
code_test = (
    "total_sold = df['Number of Units Sold'].sum()\n"
    "top_item = df.sort_values(by='Number of Units Sold', ascending=False).iloc[0]\n"
    "print(f'Total Sold: {total_sold}, Top Product: {top_item[\"Product Name\"]} ({top_item[\"Number of Units Sold\"]} units)')"
)
exec_res = execute_python.invoke({"code": code_test, "file_path": fp})
print("=== 1. Execute Python Success ===")
print("Status:", exec_res.get("status"))
print("Stdout:", exec_res.get("stdout"))

# 2. Test execute_python error reflection structure
code_err = "df['Invalid_Column'].mean()"
err_res = execute_python.invoke({"code": code_err, "file_path": fp})
print("\n=== 2. Execute Python Self-Reflection Feedback ===")
print("Status:", err_res.get("status"))
print("Error type:", err_res.get("error_type"))
print("Available cols for reflection:", err_res.get("available_columns")[:3])

# 3. Test search_definitions
s_res = search_definitions.invoke({"query": "inventory turnover ratio"})
print("\n=== 3. Search Definitions Success ===")
print("Source:", s_res.get("source"))
print("Title:", s_res.get("title"))
print("Summary:", str(s_res.get("summary"))[:120] + "...")

# 4. Test Statistics Tools
from excel_agent.tools import (  # noqa: E402
    calculate_mean,
    calculate_median,
    calculate_mode,
    calculate_quantiles,
)

mean_res = calculate_mean.invoke({"file_path": fp, "column_name": "Opening Stock"})
median_res = calculate_median.invoke({"file_path": fp, "column_name": "Opening Stock"})
mode_res = calculate_mode.invoke({"file_path": fp, "column_name": "Opening Stock"})
quantiles_res = calculate_quantiles.invoke({"file_path": fp, "column_name": "Opening Stock"})

print("\n=== 4. Statistics Tools Verification ===")
print("Mean:", mean_res)
print("Median:", median_res)
print("Mode:", mode_res)
print("Quantiles:", quantiles_res)

