from backend.excel_agent.tools.excel_tools import execute_python, search_definitions
from pathlib import Path

# Path to the Question 1 dataset
fp = str(Path(__file__).resolve().parent.parent.parent / "Question 1" / "Inventory-Records-Sample-Data.xlsx")

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
