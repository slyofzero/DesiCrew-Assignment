from pathlib import Path

from excel_agent.agent import ExcelAgent

# Resolve path to inventory dataset
dataset_path = str(Path(__file__).resolve().parent.parent / "data" / "data.xlsx")

print(f"Initializing ExcelAgent with dataset: {dataset_path}")
agent = ExcelAgent()

query = (
    "What are the top 3 best-selling products by 'Number of Units Sold', "
    "and what is the total cost value across all inventory ('Cost Price Total (USD)')?"
)

print(f"\nUser Query: {query}")
print("Executing LangGraph agent...")

result = agent.run(query=query, file_path=dataset_path)

print("\n=== Agent Final Answer ===")
print(result["answer"])

print(f"\nReflections triggered: {len(result['reflection_notes'])}")
if result["reflection_notes"]:
    for r in result["reflection_notes"]:
        print(" -", r[:100], "...")
