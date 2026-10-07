from pathlib import Path
from excel_agent.agent import ExcelAgent

# Resolve path to dataset
dataset_path = str(Path(__file__).resolve().parent.parent / "data" / "data.xlsx")
query = "How many gaming items were sold in total?"

print("==================== QUERY ====================")
print(query)
print(f"Dataset: {dataset_path}\n")

agent = ExcelAgent()
reply = agent.run(dataset_path, query)

print("==================== FINAL ANSWER ====================")
print(reply["answer"])

# Print full trace showing Reason -> Act -> Tools -> Reason -> Reflect
agent.print_trace(reply)
