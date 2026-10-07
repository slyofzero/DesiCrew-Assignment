import time
from pathlib import Path

from excel_agent.agent import AgentState, build_agent_graph

print("Building graph...", flush=True)
graph = build_agent_graph()
print("Graph built successfully!", flush=True)

dataset_path = str(Path("data/data.xlsx").resolve())
query = "How many gaming items were sold in total?"

initial_state: AgentState = {
    "messages": [],
    "file_path": dataset_path,
    "query": query,
    "current_thought": "",
    "current_decision": "call_tool",
    "tool_instruction": "",
    "thoughts": [],
    "reflection_notes": [],
    "retry_count": 0,
    "steps_taken": 0,
    "is_satisfied": False,
    "final_answer": "",
}

print(f"Starting query: {query}", flush=True)
t0 = time.time()
for step in graph.stream(initial_state):
    for node_name, output in step.items():
        elapsed = time.time() - t0
        print(f"\n[+{elapsed:.1f}s] === NODE: {node_name} ===", flush=True)
        if "current_thought" in output:
            print(f"  Thought: {output.get('current_thought')}", flush=True)
            print(f"  Decision: {output.get('current_decision')}", flush=True)
            print(f"  Tool instruction: {output.get('tool_instruction')}", flush=True)
        if "messages" in output:
            for m in output["messages"]:
                print(f"  Msg: {type(m).__name__}", flush=True)
                if hasattr(m, "tool_calls") and m.tool_calls:
                    print(f"  Tool calls: {m.tool_calls}", flush=True)
                elif hasattr(m, "content"):
                    print(f"  Content: {str(m.content)[:200]}", flush=True)
        if "is_satisfied" in output:
            print(f"  is_satisfied: {output.get('is_satisfied')}", flush=True)
            print(f"  retry_count: {output.get('retry_count')}", flush=True)
        if "final_answer" in output:
            print(f"\n[FINAL ANSWER]:\n{output['final_answer']}", flush=True)
