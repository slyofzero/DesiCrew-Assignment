import json
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

try:
    from excel_agent.llm import get_llm
    from excel_agent.tools import ALL_TOOLS
except ImportError:
    from llm import get_llm
    from tools import ALL_TOOLS


# -------------------------------------------------------------
# Structured Output Schemas for Reasoning and Reflection
# -------------------------------------------------------------
class ReasoningStep(BaseModel):
    """Structured output for the Reasoning Engine."""
    thought: str = Field(description="Step-by-step reasoning about what data subset, filter, or calculation is needed.")
    decision: Literal["call_tool", "finish"] = Field(
        description="'call_tool' if we need to execute a tool (e.g. python, metadata, search). 'finish' ONLY if all necessary observations are gathered."
    )
    tool_instruction: Optional[str] = Field(
        default=None,
        description="Clear instructions for the action executor if decision is 'call_tool'."
    )


class CriticVerdict(BaseModel):
    """Structured output for the Self-Reflection Verifier Critic."""
    verdict: Literal["approved", "rejected"] = Field(
        description="'approved' if observations accurately answer the query with proper filtering. 'rejected' if wrong, unfiltered, or incomplete."
    )
    critique: str = Field(description="Explanation of why approved or what needs correction.")


class AgentState(TypedDict):
    """LangGraph state schema with explicit Reason -> Act -> Reflect loop."""
    messages: Annotated[List[BaseMessage], add_messages]
    file_path: str
    query: str
    current_thought: str
    current_decision: str
    tool_instruction: str
    thoughts: List[str]
    reflection_notes: List[str]
    retry_count: int
    is_satisfied: bool
    final_answer: str


def build_agent_graph(model: str = "openai/gpt-4o-mini"):
    """Build and compile the LangGraph agent state graph with explicit Reason -> Act -> Reflect loop."""
    llm_pure = get_llm(model=model, temperature=0.0)
    reasoner_llm = llm_pure.with_structured_output(ReasoningStep)
    critic_llm = llm_pure.with_structured_output(CriticVerdict)
    llm_with_tools = get_llm(model=model, temperature=0.0).bind_tools(ALL_TOOLS, tool_choice="required")

    # -------------------------------------------------------------
    # 1. REASON NODE: Pure cognitive reasoning (without tools)
    # -------------------------------------------------------------
    def reason_node(state: AgentState) -> Dict[str, Any]:
        """Analyze current state, formulate a thought, and decide the next step."""
        file_path = state.get("file_path", "")
        query = state.get("query", "")
        thoughts_history = state.get("thoughts", [])
        reflections = state.get("reflection_notes", [])

        # Format observations gathered so far
        obs_history = []
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                obs_history.append(f"Observation from {msg.name}:\n{str(msg.content)[:1500]}")
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                tc_names = [tc["name"] for tc in msg.tool_calls]
                obs_history.append(f"Action taken: called {', '.join(tc_names)}")

        history_summary = "\n\n".join(obs_history) if obs_history else "No actions taken yet."
        reflections_summary = (
            f"Previous Critic Reflections/Corrections:\n" + "\n".join(reflections)
            if reflections
            else "None."
        )

        prompt = f"""You are the Reasoning Engine of an Excel Data Intelligence Agent.
Active Excel Dataset: {file_path}
User Query: "{query}"

Observations Gathered So Far:
{history_summary}

{reflections_summary}

Your task:
Analyze what information is currently available and what is missing to answer the user query accurately.
CRITICAL CONSTRAINT RULES:
- If the user asks about a specific product or category (e.g., 'gaming items', 'laptops', 'accessories'), you MUST filter `df['Product Name']`.
- NEVER run an unfiltered table-wide calculation (like `df['col'].sum()`) if a specific category was requested.
- If observations already contain the verified answer, set decision='finish'.
- If we still need to inspect metadata or execute python code, set decision='call_tool'.
"""
        step: ReasoningStep = reasoner_llm.invoke(prompt)  # type: ignore

        thought_repr = f"THOUGHT: {step.thought} | DECISION: {step.decision}"
        updated_thoughts = list(thoughts_history)
        updated_thoughts.append(thought_repr)

        return {
            "current_thought": step.thought,
            "current_decision": step.decision,
            "tool_instruction": step.tool_instruction or "",
            "thoughts": updated_thoughts,
        }

    # -------------------------------------------------------------
    # 2. ACT NODE: Emits the tool call grounded by the thought
    # -------------------------------------------------------------
    def act_node(state: AgentState) -> Dict[str, Any]:
        """Convert the current thought & instruction into an actionable tool call."""
        file_path = state.get("file_path", "")
        thought = state.get("current_thought", "")
        instruction = state.get("tool_instruction", "")

        act_prompt = [
            SystemMessage(
                content=(
                    f"You are the Action Executor for an Excel agent. Active file: {file_path}.\n"
                    f"You have access to: execute_python, search_definitions, calculate, read_metadata, read_data, create_sheet, edit_sheet.\n"
                    f"You MUST call a tool that executes the required instruction.\n"
                    f"Always pass file_path='{file_path}' when invoking data tools."
                )
            ),
            HumanMessage(
                content=(
                    f"Reasoning thought:\n{thought}\n\n"
                    f"Tool instruction:\n{instruction}\n\n"
                    f"Execute the tool call:"
                )
            )
        ]

        response = llm_with_tools.invoke(act_prompt)
        return {"messages": [response]}

    # -------------------------------------------------------------
    # 3. TOOL EXECUTION NODE
    # -------------------------------------------------------------
    tool_node = ToolNode(ALL_TOOLS)

    # -------------------------------------------------------------
    # 4. REFLECTION NODE: Critic & Self-Reflection Verifier
    # -------------------------------------------------------------
    def reflect_node(state: AgentState) -> Dict[str, Any]:
        """Critique the findings against the original query to ensure no shortcuts or hallucinations."""
        query = state.get("query", "")
        retries = state.get("retry_count", 0)

        observations = []
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                observations.append(f"Tool {msg.name} -> {msg.content}")

        obs_str = "\n".join(observations) if observations else "No observations gathered."

        critic_prompt = f"""You are the Self-Reflection Verifier Critic for an Excel Data Agent.
Original User Query: "{query}"

Observations Gathered:
{obs_str}

Evaluate if the gathered observations strictly answer the user's specific request:
1. Did the tools filter for the requested entity or category (e.g., 'gaming items'), or did they accidentally aggregate the entire dataset?
2. Are the figures grounded and verified in the tool outputs?
3. Did the agent gather enough data to answer?

Set verdict='approved' ONLY if observations properly answer the query. Set verdict='rejected' if data is missing, unfiltered, or incorrect.
"""
        critic: CriticVerdict = critic_llm.invoke(critic_prompt)  # type: ignore

        is_approved = critic.verdict == "approved"
        curr_reflections = list(state.get("reflection_notes", []))
        critique_summary = f"VERDICT: {critic.verdict.upper()}\nCRITIQUE: {critic.critique}"
        curr_reflections.append(critique_summary)

        if not is_approved and retries < 3:
            correction_msg = (
                f"[Self-Reflection Rejection]: The critic identified an issue with the findings:\n"
                f"{critic.critique}\n"
                f"Please reason about this critique and execute the corrected query."
            )
            return {
                "messages": [HumanMessage(content=correction_msg)],
                "reflection_notes": curr_reflections,
                "retry_count": retries + 1,
                "is_satisfied": False,
            }
        else:
            return {
                "reflection_notes": curr_reflections,
                "is_satisfied": True,
            }

    # -------------------------------------------------------------
    # 5. SYNTHESIZE NODE: Generates final verified answer
    # -------------------------------------------------------------
    def synthesize_node(state: AgentState) -> Dict[str, Any]:
        """Generate final plain-English answer grounded in verified observations."""
        query = state.get("query", "")
        observations = []
        for msg in state["messages"]:
            if isinstance(msg, ToolMessage):
                observations.append(f"{msg.name}: {msg.content}")

        obs_str = "\n".join(observations)

        prompt = f"""You are an Excel Data Intelligence Assistant.
User Query: "{query}"

Verified Tool Observations:
{obs_str}

Provide a clear, helpful, and concise response in plain English.
If specific products or items were counted or calculated, explicitly list them so the user has full clarity.
"""
        response = llm_pure.invoke(prompt)
        answer_text = response.content if isinstance(response.content, str) else str(response.content)

        return {
            "final_answer": answer_text,
            "messages": [AIMessage(content=answer_text)],
        }

    # -------------------------------------------------------------
    # Conditional Routing Logic
    # -------------------------------------------------------------
    def route_after_reason(state: AgentState) -> str:
        """Route based on structured decision: call_tool -> act, finish -> reflect."""
        decision = state.get("current_decision", "call_tool")
        if decision == "finish":
            return "reflect"
        return "act"

    def route_after_act(state: AgentState) -> str:
        """Route to tools if tool_calls emitted, else reflect."""
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            return "tools"
        return "reflect"

    def route_after_reflect(state: AgentState) -> str:
        """Loop back to reason if reflection rejected and retries remaining, else synthesize."""
        if not state.get("is_satisfied", False) and state.get("retry_count", 0) <= 3:
            return "reason"
        return "synthesize"

    # Assemble Graph: Reason -> Act -> Tools -> Reason -> ... -> Reflect -> (Loop if needed) -> Synthesize
    workflow = StateGraph(AgentState)
    workflow.add_node("reason", reason_node)
    workflow.add_node("act", act_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("synthesize", synthesize_node)

    workflow.add_edge(START, "reason")
    workflow.add_conditional_edges("reason", route_after_reason, {"act": "act", "reflect": "reflect"})
    workflow.add_conditional_edges("act", route_after_act, {"tools": "tools", "reflect": "reflect"})
    workflow.add_edge("tools", "reason")  # Loop back: Reason -> Act -> Tools -> Reason -> ...
    workflow.add_conditional_edges("reflect", route_after_reflect, {"reason": "reason", "synthesize": "synthesize"})
    workflow.add_edge("synthesize", END)

    return workflow.compile()


class ExcelAgent:
    """Orchestrator wrapping the explicit Reason -> Act -> Reflect LangGraph architecture."""

    def __init__(self, model: str = "openai/gpt-4o-mini"):
        self.model = model
        self.graph = build_agent_graph(model=model)

    def run(self, file_path: str, query: str, chat_history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        """Run a query through the LangGraph agent."""
        # Detect and swap if query and file_path were passed in reverse order
        if file_path.endswith("?") or (not file_path.endswith((".xlsx", ".xls")) and query.endswith((".xlsx", ".xls"))):
            file_path, query = query, file_path

        # Resolve relative paths robustly against CWD, script dir, and parent dir
        p = Path(file_path)
        if not p.is_absolute() or not p.exists():
            candidates = [
                p.resolve(),
                (Path.cwd() / p).resolve(),
                (Path(__file__).resolve().parent / p).resolve(),
                (Path(__file__).resolve().parent.parent / p).resolve(),
                (Path.cwd() / "data" / p.name).resolve(),
                (Path(__file__).resolve().parent.parent / "data" / p.name).resolve(),
            ]
            for cand in candidates:
                if cand.exists():
                    file_path = str(cand)
                    break

        initial_messages: List[BaseMessage] = list(chat_history or [])
        initial_messages.append(HumanMessage(content=query))

        initial_state: AgentState = {
            "messages": initial_messages,
            "file_path": str(file_path),
            "query": query,
            "current_thought": "",
            "current_decision": "call_tool",
            "tool_instruction": "",
            "thoughts": [],
            "reflection_notes": [],
            "retry_count": 0,
            "is_satisfied": False,
            "final_answer": "",
        }

        final_state = self.graph.invoke(initial_state)
        answer = final_state.get("final_answer") or final_state["messages"][-1].content

        return {
            "answer": answer,
            "thoughts": final_state.get("thoughts", []),
            "messages": final_state["messages"],
            "reflection_notes": final_state.get("reflection_notes", []),
            "retry_count": final_state.get("retry_count", 0),
        }

    @staticmethod
    def print_trace(reply: Dict[str, Any]) -> None:
        """Print the complete trace of Reasoning, Actions, and Reflections."""
        print("\n" + "=" * 25 + " REASON -> ACT -> REFLECT TRACE " + "=" * 25)

        # Print reasoning thoughts
        thoughts = reply.get("thoughts", [])
        if thoughts:
            print("\n[REASONING THOUGHTS]:")
            for i, th in enumerate(thoughts, 1):
                clean_th = th.strip().replace("\n", " ")
                print(f"  Step {i}: {clean_th}")

        # Print tool calls & results
        messages = reply.get("messages", [])
        has_tools = False
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                has_tools = True
                for tc in msg.tool_calls:
                    print(f"\n[ACTION: Tool Call] -> {tc['name']}")
                    print(f"  Args: {json.dumps(tc['args'], indent=2)}")
            elif isinstance(msg, ToolMessage):
                print(f"\n[OBSERVATION: Tool Output] <- {msg.name}")
                print(f"  Output: {str(msg.content)[:250]}")

        # Print reflections
        reflections = reply.get("reflection_notes", [])
        if reflections:
            print("\n[CRITIC REFLECTIONS]:")
            for r in reflections:
                print(f"  {r.strip()}")

        if not has_tools and not thoughts:
            print("No tool calls were made.")
        print("\n" + "=" * 74 + "\n")
