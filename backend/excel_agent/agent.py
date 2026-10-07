import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

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
from typing_extensions import TypedDict

try:
    from excel_agent.llm import get_llm
    from excel_agent.prompts import (
        ACT_SYSTEM_PROMPT,
        ACT_USER_PROMPT_TEMPLATE,
        CRITIC_PROMPT_TEMPLATE,
        CRITIC_REJECTION_MESSAGE_TEMPLATE,
        REASONER_PROMPT_TEMPLATE,
        SYNTHESIZE_PROMPT_TEMPLATE,
    )
    from excel_agent.tools import ALL_TOOLS
except ImportError:
    from llm import get_llm
    from prompts import (
        ACT_SYSTEM_PROMPT,
        ACT_USER_PROMPT_TEMPLATE,
        CRITIC_PROMPT_TEMPLATE,
        CRITIC_REJECTION_MESSAGE_TEMPLATE,
        REASONER_PROMPT_TEMPLATE,
        SYNTHESIZE_PROMPT_TEMPLATE,
    )
    from tools import ALL_TOOLS


# -------------------------------------------------------------
# Structured Output Schemas for Reasoning and Reflection
# -------------------------------------------------------------
class ReasoningStep(BaseModel):
    """Structured output for the Reasoning Engine."""
    thought: str = Field(description="Step-by-step reasoning about what data subset, filter, or calculation is needed.")
    is_aligned: bool = Field(
        default=True,
        description="True if query is relevant to Excel spreadsheets, data analysis, calculations, inventory, or tool capabilities. False if the query is general small talk, poetry, unrelated trivia, or out of scope."
    )
    output_as_table: bool = Field(
        default=False,
        description="True if the fetched or analyzed data consists of multiple items, rows, comparisons, or multi-field records that would be clearer presented as a markdown table. False if the answer is a single scalar number, yes/no, short statement, or greeting."
    )
    decision: Literal["call_tool", "finish"] = Field(
        description="'call_tool' if we need to execute a tool (e.g. python, metadata, search). 'finish' if answer is ready or query is out-of-scope."
    )
    tool_instruction: str | None = Field(
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
    messages: Annotated[list[BaseMessage], add_messages]
    file_path: str
    query: str
    current_thought: str
    current_decision: str
    output_as_table: bool
    tool_instruction: str
    thoughts: list[str]
    trajectory: list[dict[str, Any]]
    reflection_notes: list[str]
    retry_count: int
    steps_taken: int
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
    def reason_node(state: AgentState) -> dict[str, Any]:
        """Analyze current state, formulate a thought, and decide the next step."""
        file_path = state.get("file_path", "")
        query = state.get("query", "")
        thoughts_history = state.get("thoughts", [])
        reflections = state.get("reflection_notes", [])
        steps_taken = state.get("steps_taken", 0)

        # Format observations gathered in CURRENT turn (exclude previous chat turns)
        obs_history = []
        prior_turns: list[str] = []

        all_msgs = state.get("messages", [])
        # Find prior conversational turns (HumanMessage and AIMessage before current question)
        for msg in all_msgs:
            if isinstance(msg, HumanMessage) and msg.content != query:
                prior_turns.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                prior_turns.append(f"Assistant: {msg.content}")
            elif isinstance(msg, ToolMessage):
                obs_history.append(f"Observation from {msg.name}:\n{msg.content!s}")
            elif isinstance(msg, AIMessage) and msg.tool_calls:
                tc_names = [f"{tc['name']}({tc['args']})" for tc in msg.tool_calls]
                obs_history.append(f"Action taken: {', '.join(tc_names)}")

        conversation_history = "\n".join(prior_turns) if prior_turns else "No prior conversation history."
        history_summary = "\n\n".join(obs_history) if obs_history else "No actions taken yet."
        reflections_summary = (
            "Previous Critic Reflections/Corrections:\n" + "\n".join(reflections)
            if reflections
            else "None."
        )

        # Circuit breaker: If we have already taken 5 tool steps, force finish to reflect
        step: ReasoningStep
        if steps_taken >= 6:
            step = ReasoningStep(
                thought="Sufficient observations gathered over multiple steps. Proceeding to reflection.",
                decision="finish",
                tool_instruction=None,
            )
        else:
            prompt = REASONER_PROMPT_TEMPLATE.format(
                file_path=file_path,
                query=query,
                conversation_history=conversation_history,
                history_summary=history_summary,
                reflections_summary=reflections_summary,
            )
            raw_step = reasoner_llm.invoke(prompt)
            if isinstance(raw_step, ReasoningStep):
                step = raw_step
            elif isinstance(raw_step, dict):
                step = ReasoningStep(**raw_step)
            else:
                step = ReasoningStep(
                    thought=str(getattr(raw_step, "thought", "Proceeding with analysis.")),
                    is_aligned=getattr(raw_step, "is_aligned", True),
                    decision=getattr(raw_step, "decision", "finish"),
                    tool_instruction=getattr(raw_step, "tool_instruction", None),
                )

        # If the query is out of scope / not aligned, force finish immediately without calling tools
        if not step.is_aligned:
            step.decision = "finish"
            step.tool_instruction = None

        thought_repr = f"THOUGHT: {step.thought} | ALIGNED: {step.is_aligned} | TABLE: {step.output_as_table} | DECISION: {step.decision}"
        updated_thoughts = list(thoughts_history)
        updated_thoughts.append(thought_repr)

        # Append to trajectory
        traj = list(state.get("trajectory", []))
        traj.append({
            "type": "reason",
            "thought": step.thought,
            "is_aligned": step.is_aligned,
            "output_as_table": step.output_as_table,
            "decision": step.decision,
            "tool_instruction": step.tool_instruction or "",
        })

        return {
            "current_thought": step.thought,
            "current_decision": step.decision,
            "output_as_table": step.output_as_table,
            "tool_instruction": step.tool_instruction or "",
            "thoughts": updated_thoughts,
            "trajectory": traj,
            "steps_taken": steps_taken + 1,
        }

    # -------------------------------------------------------------
    # 2. ACT NODE: Emits the tool call grounded by the thought
    # -------------------------------------------------------------
    def act_node(state: AgentState) -> dict[str, Any]:
        """Convert the current thought & instruction into an actionable tool call."""
        file_path = state.get("file_path", "")
        thought = state.get("current_thought", "")
        instruction = state.get("tool_instruction", "")

        act_prompt = [
            SystemMessage(content=ACT_SYSTEM_PROMPT.format(file_path=file_path)),
            HumanMessage(
                content=ACT_USER_PROMPT_TEMPLATE.format(
                    thought=thought,
                    instruction=instruction,
                )
            ),
        ]

        response = llm_with_tools.invoke(act_prompt)

        # Append action and tool call intent to trajectory
        traj = list(state.get("trajectory", []))
        action_calls = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                action_calls.append({
                    "tool": tc["name"],
                    "args": tc.get("args", {}),
                    "id": tc.get("id", ""),
                })
        traj.append({
            "type": "act",
            "action_calls": action_calls,
        })

        return {
            "messages": [response],
            "trajectory": traj,
        }

    # -------------------------------------------------------------
    # 3. TOOL EXECUTION NODE
    # -------------------------------------------------------------
    tool_node = ToolNode(ALL_TOOLS)

    # -------------------------------------------------------------
    # 4. REFLECTION NODE: Critic & Self-Reflection Verifier
    # -------------------------------------------------------------
    def reflect_node(state: AgentState) -> dict[str, Any]:
        """Critique the findings against the original query to ensure no shortcuts or hallucinations."""
        query = state.get("query", "")
        retries = state.get("retry_count", 0)

        # Pair each tool action with its output so the critic can verify filtering code
        observations = []
        last_action = ""
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tc_strs = [f"{tc['name']}({tc.get('args', {})})" for tc in msg.tool_calls]
                last_action = "Action: " + ", ".join(tc_strs)
            elif isinstance(msg, ToolMessage):
                prefix = f"[{last_action}] " if last_action else ""
                observations.append(f"{prefix}Observation from {msg.name} -> {msg.content}")

        obs_str = "\n".join(observations) if observations else "No observations gathered."

        critic_prompt = CRITIC_PROMPT_TEMPLATE.format(query=query, obs_str=obs_str)
        raw_critic = critic_llm.invoke(critic_prompt)
        if isinstance(raw_critic, CriticVerdict):
            critic = raw_critic
        elif isinstance(raw_critic, dict):
            critic = CriticVerdict(**raw_critic)
        else:
            critic = CriticVerdict(
                verdict=getattr(raw_critic, "verdict", "approved"),
                critique=getattr(raw_critic, "critique", "Observations verified."),
            )

        is_approved = critic.verdict == "approved"
        curr_reflections = list(state.get("reflection_notes", []))
        critique_summary = f"VERDICT: {critic.verdict.upper()}\nCRITIQUE: {critic.critique}"
        curr_reflections.append(critique_summary)

        if not is_approved and retries < 3:
            correction_msg = CRITIC_REJECTION_MESSAGE_TEMPLATE.format(critique=critic.critique)
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
    # 5. SYNTHESIZE NODE: Output Summarizer of verified response
    # -------------------------------------------------------------
    def synthesize_node(state: AgentState) -> dict[str, Any]:
        """Summarize everything deemed as the correct response by reflection."""
        query = state.get("query", "")
        reflections = state.get("reflection_notes", [])
        reflection_critique = "\n".join(reflections) if reflections else "All findings verified and approved."

        observations = []
        last_action = ""
        for msg in state["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tc_strs = [f"{tc['name']}({tc.get('args', {})})" for tc in msg.tool_calls]
                last_action = "Action: " + ", ".join(tc_strs)
            elif isinstance(msg, ToolMessage):
                prefix = f"[{last_action}] " if last_action else ""
                observations.append(f"{prefix}{msg.name}: {msg.content}")

        obs_str = "\n".join(observations)
        output_as_table = state.get("output_as_table", False)
        table_instruction = (
            "TABLE FORMATTING REQUIRED: The reasoning engine determined that the fetched data should be formatted as a Markdown table. You MUST present the items, records, or multi-field data using clean Markdown table syntax (| Col 1 | Col 2 | ... |) with a clear header row."
            if output_as_table
            else "Do NOT format as a table unless strictly requested; present as clear concise text or bullet points."
        )

        prompt = SYNTHESIZE_PROMPT_TEMPLATE.format(
            query=query,
            reflection_critique=reflection_critique,
            obs_str=obs_str,
            table_instruction=table_instruction,
        )
        response = llm_pure.invoke(prompt)
        raw_text = response.content if isinstance(response.content, str) else str(response.content)
        answer_text = re.sub(
            r"^(?:\*{0,2}(?:Final\s+(?:Response|Answer)|Summary)\s*[:\-]?\*{0,2}\s*[:\-]?\s*)+",
            "",
            raw_text.strip(),
            flags=re.IGNORECASE,
        ).strip()

        # Append final_output to trajectory
        traj = list(state.get("trajectory", []))
        traj.append({
            "type": "final_output",
            "content": answer_text,
            "output_as_table": output_as_table,
        })

        return {
            "final_answer": answer_text,
            "trajectory": traj,
            "messages": [AIMessage(content=answer_text)],
        }

    # -------------------------------------------------------------
    # Conditional Routing Logic
    # -------------------------------------------------------------
    def route_after_reason(state: AgentState) -> str:
        """Route based on structured decision: call_tool -> act, finish -> synthesize/reflect.
        Dependent purely on the Reason node's output.
        """
        decision = state.get("current_decision", "call_tool")

        if decision == "finish":
            # If tools were already called, reflect and verify before finishing;
            # if no tools were called (e.g. general query, greeting, conversational), jump straight to synthesize
            has_tool_observations = any(isinstance(m, ToolMessage) for m in state.get("messages", []))
            if has_tool_observations:
                return "reflect"
            return "synthesize"

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
    workflow.add_conditional_edges(
        "reason",
        route_after_reason,
        {"act": "act", "reflect": "reflect", "synthesize": "synthesize"},
    )
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

    def run(self, file_path: str, query: str, chat_history: list[BaseMessage] | None = None) -> dict[str, Any]:
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

        initial_messages: list[BaseMessage] = list(chat_history or [])
        initial_messages.append(HumanMessage(content=query))

        initial_state: AgentState = {
            "messages": initial_messages,
            "file_path": str(file_path),
            "query": query,
            "current_thought": "",
            "current_decision": "call_tool",
            "tool_instruction": "",
            "thoughts": [],
            "trajectory": [],
            "reflection_notes": [],
            "retry_count": 0,
            "steps_taken": 0,
            "is_satisfied": False,
            "final_answer": "",
        }

        final_state = self.graph.invoke(initial_state)
        answer = final_state.get("final_answer") or final_state["messages"][-1].content

        # Reconstruct structured agent trajectory: reason -> act -> tool_trace -> reason -> ... -> final_output
        messages = final_state.get("messages", [])
        tool_outputs_map: dict[str, dict[str, Any]] = {}
        for m in messages:
            if isinstance(m, ToolMessage):
                tool_outputs_map[str(m.tool_call_id)] = {
                    "tool": m.name,
                    "output": m.content,
                }

        formatted_trajectory: list[dict[str, Any]] = []
        raw_traj = final_state.get("trajectory", [])
        for step in raw_traj:
            stype = step.get("type")
            if stype == "reason":
                formatted_trajectory.append({
                    "step": "reason",
                    "thought": step.get("thought", ""),
                    "is_aligned": step.get("is_aligned", True),
                    "output_as_table": step.get("output_as_table", False),
                    "decision": step.get("decision", ""),
                    "instruction": step.get("tool_instruction", ""),
                })
            elif stype == "act":
                action_calls = step.get("action_calls", [])
                formatted_trajectory.append({
                    "step": "act",
                    "calls": action_calls,
                })
                # Immediately follow with the tool_trace for those actions
                for ac in action_calls:
                    cid = str(ac.get("id", ""))
                    t_out = tool_outputs_map.get(cid)
                    formatted_trajectory.append({
                        "step": "tool_trace",
                        "tool": ac.get("tool"),
                        "args": ac.get("args"),
                        "output": t_out["output"] if t_out else None,
                    })
            elif stype == "final_output":
                formatted_trajectory.append({
                    "step": "final_output",
                    "content": step.get("content", ""),
                    "output_as_table": step.get("output_as_table", False),
                })

        return {
            "answer": answer,
            "trajectory": formatted_trajectory,
            "thoughts": final_state.get("thoughts", []),
            "messages": final_state["messages"],
            "reflection_notes": final_state.get("reflection_notes", []),
            "retry_count": final_state.get("retry_count", 0),
        }

    def stream(
        self,
        file_path: str,
        query: str,
        chat_history: list[BaseMessage] | None = None,
    ):
        """Stream the execution graph yielding step events in real-time."""
        if file_path.endswith("?") or (not file_path.endswith((".xlsx", ".xls")) and query.endswith((".xlsx", ".xls"))):
            file_path, query = query, file_path

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

        initial_messages: list[BaseMessage] = list(chat_history or [])
        initial_messages.append(HumanMessage(content=query))

        initial_state: AgentState = {
            "messages": initial_messages,
            "file_path": str(file_path),
            "query": query,
            "current_thought": "",
            "current_decision": "call_tool",
            "tool_instruction": "",
            "thoughts": [],
            "trajectory": [],
            "reflection_notes": [],
            "retry_count": 0,
            "steps_taken": 0,
            "is_satisfied": False,
            "final_answer": "",
        }

        yield from self.graph.stream(initial_state)

    @staticmethod
    def print_trace(reply: dict[str, Any]) -> None:
        """Print the complete trace of Reasoning, Actions, and Reflections in chronological order."""
        print("\n" + "=" * 25 + " AGENT TRAJECTORY " + "=" * 25)

        traj = reply.get("trajectory", [])
        if traj:
            for item in traj:
                step_type = item.get("step")
                if step_type == "reason":
                    print(f"\n[REASON]: {item.get('thought')}")
                    print(f"  Decision: {item.get('decision')} | Instruction: {item.get('instruction')}")
                elif step_type == "act":
                    calls = item.get("calls", [])
                    c_desc = [f"{c['tool']}({c.get('args', {})})" for c in calls]
                    print(f"\n[ACT]: {', '.join(c_desc)}")
                elif step_type == "tool_trace":
                    print(f"\n[TOOL_TRACE]: {item.get('tool')}")
                    print(f"  Args: {json.dumps(item.get('args'), indent=2)}")
                    print(f"  Output: {str(item.get('output'))[:300]}")
                elif step_type == "final_output":
                    print(f"\n[FINAL_OUTPUT]:\n{item.get('content')}")
        else:
            print("No trajectory recorded.")

        reflections = reply.get("reflection_notes", [])
        if reflections:
            print("\n[CRITIC REFLECTIONS]:")
            for r in reflections:
                print(f"  {r.strip()}")

        print("\n" + "=" * 70 + "\n")
