import json
import logging
import os
import re
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, SecretStr
from typing_extensions import TypedDict

try:
    from support_assistant.prompts import (
        REASONER_USER_PROMPT_TEMPLATE,
        SUPPORT_AGENT_SYSTEM_PROMPT,
        SYNTHESIZER_USER_PROMPT_TEMPLATE,
    )
    from support_assistant.tools import ALL_TOOLS
    from support_assistant.vector_store import vector_store
except ImportError:
    from prompts import (
        REASONER_USER_PROMPT_TEMPLATE,
        SUPPORT_AGENT_SYSTEM_PROMPT,
        SYNTHESIZER_USER_PROMPT_TEMPLATE,
    )
    from tools import ALL_TOOLS
    from vector_store import vector_store

logger = logging.getLogger(__name__)


def get_support_llm(model: str | None = None, temperature: float = 0.0, max_tokens: int = 1500) -> ChatOpenAI:
    """Instantiate a ChatOpenAI instance configured for the support assistant."""
    selected_model = model or os.getenv("SUPPORT_MODEL", "openai/gpt-4o-mini")
    api_key = os.getenv("AI_TOKEN", "")
    base_url = os.getenv("AI_BASE_URL", "https://aipipe.org/openrouter/v1")
    return ChatOpenAI(
        model=selected_model,
        api_key=SecretStr(api_key),
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# -------------------------------------------------------------
# 1. Structured Chain-of-Thought Reasoning Schema (5 steps)
# -------------------------------------------------------------
class CoTReasoning(BaseModel):
    """Detailed 5-step Chain-of-Thought reasoning before taking any action or answering."""

    thought_1_query_intent: str = Field(
        description="Point 1: Detailed analysis of user's query and intent."
    )
    thought_2_context_grounding: str = Field(
        description="Point 2: Evaluation of whether note chunks are needed or if existing chunks are sufficient."
    )
    thought_3_memory_anti_repetition: str = Field(
        description="Point 3: Session history audit to prevent repeating past explanations."
    )
    thought_4_topic_transition: str = Field(
        description="Point 4: Assessment of topic continuation vs switch."
    )
    thought_5_action_plan: str = Field(
        description="Point 5: Final cognitive deduction explaining the decision."
    )
    decision: Literal["retrieve", "call_tool", "finish", "reject"] = Field(
        description="'retrieve' to query the vector database for notes or fetch more chunks; 'call_tool' to use filesystem tools; 'finish' if ready to answer (or general conversation); 'reject' if completely off-topic."
    )
    search_query: str | None = Field(
        default=None,
        description="Target search query to fetch chunks from vector database if decision is 'retrieve'.",
    )
    tool_instruction: str | None = Field(
        default=None,
        description="Specific instruction for the tool node if decision is 'call_tool'.",
    )


# -------------------------------------------------------------
# 2. Agent State Schema
# -------------------------------------------------------------
class SupportAgentState(TypedDict):
    """State tracked across the Support Assistant reasoning and retrieval lifecycle."""

    query: str
    history: list[dict[str, str]]
    retrieved_chunks: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    cot_reasoning: CoTReasoning | None
    thoughts: list[str]
    trajectory: list[dict[str, Any]]
    retrieval_count: int
    steps_taken: int
    max_steps: int
    final_answer: str


# -------------------------------------------------------------
# 3. LangGraph Flow Implementation
# -------------------------------------------------------------
def build_support_agent_graph(model: str | None = None):
    """Compile the LangGraph agent state graph starting with reasoning and allowing chunk fetch looping."""
    reasoner_llm = get_support_llm(model=model, temperature=0.1, max_tokens=1500)
    llm_with_tools = get_support_llm(model=model, temperature=0.0, max_tokens=1000).bind_tools(ALL_TOOLS)
    synthesizer_llm = get_support_llm(model=model, temperature=0.2, max_tokens=2000)

    tool_map = {t.name: t for t in ALL_TOOLS}

    # 1. REASON NODE (Entrypoint): Pure Cognitive 5-point Chain-of-Thought
    def reason_node(state: SupportAgentState) -> dict[str, Any]:
        query = state["query"]
        history = state.get("history", [])
        chunks = state.get("retrieved_chunks", [])
        observations = state.get("observations", [])

        # Format history string
        hist_text = "\n".join(
            [f"Turn {i + 1} ({t.get('role', 'user')}): {t.get('content', '')}" for i, t in enumerate(history)]
        ) if history else "No prior turns in this session (Turn 1)."

        # Format chunks string
        chunks_text = "\n\n".join(
            [
                f"--- Chunk {i + 1} (Score: {c.get('score', 'N/A')}) ---\n"
                f"Source: {c.get('source', '')}\n"
                f"Section: {c.get('section', '')}\n"
                f"Content:\n{c.get('content', '')}"
                for i, c in enumerate(chunks)
            ]
        ) if chunks else "No chunks retrieved yet (Initial cognitive evaluation)."

        # Format observations string
        obs_text = json.dumps(observations, indent=2) if observations else "None"

        prompt = REASONER_USER_PROMPT_TEMPLATE.format(
            history=hist_text,
            retrieved_chunks=chunks_text,
            observations=obs_text,
            query=query,
        )

        messages = [
            SystemMessage(content=SUPPORT_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        logger.info("[SupportAgent] Invoking entrypoint 5-point CoT Reasoner...")
        raw_res = reasoner_llm.invoke(messages)
        raw_content = raw_res.content if isinstance(raw_res.content, str) else str(raw_res.content)

        parsed_data: dict[str, Any] = {}
        m = re.search(r"\{[\s\S]*\}", raw_content)
        if m:
            try:
                parsed_data = json.loads(m.group(0))
            except Exception:
                pass

        thought_1 = parsed_data.get("thought_1_query_intent", f"Analyze intent for: {query}")
        thought_2 = parsed_data.get(
            "thought_2_context_grounding",
            f"Evaluated note chunks requirement ({len(chunks)} currently loaded).",
        )
        thought_3 = parsed_data.get(
            "thought_3_memory_anti_repetition",
            "Audit conversation history to prevent redundant definitions.",
        )
        thought_4 = parsed_data.get("thought_4_topic_transition", "Ensure seamless topical continuity.")
        thought_5 = parsed_data.get("thought_5_action_plan", "Determine appropriate action.")

        raw_decision = str(parsed_data.get("decision", "")).strip().lower()

        is_greeting = any(
            re.search(rf"\b{g}\b", query.lower())
            for g in ["hi", "hello", "hey", "greetings", "good morning", "good evening", "who are you", "what can you do", "how can you help", "thanks", "thank you"]
        )
        has_tech_terms = any(
            re.search(rf"\b{term}\b", query.lower())
            for term in ["algorithm", "tree", "page", "memory", "cache", "process", "graph", "sort", "notation", "bias", "variance", "neural", "gradient", "svm", "kernel", "deadlock", "scheduling", "avl", "big-o", "omega", "theta"]
        )

        if not chunks and is_greeting and not has_tech_terms:
            decision = "finish"
        elif raw_decision in ["retrieve", "call_tool", "finish", "reject"]:
            decision = raw_decision
        elif not chunks:
            decision = "retrieve"
        else:
            decision = "finish"

        search_query = parsed_data.get("search_query")
        tool_instruction = parsed_data.get("tool_instruction")

        reasoning_res = CoTReasoning(
            thought_1_query_intent=thought_1,
            thought_2_context_grounding=thought_2,
            thought_3_memory_anti_repetition=thought_3,
            thought_4_topic_transition=thought_4,
            thought_5_action_plan=thought_5,
            decision=decision,  # type: ignore
            search_query=search_query,
            tool_instruction=tool_instruction,
        )

        thoughts = [
            f"1. Query Intent: {thought_1}",
            f"2. Context Grounding: {thought_2}",
            f"3. Memory & Anti-Repetition: {thought_3}",
            f"4. Topic Transition: {thought_4}",
            f"5. Action Plan: {thought_5}",
        ]

        step_trace = {
            "step": "reason",
            "thought": "\n\n".join(thoughts),
            "decision": decision,
            "instruction": tool_instruction or (f"Search query: {search_query}" if search_query else ""),
        }

        return {
            "cot_reasoning": reasoning_res,
            "thoughts": state["thoughts"] + thoughts,
            "trajectory": state["trajectory"] + [step_trace],
            "steps_taken": state["steps_taken"] + 1,
        }

    # 2. ROUTER: Decide whether to retrieve chunks, call tools, synthesize, or reject
    def route_reason_decision(state: SupportAgentState) -> Literal["retrieve", "act", "synthesize", "reject"]:
        cot = state.get("cot_reasoning")
        steps = state.get("steps_taken", 0)
        max_steps = state.get("max_steps", 4)
        retrieval_count = state.get("retrieval_count", 0)

        if cot:
            if cot.decision == "reject":
                return "reject"
            if cot.decision == "retrieve" and steps < max_steps and retrieval_count < 3:
                return "retrieve"
            if cot.decision == "call_tool" and steps < max_steps:
                return "act"

        return "synthesize"

    # 3. RETRIEVE NODE: Fetch chunks from vector store and loop back to reason
    def retrieve_node(state: SupportAgentState) -> dict[str, Any]:
        cot = state.get("cot_reasoning")
        target_query = (cot.search_query if cot and cot.search_query else state["query"]).strip()

        logger.info(f"[SupportAgent] Retrieving top chunks for query: '{target_query[:60]}'...")
        new_chunks = vector_store.search(target_query, top_k=5)

        # Deduplicate with existing chunks
        existing_ids = {c.get("chunk_id") for c in state.get("retrieved_chunks", [])}
        deduped = list(state.get("retrieved_chunks", []))
        added_count = 0
        for ch in new_chunks:
            if ch.get("chunk_id") not in existing_ids:
                deduped.append(ch)
                existing_ids.add(ch.get("chunk_id"))
                added_count += 1

        retrieval_trace = {
            "step": "tool_trace",
            "tool": "vector_store_retrieval",
            "args": {"query": target_query, "top_k": 5, "retrieval_turn": state.get("retrieval_count", 0) + 1},
            "output": [
                {
                    "source": c["source"],
                    "section": c["section"],
                    "score": c["score"],
                    "snippet": c["content"][:200] + "...",
                }
                for c in new_chunks
            ],
        }

        return {
            "retrieved_chunks": deduped,
            "retrieval_count": state.get("retrieval_count", 0) + 1,
            "trajectory": state["trajectory"] + [retrieval_trace],
        }

    # 4. ACT NODE: Sparse tool execution (ReAct) and loop back to reason
    def act_node(state: SupportAgentState) -> dict[str, Any]:
        cot = state.get("cot_reasoning")
        instruction = cot.tool_instruction if cot and cot.tool_instruction else "Inspect file system or search notes"

        messages = [
            SystemMessage(content=SUPPORT_AGENT_SYSTEM_PROMPT),
            HumanMessage(content=f"Execute appropriate tool call for instruction: {instruction}"),
        ]

        logger.info(f"[SupportAgent] Running tool call for instruction: {instruction}...")
        ai_msg = llm_with_tools.invoke(messages)

        new_obs = []
        new_traces = []

        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            for tc in ai_msg.tool_calls:
                t_name = tc.get("name")
                t_args = tc.get("args", {})

                if t_name in tool_map:
                    try:
                        res = tool_map[t_name].invoke(t_args)
                    except Exception as e:
                        res = {"error": str(e)}
                else:
                    res = {"error": f"Unknown tool: {t_name}"}

                new_obs.append({"tool": t_name, "args": t_args, "result": res})
                new_traces.append({
                    "step": "tool_trace",
                    "tool": t_name,
                    "args": t_args,
                    "output": res,
                })
        else:
            new_obs.append({"tool": "none", "result": "No tool call generated"})

        return {
            "observations": state.get("observations", []) + new_obs,
            "trajectory": state["trajectory"] + new_traces,
        }

    # 5. REJECT NODE: Formulate graceful rejection for out-of-scope queries
    def reject_node(state: SupportAgentState) -> dict[str, Any]:
        query = state["query"]
        logger.info(f"[SupportAgent] Rejecting out-of-scope query: '{query[:60]}'...")
        rejection_text = (
            "I am a specialized **Technical Notes Intelligence & Summarization Assistant** focused on "
            "Computer Science, Algorithms, Operating Systems, Machine Learning, and related technical notes "
            "from your knowledge base.\n\n"
            "Your query appears to be outside this domain. Please ask questions related to your technical "
            "notes, algorithms, data structures, operating systems, or machine learning!"
        )

        final_trace = {
            "step": "final_output",
            "content": rejection_text,
        }

        return {
            "final_answer": rejection_text,
            "trajectory": state["trajectory"] + [final_trace],
        }

    # 6. SYNTHESIZE NODE: Generate final grounded answer with citations or general reply
    def synthesize_node(state: SupportAgentState) -> dict[str, Any]:
        query = state["query"]
        history = state.get("history", [])
        chunks = state.get("retrieved_chunks", [])
        thoughts = state.get("thoughts", [])

        # Format history string
        hist_text = "\n".join(
            [f"Turn {i + 1} ({t.get('role', 'user')}): {t.get('content', '')}" for i, t in enumerate(history)]
        ) if history else "First turn of session."

        if not chunks:
            # General conversation / FAQ response without note citations
            general_prompt = (
                f"You are the Notes Intelligence & Summarization Assistant. The user said: '{query}'. "
                "Provide a helpful, friendly, and concise response introducing yourself as their technical notes assistant "
                "and offering to summarize, explain, or compare concepts across their technical notes. "
                "Use clean markdown formatting without redundant divider lines."
            )
            resp = synthesizer_llm.invoke([SystemMessage(content=SUPPORT_AGENT_SYSTEM_PROMPT), HumanMessage(content=general_prompt)])
            content = resp.content if isinstance(resp.content, str) else str(resp.content)
        else:
            # Formatted chunks string
            chunks_text = "\n\n".join(
                [
                    f"--- Document: {c.get('source', '')} | Section: {c.get('section', '')} ---\n{c.get('content', '')}"
                    for c in chunks
                ]
            )

            reasoning_summary = "\n".join(thoughts[-5:]) if thoughts else "Sufficient knowledge verified."

            prompt = SYNTHESIZER_USER_PROMPT_TEMPLATE.format(
                history=hist_text,
                query=query,
                retrieved_chunks=chunks_text,
                reasoning_summary=reasoning_summary,
            )

            messages = [
                SystemMessage(content=SUPPORT_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            logger.info("[SupportAgent] Synthesizing final grounded response...")
            resp = synthesizer_llm.invoke(messages)
            content = resp.content if isinstance(resp.content, str) else str(resp.content)

        final_trace = {
            "step": "final_output",
            "content": content,
        }

        return {
            "final_answer": content,
            "trajectory": state["trajectory"] + [final_trace],
        }

    # Assemble graph starting with REASON
    graph = StateGraph(SupportAgentState)
    graph.add_node("reason", reason_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("act", act_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("reject", reject_node)

    # Entrypoint is REASON
    graph.add_edge(START, "reason")
    graph.add_conditional_edges(
        "reason",
        route_reason_decision,
        {
            "retrieve": "retrieve",
            "act": "act",
            "synthesize": "synthesize",
            "reject": "reject",
        },
    )
    # Loops to allow fetching additional chunks or tool actions
    graph.add_edge("retrieve", "reason")
    graph.add_edge("act", "reason")
    graph.add_edge("synthesize", END)
    graph.add_edge("reject", END)

    return graph.compile()


class SupportAssistantAgent:
    """Wrapper class providing clean invocation and conversation tracking."""

    def __init__(self, model: str | None = None):
        self.model = model
        self.app = build_support_agent_graph(model=model)

    def run(self, query: str, history: list[dict[str, str]] | None = None) -> dict[str, Any]:
        """Execute a complete CoT + ReAct cycle for a user query."""
        initial_state: SupportAgentState = {
            "query": query,
            "history": history or [],
            "retrieved_chunks": [],
            "observations": [],
            "cot_reasoning": None,
            "thoughts": [],
            "trajectory": [],
            "retrieval_count": 0,
            "steps_taken": 0,
            "max_steps": 4,
            "final_answer": "",
        }

        result = self.app.invoke(initial_state)

        return {
            "answer": result.get("final_answer", ""),
            "thoughts": result.get("thoughts", []),
            "trajectory": result.get("trajectory", []),
            "retrieved_chunks": result.get("retrieved_chunks", []),
        }
