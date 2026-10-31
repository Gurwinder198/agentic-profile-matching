"""LangGraph recruiting agent: state, fixed screening pipeline, and the
conversational router loop.
"""

import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from llm import get_chat_model
from tools import filesystem_tools, rag_tools, screening_tools


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    job_description: str
    requirements: dict
    candidate_pool: list[dict]
    shortlist: list[dict]
    round: str
    final_recommendations: list[dict]
    new_jd_this_turn: bool


def find_candidate(candidates: list[dict], id_or_name: str) -> dict | None:
    """Case-insensitive match against candidate_id or the resume's Name: line."""
    needle = id_or_name.strip().lower()
    for candidate in candidates:
        if candidate.get("candidate_id", "").lower() == needle:
            return candidate
        resume_text = candidate.get("resume_text", "")
        for line in resume_text.splitlines():
            if line.lower().startswith("name:"):
                name = line.split(":", 1)[1].strip().lower()
                if name == needle:
                    return candidate
                break
    return None


def parse_jd_node(state: AgentState) -> dict:
    return {"messages": [HumanMessage(state["job_description"])]}


def extract_requirements_node(state: AgentState) -> dict:
    requirements = screening_tools.extract_requirements(state["job_description"])
    return {"requirements": requirements}


def search_resumes_node(state: AgentState) -> dict:
    requirements = state["requirements"]
    query = " ".join(requirements.get("must_have", []) + [requirements.get("role_title", "")])
    hits = rag_tools.rag_search_resumes(query, k=10)
    candidate_pool = [
        {"candidate_id": hit["candidate_id"], "resume_text": filesystem_tools.read_resume_file(hit["filename"])}
        for hit in hits
    ]
    return {"candidate_pool": candidate_pool}


def rank_candidates_node(state: AgentState) -> dict:
    shortlist = screening_tools.rank_candidates(state["requirements"], state["candidate_pool"])
    return {"shortlist": shortlist, "round": "deep_analysis"}


def generate_report_node(state: AgentState) -> dict:
    report_text = screening_tools.generate_report(state["requirements"], state["shortlist"])
    return {"messages": [AIMessage(report_text)]}


def build_fixed_pipeline_graph():
    graph = StateGraph(AgentState)
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("generate_report", generate_report_node)

    graph.add_edge(START, "parse_jd")
    graph.add_edge("parse_jd", "extract_requirements")
    graph.add_edge("extract_requirements", "search_resumes")
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "generate_report")
    graph.add_edge("generate_report", END)

    return graph.compile()


def human_feedback_node(state: AgentState) -> dict:
    """Identity passthrough marking where a conversational turn ends."""
    return {}


def _route_entry(state: AgentState) -> str:
    return "parse_jd" if state.get("new_jd_this_turn") else "agent_router"


def _shortlist_summary(shortlist: list[dict]) -> str:
    if not shortlist:
        return "The shortlist is currently empty."
    lines = [f"{i + 1}. {c['candidate_id']} - score {c['score']}" for i, c in enumerate(shortlist)]
    return "Current shortlist (ranked):\n" + "\n".join(lines)


def agent_router_node(state: AgentState) -> dict:
    """Conversational router: an LLM bound to tools that read/update the
    current requirements and shortlist, looping internally until it has a
    final answer for the user."""
    updates = {
        "requirements": state.get("requirements", {}),
        "shortlist": state.get("shortlist", []),
    }

    @tool
    def compare_candidates_tool(candidate_ids: list[str]) -> str:
        """Compare two or more candidates from the current shortlist side by side."""
        found = [c for c in (find_candidate(updates["shortlist"], cid) for cid in candidate_ids) if c]
        result = screening_tools.compare_candidates(found)
        return result["narrative"]

    @tool
    def explain_ranking_tool(higher_id: str, lower_id: str) -> str:
        """Explain why one candidate (higher_id) ranked above another (lower_id)."""
        higher = find_candidate(updates["shortlist"], higher_id)
        lower = find_candidate(updates["shortlist"], lower_id)
        if higher is None or lower is None:
            return "No matching candidates found."
        return screening_tools.explain_ranking(higher, lower)

    @tool
    def generate_interview_questions_tool(candidate_id: str) -> str:
        """Generate interview screening questions for one candidate from the shortlist."""
        candidate = find_candidate(updates["shortlist"], candidate_id)
        if candidate is None:
            return "No matching candidate found."
        resume_text = filesystem_tools.read_resume_file(f"{candidate['candidate_id']}.txt")
        questions = screening_tools.generate_interview_questions(
            candidate["candidate_id"], resume_text, updates["requirements"], candidate.get("gaps"))
        return "\n".join(questions)

    @tool
    def generate_final_recommendation_tool(candidate_id: str) -> str:
        """Give a hire/no-hire recommendation for one candidate from the shortlist."""
        candidate = find_candidate(updates["shortlist"], candidate_id)
        if candidate is None:
            return "No matching candidate found."
        rec = screening_tools.generate_final_recommendation(candidate)
        return f"{rec['decision'].upper()} (confidence {rec['confidence']}): {rec['justification']}"

    @tool
    def rerank_candidates_tool(new_instruction: str) -> str:
        """Update the job requirements with a new/changed criterion and
        re-rank candidates against the full resume pool, not just the
        current shortlist."""
        updated_requirements = screening_tools.update_requirements(updates["requirements"], new_instruction)
        query = " ".join(updated_requirements.get("must_have", []) + [updated_requirements.get("role_title", "")])
        fresh_hits = rag_tools.rag_search_resumes(query, k=10)
        existing_ids = {c["candidate_id"] for c in updates["shortlist"]}
        fresh_ids = {h["candidate_id"] for h in fresh_hits}
        union_ids = existing_ids | fresh_ids
        candidates_for_scoring = [
            {"candidate_id": cid, "resume_text": filesystem_tools.read_resume_file(f"{cid}.txt")}
            for cid in union_ids
        ]
        new_shortlist = screening_tools.rank_candidates(updated_requirements, candidates_for_scoring)[:10]
        new_ids = {c["candidate_id"] for c in new_shortlist}
        entered = new_ids - existing_ids
        left = existing_ids - new_ids
        updates["requirements"] = updated_requirements
        updates["shortlist"] = new_shortlist

        note_lines = [f"Updated requirements based on: {new_instruction!r}."]
        if entered:
            note_lines.append(f"Entered shortlist: {', '.join(sorted(entered))}.")
        if left:
            note_lines.append(f"Left shortlist: {', '.join(sorted(left))}.")
        note_lines.append(_shortlist_summary(new_shortlist))
        return "\n".join(note_lines)

    @tool
    def rag_search_tool(query: str) -> str:
        """Search the full resume corpus directly by a free-text query,
        independent of any existing shortlist. Use this when there is no
        shortlist yet and the user asks to find candidates."""
        hits = rag_tools.rag_search_resumes(query, k=10)
        return "\n".join(f"{h['candidate_id']} (score {h['score']:.3f}): {h['snippet']}" for h in hits)

    tools = [
        compare_candidates_tool,
        explain_ranking_tool,
        generate_interview_questions_tool,
        generate_final_recommendation_tool,
        rerank_candidates_tool,
        rag_search_tool,
    ]
    tools_by_name = {t.name: t for t in tools}

    llm_with_tools = get_chat_model().bind_tools(tools)
    system_prompt = (
        "You are a recruiting assistant helping screen candidates. "
        + _shortlist_summary(updates["shortlist"])
        + "\nIf there is no shortlist yet and the user asks to find candidates, "
        "use rag_search_tool to search the resume corpus directly."
        "\nIf a tool result says 'No matching candidates found' or "
        "'No matching candidate found', relay that exact phrase to the "
        "user rather than paraphrasing it."
    )
    conversation = [SystemMessage(system_prompt)] + list(state.get("messages", []))
    turn_start = len(conversation)

    for _ in range(5):
        ai_message = llm_with_tools.invoke(conversation)
        conversation.append(ai_message)
        if not getattr(ai_message, "tool_calls", None):
            break
        for call in ai_message.tool_calls:
            tool_fn = tools_by_name[call["name"]]
            result = tool_fn.invoke(call["args"])
            conversation.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    new_messages = conversation[turn_start:]
    return {
        "messages": new_messages,
        "requirements": updates["requirements"],
        "shortlist": updates["shortlist"],
    }


def build_graph(checkpoint_path: str = "data/checkpoints.sqlite"):
    graph = StateGraph(AgentState)
    graph.add_node("parse_jd", parse_jd_node)
    graph.add_node("extract_requirements", extract_requirements_node)
    graph.add_node("search_resumes", search_resumes_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("generate_report", generate_report_node)
    graph.add_node("agent_router", agent_router_node)
    graph.add_node("human_feedback", human_feedback_node)

    graph.add_conditional_edges(
        START, _route_entry, {"parse_jd": "parse_jd", "agent_router": "agent_router"}
    )
    graph.add_edge("parse_jd", "extract_requirements")
    graph.add_edge("extract_requirements", "search_resumes")
    graph.add_edge("search_resumes", "rank_candidates")
    graph.add_edge("rank_candidates", "generate_report")
    graph.add_edge("generate_report", "human_feedback")
    graph.add_edge("agent_router", "human_feedback")
    graph.add_edge("human_feedback", END)

    # SqliteSaver.from_conn_string() is a context manager that closes its
    # connection on exit; the checkpointer needs to outlive this function,
    # so the connection is opened directly instead.
    conn = sqlite3.connect(checkpoint_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return graph.compile(checkpointer=checkpointer)


def run_turn(
    graph,
    thread_id: str,
    user_message: str | None = None,
    job_description: str | None = None,
) -> AgentState:
    """Run one conversational turn: either a fresh/replaced job description
    (restarts the fixed pipeline) or a follow-up chat message (routes to
    the conversational tool loop)."""
    config = {"configurable": {"thread_id": thread_id}}
    if job_description is not None:
        input_state = {
            "job_description": job_description,
            "new_jd_this_turn": True,
            "requirements": {},
            "candidate_pool": [],
            "shortlist": [],
            "round": "initial_screen",
            "final_recommendations": [],
        }
    else:
        input_state = {
            "messages": [HumanMessage(user_message)] if user_message else [],
            "new_jd_this_turn": False,
        }
    return graph.invoke(input_state, config=config)
