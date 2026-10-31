"""LangGraph recruiting agent: state, fixed screening pipeline, and (from
Task 8 on) the conversational router loop.
"""

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tools import filesystem_tools, rag_tools, screening_tools


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    job_description: str
    requirements: dict
    candidate_pool: list[dict]
    shortlist: list[dict]
    round: str
    final_recommendations: list[dict]


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
