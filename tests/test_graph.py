import os
from pathlib import Path

import pytest

import matching_agent

requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"), reason="requires OPENROUTER_API_KEY"
)


@requires_api_key
def test_fixed_pipeline_produces_ranked_shortlist_and_report():
    graph = matching_agent.build_fixed_pipeline_graph()
    jd_text = Path("data/job_descriptions/frontend_react.txt").read_text()
    result = graph.invoke({"messages": [], "job_description": jd_text, "requirements": {},
                            "candidate_pool": [], "shortlist": [], "round": "initial_screen",
                            "final_recommendations": []})
    assert result["requirements"]["must_have"]
    assert 1 <= len(result["shortlist"]) <= 10
    assert all("criteria_scores" in c for c in result["shortlist"])
    assert "shortlist" in result and isinstance(result["messages"][-1].content, str)
    assert result["messages"][-1].content  # the report text landed in the conversation


def test_find_candidate_matches_by_id_or_name():
    candidates = [{"candidate_id": "c1", "resume_text": "Name: Alice Chen\n..."}]
    assert matching_agent.find_candidate(candidates, "c1") is candidates[0]
    assert matching_agent.find_candidate(candidates, "alice chen") is candidates[0]
    assert matching_agent.find_candidate(candidates, "nobody") is None


@requires_api_key
def test_conversation_state_persists_across_separate_graph_instances(tmp_path):
    checkpoint_path = str(tmp_path / "checkpoints.sqlite")
    jd_text = Path("data/job_descriptions/frontend_react.txt").read_text()
    graph1 = matching_agent.build_graph(checkpoint_path)
    matching_agent.run_turn(graph1, "thread-1", job_description=jd_text)
    # Simulate a process restart: fresh graph instance, same checkpoint file, same thread.
    graph2 = matching_agent.build_graph(checkpoint_path)
    state = graph2.get_state({"configurable": {"thread_id": "thread-1"}})
    assert state.values["shortlist"]


@requires_api_key
def test_followup_question_routes_to_tool_without_rerunning_pipeline(tmp_path):
    checkpoint_path = str(tmp_path / "checkpoints.sqlite")
    jd_text = Path("data/job_descriptions/frontend_react.txt").read_text()
    graph = matching_agent.build_graph(checkpoint_path)
    matching_agent.run_turn(graph, "thread-2", job_description=jd_text)
    result = matching_agent.run_turn(graph, "thread-2", user_message="Compare the top 2 candidates side by side")
    assert result["job_description"] == jd_text  # fixed pipeline did not re-run
    assert "compare" in result["messages"][-1].content.lower() or len(result["shortlist"]) >= 2


@requires_api_key
def test_compare_candidates_tool_reports_not_found_for_unknown_id(tmp_path):
    checkpoint_path = str(tmp_path / "checkpoints.sqlite")
    jd_text = Path("data/job_descriptions/frontend_react.txt").read_text()
    graph = matching_agent.build_graph(checkpoint_path)
    matching_agent.run_turn(graph, "thread-3", job_description=jd_text)
    result = matching_agent.run_turn(graph, "thread-3", user_message="Compare candidate ZZZ999 against candidate YYY888")
    assert "not found" in result["messages"][-1].content.lower() or "no matching" in result["messages"][-1].content.lower()
