import os
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

import matching_agent

requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"), reason="requires OPENROUTER_API_KEY"
)


class _ScriptedToolCallingModel:
    """A fake bound-tools chat model that returns a scripted sequence of
    AIMessage responses, one per .invoke() call — avoids a real LLM call
    while exercising agent_router_node's own dispatch/state-update logic."""

    def __init__(self, responses):
        self._responses = list(responses)

    def invoke(self, conversation):
        return self._responses.pop(0)


class _FakeChatModel:
    def __init__(self, responses):
        self._responses = responses

    def bind_tools(self, tools):
        return _ScriptedToolCallingModel(self._responses)


def test_compare_candidates_tool_mentions_missing_ids_alongside_found_ones(monkeypatch):
    shortlist = [{"candidate_id": "C001", "score": 90.0, "criteria_scores": {}, "strengths": [], "gaps": []}]
    state = {"messages": [], "requirements": {}, "shortlist": shortlist}

    monkeypatch.setattr(
        matching_agent.screening_tools, "compare_candidates",
        lambda candidates: {"comparison_table": [], "narrative": "C001 is strong."},
    )
    tool_call = {"name": "compare_candidates_tool", "args": {"candidate_ids": ["C001", "ZZZ999"]}, "id": "call1"}
    responses = [AIMessage(content="", tool_calls=[tool_call]), AIMessage(content="done")]
    monkeypatch.setattr(matching_agent, "get_chat_model", lambda: _FakeChatModel(responses))

    result = matching_agent.agent_router_node(state)
    tool_message = result["messages"][1]
    assert "ZZZ999" in tool_message.content
    assert "not found" in tool_message.content.lower() or "no matching" in tool_message.content.lower()
    assert "C001 is strong." in tool_message.content


def test_tool_dispatch_errors_are_fed_back_as_tool_messages_not_raised(monkeypatch):
    state = {"messages": [], "requirements": {}, "shortlist": []}
    bad_call = {"name": "nonexistent_tool", "args": {}, "id": "call1"}
    responses = [AIMessage(content="", tool_calls=[bad_call]), AIMessage(content="recovered")]
    monkeypatch.setattr(matching_agent, "get_chat_model", lambda: _FakeChatModel(responses))

    result = matching_agent.agent_router_node(state)  # must not raise
    tool_message = result["messages"][1]
    assert "failed" in tool_message.content.lower() or "error" in tool_message.content.lower()


def test_generate_final_recommendation_tool_persists_to_state(monkeypatch):
    shortlist = [{"candidate_id": "C001", "score": 90.0, "criteria_scores": {}, "strengths": [], "gaps": []}]
    state = {"messages": [], "requirements": {}, "shortlist": shortlist,
              "round": "deep_analysis", "final_recommendations": []}

    rec = {"candidate_id": "C001", "decision": "hire", "confidence": 0.9, "justification": "Strong fit."}
    monkeypatch.setattr(matching_agent.screening_tools, "generate_final_recommendation", lambda candidate: rec)
    tool_call = {"name": "generate_final_recommendation_tool", "args": {"candidate_id": "C001"}, "id": "call1"}
    responses = [AIMessage(content="", tool_calls=[tool_call]), AIMessage(content="done")]
    monkeypatch.setattr(matching_agent, "get_chat_model", lambda: _FakeChatModel(responses))

    result = matching_agent.agent_router_node(state)
    assert result["round"] == "final_recommendation"
    assert result["final_recommendations"] == [rec]


def test_run_turn_rejects_a_content_free_call(tmp_path):
    """run_turn(graph, thread_id) with neither job_description nor
    user_message would otherwise fire an LLM call against an empty
    conversation, priming context with a response to nothing — this is
    what caused the direct_search_no_jd demo scenario to misbehave."""
    checkpoint_path = str(tmp_path / "checkpoints.sqlite")
    graph = matching_agent.build_graph(checkpoint_path)
    with pytest.raises(ValueError, match="job_description.*user_message"):
        matching_agent.run_turn(graph, "thread-empty")


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
