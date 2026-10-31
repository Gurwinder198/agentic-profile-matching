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
