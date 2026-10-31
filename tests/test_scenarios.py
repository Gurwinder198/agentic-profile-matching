import os
import re
from pathlib import Path

import pytest

import matching_agent
import scenarios


@pytest.mark.skipif(not os.environ.get("OPENROUTER_API_KEY"), reason="requires OPENROUTER_API_KEY")
@pytest.mark.parametrize("scenario", scenarios.SCENARIOS, ids=lambda s: s["name"])
def test_scenario_completes_with_nonempty_responses(scenario, tmp_path):
    checkpoint_path = str(tmp_path / "checkpoints.sqlite")
    graph = matching_agent.build_graph(checkpoint_path)
    thread_id = scenario["name"]
    remaining_turns = list(scenario["turns"])

    if scenario["job_description_file"]:
        jd_text = Path("data/job_descriptions", scenario["job_description_file"]).read_text()
        last_result = matching_agent.run_turn(graph, thread_id, job_description=jd_text)
    else:
        # No JD: run_turn requires job_description or user_message, so the
        # scenario's own first turn (not a content-free call) starts it.
        first_turn = remaining_turns.pop(0)
        last_result = matching_agent.run_turn(graph, thread_id, user_message=first_turn)
    assert last_result["messages"][-1].content.strip()

    for turn in remaining_turns:
        last_result = matching_agent.run_turn(graph, thread_id, user_message=turn)
        assert last_result["messages"][-1].content.strip()

    if scenario["name"] == "direct_search_no_jd":
        reply = last_result["messages"][-1].content.lower()
        assert re.search(r"c\d{3}", reply), "expected at least one candidate id in the reply"
        assert "no matching candidates" not in reply
