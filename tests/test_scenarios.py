import os
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
    jd_text = None
    if scenario["job_description_file"]:
        jd_text = Path("data/job_descriptions", scenario["job_description_file"]).read_text()
    last_result = matching_agent.run_turn(graph, thread_id, job_description=jd_text)
    assert last_result["messages"][-1].content.strip()
    for turn in scenario["turns"]:
        last_result = matching_agent.run_turn(graph, thread_id, user_message=turn)
        assert last_result["messages"][-1].content.strip()
