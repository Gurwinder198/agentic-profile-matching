import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"), reason="requires OPENROUTER_API_KEY"
)


def test_chat_before_jd_prompts_for_jd_instead_of_crashing():
    at = AppTest.from_file("../app_streamlit.py")
    at.run()
    at.chat_input[0].set_value("Find me React candidates").run()
    assert not at.exception
    assert any("job description" in el.value.lower() for el in list(at.markdown) + list(at.chat_message))


@requires_api_key
def test_submitting_jd_then_chat_renders_shortlist(monkeypatch, tmp_path):
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoints.sqlite"))
    at = AppTest.from_file("../app_streamlit.py")
    at.run()
    at.sidebar.text_area[0].set_value(Path("data/job_descriptions/frontend_react.txt").read_text())
    at.sidebar.button[0].click().run(timeout=120)
    assert not at.exception
    assert len(at.session_state["shortlist"]) >= 1
