import os
import subprocess
import sys
from pathlib import Path

import pytest

requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"), reason="requires OPENROUTER_API_KEY"
)


@requires_api_key
def test_cli_full_conversation_via_stdin(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECKPOINT_PATH", str(tmp_path / "checkpoints.sqlite"))
    jd_text = Path("data/job_descriptions/frontend_react.txt").read_text().replace("\n", " ")
    stdin_script = f"JD: {jd_text}\nCompare the top 2 candidates\nquit\n"
    result = subprocess.run([sys.executable, "cli.py"], input=stdin_script,
                             capture_output=True, text=True, timeout=120)
    assert result.returncode == 0
    assert "Candidate" in result.stdout or "candidate" in result.stdout
