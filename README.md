# Agentic Profile Matching

A LangGraph-based recruiting agent that screens resumes against a job
description across multiple rounds, ranks and explains candidates, and
takes natural-language follow-up questions through a chat interface.

## Setup

```bash
# 1. Create and activate a virtualenv (Python 3.14)
python -m venv .venv
./.venv/Scripts/activate        # Windows
# source .venv/bin/activate      # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your OpenRouter API key
cp .env.example .env
# then edit .env and fill in OPENROUTER_API_KEY (get one at https://openrouter.ai/keys)

# 4. Generate the synthetic resume corpus and job descriptions
python -c "import data_gen; data_gen.generate_dataset()"

# 5. Build the RAG index over the generated resumes
python -c "from tools import rag_tools; rag_tools.build_index(force_rebuild=True)"
```

Steps 4-5 populate `data/resumes/` (100 resumes + `manifest.json`),
`data/job_descriptions/` (4 sample JDs), and `data/chroma_db/` (the local
vector index) — all gitignored, generated locally.

## Running the CLI

```bash
python cli.py
```

Type `JD: <paste a job description>` to start a screen (or paste one of
the generated files, e.g. `data/job_descriptions/frontend_react.txt`),
then chat with follow-up questions. Type `quit` or `exit` to stop.

## Running the Streamlit app

```bash
streamlit run app_streamlit.py
```

Paste a job description into the sidebar and click **Run Initial
Screen**, then use the chat box for follow-ups ("Compare the top 3",
"Why did X rank higher than Y?", "Give me a hire/no-hire recommendation
for the top 2").

## Running the demo scenarios

```bash
python -m scripts.run_demo_scenarios
```

Runs the 7 scripted conversation flows (satisfying the "5+ test
scenarios" requirement) end-to-end against the real model and writes
`docs/demo_transcripts.md`. The same 7 scenarios are also exercised as
automated tests in `tests/test_scenarios.py` (`pytest tests/test_scenarios.py -v`,
requires `OPENROUTER_API_KEY`).

To regenerate the state-machine diagram from the real compiled graph:

```bash
python -m scripts.generate_diagram
```

(Run as a module — `-m scripts.generate_diagram` — not as a bare script,
so the repo root is on `sys.path` and `matching_agent` resolves.) Writes
`docs/state_diagram.md`.

## Running the tests

```bash
pytest
```

Tests that call the real LLM are marked `skipif` on `OPENROUTER_API_KEY`
and will skip cleanly if it's unset; tests covering tool logic (path
traversal, empty-JD handling, the not-found contract, etc.) run without
any API key.

## Architecture notes

`matching_agent.py` implements the assignment's node pipeline exactly —
`parse_jd → extract_requirements → search_resumes → rank_candidates →
generate_report` — as a fixed sequence that runs once per new/replaced
job description. Everything after that first report is handled by a
second regime, the `agent_router` node: an LLM bound to tools (compare
candidates, explain a ranking, re-rank on new criteria, generate
interview questions, give a hire/no-hire recommendation, search the
corpus directly) that loops internally until it has a final answer.

The assignment's conceptual **"Human Feedback Loop → END"** box is
implemented as one `graph.invoke()` call per chat turn against a
SQLite-checkpointed thread (`data/checkpoints.sqlite`), rather than a
blocking LangGraph `interrupt()`. Each turn — whether it's a fresh job
description or a follow-up chat message — is a separate `run_turn()`
call; conversation and shortlist state persist between turns (and across
a process restart) via the checkpoint, which is how both paths converge
on a `human_feedback` node that simply marks the end of a turn.

The full state machine, generated directly from the compiled graph (not
hand-drawn), is in [`docs/state_diagram.md`](docs/state_diagram.md).
Real end-to-end conversation transcripts are in
[`docs/demo_transcripts.md`](docs/demo_transcripts.md).
