"""Generate the agent's state-machine diagram from the real compiled graph."""

from pathlib import Path

import matching_agent

LEGEND = (
    "This diagram is generated directly from the compiled LangGraph "
    "`StateGraph`, not hand-drawn. The graph has two regimes: a **fixed "
    "pipeline** (`parse_jd` through `generate_report`) that runs once per "
    "new/replaced job description, and a conversational **router loop** "
    "(`agent_router`) that handles every follow-up chat turn by calling "
    "tools (compare, explain, re-rank, interview questions, final "
    "recommendation) without re-running the pipeline. Both paths converge "
    "at `human_feedback`, which simply marks the end of a turn — each "
    "chat message is one `graph.invoke()` against a SQLite-checkpointed "
    "thread, which is how conversation state persists across turns."
)


def generate_diagram(out_path: str = "docs/state_diagram.md") -> None:
    """Build the real graph, render its mermaid diagram, and write it to out_path."""
    graph = matching_agent.build_graph(checkpoint_path=":memory:")
    mermaid = graph.get_graph().draw_mermaid()
    content = f"# Agent State Machine\n\n{LEGEND}\n\n```mermaid\n{mermaid}\n```\n"
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    generate_diagram()
