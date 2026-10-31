"""Run every demo scenario end-to-end and write docs/demo_transcripts.md."""

from pathlib import Path

import matching_agent
import scenarios


def _as_blockquote(text: str) -> str:
    """Quote multi-line agent text so its own markdown headers don't collide
    with this document's scenario-level headers."""
    return "\n".join(f"> {line}" if line else ">" for line in text.splitlines())


def _format_shortlist(shortlist: list[dict]) -> str:
    if not shortlist:
        return "_(no shortlist)_"
    lines = [
        f"- **{c['candidate_id']}** — score {c['score']} "
        f"(strengths: {', '.join(c.get('strengths', []))}; gaps: {', '.join(c.get('gaps', []))})"
        for c in shortlist
    ]
    return "\n".join(lines)


def main(out_path: str = "docs/demo_transcripts.md", checkpoint_path: str = "data/checkpoints.sqlite") -> None:
    graph = matching_agent.build_graph(checkpoint_path)
    sections = ["# Demo Scenario Transcripts\n"]

    for scenario in scenarios.SCENARIOS:
        thread_id = f"demo-{scenario['name']}"
        lines = [f"## {scenario['name']}\n"]

        jd_text = None
        if scenario["job_description_file"]:
            jd_text = Path("data/job_descriptions", scenario["job_description_file"]).read_text()
            lines.append(f"**Job description submitted:** `{scenario['job_description_file']}`\n")
        result = matching_agent.run_turn(graph, thread_id, job_description=jd_text)
        lines.append(f"**Agent:**\n{_as_blockquote(result['messages'][-1].content)}\n")

        for turn in scenario["turns"]:
            lines.append(f"**User:** {turn}\n")
            result = matching_agent.run_turn(graph, thread_id, user_message=turn)
            lines.append(f"**Agent:**\n{_as_blockquote(result['messages'][-1].content)}\n")

        lines.append("**Final shortlist:**\n")
        lines.append(_format_shortlist(result["shortlist"]))
        lines.append("")
        sections.append("\n".join(lines))

    Path(out_path).write_text("\n\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
