from scripts import generate_diagram


def test_generate_diagram_writes_mermaid_with_known_nodes(tmp_path):
    out_path = tmp_path / "state_diagram.md"
    generate_diagram.generate_diagram(str(out_path))
    content = out_path.read_text()
    assert "```mermaid" in content
    for node in ["parse_jd", "extract_requirements", "search_resumes",
                 "rank_candidates", "generate_report", "agent_router"]:
        assert node in content
