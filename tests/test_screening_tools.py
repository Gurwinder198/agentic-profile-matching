import os

import pytest

from tools import screening_tools

requires_api_key = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"), reason="requires OPENROUTER_API_KEY"
)


def test_extract_requirements_on_empty_jd_returns_empty_lists_without_llm_call(monkeypatch):
    monkeypatch.setattr(
        screening_tools,
        "get_chat_model",
        lambda: (_ for _ in ()).throw(AssertionError("should not call LLM")),
    )
    result = screening_tools.extract_requirements("   ")
    assert result == {"must_have": [], "nice_to_have": [], "role_title": "", "seniority": ""}


@requires_api_key
def test_extract_requirements_on_real_jd_finds_react_and_years(real_frontend_jd_text):
    result = screening_tools.extract_requirements(real_frontend_jd_text)
    assert any("react" in s.lower() for s in result["must_have"])


@requires_api_key
def test_rank_candidates_sorts_descending_and_stores_criteria_scores(sample_requirements, sample_candidates):
    ranked = screening_tools.rank_candidates(sample_requirements, sample_candidates)
    scores = [c["score"] for c in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all("criteria_scores" in c for c in ranked)


@requires_api_key
def test_generate_report_is_pure_and_includes_every_candidate(sample_requirements, ranked_sample_candidates):
    report = screening_tools.generate_report(sample_requirements, ranked_sample_candidates)
    for c in ranked_sample_candidates:
        assert c["candidate_id"] in report


def test_compare_candidates_on_empty_list_returns_not_found_without_llm_call(monkeypatch):
    monkeypatch.setattr(
        screening_tools,
        "get_chat_model",
        lambda: (_ for _ in ()).throw(AssertionError("should not call LLM")),
    )
    result = screening_tools.compare_candidates([])
    assert result["comparison_table"] == []
    assert "No matching candidates" in result["narrative"]


@requires_api_key
def test_generate_interview_questions_returns_at_least_three(sample_requirements):
    questions = screening_tools.generate_interview_questions(
        "c1", "Frontend Engineer, 2 years, HTML/CSS only", sample_requirements, gaps=["React"])
    assert len(questions) >= 3
    assert all(isinstance(q, str) and q.strip() for q in questions)
