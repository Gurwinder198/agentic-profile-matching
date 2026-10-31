"""Screening tools (Milestone 3): requirements extraction, ranking,
comparison, interview questions, and final recommendations.

Every function here is plain and state-agnostic; orchestration that reads
or writes LangGraph AgentState lives in matching_agent.py instead.
"""

from typing import Literal

from pydantic import BaseModel

from llm import get_chat_model


class JobRequirements(BaseModel):
    must_have: list[str]
    nice_to_have: list[str]
    role_title: str
    seniority: str


class CandidateScore(BaseModel):
    candidate_id: str
    score: float
    criteria_scores: dict[str, float]
    strengths: list[str]
    gaps: list[str]


class FinalRecommendation(BaseModel):
    candidate_id: str
    decision: Literal["hire", "no-hire"]
    confidence: float
    justification: str


_EMPTY_REQUIREMENTS = {"must_have": [], "nice_to_have": [], "role_title": "", "seniority": ""}


def extract_requirements(jd_text: str) -> dict:
    """Parse a job description into must-have vs nice-to-have requirements."""
    if not jd_text or not jd_text.strip():
        return dict(_EMPTY_REQUIREMENTS)

    model = get_chat_model().with_structured_output(JobRequirements, method="function_calling")
    prompt = (
        "Extract the job requirements from this job description. "
        "Separate must-have requirements from nice-to-have requirements, "
        "and identify the role title and seniority level.\n\n"
        f"Job description:\n{jd_text}"
    )
    result = model.invoke(prompt)
    return result.model_dump()


def update_requirements(current_requirements: dict, new_instruction: str) -> dict:
    """Merge a new/changed instruction into the current requirements."""
    model = get_chat_model().with_structured_output(JobRequirements, method="function_calling")
    prompt = (
        "Here are the current job requirements:\n"
        f"{current_requirements}\n\n"
        "The recruiter has given this new instruction to update the requirements:\n"
        f"{new_instruction}\n\n"
        "Return the full, updated set of requirements (must-have, nice-to-have, "
        "role title, seniority), incorporating the new instruction."
    )
    result = model.invoke(prompt)
    return result.model_dump()


def _score_is_plausible(data: dict) -> bool:
    """The LLM occasionally returns an internally-inconsistent score (e.g.
    score=0 despite high per-criterion scores and no gaps) — catch the
    clearest case of that self-contradiction."""
    criteria_scores = data.get("criteria_scores") or {}
    if not criteria_scores:
        return True
    average = sum(criteria_scores.values()) / len(criteria_scores)
    if data.get("score") == 0 and average >= 50:
        return False
    return True


def score_candidate(requirements: dict, candidate_id: str, resume_text: str, max_attempts: int = 3) -> dict:
    """Score a single candidate's resume against the given requirements."""
    model = get_chat_model().with_structured_output(CandidateScore, method="function_calling")
    prompt = (
        f"Score this candidate (candidate_id={candidate_id!r}) against the job requirements below.\n\n"
        f"Requirements:\n{requirements}\n\n"
        f"Resume:\n{resume_text}\n\n"
        "Give an overall score from 0 to 100, a per-criterion breakdown in "
        "criteria_scores (one entry per must-have and nice-to-have item, each "
        "0 to 100), a list of strengths, and a list of gaps. Set candidate_id "
        f"to exactly {candidate_id!r}."
    )
    data = None
    for _ in range(max_attempts):
        result = model.invoke(prompt)
        data = result.model_dump()
        if _score_is_plausible(data):
            return data
    return data


def _suggest_improvements(requirements: dict, scored: dict) -> list[str]:
    model = get_chat_model()
    prompt = (
        "A candidate is borderline for a role. Based on their gaps below, "
        "suggest 2-4 concrete, specific improvements that would make them "
        "qualify. Return one suggestion per line, with no numbering or bullets.\n\n"
        f"Requirements: {requirements}\n"
        f"Gaps: {scored['gaps']}\n"
        f"Strengths: {scored['strengths']}\n"
    )
    response = model.invoke(prompt)
    lines = [line.strip("-* ").strip() for line in response.content.splitlines()]
    return [line for line in lines if line]


def rank_candidates(requirements: dict, candidates: list[dict]) -> list[dict]:
    """Score every candidate, sort descending by score, and add
    improvement suggestions for borderline candidates."""
    scored = []
    for candidate in candidates:
        result = score_candidate(requirements, candidate["candidate_id"], candidate["resume_text"])
        if 60 <= result["score"] < 75:
            result["improvement_suggestions"] = _suggest_improvements(requirements, result)
        else:
            result["improvement_suggestions"] = []
        scored.append(result)
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored


def generate_report(requirements: dict, ranked_candidates: list[dict]) -> str:
    """Render a deterministic markdown report from already-scored candidates."""
    lines = [
        f"# Screening Report: {requirements.get('role_title', '')}",
        "",
        f"**Must-have:** {', '.join(requirements.get('must_have', []))}",
        f"**Nice-to-have:** {', '.join(requirements.get('nice_to_have', []))}",
        "",
    ]
    for rank, candidate in enumerate(ranked_candidates, start=1):
        lines.append(f"## {rank}. {candidate['candidate_id']} — score {candidate['score']}")
        lines.append("")
        lines.append("**Strengths:**")
        for s in candidate.get("strengths", []):
            lines.append(f"- {s}")
        lines.append("")
        lines.append("**Gaps:**")
        for g in candidate.get("gaps", []):
            lines.append(f"- {g}")
        suggestions = candidate.get("improvement_suggestions", [])
        if suggestions:
            lines.append("")
            lines.append("**Improvement suggestions:**")
            for s in suggestions:
                lines.append(f"- {s}")
        lines.append("")
    return "\n".join(lines)


def compare_candidates(candidates: list[dict]) -> dict:
    """Compare 2+ already-scored candidates, narrating only from their
    stored scores/strengths/gaps. Returns a not-found result for an empty
    list without calling the LLM."""
    if not candidates:
        return {"comparison_table": [], "narrative": "No matching candidates found."}

    comparison_table = [
        {
            "candidate_id": c["candidate_id"],
            "score": c["score"],
            "criteria_scores": c["criteria_scores"],
            "strengths": c["strengths"],
            "gaps": c["gaps"],
        }
        for c in candidates
    ]

    model = get_chat_model()
    prompt = (
        "Compare these candidates for the recruiter. Use only the scores and "
        "notes given below; do not re-evaluate the candidates.\n\n"
        f"{comparison_table}\n\n"
        "Write a short narrative explaining how they compare and why, citing "
        "the specific scores/strengths/gaps given."
    )
    response = model.invoke(prompt)
    return {"comparison_table": comparison_table, "narrative": response.content}


def explain_ranking(higher: dict, lower: dict) -> str:
    """Explain why `higher` ranked above `lower`, grounded in stored scores."""
    result = compare_candidates([higher, lower])
    return result["narrative"]


def generate_interview_questions(
    candidate_id: str,
    resume_text: str,
    requirements: dict,
    gaps: list[str] | None = None,
) -> list[str]:
    """Generate screening questions for a candidate, weighted toward probing gaps."""
    model = get_chat_model()
    prompt = (
        f"Generate at least 3 interview screening questions for candidate {candidate_id!r} "
        f"for this role.\n\nRequirements: {requirements}\n\nResume:\n{resume_text}\n\n"
        f"Known gaps to probe: {gaps or []}\n\n"
        "Return one question per line, with no numbering or bullets."
    )
    response = model.invoke(prompt)
    lines = [line.strip("-* ").strip() for line in response.content.splitlines()]
    return [line for line in lines if line]


def generate_final_recommendation(candidate: dict) -> dict:
    """Produce a hire/no-hire recommendation grounded in a candidate's stored scores."""
    model = get_chat_model().with_structured_output(FinalRecommendation, method="function_calling")
    prompt = (
        "Use only the scores and notes given below; do not re-evaluate the candidate.\n\n"
        f"Candidate ID: {candidate['candidate_id']}\n"
        f"Score: {candidate['score']}\n"
        f"Criteria scores: {candidate['criteria_scores']}\n"
        f"Strengths: {candidate['strengths']}\n"
        f"Gaps: {candidate['gaps']}\n\n"
        "Give a hire or no-hire decision, a confidence from 0 to 1, and a "
        f"justification. Set candidate_id to exactly {candidate['candidate_id']!r}."
    )
    result = model.invoke(prompt)
    return result.model_dump()
