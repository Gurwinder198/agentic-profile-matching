"""Demo conversation-flow scenarios: the single source of truth for both
the automated scenario tests and the transcript generator.
"""

SCENARIOS: list[dict] = [
    {
        "name": "direct_search_no_jd",
        "job_description_file": None,
        "turns": ["Find me candidates with React and 3+ years experience"],
    },
    {
        "name": "full_screen_and_report",
        "job_description_file": "frontend_react.txt",
        "turns": [],
    },
    {
        "name": "compare_top_three",
        "job_description_file": "frontend_react.txt",
        "turns": ["Compare the top 3 matches side by side"],
    },
    {
        "name": "explain_ranking",
        "job_description_file": "frontend_react.txt",
        "turns": ["Why did the top candidate rank higher than the second one?"],
    },
    {
        "name": "iterative_refinement",
        "job_description_file": "frontend_react.txt",
        "turns": [
            "Actually we also need strong AWS experience, not just React",
            "What changed in the rankings and why?",
        ],
    },
    {
        "name": "final_recommendation_round",
        "job_description_file": "frontend_react.txt",
        "turns": ["Give me a hire or no-hire recommendation for the top 2 candidates"],
    },
    {
        "name": "borderline_improvement_suggestions",
        "job_description_file": "frontend_react.txt",
        "turns": ["Which candidates are borderline, and what would it take for them to qualify?"],
    },
]
