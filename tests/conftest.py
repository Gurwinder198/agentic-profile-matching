from pathlib import Path

import pytest

from tools import screening_tools


@pytest.fixture
def sample_requirements():
    return {
        "must_have": ["React", "JavaScript", "3+ years of experience"],
        "nice_to_have": ["TypeScript", "Next.js"],
        "role_title": "Frontend Engineer",
        "seniority": "Mid-level",
    }


@pytest.fixture
def sample_candidates():
    return [
        {
            "candidate_id": "c1",
            "resume_text": (
                "Name: Alice Chen\n"
                "Candidate ID: c1\n"
                "Role: Frontend Engineer\n"
                "Years of Experience: 5\n"
                "\n"
                "Summary:\n"
                "Frontend Engineer with 5 years of experience in React, JavaScript, TypeScript.\n"
                "\n"
                "Skills: React, JavaScript, TypeScript, CSS, HTML, Next.js\n"
                "\n"
                "Experience:\n"
                "- Acme Corp — Frontend Engineer, 5 years\n"
                "  Worked on projects involving React, TypeScript, Next.js.\n"
                "\n"
                "Education:\n"
                "- B.S. Computer Science, State University\n"
            ),
        },
        {
            "candidate_id": "c2",
            "resume_text": (
                "Name: Bob Smith\n"
                "Candidate ID: c2\n"
                "Role: Backend Engineer\n"
                "Years of Experience: 1\n"
                "\n"
                "Summary:\n"
                "Backend Engineer with 1 year of experience in Python and SQL, some exposure to HTML.\n"
                "\n"
                "Skills: Python, SQL, REST APIs, HTML\n"
                "\n"
                "Experience:\n"
                "- Northwind Labs — Backend Engineer, 1 year\n"
                "  Worked on projects involving Python, REST APIs.\n"
                "\n"
                "Education:\n"
                "- B.S. Computer Science, Tech Institute\n"
            ),
        },
        {
            "candidate_id": "c3",
            "resume_text": (
                "Name: Carol Garcia\n"
                "Candidate ID: c3\n"
                "Role: Product Manager\n"
                "Years of Experience: 4\n"
                "\n"
                "Summary:\n"
                "Product Manager with 4 years of experience in Roadmapping, Agile, Stakeholder Management.\n"
                "\n"
                "Skills: Roadmapping, Agile, Stakeholder Management, JIRA\n"
                "\n"
                "Experience:\n"
                "- Bluepeak Software — Product Manager, 4 years\n"
                "  Worked on projects involving Roadmapping, Agile.\n"
                "\n"
                "Education:\n"
                "- B.A. Information Systems, Riverside College\n"
            ),
        },
    ]


@pytest.fixture
def ranked_sample_candidates(sample_requirements, sample_candidates):
    return screening_tools.rank_candidates(sample_requirements, sample_candidates)


@pytest.fixture
def real_frontend_jd_text():
    return Path("data/job_descriptions/frontend_react.txt").read_text()
