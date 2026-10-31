"""Synthetic resume and job-description generator.

Generates a corpus of plain-text resumes (with a ground-truth manifest) and
a handful of matching job descriptions, used throughout the project in
place of a real resume dataset.
"""

import json
import random
from pathlib import Path

ROLE_FAMILIES = {
    "Frontend Engineer": {
        "core_skills": ["React", "JavaScript", "CSS", "HTML"],
        "pool_skills": ["TypeScript", "Redux", "Next.js", "Vue", "Webpack", "Tailwind CSS", "GraphQL"],
    },
    "Backend Engineer": {
        "core_skills": ["Python", "REST APIs", "SQL"],
        "pool_skills": ["Django", "Flask", "PostgreSQL", "Java", "Go", "Microservices", "Redis"],
    },
    "Data Scientist": {
        "core_skills": ["Python", "Machine Learning", "Statistics"],
        "pool_skills": ["Pandas", "Scikit-learn", "TensorFlow", "PyTorch", "SQL", "Deep Learning", "NLP"],
    },
    "DevOps Engineer": {
        "core_skills": ["AWS", "Docker", "CI/CD"],
        "pool_skills": ["Kubernetes", "Terraform", "Jenkins", "Linux", "Ansible", "Azure", "Monitoring"],
    },
    "Product Manager": {
        "core_skills": ["Roadmapping", "Agile", "Stakeholder Management"],
        "pool_skills": ["SQL", "A/B Testing", "User Research", "JIRA", "Analytics", "Wireframing"],
    },
}

FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Elena", "Frank", "Grace", "Henry",
    "Isabel", "Jack", "Karen", "Liam", "Maria", "Nathan", "Olivia", "Priya",
    "Quinn", "Rosa", "Sam", "Tara", "Uma", "Victor", "Wendy", "Yusuf",
]
LAST_NAMES = [
    "Chen", "Smith", "Garcia", "Kim", "Patel", "Johnson", "Nguyen", "Brown",
    "Davis", "Martinez", "Lee", "Walker", "Hall", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nelson", "Baker", "Rivera", "Cooper", "Reed",
]

DEGREES = ["B.S. Computer Science", "B.A. Information Systems", "M.S. Computer Science", "B.S. Data Science"]
SCHOOLS = ["State University", "Tech Institute", "Riverside College", "Lakeside University"]
COMPANIES = ["Northwind Labs", "Bluepeak Software", "Riverstone Systems", "Alloy Digital", "Cedar Analytics"]


def render_resume_text(candidate: dict) -> str:
    """Render a candidate dict into plain-text resume content."""
    skills = candidate["skills"]
    highlight_skills = ", ".join(skills[:3])
    all_skills = ", ".join(skills)
    bullet_skills = ", ".join(skills[: min(3, len(skills))])
    company = candidate["company"]
    degree = candidate["degree"]
    school = candidate["school"]

    return (
        f"Name: {candidate['name']}\n"
        f"Candidate ID: {candidate['candidate_id']}\n"
        f"Role: {candidate['role_family']}\n"
        f"Years of Experience: {candidate['years_experience']}\n"
        "\n"
        "Summary:\n"
        f"{candidate['role_family']} with {candidate['years_experience']} years of experience in "
        f"{highlight_skills}.\n"
        "\n"
        f"Skills: {all_skills}\n"
        "\n"
        "Experience:\n"
        f"- {company} — {candidate['role_family']}, {candidate['years_experience']} years\n"
        f"  Worked on projects involving {bullet_skills}.\n"
        "\n"
        "Education:\n"
        f"- {degree}, {school}\n"
    )


def _generate_candidate(candidate_id: str, rng: random.Random) -> dict:
    role_family = rng.choice(list(ROLE_FAMILIES.keys()))
    family = ROLE_FAMILIES[role_family]
    core_skills = list(family["core_skills"])
    pool_skills = family["pool_skills"]
    sample_size = rng.randint(2, len(pool_skills))
    skills = core_skills + rng.sample(pool_skills, sample_size)

    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

    candidate = {
        "candidate_id": candidate_id,
        "name": name,
        "role_family": role_family,
        "years_experience": rng.randint(0, 15),
        "skills": skills,
        "company": rng.choice(COMPANIES),
        "degree": rng.choice(DEGREES),
        "school": rng.choice(SCHOOLS),
    }
    return candidate


_JOB_DESCRIPTIONS = {
    "frontend_react.txt": (
        "Senior Frontend Engineer (React)\n"
        "\n"
        "We are looking for a Senior Frontend Engineer to join our product team.\n"
        "\n"
        "Must-have:\n"
        "- 3+ years of experience building production web applications\n"
        "- Strong hands-on experience with React\n"
        "- Proficiency in JavaScript, HTML, and CSS\n"
        "\n"
        "Nice-to-have:\n"
        "- Experience with TypeScript\n"
        "- Experience with Next.js or Redux\n"
        "- Familiarity with GraphQL\n"
    ),
    "backend_python.txt": (
        "Backend Engineer (Python)\n"
        "\n"
        "We are hiring a Backend Engineer to build and maintain our core services.\n"
        "\n"
        "Must-have:\n"
        "- Strong Python experience\n"
        "- Experience designing REST APIs\n"
        "- Solid SQL skills\n"
        "\n"
        "Nice-to-have:\n"
        "- Experience with Django or Flask\n"
        "- Experience with microservices architectures\n"
        "- Familiarity with PostgreSQL and Redis\n"
    ),
    "data_scientist.txt": (
        "Data Scientist\n"
        "\n"
        "We are looking for a Data Scientist to drive insights from our data.\n"
        "\n"
        "Must-have:\n"
        "- Strong Python experience\n"
        "- Experience with Machine Learning\n"
        "- Solid grounding in Statistics\n"
        "\n"
        "Nice-to-have:\n"
        "- Experience with Pandas and Scikit-learn\n"
        "- Experience with Deep Learning or NLP\n"
        "- Familiarity with TensorFlow or PyTorch\n"
    ),
    "devops_engineer.txt": (
        "DevOps Engineer\n"
        "\n"
        "We are looking for a DevOps Engineer to own our cloud infrastructure.\n"
        "\n"
        "Must-have:\n"
        "- Strong experience with AWS\n"
        "- Experience with Docker\n"
        "- Experience building CI/CD pipelines\n"
        "\n"
        "Nice-to-have:\n"
        "- Experience with Kubernetes and Terraform\n"
        "- Experience with Linux administration\n"
        "- Familiarity with Ansible or Azure\n"
    ),
}


def generate_dataset(n_resumes: int = 100, seed: int = 42, out_dir: str = "data") -> None:
    """Generate a synthetic resume corpus and matching job descriptions.

    Writes ``out_dir/resumes/{candidate_id}.txt`` for each candidate,
    ``out_dir/resumes/manifest.json`` with the ground-truth metadata for
    every candidate, and four fixed job description files under
    ``out_dir/job_descriptions/``.
    """
    rng = random.Random(seed)
    base = Path(out_dir)
    resumes_dir = base / "resumes"
    jd_dir = base / "job_descriptions"
    resumes_dir.mkdir(parents=True, exist_ok=True)
    jd_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    width = len(str(n_resumes))
    for i in range(1, n_resumes + 1):
        candidate_id = f"C{i:0{width}d}"
        candidate = _generate_candidate(candidate_id, rng)
        resume_text = render_resume_text(candidate)
        (resumes_dir / f"{candidate_id}.txt").write_text(resume_text, encoding="utf-8")
        manifest.append(
            {
                "candidate_id": candidate["candidate_id"],
                "name": candidate["name"],
                "role_family": candidate["role_family"],
                "years_experience": candidate["years_experience"],
                "skills": candidate["skills"],
            }
        )

    (resumes_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    for filename, text in _JOB_DESCRIPTIONS.items():
        (jd_dir / filename).write_text(text, encoding="utf-8")
