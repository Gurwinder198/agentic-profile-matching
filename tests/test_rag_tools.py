from tools import rag_tools


def test_rag_search_resumes_ranks_relevant_resume_first(tmp_path):
    resumes_dir = tmp_path / "resumes"
    resumes_dir.mkdir()
    (resumes_dir / "c1.txt").write_text(
        "Frontend Engineer with React, JavaScript, CSS experience building web UIs.")
    (resumes_dir / "c2.txt").write_text(
        "DevOps Engineer with Kubernetes, Docker, AWS, Terraform infrastructure experience.")
    (resumes_dir / "c3.txt").write_text(
        "Data Scientist with Python, Pandas, Machine Learning, statistics experience.")
    persist_dir = tmp_path / "chroma_db"
    results = rag_tools.rag_search_resumes(
        "React frontend developer", k=3,
        resumes_dir=str(resumes_dir), persist_dir=str(persist_dir))
    assert results[0]["candidate_id"] == "c1"
    assert len(results) == 3
