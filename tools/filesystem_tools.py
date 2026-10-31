"""Filesystem tools over the resume corpus (Milestone 1)."""

from pathlib import Path

from langchain_core.tools import tool


def list_resume_files(data_dir: str = "data/resumes") -> list[str]:
    """List resume filenames (sorted), excluding manifest.json and non-.txt files."""
    return sorted(p.name for p in Path(data_dir).glob("*.txt"))


def read_resume_file(filename: str, data_dir: str = "data/resumes") -> str:
    """Read a single resume file's content by filename, guarding path traversal."""
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise ValueError(f"invalid filename: {filename!r}")
    return (Path(data_dir) / filename).read_text(encoding="utf-8")


def search_resumes_by_keyword(keyword: str, data_dir: str = "data/resumes") -> list[str]:
    """Return filenames of resumes whose content contains keyword (case-insensitive)."""
    keyword_lower = keyword.lower()
    matches = []
    for filename in list_resume_files(data_dir):
        content = read_resume_file(filename, data_dir)
        if keyword_lower in content.lower():
            matches.append(filename)
    return matches


FILESYSTEM_TOOLS = [
    tool(list_resume_files),
    tool(read_resume_file),
    tool(search_resumes_by_keyword),
]
