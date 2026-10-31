"""Filesystem tools over the resume corpus (Milestone 1)."""

from pathlib import Path

from langchain_core.tools import tool


def list_resume_files(data_dir: str = "data/resumes") -> list[str]:
    """List resume filenames (sorted), excluding manifest.json and non-.txt files."""
    return sorted(p.name for p in Path(data_dir).glob("*.txt"))


def read_resume_file(filename: str, data_dir: str = "data/resumes") -> str:
    """Read a single resume file's content by filename, guarding path traversal.

    Character checks alone ("/", "\\", leading ".") are not sufficient on
    Windows: a drive-relative segment like "C:secret.txt" contains none of
    them, but Path(data_dir) / "C:secret.txt" discards a relative data_dir
    entirely and resolves against the cwd instead. The containment check
    below (resolve both paths, require the result's parent to equal the
    resolved base) catches that case regardless of platform or form.
    """
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise ValueError(f"invalid filename: {filename!r}")
    base = Path(data_dir).resolve()
    target = (base / filename).resolve()
    if target.parent != base:
        raise ValueError(f"invalid filename: {filename!r}")
    return target.read_text(encoding="utf-8")


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
