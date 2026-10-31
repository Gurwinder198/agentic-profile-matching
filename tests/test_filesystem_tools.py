import pytest

from tools import filesystem_tools as fs_tools


def test_list_resume_files_excludes_manifest(tmp_path):
    (tmp_path / "c1.txt").write_text("...")
    (tmp_path / "c2.txt").write_text("...")
    (tmp_path / "manifest.json").write_text("[]")
    assert fs_tools.list_resume_files(str(tmp_path)) == ["c1.txt", "c2.txt"]


def test_read_resume_file_returns_content(tmp_path):
    (tmp_path / "c1.txt").write_text("Name: Alice")
    assert fs_tools.read_resume_file("c1.txt", str(tmp_path)) == "Name: Alice"


def test_read_resume_file_rejects_path_traversal(tmp_path):
    with pytest.raises(ValueError, match="invalid filename"):
        fs_tools.read_resume_file("../secrets.txt", str(tmp_path))
    with pytest.raises(ValueError, match="invalid filename"):
        fs_tools.read_resume_file("sub/dir.txt", str(tmp_path))


def test_search_resumes_by_keyword_is_case_insensitive(tmp_path):
    (tmp_path / "c1.txt").write_text("Skills: React, TypeScript")
    (tmp_path / "c2.txt").write_text("Skills: Python, SQL")
    assert fs_tools.search_resumes_by_keyword("react", str(tmp_path)) == ["c1.txt"]
