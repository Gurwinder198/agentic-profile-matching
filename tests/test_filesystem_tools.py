from pathlib import Path

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


def test_read_resume_file_rejects_windows_drive_relative_escape():
    """A Windows drive-relative path (e.g. "C:secret.txt") carries no '/'
    or '\\', but when data_dir is itself relative (the real default,
    "data/resumes") and joined naively, it discards the base entirely and
    resolves against the cwd instead, escaping it. The security property
    that matters: the secret's content must never come back, whether the
    fixed implementation raises or the (now-contained) lookup just misses."""
    relative_data_dir = "_test_relative_resumes_dir"
    Path(relative_data_dir).mkdir(exist_ok=True)
    secret_text = "SECRET: should never be readable via read_resume_file"
    cwd_secret = Path.cwd() / "_test_drive_relative_escape.txt"
    cwd_secret.write_text(secret_text)
    try:
        try:
            result = fs_tools.read_resume_file("C:_test_drive_relative_escape.txt", relative_data_dir)
        except (ValueError, OSError):
            pass  # any rejection is a safe outcome
        else:
            assert result != secret_text
    finally:
        cwd_secret.unlink()
        Path(relative_data_dir).rmdir()


def test_search_resumes_by_keyword_is_case_insensitive(tmp_path):
    (tmp_path / "c1.txt").write_text("Skills: React, TypeScript")
    (tmp_path / "c2.txt").write_text("Skills: Python, SQL")
    assert fs_tools.search_resumes_by_keyword("react", str(tmp_path)) == ["c1.txt"]
