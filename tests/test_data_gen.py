import json

import data_gen


def test_generate_dataset_writes_expected_counts(tmp_path):
    data_gen.generate_dataset(n_resumes=20, seed=1, out_dir=str(tmp_path))
    resumes = list((tmp_path / "resumes").glob("*.txt"))
    assert len(resumes) == 20
    manifest = json.loads((tmp_path / "resumes" / "manifest.json").read_text())
    assert len(manifest) == 20
    jds = list((tmp_path / "job_descriptions").glob("*.txt"))
    assert len(jds) == 4


def test_manifest_entries_match_resume_text(tmp_path):
    data_gen.generate_dataset(n_resumes=20, seed=1, out_dir=str(tmp_path))
    manifest = json.loads((tmp_path / "resumes" / "manifest.json").read_text())
    react_3plus = [c for c in manifest if "React" in c["skills"] and c["years_experience"] >= 3]
    assert len(react_3plus) >= 1
    sample = react_3plus[0]
    text = (tmp_path / "resumes" / f"{sample['candidate_id']}.txt").read_text()
    assert "React" in text and sample["name"] in text


def test_generate_dataset_is_deterministic_for_fixed_seed(tmp_path):
    out_a, out_b = tmp_path / "a", tmp_path / "b"
    data_gen.generate_dataset(n_resumes=15, seed=7, out_dir=str(out_a))
    data_gen.generate_dataset(n_resumes=15, seed=7, out_dir=str(out_b))
    assert (out_a / "resumes" / "manifest.json").read_text() == (out_b / "resumes" / "manifest.json").read_text()
