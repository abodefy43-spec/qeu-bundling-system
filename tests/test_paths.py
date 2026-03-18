from pathlib import Path

from data.paths import detect_legacy_runtime_paths, ensure_project_layout, get_project_paths


def test_ensure_project_layout_creates_canonical_directories(tmp_path: Path):
    paths = ensure_project_layout(get_project_paths(project_root=tmp_path))

    assert paths.raw_dir.exists()
    assert paths.reference_dir.exists()
    assert paths.features_dir.exists()
    assert paths.artifacts_dir.exists()
    assert paths.reports_dir.exists()
    assert paths.runs_dir.exists()


def test_detect_legacy_runtime_paths_only_reports_existing_directories(tmp_path: Path):
    (tmp_path / "output").mkdir()
    (tmp_path / "feedback").mkdir()

    assert detect_legacy_runtime_paths(project_root=tmp_path) == ["output", "feedback"]
