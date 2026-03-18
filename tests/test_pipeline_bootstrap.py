import json
from pathlib import Path

from pipelines.bootstrap import bootstrap_phase0


def test_bootstrap_phase0_writes_status_and_manifest(tmp_path: Path):
    result = bootstrap_phase0(project_root=tmp_path)

    assert result.manifest_path.exists()
    assert result.status_path.exists()
    payload = json.loads(result.status_path.read_text(encoding="utf-8"))
    assert payload["status"] == "ready_for_engine_implementation"
    assert [engine["name"] for engine in payload["engines"]] == [
        "compatible_products",
        "frequently_bought_together",
        "personalized_bundles",
    ]
