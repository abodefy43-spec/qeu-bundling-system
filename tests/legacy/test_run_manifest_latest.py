import tempfile
import unittest
from pathlib import Path

from qeu_bundling.config.paths import ensure_layout, get_paths
from qeu_bundling.core.run_manifest import (
    append_seed_history,
    resolve_latest_artifact,
    write_run_manifest,
)


class RunManifestLatestTests(unittest.TestCase):
    def test_latest_pointer_and_seed_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            paths = get_paths(project_root=base)
            ensure_layout(paths)

            artifact = paths.output_dir / "person_candidates_scored.csv"
            artifact.write_text("product_a,product_b\n1,2\n", encoding="utf-8")

            manifest_path = write_run_manifest(
                mode="quick",
                run_id="quick_test123",
                seed=99,
                started_at="2026-03-04T00:00:00+00:00",
                finished_at="2026-03-04T00:01:00+00:00",
                phases=[{"phase": "phase_06", "status": "completed", "duration_sec": 1.2}],
                artifact_paths={"person_candidates_scored": artifact},
                base_dir=base,
            )

            self.assertTrue(manifest_path.exists())
            latest = resolve_latest_artifact(
                "person_candidates_scored",
                base_dir=base,
                fallback=artifact,
            )
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertTrue(latest.exists())

            seed_path = append_seed_history(
                mode="quick",
                run_id="quick_test123",
                seed=99,
                started_at="2026-03-04T00:00:00+00:00",
                finished_at="2026-03-04T00:01:00+00:00",
                base_dir=base,
            )
            self.assertTrue(seed_path.exists())
            csv_text = seed_path.read_text(encoding="utf-8")
            self.assertIn("quick_test123", csv_text)


if __name__ == "__main__":
    unittest.main()
