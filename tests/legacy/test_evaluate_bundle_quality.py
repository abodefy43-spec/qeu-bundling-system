import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qeu_bundling.config.paths import latest_manifest_path
from qeu_bundling.core.evaluate_bundle_quality import evaluate_quality
from qeu_bundling.core.run_manifest import resolve_latest_artifact


class EvaluateBundleQualityTests(unittest.TestCase):
    def test_quality_metrics_file_and_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out = base / "output"
            proc = base / "data" / "processed"
            out.mkdir(parents=True, exist_ok=True)
            proc.mkdir(parents=True, exist_ok=True)

            final = pd.DataFrame(
                [
                    {
                        "product_a": 1,
                        "product_b": 2,
                        "product_a_name": "dates",
                        "product_b_name": "qishta",
                        "product_family_a": "dates_family",
                        "product_family_b": "dairy",
                        "purchase_score": 70,
                        "recipe_compat_score": 45,
                        "shared_categories_count": 3,
                        "anchor_score": 60,
                    }
                ]
            )
            final.to_csv(out / "person_candidates_scored.csv", index=False)
            (out / "person_reco_quality.json").write_text(
                json.dumps({"anchor_in_history_rate": 1.0}), encoding="utf-8"
            )

            payload = evaluate_quality(base_dir=base, save=True)
            self.assertIn("critical_gates_passed", payload)
            self.assertTrue((out / "bundle_quality_metrics.json").exists())
            self.assertTrue(bool(payload["critical_gates_passed"]))

    def test_quality_prefers_newer_fallback_person_quality_over_stale_manifest_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out = base / "output"
            proc = base / "data" / "processed"
            stale_run = out / "runs" / "old_run"
            out.mkdir(parents=True, exist_ok=True)
            proc.mkdir(parents=True, exist_ok=True)
            stale_run.mkdir(parents=True, exist_ok=True)

            final = pd.DataFrame(
                [
                    {
                        "product_a": 1,
                        "product_b": 2,
                        "product_a_name": "dates",
                        "product_b_name": "qishta",
                        "product_family_a": "dates_family",
                        "product_family_b": "dairy",
                        "purchase_score": 70,
                        "recipe_compat_score": 45,
                        "shared_categories_count": 3,
                        "anchor_score": 60,
                    }
                ]
            )
            final.to_csv(out / "person_candidates_scored.csv", index=False)

            stale_quality = stale_run / "person_reco_quality.json"
            stale_quality.write_text(json.dumps({"anchor_in_history_rate": 0.0}), encoding="utf-8")
            latest_manifest_path(project_root=base).write_text(
                json.dumps({"artifacts": {"person_reco_quality": str(stale_quality.resolve())}}),
                encoding="utf-8",
            )

            fresh_quality = out / "person_reco_quality.json"
            fresh_quality.write_text(json.dumps({"anchor_in_history_rate": 1.0}), encoding="utf-8")

            payload = evaluate_quality(base_dir=base, save=True)
            self.assertEqual(float(payload["anchor_in_history_rate"]), 1.0)
            self.assertTrue(bool(payload["critical_gates_passed"]))

    def test_resolve_latest_artifact_keeps_same_file_path_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            out = base / "output"
            out.mkdir(parents=True, exist_ok=True)

            artifact = out / "person_reco_quality.json"
            artifact.write_text(json.dumps({"anchor_in_history_rate": 1.0}), encoding="utf-8")
            latest_manifest_path(project_root=base).write_text(
                json.dumps({"artifacts": {"person_reco_quality": str(artifact.resolve())}}),
                encoding="utf-8",
            )

            resolved = resolve_latest_artifact("person_reco_quality", base_dir=base, fallback=artifact)
            self.assertEqual(resolved, artifact)


if __name__ == "__main__":
    unittest.main()
