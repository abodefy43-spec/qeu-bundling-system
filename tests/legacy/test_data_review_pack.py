import tempfile
import unittest
from pathlib import Path

import pandas as pd

from qeu_bundling.config.paths import ensure_layout, get_paths
from qeu_bundling.core.data_review_pack import generate_review_pack
from qeu_bundling.core.run_manifest import write_run_manifest


class DataReviewPackTests(unittest.TestCase):
    def test_generate_review_pack_outputs_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            paths = get_paths(project_root=base)
            ensure_layout(paths)

            scored = paths.output_dir / "person_candidates_scored.csv"
            pairs = paths.data_processed_candidates_dir / "person_candidate_pairs.csv"
            recipe = paths.data_processed_dir / "product_recipe_scores.csv"
            categories = paths.data_processed_dir / "product_categories.csv"
            suspicious = paths.data_processed_diagnostics_dir / "suspicious_pairs_audit.csv"

            pd.DataFrame(
                [
                    {
                        "product_a": 1,
                        "product_b": 2,
                        "product_a_name": "dates",
                        "product_b_name": "qishta",
                        "product_family_a": "dates",
                        "product_family_b": "dairy",
                        "category_a": "fruits",
                        "category_b": "dairy",
                        "purchase_score": 50,
                        "recipe_score_a": 40,
                        "recipe_score_b": 30,
                        "anchor_score": 60,
                        "complement_score": 55,
                        "final_score": 80,
                    }
                ]
            ).to_csv(scored, index=False)
            pd.DataFrame(
                [
                    {
                        "product_a": 1,
                        "product_b": 2,
                        "product_a_name": "dates",
                        "product_b_name": "qishta",
                        "final_score": 80,
                    }
                ]
            ).to_csv(pairs, index=False)
            pd.DataFrame(
                [{"product_id": 1, "product_name": "dates", "matched_ingredient": "dates", "recipe_score": 44.0}]
            ).to_csv(recipe, index=False)
            pd.DataFrame(
                [{"product_id": 1, "category": "fruits", "category_tags": "ramadan|snacks", "product_family": "dates"}]
            ).to_csv(categories, index=False)
            pd.DataFrame(
                [
                    {
                        "anchor_product_id": 1,
                        "complement_product_id": 2,
                        "anchor_name": "rice",
                        "complement_name": "butter",
                        "penalty_multiplier": 0.78,
                    }
                ]
            ).to_csv(suspicious, index=False)

            write_run_manifest(
                mode="quick",
                run_id="quick_reviewpack",
                seed=11,
                started_at="2026-03-04T00:00:00+00:00",
                finished_at="2026-03-04T00:01:00+00:00",
                phases=[],
                artifact_paths={
                    "person_candidates_scored": scored,
                    "person_candidate_pairs": pairs,
                    "suspicious_pairs_audit": suspicious,
                },
                base_dir=base,
            )

            payload = generate_review_pack(base_dir=base)
            self.assertIn("data_inventory", payload)
            self.assertTrue(Path(payload["data_inventory"]).exists())
            self.assertTrue(Path(payload["pair_scoring_breakdown"]).exists())
            self.assertTrue(Path(payload["suspicious_pairs_audit"]).exists())


if __name__ == "__main__":
    unittest.main()
