import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from qeu_bundling.pipeline.phase_08_predict import _is_weak_evidence_row, _join_pictures


class Phase08PredictTests(unittest.TestCase):
    def test_weak_evidence_free_item2_blocked_condition(self):
        weak = pd.Series(
            {
                "purchase_score": 0.0,
                "recipe_score_norm": 0.10,
                "known_prior_flag": 0,
            }
        )
        strong = pd.Series(
            {
                "purchase_score": 8.0,
                "recipe_score_norm": 0.10,
                "known_prior_flag": 0,
            }
        )
        self.assertTrue(_is_weak_evidence_row(weak))
        self.assertFalse(_is_weak_evidence_row(strong))

    def test_join_pictures_prefers_reference_override(self):
        rows = pd.DataFrame([{"product_a": 10, "product_b": 20}])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "product_pictures.csv").write_text("product_id,picture_url\n10,http://processed/a.png\n20,http://processed/b.png\n", encoding="utf-8")
            ref_dir = base / "ref"
            ref_dir.mkdir(parents=True, exist_ok=True)
            (ref_dir / "product_pictures.csv").write_text("product_id,picture_url\n20,/static/product_images/local-b.png\n", encoding="utf-8")
            with patch("qeu_bundling.pipeline.phase_08_predict.get_paths") as mocked_get_paths:
                mocked_get_paths.return_value = type("Paths", (), {"data_reference_dir": ref_dir})()
                out = _join_pictures(rows, base)

        self.assertEqual(out.loc[0, "product_a_picture"], "http://processed/a.png")
        self.assertEqual(out.loc[0, "product_b_picture"], "/static/product_images/local-b.png")


if __name__ == "__main__":
    unittest.main()
