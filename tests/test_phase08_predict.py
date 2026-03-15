import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from qeu_bundling.core.pricing import ProductPriceRecord
from qeu_bundling.pipeline.phase_08_predict import (
    _apply_margin_safe_pricing,
    _is_weak_evidence_row,
    _join_pictures,
)


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

    def test_apply_margin_safe_pricing_respects_purchase_floor_and_paid_item(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 101,
                    "product_b": 202,
                    "product_a_price": 20.0,
                    "product_b_price": 10.0,
                }
            ]
        )
        odoo_lookup = {
            101: ProductPriceRecord(sale_price=20.0, purchase_price=19.0),
            202: ProductPriceRecord(sale_price=30.0, purchase_price=10.0),
        }

        out = _apply_margin_safe_pricing(bundles, odoo_lookup)
        row = out.iloc[0]

        self.assertNotIn("discount_pred_a", out.columns)
        self.assertNotIn("discount_pred_b", out.columns)
        self.assertEqual(float(row["product_a_price"]), 20.0)
        self.assertEqual(float(row["product_b_price"]), 30.0)
        self.assertEqual(float(row["purchase_price_a"]), 19.0)
        self.assertEqual(float(row["purchase_price_b"]), 10.0)
        self.assertEqual(str(row["free_product"]), "product_a")
        self.assertEqual(str(row["paid_product"]), "product_b")
        self.assertEqual(float(row["price_after_discount_a"]), 0.0)
        # Fixed 80% margin discount for paid item (B): margin=20 => final=10 + 0.2*20 = 14
        self.assertEqual(float(row["price_after_discount_b"]), 14.0)
        self.assertEqual(float(row["margin_discount_pct"]), 80.0)
        self.assertGreaterEqual(float(row["price_after_discount_b"]), float(row["purchase_price_b"]))

    def test_apply_margin_safe_pricing_handles_missing_purchase_cost_safely(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 303,
                    "product_b": 404,
                    "product_a_price": 15.0,
                    "product_b_price": 5.0,
                }
            ]
        )
        # product_a purchase missing -> fallback purchase = sale
        odoo_lookup = {
            303: ProductPriceRecord(sale_price=15.0, purchase_price=0.0),
            404: ProductPriceRecord(sale_price=5.0, purchase_price=2.0),
        }

        out = _apply_margin_safe_pricing(bundles, odoo_lookup)
        row = out.iloc[0]

        self.assertNotIn("discount_pred_a", out.columns)
        self.assertNotIn("discount_pred_b", out.columns)
        self.assertEqual(str(row["paid_product"]), "product_a")
        self.assertEqual(str(row["free_product"]), "product_b")
        self.assertEqual(int(row["purchase_price_missing_a"]), 1)
        self.assertEqual(float(row["purchase_price_a"]), 15.0)
        # Missing purchase cost fallback should prevent below-cost discounting.
        self.assertEqual(float(row["price_after_discount_a"]), 15.0)
        self.assertEqual(float(row["price_after_discount_b"]), 0.0)


if __name__ == "__main__":
    unittest.main()
