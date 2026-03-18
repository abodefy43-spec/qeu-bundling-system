import unittest

import pandas as pd

from qeu_bundling.pipeline.phase_06_bundle_selection import (
    ANCHOR_RETENTION_PASS_A,
    RECIPE_ONLY_MIN_SCORE_NORM,
    SWEET_SAVORY_OVERRIDE_MIN_CP_SCORE,
    SWEET_SAVORY_OVERRIDE_MIN_PAIR_COUNT,
    _derive_parent_group,
    _embedding_band_score,
    _is_processed_sweet,
    _is_savory_base,
    _parent_multiplier,
    _passes_cheap_constraint,
    _retain_anchor_aware_candidates,
    _recipe_only_gate,
    _sweet_savory_block_status,
)


class BundleStrategyHelperTests(unittest.TestCase):
    def test_cheap_constraint(self):
        self.assertTrue(_passes_cheap_constraint(20.0, 10.0))
        self.assertFalse(_passes_cheap_constraint(20.0, 10.5))
        self.assertFalse(_passes_cheap_constraint(0.0, 1.0))

    def test_embedding_band_score_boundaries(self):
        self.assertEqual(_embedding_band_score(3.0), 0.0)
        self.assertGreater(_embedding_band_score(20.0), _embedding_band_score(10.0))
        self.assertGreater(_embedding_band_score(25.0), _embedding_band_score(45.0))
        self.assertEqual(_embedding_band_score(60.0), 0.0)

    def test_parent_group_and_penalty(self):
        a = _derive_parent_group("milk_dairy", "dairy")
        b = _derive_parent_group("cheese", "dairy")
        c = _derive_parent_group("chips_snacks", "snacks")
        self.assertEqual(a, "cow_produce")
        self.assertEqual(b, "cow_produce")
        self.assertEqual(c, "snack")
        self.assertLess(_parent_multiplier(a, b), 1.0)
        self.assertGreater(_parent_multiplier(a, c), 1.0)

    def test_sweet_savory_hard_block_and_override_thresholds(self):
        blocked = _is_savory_base("tomato paste", "sauce_base") and _is_processed_sweet("orange jelly", "dessert")
        self.assertTrue(blocked)
        self.assertTrue(SWEET_SAVORY_OVERRIDE_MIN_PAIR_COUNT >= 3)
        self.assertTrue(SWEET_SAVORY_OVERRIDE_MIN_CP_SCORE >= 0.15)
        blocked_pair, override_off = _sweet_savory_block_status(
            name_a="tomato paste",
            family_a="sauce_base",
            name_b="orange jelly",
            family_b="dessert",
            pair_count=0,
            copurchase_score=0.0,
        )
        blocked_pair_override, override_on = _sweet_savory_block_status(
            name_a="tomato paste",
            family_a="sauce_base",
            name_b="orange jelly",
            family_b="dessert",
            pair_count=3,
            copurchase_score=0.0,
        )
        self.assertTrue(blocked_pair)
        self.assertFalse(override_off)
        self.assertTrue(blocked_pair_override)
        self.assertTrue(override_on)

    def test_recipe_only_gate_requires_non_staple_or_high_recipe(self):
        passes, only_staples, _, reason, _ = _recipe_only_gate(
            copurchase_score=0.0,
            known_prior_flag=0,
            recipe_score_norm=0.10,
            ingredient_a="rice",
            ingredient_b="oil",
        )
        self.assertFalse(passes)
        self.assertEqual(only_staples, 1)
        self.assertEqual(reason, "blocked_recipe_only_weak")

        passes2, only_staples2, tokens2, reason2, effective2 = _recipe_only_gate(
            copurchase_score=0.0,
            known_prior_flag=0,
            recipe_score_norm=max(0.30, RECIPE_ONLY_MIN_SCORE_NORM),
            ingredient_a="chicken",
            ingredient_b="rice",
        )
        self.assertTrue(passes2)
        self.assertEqual(only_staples2, 0)
        self.assertIn("chicken", tokens2)
        self.assertEqual(reason2, "recipe_non_staple")
        self.assertGreaterEqual(effective2, RECIPE_ONLY_MIN_SCORE_NORM)

    def test_recipe_only_gate_detects_plural_staples(self):
        passes, only_staples, _, reason, _ = _recipe_only_gate(
            copurchase_score=0.0,
            known_prior_flag=0,
            recipe_score_norm=0.40,
            ingredient_a="tomatoes",
            ingredient_b="onions",
        )
        self.assertFalse(passes)
        self.assertEqual(only_staples, 1)
        self.assertEqual(reason, "blocked_recipe_only_weak")

    def test_anchor_aware_retention_keeps_per_anchor_pass_then_global_fill(self):
        rows = []
        for anchor in (1, 2, 3):
            for idx in range(5):
                rows.append(
                    {
                        "product_a": anchor,
                        "product_b": anchor * 100 + idx,
                        "product_a_name": f"anchor-{anchor}",
                        "product_b_name": f"comp-{idx}",
                        "anchor_score": float(10 - idx),
                        "complement_score": float(20 - idx),
                        "specific_shared_count": 0,
                        "specific_shared_tags": "",
                        "shared_categories": "",
                        "penalized_score": float(100 - (anchor * 10) - idx),
                    }
                )
        df = pd.DataFrame(rows)
        retained, metrics = _retain_anchor_aware_candidates(df, "penalized_score", total_cap=10, per_anchor_cap=4)

        self.assertEqual(len(retained), 10)
        self.assertEqual(metrics["raw_rows"], 15)
        self.assertEqual(metrics["pass_a_selected_rows"], min(10, 3 * ANCHOR_RETENTION_PASS_A))
        counts = retained["product_a"].value_counts().to_dict()
        self.assertGreaterEqual(counts.get(1, 0), 4)
        self.assertGreaterEqual(counts.get(2, 0), 4)
        self.assertIn(3, counts)


if __name__ == "__main__":
    unittest.main()
