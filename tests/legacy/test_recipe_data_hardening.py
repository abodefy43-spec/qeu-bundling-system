import unittest

from qeu_bundling.pipeline.phase_01_load_data import _normalise_recipe_data


class RecipeDataHardeningTests(unittest.TestCase):
    def test_missing_required_section_raises(self):
        payload = {
            "ingredients": {},
            "saudi_ramadan_dishes": {},
        }
        with self.assertRaises(ValueError):
            _normalise_recipe_data(payload)

    def test_malformed_ingredient_falls_back_safely(self):
        payload = {
            "saudi_consumption_data": {},
            "saudi_ramadan_dishes": {},
            "ingredients": {
                "dates": {
                    "qeu_relevance": "not-a-number",
                    "saudi_importance": "unknown-level",
                    "product_types": "bad-type",
                    "recipes": ["date cake", "", "date cake"],
                }
            },
        }
        out = _normalise_recipe_data(payload)
        ing = out["ingredients"]["dates"]
        self.assertEqual(ing["qeu_relevance"], 0.0)
        self.assertEqual(ing["saudi_importance"], "medium")
        self.assertEqual(ing["product_types"], [])
        self.assertEqual(ing["recipes"], ["date cake"])
        self.assertEqual(ing["recipe_count"], 1)


if __name__ == "__main__":
    unittest.main()
