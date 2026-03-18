import unittest

from qeu_bundling.pipeline.phase_05_recipe_scoring import (
    _candidate_score,
    _compile_alias_pattern,
    _coverage_penalty,
    _filter_flavor_tokens,
    _is_processed_sweet,
    _normalise,
)


class Phase05RecipeScoringTests(unittest.TestCase):
    def test_token_boundary_prevents_licorice_false_match(self):
        pattern = _compile_alias_pattern("rice")
        self.assertIsNone(pattern.search(_normalise("licorice candy")))
        self.assertIsNotNone(pattern.search(_normalise("basmati rice")))

    def test_low_signal_penalty_applies_when_specific_exists(self):
        low_signal = {
            "recipe_score": 30.0,
            "token_count": 1,
            "alias_len": 4,
            "ingredient_key": "salt",
        }
        strong = {
            "recipe_score": 30.0,
            "token_count": 1,
            "alias_len": 7,
            "ingredient_key": "chicken",
        }
        low_with_penalty = _candidate_score("salt", low_signal, has_stronger_specific=True)
        low_without_penalty = _candidate_score("salt", low_signal, has_stronger_specific=False)
        strong_score = _candidate_score("chicken", strong, has_stronger_specific=True)

        self.assertLess(low_with_penalty, low_without_penalty)
        self.assertLess(low_with_penalty, strong_score)

    def test_coverage_penalty_reduces_staple_dominance(self):
        low_share = _coverage_penalty(0.01, "rice")
        high_share = _coverage_penalty(0.20, "rice")
        self.assertLess(high_share, low_share)
        self.assertGreaterEqual(high_share, 0.60)

    def test_processed_sweet_flavor_tokens_filtered(self):
        self.assertTrue(_is_processed_sweet("orange jelly dessert", "sweet_snacks"))
        filtered = _filter_flavor_tokens(["orange", "jelly", "dessert"], is_processed_sweet=True)
        self.assertNotIn("orange", filtered)
        self.assertIn("jelly", filtered)


if __name__ == "__main__":
    unittest.main()
