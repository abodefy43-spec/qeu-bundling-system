import unittest

from qeu_bundling.pipeline.phase_07_train_models import _classifier_candidates, _regressor_candidates


class Phase07ModelCandidateTests(unittest.TestCase):
    def test_classifier_candidates_include_rf_and_xgb(self):
        candidates = _classifier_candidates(random_state=42)
        self.assertIn("RandomForest", candidates)
        self.assertIn("XGBoost", candidates)

    def test_regressor_candidates_include_rf_and_xgb(self):
        candidates = _regressor_candidates(random_state=42)
        self.assertIn("RandomForest", candidates)
        self.assertIn("XGBoost", candidates)


if __name__ == "__main__":
    unittest.main()
