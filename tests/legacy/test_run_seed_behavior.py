import unittest
from unittest.mock import patch

import pandas as pd

from qeu_bundling.pipeline.phase_06_bundle_selection import _shuffle_candidate_pool
from qeu_bundling.pipeline.phase_08_predict import _sample_ranked_pool
from qeu_bundling.runners.run_new_results import resolve_quick_seed
from qeu_bundling.runners.run_pipeline import resolve_full_seed


class RunSeedBehaviorTests(unittest.TestCase):
    def test_quick_seed_auto_generation_and_override(self):
        with patch("qeu_bundling.runners.run_new_results.secrets.randbelow", return_value=1234):
            self.assertEqual(resolve_quick_seed(None), 1235)
        self.assertEqual(resolve_quick_seed(777), 777)

    def test_full_seed_default_and_override(self):
        self.assertEqual(resolve_full_seed(None), 42)
        self.assertEqual(resolve_full_seed(9), 9)

    def test_phase6_shuffle_respects_seed(self):
        df = pd.DataFrame({"x": list(range(30))})
        a = _shuffle_candidate_pool(df, run_seed=55, max_rows=30)["x"].tolist()
        b = _shuffle_candidate_pool(df, run_seed=55, max_rows=30)["x"].tolist()
        c = _shuffle_candidate_pool(df, run_seed=56, max_rows=30)["x"].tolist()
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_phase8_shuffle_respects_seed(self):
        ranked = pd.DataFrame({"x": list(range(30))})
        a = _sample_ranked_pool(ranked, max_rows=30, run_seed=11)["x"].tolist()
        b = _sample_ranked_pool(ranked, max_rows=30, run_seed=11)["x"].tolist()
        c = _sample_ranked_pool(ranked, max_rows=30, run_seed=12)["x"].tolist()
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
