import tempfile
import unittest
from pathlib import Path

from qeu_bundling.core.feedback_memory import (
    append_feedback_row,
    build_pair_multiplier_lookup,
    load_feedback,
    pair_feedback_multiplier,
)


class FeedbackMemoryTests(unittest.TestCase):
    def test_append_and_multiplier(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "data" / "processed").mkdir(parents=True, exist_ok=True)

            append_feedback_row(
                profile_id="p1",
                anchor_product_id=1,
                complement_product_id=2,
                product_a_name="Dates Premium",
                product_b_name="Qishta",
                feedback_type="like",
                base_dir=base,
            )
            append_feedback_row(
                profile_id="p1",
                anchor_product_id=1,
                complement_product_id=3,
                product_a_name="Tuna",
                product_b_name="Salt",
                feedback_type="wrong_pair",
                base_dir=base,
            )
            df = load_feedback(base_dir=base)
            self.assertEqual(len(df), 2)

            lookup = build_pair_multiplier_lookup(base_dir=base)
            good_mult, _ = pair_feedback_multiplier("dates premium", "qishta", lookup)
            bad_mult, bad_conflict = pair_feedback_multiplier("tuna", "salt", lookup)
            self.assertGreater(good_mult, 1.0)
            self.assertLess(bad_mult, 1.0)
            self.assertTrue(bad_conflict)


if __name__ == "__main__":
    unittest.main()
