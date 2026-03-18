import tempfile
import unittest
from pathlib import Path

import pandas as pd

try:
    import pyarrow  # noqa: F401
except Exception:  # pragma: no cover
    pyarrow = None

from qeu_bundling.core.feedback_event_etl import build_feedback_feature_artifacts


@unittest.skipIf(pyarrow is None, "pyarrow is required for parquet ETL tests")
class FeedbackEventETLTests(unittest.TestCase):
    def test_feedback_event_etl_builds_clean_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "feedback").mkdir(parents=True, exist_ok=True)
            (base / "data" / "processed").mkdir(parents=True, exist_ok=True)

            raw = pd.DataFrame(
                [
                    {
                        "feedback_type": "view",
                        "user_id": "user_alpha",
                        "item_id": "100",
                        "value": "1",
                        "time_stamp": "2026-03-18 10:00:00+00",
                        "updated": "2026-03-18 10:00:00+00",
                        "comment": "",
                    },
                    {
                        "feedback_type": "add_to_cart",
                        "user_id": "user_alpha",
                        "item_id": "100",
                        "value": "1",
                        "time_stamp": "2026-03-18 10:05:00+00",
                        "updated": "2026-03-18 10:05:00+00",
                        "comment": "",
                    },
                    {
                        "feedback_type": "add_to_cart",
                        "user_id": "user_alpha",
                        "item_id": "100",
                        "value": "1",
                        "time_stamp": "2026-03-18 10:05:00+00",
                        "updated": "2026-03-18 10:06:00+00",
                        "comment": "",
                    },
                    {
                        "feedback_type": "purchase",
                        "user_id": "user_alpha",
                        "item_id": "100",
                        "value": "5",
                        "time_stamp": "2026-03-18 10:10:00+00",
                        "updated": "2026-03-18 10:10:00+00",
                        "comment": "",
                    },
                    {
                        "feedback_type": "view",
                        "user_id": "user_alpha",
                        "item_id": "200",
                        "value": "1",
                        "time_stamp": "2026-03-18 10:02:00+00",
                        "updated": "2026-03-18 10:02:00+00",
                        "comment": "",
                    },
                    {
                        "feedback_type": "add_to_cart",
                        "user_id": "user_alpha",
                        "item_id": "200",
                        "value": "1",
                        "time_stamp": "2026-03-18 10:06:00+00",
                        "updated": "2026-03-18 10:06:00+00",
                        "comment": "",
                    },
                    {
                        "feedback_type": "view",
                        "user_id": "user_alpha",
                        "item_id": "300",
                        "value": "1",
                        "time_stamp": "2026-03-18 10:03:00+00",
                        "updated": "2026-03-18 10:03:00+00",
                        "comment": "",
                    },
                    {
                        "feedback_type": "view",
                        "user_id": "sess_demo_1",
                        "item_id": "400",
                        "value": "4",
                        "time_stamp": "2026-03-18 11:00:00+00",
                        "updated": "2026-03-18 11:00:00+00",
                        "comment": "",
                    },
                    {
                        "feedback_type": "add_to_cart",
                        "user_id": "sess_demo_1",
                        "item_id": "400",
                        "value": "4",
                        "time_stamp": "2026-03-18 11:01:00+00",
                        "updated": "2026-03-18 11:01:00+00",
                        "comment": "",
                    },
                    {
                        "feedback_type": "purchase",
                        "user_id": "sess_demo_1",
                        "item_id": "500",
                        "value": "5",
                        "time_stamp": "2026-03-18 11:02:00+00",
                        "updated": "2026-03-18 11:02:00+00",
                        "comment": "",
                    },
                ]
            )
            raw_path = base / "feedback" / "feedback.csv"
            raw.to_csv(raw_path, index=False)

            result = build_feedback_feature_artifacts(
                base_dir=base,
                input_path=raw_path,
                chunk_size=3,
                carryover_rows=1,
                max_pair_items_per_session=10,
                max_view_only_negatives_per_session=10,
            )

            clean = pd.read_parquet(result.clean_events_path)
            actor_session = pd.read_parquet(result.actor_session_features_path)
            actor_item = pd.read_parquet(result.actor_item_features_path)
            session_item = pd.read_parquet(result.session_item_features_path)
            pair_labels = pd.read_parquet(result.proxy_pair_labels_path)

            self.assertEqual(len(clean), 9)
            self.assertEqual(len(actor_session), 2)
            self.assertEqual(int(result.report["clean_report"]["clean_rows"]), 9)

            durable_row = actor_item.loc[
                (actor_item["actor_id"] == "user_alpha") & (actor_item["item_id"] == "100")
            ].iloc[0]
            self.assertEqual(int(durable_row["total_views"]), 1)
            self.assertEqual(int(durable_row["total_add_to_cart"]), 1)
            self.assertEqual(int(durable_row["total_purchases"]), 1)
            self.assertEqual(str(durable_row["actor_type"]), "durable_user_like")

            session_row = actor_session.loc[actor_session["actor_id"] == "sess_demo_1"].iloc[0]
            self.assertEqual(str(session_row["actor_type"]), "session_style")
            self.assertEqual(str(session_row["session_strategy"]), "explicit_session_id")

            viewed_only = session_item.loc[
                (session_item["session_id"].str.startswith("user_alpha|"))
                & (session_item["item_id"] == "300")
            ].iloc[0]
            self.assertEqual(int(viewed_only["viewed_in_session"]), 1)
            self.assertEqual(int(viewed_only["carted_in_session"]), 0)
            self.assertEqual(int(viewed_only["purchased_in_session"]), 0)

            positive_pair = pair_labels.loc[
                (pair_labels["item_a"] == "100") & (pair_labels["item_b"] == "200")
            ].iloc[0]
            self.assertEqual(int(positive_pair["proxy_label"]), 1)
            self.assertGreater(float(positive_pair["label_score"]), 0.0)

            negative_pair = pair_labels.loc[
                (pair_labels["item_a"] == "100") & (pair_labels["item_b"] == "300")
            ].iloc[0]
            self.assertEqual(int(negative_pair["negative_view_only_sessions"]), 1)


if __name__ == "__main__":
    unittest.main()
