import unittest
from unittest.mock import patch

import pandas as pd

from qeu_bundling.presentation.app import (
    _DEFAULT_PROFILE_CACHE,
    _DEFAULT_RECOMMENDATION_CACHE,
    app,
    prewarm_local_dashboard_defaults,
)
from qeu_bundling.presentation.bundle_view import BundleViewData
from qeu_bundling.presentation.person_predictions import PersonProfile


class PeopleOnlyUiTests(unittest.TestCase):
    def test_dashboard_is_people_only(self):
        client = app.test_client()
        with patch(
            "qeu_bundling.presentation.app.translate_recommendations_for_dashboard",
            side_effect=lambda recos, cache_path: recos,
        ):
            res = client.get("/")
        self.assertEqual(res.status_code, 200)
        html = res.get_data(as_text=True)
        self.assertIn('dir="rtl"', html)
        self.assertIn("توصيات مخصصة حسب الأشخاص", html)
        self.assertIn("Add +5 Predictions", html)
        self.assertNotIn("Top 10 Compatible Bundles", html)

    def test_bundles_route_redirects_to_dashboard(self):
        client = app.test_client()
        res = client.get("/bundles", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertEqual(res.headers.get("Location"), "/")

    def test_healthz_returns_ok(self):
        client = app.test_client()
        res = client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertIn("default_person_count", payload)
        self.assertIn("latest_run_id", payload)

    def test_add_five_predictions_route_redirects(self):
        client = app.test_client()
        with patch("qeu_bundling.presentation.app._append_random_profiles", return_value=(None, 5)):
            res = client.post("/people/add-five", follow_redirects=False)
        self.assertEqual(res.status_code, 302)
        self.assertEqual(res.headers.get("Location"), "/")

    def test_prewarm_local_dashboard_defaults_seeds_default_cache(self):
        _DEFAULT_PROFILE_CACHE.clear()
        _DEFAULT_RECOMMENDATION_CACHE.clear()
        profiles = [
            PersonProfile(
                profile_id="p1",
                source="random",
                order_ids=[1],
                history_product_ids=[1, 2],
                history_items=["a", "b"],
                created_at="2026-03-11T00:00:00+00:00",
            ),
            PersonProfile(
                profile_id="p2",
                source="random",
                order_ids=[2],
                history_product_ids=[3, 4],
                history_items=["c", "d"],
                created_at="2026-03-11T00:00:00+00:00",
            ),
        ]
        recommendations = [{"person_label": "Person 1", "bundles": []}, {"person_label": "Person 2", "bundles": []}]
        view = BundleViewData(kpis={}, top10_rows=[], all_rows=[], bundles_df=pd.DataFrame([{"product_a": 1, "product_b": 2}]))

        with patch("qeu_bundling.presentation.app.LOCAL_FAST_MODE", True), patch(
            "qeu_bundling.presentation.app.DEFAULT_PERSON_COUNT", 2
        ), patch("qeu_bundling.presentation.app._default_profiles_for_run", return_value=profiles), patch(
            "qeu_bundling.presentation.app.load_bundle_view", return_value=view
        ), patch(
            "qeu_bundling.presentation.app.build_recommendations_for_profiles", return_value=recommendations
        ), patch(
            "qeu_bundling.presentation.app.translate_recommendations_for_dashboard", side_effect=lambda recos, cache_path: recos
        ):
            payload = prewarm_local_dashboard_defaults(run_id="run_local_test")

        self.assertTrue(payload["ready"])
        self.assertEqual(payload["recommendation_count"], 2)
        self.assertEqual(len(_DEFAULT_RECOMMENDATION_CACHE["run_local_test"]), 2)

if __name__ == "__main__":
    unittest.main()
