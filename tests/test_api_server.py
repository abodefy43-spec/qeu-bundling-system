import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from qeu_bundling.api.server import (
    CustomerNotFoundError,
    InsufficientHistoryError,
    ServingAssets,
    ServiceNotReadyError,
    app,
    _recommendation_records_for_user,
)
from qeu_bundling.presentation.person_predictions import OrderPool


class ApiServerTests(unittest.TestCase):
    def test_healthz_returns_ok_json(self):
        client = app.test_client()
        res = client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "application/json")
        self.assertEqual(res.get_json(), {"status": "ok"})

    def test_healthz_stays_ok_even_when_not_ready(self):
        with patch(
            "qeu_bundling.api.server._initialize_serving_state",
            side_effect=RuntimeError("init-failed"),
        ):
            client = app.test_client()
            res = client.get("/healthz")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "ok"})

    def test_readyz_returns_ready_when_state_is_initialized(self):
        with patch("qeu_bundling.api.server._initialize_serving_state", return_value=None), patch(
            "qeu_bundling.api.server._state_payload",
            return_value={"ready": True, "error": "", "artifacts": {}},
        ):
            client = app.test_client()
            res = client.get("/readyz")

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["ready"])

    def test_readyz_returns_not_ready_when_initialization_fails(self):
        with patch(
            "qeu_bundling.api.server._initialize_serving_state",
            side_effect=RuntimeError("boom"),
        ), patch(
            "qeu_bundling.api.server._state_payload",
            return_value={"ready": False, "error": "boom", "artifacts": {}},
        ):
            client = app.test_client()
            res = client.get("/readyz")

        self.assertEqual(res.status_code, 503)
        payload = res.get_json()
        self.assertEqual(payload["status"], "not_ready")
        self.assertFalse(payload["ready"])

    def test_recommendations_endpoint_returns_item_ids_and_fixed_margin_bundle_price(self):
        raw_bundles = [
            {
                "product_a": 10123,
                "product_b": 20456,
                "product_a_price": 20.00,
                "product_b_price": 12.00,
                "purchase_price_a": 10.00,
                "purchase_price_b": 8.00,
            },
            {
                "product_a": 30111,
                "product_b": 30199,
                "product_a_price": 9.00,
                "product_b_price": 14.00,
                "purchase_price_a": 6.00,
                "purchase_price_b": 4.00,
            },
            {
                "product_a": 50123,
                "product_b": 50188,
                "product_a_price": 11.00,
                "product_b_price": 6.00,
                "purchase_price_a": 6.00,
                "purchase_price_b": 3.00,
            },
            {
                "product_a": 90001,
                "product_b": 90002,
                "product_a_price": 50.00,
                "product_b_price": 30.00,
                "purchase_price_a": 20.00,
                "purchase_price_b": 10.00,
            },
        ]
        with patch("qeu_bundling.api.server._recommendation_records_for_user", return_value=raw_bundles):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "application/json")
        payload = res.get_json()
        self.assertEqual(payload["user_id"], 332323)
        self.assertEqual(len(payload["bundles"]), 3)
        self.assertEqual(
            payload["bundles"][0],
            {
                "item_1_id": 10123,
                "item_2_id": 20456,
                "bundle_price": 12.0,
            },
        )
        self.assertEqual(
            payload["bundles"][1],
            {
                "item_1_id": 30111,
                "item_2_id": 30199,
                "bundle_price": 6.0,
            },
        )
        self.assertEqual(
            payload["bundles"][2],
            {
                "item_1_id": 50123,
                "item_2_id": 50188,
                "bundle_price": 7.0,
            },
        )
        self.assertNotIn("html", res.get_data(as_text=True).lower())

    def test_recommendations_endpoint_does_not_run_initialization(self):
        raw_bundles = [
            {
                "product_a": 10123,
                "product_b": 20456,
                "product_a_price": 20.00,
                "product_b_price": 12.00,
                "purchase_price_a": 10.00,
                "purchase_price_b": 8.00,
            }
        ]
        with patch(
            "qeu_bundling.api.server._initialize_serving_state",
            side_effect=AssertionError("should_not_initialize_during_request"),
        ), patch(
            "qeu_bundling.api.server._recommendation_records_for_user",
            return_value=raw_bundles,
        ):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 200)

    def test_customer_not_found_error_payload(self):
        with patch("qeu_bundling.api.server._recommendation_records_for_user", side_effect=CustomerNotFoundError):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 404)
        self.assertEqual(
            res.get_json(),
            {"user_id": 332323, "bundles": [], "error": "customer_not_found"},
        )

    def test_insufficient_history_error_payload(self):
        with patch("qeu_bundling.api.server._recommendation_records_for_user", side_effect=InsufficientHistoryError):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 422)
        self.assertEqual(
            res.get_json(),
            {"user_id": 332323, "bundles": [], "error": "insufficient_history"},
        )

    def test_service_not_ready_error_payload(self):
        with patch("qeu_bundling.api.server._recommendation_records_for_user", side_effect=ServiceNotReadyError):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 503)
        self.assertEqual(
            res.get_json(),
            {"user_id": 332323, "bundles": [], "error": "service_not_ready"},
        )

    def test_recommendation_records_use_preloaded_assets_without_bootstrap(self):
        assets = ServingAssets(
            base_dir=Path("/tmp"),
            run_id="run-test",
            order_pool=OrderPool(
                order_product_ids={11: (1001, 1002)},
                order_product_names={11: ("Item A", "Item B")},
                preferred_order_ids=(11,),
                fallback_order_ids=(11,),
            ),
            orders_df=pd.DataFrame({"order_id": [11], "user_id": [332323]}),
            bundles_df=pd.DataFrame({"product_a": [1001], "product_b": [1002]}),
        )
        with patch("qeu_bundling.api.server._get_serving_assets", return_value=assets), patch(
            "qeu_bundling.api.server._bootstrap_runtime_artifacts_from_s3_once",
            side_effect=AssertionError("request_path_must_not_bootstrap"),
        ), patch(
            "qeu_bundling.api.server.build_recommendations_for_profiles",
            return_value=[
                {
                    "bundles": [
                        {
                            "product_a": 1001,
                            "product_b": 1002,
                            "product_a_price": 20.0,
                            "product_b_price": 12.0,
                            "purchase_price_a": 10.0,
                            "purchase_price_b": 8.0,
                        }
                    ]
                }
            ],
        ):
            bundles = _recommendation_records_for_user(332323)

        self.assertEqual(len(bundles), 1)
        self.assertEqual(int(bundles[0]["product_a"]), 1001)

    def test_server_error_returns_json_no_html(self):
        with patch("qeu_bundling.api.server._recommendation_records_for_user", side_effect=RuntimeError("boom")):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 500)
        self.assertEqual(res.mimetype, "application/json")
        body = res.get_json()
        self.assertEqual(body["error"], "recommendation_failed")
        self.assertEqual(body["bundles"], [])
        self.assertNotIn("<html", res.get_data(as_text=True).lower())


if __name__ == "__main__":
    unittest.main()
