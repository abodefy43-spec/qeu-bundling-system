import unittest
from unittest.mock import patch

from qeu_bundling.api.server import (
    CustomerNotFoundError,
    InsufficientHistoryError,
    app,
)


class ApiServerTests(unittest.TestCase):
    def test_healthz_returns_ok_json(self):
        client = app.test_client()
        res = client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "application/json")
        self.assertEqual(res.get_json(), {"status": "ok"})

    def test_recommendations_endpoint_returns_minimal_json_and_caps_to_three(self):
        raw_bundles = [
            {
                "product_a": 10123,
                "product_b": 20456,
                "product_a_price": 20.00,
                "product_b_price": 12.00,
                "price_after_discount_a": 13.45,
                "price_after_discount_b": 0.0,
            },
            {
                "product_a": 30111,
                "product_b": 30199,
                "product_a_price": 9.00,
                "product_b_price": 4.00,
                "price_after_discount_a": 7.58,
                "price_after_discount_b": 0.0,
            },
            {
                "product_a": 50123,
                "product_b": 50188,
                "product_a_price": 11.00,
                "product_b_price": 6.00,
                "price_after_discount_a": 7.90,
                "price_after_discount_b": 0.0,
            },
            {
                "product_a": 90001,
                "product_b": 90002,
                "product_a_price": 50.00,
                "product_b_price": 30.00,
                "price_after_discount_a": 35.00,
                "price_after_discount_b": 0.0,
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
                "free_item_id": 20456,
                "paid_item_discount_amount": 6.55,
            },
        )
        self.assertEqual(
            payload["bundles"][1],
            {
                "item_1_id": 30111,
                "item_2_id": 30199,
                "free_item_id": 30199,
                "paid_item_discount_amount": 1.42,
            },
        )
        self.assertEqual(
            payload["bundles"][2],
            {
                "item_1_id": 50123,
                "item_2_id": 50188,
                "free_item_id": 50188,
                "paid_item_discount_amount": 3.1,
            },
        )
        self.assertNotIn("html", res.get_data(as_text=True).lower())

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

