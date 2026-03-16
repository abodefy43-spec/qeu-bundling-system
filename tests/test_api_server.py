import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qeu_bundling.api.server import (
    CustomerNotFoundError,
    InsufficientHistoryError,
    SERVING_STATE,
    ServiceNotReadyError,
    ServingAssets,
    _initialize_serving_state,
    _recommendation_records_for_user,
    app,
)


class ApiServerTests(unittest.TestCase):
    def setUp(self):
        SERVING_STATE.ready = False
        SERVING_STATE.error = ""
        SERVING_STATE.initialized_at = ""
        SERVING_STATE.base_dir = None
        SERVING_STATE.run_id = ""
        SERVING_STATE.artifact_meta = {}
        SERVING_STATE.recommendations_by_user = {}

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
            return_value={"ready": True, "error": "", "artifacts": {}, "loaded_user_count": 12},
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
            return_value={"ready": False, "error": "boom", "artifacts": {}, "loaded_user_count": 0},
        ):
            client = app.test_client()
            res = client.get("/readyz")

        self.assertEqual(res.status_code, 503)
        payload = res.get_json()
        self.assertEqual(payload["status"], "not_ready")
        self.assertFalse(payload["ready"])

    def test_initialize_serving_state_fails_when_final_artifact_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            with patch("qeu_bundling.api.server._project_root", return_value=base_dir), patch(
                "qeu_bundling.api.server._bootstrap_runtime_artifacts_from_s3_once",
                return_value=None,
            ):
                with self.assertRaises(FileNotFoundError):
                    _initialize_serving_state(force_reload=True)

    def test_initialize_serving_state_fails_when_final_artifact_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            out_dir = base_dir / "output"
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "final_recommendations_by_user.json").write_text("{not-json}", encoding="utf-8")
            with patch("qeu_bundling.api.server._project_root", return_value=base_dir), patch(
                "qeu_bundling.api.server._bootstrap_runtime_artifacts_from_s3_once",
                return_value=None,
            ):
                with self.assertRaises(ValueError):
                    _initialize_serving_state(force_reload=True)

    def test_recommendations_endpoint_returns_precomputed_bundles(self):
        precomputed = [
            {"item_1_id": 10123, "item_2_id": 20456, "bundle_price": 12.0},
            {"item_1_id": 30111, "item_2_id": 30199, "bundle_price": 6.0},
            {"item_1_id": 50123, "item_2_id": 50188, "bundle_price": 7.0},
            {"item_1_id": 90001, "item_2_id": 90002, "bundle_price": 1.0},
        ]
        with patch("qeu_bundling.api.server._recommendation_records_for_user", return_value=precomputed):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "application/json")
        payload = res.get_json()
        self.assertEqual(payload["user_id"], 332323)
        self.assertEqual(len(payload["bundles"]), 3)
        self.assertEqual(payload["bundles"][0], {"item_1_id": 10123, "item_2_id": 20456, "bundle_price": 12.0})

    def test_recommendations_endpoint_does_not_run_initialization(self):
        with patch(
            "qeu_bundling.api.server._initialize_serving_state",
            side_effect=AssertionError("should_not_initialize_during_request"),
        ), patch(
            "qeu_bundling.api.server._recommendation_records_for_user",
            return_value=[{"item_1_id": 11, "item_2_id": 22, "bundle_price": 5.0}],
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

    def test_recommendation_records_lookup_uses_preloaded_state_only(self):
        assets = ServingAssets(
            base_dir=Path("/tmp"),
            run_id="run-test",
            recommendations_by_user={
                332323: [
                    {"item_1_id": 1001, "item_2_id": 1002, "bundle_price": 9.0},
                    {"item_1_id": 1003, "item_2_id": 1004, "bundle_price": 12.5},
                ]
            },
        )
        with patch("qeu_bundling.api.server._get_serving_assets", return_value=assets), patch(
            "qeu_bundling.api.server._bootstrap_runtime_artifacts_from_s3_once",
            side_effect=AssertionError("request_path_must_not_bootstrap"),
        ):
            bundles = _recommendation_records_for_user(332323)

        self.assertEqual(len(bundles), 2)
        self.assertEqual(int(bundles[0]["item_1_id"]), 1001)

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

    def test_invalid_user_payload_returns_customer_not_found(self):
        client = app.test_client()
        res = client.post("/api/recommendations/by-customer", json={"user_id": "<valid_user_from_filtered_orders>"})
        self.assertEqual(res.status_code, 404)
        self.assertEqual(
            res.get_json(),
            {"user_id": None, "bundles": [], "error": "customer_not_found"},
        )

    def test_loader_style_payload_is_accepted_in_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            output_dir = base_dir / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": 1,
                "generated_at": "2026-03-16T00:00:00+00:00",
                "run_id": "full_demo",
                "recommendations_by_user": {
                    "332323": [
                        {"item_1_id": 101, "item_2_id": 202, "bundle_price": 7.5},
                    ]
                },
            }
            (output_dir / "final_recommendations_by_user.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            with patch("qeu_bundling.api.server._project_root", return_value=base_dir), patch(
                "qeu_bundling.api.server._bootstrap_runtime_artifacts_from_s3_once",
                return_value=None,
            ):
                _initialize_serving_state(force_reload=True)
                bundles = _recommendation_records_for_user(332323)

            self.assertEqual(len(bundles), 1)
            self.assertEqual(int(bundles[0]["item_1_id"]), 101)


if __name__ == "__main__":
    unittest.main()
