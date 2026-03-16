import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QEU_API_EAGER_INIT", "0")

from qeu_bundling.api.server import (
    FallbackUnavailableError,
    SERVING_STATE,
    ServiceNotReadyError,
    ServingAssets,
    _initialize_on_startup_or_exit,
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
        SERVING_STATE.fallback_bundle_bank = []
        SERVING_STATE.bundle_id_lookup = {}

    def test_healthz_returns_ok_json(self):
        client = app.test_client()
        res = client.get("/healthz")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "application/json")
        self.assertEqual(res.get_json(), {"status": "ok"})

    def test_healthz_stays_ok_even_when_not_ready(self):
        with patch("qeu_bundling.api.server._state_payload", return_value={"ready": False, "error": "init-failed"}):
            client = app.test_client()
            res = client.get("/healthz")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), {"status": "ok"})

    def test_readyz_returns_ready_when_state_is_initialized(self):
        with patch(
            "qeu_bundling.api.server._state_payload",
            return_value={"ready": True, "error": "", "artifacts": {}, "loaded_user_count": 12},
        ):
            client = app.test_client()
            res = client.get("/readyz")

        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["status"], "ready")
        self.assertTrue(payload["ready"])

    def test_readyz_returns_not_ready_when_state_not_initialized(self):
        with patch(
            "qeu_bundling.api.server._state_payload",
            return_value={"ready": False, "error": "boom", "artifacts": {}, "loaded_user_count": 0},
        ):
            client = app.test_client()
            res = client.get("/readyz")

        self.assertEqual(res.status_code, 503)
        payload = res.get_json()
        self.assertEqual(payload["status"], "not_ready")
        self.assertFalse(payload["ready"])

    def test_readyz_does_not_trigger_initialization(self):
        with patch(
            "qeu_bundling.api.server._initialize_serving_state",
            side_effect=AssertionError("readyz_must_not_initialize"),
        ), patch(
            "qeu_bundling.api.server._state_payload",
            return_value={"ready": True, "error": "", "artifacts": {}, "loaded_user_count": 1},
        ):
            client = app.test_client()
            res = client.get("/readyz")

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "ready")

    def test_startup_init_exits_process_when_initialization_fails(self):
        with patch("qeu_bundling.api.server._eager_init_enabled", return_value=True), patch(
            "qeu_bundling.api.server._initialize_serving_state",
            side_effect=RuntimeError("startup-broken"),
        ):
            with self.assertRaises(SystemExit) as raised:
                _initialize_on_startup_or_exit()

        self.assertEqual(raised.exception.code, 1)

    def test_startup_init_runs_once_when_enabled(self):
        with patch("qeu_bundling.api.server._eager_init_enabled", return_value=True), patch(
            "qeu_bundling.api.server._initialize_serving_state",
            return_value=None,
        ) as mocked:
            _initialize_on_startup_or_exit()

        mocked.assert_called_once_with(force_reload=False)

    def test_startup_init_can_be_disabled(self):
        with patch("qeu_bundling.api.server._eager_init_enabled", return_value=False), patch(
            "qeu_bundling.api.server._initialize_serving_state",
            side_effect=AssertionError("should_not_initialize"),
        ):
            _initialize_on_startup_or_exit()

    def test_readyz_reflects_runtime_error_without_initializing(self):
        with patch(
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

    def test_initialize_serving_state_allows_missing_fallback_artifact(self):
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
            (output_dir / "final_recommendations_by_user.json").write_text(json.dumps(payload), encoding="utf-8")
            with patch("qeu_bundling.api.server._project_root", return_value=base_dir), patch(
                "qeu_bundling.api.server._bootstrap_runtime_artifacts_from_s3_once",
                return_value=None,
            ):
                _initialize_serving_state(force_reload=True)

            self.assertTrue(SERVING_STATE.ready)
            self.assertEqual(SERVING_STATE.fallback_bundle_bank, [])

    def test_recommendations_endpoint_returns_precomputed_bundles(self):
        precomputed = [
            {"item_1_id": 10123, "item_2_id": 20456, "bundle_price": 12.0},
            {"item_1_id": 30111, "item_2_id": 30199, "bundle_price": 6.0},
            {"item_1_id": 50123, "item_2_id": 50188, "bundle_price": 7.0},
            {"item_1_id": 90001, "item_2_id": 90002, "bundle_price": 1.0},
        ]
        with patch(
            "qeu_bundling.api.server._recommendation_records_for_user",
            return_value=(precomputed, "personalized"),
        ):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.mimetype, "application/json")
        payload = res.get_json()
        self.assertEqual(payload["user_id"], 332323)
        self.assertEqual(payload["source"], "personalized")
        self.assertEqual(len(payload["bundles"]), 3)
        self.assertEqual(payload["bundles"][0], {"item_1_id": 10123, "item_2_id": 20456, "bundle_price": 12.0})

    def test_recommendations_endpoint_does_not_run_initialization(self):
        with patch(
            "qeu_bundling.api.server._initialize_serving_state",
            side_effect=AssertionError("should_not_initialize_during_request"),
        ), patch(
            "qeu_bundling.api.server._recommendation_records_for_user",
            return_value=([{"item_1_id": 11, "item_2_id": 22, "bundle_price": 5.0}], "personalized"),
        ):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 200)

    def test_unknown_user_returns_fallback_response(self):
        fallback = [
            {"item_1_id": 901, "item_2_id": 902, "bundle_price": 9.9},
            {"item_1_id": 903, "item_2_id": 904, "bundle_price": 12.2},
        ]
        with patch(
            "qeu_bundling.api.server._recommendation_records_for_user",
            return_value=(fallback, "fallback"),
        ):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 999999})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.get_json(),
            {"user_id": 999999, "bundles": fallback, "source": "fallback"},
        )

    def test_known_user_with_empty_personalized_returns_fallback_response(self):
        fallback = [{"item_1_id": 1201, "item_2_id": 1202, "bundle_price": 14.5}]
        with patch(
            "qeu_bundling.api.server._recommendation_records_for_user",
            return_value=(fallback, "fallback"),
        ):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.get_json(),
            {"user_id": 332323, "bundles": fallback, "source": "fallback"},
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

    def test_fallback_unavailable_error_payload(self):
        with patch("qeu_bundling.api.server._recommendation_records_for_user", side_effect=FallbackUnavailableError):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": 332323})

        self.assertEqual(res.status_code, 503)
        self.assertEqual(
            res.get_json(),
            {"user_id": 332323, "bundles": [], "error": "fallback_unavailable"},
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
            fallback_bundle_bank=[{"item_1_id": 2001, "item_2_id": 2002, "bundle_price": 8.0}],
        )
        with patch("qeu_bundling.api.server._get_serving_assets", return_value=assets), patch(
            "qeu_bundling.api.server._bootstrap_runtime_artifacts_from_s3_once",
            side_effect=AssertionError("request_path_must_not_bootstrap"),
        ):
            bundles, source = _recommendation_records_for_user(332323)

        self.assertEqual(len(bundles), 3)
        self.assertEqual(source, "personalized")
        self.assertEqual(int(bundles[0]["item_1_id"]), 1001)

    def test_personalized_response_is_backfilled_to_three_when_fallback_available(self):
        assets = ServingAssets(
            base_dir=Path("/tmp"),
            run_id="run-test",
            recommendations_by_user={
                332323: [
                    {"item_1_id": 1001, "item_2_id": 1002, "bundle_price": 9.0},
                    {"item_1_id": 1003, "item_2_id": 1004, "bundle_price": 12.5},
                ]
            },
            fallback_bundle_bank=[{"item_1_id": 2001, "item_2_id": 2002, "bundle_price": 8.0}],
        )
        with patch("qeu_bundling.api.server._get_serving_assets", return_value=assets):
            bundles, source = _recommendation_records_for_user(332323)

        self.assertEqual(source, "personalized")
        self.assertEqual(len(bundles), 3)
        self.assertEqual(int(bundles[2]["item_1_id"]), 2001)

    def test_recommendation_records_include_bundle_id_when_registry_loaded(self):
        assets = ServingAssets(
            base_dir=Path("/tmp"),
            run_id="run-test",
            recommendations_by_user={
                332323: [
                    {"item_1_id": 1001, "item_2_id": 1002, "bundle_price": 9.0},
                ]
            },
            fallback_bundle_bank=[],
            bundle_id_lookup={(1001, 1002): "B00012345"},
        )
        with patch("qeu_bundling.api.server._get_serving_assets", return_value=assets):
            bundles, source = _recommendation_records_for_user(332323)

        self.assertEqual(source, "personalized")
        self.assertEqual(len(bundles), 1)
        self.assertEqual(str(bundles[0].get("bundle_id")), "B00012345")

    def test_unknown_user_uses_fallback_lookup(self):
        assets = ServingAssets(
            base_dir=Path("/tmp"),
            run_id="run-test",
            recommendations_by_user={332323: [{"item_1_id": 11, "item_2_id": 12, "bundle_price": 4.0}]},
            fallback_bundle_bank=[
                {"item_1_id": 2001, "item_2_id": 2002, "bundle_price": 8.0},
                {"item_1_id": 2003, "item_2_id": 2004, "bundle_price": 10.5},
            ],
        )
        with patch("qeu_bundling.api.server._get_serving_assets", return_value=assets):
            bundles, source = _recommendation_records_for_user(999999)

        self.assertEqual(source, "fallback")
        self.assertEqual(len(bundles), 2)
        self.assertEqual(int(bundles[0]["item_1_id"]), 2003)

    def test_known_user_with_empty_personalized_uses_fallback_lookup(self):
        assets = ServingAssets(
            base_dir=Path("/tmp"),
            run_id="run-test",
            recommendations_by_user={332323: []},
            fallback_bundle_bank=[{"item_1_id": 3001, "item_2_id": 3002, "bundle_price": 8.8}],
        )
        with patch("qeu_bundling.api.server._get_serving_assets", return_value=assets):
            bundles, source = _recommendation_records_for_user(332323)

        self.assertEqual(source, "fallback")
        self.assertEqual(len(bundles), 1)
        self.assertEqual(int(bundles[0]["item_1_id"]), 3001)

    def test_unknown_user_without_fallback_returns_degraded_error(self):
        assets = ServingAssets(
            base_dir=Path("/tmp"),
            run_id="run-test",
            recommendations_by_user={},
            fallback_bundle_bank=[],
        )
        with patch("qeu_bundling.api.server._get_serving_assets", return_value=assets):
            with self.assertRaises(FallbackUnavailableError):
                _recommendation_records_for_user(332323)

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

    def test_random_user_keyword_returns_bundles_for_selected_user(self):
        with patch("qeu_bundling.api.server._pick_random_user_id", return_value=778899), patch(
            "qeu_bundling.api.server._recommendation_records_for_user",
            return_value=([{"item_1_id": 11, "item_2_id": 22, "bundle_price": 5.0}], "personalized"),
        ):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"user_id": "random"})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.get_json(),
            {
                "user_id": 778899,
                "bundles": [{"item_1_id": 11, "item_2_id": 22, "bundle_price": 5.0}],
                "source": "personalized",
            },
        )

    def test_random_user_flag_returns_bundles_for_selected_user(self):
        with patch("qeu_bundling.api.server._pick_random_user_id", return_value=445566), patch(
            "qeu_bundling.api.server._recommendation_records_for_user",
            return_value=([{"item_1_id": 101, "item_2_id": 202, "bundle_price": 9.5}], "personalized"),
        ):
            client = app.test_client()
            res = client.post("/api/recommendations/by-customer", json={"random_user": True})

        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.get_json(),
            {
                "user_id": 445566,
                "bundles": [{"item_1_id": 101, "item_2_id": 202, "bundle_price": 9.5}],
                "source": "personalized",
            },
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
                bundles, source = _recommendation_records_for_user(332323)

            self.assertEqual(len(bundles), 1)
            self.assertEqual(source, "personalized")
            self.assertEqual(int(bundles[0]["item_1_id"]), 101)


if __name__ == "__main__":
    unittest.main()
