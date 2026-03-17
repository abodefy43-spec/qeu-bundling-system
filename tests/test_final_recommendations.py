import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from qeu_bundling.core.final_recommendations import (
    FINAL_RECOMMENDATIONS_MAX_USERS_ENV,
    FINAL_RECOMMENDATIONS_RANDOM_SEED_ENV,
    FINAL_RECOMMENDATIONS_USER_SELECTION_ENV,
    bundle_ids_artifact_path,
    materialize_final_recommendations_by_user,
    resolve_final_recommendations_max_users,
    resolve_final_recommendations_max_users_from_env,
    resolve_final_recommendations_random_seed,
    resolve_final_recommendations_user_selection_mode,
)


class FinalRecommendationsTests(unittest.TestCase):
    @staticmethod
    def _fallback_orders_df() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"order_id": 11, "user_id": 101},
                {"order_id": 22, "user_id": 202},
                {"order_id": 33, "user_id": 303},
                {"order_id": 44, "user_id": 404},
            ]
        )

    @staticmethod
    def _fallback_order_pool() -> SimpleNamespace:
        return SimpleNamespace(
            order_product_ids={
                11: [1001, 1002],
                22: [2001, 2002],
                33: [3001, 3002],
                44: [4001, 4002],
            },
            order_product_names={
                11: ["a", "b"],
                22: ["c", "d"],
                33: ["e", "f"],
                44: ["g", "h"],
            },
        )

    @staticmethod
    def _fallback_scored_df() -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "product_a": 5001,
                    "product_b": 5002,
                    "product_a_price": 15.0,
                    "product_b_price": 8.0,
                    "purchase_price_a": 9.0,
                    "purchase_price_b": 5.0,
                    "new_final_score": 95.0,
                    "model_score": 90.0,
                    "recipe_compat_score": 0.9,
                    "pair_count": 12,
                    "shared_categories_count": 5,
                    "known_prior_flag": 1,
                    "deal_signal": 0.3,
                    "pair_penalty_multiplier": 1.0,
                    "utility_penalty_multiplier": 1.0,
                    "weak_evidence_free_blocked": 0,
                    "only_staples_overlap": 0,
                    "category_a": "tea",
                    "category_b": "snacks",
                },
                {
                    "product_a": 5003,
                    "product_b": 5004,
                    "product_a_price": 18.0,
                    "product_b_price": 9.0,
                    "purchase_price_a": 11.0,
                    "purchase_price_b": 4.0,
                    "new_final_score": 92.0,
                    "model_score": 89.0,
                    "recipe_compat_score": 0.86,
                    "pair_count": 10,
                    "shared_categories_count": 4,
                    "known_prior_flag": 1,
                    "deal_signal": 0.25,
                    "pair_penalty_multiplier": 1.0,
                    "utility_penalty_multiplier": 1.0,
                    "weak_evidence_free_blocked": 0,
                    "only_staples_overlap": 0,
                    "category_a": "breakfast",
                    "category_b": "dairy",
                },
                {
                    "product_a": 5005,
                    "product_b": 5006,
                    "product_a_price": 13.0,
                    "product_b_price": 6.0,
                    "purchase_price_a": 8.0,
                    "purchase_price_b": 3.0,
                    "new_final_score": 90.0,
                    "model_score": 85.0,
                    "recipe_compat_score": 0.83,
                    "pair_count": 8,
                    "shared_categories_count": 3,
                    "known_prior_flag": 1,
                    "deal_signal": 0.2,
                    "pair_penalty_multiplier": 1.0,
                    "utility_penalty_multiplier": 1.0,
                    "weak_evidence_free_blocked": 0,
                    "only_staples_overlap": 0,
                    "category_a": "spices",
                    "category_b": "rice",
                },
                {
                    "product_a": 7001,
                    "product_b": 7001,
                    "product_a_price": 9.0,
                    "product_b_price": 2.0,
                    "purchase_price_a": 7.0,
                    "purchase_price_b": 1.0,
                    "new_final_score": 99.0,
                    "model_score": 90.0,
                    "recipe_compat_score": 0.95,
                    "pair_count": 10,
                    "shared_categories_count": 5,
                    "known_prior_flag": 1,
                    "deal_signal": 0.4,
                    "pair_penalty_multiplier": 1.0,
                    "utility_penalty_multiplier": 1.0,
                    "weak_evidence_free_blocked": 0,
                    "only_staples_overlap": 0,
                    "category_a": "invalid",
                    "category_b": "invalid",
                },
                {
                    "product_a": 7002,
                    "product_b": 7003,
                    "product_a_price": 9.0,
                    "product_b_price": 2.0,
                    "purchase_price_a": 7.0,
                    "purchase_price_b": 1.0,
                    "new_final_score": 98.0,
                    "model_score": 90.0,
                    "recipe_compat_score": 0.9,
                    "pair_count": 8,
                    "shared_categories_count": 4,
                    "known_prior_flag": 1,
                    "deal_signal": 0.4,
                    "pair_penalty_multiplier": 1.0,
                    "utility_penalty_multiplier": 1.0,
                    "weak_evidence_free_blocked": 1,
                    "only_staples_overlap": 0,
                    "category_a": "invalid",
                    "category_b": "invalid",
                },
                {
                    "product_a": 7004,
                    "product_b": 7005,
                    "product_a_price": 9.0,
                    "product_b_price": 2.0,
                    "purchase_price_a": 7.0,
                    "purchase_price_b": 1.0,
                    "new_final_score": 97.0,
                    "model_score": 90.0,
                    "recipe_compat_score": 0.2,
                    "pair_count": 0,
                    "shared_categories_count": 0,
                    "known_prior_flag": 0,
                    "deal_signal": 0.0,
                    "pair_penalty_multiplier": 1.0,
                    "utility_penalty_multiplier": 1.0,
                    "weak_evidence_free_blocked": 0,
                    "only_staples_overlap": 0,
                    "category_a": "invalid",
                    "category_b": "invalid",
                },
            ]
        )

    @staticmethod
    def _fallback_raw_recommendations(*, profiles, **kwargs):
        rows = []
        for profile in profiles:
            profile_id = str(profile.profile_id)
            user_id = int(profile_id.replace("api_user_", ""))
            if user_id == 101:
                bundles = [
                    {"product_a": 9101, "product_b": 9102, "product_a_price": 12.0, "product_b_price": 4.0, "purchase_price_a": 8.0, "purchase_price_b": 2.0},
                    {"product_a": 9103, "product_b": 9104, "product_a_price": 13.0, "product_b_price": 5.0, "purchase_price_a": 8.0, "purchase_price_b": 2.0},
                    {"product_a": 9105, "product_b": 9106, "product_a_price": 14.0, "product_b_price": 6.0, "purchase_price_a": 8.0, "purchase_price_b": 2.0},
                ]
            elif user_id == 202:
                bundles = [
                    {"product_a": 9201, "product_b": 9202, "product_a_price": 12.0, "product_b_price": 4.0, "purchase_price_a": 8.0, "purchase_price_b": 2.0},
                    {"product_a": 9203, "product_b": 9204, "product_a_price": 13.0, "product_b_price": 5.0, "purchase_price_a": 8.0, "purchase_price_b": 2.0},
                ]
            elif user_id == 303:
                bundles = [
                    {"product_a": 9301, "product_b": 9302, "product_a_price": 12.0, "product_b_price": 4.0, "purchase_price_a": 8.0, "purchase_price_b": 2.0},
                ]
            else:
                bundles = []
            rows.append({"profile_id": profile_id, "bundles": bundles})
        return rows

    def test_materialize_respects_max_users_cap_with_deterministic_order(self):
        orders_df = pd.DataFrame(
            [
                {"order_id": 22, "user_id": 202},
                {"order_id": 11, "user_id": 101},
                {"order_id": 33, "user_id": 303},
            ]
        )
        order_pool = SimpleNamespace(
            order_product_ids={
                11: [1001, 1002],
                22: [2001, 2002],
                33: [3001, 3002],
            },
            order_product_names={
                11: ["a", "b"],
                22: ["c", "d"],
                33: ["e", "f"],
            },
        )

        def _fake_recommendations(*, profiles, **kwargs):
            rows = []
            for profile in profiles:
                profile_id = str(profile.profile_id)
                user_id = int(profile_id.replace("api_user_", ""))
                rows.append(
                    {
                        "profile_id": profile_id,
                        "bundles": [
                            {
                                "product_a": int(user_id * 10 + 1),
                                "product_b": int(user_id * 10 + 2),
                                "product_a_price": 11.0,
                                "product_b_price": 7.0,
                                "purchase_price_a": 8.0,
                                "purchase_price_b": 5.0,
                            }
                        ],
                    }
                )
            return rows

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            with patch("qeu_bundling.core.final_recommendations._latest_run_id", return_value="run_cap"), patch(
                "qeu_bundling.core.final_recommendations._load_orders_frame",
                return_value=orders_df,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_order_pool",
                return_value=order_pool,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_bundle_view",
                return_value=SimpleNamespace(bundles_df=pd.DataFrame([{"x": 1}])),
            ), patch(
                "qeu_bundling.core.final_recommendations.build_recommendations_for_profiles",
                side_effect=_fake_recommendations,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_final_artifact_to_s3_if_configured",
                return_value=None,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_bundle_ids_artifact_to_s3_if_configured",
                return_value=None,
            ):
                result = materialize_final_recommendations_by_user(base_dir=base_dir, max_users=2)

            self.assertEqual(result.user_count, 2)
            payload = json.loads((base_dir / "output" / "final_recommendations_by_user.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["candidate_user_count"], 3)
            self.assertEqual(payload["selected_user_count"], 2)
            self.assertEqual(payload["max_users"], 2)
            self.assertTrue(payload["limited_mode"])
            self.assertEqual(payload["user_selection_rule"], "sorted_unique_user_id_asc")
            self.assertEqual(list(payload["recommendations_by_user"].keys()), ["101", "202"])

    def test_materialize_supports_random_user_selection_when_requested(self):
        orders_df = pd.DataFrame(
            [
                {"order_id": 11, "user_id": 101},
                {"order_id": 22, "user_id": 202},
                {"order_id": 33, "user_id": 303},
                {"order_id": 44, "user_id": 404},
            ]
        )
        order_pool = SimpleNamespace(
            order_product_ids={
                11: [1001, 1002],
                22: [2001, 2002],
                33: [3001, 3002],
                44: [4001, 4002],
            },
            order_product_names={
                11: ["a", "b"],
                22: ["c", "d"],
                33: ["e", "f"],
                44: ["g", "h"],
            },
        )

        def _fake_recommendations(*, profiles, **kwargs):
            return [
                {
                    "profile_id": str(profile.profile_id),
                    "bundles": [
                        {
                            "product_a": 11,
                            "product_b": 22,
                            "product_a_price": 10.0,
                            "product_b_price": 5.0,
                            "purchase_price_a": 7.0,
                            "purchase_price_b": 3.0,
                        }
                    ],
                }
                for profile in profiles
            ]

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            with patch("qeu_bundling.core.final_recommendations._latest_run_id", return_value="run_random"), patch(
                "qeu_bundling.core.final_recommendations._load_orders_frame",
                return_value=orders_df,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_order_pool",
                return_value=order_pool,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_bundle_view",
                return_value=SimpleNamespace(bundles_df=pd.DataFrame([{"x": 1}])),
            ), patch(
                "qeu_bundling.core.final_recommendations.build_recommendations_for_profiles",
                side_effect=_fake_recommendations,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_final_artifact_to_s3_if_configured",
                return_value=None,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_bundle_ids_artifact_to_s3_if_configured",
                return_value=None,
            ):
                materialize_final_recommendations_by_user(
                    base_dir=base_dir,
                    max_users=2,
                    user_selection_mode="random",
                    random_seed=7,
                )

            payload = json.loads((base_dir / "output" / "final_recommendations_by_user.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["user_selection_mode"], "random")
            self.assertEqual(payload["user_selection_rule"], "random_sample_from_sorted_user_ids")
            self.assertEqual(payload["random_seed"], 7)
            self.assertEqual(payload["selected_user_count"], 2)
            self.assertEqual(list(payload["recommendations_by_user"].keys()), ["101", "303"])

    def test_resolve_max_users_from_value_and_env(self):
        self.assertIsNone(resolve_final_recommendations_max_users(None))
        self.assertIsNone(resolve_final_recommendations_max_users("   "))
        self.assertIsNone(resolve_final_recommendations_max_users("0"))
        self.assertIsNone(resolve_final_recommendations_max_users("off"))
        self.assertIsNone(resolve_final_recommendations_max_users("unlimited"))
        self.assertEqual(resolve_final_recommendations_max_users("100"), 100)

        with self.assertRaisesRegex(ValueError, FINAL_RECOMMENDATIONS_MAX_USERS_ENV):
            resolve_final_recommendations_max_users("abc")
        with self.assertRaisesRegex(ValueError, FINAL_RECOMMENDATIONS_MAX_USERS_ENV):
            resolve_final_recommendations_max_users("-1")

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_final_recommendations_max_users_from_env())
        with patch.dict(os.environ, {FINAL_RECOMMENDATIONS_MAX_USERS_ENV: "0"}, clear=False):
            self.assertIsNone(resolve_final_recommendations_max_users_from_env())
        with patch.dict(os.environ, {FINAL_RECOMMENDATIONS_MAX_USERS_ENV: "100"}, clear=False):
            self.assertEqual(resolve_final_recommendations_max_users_from_env(), 100)
        with patch.dict(os.environ, {FINAL_RECOMMENDATIONS_MAX_USERS_ENV: "x"}, clear=False):
            with self.assertRaisesRegex(ValueError, FINAL_RECOMMENDATIONS_MAX_USERS_ENV):
                resolve_final_recommendations_max_users_from_env()

    def test_selection_mode_and_random_seed_validation(self):
        self.assertEqual(resolve_final_recommendations_user_selection_mode(""), "sorted")
        self.assertEqual(resolve_final_recommendations_user_selection_mode("sorted"), "sorted")
        self.assertEqual(resolve_final_recommendations_user_selection_mode("random"), "random")
        with self.assertRaisesRegex(ValueError, FINAL_RECOMMENDATIONS_USER_SELECTION_ENV):
            resolve_final_recommendations_user_selection_mode("abc")

        self.assertIsNone(resolve_final_recommendations_random_seed(""))
        self.assertEqual(resolve_final_recommendations_random_seed("42"), 42)
        with self.assertRaisesRegex(ValueError, FINAL_RECOMMENDATIONS_RANDOM_SEED_ENV):
            resolve_final_recommendations_random_seed("seed")

    def test_materialize_backfills_missing_personalized_slots_with_quality_fallback_bank(self):
        orders_df = self._fallback_orders_df()
        order_pool = self._fallback_order_pool()
        scored_df = self._fallback_scored_df()
        fallback_pair_keys = {(5001, 5002), (5003, 5004), (5005, 5006)}

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            with patch.dict(
                os.environ,
                {
                    "QEU_FALLBACK_BUNDLE_BANK_ENABLED": "1",
                    "QEU_FALLBACK_BUNDLE_BANK_TARGET_SIZE": "3",
                    "QEU_FALLBACK_BUNDLE_BANK_MAX_SIZE": "5",
                },
                clear=False,
            ), patch("qeu_bundling.core.final_recommendations._latest_run_id", return_value="run_fallback"), patch(
                "qeu_bundling.core.final_recommendations._load_orders_frame",
                return_value=orders_df,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_order_pool",
                return_value=order_pool,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_bundle_view",
                return_value=SimpleNamespace(bundles_df=scored_df),
            ), patch(
                "qeu_bundling.core.final_recommendations.build_recommendations_for_profiles",
                side_effect=self._fallback_raw_recommendations,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_final_artifact_to_s3_if_configured",
                return_value=None,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_fallback_bank_artifact_to_s3_if_configured",
                return_value=None,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_bundle_ids_artifact_to_s3_if_configured",
                return_value=None,
            ):
                materialize_final_recommendations_by_user(base_dir=base_dir, max_users=4)

            payload = json.loads((base_dir / "output" / "final_recommendations_by_user.json").read_text(encoding="utf-8"))
            by_user = payload["recommendations_by_user"]
            bank_payload = json.loads((base_dir / "output" / "fallback_bundle_bank.json").read_text(encoding="utf-8"))

            self.assertEqual(len(bank_payload["bundles"]), 3)
            actual_bank_pairs = {
                tuple(sorted((int(entry["item_1_id"]), int(entry["item_2_id"]))))
                for entry in bank_payload["bundles"]
            }
            self.assertEqual(actual_bank_pairs, fallback_pair_keys)

            self.assertEqual(len(by_user["101"]), 3)
            user_101_pairs = {
                tuple(sorted((int(b["item_1_id"]), int(b["item_2_id"]))))
                for b in by_user["101"]
            }
            self.assertTrue(user_101_pairs.isdisjoint(fallback_pair_keys))

            self.assertEqual(len(by_user["202"]), 3)
            user_202_pairs = {
                tuple(sorted((int(b["item_1_id"]), int(b["item_2_id"]))))
                for b in by_user["202"]
            }
            self.assertTrue(any(pair in fallback_pair_keys for pair in user_202_pairs))

            self.assertEqual(len(by_user["303"]), 3)
            user_303_pairs = {
                tuple(sorted((int(b["item_1_id"]), int(b["item_2_id"]))))
                for b in by_user["303"]
            }
            self.assertGreaterEqual(len(user_303_pairs & fallback_pair_keys), 2)

            self.assertEqual(len(by_user["404"]), 3)
            user_404_pairs = {
                tuple(sorted((int(b["item_1_id"]), int(b["item_2_id"]))))
                for b in by_user["404"]
            }
            self.assertEqual(user_404_pairs, fallback_pair_keys)

            for bundles in by_user.values():
                pair_keys = [tuple(sorted((int(b["item_1_id"]), int(b["item_2_id"])))) for b in bundles]
                self.assertEqual(len(pair_keys), len(set(pair_keys)))

    def test_fallback_backfill_is_deterministic_for_same_inputs(self):
        orders_df = self._fallback_orders_df()
        order_pool = self._fallback_order_pool()
        scored_df = self._fallback_scored_df()

        def _run_once(base_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
            with patch.dict(
                os.environ,
                {
                    "QEU_FALLBACK_BUNDLE_BANK_ENABLED": "1",
                    "QEU_FALLBACK_BUNDLE_BANK_TARGET_SIZE": "3",
                    "QEU_FALLBACK_BUNDLE_BANK_MAX_SIZE": "5",
                },
                clear=False,
            ), patch("qeu_bundling.core.final_recommendations._latest_run_id", return_value="run_deterministic"), patch(
                "qeu_bundling.core.final_recommendations._load_orders_frame",
                return_value=orders_df,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_order_pool",
                return_value=order_pool,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_bundle_view",
                return_value=SimpleNamespace(bundles_df=scored_df),
            ), patch(
                "qeu_bundling.core.final_recommendations.build_recommendations_for_profiles",
                side_effect=self._fallback_raw_recommendations,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_final_artifact_to_s3_if_configured",
                return_value=None,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_fallback_bank_artifact_to_s3_if_configured",
                return_value=None,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_bundle_ids_artifact_to_s3_if_configured",
                return_value=None,
            ):
                materialize_final_recommendations_by_user(base_dir=base_dir, max_users=4)
            rec_payload = json.loads((base_dir / "output" / "final_recommendations_by_user.json").read_text(encoding="utf-8"))
            bank_payload = json.loads((base_dir / "output" / "fallback_bundle_bank.json").read_text(encoding="utf-8"))
            return rec_payload, bank_payload

        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            rec_a, bank_a = _run_once(Path(tmp_a))
            rec_b, bank_b = _run_once(Path(tmp_b))

        self.assertEqual(bank_a["bundles"], bank_b["bundles"])
        self.assertEqual(rec_a["recommendations_by_user"], rec_b["recommendations_by_user"])

    def test_bundle_id_registry_is_written_and_persists_across_runs(self):
        orders_df = pd.DataFrame(
            [
                {"order_id": 11, "user_id": 101},
                {"order_id": 22, "user_id": 202},
            ]
        )
        order_pool = SimpleNamespace(
            order_product_ids={
                11: [1001, 1002],
                22: [2001, 2002],
            },
            order_product_names={
                11: ["a", "b"],
                22: ["c", "d"],
            },
        )
        scored_df = pd.DataFrame(
            [
                {"product_a": 9001, "product_b": 9002, "product_a_price": 10.0, "product_b_price": 5.0, "purchase_price_a": 7.0, "purchase_price_b": 3.0}
            ]
        )

        def _fake_reco_run1(*, profiles, **kwargs):
            rows = []
            for profile in profiles:
                user_id = int(str(profile.profile_id).replace("api_user_", ""))
                if user_id == 101:
                    rows.append(
                        {
                            "profile_id": str(profile.profile_id),
                            "bundles": [
                                {
                                    "product_a": 111,
                                    "product_b": 222,
                                    "product_a_price": 10.0,
                                    "product_b_price": 4.0,
                                    "purchase_price_a": 7.0,
                                    "purchase_price_b": 2.0,
                                }
                            ],
                        }
                    )
                elif user_id == 202:
                    rows.append(
                        {
                            "profile_id": str(profile.profile_id),
                            "bundles": [
                                {
                                    "product_a": 333,
                                    "product_b": 444,
                                    "product_a_price": 11.0,
                                    "product_b_price": 5.0,
                                    "purchase_price_a": 7.0,
                                    "purchase_price_b": 2.0,
                                }
                            ],
                        }
                    )
            return rows

        def _fake_reco_run2(*, profiles, **kwargs):
            rows = []
            for profile in profiles:
                user_id = int(str(profile.profile_id).replace("api_user_", ""))
                if user_id == 101:
                    rows.append(
                        {
                            "profile_id": str(profile.profile_id),
                            "bundles": [
                                {
                                    "product_a": 111,
                                    "product_b": 222,
                                    "product_a_price": 10.0,
                                    "product_b_price": 4.0,
                                    "purchase_price_a": 7.0,
                                    "purchase_price_b": 2.0,
                                }
                            ],
                        }
                    )
                elif user_id == 202:
                    rows.append(
                        {
                            "profile_id": str(profile.profile_id),
                            "bundles": [
                                {
                                    "product_a": 555,
                                    "product_b": 666,
                                    "product_a_price": 12.0,
                                    "product_b_price": 6.0,
                                    "purchase_price_a": 8.0,
                                    "purchase_price_b": 3.0,
                                }
                            ],
                        }
                    )
            return rows

        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)

            with patch.dict(os.environ, {"QEU_FALLBACK_BUNDLE_BANK_ENABLED": "0"}, clear=False), patch(
                "qeu_bundling.core.final_recommendations._latest_run_id",
                return_value="run_bundle_ids_1",
            ), patch(
                "qeu_bundling.core.final_recommendations._load_orders_frame",
                return_value=orders_df,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_order_pool",
                return_value=order_pool,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_bundle_view",
                return_value=SimpleNamespace(bundles_df=scored_df),
            ), patch(
                "qeu_bundling.core.final_recommendations.build_recommendations_for_profiles",
                side_effect=_fake_reco_run1,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_final_artifact_to_s3_if_configured",
                return_value=None,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_fallback_bank_artifact_to_s3_if_configured",
                return_value=None,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_bundle_ids_artifact_to_s3_if_configured",
                return_value=None,
            ):
                materialize_final_recommendations_by_user(base_dir=base_dir, max_users=2)

            ids_path = bundle_ids_artifact_path(base_dir)
            self.assertTrue(ids_path.exists())
            ids_run_1 = pd.read_csv(ids_path, dtype=str)
            self.assertEqual(len(ids_run_1), 2)
            self.assertEqual(ids_run_1["bundle_id"].tolist(), ["B00000001", "B00000002"])
            pair_to_id_run_1 = {
                (int(row["item_1_id"]), int(row["item_2_id"])): str(row["bundle_id"])
                for _, row in ids_run_1.iterrows()
            }

            with patch.dict(os.environ, {"QEU_FALLBACK_BUNDLE_BANK_ENABLED": "0"}, clear=False), patch(
                "qeu_bundling.core.final_recommendations._latest_run_id",
                return_value="run_bundle_ids_2",
            ), patch(
                "qeu_bundling.core.final_recommendations._load_orders_frame",
                return_value=orders_df,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_order_pool",
                return_value=order_pool,
            ), patch(
                "qeu_bundling.core.final_recommendations.load_bundle_view",
                return_value=SimpleNamespace(bundles_df=scored_df),
            ), patch(
                "qeu_bundling.core.final_recommendations.build_recommendations_for_profiles",
                side_effect=_fake_reco_run2,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_final_artifact_to_s3_if_configured",
                return_value=None,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_fallback_bank_artifact_to_s3_if_configured",
                return_value=None,
            ), patch(
                "qeu_bundling.core.final_recommendations._upload_bundle_ids_artifact_to_s3_if_configured",
                return_value=None,
            ):
                materialize_final_recommendations_by_user(base_dir=base_dir, max_users=2)

            ids_run_2 = pd.read_csv(ids_path, dtype=str)
            self.assertEqual(len(ids_run_2), 3)
            pair_to_id_run_2 = {
                (int(row["item_1_id"]), int(row["item_2_id"])): str(row["bundle_id"])
                for _, row in ids_run_2.iterrows()
            }
            self.assertEqual(pair_to_id_run_1[(111, 222)], pair_to_id_run_2[(111, 222)])
            self.assertEqual(pair_to_id_run_1[(333, 444)], pair_to_id_run_2[(333, 444)])
            self.assertEqual(pair_to_id_run_2[(555, 666)], "B00000003")

            pair_to_seen_count = {
                (int(row["item_1_id"]), int(row["item_2_id"])): int(row["seen_count"])
                for _, row in ids_run_2.iterrows()
            }
            self.assertEqual(pair_to_seen_count[(111, 222)], 2)
            self.assertEqual(pair_to_seen_count[(333, 444)], 1)
            self.assertEqual(pair_to_seen_count[(555, 666)], 1)


if __name__ == "__main__":
    unittest.main()
