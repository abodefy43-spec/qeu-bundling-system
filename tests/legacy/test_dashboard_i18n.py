import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from qeu_bundling.presentation.dashboard_i18n import (
    dashboard_ui_text_ar,
    translate_en_to_ar,
    translate_recommendations_for_dashboard,
)


class DashboardI18nTests(unittest.TestCase):
    def test_translate_en_to_ar_uses_cache_on_second_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "dashboard_en_ar_cache.json"
            translator = MagicMock()
            translator.translate.return_value = "صدر دجاج"

            with patch(
                "qeu_bundling.presentation.dashboard_i18n._get_translator",
                return_value=translator,
            ):
                first = translate_en_to_ar("chicken breast", cache_path)

            self.assertEqual(first, "صدر دجاج")
            self.assertEqual(translator.translate.call_count, 1)

            with patch(
                "qeu_bundling.presentation.dashboard_i18n._get_translator",
                side_effect=AssertionError("translator should not be called when cache exists"),
            ):
                second = translate_en_to_ar("chicken breast", cache_path)

            self.assertEqual(second, "صدر دجاج")

    def test_translate_en_to_ar_falls_back_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "dashboard_en_ar_cache.json"
            translator = MagicMock()
            translator.translate.side_effect = RuntimeError("translation failed")

            with patch(
                "qeu_bundling.presentation.dashboard_i18n._get_translator",
                return_value=translator,
            ):
                out = translate_en_to_ar("tomato paste", cache_path)

            self.assertEqual(out, "tomato paste")

    def test_translate_en_to_ar_skips_empty_and_numeric_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "dashboard_en_ar_cache.json"
            with patch("qeu_bundling.presentation.dashboard_i18n._get_translator") as mocked_get:
                self.assertEqual(translate_en_to_ar("", cache_path), "")
                self.assertEqual(translate_en_to_ar("SAR 12.50", cache_path), "SAR 12.50")
                self.assertEqual(translate_en_to_ar("100%", cache_path), "100%")
                mocked_get.assert_not_called()

    def test_translate_recommendations_adds_ar_fields_preserving_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "dashboard_en_ar_cache.json"
            input_recos = [
                {
                    "person_label": "Person 1",
                    "history_items": ["black tea", "potato chips"],
                    "bundles": [
                        {
                            "product_a_name": "black tea",
                            "product_b_name": "white sugar",
                            "chosen_bundle_names": ["black tea", "white sugar"],
                            "recommendation_reasons": ["Anchor from history", "Recipe-compatible"],
                            "recommendation_origin_label": "Top-bundle match",
                            "bundle_items": [
                                {"name": "black tea", "price_sar": "12.00"},
                                {"name": "white sugar", "price_sar": "2.00"},
                            ],
                        }
                    ],
                }
            ]

            with patch(
                "qeu_bundling.presentation.dashboard_i18n.translate_en_to_ar",
                side_effect=lambda text, _cache: f"AR:{text}",
            ):
                output = translate_recommendations_for_dashboard(input_recos, cache_path)

            self.assertEqual(input_recos[0]["bundles"][0]["product_a_name"], "black tea")
            self.assertEqual(output[0]["bundles"][0]["product_a_name"], "black tea")
            self.assertEqual(output[0]["bundles"][0]["product_a_name_ar"], "AR:black tea")
            self.assertEqual(output[0]["bundles"][0]["product_b_name_ar"], "AR:white sugar")
            self.assertEqual(
                output[0]["bundles"][0]["chosen_bundle_names_ar"],
                ["AR:black tea", "AR:white sugar"],
            )
            self.assertEqual(
                output[0]["bundles"][0]["recommendation_reasons_ar"],
                ["AR:Anchor from history", "AR:Recipe-compatible"],
            )
            self.assertEqual(
                output[0]["bundles"][0]["bundle_items"][0]["name_ar"],
                "AR:black tea",
            )
            self.assertEqual(
                output[0]["history_items_ar"],
                ["AR:black tea", "AR:potato chips"],
            )

    def test_ui_text_contains_expected_keys(self):
        ui = dashboard_ui_text_ar()
        self.assertIn("people_predictions_title_ar", ui)
        self.assertIn("lane_meal_ar", ui)
        self.assertIn("free_label_ar", ui)


if __name__ == "__main__":
    unittest.main()
