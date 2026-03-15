import json
from collections import Counter
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd

from qeu_bundling.config.paths import get_paths as real_get_paths
from qeu_bundling.presentation.person_predictions import (
    CURATED_CLEANING_FALLBACK_BUNDLES,
    INTENT_BOOST_MAX,
    INTENT_CLEANING,
    INTENT_DESSERT,
    INTENT_DRINK_PAIRING,
    INTENT_MEAL_BASE,
    INTENT_STAPLES,
    LANE_MEAL,
    LANE_NONFOOD,
    LANE_OCCASION,
    LANE_SNACK,
    NON_FOOD_TAG,
    OrderPool,
    PersonalizationContext,
    PersonProfile,
    ProductMatcher,
    ServingTelemetry,
    TOP_100_CURATED_FOOD_BUNDLES,
    _anchor_allowed_for_lane,
    _build_bundle_lookup,
    _build_user_intent_profile,
    _build_top_bundle_rows_by_anchor,
    _bundle_primary_intent,
    _candidate_bundle_shape_signature,
    _candidate_effective_score,
    _candidate_family_pattern_signature,
    _candidate_motif_family_signature,
    _candidate_rank_key,
    _default_intent_profile,
    _feedback_pair_boost,
    _feedback_pair_class,
    _feedback_pair_penalty,
    _fallback_candidates_for_lane,
    _intent_profile_boost,
    _maybe_swap_to_make_free_item_cheaper,
    _pick_candidate_for_anchor,
    _passes_pair_filters,
    _is_staple_product,
    _is_meal_dominant_motif_signature,
    _passes_complement_gate,
    _pick_three_lane_anchors,
    _rank_anchors,
    _rng_for_profile,
    _top_intent,
    _top_bundle_scan_limit,
    _template_signature,
    _write_person_quality_artifact,
    build_manual_profile,
    build_random_profile,
    build_recommendations_for_profiles,
    get_last_serving_profile_metrics,
)
from qeu_bundling.presentation import bundle_semantics as semantics


def _build_context() -> PersonalizationContext:
    names = {
        1: "watania chicken breast",
        2: "basmati rice",
        3: "potato chips",
        4: "cola soda",
        5: "black tea",
        6: "white sugar",
        7: "sunflower cooking oil",
        8: "chocolate biscuit",
        9: "premium dates",
        10: "fresh cream",
        11: "full fat milk",
        12: "tuna chunks",
        13: "pasta penne",
        14: "tomato paste",
        16: "fresh fish fillet",
        17: "fresh garlic",
        18: "fresh carrots",
        19: "indomie chicken noodles cup",
        20: "ground cumin spice",
        21: "black pepper powder",
        22: "whole wheat tortilla bread",
        23: "farm eggs",
        24: "triangle cheese",
        25: "salt crackers",
        26: "orange juice",
        28: "kraft cheese portions",
        101: "herbal shampoo",
        102: "silky conditioner",
        103: "laundry detergent",
        104: "fabric softener",
        105: "premium disinfectant",
        106: "premium bleach",
        107: "body soap",
        108: "paper towels pack",
    }
    prices = {
        1: 20.0,
        2: 8.0,
        3: 10.0,
        4: 4.0,
        5: 12.0,
        6: 2.0,
        7: 5.0,
        8: 3.0,
        9: 18.0,
        10: 7.0,
        11: 4.0,
        12: 14.0,
        13: 6.0,
        14: 4.0,
        16: 20.0,
        17: 2.5,
        18: 3.0,
        19: 5.0,
        20: 4.0,
        21: 4.5,
        22: 4.9,
        23: 18.9,
        24: 6.0,
        25: 3.5,
        26: 4.5,
        28: 5.5,
        101: 6.0,
        102: 5.0,
        103: 4.0,
        104: 12.0,
        105: 12.0,
        106: 11.0,
        107: 4.0,
        108: 5.5,
    }
    return PersonalizationContext(
        product_name_by_id=names,
        product_price_by_id=prices,
        product_picture_by_id={},
        neighbors={
            1: ((2, 62.0), (14, 40.0)),
            2: ((1, 62.0), (14, 44.0), (3, 55.0)),
            3: ((4, 45.0), (8, 40.0)),
            5: ((6, 35.0), (8, 28.0)),
            9: ((10, 48.0),),
            16: ((2, 56.0), (7, 52.0)),
            17: ((1, 35.0), (20, 34.0), (3, 60.0), (19, 62.0)),
            18: ((16, 33.0), (19, 59.0)),
            20: ((21, 31.0),),
            21: ((20, 31.0),),
            22: ((23, 42.0),),
            23: ((22, 42.0),),
            24: ((25, 39.0),),
            25: ((24, 39.0), (5, 35.0)),
            101: ((102, 38.0), (2, 50.0)),
            103: ((104, 35.0),),
            105: ((106, 33.0),),
            107: ((108, 30.0),),
        },
        recipe_score_by_id={
            1: 80.0,
            2: 55.0,
            3: 45.0,
            4: 30.0,
            5: 60.0,
            6: 40.0,
            7: 50.0,
            8: 48.0,
            9: 62.0,
            10: 50.0,
            14: 67.0,
            16: 78.0,
            17: 58.0,
            18: 52.0,
            19: 40.0,
            20: 64.0,
            21: 63.0,
            22: 46.0,
            23: 69.0,
            24: 52.0,
            25: 45.0,
            26: 42.0,
        },
        ingredient_by_id={
            1: "chicken",
            2: "rice",
            3: "chips",
            4: "soda",
            5: "tea",
            6: "sugar",
            7: "oil",
            8: "biscuit",
            9: "dates",
            10: "cream",
            14: "tomatoes",
            16: "fish",
            17: "garlic",
            18: "carrots",
            19: "noodles",
            20: "cumin",
            21: "black_pepper",
            22: "bread",
            23: "eggs",
            24: "cheese",
        25: "biscuits",
        26: "oranges",
        28: "cheese",
        },
        ingredient_recipe_lookup={
            "chicken": ("meal1",),
            "rice": ("meal1",),
            "tomatoes": ("meal1",),
            "fish": ("meal2",),
            "chips": ("snack1",),
            "soda": ("snack1",),
            "tea": ("occasion1",),
            "sugar": ("occasion1",),
            "biscuit": ("occasion1",),
            "dates": ("occasion2",),
            "cream": ("occasion2",),
            "garlic": ("meal1",),
            "carrots": ("meal1",),
            "noodles": ("snack1",),
            "cumin": ("meal1",),
            "black_pepper": ("meal1",),
            "bread": ("meal1",),
            "eggs": ("meal1",),
            "cheese": ("occasion1",),
            "biscuits": ("occasion1",),
            "oranges": ("snack1",),
        },
        product_family_by_id={
            1: "poultry",
            2: "rice_centric",
            3: "chips",
            4: "beverage_soda",
            5: "tea",
            6: "sugar",
            7: "cooking_oil",
            8: "biscuit",
            9: "dates_family",
            10: "dairy",
            11: "dairy",
            12: "fish",
            13: "pasta",
            14: "tomato_paste",
            16: "seafood",
            17: "produce",
            18: "produce",
            19: "noodles",
            20: "spices",
            21: "spices",
            22: "bread",
            23: "eggs",
            24: "cheese",
            25: "crackers",
            26: "beverage_juice",
            28: "cheese_snack",
            101: "personal_care",
            102: "personal_care",
            103: "laundry",
            104: "laundry",
            105: "cleaning",
            106: "cleaning",
            107: "body",
            108: "tissue",
        },
        category_by_id={
            1: "protein",
            2: "grains",
            3: "snacks",
            4: "beverages",
            5: "beverages",
            6: "baking",
            7: "oils",
            8: "snacks",
            9: "fruits",
            10: "dairy",
            11: "dairy",
            12: "protein",
            13: "grains",
            14: "vegetables",
            16: "protein",
            17: "vegetables",
            18: "vegetables",
            19: "grains",
            20: "spices",
            21: "spices",
            22: "grains",
            23: "protein",
            24: "dairy",
            25: "snacks",
            26: "beverages",
            28: "dairy",
            101: NON_FOOD_TAG,
            102: NON_FOOD_TAG,
            103: NON_FOOD_TAG,
            104: NON_FOOD_TAG,
            105: NON_FOOD_TAG,
            106: NON_FOOD_TAG,
            107: NON_FOOD_TAG,
            108: NON_FOOD_TAG,
        },
        product_brand_by_id={1: "watania", 16: "sea_brand"},
        non_food_ids=frozenset({101, 102, 103, 104, 105, 106, 107, 108}),
    )


def _build_bundles() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 1,
                "product_b": 2,
                "product_a_name": "watania chicken breast",
                "product_b_name": "basmati rice",
                "final_score": 86,
                "purchase_score": 62,
                "pair_count": 100,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "poultry",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 16,
                "product_b": 2,
                "product_a_name": "fresh fish fillet",
                "product_b_name": "basmati rice",
                "final_score": 84,
                "purchase_score": 58,
                "pair_count": 77,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "seafood",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 2,
                "product_b": 14,
                "product_a_name": "basmati rice",
                "product_b_name": "tomato paste",
                "final_score": 83,
                "purchase_score": 42,
                "pair_count": 42,
                "category_a": "grains",
                "category_b": "vegetables",
                "product_family_a": "rice_centric",
                "product_family_b": "tomato_paste",
            },
            {
                "product_a": 2,
                "product_b": 3,
                "product_a_name": "basmati rice",
                "product_b_name": "potato chips",
                "final_score": 90,
                "purchase_score": 80,
                "pair_count": 120,
                "category_a": "grains",
                "category_b": "snacks",
                "product_family_a": "rice_centric",
                "product_family_b": "chips",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 77,
                "purchase_score": 34,
                "pair_count": 36,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 3,
                "product_b": 8,
                "product_a_name": "potato chips",
                "product_b_name": "chocolate biscuit",
                "final_score": 75,
                "purchase_score": 31,
                "pair_count": 28,
                "category_a": "snacks",
                "category_b": "snacks",
                "product_family_a": "chips",
                "product_family_b": "biscuit",
            },
            {
                "product_a": 5,
                "product_b": 6,
                "product_a_name": "black tea",
                "product_b_name": "white sugar",
                "final_score": 76,
                "purchase_score": 28,
                "pair_count": 50,
                "category_a": "beverages",
                "category_b": "baking",
                "product_family_a": "tea",
                "product_family_b": "sugar",
            },
            {
                "product_a": 9,
                "product_b": 10,
                "product_a_name": "premium dates",
                "product_b_name": "fresh cream",
                "final_score": 78,
                "purchase_score": 39,
                "pair_count": 34,
                "category_a": "fruits",
                "category_b": "dairy",
                "product_family_a": "dates_family",
                "product_family_b": "dairy",
            },
            {
                "product_a": 101,
                "product_b": 102,
                "product_a_name": "herbal shampoo",
                "product_b_name": "silky conditioner",
                "final_score": 84,
                "purchase_score": 37,
                "pair_count": 31,
                "category_a": NON_FOOD_TAG,
                "category_b": NON_FOOD_TAG,
                "product_family_a": "personal_care",
                "product_family_b": "personal_care",
            },
            {
                "product_a": 101,
                "product_b": 2,
                "product_a_name": "herbal shampoo",
                "product_b_name": "basmati rice",
                "final_score": 95,
                "purchase_score": 65,
                "pair_count": 90,
                "category_a": NON_FOOD_TAG,
                "category_b": "grains",
                "product_family_a": "personal_care",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 103,
                "product_b": 104,
                "product_a_name": "laundry detergent",
                "product_b_name": "fabric softener",
                "product_a_price": 4.0,
                "product_b_price": 12.0,
                "final_score": 82,
                "purchase_score": 35,
                "pair_count": 30,
                "category_a": NON_FOOD_TAG,
                "category_b": NON_FOOD_TAG,
                "product_family_a": "laundry",
                "product_family_b": "laundry",
            },
            {
                "product_a": 105,
                "product_b": 106,
                "product_a_name": "premium disinfectant",
                "product_b_name": "premium bleach",
                "product_a_price": 12.0,
                "product_b_price": 11.0,
                "final_score": 79,
                "purchase_score": 34,
                "pair_count": 18,
                "category_a": NON_FOOD_TAG,
                "category_b": NON_FOOD_TAG,
                "product_family_a": "cleaning",
                "product_family_b": "cleaning",
            },
        ]
    )


def _build_bundles_quality_tightening() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 1,
                "product_b": 2,
                "product_a_name": "watania chicken breast",
                "product_b_name": "basmati rice",
                "final_score": 90,
                "purchase_score": 60,
                "pair_count": 80,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "poultry",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 88,
                "purchase_score": 44,
                "pair_count": 60,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 9,
                "product_b": 10,
                "product_a_name": "premium dates",
                "product_b_name": "fresh cream",
                "final_score": 87,
                "purchase_score": 42,
                "pair_count": 55,
                "category_a": "fruits",
                "category_b": "dairy",
                "product_family_a": "dates_family",
                "product_family_b": "dairy",
            },
            {
                "product_a": 12,
                "product_b": 11,
                "product_a_name": "tuna chunks",
                "product_b_name": "full fat milk",
                "final_score": 96,
                "purchase_score": 80,
                "pair_count": 120,
                "category_a": "protein",
                "category_b": "dairy",
                "product_family_a": "seafood",
                "product_family_b": "milk_dairy",
            },
            {
                "product_a": 12,
                "product_b": 117,
                "product_a_name": "tuna chunks",
                "product_b_name": "peanut butter spread",
                "final_score": 97,
                "purchase_score": 82,
                "pair_count": 130,
                "category_a": "protein",
                "category_b": "spreads",
                "product_family_a": "seafood",
                "product_family_b": "spread",
            },
            {
                "product_a": 12,
                "product_b": 13,
                "product_a_name": "tuna chunks",
                "product_b_name": "pasta penne",
                "final_score": 98,
                "purchase_score": 84,
                "pair_count": 132,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "seafood",
                "product_family_b": "noodles_pasta",
            },
            {
                "product_a": 7,
                "product_b": 113,
                "product_a_name": "sunflower cooking oil",
                "product_b_name": "chicken stock cubes",
                "final_score": 99,
                "purchase_score": 86,
                "pair_count": 140,
                "category_a": "oils",
                "category_b": "condiments",
                "product_family_a": "cooking_oil",
                "product_family_b": "seasoning",
            },
            {
                "product_a": 23,
                "product_b": 14,
                "product_a_name": "pond food: eggs",
                "product_b_name": "Rivoni tomato paste 8 x 135g",
                "final_score": 99,
                "purchase_score": 86,
                "pair_count": 140,
                "category_a": "protein",
                "category_b": "condiments",
                "product_family_a": "eggs",
                "product_family_b": "sauce",
            },
            {
                "product_a": 19,
                "product_b": 112,
                "product_a_name": "indomie chicken noodles cup",
                "product_b_name": "premium flour",
                "final_score": 95,
                "purchase_score": 78,
                "pair_count": 118,
                "category_a": "grains",
                "category_b": "baking",
                "product_family_a": "noodles",
                "product_family_b": "flour",
            },
            {
                "product_a": 113,
                "product_b": 114,
                "product_a_name": "chicken stock cubes",
                "product_b_name": "garlic mayo",
                "final_score": 94,
                "purchase_score": 76,
                "pair_count": 112,
                "category_a": "condiments",
                "category_b": "condiments",
                "product_family_a": "seasoning",
                "product_family_b": "condiments",
            },
            {
                "product_a": 7,
                "product_b": 115,
                "product_a_name": "sunflower cooking oil",
                "product_b_name": "chicken nuggets",
                "final_score": 93,
                "purchase_score": 74,
                "pair_count": 106,
                "category_a": "oils",
                "category_b": "protein",
                "product_family_a": "cooking_oil",
                "product_family_b": "protein",
            },
            {
                "product_a": 24,
                "product_b": 116,
                "product_a_name": "cream cheese spread",
                "product_b_name": "caramel dessert pudding",
                "final_score": 92,
                "purchase_score": 72,
                "pair_count": 102,
                "category_a": "dairy",
                "category_b": "desserts",
                "product_family_a": "cream_cheese",
                "product_family_b": "dessert",
            },
            {
                "product_a": 116,
                "product_b": 118,
                "product_a_name": "caramel dessert pudding",
                "product_b_name": "plain tea biscuits",
                "final_score": 96,
                "purchase_score": 82,
                "pair_count": 126,
                "category_a": "desserts",
                "category_b": "snacks",
                "product_family_a": "dessert",
                "product_family_b": "biscuit",
            },
            {
                "product_a": 11,
                "product_b": 119,
                "product_a_name": "full fat milk",
                "product_b_name": "nutella chocolate spread",
                "final_score": 95,
                "purchase_score": 80,
                "pair_count": 118,
                "category_a": "dairy",
                "category_b": "spreads",
                "product_family_a": "milk_dairy",
                "product_family_b": "spread",
            },
            {
                "product_a": 2,
                "product_b": 12,
                "product_a_name": "basmati rice",
                "product_b_name": "tuna chunks",
                "final_score": 91,
                "purchase_score": 70,
                "pair_count": 98,
                "category_a": "grains",
                "category_b": "protein",
                "product_family_a": "rice_centric",
                "product_family_b": "seafood",
            },
            {
                "product_a": 12,
                "product_b": 2,
                "product_a_name": "tuna chunks",
                "product_b_name": "basmati rice",
                "final_score": 89,
                "purchase_score": 68,
                "pair_count": 90,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "seafood",
                "product_family_b": "rice_centric",
            },
        ]
    )


def _build_bundles_swap_only() -> pd.DataFrame:
    rows = _build_bundles()
    return rows[rows["product_a"] != 101].copy()


def _build_bundles_meal_produce_snack_guardrail() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 17,
                "product_b": 3,
                "product_a_name": "fresh garlic",
                "product_b_name": "potato chips",
                "product_a_price": 2.5,
                "product_b_price": 10.0,
                "final_score": 99,
                "purchase_score": 95,
                "pair_count": 200,
                "category_a": "vegetables",
                "category_b": "snacks",
                "product_family_a": "produce",
                "product_family_b": "chips",
            },
            {
                "product_a": 17,
                "product_b": 1,
                "product_a_name": "fresh garlic",
                "product_b_name": "watania chicken breast",
                "product_a_price": 2.5,
                "product_b_price": 20.0,
                "final_score": 76,
                "purchase_score": 24,
                "pair_count": 12,
                "category_a": "vegetables",
                "category_b": "protein",
                "product_family_a": "produce",
                "product_family_b": "poultry",
            },
            {
                "product_a": 17,
                "product_b": 20,
                "product_a_name": "fresh garlic",
                "product_b_name": "ground cumin spice",
                "product_a_price": 2.5,
                "product_b_price": 4.0,
                "final_score": 74,
                "purchase_score": 20,
                "pair_count": 9,
                "category_a": "vegetables",
                "category_b": "spices",
                "product_family_a": "produce",
                "product_family_b": "spices",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "product_a_price": 10.0,
                "product_b_price": 4.0,
                "final_score": 88,
                "purchase_score": 40,
                "pair_count": 60,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 5,
                "product_b": 6,
                "product_a_name": "black tea",
                "product_b_name": "white sugar",
                "product_a_price": 12.0,
                "product_b_price": 2.0,
                "final_score": 87,
                "purchase_score": 35,
                "pair_count": 60,
                "category_a": "beverages",
                "category_b": "baking",
                "product_family_a": "tea",
                "product_family_b": "sugar",
            },
        ]
    )


def _build_bundles_meal_produce_noodles_guardrail() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 18,
                "product_b": 19,
                "product_a_name": "fresh carrots",
                "product_b_name": "indomie chicken noodles cup",
                "product_a_price": 3.0,
                "product_b_price": 5.0,
                "final_score": 99,
                "purchase_score": 92,
                "pair_count": 180,
                "category_a": "vegetables",
                "category_b": "grains",
                "product_family_a": "produce",
                "product_family_b": "noodles",
            },
            {
                "product_a": 18,
                "product_b": 16,
                "product_a_name": "fresh carrots",
                "product_b_name": "fresh fish fillet",
                "product_a_price": 3.0,
                "product_b_price": 20.0,
                "final_score": 77,
                "purchase_score": 22,
                "pair_count": 11,
                "category_a": "vegetables",
                "category_b": "protein",
                "product_family_a": "produce",
                "product_family_b": "seafood",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "product_a_price": 10.0,
                "product_b_price": 4.0,
                "final_score": 88,
                "purchase_score": 40,
                "pair_count": 60,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 5,
                "product_b": 6,
                "product_a_name": "black tea",
                "product_b_name": "white sugar",
                "product_a_price": 12.0,
                "product_b_price": 2.0,
                "final_score": 87,
                "purchase_score": 35,
                "pair_count": 60,
                "category_a": "beverages",
                "category_b": "baking",
                "product_family_a": "tea",
                "product_family_b": "sugar",
            },
        ]
    )


def _build_bundles_for_swap() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 22,
                "product_b": 23,
                "product_a_name": "whole wheat tortilla bread",
                "product_b_name": "farm eggs",
                "product_a_price": 4.9,
                "product_b_price": 18.9,
                "final_score": 96,
                "purchase_score": 70,
                "pair_count": 120,
                "category_a": "grains",
                "category_b": "protein",
                "product_family_a": "bread",
                "product_family_b": "eggs",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "product_a_price": 10.0,
                "product_b_price": 4.0,
                "final_score": 84,
                "purchase_score": 35,
                "pair_count": 40,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 5,
                "product_b": 6,
                "product_a_name": "black tea",
                "product_b_name": "white sugar",
                "product_a_price": 12.0,
                "product_b_price": 2.0,
                "final_score": 83,
                "purchase_score": 32,
                "pair_count": 45,
                "category_a": "beverages",
                "category_b": "baking",
                "product_family_a": "tea",
                "product_family_b": "sugar",
            },
        ]
    )


def _build_bundles_expensive_nonfood_only() -> pd.DataFrame:
    rows = _build_bundles()
    keep = ~((rows["product_a"] == 101) | (rows["product_a"] == 103))
    return rows[keep].copy()


def _build_bundles_snack_pattern_only() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 1,
                "product_b": 2,
                "product_a_name": "watania chicken breast",
                "product_b_name": "basmati rice",
                "final_score": 90,
                "purchase_score": 60,
                "pair_count": 80,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "poultry",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 22,
                "product_b": 24,
                "product_a_name": "whole wheat tortilla bread",
                "product_b_name": "triangle cheese",
                "final_score": 99,
                "purchase_score": 95,
                "pair_count": 200,
                "category_a": "grains",
                "category_b": "dairy",
                "product_family_a": "bread",
                "product_family_b": "cheese",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 86,
                "purchase_score": 40,
                "pair_count": 50,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 25,
                "product_b": 5,
                "product_a_name": "salt crackers",
                "product_b_name": "black tea",
                "final_score": 82,
                "purchase_score": 32,
                "pair_count": 45,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "crackers",
                "product_family_b": "tea",
            },
            {
                "product_a": 9,
                "product_b": 10,
                "product_a_name": "premium dates",
                "product_b_name": "fresh cream",
                "final_score": 84,
                "purchase_score": 35,
                "pair_count": 30,
                "category_a": "fruits",
                "category_b": "dairy",
                "product_family_a": "dates_family",
                "product_family_b": "dairy",
            },
        ]
    )


def _build_bundles_snack_negative_block() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 1,
                "product_b": 2,
                "product_a_name": "watania chicken breast",
                "product_b_name": "basmati rice",
                "final_score": 88,
                "purchase_score": 55,
                "pair_count": 70,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "poultry",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 17,
                "product_b": 3,
                "product_a_name": "fresh garlic",
                "product_b_name": "potato chips",
                "final_score": 98,
                "purchase_score": 90,
                "pair_count": 140,
                "category_a": "vegetables",
                "category_b": "snacks",
                "product_family_a": "produce",
                "product_family_b": "chips",
            },
            {
                "product_a": 18,
                "product_b": 19,
                "product_a_name": "fresh carrots",
                "product_b_name": "indomie chicken noodles cup",
                "final_score": 97,
                "purchase_score": 88,
                "pair_count": 130,
                "category_a": "vegetables",
                "category_b": "grains",
                "product_family_a": "produce",
                "product_family_b": "noodles",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 85,
                "purchase_score": 42,
                "pair_count": 55,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 9,
                "product_b": 10,
                "product_a_name": "premium dates",
                "product_b_name": "fresh cream",
                "final_score": 84,
                "purchase_score": 35,
                "pair_count": 30,
                "category_a": "fruits",
                "category_b": "dairy",
                "product_family_a": "dates_family",
                "product_family_b": "dairy",
            },
        ]
    )


def _build_bundles_occasion_cross_lane_conflict() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 1,
                "product_b": 2,
                "product_a_name": "watania chicken breast",
                "product_b_name": "basmati rice",
                "final_score": 90,
                "purchase_score": 60,
                "pair_count": 80,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "poultry",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 5,
                "product_b": 6,
                "product_a_name": "black tea",
                "product_b_name": "white sugar",
                "final_score": 95,
                "purchase_score": 45,
                "pair_count": 70,
                "category_a": "beverages",
                "category_b": "baking",
                "product_family_a": "tea",
                "product_family_b": "sugar",
            },
            {
                "product_a": 25,
                "product_b": 5,
                "product_a_name": "salt crackers",
                "product_b_name": "black tea",
                "final_score": 93,
                "purchase_score": 44,
                "pair_count": 60,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "crackers",
                "product_family_b": "tea",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 82,
                "purchase_score": 35,
                "pair_count": 50,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
        ]
    )


def _build_bundles_packaging_leak() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 1,
                "product_b": 2,
                "product_a_name": "watania chicken breast",
                "product_b_name": "basmati rice",
                "final_score": 90,
                "purchase_score": 60,
                "pair_count": 80,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "poultry",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 3,
                "product_b": 109,
                "product_a_name": "potato chips",
                "product_b_name": "tea cups pack",
                "final_score": 99,
                "purchase_score": 95,
                "pair_count": 140,
                "category_a": "snacks",
                "category_b": "snacks",
                "product_family_a": "chips",
                "product_family_b": "party_supplies",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 86,
                "purchase_score": 42,
                "pair_count": 55,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 5,
                "product_b": 6,
                "product_a_name": "black tea",
                "product_b_name": "white sugar",
                "final_score": 88,
                "purchase_score": 35,
                "pair_count": 60,
                "category_a": "beverages",
                "category_b": "baking",
                "product_family_a": "tea",
                "product_family_b": "sugar",
            },
        ]
    )


def _build_bundles_occasion_fat_cheese_block() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 7,
                "product_b": 24,
                "product_a_name": "sunflower cooking oil",
                "product_b_name": "triangle cheese",
                "final_score": 99,
                "purchase_score": 92,
                "pair_count": 140,
                "category_a": "oils",
                "category_b": "dairy",
                "product_family_a": "cooking_oil",
                "product_family_b": "cheese",
            },
            {
                "product_a": 5,
                "product_b": 6,
                "product_a_name": "black tea",
                "product_b_name": "white sugar",
                "final_score": 88,
                "purchase_score": 35,
                "pair_count": 60,
                "category_a": "beverages",
                "category_b": "baking",
                "product_family_a": "tea",
                "product_family_b": "sugar",
            },
            {
                "product_a": 1,
                "product_b": 2,
                "product_a_name": "watania chicken breast",
                "product_b_name": "basmati rice",
                "final_score": 90,
                "purchase_score": 60,
                "pair_count": 80,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "poultry",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 86,
                "purchase_score": 42,
                "pair_count": 55,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
        ]
    )


def _build_bundles_feedback_scoring() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 1,
                "product_b": 2,
                "product_a_name": "watania chicken breast",
                "product_b_name": "basmati rice",
                "final_score": 88,
                "purchase_score": 56,
                "pair_count": 70,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "poultry",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 9,
                "product_b": 10,
                "product_a_name": "premium dates",
                "product_b_name": "fresh cream",
                "final_score": 86,
                "purchase_score": 38,
                "pair_count": 32,
                "category_a": "fruits",
                "category_b": "dairy",
                "product_family_a": "dates_family",
                "product_family_b": "dairy",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 90,
                "purchase_score": 39,
                "pair_count": 66,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 3,
                "product_b": 28,
                "product_a_name": "potato chips",
                "product_b_name": "kraft cheese portions",
                "final_score": 76,
                "purchase_score": 30,
                "pair_count": 34,
                "category_a": "snacks",
                "category_b": "dairy",
                "product_family_a": "chips",
                "product_family_b": "cheese_snack",
            },
        ]
    )


def _mixed_profile(profile_id: str = "p1") -> PersonProfile:
    return PersonProfile(
        profile_id=profile_id,
        source="manual",
        order_ids=[1001, 1002],
        history_product_ids=[1, 3, 5, 16],
        history_items=["watania chicken breast", "potato chips", "black tea", "fresh fish fillet"],
        created_at="2026-03-05T00:00:00+00:00",
        history_counts={1: 3, 3: 2, 5: 2, 16: 1},
    )


def _build_bundles_feedback_class_order() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 1,
                "product_b": 2,
                "product_a_name": "watania chicken breast",
                "product_b_name": "basmati rice",
                "final_score": 88,
                "purchase_score": 56,
                "pair_count": 70,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "poultry",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 9,
                "product_b": 10,
                "product_a_name": "premium dates",
                "product_b_name": "fresh cream",
                "final_score": 86,
                "purchase_score": 38,
                "pair_count": 32,
                "category_a": "fruits",
                "category_b": "dairy",
                "product_family_a": "dates_family",
                "product_family_b": "dairy",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 86,
                "purchase_score": 35,
                "pair_count": 50,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 3,
                "product_b": 28,
                "product_a_name": "potato chips",
                "product_b_name": "kraft cheese portions",
                "final_score": 85,
                "purchase_score": 35,
                "pair_count": 50,
                "category_a": "snacks",
                "category_b": "dairy",
                "product_family_a": "chips",
                "product_family_b": "cheese_snack",
            },
        ]
    )


def _build_bundles_diversity_pressure() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 1,
                "product_b": 2,
                "product_a_name": "watania chicken breast",
                "product_b_name": "basmati rice",
                "final_score": 93,
                "purchase_score": 66,
                "pair_count": 84,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "poultry",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 16,
                "product_b": 2,
                "product_a_name": "fresh fish fillet",
                "product_b_name": "basmati rice",
                "final_score": 92,
                "purchase_score": 64,
                "pair_count": 78,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "seafood",
                "product_family_b": "rice_centric",
            },
            {
                "product_a": 23,
                "product_b": 22,
                "product_a_name": "farm eggs",
                "product_b_name": "whole wheat tortilla bread",
                "final_score": 91,
                "purchase_score": 63,
                "pair_count": 76,
                "category_a": "protein",
                "category_b": "grains",
                "product_family_a": "eggs",
                "product_family_b": "bread",
            },
            {
                "product_a": 5,
                "product_b": 8,
                "product_a_name": "black tea",
                "product_b_name": "chocolate biscuit",
                "final_score": 92,
                "purchase_score": 58,
                "pair_count": 74,
                "category_a": "beverages",
                "category_b": "snacks",
                "product_family_a": "tea",
                "product_family_b": "biscuit",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 91,
                "purchase_score": 57,
                "pair_count": 70,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 3,
                "product_b": 28,
                "product_a_name": "potato chips",
                "product_b_name": "kraft cheese portions",
                "final_score": 90,
                "purchase_score": 56,
                "pair_count": 68,
                "category_a": "snacks",
                "category_b": "dairy",
                "product_family_a": "chips",
                "product_family_b": "cheese_snack",
            },
            {
                "product_a": 9,
                "product_b": 10,
                "product_a_name": "premium dates",
                "product_b_name": "fresh cream",
                "final_score": 92,
                "purchase_score": 56,
                "pair_count": 66,
                "category_a": "fruits",
                "category_b": "dairy",
                "product_family_a": "dates_family",
                "product_family_b": "dairy",
            },
            {
                "product_a": 9,
                "product_b": 11,
                "product_a_name": "premium dates",
                "product_b_name": "full fat milk",
                "final_score": 91,
                "purchase_score": 54,
                "pair_count": 64,
                "category_a": "fruits",
                "category_b": "dairy",
                "product_family_a": "dates_family",
                "product_family_b": "dairy",
            },
            {
                "product_a": 5,
                "product_b": 11,
                "product_a_name": "black tea",
                "product_b_name": "full fat milk",
                "final_score": 90,
                "purchase_score": 53,
                "pair_count": 62,
                "category_a": "beverages",
                "category_b": "dairy",
                "product_family_a": "tea",
                "product_family_b": "dairy",
            },
        ]
    )


def _build_bundles_prefer_top_over_early_copurchase() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "product_a": 2,
                "product_b": 3,
                "product_a_name": "basmati rice",
                "product_b_name": "potato chips",
                "final_score": 99,
                "purchase_score": 82,
                "pair_count": 140,
                "category_a": "grains",
                "category_b": "snacks",
                "product_family_a": "rice_centric",
                "product_family_b": "chips",
            },
            {
                "product_a": 1,
                "product_b": 14,
                "product_a_name": "watania chicken breast",
                "product_b_name": "tomato paste",
                "final_score": 98,
                "purchase_score": 72,
                "pair_count": 110,
                "category_a": "protein",
                "category_b": "vegetables",
                "product_family_a": "poultry",
                "product_family_b": "tomato_paste",
            },
            {
                "product_a": 3,
                "product_b": 4,
                "product_a_name": "potato chips",
                "product_b_name": "cola soda",
                "final_score": 91,
                "purchase_score": 52,
                "pair_count": 88,
                "category_a": "snacks",
                "category_b": "beverages",
                "product_family_a": "chips",
                "product_family_b": "beverage_soda",
            },
            {
                "product_a": 5,
                "product_b": 8,
                "product_a_name": "black tea",
                "product_b_name": "chocolate biscuit",
                "final_score": 90,
                "purchase_score": 45,
                "pair_count": 80,
                "category_a": "beverages",
                "category_b": "snacks",
                "product_family_a": "tea",
                "product_family_b": "biscuit",
            },
        ]
    )


def _feedback_snack_profile(profile_id: str) -> PersonProfile:
    return PersonProfile(
        profile_id=profile_id,
        source="manual",
        order_ids=[5101],
        history_product_ids=[3, 1, 9],
        history_items=["potato chips", "watania chicken breast", "premium dates"],
        created_at="2026-03-05T00:00:00+00:00",
        history_counts={3: 6, 1: 3, 9: 3},
    )


class PersonPredictionsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._base_dir = Path(cls._tmpdir.name)
        cls._paths_patcher = patch(
            "qeu_bundling.presentation.person_predictions.get_paths",
            side_effect=lambda project_root=None: real_get_paths(project_root=cls._base_dir),
        )
        cls._paths_patcher.start()

    @classmethod
    def tearDownClass(cls):
        cls._paths_patcher.stop()
        cls._tmpdir.cleanup()
        super().tearDownClass()

    def _run_stubbed_selection_case(
        self,
        *,
        case_id: str,
        lane_pairs: list[tuple[str, int, int, float]],
    ) -> list[dict[str, object]]:
        context = _build_context()
        history_ids = sorted(
            {
                int(pid)
                for _lane, anchor, complement, _score in lane_pairs
                for pid in (int(anchor), int(complement))
                if int(pid) > 0
            }
        )
        profile = PersonProfile(
            profile_id=f"p_stub_{case_id}",
            source="manual",
            order_ids=[99000 + len(history_ids)],
            history_product_ids=history_ids,
            history_items=[str(context.product_name_by_id.get(pid, f"Product {pid}")) for pid in history_ids],
            created_at="2026-03-15T00:00:00+00:00",
            history_counts={int(pid): 1 for pid in history_ids},
        )

        lane_ranked: dict[str, list[tuple[int, float]]] = {LANE_MEAL: [], LANE_SNACK: [], LANE_OCCASION: []}
        candidate_map: dict[tuple[str, int], tuple[int, float]] = {}
        for lane, anchor, complement, score in lane_pairs:
            lane_ranked.setdefault(str(lane), []).append((int(anchor), float(score)))
            key = (str(lane).strip().lower(), int(anchor))
            candidate_map[key] = (int(complement), float(score))

        def fake_rank(*_args, **_kwargs):
            return lane_ranked

        def fake_pick_candidate_for_anchor(*, anchor, lane, **_kwargs):
            key = (str(lane).strip().lower(), int(anchor))
            payload = candidate_map.get(key)
            if payload is None:
                return None, None, 0.0, 0
            complement, score = payload
            choice = {
                "anchor": int(anchor),
                "complement": int(complement),
                "source": "top_bundle",
                "cp_score": 90.0,
                "pair_count": 40,
                "recipe_compat": 0.9,
                "pair_strength": semantics.STRENGTH_STRONG,
                "lane_fit_score": 0.95,
                "template_strength": 1.0,
                "category_strength": 1.0,
                "prior_bonus": 0.2,
            }
            return choice, (int(anchor), int(complement)), float(score), 0

        def fake_effective_score(candidate, *_args, **_kwargs):
            return float(candidate.get("base_score", 0.0))

        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context), patch(
            "qeu_bundling.presentation.person_predictions.build_pair_multiplier_lookup", return_value={}
        ), patch(
            "qeu_bundling.presentation.person_predictions._rank_anchors_by_lane", side_effect=fake_rank
        ), patch(
            "qeu_bundling.presentation.person_predictions._pick_three_lane_anchors", return_value={}
        ), patch(
            "qeu_bundling.presentation.person_predictions._pick_candidate_for_anchor",
            side_effect=fake_pick_candidate_for_anchor,
        ), patch(
            "qeu_bundling.presentation.person_predictions._fallback_candidates_for_lane", return_value=[]
        ), patch(
            "qeu_bundling.presentation.person_predictions._candidate_personalization_boost", return_value=0.0
        ), patch(
            "qeu_bundling.presentation.person_predictions._candidate_is_strong_finalist", return_value=True
        ), patch(
            "qeu_bundling.presentation.person_predictions._candidate_effective_score",
            side_effect=fake_effective_score,
        ), patch(
            "qeu_bundling.presentation.person_predictions._final_human_quality_reject_reason", return_value=None
        ), patch(
            "qeu_bundling.presentation.person_predictions._passes_choice_score_floor", return_value=True
        ), patch(
            "qeu_bundling.presentation.person_predictions._write_person_quality_artifact", return_value=None
        ):
            recs = build_recommendations_for_profiles(
                bundles_df=pd.DataFrame(),
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id=f"run_stub_{case_id}",
            )
        self.assertEqual(len(recs), 1)
        return recs[0].get("bundles", [])

    def test_no_anchor_reuse_within_person_selection(self):
        bundles = self._run_stubbed_selection_case(
            case_id="no_anchor_reuse",
            lane_pairs=[
                (LANE_SNACK, 8, 11, 9.0),
                (LANE_OCCASION, 8, 9, 8.5),
                (LANE_MEAL, 1, 14, 8.0),
            ],
        )
        anchor_ids = [int(bundle.get("anchor_product_id", -1)) for bundle in bundles]
        self.assertLessEqual(anchor_ids.count(8), 1)

    def test_no_complement_reuse_within_person_selection(self):
        bundles = self._run_stubbed_selection_case(
            case_id="no_complement_reuse",
            lane_pairs=[
                (LANE_MEAL, 1, 14, 9.0),
                (LANE_SNACK, 8, 14, 8.5),
                (LANE_OCCASION, 5, 9, 8.0),
            ],
        )
        all_item_ids = [
            int(pid)
            for bundle in bundles
            for pid in (int(bundle.get("product_a", -1)), int(bundle.get("product_b", -1)))
            if int(pid) > 0
        ]
        self.assertLessEqual(all_item_ids.count(14), 1)

    def test_no_cross_role_item_reuse_within_person_selection(self):
        bundles = self._run_stubbed_selection_case(
            case_id="no_cross_role_reuse",
            lane_pairs=[
                (LANE_SNACK, 8, 9, 9.0),
                (LANE_OCCASION, 9, 5, 8.5),
                (LANE_MEAL, 1, 14, 8.0),
            ],
        )
        all_item_ids = [
            int(pid)
            for bundle in bundles
            for pid in (int(bundle.get("product_a", -1)), int(bundle.get("product_b", -1)))
            if int(pid) > 0
        ]
        self.assertLessEqual(all_item_ids.count(9), 1)

    def test_manual_parser_accepts_lines_and_commas(self):
        matcher = ProductMatcher(
            product_name_by_id={1: "tuna", 2: "mayonnaise", 3: "bread"},
            normalized_name_to_ids={"tuna": (1,), "mayonnaise": (2,), "bread": (3,)},
            normalized_names=("bread", "mayonnaise", "tuna"),
        )
        result = build_manual_profile("tuna, mayonnaise\n12345, unknown", matcher)
        self.assertIsNotNone(result.profile)
        assert result.profile is not None
        self.assertEqual(set(result.profile.history_product_ids), {1, 2})
        self.assertGreaterEqual(result.matched_count, 2)
        self.assertTrue(any("Unknown product id" in w for w in result.warnings))

    def test_random_profile_prefers_two_orders_and_falls_back_to_one(self):
        pool = OrderPool(
            order_product_ids={100: (1, 2), 101: (3, 4), 102: (5,)},
            order_product_names={100: ("A", "B"), 101: ("C", "D"), 102: ("E",)},
            preferred_order_ids=(100, 101),
            fallback_order_ids=(100, 101, 102),
        )
        profile = build_random_profile(pool, preferred_orders=2, fallback_orders=1, rng=random.Random(3))
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(len(profile.order_ids), 2)

    def test_food_bundles_allow_missing_lanes_without_forcing_junk(self):
        bundles = _build_bundles()
        profile = _mixed_profile("p_three_food")
        context = _build_context()
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_three_food",
            )
        self.assertEqual(len(recs), 1)
        lanes = [str(bundle.get("lane")) for bundle in recs[0]["bundles"]]
        self.assertTrue(set(lanes).issubset({LANE_MEAL, LANE_SNACK, LANE_OCCASION, LANE_NONFOOD}))
        self.assertGreaterEqual(len(lanes), 1)
        self.assertIn("missing_food_lanes", recs[0])
        self.assertTrue(set(recs[0]["missing_food_lanes"]).issubset({LANE_MEAL, LANE_SNACK, LANE_OCCASION}))

    def test_lane_set_food_exact_subset(self):
        bundles = _build_bundles()
        profile = _mixed_profile("p_lane_subset")
        context = _build_context()
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_lane_subset",
            )
        lane_set = {str(bundle.get("lane")) for bundle in recs[0]["bundles"]}
        self.assertTrue(lane_set.issubset({LANE_MEAL, LANE_SNACK, LANE_OCCASION, LANE_NONFOOD}))
        self.assertNotIn("staples", lane_set)

    def test_anchor_selection_is_deterministic_across_salts(self):
        bundles = _build_bundles()
        profile = _mixed_profile("p_seed")
        context = _build_context()
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            with patch("qeu_bundling.presentation.person_predictions.NONFOOD_INCLUDE_RATE", 0.0):
                recs_a = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_seed_a",
                    rng_salt="salt_a",
                )
                recs_b = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_seed_b",
                    rng_salt="salt_b",
                )
                recs_c = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_seed_c",
                    rng_salt=None,
                )
        pairs_a = [(int(b["anchor_product_id"]), int(b["complement_product_id"])) for b in recs_a[0]["bundles"]]
        pairs_b = [(int(b["anchor_product_id"]), int(b["complement_product_id"])) for b in recs_b[0]["bundles"]]
        pairs_c = [(int(b["anchor_product_id"]), int(b["complement_product_id"])) for b in recs_c[0]["bundles"]]
        self.assertEqual(pairs_a, pairs_b)
        self.assertEqual(pairs_a, pairs_c)

    def test_pick_three_lane_anchors_breaks_ties_by_product_id(self):
        selected = _pick_three_lane_anchors(
            {
                LANE_MEAL: [(2, 0.9), (1, 0.9)],
                LANE_SNACK: [(4, 0.8), (3, 0.8)],
                LANE_OCCASION: [(6, 0.7), (5, 0.7)],
            },
            random.Random(11),
        )
        self.assertEqual(selected, {LANE_MEAL: 1, LANE_SNACK: 3, LANE_OCCASION: 5})

    def test_pair_uniqueness_is_enforced_per_person_not_globally(self):
        bundles = _build_bundles()
        context = _build_context()
        profiles = [_mixed_profile("p_dup_1"), _mixed_profile("p_dup_2"), _mixed_profile("p_dup_3")]
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=profiles,
                max_people=3,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_dedupe",
            )
        for rec in recs:
            pairs = []
            for bundle in rec["bundles"]:
                pairs.append((int(bundle["anchor_product_id"]), int(bundle["complement_product_id"])))
            self.assertEqual(len(pairs), len(set(pairs)))

    def test_anchor_uniqueness_is_enforced_per_person(self):
        bundles = _build_bundles()
        context = _build_context()
        profiles = [
            _mixed_profile("p_cap_1"),
            _mixed_profile("p_cap_2"),
            _mixed_profile("p_cap_3"),
            _mixed_profile("p_cap_4"),
        ]
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=profiles,
                max_people=4,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_anchor_cap",
            )
        for rec in recs:
            anchors: list[int] = []
            for bundle in rec["bundles"]:
                lane = str(bundle.get("lane"))
                if lane not in {LANE_MEAL, LANE_SNACK, LANE_OCCASION}:
                    continue
                anchors.append(int(bundle.get("anchor_product_id", -1)))
            self.assertEqual(len(anchors), len(set(anchors)))

    def test_occasion_tea_sugar_rejected(self):
        context = _build_context()
        allowed = _passes_complement_gate(
            anchor=5,
            complement=6,
            context=context,
            cp_score=0.0,
            recipe_compat=0.0,
            prior_bonus=0.0,
            lane=LANE_OCCASION,
            pair_count=0,
        )
        self.assertFalse(allowed)

    def test_weak_semantic_snack_requires_stronger_evidence(self):
        context = _build_context()
        low_evidence_allowed = _passes_complement_gate(
            anchor=3,
            complement=4,
            context=context,
            cp_score=6.0,
            recipe_compat=0.0,
            prior_bonus=0.0,
            lane=LANE_SNACK,
            pair_count=1,
        )
        self.assertFalse(low_evidence_allowed)
        stronger_evidence_allowed = _passes_complement_gate(
            anchor=3,
            complement=4,
            context=context,
            cp_score=34.0,
            recipe_compat=0.20,
            prior_bonus=0.0,
            lane=LANE_SNACK,
            pair_count=12,
        )
        self.assertFalse(stronger_evidence_allowed)

    def test_free_product_b_for_all_bundles(self):
        bundles = _build_bundles()
        profile = _mixed_profile("p_free")
        context = _build_context()
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_free",
            )
        self.assertEqual(len(recs), 1)
        for bundle in recs[0]["bundles"]:
            self.assertEqual(bundle.get("free_product"), "product_b")
            if "price_after_discount_b" in bundle:
                self.assertEqual(float(bundle["price_after_discount_b"]), 0.0)
            if "price_after_b_sar" in bundle:
                self.assertEqual(str(bundle["price_after_b_sar"]), "0.00")

    def test_no_product_repeats_across_lanes_for_person(self):
        bundles = _build_bundles_occasion_cross_lane_conflict()
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_occasion_cross_lane_dedupe",
            source="manual",
            order_ids=[3101],
            history_product_ids=[1, 2, 3, 4, 5, 6, 25],
            history_items=[
                "watania chicken breast",
                "basmati rice",
                "potato chips",
                "cola soda",
                "black tea",
                "white sugar",
                "salt crackers",
            ],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={1: 3, 2: 3, 3: 2, 4: 2, 5: 5, 6: 4, 25: 4},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            with patch("qeu_bundling.presentation.person_predictions.NONFOOD_INCLUDE_RATE", 0.0):
                recs = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_occasion_cross_lane_dedupe",
                )
        self.assertEqual(len(recs), 1)
        bundles_for_person = recs[0].get("bundles", [])
        anchors = [int(bundle.get("anchor_product_id", -1)) for bundle in bundles_for_person]
        pair_ids = [tuple(sorted((int(bundle.get("product_a", -1)), int(bundle.get("product_b", -1))))) for bundle in bundles_for_person]
        self.assertEqual(len(anchors), len(set(anchors)))
        self.assertEqual(len(pair_ids), len(set(pair_ids)))

    def test_food_lanes_reject_packaging_utility_items_anchor_or_complement(self):
        context = _build_context()
        context.product_name_by_id[109] = "tea cups pack"
        context.product_price_by_id[109] = 3.0
        context.category_by_id[109] = "snacks"
        context.product_family_by_id[109] = "party_supplies"

        allowed_history = {3, 5}
        self.assertFalse(
            _passes_pair_filters(
                anchor=3,
                complement=109,
                history_ids=allowed_history,
                context=context,
                lane=LANE_SNACK,
            )
        )
        self.assertFalse(
            _passes_pair_filters(
                anchor=109,
                complement=5,
                history_ids=allowed_history,
                context=context,
                lane=LANE_OCCASION,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=3,
                complement=109,
                context=context,
                cp_score=90.0,
                recipe_compat=0.9,
                prior_bonus=1.0,
                lane=LANE_SNACK,
                pair_count=100,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=109,
                complement=5,
                context=context,
                cp_score=90.0,
                recipe_compat=0.9,
                prior_bonus=1.0,
                lane=LANE_OCCASION,
                pair_count=100,
            )
        )

    def test_nonfood_leakage_prevention_in_food_lanes(self):
        context = _build_context()
        self.assertFalse(
            _passes_complement_gate(
                anchor=3,
                complement=107,
                context=context,
                cp_score=95.0,
                recipe_compat=0.9,
                prior_bonus=1.0,
                lane=LANE_SNACK,
                pair_count=100,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=1,
                complement=103,
                context=context,
                cp_score=95.0,
                recipe_compat=0.9,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=100,
            )
        )

    def test_meal_lane_rejects_snack_pairs(self):
        context = _build_context()
        self.assertFalse(
            _passes_complement_gate(
                anchor=13,
                complement=3,
                context=context,
                cp_score=96.0,
                recipe_compat=0.9,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=110,
            )
        )

    def test_dessert_plus_meat_rejected(self):
        context = _build_context()
        self.assertFalse(
            _passes_complement_gate(
                anchor=8,
                complement=1,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=100,
            )
        )

    def test_snack_lane_rejects_grain_pair(self):
        context = _build_context()
        self.assertFalse(
            _passes_complement_gate(
                anchor=3,
                complement=2,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_SNACK,
                pair_count=100,
            )
        )

    def test_meal_rejects_spice_only_anchor(self):
        context = _build_context()
        self.assertFalse(_anchor_allowed_for_lane(20, LANE_MEAL, context))
        self.assertFalse(_anchor_allowed_for_lane(21, LANE_MEAL, context))

    def test_occasion_rejects_water_anchor(self):
        context = _build_context()
        context.product_name_by_id[110] = "premium mineral water"
        context.product_price_by_id[110] = 1.5
        context.category_by_id[110] = "beverages"
        context.product_family_by_id[110] = "water"
        self.assertFalse(_anchor_allowed_for_lane(110, LANE_OCCASION, context))

    def test_snack_rejects_pantry_anchors(self):
        context = _build_context()
        context.product_name_by_id[112] = "premium flour"
        context.product_price_by_id[112] = 3.0
        context.category_by_id[112] = "grains"
        context.product_family_by_id[112] = "flour"
        self.assertFalse(_anchor_allowed_for_lane(2, LANE_SNACK, context))
        self.assertFalse(_anchor_allowed_for_lane(7, LANE_SNACK, context))
        self.assertFalse(_anchor_allowed_for_lane(112, LANE_SNACK, context))

    def test_meal_rejects_chicken_plus_fruit(self):
        context = _build_context()
        self.assertFalse(
            _passes_complement_gate(
                anchor=1,
                complement=26,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )

    def test_occasion_rejects_dessert_plus_water(self):
        context = _build_context()
        context.product_name_by_id[110] = "premium mineral water"
        context.product_price_by_id[110] = 1.5
        context.category_by_id[110] = "beverages"
        context.product_family_by_id[110] = "water"
        self.assertFalse(
            _passes_complement_gate(
                anchor=8,
                complement=110,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_OCCASION,
                pair_count=120,
            )
        )

    def test_occasion_rejects_tea_plus_samosa_chips(self):
        context = _build_context()
        context.product_name_by_id[111] = "siniora beef samosa chips"
        context.product_price_by_id[111] = 6.0
        context.category_by_id[111] = "snacks"
        context.product_family_by_id[111] = "chips"
        self.assertFalse(
            _passes_complement_gate(
                anchor=5,
                complement=111,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_OCCASION,
                pair_count=120,
            )
        )

    def test_meal_rejects_tuna_plus_flour(self):
        context = _build_context()
        context.product_name_by_id[112] = "premium flour"
        context.product_price_by_id[112] = 3.0
        context.category_by_id[112] = "grains"
        context.product_family_by_id[112] = "flour"
        self.assertFalse(
            _passes_complement_gate(
                anchor=12,
                complement=112,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )

    def test_meal_rejects_spice_plus_spice(self):
        context = _build_context()
        self.assertFalse(
            _passes_complement_gate(
                anchor=21,
                complement=20,
                context=context,
                cp_score=45.0,
                recipe_compat=0.25,
                prior_bonus=0.0,
                lane=LANE_MEAL,
                pair_count=12,
            )
        )

    def test_meal_rejects_eggs_plus_cream_rice_cake(self):
        context = _build_context()
        context.product_name_by_id[113] = "topokki rice cake with cream flavor"
        context.product_price_by_id[113] = 7.8
        context.category_by_id[113] = "grains"
        context.product_family_by_id[113] = "rice_centric"
        self.assertFalse(
            _passes_complement_gate(
                anchor=23,
                complement=113,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )

    def test_meal_rejects_low_expression_pantry_pairs(self):
        context = _build_context()
        context.product_name_by_id[114] = "natural sea salt"
        context.product_price_by_id[114] = 2.2
        context.category_by_id[114] = "spices"
        context.product_family_by_id[114] = "spices"
        self.assertFalse(
            _passes_complement_gate(
                anchor=2,
                complement=7,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=7,
                complement=114,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )

    def test_meal_rejects_rice_plus_honey_mustard_sauce(self):
        context = _build_context()
        context.product_name_by_id[130] = "honey mustard sauce"
        context.product_price_by_id[130] = 4.2
        context.category_by_id[130] = "condiments"
        context.product_family_by_id[130] = "sauces"
        self.assertFalse(
            _passes_complement_gate(
                anchor=2,
                complement=130,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )

    def test_meal_rejects_eggs_plus_carrots_but_keeps_eggs_tomatoes_and_tomato_chicken(self):
        context = _build_context()
        context.product_name_by_id[131] = "fresh tomatoes"
        context.product_price_by_id[131] = 3.3
        context.category_by_id[131] = "vegetables"
        context.product_family_by_id[131] = "produce"
        self.assertFalse(
            _passes_complement_gate(
                anchor=23,
                complement=18,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )
        self.assertTrue(
            _passes_complement_gate(
                anchor=23,
                complement=131,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )
        self.assertTrue(
            _passes_complement_gate(
                anchor=1,
                complement=131,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )

    def test_meal_rejects_eggs_milk_and_oats_chicken(self):
        context = _build_context()
        context.product_name_by_id[115] = "white oats 500g"
        context.product_price_by_id[115] = 5.0
        context.category_by_id[115] = "grains"
        context.product_family_by_id[115] = "oats"
        self.assertFalse(
            _passes_complement_gate(
                anchor=23,
                complement=11,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=115,
                complement=1,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )

    def test_snack_rejects_moisturizing_cream_with_dates(self):
        context = _build_context()
        context.product_name_by_id[132] = "skin soothing moisturizing cream 50 ml"
        context.product_price_by_id[132] = 12.0
        context.category_by_id[132] = "personal care"
        context.product_family_by_id[132] = "personal_care"
        self.assertFalse(
            _passes_pair_filters(
                anchor=132,
                complement=9,
                history_ids={132, 9},
                context=context,
                lane=LANE_SNACK,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=132,
                complement=9,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_SNACK,
                pair_count=120,
            )
        )

    def test_occasion_rejects_cream_cheese_condensed_and_cooking_cream_biscuit(self):
        context = _build_context()
        context.product_name_by_id[116] = "sweetened condensed milk"
        context.product_price_by_id[116] = 4.5
        context.category_by_id[116] = "dairy"
        context.product_family_by_id[116] = "milk_dairy"
        context.product_name_by_id[117] = "cooking cream"
        context.product_price_by_id[117] = 5.2
        context.category_by_id[117] = "dairy"
        context.product_family_by_id[117] = "dairy"
        self.assertFalse(
            _passes_complement_gate(
                anchor=24,
                complement=116,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_OCCASION,
                pair_count=120,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=117,
                complement=25,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_OCCASION,
                pair_count=120,
            )
        )

    def test_occasion_rejects_milk_vimto_and_biscuit_cheese(self):
        context = _build_context()
        context.product_name_by_id[118] = "vimto syrup concentrate"
        context.product_price_by_id[118] = 5.8
        context.category_by_id[118] = "beverages"
        context.product_family_by_id[118] = "beverage_soda"
        self.assertFalse(
            _passes_complement_gate(
                anchor=11,
                complement=118,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_OCCASION,
                pair_count=120,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=25,
                complement=24,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_OCCASION,
                pair_count=120,
            )
        )

    def test_snack_rejects_mozzarella_style_cheese_pair(self):
        context = _build_context()
        context.product_name_by_id[119] = "grated mozzarella cheese"
        context.product_price_by_id[119] = 7.9
        context.category_by_id[119] = "dairy"
        context.product_family_by_id[119] = "cheese"
        self.assertFalse(
            _passes_complement_gate(
                anchor=3,
                complement=119,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_SNACK,
                pair_count=120,
            )
        )

    def test_occasion_milk_biscuit_requires_beverage_milk(self):
        context = _build_context()
        context.product_name_by_id[116] = "sweetened condensed milk"
        context.product_price_by_id[116] = 4.5
        context.category_by_id[116] = "dairy"
        context.product_family_by_id[116] = "milk_dairy"
        context.product_name_by_id[120] = "evaporated milk"
        context.product_price_by_id[120] = 3.8
        context.category_by_id[120] = "dairy"
        context.product_family_by_id[120] = "milk_dairy"
        self.assertFalse(
            _passes_complement_gate(
                anchor=120,
                complement=8,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_OCCASION,
                pair_count=120,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=116,
                complement=8,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=1.0,
                lane=LANE_OCCASION,
                pair_count=120,
            )
        )

    def test_snack_accepts_biscuit_plus_milk_as_explicit_pattern(self):
        context = _build_context()
        self.assertFalse(
            _passes_complement_gate(
                anchor=8,
                complement=11,
                context=context,
                cp_score=55.0,
                recipe_compat=0.0,
                prior_bonus=0.0,
                lane=LANE_SNACK,
                pair_count=24,
            )
        )

    def test_low_evidence_suspicious_defaults_are_rejected(self):
        context = _build_context()
        self.assertFalse(
            _passes_complement_gate(
                anchor=2,
                complement=14,
                context=context,
                cp_score=22.0,
                recipe_compat=0.16,
                prior_bonus=0.0,
                lane=LANE_MEAL,
                pair_count=8,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=5,
                complement=8,
                context=context,
                cp_score=28.0,
                recipe_compat=0.10,
                prior_bonus=0.0,
                lane=LANE_OCCASION,
                pair_count=10,
            )
        )

    def test_snack_contains_no_packaging_utility_items(self):
        bundles = _build_bundles_packaging_leak()
        context = _build_context()
        context.product_name_by_id[109] = "tea cups pack"
        context.product_price_by_id[109] = 3.0
        context.category_by_id[109] = "snacks"
        context.product_family_by_id[109] = "party_supplies"
        profile = PersonProfile(
            profile_id="p_snack_packaging_block",
            source="manual",
            order_ids=[3501],
            history_product_ids=[3, 5, 1, 2],
            history_items=["potato chips", "black tea", "watania chicken breast", "basmati rice"],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={3: 5, 5: 4, 1: 2, 2: 2},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_snack_packaging_block",
            )
        self.assertEqual(len(recs), 1)
        snack = [b for b in recs[0]["bundles"] if str(b.get("lane")) == LANE_SNACK]
        self.assertEqual(len(snack), 1)
        names = f"{snack[0].get('product_a_name', '').lower()}::{snack[0].get('product_b_name', '').lower()}"
        self.assertNotIn("cup", names)
        self.assertNotIn("cups", names)

    def test_occasion_blocks_fat_plus_cheese_spread(self):
        context = _build_context()
        blocked = _passes_complement_gate(
            anchor=7,
            complement=24,
            context=context,
            cp_score=95.0,
            recipe_compat=0.8,
            prior_bonus=1.0,
            lane=LANE_OCCASION,
            pair_count=120,
        )
        self.assertFalse(blocked)

        bundles = _build_bundles_occasion_fat_cheese_block()
        profile = PersonProfile(
            profile_id="p_occasion_fat_cheese_block",
            source="manual",
            order_ids=[3601],
            history_product_ids=[7, 24, 5, 6, 1, 3],
            history_items=[
                "sunflower cooking oil",
                "triangle cheese",
                "black tea",
                "white sugar",
                "watania chicken breast",
                "potato chips",
            ],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={7: 8, 24: 7, 5: 5, 6: 4, 1: 3, 3: 3},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_occasion_fat_cheese_block",
            )
        self.assertEqual(len(recs), 1)
        for bundle in recs[0].get("bundles", []):
            pair = {int(bundle.get("product_a", -1)), int(bundle.get("product_b", -1))}
            self.assertNotEqual(pair, {7, 24})

    def test_snack_is_pattern_driven_only(self):
        bundles = _build_bundles_snack_pattern_only()
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_snack_pattern_only",
            source="manual",
            order_ids=[3001],
            history_product_ids=[22, 3, 5, 1, 9],
            history_items=[
                "whole wheat tortilla bread",
                "potato chips",
                "black tea",
                "watania chicken breast",
                "premium dates",
            ],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={22: 5, 3: 4, 5: 3, 1: 3, 9: 2},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_snack_pattern_only",
            )
        self.assertEqual(len(recs), 1)
        snack = [b for b in recs[0]["bundles"] if str(b.get("lane")) == LANE_SNACK]
        for row in snack:
            self.assertIn(
                str(row.get("snack_pattern", "")),
                {"drink_snack", "tea_snack", "cookie_milk", "sweet_milk", "wafer_chocolate", "dates_cream", "nuts_drink", "cheese_snack"},
            )

    def test_snack_never_bread_adjacent(self):
        bundles = _build_bundles_snack_pattern_only()
        context = _build_context()
        profile = _mixed_profile("p_snack_no_bread")
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_snack_no_bread",
            )
        snack = [b for b in recs[0]["bundles"] if str(b.get("lane")) == LANE_SNACK]
        self.assertGreaterEqual(len(snack), 1)
        for row in snack:
            names = f"{row.get('product_a_name', '').lower()}::{row.get('product_b_name', '').lower()}"
            self.assertNotIn("bread", names)
            self.assertNotIn("tortilla", names)
            self.assertNotIn("toast", names)
            self.assertNotIn("wrap", names)

    def test_meal_snack_not_same_theme(self):
        bundles = _build_bundles_snack_pattern_only()
        context = _build_context()
        profile = _mixed_profile("p_theme_diverse")
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_theme_diverse",
            )
        meal = [b for b in recs[0]["bundles"] if str(b.get("lane")) == LANE_MEAL][0]
        snack = [b for b in recs[0]["bundles"] if str(b.get("lane")) == LANE_SNACK][0]
        self.assertNotEqual(str(meal.get("bundle_theme", "")), str(snack.get("bundle_theme", "")))

    def test_blocks_garlic_chips_and_produce_noodles_in_snack(self):
        bundles = _build_bundles_snack_negative_block()
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_snack_blocks_produce",
            source="manual",
            order_ids=[3002],
            history_product_ids=[17, 18, 3, 5, 1, 9],
            history_items=[
                "fresh garlic",
                "fresh carrots",
                "potato chips",
                "black tea",
                "watania chicken breast",
                "premium dates",
            ],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={17: 4, 18: 4, 3: 4, 5: 3, 1: 3, 9: 2},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_snack_blocks_produce",
            )
        names = " ".join(
            f"{b.get('product_a_name', '').lower()}::{b.get('product_b_name', '').lower()}" for b in recs[0]["bundles"]
        )
        self.assertNotIn("garlic::potato chips", names)
        self.assertNotIn("carrots", names)
        self.assertNotIn("indomie", names)
        self.assertNotIn("noodles", names)

    def test_meal_blocks_produce_plus_chips(self):
        bundles = _build_bundles_meal_produce_snack_guardrail()
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_meal_produce_snack",
            source="manual",
            order_ids=[2001],
            history_product_ids=[17, 3, 5],
            history_items=["fresh garlic", "potato chips", "black tea"],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={17: 8, 3: 2, 5: 2},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_meal_guardrail_snack",
            )
        self.assertEqual(len(recs), 1)
        meal = [b for b in recs[0]["bundles"] if str(b.get("lane")) == LANE_MEAL]
        self.assertGreaterEqual(len(meal), 1)
        for row in meal:
            pair_names = f"{row.get('product_a_name', '').lower()}::{row.get('product_b_name', '').lower()}"
            self.assertNotIn("potato chips", pair_names)

    def test_meal_blocks_produce_plus_noodles(self):
        bundles = _build_bundles_meal_produce_noodles_guardrail()
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_meal_produce_noodles",
            source="manual",
            order_ids=[2002],
            history_product_ids=[18, 3, 5],
            history_items=["fresh carrots", "potato chips", "black tea"],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={18: 8, 3: 2, 5: 2},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_meal_guardrail_noodles",
            )
        self.assertEqual(len(recs), 1)
        meal = [b for b in recs[0]["bundles"] if str(b.get("lane")) == LANE_MEAL]
        self.assertGreaterEqual(len(meal), 1)
        for row in meal:
            pair_names = f"{row.get('product_a_name', '').lower()}::{row.get('product_b_name', '').lower()}"
            self.assertNotIn("noodles", pair_names)

    def test_fallback_templates_recover_lane_when_normal_path_fails(self):
        bundles = _build_bundles()
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_fallback_recovery",
            source="manual",
            order_ids=[3801, 3802],
            history_product_ids=[1, 2, 3, 4, 5, 6, 9, 10, 14],
            history_items=[
                "watania chicken breast",
                "basmati rice",
                "potato chips",
                "cola soda",
                "black tea",
                "white sugar",
                "premium dates",
                "fresh cream",
                "tomato paste",
            ],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={1: 3, 2: 4, 3: 3, 4: 2, 5: 3, 6: 3, 9: 2, 10: 2, 14: 2},
        )
        from qeu_bundling.presentation import person_predictions as pp

        original_pick = pp._pick_candidate_for_anchor
        call_counters = {"normal": 0, "fallback": 0}

        def _side_effect(*args, **kwargs):
            if kwargs.get("allowed_complements"):
                call_counters["fallback"] += 1
                return original_pick(*args, **kwargs)
            call_counters["normal"] += 1
            return None, None, 0.0, 0

        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            with patch("qeu_bundling.presentation.person_predictions._pick_candidate_for_anchor", side_effect=_side_effect):
                recs = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_fallback_recovery",
                )
        self.assertEqual(len(recs), 1)
        self.assertGreater(call_counters["normal"], 0)
        self.assertGreater(call_counters["fallback"], 0)
        lanes = {str(bundle.get("lane")) for bundle in recs[0]["bundles"]}
        self.assertTrue(bool(lanes & {LANE_MEAL, LANE_SNACK, LANE_OCCASION}))

    def test_lane_selection_prefers_top_bundle_from_later_anchor_over_early_copurchase(self):
        bundles = _build_bundles_prefer_top_over_early_copurchase()
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_prefer_top_later_anchor",
            source="manual",
            order_ids=[3901],
            history_product_ids=[2, 1, 14, 3, 4, 5, 8],
            history_items=[
                "basmati rice",
                "watania chicken breast",
                "tomato paste",
                "potato chips",
                "cola soda",
                "black tea",
                "chocolate biscuit",
            ],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={2: 9, 1: 3, 14: 2, 3: 2, 4: 1, 5: 1, 8: 1},
        )
        forced_anchors = {LANE_MEAL: 2, LANE_SNACK: 3, LANE_OCCASION: 5}
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            with patch("qeu_bundling.presentation.person_predictions.NONFOOD_INCLUDE_RATE", 0.0):
                with patch("qeu_bundling.presentation.person_predictions._pick_three_lane_anchors", return_value=forced_anchors):
                    recs = build_recommendations_for_profiles(
                        bundles_df=bundles,
                        profiles=[profile],
                        max_people=1,
                        row_to_record=lambda row: row.to_dict(),
                        run_id="run_prefer_top_later_anchor",
                    )
        self.assertEqual(len(recs), 1)
        self.assertTrue(
            any(
                str(bundle.get("recommendation_origin")) == "top_bundle"
                and {int(bundle.get("product_a", -1)), int(bundle.get("product_b", -1))} == {1, 14}
                for bundle in recs[0]["bundles"]
            )
        )

    def test_fallback_templates_do_not_admit_weak_risky_visible_pairs(self):
        context = _build_context()
        context.product_name_by_id[116] = "sweetened condensed milk"
        context.product_price_by_id[116] = 4.5
        context.category_by_id[116] = "dairy"
        context.product_family_by_id[116] = "milk_dairy"
        context.product_name_by_id[119] = "grated mozzarella cheese"
        context.product_price_by_id[119] = 7.9
        context.category_by_id[119] = "dairy"
        context.product_family_by_id[119] = "cheese"
        context.neighbors[3] = ((119, 70.0),)
        context.neighbors[11] = ((8, 70.0),)
        context.neighbors[116] = ((8, 70.0),)
        bundle_lookup: dict[tuple[int, int], pd.Series] = {}
        snack_fallback = _fallback_candidates_for_lane(
            history_ids={3, 119},
            lane=LANE_SNACK,
            context=context,
            top_bundle_rows_by_anchor={},
            bundle_lookup=bundle_lookup,
        )
        occasion_fallback = _fallback_candidates_for_lane(
            history_ids={8, 11, 116},
            lane=LANE_OCCASION,
            context=context,
            top_bundle_rows_by_anchor={},
            bundle_lookup=bundle_lookup,
        )
        meal_fallback = _fallback_candidates_for_lane(
            history_ids={2, 14},
            lane=LANE_MEAL,
            context=context,
            top_bundle_rows_by_anchor={},
            bundle_lookup=bundle_lookup,
        )
        self.assertFalse(any({int(a), int(b)} == {3, 119} for a, b, _row, _source in snack_fallback))
        self.assertFalse(any({int(a), int(b)} == {11, 8} for a, b, _row, _source in occasion_fallback))
        self.assertFalse(any({int(a), int(b)} == {116, 8} for a, b, _row, _source in occasion_fallback))
        self.assertFalse(any({int(a), int(b)} == {2, 14} for a, b, _row, _source in meal_fallback))

    def test_fallback_templates_prefer_valid_orientation_for_history(self):
        context = _build_context()
        context.product_name_by_id[120] = "evaporated milk"
        context.product_price_by_id[120] = 3.8
        context.category_by_id[120] = "dairy"
        context.product_family_by_id[120] = "milk_dairy"
        row = pd.Series(
            {
                "product_a": 5,
                "product_b": 120,
                "product_a_name": "black tea",
                "product_b_name": "evaporated milk",
                "final_score": 83,
                "purchase_score": 38,
                "pair_count": 20,
                "category_a": "beverages",
                "category_b": "dairy",
                "product_family_a": "tea",
                "product_family_b": "milk_dairy",
            }
        )
        fallback = _fallback_candidates_for_lane(
            history_ids={120},
            lane=LANE_OCCASION,
            context=context,
            top_bundle_rows_by_anchor={5: [row], 120: [row]},
            bundle_lookup={(5, 120): row},
        )
        self.assertTrue(fallback)
        anchor, complement, _row, source = fallback[0]
        self.assertEqual({int(anchor), int(complement)}, {120, 5})
        self.assertTrue(str(source).startswith("fallback:occasion:"))

    def test_fallback_templates_reject_weak_snack_biscuit_milk(self):
        context = _build_context()
        row = pd.Series(
            {
                "product_a": 8,
                "product_b": 11,
                "product_a_name": "chocolate biscuit",
                "product_b_name": "full fat milk",
                "final_score": 81,
                "purchase_score": 36,
                "pair_count": 14,
                "category_a": "snacks",
                "category_b": "dairy",
                "product_family_a": "biscuit",
                "product_family_b": "milk_dairy",
            }
        )
        fallback = _fallback_candidates_for_lane(
            history_ids={8, 11},
            lane=LANE_SNACK,
            context=context,
            top_bundle_rows_by_anchor={8: [row], 11: [row]},
            bundle_lookup={(8, 11): row},
        )
        self.assertFalse(any({int(a), int(b)} == {8, 11} for a, b, _row, _source in fallback))

    def test_fallback_templates_allow_strong_occasion_tea_biscuit(self):
        context = _build_context()
        row = pd.Series(
            {
                "product_a": 5,
                "product_b": 8,
                "product_a_name": "black tea",
                "product_b_name": "chocolate biscuit",
                "final_score": 82,
                "purchase_score": 37,
                "pair_count": 15,
                "category_a": "beverages",
                "category_b": "snacks",
                "product_family_a": "tea",
                "product_family_b": "biscuit",
            }
        )
        fallback = _fallback_candidates_for_lane(
            history_ids={5, 8},
            lane=LANE_OCCASION,
            context=context,
            top_bundle_rows_by_anchor={5: [row], 8: [row]},
            bundle_lookup={(5, 8): row},
        )
        self.assertTrue(
            any(
                {int(a), int(b)} == {5, 8} and str(source).startswith("fallback:occasion:")
                for a, b, _row, source in fallback
            )
        )

    def test_spices_bundle_rejected_in_meal(self):
        context = _build_context()
        allowed = _passes_complement_gate(
            anchor=21,
            complement=20,
            context=context,
            cp_score=45.0,
            recipe_compat=0.25,
            prior_bonus=0.0,
            lane=LANE_MEAL,
            pair_count=12,
        )
        self.assertFalse(allowed)

    def test_nonfood_lane_included_for_some_profiles_approx_share(self):
        bundles = _build_bundles()
        context = _build_context()
        positives: list[str] = []
        negatives: list[str] = []
        idx = 0
        while len(positives) < 2 or len(negatives) < 8:
            pid = f"p_nonfood_share_{idx}"
            score_rng = _rng_for_profile("run_nonfood_share", pid, rng_salt=None)
            if score_rng.random() < 0.20:
                positives.append(pid)
            else:
                negatives.append(pid)
            idx += 1
            if idx > 2000:
                break
        profile_ids = positives[:2] + negatives[:8]
        profiles = [_mixed_profile(pid) for pid in profile_ids]
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=profiles,
                max_people=10,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_nonfood_share",
            )
        nonfood_count = sum(
            1
            for rec in recs
            if any(str(bundle.get("lane")) == LANE_NONFOOD for bundle in rec.get("bundles", []))
        )
        self.assertGreaterEqual(nonfood_count, 2)
        self.assertLessEqual(nonfood_count, 10)

    def test_nonfood_lane_rate_is_bounded(self):
        bundles = _build_bundles()
        context = _build_context()
        positives: list[str] = []
        negatives: list[str] = []
        idx = 0
        while len(positives) < 4 or len(negatives) < 16:
            pid = f"p_nonfood_bound_{idx}"
            gate_rng = _rng_for_profile("run_nonfood_bound", pid, rng_salt="::nonfood_gate")
            if gate_rng.random() < 0.20:
                positives.append(pid)
            else:
                negatives.append(pid)
            idx += 1
            if idx > 4000:
                break
        profile_ids = positives[:4] + negatives[:16]
        profiles = [_mixed_profile(pid) for pid in profile_ids]
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=profiles,
                max_people=20,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_nonfood_bound",
            )
        nonfood_count = sum(
            1
            for rec in recs
            if any(str(bundle.get("lane")) == LANE_NONFOOD for bundle in rec.get("bundles", []))
        )
        self.assertGreaterEqual(nonfood_count, 4)
        self.assertLessEqual(nonfood_count, 20)

    def test_nonfood_pairs_must_be_close(self):
        context = _build_context()
        allowed = _passes_complement_gate(
            anchor=101,
            complement=102,
            context=context,
            cp_score=30.0,
            recipe_compat=0.0,
            prior_bonus=0.0,
            lane=LANE_NONFOOD,
            pair_count=10,
        )
        rejected = _passes_complement_gate(
            anchor=101,
            complement=2,
            context=context,
            cp_score=90.0,
            recipe_compat=0.9,
            prior_bonus=1.0,
            lane=LANE_NONFOOD,
            pair_count=100,
        )
        self.assertTrue(allowed)
        self.assertFalse(rejected)

    def test_nonfood_lane_category_close(self):
        bundles = _build_bundles()
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_nonfood_close_lane",
            source="manual",
            order_ids=[3401],
            history_product_ids=[101, 103, 1, 3, 5],
            history_items=["herbal shampoo", "laundry detergent", "watania chicken breast", "potato chips", "black tea"],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={101: 3, 103: 3, 1: 2, 3: 2, 5: 2},
        )

        def _coarse_nonfood_group(name: str) -> str:
            text = str(name).lower()
            if any(token in text for token in ("shampoo", "conditioner")):
                return "hair"
            if any(token in text for token in ("detergent", "softener", "disinfectant", "bleach", "dishwashing", "cleaner")):
                return "cleaning"
            if any(token in text for token in ("soap", "body wash", "shower gel")):
                return "body"
            if any(token in text for token in ("tissue", "paper towel", "toilet paper")):
                return "tissue"
            return "other"

        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            with patch("qeu_bundling.presentation.person_predictions.NONFOOD_INCLUDE_RATE", 1.0):
                recs = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_nonfood_lane_close",
                )
        nonfood = [b for b in recs[0]["bundles"] if str(b.get("lane")) == LANE_NONFOOD]
        self.assertLessEqual(len(nonfood), 1)
        if nonfood:
            row = nonfood[0]
            group_a = _coarse_nonfood_group(str(row.get("product_a_name", "")))
            group_b = _coarse_nonfood_group(str(row.get("product_b_name", "")))
            self.assertEqual(group_a, group_b)

    def test_swap_makes_free_item_cheaper_when_possible(self):
        bundles = _build_bundles_for_swap()
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_swap_to_cheaper_free",
            source="manual",
            order_ids=[2201],
            history_product_ids=[22, 3, 5],
            history_items=["whole wheat tortilla bread", "potato chips", "black tea"],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={22: 9, 3: 2, 5: 2},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_swap_cheaper_free",
            )
        self.assertEqual(len(recs), 1)
        meal = next(
            b
            for b in recs[0]["bundles"]
            if {int(b.get("product_a", -1)), int(b.get("product_b", -1))} == {22, 23}
        )
        self.assertTrue(bool(meal.get("swapped")))
        self.assertEqual(meal.get("swap_reason"), "make_free_item_cheaper")
        self.assertEqual(meal.get("free_product"), "product_b")
        self.assertLessEqual(float(meal.get("product_b_price", 0.0)), float(meal.get("product_a_price", 0.0)))
        self.assertEqual(int(meal.get("anchor_product_id", -1)), 22)
        self.assertEqual(int(meal.get("complement_product_id", -1)), 23)
        self.assertTrue(bool(meal.get("anchor_in_history", False)))
        self.assertEqual(str(meal.get("price_after_b_sar", "")), "0.00")

    def test_no_swap_when_prices_missing(self):
        context = _build_context()
        display = {
            "product_a": 22,
            "product_b": 23,
            "product_a_name": "whole wheat tortilla bread",
            "product_b_name": "farm eggs",
        }
        _maybe_swap_to_make_free_item_cheaper(display, lane=LANE_MEAL, context=context)
        self.assertFalse(bool(display.get("swapped")))
        self.assertEqual(display.get("free_product"), "product_b")
        self.assertEqual(int(display.get("product_a", -1)), 22)
        self.assertEqual(int(display.get("product_b", -1)), 23)

    def test_staple_anchor_penalty_reduces_staple_priority(self):
        profile = PersonProfile(
            profile_id="p_staple_rank",
            source="manual",
            order_ids=[1],
            history_product_ids=[2, 1],
            history_items=["basmati rice", "watania chicken breast"],
            created_at="2026-03-05T00:00:00+00:00",
            history_counts={2: 2, 1: 1},
        )
        context = _build_context()
        ranked = _rank_anchors(profile, context, random.Random(11))
        self.assertTrue(ranked)
        self.assertNotEqual(ranked[0], 2)
        self.assertTrue(_is_staple_product("basmati rice", "rice_centric"))

    def test_feedback_good_pair_boosts_ranking(self):
        feedback_lookup = {
            "pair_boosts": {(3, 28): 0.55},
            "pair_penalties": {},
            "pair_overrides": set(),
            "good_pairs": {(3, 28)},
            "bad_pairs": set(),
        }
        with patch("qeu_bundling.presentation.person_predictions.FEEDBACK_LOOKUP", feedback_lookup):
            self.assertGreater(_feedback_pair_boost(3, 28), 0.0)

    def test_feedback_bad_pair_penalizes_ranking(self):
        bad_feedback = {
            "pair_boosts": {},
            "pair_penalties": {(3, 28): 0.70},
            "pair_overrides": set(),
            "good_pairs": set(),
            "bad_pairs": {(3, 28)},
        }
        with patch("qeu_bundling.presentation.person_predictions.FEEDBACK_LOOKUP", bad_feedback):
            self.assertGreater(_feedback_pair_penalty(3, 28), 0.0)

    def test_feedback_override_does_not_bypass_hard_invalid(self):
        context = _build_context()
        context.product_name_by_id[113] = "topokki rice cake with cream flavor"
        context.product_price_by_id[113] = 7.8
        context.category_by_id[113] = "grains"
        context.product_family_by_id[113] = "rice_centric"

        self.assertFalse(
            _passes_pair_filters(
                anchor=23,
                complement=113,
                history_ids={23, 113},
                context=context,
                lane=LANE_MEAL,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=23,
                complement=113,
                context=context,
                cp_score=90.0,
                recipe_compat=0.8,
                prior_bonus=0.0,
                lane=LANE_MEAL,
                pair_count=20,
            )
        )
        feedback_lookup = {
            "pair_boosts": {(23, 113): 0.30},
            "pair_penalties": {},
            "pair_overrides": {(23, 113)},
            "good_pairs": {(23, 113)},
            "bad_pairs": set(),
        }
        with patch("qeu_bundling.presentation.person_predictions.FEEDBACK_LOOKUP", feedback_lookup):
            self.assertFalse(
                _passes_pair_filters(
                    anchor=23,
                    complement=113,
                    history_ids={23, 113},
                    context=context,
                    lane=LANE_MEAL,
                )
            )
            self.assertFalse(
                _passes_complement_gate(
                    anchor=23,
                    complement=113,
                    context=context,
                    cp_score=90.0,
                    recipe_compat=0.8,
                    prior_bonus=0.0,
                    lane=LANE_MEAL,
                    pair_count=20,
                )
            )

    def test_feedback_scoring_keeps_determinism(self):
        bundles = _build_bundles_feedback_scoring()
        context = _build_context()
        profile = _feedback_snack_profile("p_feedback_deterministic")
        feedback_lookup = {
            "pair_boosts": {(3, 28): 0.40},
            "pair_penalties": {(3, 4): 0.15},
            "pair_overrides": set(),
            "good_pairs": {(3, 28)},
            "bad_pairs": {(3, 4)},
        }
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            with patch("qeu_bundling.presentation.person_predictions.FEEDBACK_LOOKUP", feedback_lookup):
                recs_a = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_feedback_det",
                    rng_salt="same",
                )
                recs_b = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_feedback_det",
                    rng_salt="same",
                )
        pairs_a = [(int(b["anchor_product_id"]), int(b["complement_product_id"])) for b in recs_a[0]["bundles"]]
        pairs_b = [(int(b["anchor_product_id"]), int(b["complement_product_id"])) for b in recs_b[0]["bundles"]]
        self.assertEqual(pairs_a, pairs_b)

    def test_feedback_classes_strong_outrank_staple(self):
        feedback_lookup = {
            "strong_pairs": {(3, 28)},
            "staple_pairs": {(3, 4)},
            "weak_pairs": set(),
            "trash_pairs": set(),
            "pair_boosts": {},
            "pair_penalties": {},
            "pair_overrides": set(),
            "good_pairs": {(3, 28)},
            "bad_pairs": set(),
        }
        with patch("qeu_bundling.presentation.person_predictions.FEEDBACK_LOOKUP", feedback_lookup):
            self.assertEqual(_feedback_pair_class(3, 28), "strong")

    def test_feedback_classes_staple_outrank_weak(self):
        bundles = _build_bundles_feedback_class_order()
        bundles = bundles[bundles["product_a"] == 3].reset_index(drop=True)
        bundles = pd.concat(
            [
                bundles,
                pd.DataFrame(
                    [
                        {
                            "product_a": 3,
                            "product_b": 27,
                            "product_a_name": "potato chips",
                            "product_b_name": "cola soda zero",
                            "final_score": 87,
                            "purchase_score": 35,
                            "pair_count": 50,
                            "category_a": "snacks",
                            "category_b": "beverages",
                            "product_family_a": "chips",
                            "product_family_b": "beverage_soda",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        feedback_lookup = {
            "strong_pairs": set(),
            "staple_pairs": {(3, 28)},
            "weak_pairs": {(3, 27)},
            "trash_pairs": set(),
            "pair_boosts": {},
            "pair_penalties": {},
            "pair_overrides": set(),
            "good_pairs": set(),
            "bad_pairs": set(),
        }
        with patch("qeu_bundling.presentation.person_predictions.FEEDBACK_LOOKUP", feedback_lookup):
            self.assertEqual(_feedback_pair_class(3, 28), "staple")

    def test_feedback_classes_trash_heavily_penalized(self):
        bundles = _build_bundles_feedback_class_order()
        bundles = bundles[bundles["product_a"] == 3].reset_index(drop=True)
        bundles = pd.concat(
            [
                bundles,
                pd.DataFrame(
                    [
                        {
                            "product_a": 3,
                            "product_b": 27,
                            "product_a_name": "potato chips",
                            "product_b_name": "cola soda zero",
                            "final_score": 92,
                            "purchase_score": 38,
                            "pair_count": 65,
                            "category_a": "snacks",
                            "category_b": "beverages",
                            "product_family_a": "chips",
                            "product_family_b": "beverage_soda",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        feedback_lookup = {
            "strong_pairs": set(),
            "staple_pairs": {(3, 4)},
            "weak_pairs": set(),
            "trash_pairs": {(3, 28)},
            "pair_boosts": {},
            "pair_penalties": {},
            "pair_overrides": set(),
            "good_pairs": set(),
            "bad_pairs": {(3, 28)},
        }
        with patch("qeu_bundling.presentation.person_predictions.FEEDBACK_LOOKUP", feedback_lookup):
            self.assertEqual(_feedback_pair_class(3, 28), "trash")
            self.assertEqual(_feedback_pair_class(3, 4), "staple")

    def test_semantic_diagnostics_are_additive_in_bundle_payload(self):
        bundles = _build_bundles()
        context = _build_context()
        profile = _mixed_profile("p_semantic_diag")
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_semantic_diag",
            )
        self.assertEqual(len(recs), 1)
        self.assertGreaterEqual(len(recs[0].get("bundles", [])), 1)
        bundle = recs[0]["bundles"][0]
        self.assertIn("semantic_roles_a", bundle)
        self.assertIn("semantic_roles_b", bundle)
        self.assertIn("pair_relation", bundle)
        self.assertIn("pair_strength", bundle)
        self.assertIn("lane_fit_score", bundle)
        self.assertIn("internal_lane_fit", bundle)
        self.assertIn("semantic_engine_version", bundle)

    def test_finalized_bundle_payload_keeps_origin_and_ids(self):
        bundles = _build_bundles()
        context = _build_context()
        profile = _mixed_profile("p_payload_fields")
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_payload_fields",
            )
        self.assertEqual(len(recs), 1)
        self.assertTrue(recs[0].get("bundles"))
        bundle = recs[0]["bundles"][0]
        self.assertIn("recommendation_origin", bundle)
        self.assertIn("recommendation_origin_label", bundle)
        self.assertIn("anchor_product_id", bundle)
        self.assertIn("complement_product_id", bundle)
        self.assertIn("free_product", bundle)
        self.assertEqual(str(bundle.get("free_product")), "product_b")
        self.assertGreater(int(bundle.get("anchor_product_id", 0)), 0)
        self.assertGreater(int(bundle.get("complement_product_id", 0)), 0)

    def test_missing_lane_allowed_when_no_semantic_candidate_survives(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 11,
                    "product_b": 17,
                    "product_a_name": "full cream milk",
                    "product_b_name": "fresh onions",
                    "final_score": 99,
                    "purchase_score": 88,
                    "pair_count": 90,
                    "category_a": "dairy",
                    "category_b": "vegetables",
                    "product_family_a": "milk_dairy",
                    "product_family_b": "vegetables",
                }
            ]
        )
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_missing_lane",
            source="manual",
            order_ids=[9901],
            history_product_ids=[11, 17],
            history_items=["full cream milk", "fresh onions"],
            created_at="2026-03-08T00:00:00+00:00",
            history_counts={11: 3, 17: 3},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_missing_lane",
            )
        self.assertEqual(len(recs), 1)
        bundles_out = recs[0].get("bundles", [])
        self.assertLessEqual(len(bundles_out), 1)
        self.assertTrue(all(str(bundle.get("lane", "")).strip().lower() == LANE_NONFOOD for bundle in bundles_out))
        self.assertEqual(set(recs[0].get("missing_food_lanes", [])), {LANE_MEAL, LANE_SNACK, LANE_OCCASION})

    def test_missing_lane_allowed_when_only_weak_visible_candidates_exist(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 3,
                    "product_b": 4,
                    "product_a_name": "potato chips",
                    "product_b_name": "cola soda",
                    "final_score": 62,
                    "purchase_score": 6,
                    "pair_count": 1,
                    "category_a": "snacks",
                    "category_b": "beverages",
                    "product_family_a": "chips",
                    "product_family_b": "beverage_soda",
                }
            ]
        )
        context = _build_context()
        object.__setattr__(context, "neighbors", {})
        profile = PersonProfile(
            profile_id="p_weak_only_missing",
            source="manual",
            order_ids=[9902],
            history_product_ids=[3, 4],
            history_items=["potato chips", "cola soda"],
            created_at="2026-03-08T00:00:00+00:00",
            history_counts={3: 3, 4: 3},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_missing_lane_weak_only",
            )
        self.assertEqual(len(recs), 1)
        self.assertEqual(len(recs[0].get("bundles", [])), 0)
        self.assertEqual(set(recs[0].get("missing_food_lanes", [])), {LANE_MEAL, LANE_SNACK, LANE_OCCASION})

    def test_feedback_override_does_not_bypass_visible_expression_floor(self):
        context = _build_context()
        context.product_name_by_id[121] = "white oats 500g"
        context.product_price_by_id[121] = 5.0
        context.category_by_id[121] = "grains"
        context.product_family_by_id[121] = "oats"

        self.assertFalse(
            _passes_pair_filters(
                anchor=121,
                complement=23,
                history_ids={121, 23},
                context=context,
                lane=LANE_MEAL,
            )
        )
        self.assertFalse(
            _passes_complement_gate(
                anchor=121,
                complement=23,
                context=context,
                cp_score=95.0,
                recipe_compat=0.8,
                prior_bonus=0.0,
                lane=LANE_MEAL,
                pair_count=120,
            )
        )
        feedback_lookup = {
            "pair_boosts": {(23, 121): 0.40},
            "pair_penalties": {},
            "pair_overrides": {(23, 121)},
            "good_pairs": {(23, 121)},
            "bad_pairs": set(),
        }
        with patch("qeu_bundling.presentation.person_predictions.FEEDBACK_LOOKUP", feedback_lookup):
            self.assertFalse(
                _passes_pair_filters(
                    anchor=121,
                    complement=23,
                    history_ids={121, 23},
                    context=context,
                    lane=LANE_MEAL,
                )
            )
            self.assertFalse(
                _passes_complement_gate(
                    anchor=121,
                    complement=23,
                    context=context,
                    cp_score=95.0,
                    recipe_compat=0.8,
                    prior_bonus=0.0,
                    lane=LANE_MEAL,
                    pair_count=120,
                )
            )

    def test_top_bundle_scan_limit_is_lane_specific(self):
        self.assertEqual(_top_bundle_scan_limit(LANE_MEAL), 80)
        self.assertEqual(_top_bundle_scan_limit(LANE_SNACK), 140)
        self.assertEqual(_top_bundle_scan_limit(LANE_OCCASION), 180)

    def test_person_quality_artifact_counts_fallback_and_telemetry(self):
        context = _build_context()
        recommendations = [
            {
                "person_label": "Person 1",
                "bundles": [
                    {
                        "lane": LANE_MEAL,
                        "product_a": 1,
                        "product_b": 2,
                        "anchor_product_id": 1,
                        "complement_product_id": 2,
                        "recommendation_origin": "top_bundle",
                        "anchor_in_history": True,
                        "history_match_count": 2,
                    },
                    {
                        "lane": LANE_OCCASION,
                        "product_a": 5,
                        "product_b": 8,
                        "anchor_product_id": 5,
                        "complement_product_id": 8,
                        "recommendation_origin": "fallback_food",
                        "anchor_in_history": True,
                        "history_match_count": 2,
                    },
                ],
            }
        ]
        telemetry = ServingTelemetry(
            overall={"rejected_score_floor": 3, "rejected_visible_expression": 2, "chose_food_fallback": 1},
            by_lane={LANE_OCCASION: {"rejected_score_floor": 3, "chose_food_fallback": 1}},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            paths = real_get_paths(project_root=base_dir)
            paths.output_dir.mkdir(parents=True, exist_ok=True)
            with patch("qeu_bundling.presentation.person_predictions.get_paths", return_value=paths):
                _write_person_quality_artifact(
                    base_dir,
                    recommendations,
                    context=context,
                    run_id="run-1",
                    serving_telemetry=telemetry,
                )
            payload = json.loads((paths.output_dir / "person_reco_quality.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["top_bundle_count"], 1)
        self.assertEqual(payload["food_fallback_count"], 1)
        self.assertEqual(payload["cleaning_fallback_count"], 0)
        self.assertEqual(payload["template_fallback_count"], 1)
        self.assertEqual(payload["fallback_count"], 1)
        self.assertEqual(payload["fallback_share"], 0.5)
        self.assertEqual(payload["rejected_score_floor"], 3)
        self.assertEqual(payload["serving_telemetry_by_lane"][LANE_OCCASION]["chose_food_fallback"], 1)

    def test_each_profile_gets_exactly_three_two_item_bundles(self):
        bundles = _build_bundles()
        context = _build_context()
        profile = _mixed_profile("p_exact_three")
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_exact_three",
            )
        self.assertEqual(len(recs), 1)
        bundles_out = recs[0].get("bundles", [])
        self.assertEqual(len(bundles_out), 3)
        anchors = [int(b.get("anchor_product_id", -1)) for b in bundles_out]
        self.assertEqual(len(anchors), len(set(anchors)))
        pair_keys = [tuple(sorted((int(b.get("product_a", -1)), int(b.get("product_b", -1))))) for b in bundles_out]
        self.assertEqual(len(pair_keys), len(set(pair_keys)))
        for bundle in bundles_out:
            self.assertGreater(int(bundle.get("product_a", 0)), 0)
            self.assertGreater(int(bundle.get("product_b", 0)), 0)

    def test_curated_fallback_library_has_required_counts(self):
        meal = [item for item in TOP_100_CURATED_FOOD_BUNDLES if str(item.get("lane")) == LANE_MEAL]
        snack = [item for item in TOP_100_CURATED_FOOD_BUNDLES if str(item.get("lane")) == LANE_SNACK]
        occasion = [item for item in TOP_100_CURATED_FOOD_BUNDLES if str(item.get("lane")) == LANE_OCCASION]
        self.assertEqual(len(TOP_100_CURATED_FOOD_BUNDLES), 100)
        self.assertEqual(len(meal), 40)
        self.assertEqual(len(snack), 30)
        self.assertEqual(len(occasion), 30)
        self.assertTrue(CURATED_CLEANING_FALLBACK_BUNDLES)
        required = {"id", "lane", "priority", "anchor_hint", "complement_hint", "source_group"}
        for entry in TOP_100_CURATED_FOOD_BUNDLES:
            self.assertTrue(required.issubset(set(entry.keys())))
            self.assertEqual(str(entry.get("source_group")), "fallback")

    def test_cross_domain_pairs_are_rejected_in_main_and_fallback_paths(self):
        context = _build_context()
        context.product_name_by_id[201] = "heavy meat grinder"
        context.product_price_by_id[201] = 99.0
        context.category_by_id[201] = "appliances"
        context.product_family_by_id[201] = "kitchen_tool"
        self.assertFalse(_passes_pair_filters(2, 103, {2, 103}, context, lane=LANE_MEAL))
        self.assertFalse(_passes_pair_filters(2, 201, {2, 201}, context, lane=LANE_MEAL))

    def test_source_priority_prefers_top_bundle_over_copurchase_and_fallback(self):
        top_candidate = {
            "source": "top_bundle",
            "personal_score": 0.40,
            "lane_fit_score": 0.60,
            "pair_strength": "strong",
            "anchor": 1,
            "complement": 2,
        }
        template_candidate = {
            "source": "fallback:meal:meal_001",
            "personal_score": 0.90,
            "lane_fit_score": 0.95,
            "pair_strength": "strong",
            "anchor": 23,
            "complement": 22,
        }
        copurchase_candidate = {
            "source": "copurchase_fallback",
            "personal_score": 0.99,
            "lane_fit_score": 0.99,
            "pair_strength": "strong",
            "anchor": 5,
            "complement": 8,
        }
        self.assertLess(_candidate_rank_key(top_candidate), _candidate_rank_key(template_candidate))
        self.assertLess(_candidate_rank_key(copurchase_candidate), _candidate_rank_key(template_candidate))

    def test_candidate_first_prefers_copurchase_over_fallback_when_top_missing(self):
        template_candidate = {
            "source": "fallback:occasion:occasion_001",
            "personal_score": 0.45,
            "lane_fit_score": 0.72,
            "pair_strength": "weak_valid",
            "anchor": 5,
            "complement": 8,
        }
        copurchase_candidate = {
            "source": "copurchase_fallback",
            "personal_score": 0.98,
            "lane_fit_score": 0.91,
            "pair_strength": "strong",
            "anchor": 5,
            "complement": 6,
        }
        self.assertLess(_candidate_rank_key(copurchase_candidate), _candidate_rank_key(template_candidate))

    def test_fallback_candidates_require_graph_or_bundle_evidence(self):
        context = _build_context()
        object.__setattr__(context, "neighbors", {})
        fallback = _fallback_candidates_for_lane(
            history_ids={23, 22},
            lane=LANE_MEAL,
            context=context,
            top_bundle_rows_by_anchor={},
            bundle_lookup={},
        )
        self.assertEqual(fallback, [])

    def test_final_food_bundles_do_not_emit_weak_valid_strength(self):
        bundles = _build_bundles()
        context = _build_context()
        profile = _mixed_profile("p_no_weak_final")
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_no_weak_final",
            )
        self.assertEqual(len(recs), 1)
        for bundle in recs[0].get("bundles", []):
            if str(bundle.get("lane", "")).strip().lower() in {LANE_MEAL, LANE_SNACK, LANE_OCCASION}:
                self.assertNotEqual(str(bundle.get("pair_strength", "")), "weak_valid")

    def test_candidate_first_selection_does_not_force_lane_quotas(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 1,
                    "product_b": 2,
                    "product_a_name": "watania chicken breast",
                    "product_b_name": "basmati rice",
                    "final_score": 92,
                    "purchase_score": 66,
                    "pair_count": 80,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "poultry",
                    "product_family_b": "rice_centric",
                },
                {
                    "product_a": 16,
                    "product_b": 2,
                    "product_a_name": "fresh fish fillet",
                    "product_b_name": "basmati rice",
                    "final_score": 90,
                    "purchase_score": 62,
                    "pair_count": 70,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "seafood",
                    "product_family_b": "rice_centric",
                },
                {
                    "product_a": 23,
                    "product_b": 22,
                    "product_a_name": "farm eggs",
                    "product_b_name": "whole wheat tortilla bread",
                    "final_score": 89,
                    "purchase_score": 60,
                    "pair_count": 68,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "eggs",
                    "product_family_b": "bread",
                },
            ]
        )
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_candidate_first_no_quota",
            source="manual",
            order_ids=[8801],
            history_product_ids=[1, 2, 16, 23, 22],
            history_items=["watania chicken breast", "basmati rice", "fresh fish fillet", "farm eggs", "whole wheat tortilla bread"],
            created_at="2026-03-10T00:00:00+00:00",
            history_counts={1: 3, 2: 4, 16: 2, 23: 3, 22: 2},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_candidate_first_no_quota",
            )
        self.assertEqual(len(recs), 1)
        bundles_out = recs[0].get("bundles", [])
        self.assertEqual(len(bundles_out), 3)
        lanes = {str(bundle.get("lane", "")).strip().lower() for bundle in bundles_out}
        self.assertTrue(lanes.issubset({LANE_MEAL, LANE_NONFOOD}))
        self.assertIn(LANE_MEAL, lanes)
        item_ids = [
            int(pid)
            for bundle in bundles_out
            for pid in (int(bundle.get("product_a", -1)), int(bundle.get("product_b", -1)))
            if int(pid) > 0
        ]
        self.assertEqual(len(item_ids), len(set(item_ids)))
        self.assertEqual(set(recs[0].get("missing_food_lanes", [])), {LANE_SNACK, LANE_OCCASION})

    def test_random_profile_resamples_instead_of_weak_fill(self):
        bundles = _build_bundles()
        context = _build_context()
        weak_random = PersonProfile(
            profile_id="p_random_weak",
            source="random",
            order_ids=[9903],
            history_product_ids=[11, 17],
            history_items=["full fat milk", "fresh garlic"],
            created_at="2026-03-08T00:00:00+00:00",
            history_counts={11: 1, 17: 1},
        )
        resampled_random = PersonProfile(
            profile_id="p_random_resampled",
            source="random",
            order_ids=[9904],
            history_product_ids=[1, 2, 3, 4, 5, 8, 9, 10],
            history_items=[
                "watania chicken breast",
                "basmati rice",
                "potato chips",
                "cola soda",
                "black tea",
                "chocolate biscuit",
                "premium dates",
                "fresh cream",
            ],
            created_at="2026-03-08T00:00:00+00:00",
            history_counts={1: 2, 2: 2, 3: 2, 4: 1, 5: 2, 8: 2, 9: 1, 10: 1},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            with patch("qeu_bundling.presentation.person_predictions.build_random_profile", side_effect=[resampled_random, None]):
                recs = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[weak_random],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_random_resample",
                )
        self.assertEqual(len(recs), 1)
        self.assertEqual(str(recs[0].get("profile_id")), "p_random_resampled")
        self.assertEqual(len(recs[0].get("bundles", [])), 3)

    def _run_quality_tightening_case(self) -> list[dict[str, object]]:
        bundles = _build_bundles_quality_tightening()
        context = _build_context()
        context.product_name_by_id.update(
            {
                112: "premium flour",
                113: "chicken stock cubes",
                114: "garlic mayo",
                115: "chicken nuggets",
                116: "caramel dessert pudding",
                117: "peanut butter spread",
                118: "plain tea biscuits",
                119: "nutella chocolate spread",
            }
        )
        context.product_price_by_id.update({112: 3.0, 113: 4.0, 114: 5.0, 115: 12.0, 116: 7.0, 117: 9.0, 118: 4.0, 119: 12.0})
        context.category_by_id.update(
            {
                112: "baking",
                113: "condiments",
                114: "condiments",
                115: "protein",
                116: "desserts",
                117: "spreads",
                118: "snacks",
                119: "spreads",
            }
        )
        context.product_family_by_id.update(
            {
                112: "flour",
                113: "seasoning",
                114: "condiments",
                115: "protein",
                116: "dessert",
                117: "spread",
                118: "biscuit",
                119: "spread",
            }
        )
        profile = PersonProfile(
            profile_id="p_quality_tightening",
            source="manual",
            order_ids=[9001],
            history_product_ids=[1, 2, 3, 4, 9, 10, 7, 11, 12, 13, 19, 24, 112, 113, 114, 115, 116, 117, 118, 119],
            history_items=[
                "watania chicken breast",
                "basmati rice",
                "potato chips",
                "cola soda",
                "premium dates",
                "fresh cream",
                "sunflower cooking oil",
                "full fat milk",
                "tuna chunks",
                "pasta penne",
                "indomie chicken noodles cup",
                "cream cheese spread",
                "premium flour",
                "chicken stock cubes",
                "garlic mayo",
                "chicken nuggets",
                "caramel dessert pudding",
                "peanut butter spread",
                "plain tea biscuits",
                "nutella chocolate spread",
            ],
            created_at="2026-03-12T00:00:00+00:00",
            history_counts={
                1: 2,
                2: 3,
                3: 2,
                4: 2,
                9: 2,
                10: 2,
                7: 2,
                11: 1,
                12: 2,
                13: 1,
                19: 2,
                24: 1,
                112: 1,
                113: 1,
                114: 1,
                115: 1,
                116: 1,
                117: 1,
                118: 1,
                119: 1,
            },
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_quality_tightening_case",
            )
        self.assertEqual(len(recs), 1)
        bundles_out = recs[0].get("bundles", [])
        self.assertEqual(len(bundles_out), 3)
        return bundles_out

    def test_final_unordered_pair_uniqueness_per_person(self):
        bundles_out = self._run_quality_tightening_case()
        unordered_pairs = [
            tuple(sorted((int(bundle.get("product_a", -1)), int(bundle.get("product_b", -1)))))
            for bundle in bundles_out
        ]
        self.assertEqual(len(unordered_pairs), len(set(unordered_pairs)))

    def test_tuna_plus_milk_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("full fat milk + tuna chunks", pair_names)

    def test_tuna_plus_rice_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("basmati rice + tuna chunks", pair_names)

    def test_tuna_plus_pasta_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("pasta penne + tuna chunks", pair_names)

    def test_tuna_plus_peanut_butter_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("peanut butter spread + tuna chunks", pair_names)

    def test_noodles_plus_flour_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("indomie chicken noodles cup + premium flour", pair_names)

    def test_stock_cubes_plus_mayo_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("chicken stock cubes + garlic mayo", pair_names)

    def test_oil_plus_stock_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("chicken stock cubes + sunflower cooking oil", pair_names)

    def test_eggs_plus_tomato_paste_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("pond food: eggs + rivoni tomato paste 8 x 135g", pair_names)

    def test_dessert_plus_plain_tea_biscuit_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("caramel dessert pudding + plain tea biscuits", pair_names)

    def test_nutella_plus_plain_milk_as_occasion_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("full fat milk + nutella chocolate spread", pair_names)

    def test_savory_protein_plus_flour_is_rejected_from_final_output(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 1,
                    "product_b": 2,
                    "product_a_name": "national chilled chicken 100 grams",
                    "product_b_name": "foam premium flour 1kg",
                    "final_score": 96,
                    "purchase_score": 72,
                    "pair_count": 95,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "poultry",
                    "product_family_b": "flour",
                },
                {
                    "product_a": 5,
                    "product_b": 8,
                    "product_a_name": "black tea",
                    "product_b_name": "chocolate biscuit",
                    "final_score": 91,
                    "purchase_score": 60,
                    "pair_count": 78,
                    "category_a": "beverages",
                    "category_b": "snacks",
                    "product_family_a": "tea",
                    "product_family_b": "biscuit",
                },
                {
                    "product_a": 9,
                    "product_b": 10,
                    "product_a_name": "premium dates",
                    "product_b_name": "fresh cream",
                    "final_score": 90,
                    "purchase_score": 58,
                    "pair_count": 74,
                    "category_a": "fruits",
                    "category_b": "dairy",
                    "product_family_a": "dates_family",
                    "product_family_b": "dairy",
                },
                {
                    "product_a": 23,
                    "product_b": 22,
                    "product_a_name": "farm eggs",
                    "product_b_name": "whole wheat tortilla bread",
                    "final_score": 89,
                    "purchase_score": 55,
                    "pair_count": 70,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "eggs",
                    "product_family_b": "bread",
                },
            ]
        )
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_savory_flour_reject",
            source="manual",
            order_ids=[9701],
            history_product_ids=[1, 2, 5, 8, 9, 10, 23, 22],
            history_items=[
                "national chilled chicken 100 grams",
                "foam premium flour 1kg",
                "black tea",
                "chocolate biscuit",
                "premium dates",
                "fresh cream",
                "farm eggs",
                "whole wheat tortilla bread",
            ],
            created_at="2026-03-12T00:00:00+00:00",
            history_counts={1: 6, 2: 6, 5: 3, 8: 3, 9: 2, 10: 2, 23: 2, 22: 2},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_savory_flour_reject",
            )
        pair_names = {
            " + ".join(
                sorted(
                    [
                        str(bundle.get("product_a_name", "")).lower(),
                        str(bundle.get("product_b_name", "")).lower(),
                    ]
                )
            )
            for bundle in recs[0].get("bundles", [])
        }
        self.assertNotIn("foam premium flour 1kg + national chilled chicken 100 grams", pair_names)

    def test_oil_plus_nuggets_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("chicken nuggets + sunflower cooking oil", pair_names)

    def test_cream_cheese_plus_dessert_duplication_is_rejected_from_final_output(self):
        bundles_out = self._run_quality_tightening_case()
        pair_names = {" + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()])) for bundle in bundles_out}
        self.assertNotIn("caramel dessert pudding + cream cheese spread", pair_names)

    def test_human_preference_boost_prioritizes_chicken_bread_over_chicken_tomato_paste(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 1,
                    "product_b": 22,
                    "product_a_name": "watania chicken breast",
                    "product_b_name": "whole wheat tortilla bread",
                    "final_score": 82,
                    "purchase_score": 40,
                    "pair_count": 20,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "poultry",
                    "product_family_b": "bread_carb",
                },
                {
                    "product_a": 1,
                    "product_b": 14,
                    "product_a_name": "watania chicken breast",
                    "product_b_name": "tomato paste",
                    "final_score": 86,
                    "purchase_score": 42,
                    "pair_count": 24,
                    "category_a": "protein",
                    "category_b": "condiments",
                    "product_family_a": "poultry",
                    "product_family_b": "sauce",
                },
                {
                    "product_a": 3,
                    "product_b": 4,
                    "product_a_name": "potato chips",
                    "product_b_name": "cola soda",
                    "final_score": 80,
                    "purchase_score": 38,
                    "pair_count": 18,
                    "category_a": "snacks",
                    "category_b": "beverages",
                    "product_family_a": "chips",
                    "product_family_b": "beverage_soda",
                },
                {
                    "product_a": 9,
                    "product_b": 10,
                    "product_a_name": "premium dates",
                    "product_b_name": "fresh cream",
                    "final_score": 79,
                    "purchase_score": 35,
                    "pair_count": 16,
                    "category_a": "fruits",
                    "category_b": "dairy",
                    "product_family_a": "dates_family",
                    "product_family_b": "dairy",
                },
            ]
        )
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_human_preference_boost",
            source="manual",
            order_ids=[9201],
            history_product_ids=[1, 22, 14, 3, 4, 9, 10],
            history_items=[
                "watania chicken breast",
                "whole wheat tortilla bread",
                "tomato paste",
                "potato chips",
                "cola soda",
                "premium dates",
                "fresh cream",
            ],
            created_at="2026-03-12T00:00:00+00:00",
            history_counts={1: 2, 22: 2, 14: 2, 3: 2, 4: 2, 9: 1, 10: 1},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_human_preference_boost",
            )
        self.assertEqual(len(recs), 1)
        bundles_out = recs[0].get("bundles", [])
        self.assertEqual(len(bundles_out), 3)
        pair_names = {
            " + ".join(sorted([str(bundle.get("product_a_name", "")).lower(), str(bundle.get("product_b_name", "")).lower()]))
            for bundle in bundles_out
        }
        tortilla_key = " + ".join(sorted(["whole wheat tortilla bread", "watania chicken breast"]))
        tomato_key = " + ".join(sorted(["tomato paste", "watania chicken breast"]))
        self.assertIn(tortilla_key, pair_names)
        self.assertNotIn(tomato_key, pair_names)
        chicken_pair_count = sum(
            1
            for bundle in bundles_out
            if "watania chicken breast"
            in {
                str(bundle.get("product_a_name", "")).lower(),
                str(bundle.get("product_b_name", "")).lower(),
            }
        )
        self.assertEqual(chicken_pair_count, 1)

    def test_fallback_motif_penalty_reduces_repetition(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 1,
                    "product_b": 2,
                    "product_a_name": "watania chicken breast",
                    "product_b_name": "basmati rice",
                    "final_score": 88,
                    "purchase_score": 58,
                    "pair_count": 70,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "poultry",
                    "product_family_b": "rice_centric",
                }
            ]
        )
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_fallback_motif_reduce",
            source="manual",
            order_ids=[9010],
            history_product_ids=[1, 2, 5, 9, 26],
            history_items=["watania chicken breast", "basmati rice", "black tea", "premium dates", "orange juice"],
            created_at="2026-03-12T00:00:00+00:00",
            history_counts={1: 2, 2: 2, 5: 3, 9: 2, 26: 1},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_fallback_motif_reduce",
            )
        bundles_out = recs[0].get("bundles", [])
        self.assertEqual(len(bundles_out), 3)

        def _motif_key(bundle: dict[str, object]) -> str:
            pair_text = " ".join(
                sorted(
                    [
                        str(bundle.get("product_a_name", "")).lower(),
                        str(bundle.get("product_b_name", "")).lower(),
                    ]
                )
            )
            if "tea" in pair_text and "evaporated milk" in pair_text:
                return "tea_evap"
            if "coffee" in pair_text and "evaporated milk" in pair_text:
                return "coffee_evap"
            if "dates" in pair_text and "evaporated milk" in pair_text:
                return "dates_evap"
            if "dates" in pair_text and "condensed milk" in pair_text:
                return "dates_condensed"
            return ""

        motif_counts: dict[str, int] = {}
        for bundle in bundles_out:
            key = _motif_key(bundle)
            if key:
                motif_counts[key] = int(motif_counts.get(key, 0)) + 1
        self.assertTrue(all(count <= 1 for count in motif_counts.values()))

    def test_all_profiles_still_get_exactly_three_after_quality_tightening(self):
        bundles = _build_bundles()
        context = _build_context()
        profiles = [_mixed_profile(f"p_quality_tight_{idx}") for idx in range(4)]
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=profiles,
                max_people=4,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_quality_tightening_exact_three",
            )
        self.assertEqual(len(recs), 4)
        for rec in recs:
            self.assertEqual(len(rec.get("bundles", [])), 3)

    def test_cleaning_bundle_capped_at_one_and_only_when_needed(self):
        bundles = _build_bundles()
        context = _build_context()
        profile = _mixed_profile("p_cleaning_cap")
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            with patch("qeu_bundling.presentation.person_predictions.NONFOOD_INCLUDE_RATE", 1.0):
                recs = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_cleaning_cap",
                )
        nonfood = [b for b in recs[0].get("bundles", []) if str(b.get("lane", "")).strip().lower() == LANE_NONFOOD]
        self.assertLessEqual(len(nonfood), 1)

    def test_curated_fallback_library_fills_to_exact_three_deterministically(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 1,
                    "product_b": 2,
                    "product_a_name": "watania chicken breast",
                    "product_b_name": "basmati rice",
                    "final_score": 82,
                    "purchase_score": 44,
                    "pair_count": 18,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "poultry",
                    "product_family_b": "rice_centric",
                }
            ]
        )
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_fallback_exact_three",
            source="manual",
            order_ids=[7101],
            history_product_ids=[23, 22, 5, 8, 9, 10, 103, 104],
            history_items=["farm eggs", "whole wheat tortilla bread", "black tea", "chocolate biscuit", "premium dates", "fresh cream", "laundry detergent", "fabric softener"],
            created_at="2026-03-10T00:00:00+00:00",
            history_counts={23: 3, 22: 2, 5: 5, 8: 4, 9: 2, 10: 2, 103: 1, 104: 1},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs_a = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_fallback_exact_three",
            )
            recs_b = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_fallback_exact_three",
            )
        self.assertEqual(len(recs_a[0]["bundles"]), 3)
        self.assertEqual(
            [(int(b["anchor_product_id"]), int(b["complement_product_id"])) for b in recs_a[0]["bundles"]],
            [(int(b["anchor_product_id"]), int(b["complement_product_id"])) for b in recs_b[0]["bundles"]],
        )

    def test_recent_intent_beats_generic_bundle_when_quality_is_close(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 2,
                    "product_b": 1,
                    "product_a_name": "basmati rice",
                    "product_b_name": "watania chicken breast",
                    "final_score": 88,
                    "purchase_score": 50,
                    "pair_count": 40,
                    "category_a": "grains",
                    "category_b": "protein",
                    "product_family_a": "rice_centric",
                    "product_family_b": "poultry",
                },
                {
                    "product_a": 5,
                    "product_b": 8,
                    "product_a_name": "black tea",
                    "product_b_name": "chocolate biscuit",
                    "final_score": 86,
                    "purchase_score": 48,
                    "pair_count": 42,
                    "category_a": "beverages",
                    "category_b": "snacks",
                    "product_family_a": "tea",
                    "product_family_b": "biscuit",
                },
                {
                    "product_a": 9,
                    "product_b": 10,
                    "product_a_name": "premium dates",
                    "product_b_name": "fresh cream",
                    "final_score": 84,
                    "purchase_score": 37,
                    "pair_count": 30,
                    "category_a": "fruits",
                    "category_b": "dairy",
                    "product_family_a": "dates_family",
                    "product_family_b": "dairy",
                },
            ]
        )
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_recent_intent",
            source="manual",
            order_ids=[8101],
            history_product_ids=[1, 2, 5, 8, 5, 8],
            history_items=["watania chicken breast", "basmati rice", "black tea", "chocolate biscuit", "black tea", "chocolate biscuit"],
            created_at="2026-03-10T00:00:00+00:00",
            history_counts={1: 5, 2: 5, 5: 2, 8: 2},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_recent_intent",
            )
        first_pair = {int(recs[0]["bundles"][0]["product_a"]), int(recs[0]["bundles"][0]["product_b"])}
        self.assertEqual(first_pair, {1, 2})

    def test_exposure_penalty_reduces_batch_pair_dominance_deterministically(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 5,
                    "product_b": 8,
                    "product_a_name": "black tea",
                    "product_b_name": "chocolate biscuit",
                    "final_score": 88,
                    "purchase_score": 52,
                    "pair_count": 44,
                    "category_a": "beverages",
                    "category_b": "snacks",
                    "product_family_a": "tea",
                    "product_family_b": "biscuit",
                },
                {
                    "product_a": 9,
                    "product_b": 10,
                    "product_a_name": "premium dates",
                    "product_b_name": "fresh cream",
                    "final_score": 87,
                    "purchase_score": 50,
                    "pair_count": 42,
                    "category_a": "fruits",
                    "category_b": "dairy",
                    "product_family_a": "dates_family",
                    "product_family_b": "dairy",
                },
                {
                    "product_a": 23,
                    "product_b": 22,
                    "product_a_name": "farm eggs",
                    "product_b_name": "whole wheat tortilla bread",
                    "final_score": 86,
                    "purchase_score": 45,
                    "pair_count": 36,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "eggs",
                    "product_family_b": "bread",
                },
            ]
        )
        context = _build_context()
        profiles = [_mixed_profile(f"p_exposure_{idx}") for idx in range(5)]
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=profiles,
                max_people=5,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_exposure_penalty",
            )
        self.assertTrue(all(len(rec.get("bundles", [])) == 3 for rec in recs))
        motifs: list[tuple[int, int]] = []
        for rec in recs:
            for bundle in rec.get("bundles", []):
                motifs.append(tuple(sorted((int(bundle["product_a"]), int(bundle["product_b"])))))
        self.assertGreater(len(set(motifs)), 1)

    def test_repeated_exact_pair_exposure_reduces_across_people(self):
        bundles = _build_bundles_diversity_pressure()
        context = _build_context()
        profiles = [
            PersonProfile(
                profile_id=f"p_exact_repeat_{idx}",
                source="manual",
                order_ids=[9100 + idx],
                history_product_ids=[1, 2, 3, 5, 8, 9, 10, 16, 22, 23],
                history_items=[
                    "watania chicken breast",
                    "basmati rice",
                    "potato chips",
                    "black tea",
                    "chocolate biscuit",
                    "premium dates",
                    "fresh cream",
                    "fresh fish fillet",
                    "whole wheat tortilla bread",
                    "farm eggs",
                ],
                created_at="2026-03-11T00:00:00+00:00",
                history_counts={1: 4, 2: 4, 3: 3, 5: 3, 8: 3, 9: 2, 10: 2, 16: 2, 22: 2, 23: 2},
            )
            for idx in range(6)
        ]

        def _run(disable_diversity: bool) -> list[dict[str, object]]:
            with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
                if not disable_diversity:
                    return build_recommendations_for_profiles(
                        bundles_df=bundles,
                        profiles=profiles,
                        max_people=len(profiles),
                        row_to_record=lambda row: row.to_dict(),
                        run_id="run_exact_repeat_enabled",
                    )
                with patch("qeu_bundling.presentation.person_predictions.EXPOSURE_PAIR_PENALTY", 0.0), patch(
                    "qeu_bundling.presentation.person_predictions.EXPOSURE_TEMPLATE_SIGNATURE_PENALTY", 0.0
                ), patch("qeu_bundling.presentation.person_predictions.EXPOSURE_MOTIF_FAMILY_PENALTY", 0.0), patch(
                    "qeu_bundling.presentation.person_predictions.EXPOSURE_FAMILY_PATTERN_PENALTY", 0.0
                ), patch("qeu_bundling.presentation.person_predictions.EXPOSURE_BUNDLE_SHAPE_PENALTY", 0.0), patch(
                    "qeu_bundling.presentation.person_predictions.EXPOSURE_FALLBACK_MOTIF_PENALTY", 0.0
                ), patch("qeu_bundling.presentation.person_predictions.RARER_STRONG_ALTERNATIVE_BONUS_STEP", 0.0):
                    return build_recommendations_for_profiles(
                        bundles_df=bundles,
                        profiles=profiles,
                        max_people=len(profiles),
                        row_to_record=lambda row: row.to_dict(),
                        run_id="run_exact_repeat_disabled",
                    )

        recs_enabled = _run(disable_diversity=False)
        recs_disabled = _run(disable_diversity=True)

        def _repeated_exact_pairs(recs: list[dict[str, object]]) -> int:
            pair_counter: Counter[tuple[int, int]] = Counter()
            for rec in recs:
                for bundle in rec.get("bundles", []):
                    pair_key = tuple(sorted((int(bundle["product_a"]), int(bundle["product_b"]))))
                    pair_counter[pair_key] += 1
            return int(sum(max(0, count - 1) for count in pair_counter.values()))

        self.assertLessEqual(_repeated_exact_pairs(recs_enabled), _repeated_exact_pairs(recs_disabled))

    def test_shopper_visible_family_mapping_examples(self):
        context = _build_context()
        names = dict(context.product_name_by_id)
        categories = dict(context.category_by_id)
        families = dict(context.product_family_by_id)
        prices = dict(context.product_price_by_id)
        names.update(
            {
                301: "americana minced lamb 400g",
                302: "boni condensed milk",
                303: "mcvities tea biscuits",
                304: "tea milk evaporated milk",
                305: "fresh labneh",
            }
        )
        categories.update(
            {
                301: "protein",
                302: "dairy",
                303: "snacks",
                304: "dairy",
                305: "dairy",
            }
        )
        families.update(
            {
                301: "meat",
                302: "milk",
                303: "biscuit",
                304: "milk",
                305: "labneh",
            }
        )
        prices.update({301: 18.0, 302: 7.0, 303: 4.0, 304: 4.5, 305: 9.0})
        extended_context = PersonalizationContext(
            product_name_by_id=names,
            product_price_by_id=prices,
            product_picture_by_id=dict(context.product_picture_by_id),
            neighbors=dict(context.neighbors),
            recipe_score_by_id=dict(context.recipe_score_by_id),
            ingredient_by_id=dict(context.ingredient_by_id),
            ingredient_recipe_lookup=dict(context.ingredient_recipe_lookup),
            product_family_by_id=families,
            category_by_id=categories,
            product_brand_by_id=dict(context.product_brand_by_id),
            non_food_ids=frozenset(context.non_food_ids),
        )
        cases = [
            (
                {"anchor": 1, "complement": 22, "lane": LANE_MEAL, "pair_relation": semantics.REL_EAT},
                "meal:chicken_wrap_meal",
            ),
            (
                {"anchor": 2, "complement": 301, "lane": LANE_MEAL, "pair_relation": semantics.REL_EAT},
                "meal:rice_meat_meal",
            ),
            (
                {"anchor": 9, "complement": 302, "lane": LANE_OCCASION, "pair_relation": semantics.REL_DESSERT},
                "occasion:dates_milk_treat",
            ),
            (
                {"anchor": 303, "complement": 304, "lane": LANE_SNACK, "pair_relation": semantics.REL_DRINK},
                "snack:biscuit_milk_tea_snack",
            ),
            (
                {"anchor": 1, "complement": 7, "lane": LANE_MEAL, "pair_relation": semantics.REL_COOK},
                "meal:chicken_fat_utilitarian",
            ),
            (
                {"anchor": 23, "complement": 7, "lane": LANE_MEAL, "pair_relation": semantics.REL_COOK},
                "meal:egg_fat_utilitarian",
            ),
            (
                {"anchor": 305, "complement": 3, "lane": LANE_SNACK, "pair_relation": semantics.REL_EAT},
                "snack:labneh_crunchy_snack",
            ),
        ]
        for candidate, expected in cases:
            got = _candidate_motif_family_signature(candidate, extended_context)
            self.assertEqual(got, expected)

    def test_repeated_motif_family_exposure_reduces_similar_bundle_shapes(self):
        context = _build_context()
        overused = {
            "anchor": 1,
            "complement": 2,
            "lane": LANE_MEAL,
            "pool_score": 2.35,
            "personal_score": 2.35,
            "source": "top_bundle",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_EAT,
            "theme": "rice_protein",
        }
        fresh = {
            "anchor": 23,
            "complement": 22,
            "lane": LANE_MEAL,
            "pool_score": 2.28,
            "personal_score": 2.28,
            "source": "top_bundle",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_EAT,
            "theme": "chicken_bread_tortilla_family",
        }

        overused_template = _template_signature(overused, context)
        overused_motif = _candidate_motif_family_signature(overused, context)
        overused_family_pattern = _candidate_family_pattern_signature(overused, context)
        overused_shape = _candidate_bundle_shape_signature(overused, context)

        score_overused = _candidate_effective_score(
            overused,
            context,
            global_pair_exposure={tuple(sorted((1, 2))): 4},
            global_template_exposure={overused_template: 4},
            global_motif_family_exposure={overused_motif: 4},
            global_family_pattern_exposure={overused_family_pattern: 4},
            global_bundle_shape_exposure={overused_shape: 4},
        )
        score_fresh = _candidate_effective_score(
            fresh,
            context,
            global_pair_exposure={},
            global_template_exposure={},
            global_motif_family_exposure={},
            global_family_pattern_exposure={},
            global_bundle_shape_exposure={},
        )
        self.assertGreater(score_fresh, score_overused)

    def test_dominant_meal_family_saturates_after_two_uses(self):
        context = _build_context()
        dominant = {
            "anchor": 1,
            "complement": 2,
            "lane": LANE_MEAL,
            "pool_score": 2.28,
            "personal_score": 2.28,
            "source": "top_bundle",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_EAT,
            "theme": "rice_protein",
        }
        alternative = {
            "anchor": 1,
            "complement": 22,
            "lane": LANE_MEAL,
            "pool_score": 2.19,
            "personal_score": 2.19,
            "source": "top_bundle",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_EAT,
            "theme": "bread_meal",
        }
        dominant_motif = _candidate_motif_family_signature(dominant, context)
        dominant_family = _candidate_family_pattern_signature(dominant, context)
        dominant_shape = _candidate_bundle_shape_signature(dominant, context)
        dominant_template = _template_signature(dominant, context)
        score_dominant = _candidate_effective_score(
            dominant,
            context,
            global_pair_exposure={tuple(sorted((1, 2))): 2},
            global_template_exposure={dominant_template: 2},
            global_motif_family_exposure={dominant_motif: 3},
            global_family_pattern_exposure={dominant_family: 3},
            global_bundle_shape_exposure={dominant_shape: 3},
        )
        score_alternative = _candidate_effective_score(
            alternative,
            context,
            global_pair_exposure={},
            global_template_exposure={},
            global_motif_family_exposure={},
            global_family_pattern_exposure={},
            global_bundle_shape_exposure={},
        )
        self.assertGreater(score_alternative, score_dominant)

    def test_rarer_motif_family_beats_overused_meal_family_when_close(self):
        context = _build_context()
        overused_meal = {
            "anchor": 1,
            "complement": 2,
            "lane": LANE_MEAL,
            "pool_score": 2.24,
            "personal_score": 2.24,
            "source": "top_bundle",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_EAT,
            "theme": "rice_protein",
        }
        rarer_meal = {
            "anchor": 23,
            "complement": 22,
            "lane": LANE_MEAL,
            "pool_score": 2.20,
            "personal_score": 2.20,
            "source": "top_bundle",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_EAT,
            "theme": "bread_meal",
        }
        overused_sig = _candidate_motif_family_signature(overused_meal, context)
        overused_family = _candidate_family_pattern_signature(overused_meal, context)
        overused_shape = _candidate_bundle_shape_signature(overused_meal, context)
        overused_template = _template_signature(overused_meal, context)
        score_overused = _candidate_effective_score(
            overused_meal,
            context,
            global_pair_exposure={tuple(sorted((1, 2))): 2},
            global_template_exposure={overused_template: 2},
            global_motif_family_exposure={overused_sig: 3},
            global_family_pattern_exposure={overused_family: 3},
            global_bundle_shape_exposure={overused_shape: 3},
        )
        score_rarer = _candidate_effective_score(
            rarer_meal,
            context,
            global_pair_exposure={},
            global_template_exposure={},
            global_motif_family_exposure={},
            global_family_pattern_exposure={},
            global_bundle_shape_exposure={},
        )
        self.assertGreater(score_rarer, score_overused)

    def test_utilitarian_oil_plus_protein_is_strongly_downranked(self):
        context = _build_context()
        utilitarian = {
            "anchor": 7,
            "complement": 1,
            "lane": LANE_MEAL,
            "pool_score": 2.20,
            "personal_score": 2.20,
            "source": "copurchase_fallback",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_EAT,
            "theme": "meal_generic",
        }
        attractive = {
            "anchor": 1,
            "complement": 22,
            "lane": LANE_MEAL,
            "pool_score": 2.16,
            "personal_score": 2.16,
            "source": "copurchase_fallback",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_EAT,
            "theme": "bread_meal",
        }
        score_utilitarian = _candidate_effective_score(
            utilitarian,
            context,
            global_pair_exposure={},
            global_template_exposure={},
            global_motif_family_exposure={},
            global_family_pattern_exposure={},
            global_bundle_shape_exposure={},
        )
        score_attractive = _candidate_effective_score(
            attractive,
            context,
            global_pair_exposure={},
            global_template_exposure={},
            global_motif_family_exposure={},
            global_family_pattern_exposure={},
            global_bundle_shape_exposure={},
        )
        self.assertGreater(score_attractive, score_utilitarian)

    def test_meal_dominant_family_saturation_penalty_is_stronger_than_non_dominant(self):
        context = _build_context()
        dominant = {
            "anchor": 1,
            "complement": 2,
            "lane": LANE_MEAL,
            "pool_score": 2.36,
            "personal_score": 2.36,
            "source": "top_bundle",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_EAT,
            "theme": "rice_protein",
        }
        non_dominant = {
            "anchor": 23,
            "complement": 22,
            "lane": LANE_MEAL,
            "pool_score": 2.36,
            "personal_score": 2.36,
            "source": "top_bundle",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_EAT,
            "theme": "egg_breakfast_meal",
        }
        self.assertTrue(_is_meal_dominant_motif_signature(_candidate_motif_family_signature(dominant, context)))
        self.assertFalse(_is_meal_dominant_motif_signature(_candidate_motif_family_signature(non_dominant, context)))

        dom_template = _template_signature(dominant, context)
        dom_motif = _candidate_motif_family_signature(dominant, context)
        dom_family = _candidate_family_pattern_signature(dominant, context)
        dom_shape = _candidate_bundle_shape_signature(dominant, context)
        non_template = _template_signature(non_dominant, context)
        non_motif = _candidate_motif_family_signature(non_dominant, context)
        non_family = _candidate_family_pattern_signature(non_dominant, context)
        non_shape = _candidate_bundle_shape_signature(non_dominant, context)

        score_dominant = _candidate_effective_score(
            dominant,
            context,
            global_pair_exposure={tuple(sorted((1, 2))): 4},
            global_template_exposure={dom_template: 4},
            global_motif_family_exposure={dom_motif: 4},
            global_family_pattern_exposure={dom_family: 4},
            global_bundle_shape_exposure={dom_shape: 4},
        )
        score_non_dominant = _candidate_effective_score(
            non_dominant,
            context,
            global_pair_exposure={tuple(sorted((23, 22))): 4},
            global_template_exposure={non_template: 4},
            global_motif_family_exposure={non_motif: 4},
            global_family_pattern_exposure={non_family: 4},
            global_bundle_shape_exposure={non_shape: 4},
        )
        self.assertGreater(score_non_dominant, score_dominant)

    def test_repeated_rice_meat_meal_domination_is_reduced_across_batch(self):
        bundles = _build_bundles_diversity_pressure()
        context = _build_context()
        profiles = [
            PersonProfile(
                profile_id=f"p_meal_domination_{idx}",
                source="manual",
                order_ids=[9400 + idx],
                history_product_ids=[1, 2, 16, 23, 22, 5, 8, 9, 10],
                history_items=[
                    "watania chicken breast",
                    "basmati rice",
                    "fresh fish fillet",
                    "farm eggs",
                    "whole wheat tortilla bread",
                    "black tea",
                    "chocolate biscuit",
                    "premium dates",
                    "fresh cream",
                ],
                created_at="2026-03-12T00:00:00+00:00",
                history_counts={1: 5, 2: 5, 16: 4, 23: 3, 22: 3, 5: 2, 8: 2, 9: 1, 10: 1},
            )
            for idx in range(7)
        ]

        def _run(disable_meal_dominance_pressure: bool) -> list[dict[str, object]]:
            with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
                if not disable_meal_dominance_pressure:
                    return build_recommendations_for_profiles(
                        bundles_df=bundles,
                        profiles=profiles,
                        max_people=len(profiles),
                        row_to_record=lambda row: row.to_dict(),
                        run_id="run_meal_domination_pressure_on",
                    )
                with patch("qeu_bundling.presentation.person_predictions.EXPOSURE_MEAL_DOMINANT_MOTIF_PENALTY", 0.0), patch.dict(
                    "qeu_bundling.presentation.person_predictions.HUMAN_SOFT_PENALTY_BY_PATTERN",
                    {"meal_rice_meat_overused": 0.0},
                    clear=False,
                ), patch("qeu_bundling.presentation.person_predictions.EXPOSURE_MOTIF_FAMILY_PENALTY", 0.0), patch(
                    "qeu_bundling.presentation.person_predictions.EXPOSURE_FAMILY_PATTERN_PENALTY", 0.0
                ), patch("qeu_bundling.presentation.person_predictions.EXPOSURE_BUNDLE_SHAPE_PENALTY", 0.0), patch(
                    "qeu_bundling.presentation.person_predictions.EXPOSURE_MEAL_FAMILY_SATURATION_PENALTY", 0.0
                ), patch("qeu_bundling.presentation.person_predictions.EXPOSURE_MEAL_PATTERN_SATURATION_PENALTY", 0.0), patch(
                    "qeu_bundling.presentation.person_predictions.EXPOSURE_MEAL_SHAPE_SATURATION_PENALTY", 0.0
                ):
                    return build_recommendations_for_profiles(
                        bundles_df=bundles,
                        profiles=profiles,
                        max_people=len(profiles),
                        row_to_record=lambda row: row.to_dict(),
                        run_id="run_meal_domination_pressure_off",
                    )

        def _dominant_meal_count(recs: list[dict[str, object]]) -> int:
            total = 0
            for rec in recs:
                for bundle in rec.get("bundles", []):
                    lane = str(bundle.get("lane", "")).strip().lower()
                    if lane != LANE_MEAL:
                        continue
                    text = " ".join(
                        sorted(
                            [
                                str(bundle.get("product_a_name", "")).lower(),
                                str(bundle.get("product_b_name", "")).lower(),
                            ]
                        )
                    )
                    rice_meat = ("rice" in text) and any(token in text for token in ("chicken", "beef", "lamb", "meat", "fish"))
                    chicken_base = ("chicken" in text) and ("tomato paste" in text or "oil" in text)
                    if rice_meat or chicken_base:
                        total += 1
            return total

        def _repeated_meal_motif_family_count(recs: list[dict[str, object]]) -> int:
            motif_counter: Counter[str] = Counter()
            for rec in recs:
                for bundle in rec.get("bundles", []):
                    lane = str(bundle.get("lane", "")).strip().lower()
                    if lane != LANE_MEAL:
                        continue
                    candidate = {
                        "anchor": int(bundle.get("product_a", -1)),
                        "complement": int(bundle.get("product_b", -1)),
                        "lane": lane,
                        "theme": str(bundle.get("bundle_theme", "")),
                        "pair_relation": str(bundle.get("pair_relation", "")),
                    }
                    motif = _candidate_motif_family_signature(candidate, context)
                    motif_counter[motif] += 1
            return int(sum(max(0, count - 1) for count in motif_counter.values()))

        recs_pressure_on = _run(disable_meal_dominance_pressure=False)
        recs_pressure_off = _run(disable_meal_dominance_pressure=True)
        self.assertLessEqual(_dominant_meal_count(recs_pressure_on), _dominant_meal_count(recs_pressure_off))
        self.assertLessEqual(
            _repeated_meal_motif_family_count(recs_pressure_on),
            _repeated_meal_motif_family_count(recs_pressure_off),
        )

    def test_personalization_prefers_person_specific_strong_pair_over_overused_generic(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 1,
                    "product_b": 2,
                    "product_a_name": "watania chicken breast",
                    "product_b_name": "basmati rice",
                    "final_score": 93,
                    "purchase_score": 64,
                    "pair_count": 80,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "poultry",
                    "product_family_b": "rice_centric",
                },
                {
                    "product_a": 23,
                    "product_b": 22,
                    "product_a_name": "farm eggs",
                    "product_b_name": "whole wheat tortilla bread",
                    "final_score": 92,
                    "purchase_score": 63,
                    "pair_count": 78,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "eggs",
                    "product_family_b": "bread",
                },
                {
                    "product_a": 5,
                    "product_b": 8,
                    "product_a_name": "black tea",
                    "product_b_name": "chocolate biscuit",
                    "final_score": 80,
                    "purchase_score": 42,
                    "pair_count": 55,
                    "category_a": "beverages",
                    "category_b": "snacks",
                    "product_family_a": "tea",
                    "product_family_b": "biscuit",
                },
                {
                    "product_a": 9,
                    "product_b": 10,
                    "product_a_name": "premium dates",
                    "product_b_name": "fresh cream",
                    "final_score": 79,
                    "purchase_score": 41,
                    "pair_count": 52,
                    "category_a": "fruits",
                    "category_b": "dairy",
                    "product_family_a": "dates_family",
                    "product_family_b": "dairy",
                },
            ]
        )
        context = _build_context()
        profile_generic = PersonProfile(
            profile_id="p_personal_generic",
            source="manual",
            order_ids=[9301],
            history_product_ids=[1, 2, 5, 8, 9, 10],
            history_items=["watania chicken breast", "basmati rice", "black tea", "chocolate biscuit", "premium dates", "fresh cream"],
            created_at="2026-03-11T00:00:00+00:00",
            history_counts={1: 6, 2: 6, 5: 3, 8: 3, 9: 2, 10: 2},
        )
        profile_specific = PersonProfile(
            profile_id="p_personal_specific",
            source="manual",
            order_ids=[9302],
            history_product_ids=[23, 22, 23, 22, 1, 2, 5, 8, 9, 10],
            history_items=["farm eggs", "whole wheat tortilla bread", "watania chicken breast", "basmati rice", "black tea", "chocolate biscuit", "premium dates", "fresh cream"],
            created_at="2026-03-11T00:00:00+00:00",
            history_counts={23: 8, 22: 8, 1: 1, 2: 1, 5: 2, 8: 2, 9: 1, 10: 1},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[profile_generic, profile_specific],
                max_people=2,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_personal_specific_preference",
            )
        first_generic = {int(recs[0]["bundles"][0]["product_a"]), int(recs[0]["bundles"][0]["product_b"])}
        first_specific = {int(recs[1]["bundles"][0]["product_a"]), int(recs[1]["bundles"][0]["product_b"])}
        self.assertIn(first_generic, ({1, 2}, {5, 8}))
        self.assertEqual(first_specific, {22, 23})

    def test_person_specific_snack_or_occasion_preference_beats_generic_meal_default(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 1,
                    "product_b": 2,
                    "product_a_name": "watania chicken breast",
                    "product_b_name": "basmati rice",
                    "final_score": 94,
                    "purchase_score": 66,
                    "pair_count": 86,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "poultry",
                    "product_family_b": "rice_centric",
                },
                {
                    "product_a": 5,
                    "product_b": 8,
                    "product_a_name": "black tea",
                    "product_b_name": "chocolate biscuit",
                    "final_score": 93,
                    "purchase_score": 65,
                    "pair_count": 84,
                    "category_a": "beverages",
                    "category_b": "snacks",
                    "product_family_a": "tea",
                    "product_family_b": "biscuit",
                },
                {
                    "product_a": 9,
                    "product_b": 10,
                    "product_a_name": "premium dates",
                    "product_b_name": "fresh cream",
                    "final_score": 92,
                    "purchase_score": 63,
                    "pair_count": 80,
                    "category_a": "fruits",
                    "category_b": "dairy",
                    "product_family_a": "dates_family",
                    "product_family_b": "dairy",
                },
                {
                    "product_a": 3,
                    "product_b": 4,
                    "product_a_name": "potato chips",
                    "product_b_name": "cola soda",
                    "final_score": 91,
                    "purchase_score": 62,
                    "pair_count": 78,
                    "category_a": "snacks",
                    "category_b": "beverages",
                    "product_family_a": "chips",
                    "product_family_b": "beverage_soda",
                },
            ]
        )
        context = _build_context()
        snack_profile = PersonProfile(
            profile_id="p_snack_personal_bias",
            source="manual",
            order_ids=[9501],
            history_product_ids=[5, 8, 5, 8, 3, 4, 9, 10, 26, 1, 2],
            history_items=[
                "black tea",
                "chocolate biscuit",
                "potato chips",
                "cola soda",
                "premium dates",
                "fresh cream",
                "orange juice",
                "watania chicken breast",
                "basmati rice",
            ],
            created_at="2026-03-12T00:00:00+00:00",
            history_counts={5: 8, 8: 8, 3: 6, 4: 6, 9: 4, 10: 4, 26: 3, 1: 1, 2: 1},
        )
        meal_profile = PersonProfile(
            profile_id="p_meal_personal_bias",
            source="manual",
            order_ids=[9502],
            history_product_ids=[1, 2, 1, 2, 16, 22, 23, 5, 8],
            history_items=[
                "watania chicken breast",
                "basmati rice",
                "fresh fish fillet",
                "whole wheat tortilla bread",
                "farm eggs",
                "black tea",
                "chocolate biscuit",
            ],
            created_at="2026-03-12T00:00:00+00:00",
            history_counts={1: 8, 2: 8, 16: 3, 22: 2, 23: 2, 5: 1, 8: 1},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            snack_recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[snack_profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_snack_personal_bias",
            )
            meal_recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[meal_profile],
                max_people=1,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_meal_personal_bias",
            )
        snack_first = snack_recs[0]["bundles"][0]
        meal_first = meal_recs[0]["bundles"][0]
        snack_first_pair = {int(snack_first["product_a"]), int(snack_first["product_b"])}
        meal_first_pair = {int(meal_first["product_a"]), int(meal_first["product_b"])}
        self.assertNotEqual(snack_first_pair, {1, 2})
        self.assertIn(str(snack_first.get("lane", "")).strip().lower(), {LANE_SNACK, LANE_OCCASION})
        self.assertEqual(meal_first_pair, {1, 2})

    def test_personalized_wrap_style_can_beat_overused_generic_meal_family(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 1,
                    "product_b": 2,
                    "product_a_name": "watania chicken breast",
                    "product_b_name": "basmati rice",
                    "final_score": 95,
                    "purchase_score": 68,
                    "pair_count": 92,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "poultry",
                    "product_family_b": "rice_centric",
                },
                {
                    "product_a": 1,
                    "product_b": 22,
                    "product_a_name": "watania chicken breast",
                    "product_b_name": "whole wheat tortilla bread",
                    "final_score": 93,
                    "purchase_score": 64,
                    "pair_count": 84,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "poultry",
                    "product_family_b": "bread",
                },
                {
                    "product_a": 5,
                    "product_b": 8,
                    "product_a_name": "black tea",
                    "product_b_name": "chocolate biscuit",
                    "final_score": 90,
                    "purchase_score": 56,
                    "pair_count": 72,
                    "category_a": "beverages",
                    "category_b": "snacks",
                    "product_family_a": "tea",
                    "product_family_b": "biscuit",
                },
                {
                    "product_a": 9,
                    "product_b": 10,
                    "product_a_name": "premium dates",
                    "product_b_name": "fresh cream",
                    "final_score": 89,
                    "purchase_score": 55,
                    "pair_count": 70,
                    "category_a": "fruits",
                    "category_b": "dairy",
                    "product_family_a": "dates_family",
                    "product_family_b": "dairy",
                },
            ]
        )
        context = _build_context()
        generic_profile = PersonProfile(
            profile_id="p_wrap_generic_lead",
            source="manual",
            order_ids=[9601],
            history_product_ids=[1, 2, 1, 2, 16, 5, 8, 9, 10],
            history_items=[
                "watania chicken breast",
                "basmati rice",
                "fresh fish fillet",
                "black tea",
                "chocolate biscuit",
                "premium dates",
                "fresh cream",
            ],
            created_at="2026-03-12T00:00:00+00:00",
            history_counts={1: 8, 2: 8, 16: 4, 5: 2, 8: 2, 9: 1, 10: 1},
        )
        wrap_profile = PersonProfile(
            profile_id="p_wrap_specific_follow",
            source="manual",
            order_ids=[9602],
            history_product_ids=[1, 22, 1, 22, 2, 2, 5, 8, 9, 10],
            history_items=[
                "watania chicken breast",
                "whole wheat tortilla bread",
                "basmati rice",
                "black tea",
                "chocolate biscuit",
                "premium dates",
                "fresh cream",
            ],
            created_at="2026-03-12T00:00:00+00:00",
            history_counts={1: 8, 22: 7, 2: 6, 5: 3, 8: 3, 9: 2, 10: 2},
        )

        def _meal_pair_for_second(recs: list[dict[str, object]]) -> set[int]:
            second = recs[1]
            meal_bundles = [b for b in second.get("bundles", []) if str(b.get("lane", "")).strip().lower() == LANE_MEAL]
            self.assertTrue(meal_bundles)
            return {int(meal_bundles[0]["product_a"]), int(meal_bundles[0]["product_b"])}

        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs_on = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=[generic_profile, wrap_profile],
                max_people=2,
                row_to_record=lambda row: row.to_dict(),
                run_id="run_wrap_personalization_on",
            )
        meal_pair_on = _meal_pair_for_second(recs_on)
        self.assertEqual(meal_pair_on, {1, 22})
        second_pairs = {
            tuple(sorted((int(bundle["product_a"]), int(bundle["product_b"]))))
            for bundle in recs_on[1].get("bundles", [])
        }
        self.assertNotIn((1, 2), second_pairs)

    def test_rarer_strong_alternative_can_beat_overused_safe_default(self):
        context = _build_context()
        overused_safe = {
            "anchor": 5,
            "complement": 8,
            "lane": LANE_SNACK,
            "pool_score": 2.24,
            "personal_score": 2.24,
            "source": "top_bundle",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_DRINK,
            "theme": "tea_snack",
        }
        rarer_strong = {
            "anchor": 3,
            "complement": 4,
            "lane": LANE_SNACK,
            "pool_score": 2.18,
            "personal_score": 2.18,
            "source": "top_bundle",
            "pair_strength": semantics.STRENGTH_STRONG,
            "pair_relation": semantics.REL_DRINK,
            "theme": "drink_snack",
        }
        safe_template = _template_signature(overused_safe, context)
        safe_motif = _candidate_motif_family_signature(overused_safe, context)
        safe_family_pattern = _candidate_family_pattern_signature(overused_safe, context)
        safe_shape = _candidate_bundle_shape_signature(overused_safe, context)

        score_safe = _candidate_effective_score(
            overused_safe,
            context,
            global_pair_exposure={tuple(sorted((5, 8))): 3},
            global_template_exposure={safe_template: 3},
            global_motif_family_exposure={safe_motif: 4},
            global_family_pattern_exposure={safe_family_pattern: 3},
            global_bundle_shape_exposure={safe_shape: 4},
        )
        score_rare = _candidate_effective_score(
            rarer_strong,
            context,
            global_pair_exposure={},
            global_template_exposure={},
            global_motif_family_exposure={},
            global_family_pattern_exposure={},
            global_bundle_shape_exposure={},
        )
        self.assertGreater(score_rare, score_safe)

    def test_diversity_tightening_keeps_exactly_three_bundles_per_profile(self):
        bundles = _build_bundles_diversity_pressure()
        context = _build_context()
        profiles = [_mixed_profile(f"p_three_after_diversity_{idx}") for idx in range(7)]
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=profiles,
                max_people=len(profiles),
                row_to_record=lambda row: row.to_dict(),
                run_id="run_three_after_diversity",
            )
        self.assertTrue(all(len(rec.get("bundles", [])) == 3 for rec in recs))

    def test_diversity_tightening_is_deterministic_for_same_run(self):
        bundles = _build_bundles_diversity_pressure()
        context = _build_context()
        profiles = [_mixed_profile(f"p_deterministic_diversity_{idx}") for idx in range(6)]
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            recs_a = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=profiles,
                max_people=len(profiles),
                row_to_record=lambda row: row.to_dict(),
                run_id="run_diversity_deterministic",
            )
            recs_b = build_recommendations_for_profiles(
                bundles_df=bundles,
                profiles=profiles,
                max_people=len(profiles),
                row_to_record=lambda row: row.to_dict(),
                run_id="run_diversity_deterministic",
            )

        def _pair_layout(recs: list[dict[str, object]]) -> list[list[tuple[int, int]]]:
            out: list[list[tuple[int, int]]] = []
            for rec in recs:
                person_pairs: list[tuple[int, int]] = []
                for bundle in rec.get("bundles", []):
                    pair = tuple(sorted((int(bundle["product_a"]), int(bundle["product_b"]))))
                    person_pairs.append(pair)
                out.append(person_pairs)
            return out

        self.assertEqual(_pair_layout(recs_a), _pair_layout(recs_b))

    def test_bundle_primary_intent_tagging_examples(self):
        context = _build_context()
        cases = [
            ({"anchor": 5, "complement": 11, "lane": LANE_OCCASION}, LANE_OCCASION, INTENT_DRINK_PAIRING),
            ({"anchor": 9, "complement": 10, "lane": LANE_OCCASION}, LANE_OCCASION, INTENT_DESSERT),
            ({"anchor": 1, "complement": 2, "lane": LANE_MEAL}, LANE_MEAL, INTENT_MEAL_BASE),
            ({"anchor": 12, "complement": 2, "lane": LANE_MEAL}, LANE_MEAL, INTENT_STAPLES),
            ({"anchor": 103, "complement": 104, "lane": LANE_NONFOOD}, LANE_NONFOOD, INTENT_CLEANING),
        ]
        for candidate, lane_hint, expected in cases:
            with self.subTest(candidate=candidate, expected=expected):
                self.assertEqual(_bundle_primary_intent(candidate, context, lane_hint=lane_hint), expected)

        cross_domain_candidate = {"anchor": 103, "complement": 2, "lane": LANE_MEAL}
        self.assertNotEqual(_bundle_primary_intent(cross_domain_candidate, context, lane_hint=LANE_MEAL), INTENT_CLEANING)

    def test_user_intent_profile_no_history_uses_priors(self):
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_intent_prior_only",
            source="manual",
            order_ids=[],
            history_product_ids=[],
            history_items=[],
            created_at="2026-03-15T00:00:00+00:00",
            history_counts={},
        )
        profile_weights = _build_user_intent_profile(profile, context)
        prior_weights = _default_intent_profile()
        self.assertAlmostEqual(sum(profile_weights.values()), 1.0, places=6)
        for intent, prior_weight in prior_weights.items():
            self.assertAlmostEqual(float(profile_weights[intent]), float(prior_weight), places=6)

    def test_user_intent_profile_detects_cleaning_preference(self):
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_cleaning_intent",
            source="manual",
            order_ids=[9911],
            history_product_ids=[103, 104, 105, 106, 103, 104],
            history_items=[
                "laundry detergent",
                "fabric softener",
                "premium disinfectant",
                "premium bleach",
                "laundry detergent",
                "fabric softener",
            ],
            created_at="2026-03-15T00:00:00+00:00",
            history_counts={103: 4, 104: 4, 105: 2, 106: 2},
        )
        profile_weights = _build_user_intent_profile(profile, context)
        top_intent, _weight = _top_intent(profile_weights)
        self.assertEqual(top_intent, INTENT_CLEANING)

    def test_intent_profile_boost_is_small_and_capped(self):
        profile_weights = {k: 0.0 for k in _default_intent_profile()}
        profile_weights[INTENT_DRINK_PAIRING] = 1.0
        boost_match = _intent_profile_boost(INTENT_DRINK_PAIRING, profile_weights)
        boost_mismatch = _intent_profile_boost(INTENT_MEAL_BASE, profile_weights)
        self.assertGreater(boost_match, 0.0)
        self.assertLess(boost_mismatch, 0.0)
        self.assertLessEqual(abs(boost_match), float(INTENT_BOOST_MAX))
        self.assertLessEqual(abs(boost_mismatch), float(INTENT_BOOST_MAX))

    def test_intent_diversity_soft_cap_does_not_block_third_bundle_when_only_option(self):
        bundles = pd.DataFrame(
            [
                {
                    "product_a": 1,
                    "product_b": 2,
                    "product_a_name": "watania chicken breast",
                    "product_b_name": "basmati rice",
                    "final_score": 92,
                    "purchase_score": 55,
                    "pair_count": 44,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "poultry",
                    "product_family_b": "rice_centric",
                },
                {
                    "product_a": 16,
                    "product_b": 13,
                    "product_a_name": "fresh fish fillet",
                    "product_b_name": "pasta penne",
                    "final_score": 91,
                    "purchase_score": 54,
                    "pair_count": 42,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "seafood",
                    "product_family_b": "pasta",
                },
                {
                    "product_a": 23,
                    "product_b": 22,
                    "product_a_name": "farm eggs",
                    "product_b_name": "whole wheat tortilla bread",
                    "final_score": 90,
                    "purchase_score": 53,
                    "pair_count": 40,
                    "category_a": "protein",
                    "category_b": "grains",
                    "product_family_a": "eggs",
                    "product_family_b": "bread",
                },
            ]
        )
        context = _build_context()
        profile = PersonProfile(
            profile_id="p_intent_soft_cap_override",
            source="manual",
            order_ids=[9922],
            history_product_ids=[1, 2, 16, 13, 23, 22],
            history_items=[
                "watania chicken breast",
                "basmati rice",
                "fresh fish fillet",
                "pasta penne",
                "farm eggs",
                "whole wheat tortilla bread",
            ],
            created_at="2026-03-15T00:00:00+00:00",
            history_counts={1: 4, 2: 4, 16: 3, 13: 3, 23: 2, 22: 2},
        )
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            with patch("qeu_bundling.presentation.person_predictions.INTENT_DIVERSITY_MAX_PER_PERSON", 1):
                recs = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_intent_soft_cap_override",
                )
        self.assertEqual(len(recs[0].get("bundles", [])), 3)
        selected_intents = [
            _bundle_primary_intent(
                {
                    "anchor": int(bundle["anchor_product_id"]),
                    "complement": int(bundle["complement_product_id"]),
                    "lane": str(bundle.get("lane", "")).strip().lower(),
                },
                context,
                lane_hint=str(bundle.get("lane", "")).strip().lower(),
            )
            for bundle in recs[0].get("bundles", [])
        ]
        intent_counts = Counter(selected_intents)
        self.assertGreaterEqual(max(intent_counts.values()), 2)

    def test_serving_profiling_mode_collects_stage_metrics(self):
        bundles = _build_bundles()
        context = _build_context()
        profile = _mixed_profile("p_serving_profile_metrics")
        with patch("qeu_bundling.presentation.person_predictions.load_personalization_context", return_value=context):
            with patch.dict("os.environ", {"QEU_SERVING_PROFILE": "1"}, clear=False):
                recs = build_recommendations_for_profiles(
                    bundles_df=bundles,
                    profiles=[profile],
                    max_people=1,
                    row_to_record=lambda row: row.to_dict(),
                    run_id="run_serving_profile_metrics",
                )
        self.assertEqual(len(recs), 1)
        metrics = get_last_serving_profile_metrics()
        self.assertTrue(bool(metrics.get("enabled")))
        self.assertGreaterEqual(int(metrics.get("profile_count", 0)), 1)
        stage_totals = metrics.get("stage_totals_sec", {})
        self.assertIn("candidate_generation", stage_totals)
        self.assertIn("selection_fill", stage_totals)
        self.assertIn("profile_total", stage_totals)

    def test_dashboard_layer_does_not_inject_extra_cleaning_bundle(self):
        from qeu_bundling.presentation.app import _apply_cleaning_display_fallback

        recommendations = [
            {
                "person_label": "Person 1",
                "bundles": [
                    {"lane": LANE_MEAL, "product_a": 1, "product_b": 2},
                    {"lane": LANE_SNACK, "product_a": 3, "product_b": 4},
                ],
                "missing_food_lanes": [LANE_OCCASION],
            }
        ]
        result = _apply_cleaning_display_fallback(recommendations)
        self.assertEqual(len(result[0]["bundles"]), 2)


if __name__ == "__main__":
    unittest.main()
