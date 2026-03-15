"""Session-scoped person profile generation and recommendation scoring helpers."""
    
from __future__ import annotations

from dataclasses import dataclass, field                                                                                                                                                                                                                                                                                                                                                                                                                           
from difflib import get_close_matches
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import os
import random
import re
import secrets
import time
from datetime import datetime, timezone

import pandas as pd

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.feedback_memory import build_pair_multiplier_lookup, pair_feedback_multiplier
from qeu_bundling.presentation import bundle_semantics as semantics
from qeu_bundling.review.feedback_loader import build_feedback_lookup, load_bundle_feedback


MAX_NEIGHBORS_PER_ANCHOR = 120
MAX_CANDIDATES_PER_PROFILE = 180
CHEAP_RATIO_MAX = 0.50
MAX_TOP_BUNDLE_CANDIDATES = 80
MAX_TOP_BUNDLE_CANDIDATES_BY_LANE = {"meal": 80, "snack": 140, "occasion": 180, "nonfood": 80}
MAX_COPURCHASE_FALLBACK = 80
MAX_BUNDLES_PER_PERSON = 3
MAX_CLEANING_BUNDLES_PER_PERSON = 1
EXPOSURE_PAIR_PENALTY = 0.30
EXPOSURE_TEMPLATE_SIGNATURE_PENALTY = 0.14
EXPOSURE_FALLBACK_MOTIF_PENALTY = 0.18
EXPOSURE_MOTIF_FAMILY_PENALTY = 0.60
EXPOSURE_FAMILY_PATTERN_PENALTY = 0.36
EXPOSURE_BUNDLE_SHAPE_PENALTY = 0.32
EXPOSURE_MEAL_DOMINANT_MOTIF_PENALTY = 0.78
EXPOSURE_MEAL_FAMILY_SATURATION_THRESHOLD = 2
EXPOSURE_MEAL_FAMILY_SATURATION_PENALTY = 1.35
EXPOSURE_MEAL_PATTERN_SATURATION_PENALTY = 0.84
EXPOSURE_MEAL_SHAPE_SATURATION_PENALTY = 0.62
EXPOSURE_MEAL_DOMINANT_FALLBACK_EXTRA = 0.44
EXPOSURE_SNACK_FAMILY_SATURATION_THRESHOLD = 3
EXPOSURE_SNACK_FAMILY_SATURATION_PENALTY = 0.56
EXPOSURE_SNACK_DOMINANT_FAMILY_EXTRA = 0.34
MEAL_DOMINANT_HARD_DECAY_START = 3
MEAL_DOMINANT_HARD_DECAY_STEP = 0.62
MEAL_CHICKEN_DEFAULT_EXTRA_DECAY = 0.40
MEAL_RICE_MEAT_EXTRA_DECAY = 0.48
EXPOSURE_SURGE_THRESHOLD_PAIR = 1
EXPOSURE_SURGE_THRESHOLD_TEMPLATE = 1
EXPOSURE_SURGE_THRESHOLD_MOTIF = 1
EXPOSURE_SURGE_THRESHOLD_FAMILY = 1
EXPOSURE_SURGE_THRESHOLD_SHAPE = 1
EXPOSURE_SURGE_MULTIPLIER = 0.85
EXPOSURE_SURGE_POWER = 1.35
RARER_STRONG_ALTERNATIVE_BONUS_STEP = 0.12
RARER_STRONG_ALTERNATIVE_BONUS_CAP = 0.24
RARER_STRONG_FAMILY_BONUS_STEP = 0.16
RARER_STRONG_FAMILY_BONUS_CAP = 0.42
NON_MEAL_COVERAGE_MARGIN = 0.56
FAMILY_OVERUSE_RANK_PENALTY_STEP = 0.18
FAMILY_OVERUSE_RANK_PENALTY_CAP = 0.54
FAMILY_RARITY_CLOSE_SCORE_MARGIN = 0.55
FAMILY_RARITY_CLOSE_SCORE_BONUS = 0.24
MEAL_DOMINANT_CLOSE_SCORE_EXTRA_PENALTY = 0.34
FAMILY_CLOSE_SCORE_OVERUSE_THRESHOLD = 2
GENERIC_THEME_PENALTY = 0.08
WEAK_CANDIDATE_PENALTY = 0.18
TOP_TRIO_CANDIDATES_PER_LANE = 18
RECENCY_BOOST_WEIGHT = 0.28
COUNT_BOOST_WEIGHT = 0.08
LANE_INTENT_BOOST_WEIGHT = 0.16
PERSONAL_ANCHOR_HISTORY_BOOST = 0.22
PERSONAL_COMPLEMENT_HISTORY_BOOST = 0.12
PERSONAL_PRODUCT_REPEAT_WEIGHT = 0.16
PERSONAL_FAMILY_AFFINITY_WEIGHT = 0.14
PERSONAL_CATEGORY_AFFINITY_WEIGHT = 0.10
PERSONAL_STRONG_ANCHOR_MATCH_BONUS = 0.08
PERSONAL_ESCAPE_GENERIC_MEAL_PENALTY = 0.16
PERSONAL_ESCAPE_GENERIC_MEAL_AFFINITY_PENALTY = 0.06
PERSONAL_NONMEAL_HISTORY_ESCAPE_BONUS = 0.10
PERSONAL_WRAP_STYLE_HISTORY_BONUS = 0.08
PERSONAL_SNACK_OCCASION_PATTERN_BONUS = 0.06
PERSONAL_SHOPPER_FAMILY_ALIGNMENT_WEIGHT = 0.22
PERSONAL_SHOPPER_FAMILY_ESCAPE_PENALTY = 0.12
NON_FOOD_TAG = "non_food"
LANE_NONFOOD = "nonfood"
STRICT_RECIPE_COMPAT_MIN = 0.16
STRICT_COPURCHASE_MIN = 22.0
STRICT_PAIR_COUNT_MIN = 6
UTILITY_WEAK_CP_MAX = 38.0
UTILITY_WEAK_RECIPE_MAX = 0.22
HISTORY_COMPLEMENT_BOOST = 0.08
BRAND_MATCH_BOOST = 0.07
BRAND_MISMATCH_PENALTY = 0.06
DIVERSITY_MAX_SAME_ANCHOR = 2
DIVERSITY_MAX_SAME_COMPLEMENT_FAMILY = 3
DIVERSITY_MAX_TEMPLATE_SIGNATURE = 2
MAX_DIVERSITY_RETRIES_PER_PROFILE = 8
STAPLE_ANCHOR_PENALTY = 0.20
STAPLE_ANCHOR_PENALTY_STAPLE_HEAVY = 0.60
STAPLE_HEAVY_PROFILE_THRESHOLD = 0.70
MAX_STAPLE_ANCHORS_TOP5 = 1
STAPLE_NAME_HINTS = frozenset({"rice", "oil", "sugar", "salt", "flour", "water"})
LOW_PREMIUM_STAPLES = frozenset({"rice", "oil", "sugar", "salt", "water", "flour"})
MAX_SAME_ANCHOR_PER_PAGE = 2
TOP10_ANCHOR_SIZE = 18
MAX_CURATED_FALLBACK_TEMPLATES_PER_LANE = 40
NONFOOD_INCLUDE_RATE = 0.20
PREMIUM_TOP_N = 5
FALLBACK_MOTIF_REPEAT_CAP_PER_PERSON = 1
FALLBACK_EVAP_MOTIF_CAP_PER_PERSON = 2
FEEDBACK_REVIEW_REL_PATH = Path("review") / "bundle_feedback.csv"
FEEDBACK_STRONG_BOOST = 0.75
FEEDBACK_WEAK_PENALTY = 0.45
FEEDBACK_TRASH_PENALTY = 1.00
SERVING_PROFILING_ENV = "QEU_SERVING_PROFILE"
USE_NEW_BUNDLE_SEMANTICS = str(os.getenv("QEU_USE_NEW_BUNDLE_SEMANTICS", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
STRICT_SEMANTIC_FILTERING = str(os.getenv("QEU_STRICT_SEMANTIC_FILTERING", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
ENABLE_INTERNAL_STAPLES = str(os.getenv("QEU_ENABLE_INTERNAL_STAPLES", "1")).strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
ENABLE_STAPLES_LANE = str(os.getenv("QEU_ENABLE_STAPLES_LANE", "0")).strip().lower() in {"1", "true", "yes", "on"}
SEMANTIC_ENGINE_VERSION = "v2"
LANE_MEAL = "meal"
LANE_SNACK = "snack"
LANE_OCCASION = "occasion"
LANE_ORDER = (LANE_MEAL, LANE_SNACK, LANE_OCCASION)
FOOD_LANE_ORDER = LANE_ORDER
ALL_LANE_ORDER = (LANE_MEAL, LANE_SNACK, LANE_OCCASION, LANE_NONFOOD)
LANE_LABELS = {
    LANE_MEAL: "MEAL",
    LANE_SNACK: "SNACK",
    LANE_OCCASION: "OCCASION",
    LANE_NONFOOD: "NONFOOD",
}
LANE_MEAL_NAME_HINTS = frozenset(
    {
        "chicken",
        "meat",
        "beef",
        "fish",
        "rice",
        "pasta",
        "vegetable",
        "tomato",
        "onion",
        "garlic",
        "oil",
        "sauce",
    }
)
LANE_SNACK_NAME_HINTS = frozenset(
    {
        "chips",
        "soda",
        "cola",
        "juice",
        "biscuit",
        "cookie",
        "candy",
        "chocolate",
        "dessert",
        "wafer",
    }
)
LANE_OCCASION_NAME_HINTS = frozenset(
    {
        "tea",
        "coffee",
        "sugar",
        "dates",
        "cream",
        "qishta",
        "biscuit",
        "cookie",
        "milk",
        "nuts",
    }
)
LANE_MEAL_CATEGORY_HINTS = frozenset({"protein", "vegetables", "grains", "oils", "dairy", "condiments"})
LANE_SNACK_CATEGORY_HINTS = frozenset({"snacks", "desserts"})
LANE_OCCASION_CATEGORY_HINTS = frozenset({"beverages", "tea", "coffee", "dairy"})
LANE_MEAL_FAMILY_HINTS = frozenset(
    {
        "poultry",
        "fish",
        "seafood",
        "meat",
        "rice",
        "grain",
        "oil",
        "sauce",
        "milk_dairy",
    }
)
LANE_SNACK_FAMILY_HINTS = frozenset({"chips", "snack", "chocolate", "dessert", "candy", "biscuit", "cookie"})
LANE_OCCASION_FAMILY_HINTS = frozenset({"dates", "tea", "coffee", "milk", "dairy", "nuts"})
LANE_NONFOOD_NAME_HINTS = frozenset(
    {
        "shampoo",
        "conditioner",
        "soap",
        "body wash",
        "detergent",
        "softener",
        "disinfectant",
        "cleaner",
        "bleach",
        "tissue",
        "paper towel",
        "trash bag",
        "dishwashing",
    }
)
LANE_NONFOOD_CATEGORY_HINTS = frozenset({"non_food", "household", "personal_care", "cleaning"})
LANE_NONFOOD_FAMILY_HINTS = frozenset({"clean", "care", "soap", "detergent", "tissue", "paper", "bag"})
SAVORY_BASE_HINTS = frozenset({"tomato paste", "bouillon", "seasoning", "sauce", "cooking oil", "oil"})
PROCESSED_SWEET_HINTS = frozenset({"jelly", "jam", "candy", "dessert", "chocolate", "biscuit", "cookie", "wafer"})

KNOWN_LANE_PATTERNS = {
    LANE_OCCASION: (
        ("tea", "sugar"),
        ("tea", "biscuit"),
        ("coffee", "biscuit"),
        ("tea", "evaporated milk"),
        ("coffee", "evaporated milk"),
        ("dates", "cream"),
        ("dates", "milk"),
        ("dates", "qishta"),
        ("coffee", "dates"),
        ("dessert", "fresh cream"),
    ),
    LANE_SNACK: (
        ("chips", "soda"),
        ("biscuit", "tea"),
        ("biscuit", "milk"),
        ("cookie", "milk"),
        ("chocolate", "milk"),
        ("wafer", "chocolate"),
        ("chips", "cheese"),
        ("cracker", "cream cheese"),
    ),
    LANE_MEAL: (
        ("rice", "chicken"),
        ("pasta", "sauce"),
        ("fish", "oil"),
        ("rice", "spice"),
        ("rice", "tomato"),
        ("cake", "eggs"),
        ("flour", "yeast"),
        ("flour", "baking powder"),
    ),
    LANE_NONFOOD: (
        ("shampoo", "conditioner"),
        ("detergent", "softener"),
        ("soap", "body wash"),
        ("tissue", "paper towel"),
        ("disinfectant", "bleach"),
        ("dishwashing", "sponge"),
    ),
}
MEAL_CATEGORY_COMPLEMENTS = frozenset(
    {
        ("protein", "grains"),
        ("protein", "vegetables"),
        ("protein", "oils"),
        ("vegetables", "oils"),
        ("grains", "oils"),
        ("dairy", "grains"),
    }
)
SNACK_CATEGORY_COMPLEMENTS = frozenset(
    {
        ("snacks", "beverages"),
        ("desserts", "dairy"),
        ("desserts", "beverages"),
    }
)
OCCASION_CATEGORY_COMPLEMENTS = frozenset(
    {
        ("beverages", "snacks"),
        ("beverages", "dairy"),
        ("beverages", "grains"),
        ("fruits", "dairy"),
    }
)
NONFOOD_CATEGORY_COMPLEMENTS = frozenset(
    {
        ("non_food", "non_food"),
        ("household", "household"),
        ("personal_care", "personal_care"),
        ("cleaning", "cleaning"),
        ("paper_goods", "paper_goods"),
    }
)
UTILITY_TOKENS = frozenset(
    {
        "salt",
        "sugar",
        "water",
        "oil",
        "pepper",
        "spice",
        "powder",
        "seasoning",
        "mint",
        "herb",
        "cumin",
    }
)
UTILITY_CATEGORIES = frozenset({"spices", "condiments", "herbs"})
BRAND_STOPWORDS = frozenset(
    {
        "fresh",
        "premium",
        "classic",
        "large",
        "small",
        "pack",
        "pcs",
        "piece",
        "kg",
        "g",
        "ml",
        "l",
    }
)
_NORMALIZE_TEXT_PATTERN = re.compile(r"[^a-z0-9\u0600-\u06FF]+")
_LAST_SERVING_PROFILE_METRICS: dict[str, object] = {}

KNOWN_COMPLEMENT_PRIORS = {
    "dates": {"qishta", "cream", "gaimar", "قشطة"},
    "tuna": {"mayonnaise", "mayo"},
    "rice": {"chicken", "oil"},
}
BAD_PAIR_PATTERNS = (
    ("rice", "chips"),
    ("tea", "spaghetti"),
    ("tea", "pasta"),
    ("tomato", "soda"),
    ("tomato", "candy"),
    ("tomato", "chips"),
    ("chicken", "peanut butter"),
    ("milk", "soup powder"),
    ("tomato paste", "nuggets"),
    ("burger", "rice pouch"),
    ("burger", "ready rice"),
    ("olive oil", "fish biscuits"),
    ("ketchup", "flour"),
)
FINAL_QUALITY_MILK_BLOCK_HINTS = frozenset(
    {
        "fresh milk",
        "full fat milk",
        "full cream milk",
        "low fat milk",
        "whole milk",
        "milk drink",
        "evaporated milk",
        "flavored milk",
        "flavoured milk",
        "milk",
    }
)
FINAL_QUALITY_MILK_EXCLUSION_HINTS = frozenset(
    {
        "condensed milk",
        "milk powder",
        "baby milk",
        "infant formula",
        "formula",
        "cream cheese",
        "cheese",
        "labneh",
        "yogurt",
        "yoghurt",
        "fresh cream",
        "cooking cream",
    }
)
FINAL_QUALITY_NOODLE_HINTS = frozenset(
    {"noodles", "indomie", "ramen", "instant noodle", "cup noodles", "topokki", "mi sidap", "korean noodle"}
)
FINAL_QUALITY_BAKING_HINTS = frozenset({"flour", "semolina", "cornstarch", "baking powder", "yeast", "baking mix"})
FINAL_QUALITY_STOCK_HINTS = frozenset({"stock cube", "stock cubes", "chicken stock", "bouillon", "broth cube"})
FINAL_QUALITY_MAYO_HINTS = frozenset({"mayo", "mayonnaise", "garlic mayo", "garlic mayonnaise"})
FINAL_QUALITY_PEANUT_BUTTER_HINTS = frozenset({"peanut butter"})
FINAL_QUALITY_NUGGET_HINTS = frozenset({"nuggets", "chicken nuggets"})
FINAL_QUALITY_INSTANT_NOODLE_HINTS = frozenset({"indomie", "instant noodle", "cup noodles", "ramen", "mi sidap"})
FINAL_QUALITY_DESSERT_HINTS = frozenset(
    {"dessert", "pudding", "custard", "caramel", "cake", "brownie", "sweet", "jelly", "mousse"}
)
FINAL_QUALITY_CREAM_CHEESE_HINTS = frozenset({"cream cheese", "cheese spread", "kiri", "puck", "triangle cheese", "labneh"})
FINAL_QUALITY_BISCUIT_PLAIN_HINTS = frozenset({"tea biscuits", "tea biscuit", "marie biscuit", "plain biscuit", "biscuits", "biscuit"})
FINAL_QUALITY_NUTELLA_HINTS = frozenset({"nutella", "hazelnut spread", "chocolate spread"})
FINAL_QUALITY_STOCK_SEASONING_HINTS = frozenset(
    {"stock cube", "stock cubes", "bouillon", "seasoning cube", "seasoning cubes", "chicken stock"}
)
FINAL_QUALITY_SAVORY_READY_HINTS = frozenset(
    {
        "nuggets",
        "burger",
        "sausage",
        "instant noodle",
        "indomie",
        "topokki",
        "ready meal",
        "frozen meal",
        "spicy noodles",
        "mi sidap",
    }
)
FINAL_QUALITY_TUNA_HINTS = frozenset({"tuna"})
FINAL_QUALITY_RICE_FAMILY_HINTS = frozenset({"rice", "basmati", "sella", "long grain rice", "biryani rice"})
FINAL_QUALITY_PASTA_FAMILY_HINTS = frozenset(
    {"pasta", "spaghetti", "macaroni", "penne", "fusilli", "linguine", "vermicelli", "lasagna"}
)
FINAL_QUALITY_CRACKER_HINTS = frozenset({"cracker", "crackers"})

HUMAN_SOFT_PENALTY_BY_PATTERN = {
    "rice_tuna": 0.12,
    "chicken_tomato_paste": 0.16,
    "chicken_olive_oil": 0.22,
    "utilitarian_fat_protein_meal": 0.68,
    "utilitarian_egg_fat_meal": 0.52,
    "eggs_feta": 0.08,
    "labneh_chips": 0.10,
    "dates_evap_milk": 0.09,
    "cream_dessert_duplication": 0.30,
    "biscuits_coconut_milk": 0.22,
    "milk_cocoa_powder": 0.24,
    "meal_utilitarian_stock_oil": 0.30,
    "eggs_tomato_paste": 0.28,
    "dessert_plain_biscuit": 0.24,
    "nutella_plain_milk_occasion": 0.24,
    "meal_rice_meat_overused": 0.22,
}
HUMAN_BOOST_BY_PATTERN = {
    "meal_rice_meat": 0.08,
    "meal_rice_eggs": 0.12,
    "meal_chicken_bread_tortilla": 0.30,
    "meal_eggs_bread": 0.26,
    "meal_eggs_labneh": 0.24,
    "meal_spring_roll_cheese": 0.24,
    "meal_minced_meat_wrap": 0.26,
    "snack_biscuits_milk": 0.20,
    "snack_biscuits_chocolate": 0.15,
    "snack_nutella_bread": 0.26,
    "snack_chocolate_milk": 0.16,
    "snack_real_pattern_bonus": 0.16,
    "occasion_tea_evap_milk": 0.20,
    "occasion_coffee_evap_milk": 0.20,
    "occasion_dates_condensed_milk": 0.24,
    "fastfood_nuggets_bread": 0.28,
    "fastfood_nuggets_fries": 0.18,
    "fastfood_burger_bread": 0.18,
    "fastfood_burger_cheese": 0.16,
    "fastfood_burger_sauce": 0.14,
    "meal_labneh_bread": 0.24,
}

HUMAN_HINTS_CHICKEN = frozenset({"chicken"})
HUMAN_HINTS_TOMATO_PASTE = frozenset({"tomato paste"})
HUMAN_HINTS_OLIVE_OIL = frozenset({"olive oil"})
HUMAN_HINTS_EGGS = frozenset({"eggs", "egg"})
HUMAN_HINTS_FETA = frozenset({"feta"})
HUMAN_HINTS_LABNEH = frozenset({"labneh"})
HUMAN_HINTS_CHIPS = frozenset({"chips", "crisps"})
HUMAN_HINTS_DATES = frozenset({"dates"})
HUMAN_HINTS_EVAP_MILK = frozenset({"evaporated milk"})
HUMAN_HINTS_CONDENSED_MILK = frozenset({"condensed milk"})
HUMAN_HINTS_RICE = frozenset({"rice", "basmati", "sella", "biryani rice"})
HUMAN_HINTS_MEAT = frozenset({"meat", "beef", "lamb", "minced beef", "minced lamb"})
HUMAN_HINTS_BREAD = frozenset({"bread", "toast", "bun"})
HUMAN_HINTS_TORTILLA = frozenset({"tortilla", "wrap"})
HUMAN_HINTS_BISCUITS = frozenset({"biscuit", "biscuits", "cookie", "cookies", "wafer", "cracker", "crackers"})
HUMAN_HINTS_MILK = frozenset({"milk"})
HUMAN_HINTS_CHOCOLATE = frozenset({"chocolate", "cocoa", "nutella"})
HUMAN_HINTS_NUTELLA = frozenset({"nutella"})
HUMAN_HINTS_TEA = frozenset({"tea"})
HUMAN_HINTS_COFFEE = frozenset({"coffee"})
HUMAN_HINTS_NUGGETS = frozenset({"nuggets"})
HUMAN_HINTS_FRIES = frozenset({"fries", "french fries", "potato fries"})
HUMAN_HINTS_BURGER = frozenset({"burger"})
HUMAN_HINTS_CHEESE = frozenset({"cheese", "cheddar", "mozzarella", "feta"})
HUMAN_HINTS_SAUCE = frozenset({"sauce", "ketchup", "mayo", "mayonnaise", "dip"})
HUMAN_HINTS_COCONUT_MILK = frozenset({"coconut milk"})
HUMAN_HINTS_COCOA_POWDER = frozenset({"cocoa powder"})
HUMAN_HINTS_CREAM_TOKEN = frozenset({"cream"})
HUMAN_HINTS_FISH = frozenset({"fish", "seafood"})
HUMAN_HINTS_SPRING_ROLL = frozenset({"spring roll", "spring rolls"})
HUMAN_HINTS_MINCED_MEAT = frozenset({"minced beef", "minced lamb", "ground beef", "ground lamb", "minced"})
HUMAN_HINTS_STOCK = frozenset({"stock cube", "stock cubes", "bouillon", "seasoning cube", "seasoning cubes", "chicken stock"})
HUMAN_HINTS_FAT_BASE = frozenset({"oil", "olive oil", "sunflower oil", "ghee", "butter", "margarine"})
HUMAN_HINTS_PLAIN_COOKING_FAT = frozenset({"oil", "olive oil", "sunflower oil", "vegetable oil", "ghee", "butter"})
HUMAN_HINTS_PROTEIN_SAVORY = frozenset(
    {"chicken", "beef", "lamb", "meat", "fish", "seafood", "tuna", "burger", "sausage", "nuggets", "shawarma", "fillet"}
)
HUMAN_HINTS_SPICE_SEASONING = frozenset({"spice", "spices", "seasoning", "stock", "bouillon", "masala", "cube"})
MOTIF_HINTS_MILK_TEA = frozenset({"tea", "coffee", "milk", "evaporated milk", "tea milk"})
MOTIF_HINTS_DATES_MILK = frozenset({"milk", "evaporated milk", "condensed milk", "fresh cream", "qishta"})
MOTIF_HINTS_TOMATO_BASE = frozenset({"tomato paste", "tomato sauce", "ketchup"})
MOTIF_HINTS_CRUNCHY_SNACK = frozenset({"chips", "crisps", "cracker", "crackers", "nachos", "spring roll chips"})
SHOPPER_FAMILY_BASE_ADJUSTMENT: dict[str, float] = {
    "meal:chicken_wrap_meal": 0.20,
    "meal:nuggets_bread_fastmeal": 0.22,
    "meal:labneh_bread_meal": 0.20,
    "meal:minced_meat_wrap_meal": 0.18,
    "meal:egg_breakfast_meal": 0.16,
    "meal:rice_egg_meal": 0.12,
    "meal:rice_meat_meal": 0.06,
    "meal:rice_chicken_meal": 0.04,
    "meal:chicken_tomato_meal": 0.03,
    "meal:protein_bread_meal": 0.10,
    "meal:produce_protein_meal": 0.06,
    "meal:protein_grain_meal": -0.04,
    "meal:protein_noodles_meal": -0.10,
    "meal:protein_starch_generic_meal": -0.12,
    "snack:biscuit_milk_tea_snack": 0.05,
    "snack:wafer_chocolate_snack": 0.14,
    "snack:labneh_crunchy_snack": 0.06,
    "snack:nutella_snack_pair": 0.14,
    "snack:cheese_cracker_snack": 0.12,
    "snack:drink_snack_pair": 0.08,
    "snack:snack_dairy_pair": 0.06,
    "snack:dessert_dairy_pair": 0.05,
    "occasion:tea_milk_drink": 0.10,
    "occasion:coffee_milk_drink": 0.10,
    "occasion:dates_milk_treat": 0.24,
    "occasion:dates_cream_treat": 0.22,
    "occasion:dessert_cream_treat": 0.18,
    "occasion:beverage_dairy_treat": 0.12,
    "occasion:dates_dairy_treat": 0.16,
    "meal:chicken_fat_utilitarian": -0.55,
    "meal:protein_oil_utilitarian": -0.60,
    "meal:egg_fat_utilitarian": -0.40,
    "meal:fat_spice_utilitarian": -0.46,
    "meal:protein_plain_fat_utilitarian": -0.54,
}
SHOPPER_FAMILY_MEAL_DOMINANT = frozenset(
    {
        "meal:rice_meat_meal",
        "meal:rice_chicken_meal",
        "meal:chicken_tomato_meal",
        "meal:chicken_wrap_meal",
        "meal:chicken_fat_utilitarian",
        "meal:protein_oil_utilitarian",
        "meal:protein_plain_fat_utilitarian",
    }
)
SHOPPER_FAMILY_SNACK_DOMINANT = frozenset(
    {
        "snack:biscuit_milk_tea_snack",
        "snack:labneh_crunchy_snack",
    }
)
SHOPPER_FAMILY_OCCASION_DOMINANT = frozenset(
    {
        "occasion:tea_milk_drink",
        "occasion:coffee_milk_drink",
        "occasion:dates_milk_treat",
        "occasion:beverage_dairy_treat",
    }
)
SHOPPER_FAMILY_UTILITARIAN = frozenset(
    {
        "meal:chicken_fat_utilitarian",
        "meal:protein_oil_utilitarian",
        "meal:egg_fat_utilitarian",
        "meal:fat_spice_utilitarian",
        "meal:protein_plain_fat_utilitarian",
    }
)
DOMINANT_FAMILY_EXTRA_DECAY_STEP = 0.82
UTILITARIAN_FAMILY_EXTRA_DECAY_STEP = 0.92
DOMINANT_MEAL_FAMILY_STRONG_DECAY_STEP = 1.42
DOMINANT_MEAL_FAMILY_STRONG_DECAY_POWER = 1.28
STRICT_FALLBACK_CP_MIN = {LANE_MEAL: 26.0, LANE_SNACK: 24.0, LANE_OCCASION: 30.0}
STRICT_FALLBACK_PAIR_COUNT_MIN = {LANE_MEAL: 10, LANE_SNACK: 9, LANE_OCCASION: 10}
STRICT_FALLBACK_LANE_FIT_MIN = {LANE_MEAL: 0.64, LANE_SNACK: 0.78, LANE_OCCASION: 0.84}
STRICT_FALLBACK_TEMPLATE_MIN = {LANE_MEAL: 0.62, LANE_SNACK: 0.78, LANE_OCCASION: 0.80}
STRICT_FALLBACK_RISK_HINTS = frozenset(
    {
        "oil",
        "ghee",
        "butter",
        "spice",
        "seasoning",
        "stock cube",
        "bouillon",
        "ketchup",
        "mayo",
        "mayonnaise",
        "evaporated milk",
        "condensed milk",
    }
)
FALLBACK_MOTIF_KEY_TEA_EVAP = "tea_evap_milk"
FALLBACK_MOTIF_KEY_COFFEE_EVAP = "coffee_evap_milk"
FALLBACK_MOTIF_KEY_DATES_EVAP = "dates_evap_milk"
FALLBACK_MOTIF_KEY_DATES_COND = "dates_condensed_milk"
CONTROLLED_FALLBACK_MOTIFS = frozenset(
    {
        FALLBACK_MOTIF_KEY_TEA_EVAP,
        FALLBACK_MOTIF_KEY_COFFEE_EVAP,
        FALLBACK_MOTIF_KEY_DATES_EVAP,
        FALLBACK_MOTIF_KEY_DATES_COND,
    }
)
EVAP_REPETITIVE_FALLBACK_MOTIFS = frozenset(
    {FALLBACK_MOTIF_KEY_TEA_EVAP, FALLBACK_MOTIF_KEY_COFFEE_EVAP, FALLBACK_MOTIF_KEY_DATES_EVAP}
)
MEAL_DOMINANT_MOTIF_KEYWORDS = frozenset(
    {
        "meal:rice_meat_meal",
        "meal:rice_chicken_meal",
        "meal:chicken_tomato_meal",
        "meal:chicken_fat_utilitarian",
        "meal:protein_oil_utilitarian",
        "meal:protein_plain_fat_utilitarian",
    }
)
SNACK_DOMINANT_MOTIF_KEYWORDS = frozenset(
    {
        "snack:biscuit_milk_tea_snack",
        "snack:wafer_chocolate_snack",
    }
)
NONFOOD_TEXT_HINTS = frozenset(
    {
        "cat food",
        "dog food",
        "pet food",
        "litter",
        "shampoo",
        "blush",
        "lip balm",
        "cleaning",
        "bathroom",
        "bicarbonate",
        "baking soda",
        "detergent",
        "sanitizer",
        "bleach",
    }
)
GROUP_PRODUCE = "produce"
GROUP_SNACKS = "snacks"
GROUP_NOODLES = "noodles_pasta"
GROUP_SPICES = "spices"
GROUP_CARBS = "carbs_bread"
GROUP_PROTEIN = "protein"
GROUP_DAIRY = "dairy"
GROUP_SWEETS = "sweets"
GROUP_BEVERAGES = "beverages"
GROUP_BREAD_CARB = "bread_carb"
GROUP_CHIPS = "chips"
GROUP_CRACKERS = "crackers"
GROUP_COOKIES = "cookies"
GROUP_CHOCOLATE = "chocolate"
GROUP_CANDY = "candy"
GROUP_NUTS = "nuts"
GROUP_TEA = "tea"
GROUP_COFFEE = "coffee"
GROUP_MILK = "milk"
GROUP_SODA = "soda"
GROUP_JUICE = "juice"
GROUP_DATES = "dates"
GROUP_CREAM = "cream"
GROUP_CREAM_CHEESE = "cream_cheese"
GROUP_CHEESE = "cheese"
GROUP_NOODLES_PASTA = "noodles_pasta"
GROUP_RICE_GRAINS = "rice_grains"
GROUP_NONFOOD_CLEANING = "nonfood_cleaning"
GROUP_NONFOOD_HAIR = "nonfood_hair"
GROUP_NONFOOD_BODY = "nonfood_body"
GROUP_NONFOOD_TISSUE = "nonfood_tissue"
GROUP_NONFOOD_OTHER = "nonfood_other"
SEM_GRAINS = "grains"
SEM_PROTEIN = "protein"
SEM_DAIRY = "dairy"
SEM_SNACKS = "snacks"
SEM_PRODUCE = "produce"
SEM_DESSERT = "dessert"
SEM_BEVERAGE = "beverage"
SEM_SAUCE = "sauce"
SEM_SPICES = "spices"
SEM_DETERGENT = "detergent"
SEM_COSMETICS = "cosmetics"
SEM_SOAP = "soap"
SEM_CLEANING = "cleaning"
SEM_PACKAGING = "packaging"
SEM_UNKNOWN = "unknown"
FOOD_SEMANTIC_GROUPS = frozenset(
    {SEM_GRAINS, SEM_PROTEIN, SEM_DAIRY, SEM_SNACKS, SEM_PRODUCE, SEM_DESSERT, SEM_BEVERAGE, SEM_SAUCE, SEM_SPICES}
)
NONFOOD_SEMANTIC_GROUPS = frozenset({SEM_DETERGENT, SEM_COSMETICS, SEM_SOAP, SEM_CLEANING, SEM_PACKAGING})
SAUCE_HINTS = frozenset({"sauce", "paste", "ketchup", "mayo", "mayonnaise", "dip"})
SAVORY_SEMANTIC_GROUPS = frozenset({SEM_PROTEIN, SEM_GRAINS, SEM_PRODUCE, SEM_SAUCE, SEM_SPICES})
SNACK_APPROVED_PATTERNS = frozenset(
    {
        "drink_snack",
        "tea_snack",
        "cookie_milk",
        "sweet_milk",
        "wafer_chocolate",
        "dates_cream",
        "nuts_drink",
        "cheese_snack",
    }
)
SNACK_ANCHOR_ALLOWED_GROUPS = frozenset(
    {
        GROUP_CHIPS,
        GROUP_CRACKERS,
        GROUP_COOKIES,
        GROUP_CHOCOLATE,
        GROUP_CANDY,
        GROUP_NUTS,
        GROUP_TEA,
        GROUP_COFFEE,
        GROUP_SODA,
        GROUP_DATES,
        GROUP_CHEESE,
        GROUP_CREAM_CHEESE,
    }
)
SNACK_ANCHOR_BLOCKED_GROUPS = frozenset(
    {
        GROUP_BREAD_CARB,
        GROUP_PRODUCE,
        GROUP_RICE_GRAINS,
        GROUP_NOODLES_PASTA,
        GROUP_PROTEIN,
        GROUP_NONFOOD_CLEANING,
        GROUP_NONFOOD_HAIR,
        GROUP_NONFOOD_BODY,
        GROUP_NONFOOD_TISSUE,
        GROUP_NONFOOD_OTHER,
    }
)
NONFOOD_GROUP_MAP = {
    GROUP_NONFOOD_CLEANING: "cleaning",
    GROUP_NONFOOD_HAIR: "hair",
    GROUP_NONFOOD_BODY: "body",
    GROUP_NONFOOD_TISSUE: "tissue",
    GROUP_NONFOOD_OTHER: "other",
}
GROUP_PRIMARY_ORDER = (
    GROUP_BREAD_CARB,
    GROUP_CHIPS,
    GROUP_CRACKERS,
    GROUP_COOKIES,
    GROUP_CHOCOLATE,
    GROUP_CANDY,
    GROUP_NUTS,
    GROUP_TEA,
    GROUP_COFFEE,
    GROUP_SODA,
    GROUP_JUICE,
    GROUP_DATES,
    GROUP_CREAM_CHEESE,
    GROUP_CHEESE,
    GROUP_CREAM,
    GROUP_MILK,
    GROUP_PRODUCE,
    GROUP_NOODLES_PASTA,
    GROUP_RICE_GRAINS,
    GROUP_PROTEIN,
    GROUP_SPICES,
    GROUP_NONFOOD_CLEANING,
    GROUP_NONFOOD_HAIR,
    GROUP_NONFOOD_BODY,
    GROUP_NONFOOD_TISSUE,
    GROUP_NONFOOD_OTHER,
)
GROUP_OVERLAP_IGNORE = frozenset({GROUP_MILK, GROUP_CREAM, GROUP_CHEESE, GROUP_CREAM_CHEESE})
PRODUCT_GROUP_HINTS: dict[str, frozenset[str]] = {
    GROUP_PRODUCE: frozenset(
        {"onion", "garlic", "parsley", "mint", "vegetable", "carrot", "tomato", "cucumber", "pepper", "eggplant"}
    ),
    GROUP_SNACKS: frozenset(
        {"chips", "crisps", "popcorn", "nachos", "samosa chips", "potato sticks", "cracker"}
    ),
    GROUP_NOODLES: frozenset({"noodles", "indomie", "pasta", "spaghetti", "vermicelli"}),
    GROUP_SPICES: frozenset(
        {"pepper", "cumin", "spice", "masala", "stock", "seasoning", "bouillon", "turmeric", "cardamom"}
    ),
    GROUP_CARBS: frozenset({"bread", "tortilla", "toast", "rice", "flour", "oats", "semolina"}),
    GROUP_PROTEIN: frozenset({"chicken", "meat", "beef", "lamb", "tuna", "fish", "nuggets", "burger", "sausage", "eggs"}),
    GROUP_DAIRY: frozenset({"milk", "cheese", "cream", "yogurt", "labneh", "condensed"}),
    GROUP_SWEETS: frozenset({"cake", "biscuit", "cookie", "chocolate", "caramel", "custard", "cocoa", "dessert"}),
    GROUP_BEVERAGES: frozenset({"pepsi", "cola", "soda", "vimto", "juice", "tea", "coffee", "drink"}),
}
MEAL_REJECT_PRODUCE_SNACK_KEY = "meal_reject_produce_snack_count"
MEAL_REJECT_PRODUCE_NOODLES_KEY = "meal_reject_produce_noodles_count"
STAPLE_ALLOWED_COMPLEMENT_HINTS = {
    "rice": frozenset({"chicken", "meat", "fish", "spice", "tomato", "onion", "garlic", "beans", "vegetable"}),
    "flour": frozenset({"yeast", "baking powder", "cocoa", "sugar", "eggs", "milk", "butter"}),
    "sugar": frozenset({"tea", "coffee", "cake", "cocoa", "milk", "dessert", "biscuit"}),
    "oil": frozenset({"chicken", "meat", "fish", "vegetable", "spice", "sauce", "onion", "garlic"}),
    "salt": frozenset({"chicken", "meat", "fish", "vegetable", "soup", "spice", "tomato"}),
    "water": frozenset({"juice", "drink", "tea", "coffee"}),
}
LANE_CP_THRESHOLDS = {
    LANE_MEAL: 18.0,
    LANE_SNACK: 18.0,
    LANE_OCCASION: 22.0,
    LANE_NONFOOD: 8.0,
}
LANE_PAIR_COUNT_THRESHOLDS = {
    LANE_MEAL: 8,
    LANE_SNACK: 8,
    LANE_OCCASION: 10,
    LANE_NONFOOD: 3,
}
LANE_RECIPE_THRESHOLDS = {
    LANE_MEAL: 0.16,
    LANE_SNACK: 0.12,
    LANE_OCCASION: 0.10,
    LANE_NONFOOD: 1.00,
}
LANE_RECIPE_WEIGHTS = {
    LANE_MEAL: 0.14,
    LANE_SNACK: 0.06,
    LANE_OCCASION: 0.05,
    LANE_NONFOOD: 0.00,
}
LANE_CP_WEIGHTS = {
    LANE_MEAL: 0.03,
    LANE_SNACK: 0.05,
    LANE_OCCASION: 0.03,
    LANE_NONFOOD: 0.04,
}
LANE_ORIGIN_SCORE_FLOORS: dict[str, dict[str, float]] = {
    LANE_MEAL: {"top_bundle": 1.00, "copurchase_fallback": 1.15, "fallback_food": 1.28, "fallback_cleaning": 10.0},
    LANE_SNACK: {"top_bundle": 0.95, "copurchase_fallback": 1.00, "fallback_food": 1.08, "fallback_cleaning": 10.0},
    LANE_OCCASION: {"top_bundle": 0.98, "copurchase_fallback": 1.08, "fallback_food": 1.18, "fallback_cleaning": 10.0},
    LANE_NONFOOD: {"top_bundle": 0.0, "copurchase_fallback": 0.0, "fallback_food": 0.0, "fallback_cleaning": 0.0},
}
NONFOOD_GROUP_HINTS: dict[str, frozenset[str]] = {
    "personal_care": frozenset(
        {
            "shampoo",
            "conditioner",
            "soap",
            "body wash",
            "lotion",
            "deodorant",
            "moisturizing",
            "moisturiser",
            "moisturizer",
            "skin care",
            "skincare",
            "face cream",
            "hand cream",
            "underarm cream",
            "serum",
        }
    ),
    "laundry": frozenset({"detergent", "softener", "bleach", "stain"}),
    "household_cleaning": frozenset({"disinfectant", "cleaner", "dishwashing", "dish soap", "floor cleaner"}),
    "paper_goods": frozenset({"tissue", "toilet paper", "paper towel", "napkin"}),
    "disposables": frozenset({"trash bag", "waste bag", "foil", "wrap", "plastic cup", "plastic plate", "tablecloth"}),
}
PACKAGING_UTILITY_HINTS = frozenset(
    {
        "cup",
        "cups",
        "plate",
        "plates",
        "tissue",
        "tissues",
        "paper",
        "towel",
        "towels",
        "napkin",
        "napkins",
        "bag",
        "bags",
        "trash",
        "bin",
        "foil",
        "wrap",
        "container",
        "packaging",
        "detergent",
        "dishwashing",
        "cleaner",
        "disinfectant",
        "bleach",
        "shampoo",
        "conditioner",
        "soap",
        "body soap",
        "moisturizing",
        "moisturiser",
        "moisturizer",
        "skin care",
        "skincare",
        "face cream",
        "hand cream",
        "underarm cream",
        "serum",
    }
)
FAT_OIL_HINTS = frozenset({"ghee", "oil", "butter", "margarine", "shortening", "fat"})
CHEESE_SPREAD_HINTS = frozenset({"cheese", "cream cheese", "labneh", "spread"})
FRUIT_HINTS = frozenset({"banana", "strawberry", "strawberries", "orange", "oranges", "apple", "mango", "grape", "fruit"})
SAVORY_PROTEIN_HINTS = frozenset({"chicken", "beef", "lamb", "tuna", "fish", "meat", "burger", "nuggets", "sausage", "samosa"})
PANTRY_HINTS = frozenset(
    {
        "flour",
        "cornstarch",
        "baking powder",
        "yeast",
        "semolina",
        "oil",
        "salt",
        "sugar",
        "spice",
        "powder",
    }
)
MEAL_ANCHOR_HARD_BLOCK_HINTS = frozenset({"baking powder", "cornstarch", "water", "candy", "chips"})
OCCASION_ANCHOR_HARD_BLOCK_HINTS = frozenset({"water", "rice", "flour", "pasta", "spaghetti", "noodles", "indomie"})
APPLIANCE_TOOL_HINTS = frozenset(
    {
        "grinder",
        "meat grinder",
        "blender",
        "mixer",
        "oven",
        "microwave",
        "toaster",
        "knife",
        "pan",
        "pot",
        "tool",
        "appliance",
    }
)

def _build_curated_lane_library(
    lane: str,
    left_hints: tuple[str, ...],
    right_hints: tuple[str, ...],
    *,
    base_priority: int,
    base_score: float,
    base_cp_score: float,
    notes: str,
    limit: int = 200,
) -> tuple[dict[str, object], ...]:
    items: list[dict[str, object]] = []
    rank = 0
    for left in left_hints:
        for right in right_hints:
            if left == right:
                continue
            rank += 1
            items.append(
                {
                    "id": f"{lane}_{left.replace(' ', '_')}_{right.replace(' ', '_')}_v1",
                    "lane": lane,
                    "left": left,
                    "right": right,
                    "priority": int(base_priority + rank),
                    "score": float(base_score - min(8.0, (rank - 1) * 0.03)),
                    "cp_score": float(max(24.0, base_cp_score - min(10.0, (rank - 1) * 0.02))),
                    "pair_count": int(max(8, 20 - ((rank - 1) % 12))),
                    "notes": notes,
                }
            )
            if len(items) >= limit:
                return tuple(items)
    return tuple(items)


def _build_curated_pattern_library(
    lane: str,
    patterns: tuple[tuple[str, str], ...],
    *,
    base_priority: int,
    base_score: float,
    base_cp_score: float,
    notes: str,
) -> tuple[dict[str, object], ...]:
    items: list[dict[str, object]] = []
    for rank, (left, right) in enumerate(patterns, start=1):
        if not str(left).strip() or not str(right).strip() or str(left).strip() == str(right).strip():
            continue
        items.append(
            {
                "id": f"{lane}_{str(left).replace(' ', '_')}_{str(right).replace(' ', '_')}_v2",
                "lane": lane,
                "left": str(left),
                "right": str(right),
                "priority": int(base_priority + rank),
                "score": float(base_score - min(6.0, (rank - 1) * 0.04)),
                "cp_score": float(max(26.0, base_cp_score - min(8.0, (rank - 1) * 0.03))),
                "pair_count": int(max(10, 20 - ((rank - 1) % 8))),
                "notes": notes,
            }
        )
    return tuple(items)


CURATED_MEAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("eggs", "tomatoes"),
    ("eggs", "bread"),
    ("rice", "eggs"),
    ("rice", "lamb"),
    ("rice", "beef"),
    ("ghee", "lamb"),
    ("ghee", "chicken"),
    ("nuggets", "toast"),
    ("chicken", "tortilla"),
    ("labneh", "bread"),
    ("spring rolls", "kraft cheese"),
    ("rice", "chicken"),
    ("rice", "minced lamb"),
    ("rice", "minced beef"),
    ("chicken", "bread"),
    ("chicken", "tomato sauce"),
    ("fish", "rice"),
    ("tuna", "mayo"),
    ("pasta", "sauce"),
    ("pasta", "cheese"),
    ("lentils", "rice"),
    ("beans", "rice"),
    ("eggs", "cheese"),
    ("eggs", "labneh"),
    ("minced lamb", "tortilla"),
    ("minced beef", "tortilla"),
    ("chicken", "garlic"),
    ("chicken", "onions"),
    ("beef", "tomato sauce"),
    ("lamb", "tomato sauce"),
    ("rice", "tomatoes"),
    ("rice", "garlic"),
    ("rice", "onions"),
    ("fish", "tortilla"),
    ("tuna", "bread"),
    ("chicken nuggets", "toast"),
    ("burger buns", "minced beef"),
    ("samosa", "tomato sauce"),
    ("eggs", "ghee"),
    ("labneh", "toast"),
)
CURATED_SNACK_PATTERNS: tuple[tuple[str, str], ...] = (
    ("tea biscuits", "tea milk"),
    ("biscuits", "milk"),
    ("cookies", "milk"),
    ("chips", "cola"),
    ("chips", "juice"),
    ("crackers", "cheese"),
    ("wafer", "chocolate"),
    ("chocolate", "milk"),
    ("nuts", "juice"),
    ("popcorn", "cola"),
    ("brownie", "milk"),
    ("cake", "milk"),
    ("dates", "milk"),
    ("tea", "biscuit"),
    ("coffee", "biscuit"),
    ("granola", "yogurt"),
    ("pretzel", "cheese"),
    ("chips", "dip"),
    ("nachos", "cheese"),
    ("salt crackers", "cheese"),
    ("tea biscuits", "evaporated milk"),
    ("cocoa", "milk"),
    ("cookies", "tea"),
    ("crackers", "tea"),
    ("wafer", "milk"),
    ("nuts", "tea"),
    ("chocolate", "cookies"),
    ("cheese portions", "crackers"),
    ("dessert", "milk"),
    ("orange juice", "cookies"),
)
CURATED_OCCASION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("tea", "evaporated milk"),
    ("coffee beans", "evaporated milk"),
    ("dates", "cream"),
    ("dates", "condensed milk"),
    ("tea biscuits", "tea milk"),
    ("coffee", "biscuit"),
    ("tea", "biscuit"),
    ("arabic coffee", "dates"),
    ("dessert", "fresh cream"),
    ("coffee", "evaporated milk"),
    ("tea", "milk"),
    ("qishta", "dates"),
    ("mamoul", "coffee"),
    ("baklava", "coffee"),
    ("brownie", "coffee"),
    ("wafer", "coffee"),
    ("nuts", "coffee"),
    ("dates", "evaporated milk"),
    ("dates", "fresh milk"),
    ("chocolate biscuit", "tea"),
    ("cookies", "tea"),
    ("cookies", "coffee"),
    ("caramel dessert", "fresh cream"),
    ("pudding", "fresh cream"),
    ("saffron tea", "evaporated milk"),
    ("tea", "cookies"),
    ("coffee", "cookies"),
    ("dates", "qishta"),
    ("dessert", "evaporated milk"),
    ("tea", "tea milk"),
)
CURATED_CLEANING_PATTERNS: tuple[tuple[str, str], ...] = (
    ("detergent", "softener"),
    ("disinfectant", "bleach"),
    ("shampoo", "conditioner"),
    ("soap", "body wash"),
    ("paper towels", "tissue"),
    ("dish soap", "sponge"),
    ("trash bags", "paper towels"),
    ("toilet cleaner", "disinfectant"),
)


def _build_curated_bundle_list(
    lane: str,
    patterns: tuple[tuple[str, str], ...],
    *,
    base_priority: int,
    source_group: str,
) -> tuple[dict[str, object], ...]:
    out: list[dict[str, object]] = []
    for idx, (anchor_hint, complement_hint) in enumerate(patterns, start=1):
        out.append(
            {
                "id": f"{lane}_{idx:03d}",
                "lane": lane,
                "priority": int(base_priority + idx),
                "anchor_hint": str(anchor_hint),
                "complement_hint": str(complement_hint),
                "source_group": str(source_group),
            }
        )
    return tuple(out)


TOP_100_CURATED_FOOD_BUNDLES: tuple[dict[str, object], ...] = (
    _build_curated_bundle_list(LANE_MEAL, CURATED_MEAL_PATTERNS, base_priority=1000, source_group="fallback")
    + _build_curated_bundle_list(LANE_SNACK, CURATED_SNACK_PATTERNS, base_priority=2000, source_group="fallback")
    + _build_curated_bundle_list(LANE_OCCASION, CURATED_OCCASION_PATTERNS, base_priority=3000, source_group="fallback")
)
CURATED_CLEANING_FALLBACK_BUNDLES: tuple[dict[str, object], ...] = _build_curated_bundle_list(
    LANE_NONFOOD,
    CURATED_CLEANING_PATTERNS,
    base_priority=4000,
    source_group="fallback_cleaning",
)


def _validate_curated_bundle_catalog(entries: tuple[dict[str, object], ...], *, expected_size: int) -> None:
    if len(entries) != expected_size:
        raise ValueError(f"curated fallback catalog must contain exactly {expected_size} entries, found {len(entries)}")
    required = {"id", "lane", "priority", "anchor_hint", "complement_hint", "source_group"}
    valid_lanes = {LANE_MEAL, LANE_SNACK, LANE_OCCASION}
    for entry in entries:
        missing = required - set(entry.keys())
        if missing:
            raise ValueError(f"curated fallback entry missing fields: {sorted(missing)}")
        lane = str(entry.get("lane", "")).strip().lower()
        if lane not in valid_lanes:
            raise ValueError(f"invalid curated lane '{lane}'")
        if not str(entry.get("anchor_hint", "")).strip() or not str(entry.get("complement_hint", "")).strip():
            raise ValueError("curated fallback hints must be non-empty")


_validate_curated_bundle_catalog(TOP_100_CURATED_FOOD_BUNDLES, expected_size=100)


@dataclass(frozen=True)
class PersonProfile:
    profile_id: str
    source: str
    order_ids: list[int]
    history_product_ids: list[int]
    history_items: list[str]
    created_at: str
    history_counts: dict[int, int] = field(default_factory=dict)


@dataclass
class PersonRecommendationState:
    profiles: list[PersonProfile] = field(default_factory=list)
    recommendations: list[dict[str, object]] = field(default_factory=list)
    run_id: str = ""
    last_updated: float = 0.0


@dataclass
class ManualProfileBuildResult:
    profile: PersonProfile | None
    warnings: list[str] = field(default_factory=list)
    matched_count: int = 0


@dataclass
class ServingTelemetry:
    overall: dict[str, int] = field(default_factory=dict)
    by_lane: dict[str, dict[str, int]] = field(default_factory=dict)


@dataclass
class _ServingProfileRecorder:
    enabled: bool = False
    per_profile: list[dict[str, object]] = field(default_factory=list)
    stage_totals: dict[str, float] = field(default_factory=dict)

    def add_stage(self, stage: str, seconds: float) -> None:
        if not self.enabled:
            return
        key = str(stage)
        self.stage_totals[key] = float(self.stage_totals.get(key, 0.0)) + float(seconds)

    def add_profile(self, payload: dict[str, object]) -> None:
        if not self.enabled:
            return
        self.per_profile.append(dict(payload))


@dataclass(frozen=True)
class OrderPool:
    order_product_ids: dict[int, tuple[int, ...]]
    order_product_names: dict[int, tuple[str, ...]]
    preferred_order_ids: tuple[int, ...]
    fallback_order_ids: tuple[int, ...]


@dataclass(frozen=True)
class ProductMatcher:
    product_name_by_id: dict[int, str]
    normalized_name_to_ids: dict[str, tuple[int, ...]]
    normalized_names: tuple[str, ...]


@dataclass(frozen=True)
class PersonalizationContext:
    product_name_by_id: dict[int, str]
    product_price_by_id: dict[int, float]
    product_picture_by_id: dict[int, str]
    neighbors: dict[int, tuple[tuple[int, float], ...]]
    recipe_score_by_id: dict[int, float] = field(default_factory=dict)
    ingredient_by_id: dict[int, str] = field(default_factory=dict)
    ingredient_recipe_lookup: dict[str, tuple[str, ...]] = field(default_factory=dict)
    product_family_by_id: dict[int, str] = field(default_factory=dict)
    category_by_id: dict[int, str] = field(default_factory=dict)
    product_brand_by_id: dict[int, str] = field(default_factory=dict)
    non_food_ids: frozenset[int] = field(default_factory=frozenset)


@dataclass(frozen=True)
class PairAnalysis:
    anchor: int
    complement: int
    lane: str
    pair_row: pd.Series | None
    anchor_name: str
    complement_name: str
    anchor_category: str
    complement_category: str
    anchor_family: str
    complement_family: str
    anchor_roles: frozenset[str]
    complement_roles: frozenset[str]
    anchor_groups: set[str]
    complement_groups: set[str]
    anchor_nonfood: bool
    complement_nonfood: bool
    anchor_packaging: bool
    complement_packaging: bool
    semantic: semantics.BundleSemantics
    visible_ok: bool
    visible_reason: str | None


def _pair_analysis(
    anchor: int,
    complement: int,
    lane: str,
    context: PersonalizationContext,
    pair_row: pd.Series | None = None,
) -> PairAnalysis:
    anchor_name, anchor_category, anchor_family, _anchor_text = _semantic_product_text(int(anchor), context, row=pair_row, side="a")
    complement_name, complement_category, complement_family, _comp_text = _semantic_product_text(
        int(complement), context, row=pair_row, side="b"
    )
    anchor_roles = semantics.infer_product_roles(anchor_name, anchor_category, anchor_family)
    complement_roles = semantics.infer_product_roles(complement_name, complement_category, complement_family)
    anchor_groups = _group_labels_from_text(anchor_name, anchor_category, anchor_family)
    complement_groups = _group_labels_from_text(complement_name, complement_category, complement_family)
    semantic = _semantic_pair_snapshot(int(anchor), int(complement), lane, context, pair_row=pair_row)
    visible_ok, visible_reason = _semantic_visible_expression_ok(int(anchor), int(complement), lane, context, pair_row=pair_row)
    return PairAnalysis(
        anchor=int(anchor),
        complement=int(complement),
        lane=str(lane),
        pair_row=pair_row,
        anchor_name=anchor_name,
        complement_name=complement_name,
        anchor_category=anchor_category,
        complement_category=complement_category,
        anchor_family=anchor_family,
        complement_family=complement_family,
        anchor_roles=anchor_roles,
        complement_roles=complement_roles,
        anchor_groups=anchor_groups,
        complement_groups=complement_groups,
        anchor_nonfood=_is_nonfood_product(int(anchor), context, row=pair_row, side="a"),
        complement_nonfood=_is_nonfood_product(int(complement), context, row=pair_row, side="b"),
        anchor_packaging=_is_packaging_or_utility_item(int(anchor), context, row=pair_row, side="a"),
        complement_packaging=_is_packaging_or_utility_item(int(complement), context, row=pair_row, side="b"),
        semantic=semantic,
        visible_ok=bool(visible_ok),
        visible_reason=str(visible_reason) if visible_reason else None,
    )


def _source_group_from_source(source: str) -> str:
    origin = str(source or "").strip().lower()
    if origin == "top_bundle":
        return "top_bundle"
    if origin == "copurchase_fallback":
        return "copurchase_fallback"
    if origin == "fallback_cleaning":
        return "fallback_cleaning"
    if origin == "fallback_food":
        return "fallback_food"
    if origin.startswith("fallback_cleaning:"):
        return "fallback_cleaning"
    if origin.startswith("fallback:") or origin.startswith("fallback_template:") or origin.startswith("fallback_"):
        return "fallback_food"
    return "other"


def _candidate_rank_key(candidate: dict[str, object]) -> tuple[int, float, float, float, int, int]:
    source_group = _source_group_from_source(str(candidate.get("source", "")))
    source_priority_map = {
        "top_bundle": 0,
        "copurchase_fallback": 1,
        "fallback_food": 2,
        "fallback_cleaning": 3,
        "other": 4,
    }
    source_priority = int(source_priority_map.get(source_group, 4))
    return (
        source_priority,
        -float(candidate["personal_score"]),
        -float(candidate.get("lane_fit_score", 0.0)),
        -float(semantics.semantic_score_prior(str(candidate.get("pair_strength", "")))),
        int(candidate.get("complement", -1)),
        int(candidate.get("anchor", -1)),
    )


def _top_bundle_scan_limit(lane: str) -> int:
    return int(MAX_TOP_BUNDLE_CANDIDATES_BY_LANE.get(str(lane).strip().lower(), MAX_TOP_BUNDLE_CANDIDATES))


def _increment_counter(counter: dict[str, int], key: str, amount: int = 1) -> None:
    counter[str(key)] = int(counter.get(str(key), 0)) + int(amount)


def _record_serving_telemetry(telemetry: ServingTelemetry | None, lane: str, key: str, amount: int = 1) -> None:
    if telemetry is None:
        return
    _increment_counter(telemetry.overall, key, amount=amount)
    lane_key = str(lane).strip().lower()
    lane_bucket = telemetry.by_lane.setdefault(lane_key, {})
    _increment_counter(lane_bucket, key, amount=amount)


def _serving_profiling_enabled() -> bool:
    raw = str(os.getenv(SERVING_PROFILING_ENV, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(0.0, min(1.0, float(percentile))) * float(len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = float(rank - lower)
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def get_last_serving_profile_metrics() -> dict[str, object]:
    payload = _LAST_SERVING_PROFILE_METRICS
    return dict(payload) if isinstance(payload, dict) else {}


def _anchor_rank_sort_key(item: tuple[int, float]) -> tuple[float, int]:
    pid, score = item
    return (-float(score), int(pid))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@lru_cache(maxsize=200_000)
def _normalise_text_cached(value: str) -> str:
    return _NORMALIZE_TEXT_PATTERN.sub(" ", str(value).lower()).strip()


def _normalise_text(value: object) -> str:
    return _normalise_text_cached(str(value))


def _load_brand_alias_tokens(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("brands", []) if isinstance(payload, dict) else []
    out: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        brand = _normalise_text(row.get("brand", ""))
        aliases_raw = row.get("aliases", [])
        aliases = {_normalise_text(x) for x in aliases_raw if _normalise_text(x)}
        if brand:
            aliases.add(brand)
        if brand and aliases:
            out[brand] = aliases
    return out


def _infer_brand(name: str, brand_alias_tokens: dict[str, set[str]]) -> str:
    norm = _normalise_text(name)
    if not norm:
        return ""
    text_tokens = {t for t in norm.split() if t}
    for brand, aliases in brand_alias_tokens.items():
        if text_tokens & aliases:
            return brand
    # safe heuristic fallback: first strong token if not generic
    parts = [p for p in norm.split() if p and p not in BRAND_STOPWORDS and not any(ch.isdigit() for ch in p)]
    if not parts:
        return ""
    first = parts[0]
    if len(first) < 4:
        return ""
    return first


def _empty_order_pool() -> OrderPool:
    return OrderPool(
        order_product_ids={},
        order_product_names={},
        preferred_order_ids=(),
        fallback_order_ids=(),
    )


def _empty_product_matcher() -> ProductMatcher:
    return ProductMatcher(product_name_by_id={}, normalized_name_to_ids={}, normalized_names=())


def _empty_personalization_context() -> PersonalizationContext:
    return PersonalizationContext(
        product_name_by_id={},
        product_price_by_id={},
        product_picture_by_id={},
        neighbors={},
        recipe_score_by_id={},
        ingredient_by_id={},
        ingredient_recipe_lookup={},
        product_family_by_id={},
        category_by_id={},
        product_brand_by_id={},
        non_food_ids=frozenset(),
    )


def _feedback_csv_path() -> Path:
    try:
        return get_paths().output_dir / FEEDBACK_REVIEW_REL_PATH
    except Exception:
        return Path("output") / FEEDBACK_REVIEW_REL_PATH


def _load_feedback_lookup_once() -> dict[str, object]:
    try:
        feedback_df = load_bundle_feedback(_feedback_csv_path())
        lookup = build_feedback_lookup(feedback_df)
        return lookup if isinstance(lookup, dict) else {}
    except Exception:
        return {}


FEEDBACK_LOOKUP: dict[str, object] = _load_feedback_lookup_once()


def _feedback_pair_key(anchor: int, complement: int) -> tuple[int, int] | None:
    if int(anchor) <= 0 or int(complement) <= 0:
        return None
    return _pair_key(int(anchor), int(complement))


def _feedback_pair_boost(anchor: int, complement: int) -> float:
    key = _feedback_pair_key(anchor, complement)
    if key is None:
        return 0.0
    if not isinstance(FEEDBACK_LOOKUP, dict):
        return 0.0
    boosts = FEEDBACK_LOOKUP.get("pair_boosts", {})
    strong_pairs = FEEDBACK_LOOKUP.get("strong_pairs", set())
    boost = float(boosts.get(key, 0.0))
    if key in strong_pairs:
        boost = max(boost, FEEDBACK_STRONG_BOOST)
    return float(boost)


def _feedback_pair_penalty(anchor: int, complement: int) -> float:
    key = _feedback_pair_key(anchor, complement)
    if key is None:
        return 0.0
    if not isinstance(FEEDBACK_LOOKUP, dict):
        return 0.0
    penalties = FEEDBACK_LOOKUP.get("pair_penalties", {})
    weak_pairs = FEEDBACK_LOOKUP.get("weak_pairs", set())
    trash_pairs = FEEDBACK_LOOKUP.get("trash_pairs", set())
    penalty = float(penalties.get(key, 0.0))
    if key in weak_pairs:
        penalty = max(penalty, FEEDBACK_WEAK_PENALTY)
    if key in trash_pairs:
        penalty = max(penalty, FEEDBACK_TRASH_PENALTY)
    return float(penalty)


def _feedback_pair_class(anchor: int, complement: int) -> str:
    key = _feedback_pair_key(anchor, complement)
    if key is None or not isinstance(FEEDBACK_LOOKUP, dict):
        return ""
    if key in FEEDBACK_LOOKUP.get("strong_pairs", set()):
        return "strong"
    if key in FEEDBACK_LOOKUP.get("staple_pairs", set()):
        return "staple"
    if key in FEEDBACK_LOOKUP.get("weak_pairs", set()):
        return "weak"
    if key in FEEDBACK_LOOKUP.get("trash_pairs", set()):
        return "trash"
    return ""


def _feedback_pair_override(anchor: int, complement: int) -> bool:
    key = _feedback_pair_key(anchor, complement)
    if key is None:
        return False
    overrides = FEEDBACK_LOOKUP.get("pair_overrides", set()) if isinstance(FEEDBACK_LOOKUP, dict) else set()
    return key in overrides


@lru_cache(maxsize=4)
def _load_order_pool_cached(path_str: str, mtime_ns: int) -> OrderPool:
    del mtime_ns
    try:
        orders = pd.read_pickle(path_str)
    except Exception:
        return _empty_order_pool()

    required = {"order_id", "product_id", "product_name"}
    if orders.empty or not required.issubset(set(orders.columns)):
        return _empty_order_pool()

    orders = orders.copy()
    orders["order_id"] = pd.to_numeric(orders["order_id"], errors="coerce")
    orders["product_id"] = pd.to_numeric(orders["product_id"], errors="coerce")
    orders = orders.dropna(subset=["order_id", "product_id"])
    if orders.empty:
        return _empty_order_pool()

    orders["order_id"] = orders["order_id"].astype("int64")
    orders["product_id"] = orders["product_id"].astype("int64")
    orders["product_name"] = orders["product_name"].astype(str).str.strip()

    order_products = orders[["order_id", "product_id"]].drop_duplicates()
    order_pid_series = order_products.groupby("order_id")["product_id"].agg(list)
    order_product_ids = {
        int(order_id): tuple(int(pid) for pid in pids)
        for order_id, pids in order_pid_series.items()
        if pids
    }

    name_pairs = (
        orders[["order_id", "product_id", "product_name"]]
        .drop_duplicates(subset=["order_id", "product_id"])
        .copy()
    )
    order_product_names: dict[int, tuple[str, ...]] = {}
    for order_id, group in name_pairs.groupby("order_id", sort=False):
        names = [str(name).strip() for name in group["product_name"].tolist() if str(name).strip()]
        if names:
            order_product_names[int(order_id)] = tuple(names)

    order_sizes = order_products.groupby("order_id")["product_id"].nunique().sort_values(ascending=False)
    preferred_order_ids = tuple(int(oid) for oid in order_sizes[order_sizes >= 2].index.tolist())
    fallback_order_ids = tuple(int(oid) for oid in order_sizes[order_sizes >= 1].index.tolist())

    return OrderPool(
        order_product_ids=order_product_ids,
        order_product_names=order_product_names,
        preferred_order_ids=preferred_order_ids,
        fallback_order_ids=fallback_order_ids,
    )


def load_order_pool(base_dir: Path) -> OrderPool:
    paths = get_paths(project_root=base_dir)
    orders_path = paths.data_processed_dir / "filtered_orders.pkl"
    if not orders_path.exists():
        return _empty_order_pool()
    return _load_order_pool_cached(str(orders_path.resolve()), int(orders_path.stat().st_mtime_ns))


@lru_cache(maxsize=4)
def _load_product_matcher_cached(path_str: str, mtime_ns: int) -> ProductMatcher:
    del mtime_ns
    try:
        orders = pd.read_pickle(path_str)
    except Exception:
        return _empty_product_matcher()

    required = {"product_id", "product_name"}
    if orders.empty or not required.issubset(set(orders.columns)):
        return _empty_product_matcher()

    rows = orders[["product_id", "product_name"]].copy()
    rows["product_id"] = pd.to_numeric(rows["product_id"], errors="coerce")
    rows = rows.dropna(subset=["product_id"])
    if rows.empty:
        return _empty_product_matcher()

    rows["product_id"] = rows["product_id"].astype("int64")
    rows["product_name"] = rows["product_name"].astype(str).str.strip()
    rows = rows[rows["product_name"] != ""]
    if rows.empty:
        return _empty_product_matcher()

    dedup = rows.drop_duplicates(subset=["product_id"], keep="first")
    product_name_by_id = {int(pid): str(name) for pid, name in zip(dedup["product_id"], dedup["product_name"])}

    normalized_name_to_ids: dict[str, list[int]] = {}
    for pid, name in product_name_by_id.items():
        norm = _normalise_text(name)
        if not norm:
            continue
        normalized_name_to_ids.setdefault(norm, []).append(int(pid))

    normalized_name_to_ids_sorted = {
        key: tuple(sorted(set(ids)))
        for key, ids in normalized_name_to_ids.items()
    }

    return ProductMatcher(
        product_name_by_id=product_name_by_id,
        normalized_name_to_ids=normalized_name_to_ids_sorted,
        normalized_names=tuple(sorted(normalized_name_to_ids_sorted.keys())),
    )


def load_product_matcher(base_dir: Path) -> ProductMatcher:
    paths = get_paths(project_root=base_dir)
    orders_path = paths.data_processed_dir / "filtered_orders.pkl"
    if not orders_path.exists():
        return _empty_product_matcher()
    return _load_product_matcher_cached(str(orders_path.resolve()), int(orders_path.stat().st_mtime_ns))


@lru_cache(maxsize=4)
def _load_personalization_context_cached(
    orders_path_str: str,
    orders_mtime_ns: int,
    copurchase_path_str: str,
    copurchase_mtime_ns: int,
    recipe_path_str: str,
    recipe_mtime_ns: int,
    categories_path_str: str,
    categories_mtime_ns: int,
    ingredient_links_path_str: str,
    ingredient_links_mtime_ns: int,
    brand_alias_path_str: str,
    brand_alias_mtime_ns: int,
) -> PersonalizationContext:
    del (
        orders_mtime_ns,
        copurchase_mtime_ns,
        recipe_mtime_ns,
        categories_mtime_ns,
        ingredient_links_mtime_ns,
        brand_alias_mtime_ns,
    )

    try:
        orders = pd.read_pickle(orders_path_str)
    except Exception:
        return _empty_personalization_context()

    if orders.empty or not {"product_id", "product_name", "unit_price"}.issubset(set(orders.columns)):
        return _empty_personalization_context()

    rows = orders.copy()
    rows["product_id"] = pd.to_numeric(rows["product_id"], errors="coerce")
    rows["unit_price"] = pd.to_numeric(rows["unit_price"], errors="coerce")
    rows = rows.dropna(subset=["product_id"])
    if rows.empty:
        return _empty_personalization_context()

    rows["product_id"] = rows["product_id"].astype("int64")
    rows["product_name"] = rows["product_name"].astype(str).str.strip()

    name_rows = rows[["product_id", "product_name"]].copy()
    name_rows = name_rows[name_rows["product_name"] != ""]
    name_rows = name_rows.drop_duplicates(subset=["product_id"], keep="first")
    product_name_by_id = {
        int(pid): str(name)
        for pid, name in zip(name_rows["product_id"], name_rows["product_name"])
    }

    price_rows = rows.dropna(subset=["unit_price"])
    product_price_by_id = {
        int(pid): float(price)
        for pid, price in price_rows.groupby("product_id")["unit_price"].median().items()
    }

    product_picture_by_id: dict[int, str] = {}
    if "product_picture" in rows.columns:
        pic_rows = rows[["product_id", "product_picture"]].copy()
        pic_rows["product_picture"] = pic_rows["product_picture"].fillna("").astype(str).str.strip()
        pic_rows = pic_rows[pic_rows["product_picture"] != ""]
        pic_rows = pic_rows.drop_duplicates(subset=["product_id"], keep="first")
        product_picture_by_id = {
            int(pid): str(pic)
            for pid, pic in zip(pic_rows["product_id"], pic_rows["product_picture"])
        }

    neighbors: dict[int, list[tuple[int, float]]] = {}
    cp_path = Path(copurchase_path_str)
    if cp_path.exists():
        try:
            cp = pd.read_csv(cp_path)
            cp["product_a"] = pd.to_numeric(cp.get("product_a"), errors="coerce")
            cp["product_b"] = pd.to_numeric(cp.get("product_b"), errors="coerce")
            cp["score"] = pd.to_numeric(cp.get("score"), errors="coerce").fillna(0.0)
            cp = cp.dropna(subset=["product_a", "product_b"])
            cp = cp[(cp["score"] > 0) & (cp["product_a"] != cp["product_b"])]
            for row in cp.itertuples(index=False):
                a = int(row.product_a)
                b = int(row.product_b)
                s = float(row.score)
                neighbors.setdefault(a, []).append((b, s))
                neighbors.setdefault(b, []).append((a, s))
        except Exception:
            neighbors = {}

    neighbors_sorted: dict[int, tuple[tuple[int, float], ...]] = {}
    for pid, pairs in neighbors.items():
        pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
        dedup: dict[int, float] = {}
        for n_pid, score in pairs_sorted:
            if n_pid not in dedup or score > dedup[n_pid]:
                dedup[n_pid] = score
        compact = sorted(dedup.items(), key=lambda x: x[1], reverse=True)
        neighbors_sorted[int(pid)] = tuple((int(n), float(s)) for n, s in compact[:MAX_NEIGHBORS_PER_ANCHOR])

    recipe_score_by_id: dict[int, float] = {}
    ingredient_by_id: dict[int, str] = {}
    recipe_path = Path(recipe_path_str)
    if recipe_path.exists():
        try:
            recipe_df = pd.read_csv(recipe_path)
            if not recipe_df.empty and "product_id" in recipe_df.columns:
                recipe_df["product_id"] = pd.to_numeric(recipe_df["product_id"], errors="coerce")
                recipe_df = recipe_df.dropna(subset=["product_id"]).copy()
                recipe_df["product_id"] = recipe_df["product_id"].astype("int64")
                if "recipe_score" in recipe_df.columns:
                    recipe_score_by_id = {
                        int(pid): float(score)
                        for pid, score in zip(
                            recipe_df["product_id"],
                            pd.to_numeric(recipe_df["recipe_score"], errors="coerce").fillna(0.0),
                        )
                    }
                if "matched_ingredient" in recipe_df.columns:
                    ingredient_by_id = {
                        int(pid): str(ing).strip().lower()
                        for pid, ing in zip(recipe_df["product_id"], recipe_df["matched_ingredient"].fillna(""))
                    }
        except Exception:
            recipe_score_by_id = {}
            ingredient_by_id = {}

    ingredient_recipe_lookup: dict[str, tuple[str, ...]] = {}
    ingredient_links_path = Path(ingredient_links_path_str)
    if ingredient_links_path.exists():
        try:
            links_df = pd.read_csv(ingredient_links_path)
            if not links_df.empty and {"ingredient_key", "recipe_key"}.issubset(set(links_df.columns)):
                lookup: dict[str, set[str]] = {}
                for ingredient, recipe in zip(links_df["ingredient_key"], links_df["recipe_key"]):
                    ing = str(ingredient).strip().lower()
                    rec = str(recipe).strip().lower()
                    if not ing or not rec:
                        continue
                    lookup.setdefault(ing, set()).add(rec)
                ingredient_recipe_lookup = {k: tuple(sorted(v)) for k, v in lookup.items()}
        except Exception:
            ingredient_recipe_lookup = {}

    product_family_by_id: dict[int, str] = {}
    category_by_id: dict[int, str] = {}
    non_food_ids: set[int] = set()
    categories_path = Path(categories_path_str)
    if categories_path.exists():
        try:
            categories_df = pd.read_csv(categories_path)
            if not categories_df.empty and "product_id" in categories_df.columns:
                categories_df["product_id"] = pd.to_numeric(categories_df["product_id"], errors="coerce")
                categories_df = categories_df.dropna(subset=["product_id"]).copy()
                categories_df["product_id"] = categories_df["product_id"].astype("int64")
                category_col = (
                    categories_df["category"].fillna("").astype(str)
                    if "category" in categories_df.columns
                    else pd.Series("", index=categories_df.index, dtype="object")
                )
                family_col = (
                    categories_df["product_family"].fillna("").astype(str)
                    if "product_family" in categories_df.columns
                    else pd.Series("", index=categories_df.index, dtype="object")
                )
                tags_col = (
                    categories_df["category_tags"].fillna("").astype(str)
                    if "category_tags" in categories_df.columns
                    else pd.Series("", index=categories_df.index, dtype="object")
                )
                for pid, cat, family, tags in zip(categories_df["product_id"], category_col, family_col, tags_col):
                    pid_int = int(pid)
                    category_text = str(cat).strip().lower()
                    family_text = str(family).strip()
                    tags_set = {t for t in str(tags).strip().split("|") if t}
                    category_by_id[pid_int] = category_text
                    product_family_by_id[pid_int] = family_text
                    if category_text == NON_FOOD_TAG or NON_FOOD_TAG in tags_set:
                        non_food_ids.add(pid_int)
        except Exception:
            product_family_by_id = {}
            category_by_id = {}
            non_food_ids = set()

    brand_alias_tokens = _load_brand_alias_tokens(Path(brand_alias_path_str))
    product_brand_by_id: dict[int, str] = {}
    for pid, name in product_name_by_id.items():
        product_brand_by_id[int(pid)] = _infer_brand(str(name), brand_alias_tokens)

    return PersonalizationContext(
        product_name_by_id=product_name_by_id,
        product_price_by_id=product_price_by_id,
        product_picture_by_id=product_picture_by_id,
        neighbors=neighbors_sorted,
        recipe_score_by_id=recipe_score_by_id,
        ingredient_by_id=ingredient_by_id,
        ingredient_recipe_lookup=ingredient_recipe_lookup,
        product_family_by_id=product_family_by_id,
        category_by_id=category_by_id,
        product_brand_by_id=product_brand_by_id,
        non_food_ids=frozenset(non_food_ids),
    )


def load_personalization_context(base_dir: Path) -> PersonalizationContext:
    paths = get_paths(project_root=base_dir)
    orders_path = paths.data_processed_dir / "filtered_orders.pkl"
    cp_path = paths.data_processed_dir / "copurchase_scores.csv"
    recipe_path = paths.data_processed_dir / "product_recipe_scores.csv"
    categories_path = paths.data_processed_dir / "product_categories.csv"
    ingredient_links_path = paths.data_processed_dir / "ingredient_recipe_links.csv"
    brand_alias_path = paths.data_reference_dir / "brand_aliases.json"
    if not orders_path.exists():
        return _empty_personalization_context()
    cp_mtime = int(cp_path.stat().st_mtime_ns) if cp_path.exists() else 0
    recipe_mtime = int(recipe_path.stat().st_mtime_ns) if recipe_path.exists() else 0
    categories_mtime = int(categories_path.stat().st_mtime_ns) if categories_path.exists() else 0
    ingredient_links_mtime = int(ingredient_links_path.stat().st_mtime_ns) if ingredient_links_path.exists() else 0
    brand_alias_mtime = int(brand_alias_path.stat().st_mtime_ns) if brand_alias_path.exists() else 0
    return _load_personalization_context_cached(
        str(orders_path.resolve()),
        int(orders_path.stat().st_mtime_ns),
        str(cp_path.resolve()),
        cp_mtime,
        str(recipe_path.resolve()),
        recipe_mtime,
        str(categories_path.resolve()),
        categories_mtime,
        str(ingredient_links_path.resolve()),
        ingredient_links_mtime,
        str(brand_alias_path.resolve()),
        brand_alias_mtime,
    )


@lru_cache(maxsize=4)
def _load_category_importance_lookup_cached(path_str: str, mtime_ns: int) -> dict[str, float]:
    del mtime_ns
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "category" not in df.columns or "final_score" not in df.columns:
        return {}
    score = pd.to_numeric(df["final_score"], errors="coerce").fillna(0.0)
    lo = float(score.min()) if len(score) else 0.0
    hi = float(score.max()) if len(score) else 0.0
    span = max(hi - lo, 1e-9)
    out: dict[str, float] = {}
    for cat, raw in zip(df["category"], score):
        key = _normalise_text(cat)
        if not key:
            continue
        norm = (float(raw) - lo) / span
        out[key] = max(out.get(key, 0.0), float(max(0.0, min(1.0, norm))))
    return out


def _load_category_importance_lookup(base_dir: Path) -> dict[str, float]:
    path = get_paths(project_root=base_dir).data_reference_dir / "category_importance.csv"
    if not path.exists():
        return {}
    return _load_category_importance_lookup_cached(str(path.resolve()), int(path.stat().st_mtime_ns))


def _category_importance_norm(
    pid: int,
    context: PersonalizationContext,
    category_importance_lookup: dict[str, float],
) -> float:
    category = _normalise_text(context.category_by_id.get(int(pid), ""))
    if not category:
        return 0.0
    return float(category_importance_lookup.get(category, 0.0))


def _new_profile_id() -> str:
    return f"person_{secrets.token_hex(6)}"


def build_random_profile(
    order_pool: OrderPool,
    preferred_orders: int = 2,
    fallback_orders: int = 1,
    rng: random.Random | None = None,
) -> PersonProfile | None:
    generator = rng or random.Random()

    selected_order_ids: list[int] = []
    preferred = list(order_pool.preferred_order_ids)
    if len(preferred) >= preferred_orders:
        selected_order_ids = generator.sample(preferred, preferred_orders)
    elif preferred:
        selected_order_ids = [int(preferred[0])]

    if not selected_order_ids:
        fallback = list(order_pool.fallback_order_ids)
        if not fallback:
            return None
        take = max(1, min(fallback_orders, len(fallback)))
        selected_order_ids = generator.sample(fallback, take)

    history_ids: set[int] = set()
    history_counts: dict[int, int] = {}
    history_items: list[str] = []
    seen_names: set[str] = set()
    for order_id in selected_order_ids:
        for pid in order_pool.order_product_ids.get(int(order_id), ()):  # type: ignore[arg-type]
            pid_int = int(pid)
            history_ids.add(pid_int)
            history_counts[pid_int] = int(history_counts.get(pid_int, 0)) + 1
        for name in order_pool.order_product_names.get(int(order_id), ()):  # type: ignore[arg-type]
            text = str(name).strip()
            if text and text not in seen_names:
                seen_names.add(text)
                history_items.append(text)

    if not history_ids:
        return None

    return PersonProfile(
        profile_id=_new_profile_id(),
        source="random",
        order_ids=sorted(int(x) for x in selected_order_ids),
        history_product_ids=sorted(history_ids),
        history_items=history_items,
        created_at=_utc_now_iso(),
        history_counts={int(k): int(v) for k, v in sorted(history_counts.items())},
    )


def build_default_profiles(
    order_pool: OrderPool,
    count: int = 10,
    rng: random.Random | None = None,
) -> list[PersonProfile]:
    if count <= 0:
        return []

    generator = rng or random.Random()
    profiles: list[PersonProfile] = []
    seen_signatures: set[tuple[int, ...]] = set()
    attempts = max(40, count * 10)

    for _ in range(attempts):
        profile = build_random_profile(order_pool, rng=generator)
        if profile is None:
            continue
        signature = tuple(profile.history_product_ids)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        profiles.append(profile)
        if len(profiles) >= count:
            break

    return profiles


def _match_text_token(token: str, matcher: ProductMatcher) -> tuple[int | None, str | None]:
    norm = _normalise_text(token)
    if not norm:
        return None, None

    exact_ids = matcher.normalized_name_to_ids.get(norm)
    if exact_ids:
        pid = int(exact_ids[0])
        return pid, matcher.product_name_by_id.get(pid, token)

    close = get_close_matches(norm, matcher.normalized_names, n=1, cutoff=0.86)
    if not close:
        return None, None

    fuzzy_ids = matcher.normalized_name_to_ids.get(close[0], ())
    if not fuzzy_ids:
        return None, None
    pid = int(fuzzy_ids[0])
    return pid, matcher.product_name_by_id.get(pid, token)


@lru_cache(maxsize=200_000)
def _product_text_from_components(name: str, category: str, family: str) -> str:
    parts = (
        _normalise_text(name),
        _normalise_text(category),
        _normalise_text(family),
    )
    return " ".join(part for part in parts if part)


def _product_text_for_pid(pid: int, context: PersonalizationContext) -> str:
    return _product_text_from_components(
        str(context.product_name_by_id.get(int(pid), "")),
        str(context.category_by_id.get(int(pid), "")),
        str(context.product_family_by_id.get(int(pid), "")),
    )


def _hint_match_score(hint: str, text: str) -> float:
    hint_norm = _normalise_text(hint)
    text_norm = _normalise_text(text)
    if not hint_norm or not text_norm:
        return 0.0
    if hint_norm in text_norm:
        return 1.0
    hint_tokens = {token for token in hint_norm.split() if token}
    text_tokens = {token for token in text_norm.split() if token}
    if not hint_tokens or not text_tokens:
        return 0.0
    overlap = len(hint_tokens & text_tokens)
    if overlap <= 0:
        return 0.0
    return float(overlap / len(hint_tokens))


def build_manual_profile(raw_text: str, product_matcher: ProductMatcher) -> ManualProfileBuildResult:
    text = str(raw_text or "").strip()
    if not text:
        return ManualProfileBuildResult(profile=None, warnings=["No input provided."], matched_count=0)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ManualProfileBuildResult(profile=None, warnings=["No valid order lines found."], matched_count=0)

    warnings: list[str] = []
    history_ids: set[int] = set()
    history_counts: dict[int, int] = {}
    history_items: list[str] = []
    seen_items: set[str] = set()
    pseudo_order_ids: list[int] = []
    matched_count = 0

    for line_idx, line in enumerate(lines, start=1):
        tokens = [token.strip() for token in line.split(",") if token.strip()]
        if not tokens:
            continue
        pseudo_order_ids.append(-line_idx)
        for token in tokens:
            pid: int | None = None
            name: str | None = None

            if token.isdigit():
                pid_int = int(token)
                if pid_int in product_matcher.product_name_by_id:
                    pid = pid_int
                    name = product_matcher.product_name_by_id[pid_int]
                else:
                    warnings.append(f"Unknown product id: {token}")
                    continue
            else:
                pid, name = _match_text_token(token, product_matcher)
                if pid is None:
                    warnings.append(f"Unmatched item: {token}")
                    continue

            history_ids.add(int(pid))
            history_counts[int(pid)] = int(history_counts.get(int(pid), 0)) + 1
            matched_count += 1
            item_name = str(name or token).strip()
            if item_name and item_name not in seen_items:
                seen_items.add(item_name)
                history_items.append(item_name)

    if not history_ids:
        if not warnings:
            warnings.append("Could not match any products from the provided input.")
        return ManualProfileBuildResult(profile=None, warnings=warnings, matched_count=0)

    profile = PersonProfile(
        profile_id=_new_profile_id(),
        source="manual",
        order_ids=sorted(pseudo_order_ids),
        history_product_ids=sorted(history_ids),
        history_items=history_items,
        created_at=_utc_now_iso(),
        history_counts={int(k): int(v) for k, v in sorted(history_counts.items())},
    )
    return ManualProfileBuildResult(profile=profile, warnings=warnings, matched_count=matched_count)


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _safe_int(value: object, default: int = -1) -> int:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return default
    return int(num)


def _safe_float(value: object, default: float = 0.0) -> float:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return default
    return float(num)


def _confidence_from_score(score_value: float) -> float:
    """Map ranking score to a stable confidence display that avoids constant 100%."""
    score = float(score_value)
    bounded = 100.0 / (1.0 + math.exp(-2.0 * (score - 1.0)))
    return round(float(max(0.0, min(99.5, bounded))), 1)


def _safe_text_row(row: pd.Series | None, key: str) -> str:
    if row is None:
        return ""
    value = row.get(key, "")
    return str(value).strip()


def _safe_float_row(row: pd.Series | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    return _safe_float(row.get(key, default), default=default)


def _row_side_for_pid(row: pd.Series | None, pid: int, side: str | None = None) -> str | None:
    if row is None:
        return None
    pid_int = int(pid)
    preferred = side if side in {"a", "b"} else None
    if preferred is not None and _safe_int(row.get(f"product_{preferred}"), default=-1) == pid_int:
        return preferred
    for candidate in ("a", "b"):
        if _safe_int(row.get(f"product_{candidate}"), default=-1) == pid_int:
            return candidate
    return preferred


def _product_name(pid: int, context: PersonalizationContext, row: pd.Series | None = None, side: str | None = None) -> str:
    row_side = _row_side_for_pid(row, pid, side)
    if row_side in {"a", "b"} and row is not None:
        key = "product_a_name" if row_side == "a" else "product_b_name"
        row_name = _safe_text_row(row, key)
        if row_name:
            return row_name
    return str(context.product_name_by_id.get(int(pid), "")).strip()


def _product_category(pid: int, context: PersonalizationContext, row: pd.Series | None = None, side: str | None = None) -> str:
    row_side = _row_side_for_pid(row, pid, side)
    if row_side in {"a", "b"} and row is not None:
        key = "category_a" if row_side == "a" else "category_b"
        row_cat = _normalise_text(_safe_text_row(row, key))
        if row_cat:
            return row_cat
    return _normalise_text(context.category_by_id.get(int(pid), ""))


def _product_family(pid: int, context: PersonalizationContext, row: pd.Series | None = None, side: str | None = None) -> str:
    row_side = _row_side_for_pid(row, pid, side)
    if row_side in {"a", "b"} and row is not None:
        key = "product_family_a" if row_side == "a" else "product_family_b"
        row_family = _normalise_text(_safe_text_row(row, key))
        if row_family:
            return row_family
    return _normalise_text(context.product_family_by_id.get(int(pid), ""))


def _product_price(pid: int, context: PersonalizationContext, row: pd.Series | None = None, side: str | None = None) -> float:
    row_side = _row_side_for_pid(row, pid, side)
    if row_side in {"a", "b"} and row is not None:
        key = "product_a_price" if row_side == "a" else "product_b_price"
        row_price = _safe_float_row(row, key, default=0.0)
        if row_price > 0:
            return row_price
    return float(context.product_price_by_id.get(int(pid), 0.0))


def _semantic_product_text(pid: int, context: PersonalizationContext, row: pd.Series | None = None, side: str | None = None) -> tuple[str, str, str, str]:
    name = _product_name(pid, context, row=row, side=side)
    category = _product_category(pid, context, row=row, side=side)
    family = _product_family(pid, context, row=row, side=side)
    text = semantics.normalize_product_text(name, category, family)
    return name, category, family, text


def _semantic_roles_for_pid(pid: int, context: PersonalizationContext, row: pd.Series | None = None, side: str | None = None) -> frozenset[str]:
    name, category, family, _text = _semantic_product_text(pid, context, row=row, side=side)
    return semantics.infer_product_roles(name, category, family)


def _semantic_pair_snapshot(
    anchor_pid: int,
    complement_pid: int,
    lane: str,
    context: PersonalizationContext,
    pair_row: pd.Series | None = None,
) -> semantics.BundleSemantics:
    name_a, category_a, family_a, _text_a = _semantic_product_text(anchor_pid, context, row=pair_row, side="a")
    name_b, category_b, family_b, _text_b = _semantic_product_text(complement_pid, context, row=pair_row, side="b")
    normalized_lane = lane if lane in {semantics.LANE_MEAL, semantics.LANE_SNACK, semantics.LANE_OCCASION, semantics.LANE_STAPLES, semantics.LANE_NONFOOD} else semantics.LANE_MEAL
    return semantics.classify_bundle_semantics(
        lane=normalized_lane,
        name_a=name_a,
        category_a=category_a,
        family_a=family_a,
        name_b=name_b,
        category_b=category_b,
        family_b=family_b,
    )


def _semantic_visible_expression_ok(
    anchor_pid: int,
    complement_pid: int,
    lane: str,
    context: PersonalizationContext,
    pair_row: pd.Series | None = None,
) -> tuple[bool, str | None]:
    if lane not in FOOD_LANE_ORDER:
        return True, None
    sem = _semantic_pair_snapshot(int(anchor_pid), int(complement_pid), lane, context, pair_row=pair_row)
    _name_a, _cat_a, _fam_a, text_a = _semantic_product_text(int(anchor_pid), context, row=pair_row, side="a")
    _name_b, _cat_b, _fam_b, text_b = _semantic_product_text(int(complement_pid), context, row=pair_row, side="b")
    ok, reason = semantics.visible_lane_expression_ok(
        lane=lane,
        relation=str(sem.relation),
        strength=str(sem.strength),
        roles_a=frozenset(sem.roles_a),
        roles_b=frozenset(sem.roles_b),
        text_a=text_a,
        text_b=text_b,
    )
    return bool(ok), (str(reason) if reason else None)


def _nonfood_group(name: str, category: str, family: str) -> str:
    text = f"{_normalise_text(name)} {_normalise_text(category)} {_normalise_text(family)}"
    for group, hints in NONFOOD_GROUP_HINTS.items():
        if any(hint in text for hint in hints):
            return group
    return ""


def _is_nonfood_product(
    pid: int,
    context: PersonalizationContext,
    row: pd.Series | None = None,
    side: str | None = None,
) -> bool:
    if int(pid) in context.non_food_ids:
        return True
    category = _product_category(pid, context, row=row, side=side)
    if category == NON_FOOD_TAG:
        return True
    family = _product_family(pid, context, row=row, side=side)
    name = _product_name(pid, context, row=row, side=side)
    if _nonfood_group(name, category, family):
        return True
    text = f"{_normalise_text(name)} {_normalise_text(category)} {_normalise_text(family)}"
    return any(token in text for token in ("dishwashing", "detergent", "cleaner", "shampoo", "soap", "tissue", "bleach"))


def _is_appliance_or_tool_product(
    pid: int,
    context: PersonalizationContext,
    row: pd.Series | None = None,
    side: str | None = None,
) -> bool:
    name = _product_name(pid, context, row=row, side=side)
    category = _product_category(pid, context, row=row, side=side)
    family = _product_family(pid, context, row=row, side=side)
    text = f"{_normalise_text(name)} {_normalise_text(category)} {_normalise_text(family)}"
    return any(hint in text for hint in APPLIANCE_TOOL_HINTS)


def _product_domain(
    pid: int,
    context: PersonalizationContext,
    row: pd.Series | None = None,
    side: str | None = None,
) -> str:
    if _is_appliance_or_tool_product(pid, context, row=row, side=side):
        return "appliance"
    if _is_nonfood_product(pid, context, row=row, side=side):
        return "household"
    return "food"


def _is_packaging_or_utility_item(
    pid: int,
    context: PersonalizationContext,
    row: pd.Series | None = None,
    side: str | None = None,
) -> bool:
    name = _product_name(pid, context, row=row, side=side)
    category = _product_category(pid, context, row=row, side=side)
    family = _product_family(pid, context, row=row, side=side)
    text = f"{_normalise_text(name)} {_normalise_text(category)} {_normalise_text(family)}"
    tokens = _token_set(text)
    if tokens & PACKAGING_UTILITY_HINTS:
        return True
    return any(hint in text for hint in PACKAGING_UTILITY_HINTS)


def _is_fat_or_oil_item(name: str, category: str, family: str) -> bool:
    text = f"{_normalise_text(name)} {_normalise_text(category)} {_normalise_text(family)}"
    tokens = _token_set(text)
    if tokens & FAT_OIL_HINTS:
        return True
    return any(hint in text for hint in FAT_OIL_HINTS)


def _is_cheese_spread_item(name: str, category: str, family: str) -> bool:
    text = f"{_normalise_text(name)} {_normalise_text(category)} {_normalise_text(family)}"
    tokens = _token_set(text)
    if tokens & CHEESE_SPREAD_HINTS:
        return True
    return any(hint in text for hint in CHEESE_SPREAD_HINTS)


def _build_bundle_lookup(bundles_df: pd.DataFrame) -> dict[tuple[int, int], pd.Series]:
    lookup: dict[tuple[int, int], pd.Series] = {}
    if bundles_df.empty:
        return lookup
    ranked = bundles_df.copy()
    ranked["__score__"] = pd.to_numeric(ranked.get("final_score", 0.0), errors="coerce").fillna(0.0)
    ranked = ranked.sort_values("__score__", ascending=False)
    for _, row in ranked.iterrows():
        a = _safe_int(row.get("product_a"), default=-1)
        b = _safe_int(row.get("product_b"), default=-1)
        if a <= 0 or b <= 0 or a == b:
            continue
        key = _pair_key(a, b)
        if key not in lookup:
            lookup[key] = row
    return lookup


def _build_top_bundle_rows_by_anchor(bundles_df: pd.DataFrame) -> dict[int, list[pd.Series]]:
    by_anchor: dict[int, list[pd.Series]] = {}
    if bundles_df.empty:
        return by_anchor

    ranked = bundles_df.copy()
    ranked["__score__"] = pd.to_numeric(ranked.get("final_score", 0.0), errors="coerce").fillna(0.0)
    ranked = ranked.sort_values("__score__", ascending=False)

    for _, row in ranked.iterrows():
        a = _safe_int(row.get("product_a"), default=-1)
        b = _safe_int(row.get("product_b"), default=-1)
        if a <= 0 or b <= 0 or a == b:
            continue
        by_anchor.setdefault(a, []).append(row)
        by_anchor.setdefault(b, []).append(row)
    return by_anchor


def _build_display_dict_from_row(row: pd.Series) -> dict[str, object]:
    d = row.to_dict()

    product_a_price = float(pd.to_numeric(d.get("product_a_price", 0.0), errors="coerce") or 0.0)
    product_b_price = float(pd.to_numeric(d.get("product_b_price", 0.0), errors="coerce") or 0.0)
    d.setdefault("product_a_price", product_a_price)
    d.setdefault("product_b_price", product_b_price)

    d["price_a_sar"] = f"{product_a_price:,.2f}"
    d["price_b_sar"] = f"{product_b_price:,.2f}"

    discount_a = float(pd.to_numeric(d.get("discount_pred_a", d.get("discount_a", 12.0)), errors="coerce") or 0.0)
    discount_b = float(pd.to_numeric(d.get("discount_pred_b", d.get("discount_b", 12.0)), errors="coerce") or 0.0)
    d["discount_a"] = discount_a
    d["discount_b"] = discount_b
    d["discount_pred_a"] = discount_a
    d["discount_pred_b"] = discount_b

    free = str(d.get("free_product", d.get("free_product_raw", ""))).strip().lower()
    if free not in {"product_a", "product_b", "product_c"}:
        free = "product_a" if product_a_price <= product_b_price else "product_b"
    d["free_product"] = free

    after_a = 0.0 if free == "product_a" else max(0.0, product_a_price * (1 - discount_a / 100.0))
    after_b = 0.0 if free == "product_b" else max(0.0, product_b_price * (1 - discount_b / 100.0))
    d["price_after_a_sar"] = f"{after_a:,.2f}"
    d["price_after_b_sar"] = f"{after_b:,.2f}"

    d.setdefault("is_triple", bool(d.get("is_triple_bundle", False)))
    d.setdefault("product_c_name", "")
    d.setdefault("price_c_sar", "0.00")
    d.setdefault("price_after_c_sar", "0.00")
    d.setdefault("product_a_picture", str(d.get("product_a_picture", "") or ""))
    d.setdefault("product_b_picture", str(d.get("product_b_picture", "") or ""))
    d.setdefault("product_c_picture", str(d.get("product_c_picture", "") or ""))
    return d


def _build_constrained_pair_record(anchor: int, complement: int, context: PersonalizationContext) -> dict[str, object]:
    name_a = str(context.product_name_by_id.get(anchor, f"Product {anchor}"))
    name_b = str(context.product_name_by_id.get(complement, f"Product {complement}"))
    price_a = float(context.product_price_by_id.get(anchor, 0.0))
    price_b = float(context.product_price_by_id.get(complement, 0.0))

    if price_a <= 0 and price_b > 0:
        price_a = round(price_b * 1.2, 2)
    if price_b <= 0 and price_a > 0:
        price_b = round(price_a * 0.6, 2)
    if price_a <= 0 and price_b <= 0:
        price_a = 10.0
        price_b = 5.0

    free = "product_a" if price_a <= price_b else "product_b"
    discount_a = 12.0
    discount_b = 12.0

    after_a = 0.0 if free == "product_a" else max(0.0, price_a * (1 - discount_a / 100.0))
    after_b = 0.0 if free == "product_b" else max(0.0, price_b * (1 - discount_b / 100.0))

    return {
        "product_a": int(anchor),
        "product_b": int(complement),
        "product_a_name": name_a,
        "product_b_name": name_b,
        "product_a_price": round(price_a, 2),
        "product_b_price": round(price_b, 2),
        "price_a_sar": f"{price_a:,.2f}",
        "price_b_sar": f"{price_b:,.2f}",
        "price_after_a_sar": f"{after_a:,.2f}",
        "price_after_b_sar": f"{after_b:,.2f}",
        "free_product": free,
        "discount_a": discount_a,
        "discount_b": discount_b,
        "discount_pred_a": discount_a,
        "discount_pred_b": discount_b,
        "product_a_picture": str(context.product_picture_by_id.get(anchor, "") or ""),
        "product_b_picture": str(context.product_picture_by_id.get(complement, "") or ""),
        "product_c_name": "",
        "price_c_sar": "0.00",
        "price_after_c_sar": "0.00",
        "is_triple": False,
        "has_ramadan": 0,
    }


def _normalised_values(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    lo = min(values.values())
    hi = max(values.values())
    span = hi - lo
    if span <= 1e-9:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / span for k, v in values.items()}


def _token_set(text: str) -> set[str]:
    return {token for token in _normalise_text(text).split() if token}


def _text_has_any_hint(text: str, hints: frozenset[str]) -> bool:
    norm = _normalise_text(text)
    return any(hint in norm for hint in hints)


def _is_plain_milk_beverage_text(text: str) -> bool:
    norm = _normalise_text(text)
    if "milk" not in norm:
        return False
    if _text_has_any_hint(norm, FINAL_QUALITY_MILK_EXCLUSION_HINTS):
        return False
    return _text_has_any_hint(norm, FINAL_QUALITY_MILK_BLOCK_HINTS)


def _candidate_pair_fields(
    candidate: dict[str, object],
    context: PersonalizationContext,
) -> tuple[int, int, str, str, str, str, str, str, pd.Series | None]:
    anchor = int(_safe_int(candidate.get("anchor"), default=-1))
    complement = int(_safe_int(candidate.get("complement"), default=-1))
    row = candidate.get("bundle_row")
    pair_row = row if isinstance(row, pd.Series) else None
    name_a = _product_name(anchor, context, row=pair_row, side="a")
    name_b = _product_name(complement, context, row=pair_row, side="b")
    cat_a = _product_category(anchor, context, row=pair_row, side="a")
    cat_b = _product_category(complement, context, row=pair_row, side="b")
    fam_a = _product_family(anchor, context, row=pair_row, side="a")
    fam_b = _product_family(complement, context, row=pair_row, side="b")
    return anchor, complement, name_a, name_b, cat_a, cat_b, fam_a, fam_b, pair_row


def _pair_matches_hints(text_a: str, text_b: str, left_hints: frozenset[str], right_hints: frozenset[str]) -> bool:
    return bool(
        (_text_has_any_hint(text_a, left_hints) and _text_has_any_hint(text_b, right_hints))
        or (_text_has_any_hint(text_b, left_hints) and _text_has_any_hint(text_a, right_hints))
    )


def _candidate_family_pattern_signature(candidate: dict[str, object], context: PersonalizationContext) -> str:
    raw = candidate.get("pair_fingerprint")
    if isinstance(raw, tuple) and len(raw) == 2 and str(raw[0]).strip():
        lane = str(candidate.get("lane", "")).strip().lower() or LANE_MEAL
        return f"{lane}:{str(raw[0]).strip().lower()}"
    (
        _anchor,
        _complement,
        _name_a,
        _name_b,
        _cat_a,
        _cat_b,
        fam_a,
        fam_b,
        _pair_row,
    ) = _candidate_pair_fields(candidate, context)
    fam_pair = tuple(sorted((str(fam_a).strip().lower() or "na", str(fam_b).strip().lower() or "nb")))
    lane = str(candidate.get("lane", "")).strip().lower() or LANE_MEAL
    return f"{lane}:{fam_pair[0]}::{fam_pair[1]}"


def _candidate_bundle_shape_signature(candidate: dict[str, object], context: PersonalizationContext) -> str:
    raw = candidate.get("pair_fingerprint")
    lane = str(candidate.get("lane", "")).strip().lower() or LANE_MEAL
    theme = str(candidate.get("theme", "")).strip().lower() or "none"
    relation = str(candidate.get("pair_relation", "")).strip().lower() or "none"
    if isinstance(raw, tuple) and len(raw) == 2 and str(raw[1]).strip():
        return f"{lane}:{str(raw[1]).strip().lower()}:{theme}:{relation}"
    (
        anchor,
        complement,
        name_a,
        name_b,
        cat_a,
        cat_b,
        fam_a,
        fam_b,
        pair_row,
    ) = _candidate_pair_fields(candidate, context)
    groups_a = _group_labels_from_text(name_a, cat_a, fam_a)
    groups_b = _group_labels_from_text(name_b, cat_b, fam_b)
    sem_a = _product_semantic_group(anchor, context, row=pair_row, side="a")
    sem_b = _product_semantic_group(complement, context, row=pair_row, side="b")
    group_pair = tuple(sorted((_primary_group(groups_a), _primary_group(groups_b))))
    sem_pair = tuple(sorted((str(sem_a), str(sem_b))))
    return f"{lane}:{group_pair[0]}::{group_pair[1]}:{sem_pair[0]}::{sem_pair[1]}:{theme}:{relation}"


def _fallback_shopper_family_signature(
    lane: str,
    groups_a: set[str],
    groups_b: set[str],
    relation: str,
) -> str:
    primary_a = str(_primary_group(groups_a))
    primary_b = str(_primary_group(groups_b))
    group_pair = tuple(sorted((primary_a, primary_b)))
    rel = str(relation).strip().lower() or "pair"
    lane_key = str(lane).strip().lower() or LANE_MEAL
    pair_set = {primary_a, primary_b}
    if lane_key == LANE_MEAL:
        if GROUP_RICE_GRAINS in pair_set and GROUP_PROTEIN in pair_set:
            return "meal:protein_grain_meal"
        if GROUP_BREAD_CARB in pair_set and GROUP_PROTEIN in pair_set:
            return "meal:protein_bread_meal"
        if GROUP_NOODLES_PASTA in pair_set and GROUP_PROTEIN in pair_set:
            return "meal:protein_noodles_meal"
        if GROUP_PRODUCE in pair_set and GROUP_PROTEIN in pair_set:
            return "meal:produce_protein_meal"
        if GROUP_RICE_GRAINS in pair_set and GROUP_PRODUCE in pair_set:
            return "meal:produce_grain_meal"
        if GROUP_PROTEIN in pair_set and (GROUP_RICE_GRAINS in pair_set or GROUP_BREAD_CARB in pair_set or GROUP_NOODLES_PASTA in pair_set):
            return "meal:protein_starch_generic_meal"
    if lane_key == LANE_SNACK:
        beverage_groups = {GROUP_TEA, GROUP_COFFEE, GROUP_SODA, GROUP_JUICE, GROUP_BEVERAGES}
        snack_groups = {GROUP_CHIPS, GROUP_CRACKERS, GROUP_COOKIES, GROUP_CHOCOLATE, GROUP_CANDY, GROUP_NUTS, GROUP_SNACKS}
        dairy_groups = {GROUP_DAIRY, GROUP_MILK, GROUP_CREAM, GROUP_CREAM_CHEESE, GROUP_CHEESE}
        dessert_groups = {GROUP_COOKIES, GROUP_CHOCOLATE, GROUP_CANDY, GROUP_DATES, GROUP_SWEETS}
        union = set(groups_a) | set(groups_b)
        if union & snack_groups and union & beverage_groups:
            return "snack:drink_snack_pair"
        if union & snack_groups and union & dairy_groups:
            return "snack:snack_dairy_pair"
        if union & dessert_groups and union & dairy_groups:
            return "snack:dessert_dairy_pair"
    if lane_key == LANE_OCCASION:
        beverage_groups = {GROUP_TEA, GROUP_COFFEE, GROUP_BEVERAGES}
        dairy_groups = {GROUP_DAIRY, GROUP_MILK, GROUP_CREAM, GROUP_CREAM_CHEESE, GROUP_CHEESE}
        union = set(groups_a) | set(groups_b)
        if union & beverage_groups and union & dairy_groups:
            return "occasion:beverage_dairy_treat"
        if GROUP_DATES in union and union & dairy_groups:
            return "occasion:dates_dairy_treat"
    return f"{lane_key}:{group_pair[0]}_{group_pair[1]}_{rel}_family"


def _shopper_visible_family_signature(
    lane: str,
    text_a: str,
    text_b: str,
    groups_a: set[str],
    groups_b: set[str],
) -> str:
    lane_key = str(lane).strip().lower()
    if lane_key == LANE_MEAL:
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_NUGGETS, HUMAN_HINTS_BREAD | HUMAN_HINTS_FRIES):
            return "meal:nuggets_bread_fastmeal"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_MINCED_MEAT, HUMAN_HINTS_TORTILLA | HUMAN_HINTS_BREAD):
            return "meal:minced_meat_wrap_meal"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHICKEN, HUMAN_HINTS_TORTILLA | HUMAN_HINTS_BREAD):
            return "meal:chicken_wrap_meal"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_LABNEH, HUMAN_HINTS_BREAD):
            return "meal:labneh_bread_meal"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_EGGS, HUMAN_HINTS_BREAD):
            return "meal:egg_breakfast_meal"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_RICE, HUMAN_HINTS_EGGS):
            return "meal:rice_egg_meal"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_RICE, HUMAN_HINTS_CHICKEN):
            return "meal:rice_chicken_meal"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_RICE, HUMAN_HINTS_MEAT | HUMAN_HINTS_FISH):
            return "meal:rice_meat_meal"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHICKEN, MOTIF_HINTS_TOMATO_BASE):
            return "meal:chicken_tomato_meal"

        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_EGGS, HUMAN_HINTS_PLAIN_COOKING_FAT):
            return "meal:egg_fat_utilitarian"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_PLAIN_COOKING_FAT, HUMAN_HINTS_SPICE_SEASONING):
            return "meal:fat_spice_utilitarian"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHICKEN, HUMAN_HINTS_PLAIN_COOKING_FAT):
            return "meal:chicken_fat_utilitarian"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_PROTEIN_SAVORY, HUMAN_HINTS_PLAIN_COOKING_FAT):
            return "meal:protein_plain_fat_utilitarian"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_FAT_BASE, HUMAN_HINTS_PROTEIN_SAVORY):
            return "meal:protein_oil_utilitarian"
        return ""

    if lane_key == LANE_SNACK:
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_LABNEH, HUMAN_HINTS_CHIPS | MOTIF_HINTS_CRUNCHY_SNACK):
            return "snack:labneh_crunchy_snack"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHEESE, frozenset({"cracker", "crackers", "biscuit", "biscuits"})):
            return "snack:cheese_cracker_snack"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_NUTELLA, HUMAN_HINTS_BREAD | HUMAN_HINTS_BISCUITS):
            return "snack:nutella_snack_pair"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_BISCUITS, MOTIF_HINTS_MILK_TEA):
            return "snack:biscuit_milk_tea_snack"
        if _pair_matches_hints(text_a, text_b, frozenset({"wafer", "wafer biscuit", "cookies", "biscrem"}), HUMAN_HINTS_CHOCOLATE):
            return "snack:wafer_chocolate_snack"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_BISCUITS, HUMAN_HINTS_CHOCOLATE):
            return "snack:wafer_chocolate_snack"
        if (set(groups_a) | set(groups_b)) & {GROUP_CHIPS, GROUP_SNACKS} and (set(groups_a) | set(groups_b)) & {
            GROUP_SODA,
            GROUP_JUICE,
            GROUP_TEA,
            GROUP_COFFEE,
            GROUP_BEVERAGES,
        }:
            return "snack:drink_snack_pair"
        if (set(groups_a) | set(groups_b)) & {GROUP_SNACKS, GROUP_CHIPS, GROUP_COOKIES, GROUP_CHOCOLATE} and (
            (set(groups_a) | set(groups_b))
            & {GROUP_DAIRY, GROUP_MILK, GROUP_CREAM, GROUP_CREAM_CHEESE, GROUP_CHEESE}
        ):
            return "snack:snack_dairy_pair"
        return ""

    if lane_key == LANE_OCCASION:
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_DATES, HUMAN_HINTS_CREAM_TOKEN):
            return "occasion:dates_cream_treat"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_DATES, HUMAN_HINTS_MILK | HUMAN_HINTS_EVAP_MILK | HUMAN_HINTS_CONDENSED_MILK):
            return "occasion:dates_milk_treat"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_COFFEE, HUMAN_HINTS_MILK | HUMAN_HINTS_EVAP_MILK):
            return "occasion:coffee_milk_drink"
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_TEA, HUMAN_HINTS_MILK | HUMAN_HINTS_EVAP_MILK):
            return "occasion:tea_milk_drink"
        if _pair_matches_hints(text_a, text_b, FINAL_QUALITY_DESSERT_HINTS, HUMAN_HINTS_CREAM_TOKEN):
            return "occasion:dessert_cream_treat"
        union = set(groups_a) | set(groups_b)
        if union & {GROUP_TEA, GROUP_COFFEE, GROUP_BEVERAGES} and union & {
            GROUP_DAIRY,
            GROUP_MILK,
            GROUP_CREAM,
            GROUP_CREAM_CHEESE,
            GROUP_CHEESE,
        }:
            return "occasion:beverage_dairy_treat"
        if GROUP_DATES in union and union & {GROUP_DAIRY, GROUP_MILK, GROUP_CREAM, GROUP_CREAM_CHEESE, GROUP_CHEESE}:
            return "occasion:dates_dairy_treat"
        return ""

    if lane_key == LANE_NONFOOD:
        return "nonfood:household_bundle"

    del groups_a, groups_b
    return ""


def _candidate_motif_family_signature(candidate: dict[str, object], context: PersonalizationContext) -> str:
    existing = str(candidate.get("motif_family_signature", "")).strip().lower()
    if existing:
        return existing
    (
        _anchor,
        _complement,
        name_a,
        name_b,
        cat_a,
        cat_b,
        fam_a,
        fam_b,
        _pair_row,
    ) = _candidate_pair_fields(candidate, context)
    lane = str(candidate.get("lane", "")).strip().lower() or LANE_MEAL
    text_a = semantics.normalize_product_text(name_a, cat_a, fam_a)
    text_b = semantics.normalize_product_text(name_b, cat_b, fam_b)
    groups_a = _group_labels_from_text(name_a, cat_a, fam_a)
    groups_b = _group_labels_from_text(name_b, cat_b, fam_b)
    shopper_family = _shopper_visible_family_signature(lane, text_a, text_b, groups_a, groups_b)
    if shopper_family:
        return str(shopper_family)
    relation = str(candidate.get("pair_relation", "")).strip().lower()
    return _fallback_shopper_family_signature(lane, groups_a, groups_b, relation)


def _is_utilitarian_shopper_family_signature(signature: str) -> bool:
    sig = str(signature or "").strip().lower()
    if not sig:
        return False
    if sig in SHOPPER_FAMILY_UTILITARIAN:
        return True
    if "utilitarian" in sig:
        return True
    if sig.startswith(f"{LANE_MEAL}:") and ("_plain_fat_" in sig or "_fat_" in sig):
        return True
    return False


def _is_dominant_shopper_family_signature(signature: str) -> bool:
    sig = str(signature or "").strip().lower()
    if not sig:
        return False
    if (
        sig in SHOPPER_FAMILY_MEAL_DOMINANT
        or sig in SHOPPER_FAMILY_SNACK_DOMINANT
        or sig in SHOPPER_FAMILY_OCCASION_DOMINANT
    ):
        return True
    if sig.startswith(f"{LANE_MEAL}:"):
        if "utilitarian" in sig:
            return True
        if any(
            token in sig
            for token in (
                "rice_meat_meal",
                "rice_chicken_meal",
                "protein_grain_meal",
                "protein_starch_generic_meal",
                "chicken_tomato_meal",
            )
        ):
            return True
    return False


def _is_meal_dominant_motif_signature(signature: str) -> bool:
    sig = str(signature or "").strip().lower()
    if not sig or not sig.startswith(f"{LANE_MEAL}:"):
        return False
    if _is_dominant_shopper_family_signature(sig):
        return True
    return any(keyword in sig for keyword in MEAL_DOMINANT_MOTIF_KEYWORDS)


def _is_meal_dominant_pair_text(text_a: str, text_b: str) -> bool:
    rice_meat = _pair_matches_hints(text_a, text_b, HUMAN_HINTS_RICE, HUMAN_HINTS_MEAT | HUMAN_HINTS_CHICKEN | HUMAN_HINTS_FISH)
    rice_eggs = _pair_matches_hints(text_a, text_b, HUMAN_HINTS_RICE, HUMAN_HINTS_EGGS)
    chicken_oil = _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHICKEN, HUMAN_HINTS_OLIVE_OIL | frozenset({"oil", "ghee"}))
    chicken_tomato = _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHICKEN, HUMAN_HINTS_TOMATO_PASTE)
    return bool(rice_meat or rice_eggs or chicken_oil or chicken_tomato)


def _progressive_exposure_penalty(count: int, base_penalty: float, threshold: int) -> float:
    exposure = max(0, int(count))
    if exposure <= 0 or float(base_penalty) <= 0.0:
        return 0.0
    linear = float(base_penalty) * float(exposure)
    over = max(0, exposure - int(threshold))
    if over <= 0:
        return linear
    surge = float(base_penalty) * EXPOSURE_SURGE_MULTIPLIER * float(over ** EXPOSURE_SURGE_POWER)
    return float(linear + surge)


def _human_preference_score_adjustment(candidate: dict[str, object], context: PersonalizationContext) -> float:
    (
        _anchor,
        _complement,
        name_a,
        name_b,
        cat_a,
        cat_b,
        fam_a,
        fam_b,
        _pair_row,
    ) = _candidate_pair_fields(candidate, context)
    lane = str(candidate.get("lane", "")).strip().lower()
    text_a = semantics.normalize_product_text(name_a, cat_a, fam_a)
    text_b = semantics.normalize_product_text(name_b, cat_b, fam_b)
    adjustment = 0.0

    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_RICE, FINAL_QUALITY_TUNA_HINTS):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["rice_tuna"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_OLIVE_OIL | frozenset({"oil", "ghee"}), HUMAN_HINTS_STOCK):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["meal_utilitarian_stock_oil"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHICKEN, HUMAN_HINTS_TOMATO_PASTE):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["chicken_tomato_paste"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHICKEN, HUMAN_HINTS_OLIVE_OIL):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["chicken_olive_oil"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_EGGS, HUMAN_HINTS_TOMATO_PASTE):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["eggs_tomato_paste"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_EGGS, HUMAN_HINTS_FETA):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["eggs_feta"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_LABNEH, HUMAN_HINTS_CHIPS):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["labneh_chips"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_DATES, HUMAN_HINTS_EVAP_MILK):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["dates_evap_milk"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_BISCUITS, HUMAN_HINTS_COCONUT_MILK):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["biscuits_coconut_milk"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_MILK, HUMAN_HINTS_COCOA_POWDER):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["milk_cocoa_powder"])
    if _pair_matches_hints(text_a, text_b, FINAL_QUALITY_DESSERT_HINTS, FINAL_QUALITY_BISCUIT_PLAIN_HINTS):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["dessert_plain_biscuit"])
    if (
        lane == LANE_OCCASION
        and _pair_matches_hints(text_a, text_b, FINAL_QUALITY_NUTELLA_HINTS, HUMAN_HINTS_MILK)
        and (_is_plain_milk_beverage_text(text_a) or _is_plain_milk_beverage_text(text_b))
    ):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["nutella_plain_milk_occasion"])
    if (
        _text_has_any_hint(text_a, HUMAN_HINTS_CREAM_TOKEN)
        and _text_has_any_hint(text_b, HUMAN_HINTS_CREAM_TOKEN)
        and (_text_has_any_hint(text_a, FINAL_QUALITY_DESSERT_HINTS) or _text_has_any_hint(text_b, FINAL_QUALITY_DESSERT_HINTS))
    ):
        adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["cream_dessert_duplication"])

    if lane == LANE_MEAL:
        fat_protein_pair = _pair_matches_hints(
            text_a,
            text_b,
            HUMAN_HINTS_FAT_BASE,
            HUMAN_HINTS_CHICKEN | HUMAN_HINTS_MEAT | HUMAN_HINTS_FISH,
        )
        fat_egg_pair = _pair_matches_hints(text_a, text_b, HUMAN_HINTS_FAT_BASE, HUMAN_HINTS_EGGS)
        has_real_dish_pattern = bool(
            _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHICKEN, HUMAN_HINTS_BREAD | HUMAN_HINTS_TORTILLA)
            or _pair_matches_hints(text_a, text_b, HUMAN_HINTS_RICE, HUMAN_HINTS_MEAT | HUMAN_HINTS_EGGS)
            or _pair_matches_hints(text_a, text_b, MOTIF_HINTS_TOMATO_BASE, HUMAN_HINTS_CHICKEN)
            or _pair_matches_hints(text_a, text_b, HUMAN_HINTS_MINCED_MEAT, HUMAN_HINTS_TORTILLA | HUMAN_HINTS_BREAD)
            or _pair_matches_hints(text_a, text_b, HUMAN_HINTS_SPRING_ROLL, HUMAN_HINTS_CHEESE)
        )
        if fat_protein_pair and not has_real_dish_pattern:
            adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["utilitarian_fat_protein_meal"])
        if fat_egg_pair and not has_real_dish_pattern:
            adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["utilitarian_egg_fat_meal"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_RICE, HUMAN_HINTS_MEAT | HUMAN_HINTS_CHICKEN):
            adjustment -= float(HUMAN_SOFT_PENALTY_BY_PATTERN["meal_rice_meat_overused"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_RICE, HUMAN_HINTS_MEAT):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["meal_rice_meat"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_RICE, HUMAN_HINTS_EGGS):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["meal_rice_eggs"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHICKEN, HUMAN_HINTS_BREAD | HUMAN_HINTS_TORTILLA):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["meal_chicken_bread_tortilla"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_EGGS, HUMAN_HINTS_BREAD):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["meal_eggs_bread"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_EGGS, HUMAN_HINTS_LABNEH):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["meal_eggs_labneh"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_SPRING_ROLL, HUMAN_HINTS_CHEESE):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["meal_spring_roll_cheese"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_MINCED_MEAT, HUMAN_HINTS_TORTILLA | HUMAN_HINTS_BREAD):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["meal_minced_meat_wrap"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_LABNEH, HUMAN_HINTS_BREAD):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["meal_labneh_bread"])
    if lane == LANE_SNACK:
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_BISCUITS, HUMAN_HINTS_MILK):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["snack_biscuits_milk"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_BISCUITS, HUMAN_HINTS_CHOCOLATE):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["snack_biscuits_chocolate"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_NUTELLA, HUMAN_HINTS_BREAD):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["snack_nutella_bread"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_CHOCOLATE, HUMAN_HINTS_MILK):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["snack_chocolate_milk"])
        if str(candidate.get("snack_pattern", "")).strip():
            adjustment += float(HUMAN_BOOST_BY_PATTERN["snack_real_pattern_bonus"])
    if lane == LANE_OCCASION:
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_TEA, HUMAN_HINTS_EVAP_MILK):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["occasion_tea_evap_milk"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_COFFEE, HUMAN_HINTS_EVAP_MILK):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["occasion_coffee_evap_milk"])
        if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_DATES, HUMAN_HINTS_CONDENSED_MILK):
            adjustment += float(HUMAN_BOOST_BY_PATTERN["occasion_dates_condensed_milk"])

    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_NUGGETS, HUMAN_HINTS_BREAD):
        adjustment += float(HUMAN_BOOST_BY_PATTERN["fastfood_nuggets_bread"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_NUGGETS, HUMAN_HINTS_FRIES):
        adjustment += float(HUMAN_BOOST_BY_PATTERN["fastfood_nuggets_fries"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_BURGER, HUMAN_HINTS_BREAD):
        adjustment += float(HUMAN_BOOST_BY_PATTERN["fastfood_burger_bread"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_BURGER, HUMAN_HINTS_CHEESE):
        adjustment += float(HUMAN_BOOST_BY_PATTERN["fastfood_burger_cheese"])
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_BURGER, HUMAN_HINTS_SAUCE):
        adjustment += float(HUMAN_BOOST_BY_PATTERN["fastfood_burger_sauce"])

    return float(adjustment)


def _controlled_fallback_motif_key(candidate: dict[str, object], context: PersonalizationContext) -> str:
    source_group = _source_group_from_source(str(candidate.get("source", "")))
    if source_group != "fallback_food":
        return ""
    (
        _anchor,
        _complement,
        name_a,
        name_b,
        cat_a,
        cat_b,
        fam_a,
        fam_b,
        _pair_row,
    ) = _candidate_pair_fields(candidate, context)
    text_a = semantics.normalize_product_text(name_a, cat_a, fam_a)
    text_b = semantics.normalize_product_text(name_b, cat_b, fam_b)
    pair_text = f"{text_a}::{text_b}"
    if _text_has_any_hint(pair_text, frozenset({"tea"})) and _text_has_any_hint(pair_text, frozenset({"evaporated milk"})):
        return FALLBACK_MOTIF_KEY_TEA_EVAP
    if _text_has_any_hint(pair_text, frozenset({"coffee"})) and _text_has_any_hint(pair_text, frozenset({"evaporated milk"})):
        return FALLBACK_MOTIF_KEY_COFFEE_EVAP
    if _text_has_any_hint(pair_text, frozenset({"dates"})) and _text_has_any_hint(pair_text, frozenset({"evaporated milk"})):
        return FALLBACK_MOTIF_KEY_DATES_EVAP
    if _text_has_any_hint(pair_text, frozenset({"dates"})) and _text_has_any_hint(pair_text, frozenset({"condensed milk"})):
        return FALLBACK_MOTIF_KEY_DATES_COND
    return ""


def _final_human_quality_reject_reason(
    anchor: int,
    complement: int,
    lane: str,
    context: PersonalizationContext,
    *,
    pair_row: pd.Series | None = None,
) -> str | None:
    if int(anchor) <= 0 or int(complement) <= 0 or int(anchor) == int(complement):
        return "invalid_pair"
    if lane not in FOOD_LANE_ORDER:
        return None
    analysis = _pair_analysis(int(anchor), int(complement), lane, context, pair_row=pair_row)
    text_a = semantics.normalize_product_text(analysis.anchor_name, analysis.anchor_category, analysis.anchor_family)
    text_b = semantics.normalize_product_text(analysis.complement_name, analysis.complement_category, analysis.complement_family)
    text_pair = f"{text_a}::{text_b}"

    tuna_a = _text_has_any_hint(text_a, FINAL_QUALITY_TUNA_HINTS)
    tuna_b = _text_has_any_hint(text_b, FINAL_QUALITY_TUNA_HINTS)
    rice_family_a = _text_has_any_hint(text_a, FINAL_QUALITY_RICE_FAMILY_HINTS) and not _text_has_any_hint(
        text_a, FINAL_QUALITY_CRACKER_HINTS
    )
    rice_family_b = _text_has_any_hint(text_b, FINAL_QUALITY_RICE_FAMILY_HINTS) and not _text_has_any_hint(
        text_b, FINAL_QUALITY_CRACKER_HINTS
    )
    pasta_family_a = _text_has_any_hint(text_a, FINAL_QUALITY_PASTA_FAMILY_HINTS)
    pasta_family_b = _text_has_any_hint(text_b, FINAL_QUALITY_PASTA_FAMILY_HINTS)
    if (tuna_a and (rice_family_b or pasta_family_b)) or (tuna_b and (rice_family_a or pasta_family_a)):
        return "tuna_with_rice_or_pasta_pair"
    milk_a = _is_plain_milk_beverage_text(text_a)
    milk_b = _is_plain_milk_beverage_text(text_b)
    if (tuna_a and milk_b) or (tuna_b and milk_a):
        return "tuna_milk_pair"
    peanut_a = _text_has_any_hint(text_a, FINAL_QUALITY_PEANUT_BUTTER_HINTS)
    peanut_b = _text_has_any_hint(text_b, FINAL_QUALITY_PEANUT_BUTTER_HINTS)
    if (tuna_a and peanut_b) or (tuna_b and peanut_a):
        return "tuna_peanut_butter_pair"

    noodles_a = _text_has_any_hint(text_a, FINAL_QUALITY_NOODLE_HINTS)
    noodles_b = _text_has_any_hint(text_b, FINAL_QUALITY_NOODLE_HINTS)
    baking_a = _text_has_any_hint(text_a, FINAL_QUALITY_BAKING_HINTS)
    baking_b = _text_has_any_hint(text_b, FINAL_QUALITY_BAKING_HINTS)
    if (noodles_a and baking_b) or (noodles_b and baking_a):
        return "noodles_baking_pair"
    spicy_a = _text_has_any_hint(text_a, frozenset({"spicy"}))
    spicy_b = _text_has_any_hint(text_b, frozenset({"spicy"}))
    if ((noodles_a and spicy_a) and baking_b) or ((noodles_b and spicy_b) and baking_a):
        return "spicy_noodles_baking_pair"
    savory_protein_baking = _pair_matches_hints(
        text_a,
        text_b,
        HUMAN_HINTS_CHICKEN | HUMAN_HINTS_MEAT | HUMAN_HINTS_FISH,
        FINAL_QUALITY_BAKING_HINTS,
    )
    egg_baking_pair = _pair_matches_hints(text_a, text_b, HUMAN_HINTS_EGGS, FINAL_QUALITY_BAKING_HINTS)
    if savory_protein_baking and not egg_baking_pair:
        return "savory_protein_baking_pair"

    stock_a = _text_has_any_hint(text_a, FINAL_QUALITY_STOCK_HINTS)
    stock_b = _text_has_any_hint(text_b, FINAL_QUALITY_STOCK_HINTS)
    stock_seasoning_a = _text_has_any_hint(text_a, FINAL_QUALITY_STOCK_SEASONING_HINTS)
    stock_seasoning_b = _text_has_any_hint(text_b, FINAL_QUALITY_STOCK_SEASONING_HINTS)
    mayo_a = _text_has_any_hint(text_a, FINAL_QUALITY_MAYO_HINTS)
    mayo_b = _text_has_any_hint(text_b, FINAL_QUALITY_MAYO_HINTS)
    if (stock_a and mayo_b) or (stock_b and mayo_a):
        return "stock_mayo_pair"

    oil_a = _is_fat_or_oil_item(analysis.anchor_name, analysis.anchor_category, analysis.anchor_family)
    oil_b = _is_fat_or_oil_item(analysis.complement_name, analysis.complement_category, analysis.complement_family)
    if (oil_a and stock_seasoning_b) or (oil_b and stock_seasoning_a):
        return "oil_stock_seasoning_pair"
    nugget_a = _text_has_any_hint(text_a, FINAL_QUALITY_NUGGET_HINTS)
    nugget_b = _text_has_any_hint(text_b, FINAL_QUALITY_NUGGET_HINTS)
    if (oil_a and nugget_b) or (oil_b and nugget_a):
        return "oil_nuggets_pair"

    instant_noodle_a = _text_has_any_hint(text_a, FINAL_QUALITY_INSTANT_NOODLE_HINTS)
    instant_noodle_b = _text_has_any_hint(text_b, FINAL_QUALITY_INSTANT_NOODLE_HINTS)
    if (oil_a and instant_noodle_b) or (oil_b and instant_noodle_a):
        return "oil_instant_noodles_pair"

    cream_cheese_a = _text_has_any_hint(text_a, FINAL_QUALITY_CREAM_CHEESE_HINTS)
    cream_cheese_b = _text_has_any_hint(text_b, FINAL_QUALITY_CREAM_CHEESE_HINTS)
    dessert_a = _text_has_any_hint(text_a, FINAL_QUALITY_DESSERT_HINTS)
    dessert_b = _text_has_any_hint(text_b, FINAL_QUALITY_DESSERT_HINTS)
    if (cream_cheese_a and dessert_b) or (cream_cheese_b and dessert_a):
        return "cream_cheese_dessert_pair"
    cake_a = _text_has_any_hint(text_a, frozenset({"cake"}))
    cake_b = _text_has_any_hint(text_b, frozenset({"cake"}))
    if (cake_a and cream_cheese_b and dessert_a) or (cake_b and cream_cheese_a and dessert_b):
        return "cake_cream_cheese_duplication"
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_EGGS, HUMAN_HINTS_TOMATO_PASTE):
        return "eggs_tomato_paste_pair"
    if _pair_matches_hints(text_a, text_b, FINAL_QUALITY_DESSERT_HINTS, FINAL_QUALITY_BISCUIT_PLAIN_HINTS):
        return "dessert_plain_biscuit_pair"
    if ((dessert_a and milk_b) or (dessert_b and milk_a)) and not _pair_matches_hints(
        text_a,
        text_b,
        HUMAN_HINTS_CHOCOLATE,
        HUMAN_HINTS_MILK,
    ):
        return "dessert_plain_milk_pair"
    if (
        lane == LANE_OCCASION
        and _pair_matches_hints(text_a, text_b, FINAL_QUALITY_NUTELLA_HINTS, HUMAN_HINTS_MILK)
        and (_is_plain_milk_beverage_text(text_a) or _is_plain_milk_beverage_text(text_b))
    ):
        return "nutella_plain_milk_occasion_pair"

    protein_milk = (
        (semantics.ROLE_PROTEIN in analysis.anchor_roles and milk_b)
        or (semantics.ROLE_PROTEIN in analysis.complement_roles and milk_a)
    )
    if protein_milk:
        relation = str(analysis.semantic.relation)
        strength = str(analysis.semantic.strength)
        allows_display = bool(
            lane in {LANE_SNACK, LANE_OCCASION}
            and relation in {semantics.REL_DRINK, semantics.REL_DESSERT}
            and strength == semantics.STRENGTH_STRONG
        )
        if not allows_display:
            return "protein_dairy_beverage_pair"

    savory_ready_a = _text_has_any_hint(text_a, FINAL_QUALITY_SAVORY_READY_HINTS)
    savory_ready_b = _text_has_any_hint(text_b, FINAL_QUALITY_SAVORY_READY_HINTS)
    if (savory_ready_a and baking_b) or (savory_ready_b and baking_a):
        return "savory_ready_baking_pair"

    if _text_has_any_hint(text_pair, frozenset({"stock cube", "bouillon"})) and _text_has_any_hint(
        text_pair, frozenset({"garlic mayo", "mayonnaise", "mayo"})
    ):
        return "stock_mayo_pair"

    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_MILK, HUMAN_HINTS_COCOA_POWDER):
        return "milk_cocoa_powder_pair"
    if _pair_matches_hints(text_a, text_b, HUMAN_HINTS_BISCUITS, HUMAN_HINTS_COCONUT_MILK):
        return "biscuits_coconut_milk_pair"
    if (
        _text_has_any_hint(text_a, HUMAN_HINTS_CREAM_TOKEN)
        and _text_has_any_hint(text_b, HUMAN_HINTS_CREAM_TOKEN)
        and (_text_has_any_hint(text_a, FINAL_QUALITY_DESSERT_HINTS) or _text_has_any_hint(text_b, FINAL_QUALITY_DESSERT_HINTS))
    ):
        return "cream_dessert_duplication_pair"

    return None


def _fallback_quality_reject_reason(candidate: dict[str, object], context: PersonalizationContext) -> str | None:
    source_group = _source_group_from_source(str(candidate.get("source", "")))
    if source_group != "fallback_food":
        return None
    lane = str(candidate.get("lane", "")).strip().lower()
    if lane not in FOOD_LANE_ORDER:
        return None
    strength = str(candidate.get("pair_strength", "")).strip()
    cp_score = float(candidate.get("cp_score", 0.0))
    pair_count = int(_safe_int(candidate.get("pair_count"), default=0))
    lane_fit = float(candidate.get("lane_fit_score", 0.0))
    template_strength = float(candidate.get("template_strength", 0.0))
    category_strength = float(candidate.get("category_strength", 0.0))
    recipe = float(candidate.get("recipe_compat", 0.0))
    prior_bonus = float(candidate.get("prior_bonus", 0.0))

    if strength == semantics.STRENGTH_TRASH or strength == semantics.STRENGTH_WEAK:
        return "fallback_weak_strength"
    if lane in {LANE_SNACK, LANE_OCCASION} and strength != semantics.STRENGTH_STRONG:
        return "fallback_non_strong_in_snack_occasion"
    if cp_score < float(STRICT_FALLBACK_CP_MIN.get(lane, 0.0)):
        return "fallback_cp_too_low"
    if pair_count < int(STRICT_FALLBACK_PAIR_COUNT_MIN.get(lane, 0)):
        return "fallback_pair_count_too_low"
    if lane_fit < float(STRICT_FALLBACK_LANE_FIT_MIN.get(lane, 0.0)):
        return "fallback_lane_fit_too_low"
    if template_strength < float(STRICT_FALLBACK_TEMPLATE_MIN.get(lane, 0.0)) and category_strength < 0.66:
        return "fallback_template_quality_too_low"

    (
        _anchor,
        _complement,
        name_a,
        name_b,
        cat_a,
        cat_b,
        fam_a,
        fam_b,
        _pair_row,
    ) = _candidate_pair_fields(candidate, context)
    text_a = semantics.normalize_product_text(name_a, cat_a, fam_a)
    text_b = semantics.normalize_product_text(name_b, cat_b, fam_b)
    text_pair = f"{text_a}::{text_b}"
    has_risky_hint = _text_has_any_hint(text_pair, STRICT_FALLBACK_RISK_HINTS)
    if has_risky_hint:
        if cp_score < 34.0 or pair_count < 14:
            return "fallback_risky_evidence_too_low"
        if template_strength < 0.80 and category_strength < 0.72:
            return "fallback_risky_pattern_too_weak"
        if recipe < 0.16 and prior_bonus <= 0.0:
            return "fallback_risky_recipe_support_too_low"

    motif_key = _controlled_fallback_motif_key(candidate, context)
    if motif_key in EVAP_REPETITIVE_FALLBACK_MOTIFS:
        if cp_score < 32.0 or pair_count < 12:
            return "fallback_evap_motif_evidence_too_low"
        if template_strength < 0.82 and lane == LANE_OCCASION:
            return "fallback_evap_motif_template_too_low"

    if _is_plain_milk_beverage_text(text_a) or _is_plain_milk_beverage_text(text_b):
        if lane == LANE_MEAL and prior_bonus <= 0.0 and recipe < 0.18:
            return "fallback_plain_milk_meal_reject"

    return None


def _normalize_name_tokens(name: str) -> set[str]:
    norm = _normalise_text(name)
    parts = [part for part in norm.split() if part]
    tokens = set(parts)
    for idx in range(len(parts) - 1):
        tokens.add(f"{parts[idx]} {parts[idx + 1]}")
    return tokens


def _group_labels_from_text(name: str, category: str, family: str) -> set[str]:
    text = f"{_normalise_text(name)} {_normalise_text(category)} {_normalise_text(family)}"
    tokens = _normalize_name_tokens(text)
    groups: set[str] = set()

    bread_tokens = {"bread", "tortilla", "toast", "naan", "pita", "wrap", "loaf", "bun", "paratha"}
    chips_tokens = {"chips", "crisps", "popcorn", "nachos", "potato chips"}
    cracker_tokens = {"cracker", "crackers"}
    cookie_tokens = {"biscuit", "biscuits", "cookie", "cookies", "wafer", "tea biscuit"}
    chocolate_tokens = {"chocolate", "cocoa"}
    candy_tokens = {"candy", "caramel", "toffee", "sweet"}
    nuts_tokens = {"nuts", "almond", "cashew", "pistachio", "peanut"}
    tea_tokens = {"tea", "chai"}
    coffee_tokens = {"coffee", "nescafe", "espresso", "cappuccino"}
    milk_tokens = {"milk", "evaporated", "condensed"}
    soda_tokens = {"soda", "cola", "pepsi", "soft drink"}
    juice_tokens = {"juice", "vimto"}
    dates_tokens = {"dates"}
    cream_tokens = {"cream", "whip", "custard", "caramel"}
    cream_cheese_tokens = {"cream cheese", "kiri", "puck", "labneh"}
    cheese_tokens = {"cheese", "mozzarella", "feta", "triangles"}
    produce_tokens = {"onion", "garlic", "parsley", "mint", "vegetables", "vegetable", "carrots", "tomato", "cucumber"}
    spices_tokens = {"pepper", "cumin", "spice", "masala", "seasoning", "stock", "bouillon", "turmeric", "cardamom"}
    noodles_tokens = {"noodles", "indomie", "pasta", "spaghetti", "vermicelli"}
    grains_tokens = {"rice", "bulgur", "grains", "grain", "flour", "oats", "semolina"}
    protein_tokens = {"chicken", "beef", "lamb", "tuna", "fish", "protein", "nuggets", "burger", "sausage", "eggs"}

    cleaning_tokens = {"dishwashing", "detergent", "cleaner", "disinfectant", "bleach", "floor cleaner"}
    hair_tokens = {"shampoo", "conditioner"}
    body_tokens = {"soap", "body wash", "shower gel"}
    tissue_tokens = {"tissue", "paper towel", "paper towels", "toilet paper"}
    nonfood_tokens = {"trash bag", "waste bag", "foil", "wrap", "cleaning", "personal care", NON_FOOD_TAG}

    if tokens & bread_tokens:
        groups.update({GROUP_BREAD_CARB, GROUP_CARBS})
    if tokens & chips_tokens:
        groups.update({GROUP_CHIPS, GROUP_SNACKS})
    if tokens & cracker_tokens:
        if tokens & bread_tokens:
            groups.update({GROUP_BREAD_CARB, GROUP_CARBS})
        else:
            groups.update({GROUP_CRACKERS, GROUP_SNACKS})
    if tokens & cookie_tokens:
        groups.update({GROUP_COOKIES, GROUP_SNACKS, GROUP_SWEETS})
    if tokens & chocolate_tokens:
        groups.update({GROUP_CHOCOLATE, GROUP_SNACKS, GROUP_SWEETS})
    if tokens & candy_tokens:
        groups.update({GROUP_CANDY, GROUP_SNACKS, GROUP_SWEETS})
    if tokens & nuts_tokens:
        groups.update({GROUP_NUTS, GROUP_SNACKS})
    if tokens & tea_tokens:
        groups.update({GROUP_TEA, GROUP_BEVERAGES})
    if tokens & coffee_tokens:
        groups.update({GROUP_COFFEE, GROUP_BEVERAGES})
    if tokens & milk_tokens:
        groups.update({GROUP_MILK, GROUP_DAIRY})
    if tokens & soda_tokens:
        groups.update({GROUP_SODA, GROUP_BEVERAGES})
    if tokens & juice_tokens:
        groups.update({GROUP_JUICE, GROUP_BEVERAGES})
    if tokens & dates_tokens:
        groups.add(GROUP_DATES)
    if tokens & cream_tokens:
        groups.update({GROUP_CREAM, GROUP_DAIRY})
    if tokens & cream_cheese_tokens:
        groups.update({GROUP_CREAM_CHEESE, GROUP_CHEESE, GROUP_DAIRY})
    if tokens & cheese_tokens:
        groups.update({GROUP_CHEESE, GROUP_DAIRY})
    if tokens & produce_tokens:
        groups.add(GROUP_PRODUCE)
    if tokens & spices_tokens:
        groups.add(GROUP_SPICES)
    if tokens & noodles_tokens:
        groups.update({GROUP_NOODLES_PASTA, GROUP_NOODLES})
    if tokens & grains_tokens:
        groups.update({GROUP_RICE_GRAINS, GROUP_CARBS})
    if tokens & protein_tokens:
        groups.add(GROUP_PROTEIN)

    if tokens & cleaning_tokens:
        groups.add(GROUP_NONFOOD_CLEANING)
    if tokens & hair_tokens:
        groups.add(GROUP_NONFOOD_HAIR)
    if tokens & body_tokens:
        groups.add(GROUP_NONFOOD_BODY)
    if tokens & tissue_tokens:
        groups.add(GROUP_NONFOOD_TISSUE)
    if tokens & nonfood_tokens:
        groups.add(GROUP_NONFOOD_OTHER)

    # Category/family fallbacks.
    if "snack" in text:
        groups.add(GROUP_SNACKS)
    if "beverage" in text:
        groups.add(GROUP_BEVERAGES)
    if "dairy" in text:
        groups.add(GROUP_DAIRY)
    if "spice" in text or "herb" in text:
        groups.add(GROUP_SPICES)
    if "vegetable" in text:
        groups.add(GROUP_PRODUCE)
    if "grain" in text:
        groups.update({GROUP_RICE_GRAINS, GROUP_CARBS})
    if "protein" in text:
        groups.add(GROUP_PROTEIN)
    if NON_FOOD_TAG in text:
        groups.add(GROUP_NONFOOD_OTHER)

    return groups


def _group_labels_for_pid(
    pid: int,
    context: PersonalizationContext,
    row: pd.Series | None = None,
    side: str | None = None,
) -> set[str]:
    return _group_labels_from_text(
        _product_name(pid, context, row=row, side=side),
        _product_category(pid, context, row=row, side=side),
        _product_family(pid, context, row=row, side=side),
    )


def _product_semantic_group(
    pid: int,
    context: PersonalizationContext,
    row: pd.Series | None = None,
    side: str | None = None,
) -> str:
    name = _product_name(pid, context, row=row, side=side)
    category = _product_category(pid, context, row=row, side=side)
    family = _product_family(pid, context, row=row, side=side)
    text = f"{_normalise_text(name)} {_normalise_text(category)} {_normalise_text(family)}"
    groups = _group_labels_from_text(name, category, family)

    if _is_nonfood_product(pid, context, row=row, side=side) or groups & {
        GROUP_NONFOOD_CLEANING,
        GROUP_NONFOOD_HAIR,
        GROUP_NONFOOD_BODY,
        GROUP_NONFOOD_TISSUE,
        GROUP_NONFOOD_OTHER,
    }:
        if any(token in text for token in ("detergent", "softener", "laundry")):
            return SEM_DETERGENT
        if any(token in text for token in ("shampoo", "conditioner", "deodorant", "lotion", "cosmetic", "underarm")):
            return SEM_COSMETICS
        if any(token in text for token in ("soap", "body wash", "shower gel")):
            return SEM_SOAP
        if any(token in text for token in ("paper", "tissue", "napkin", "bag", "foil", "wrap", "cup", "plate", "container")):
            return SEM_PACKAGING
        return SEM_CLEANING

    if any(token in text for token in SAUCE_HINTS):
        return SEM_SAUCE
    if groups & {GROUP_PROTEIN}:
        return SEM_PROTEIN
    if groups & {GROUP_PRODUCE}:
        return SEM_PRODUCE
    if groups & {GROUP_RICE_GRAINS, GROUP_NOODLES_PASTA, GROUP_BREAD_CARB, GROUP_CARBS}:
        return SEM_GRAINS
    if groups & {GROUP_SPICES}:
        return SEM_SPICES
    if groups & {GROUP_TEA, GROUP_COFFEE, GROUP_SODA, GROUP_JUICE, GROUP_BEVERAGES}:
        return SEM_BEVERAGE
    if groups & {GROUP_CHIPS, GROUP_CRACKERS, GROUP_NUTS, GROUP_SNACKS}:
        return SEM_SNACKS
    if groups & {GROUP_CHOCOLATE, GROUP_CANDY, GROUP_COOKIES, GROUP_DATES}:
        return SEM_DESSERT
    if any(token in text for token in ("dessert", "cake", "custard", "caramel", "nutella", "sweet", "sugar")):
        return SEM_DESSERT
    if groups & {GROUP_DAIRY, GROUP_MILK, GROUP_CREAM, GROUP_CREAM_CHEESE, GROUP_CHEESE}:
        return SEM_DAIRY
    if "grain" in text or "rice" in text or "pasta" in text or "flour" in text:
        return SEM_GRAINS
    if "beverage" in text or "drink" in text:
        return SEM_BEVERAGE
    if "snack" in text:
        return SEM_SNACKS
    return SEM_UNKNOWN


def _sem_pair(anchor_sem: str, comp_sem: str) -> frozenset[str]:
    return frozenset({str(anchor_sem), str(comp_sem)})


def _is_occasion_biscuit_pair(anchor_groups: set[str], comp_groups: set[str]) -> bool:
    beverage_side_a = bool(anchor_groups & {GROUP_TEA, GROUP_COFFEE, GROUP_BEVERAGES})
    beverage_side_b = bool(comp_groups & {GROUP_TEA, GROUP_COFFEE, GROUP_BEVERAGES})
    biscuit_side_a = bool(anchor_groups & {GROUP_COOKIES, GROUP_CRACKERS})
    biscuit_side_b = bool(comp_groups & {GROUP_COOKIES, GROUP_CRACKERS})
    return bool((beverage_side_a and biscuit_side_b) or (beverage_side_b and biscuit_side_a))


def _is_tea_coffee_milk_pair(anchor_groups: set[str], comp_groups: set[str]) -> bool:
    tea_or_coffee_a = bool(anchor_groups & {GROUP_TEA, GROUP_COFFEE})
    tea_or_coffee_b = bool(comp_groups & {GROUP_TEA, GROUP_COFFEE})
    milk_a = bool(anchor_groups & {GROUP_MILK})
    milk_b = bool(comp_groups & {GROUP_MILK})
    return bool((tea_or_coffee_a and milk_b) or (tea_or_coffee_b and milk_a))


def _semantic_lane_compatible(
    anchor: int,
    complement: int,
    lane: str,
    context: PersonalizationContext,
    *,
    pair_row: pd.Series | None = None,
    anchor_groups: set[str] | None = None,
    comp_groups: set[str] | None = None,
) -> bool:
    groups_a = set(anchor_groups or _group_labels_for_pid(anchor, context, row=pair_row, side="a"))
    groups_b = set(comp_groups or _group_labels_for_pid(complement, context, row=pair_row, side="b"))
    sem_a = _product_semantic_group(anchor, context, row=pair_row, side="a")
    sem_b = _product_semantic_group(complement, context, row=pair_row, side="b")
    sem_pair = _sem_pair(sem_a, sem_b)

    a_is_nonfood = sem_a in NONFOOD_SEMANTIC_GROUPS
    b_is_nonfood = sem_b in NONFOOD_SEMANTIC_GROUPS

    if lane != LANE_NONFOOD and (a_is_nonfood or b_is_nonfood):
        return False
    if lane == LANE_NONFOOD:
        return bool(a_is_nonfood and b_is_nonfood and sem_a == sem_b)

    if SEM_DESSERT in sem_pair and sem_pair & SAVORY_SEMANTIC_GROUPS:
        return False

    if lane == LANE_MEAL:
        if sem_pair <= {SEM_SPICES}:
            return True
        if sem_pair == frozenset({SEM_PRODUCE, SEM_SPICES}):
            return True
        if SEM_SNACKS in sem_pair or SEM_DESSERT in sem_pair:
            return False
        if SEM_BEVERAGE in sem_pair:
            return False
        if sem_pair == frozenset({SEM_GRAINS, SEM_DAIRY}) and (GROUP_NOODLES_PASTA in groups_a or GROUP_NOODLES_PASTA in groups_b):
            return False
        allowed = {
            frozenset({SEM_GRAINS, SEM_PROTEIN}),
            frozenset({SEM_GRAINS, SEM_SAUCE}),
            frozenset({SEM_PROTEIN, SEM_SAUCE}),
            frozenset({SEM_GRAINS, SEM_PRODUCE}),
            frozenset({SEM_PROTEIN, SEM_PRODUCE}),
            frozenset({SEM_PRODUCE, SEM_SAUCE}),
            frozenset({SEM_PROTEIN, SEM_SPICES}),
            frozenset({SEM_GRAINS, SEM_SPICES}),
        }
        return sem_pair in allowed

    if lane == LANE_SNACK:
        if sem_pair & {SEM_GRAINS, SEM_PROTEIN, SEM_PRODUCE}:
            return False
        if sem_pair in {
            frozenset({SEM_SNACKS, SEM_SNACKS}),
            frozenset({SEM_SNACKS, SEM_DAIRY}),
            frozenset({SEM_DESSERT, SEM_DAIRY}),
        }:
            return True
        snack_pattern = _snack_pattern_key(groups_a, groups_b)
        return bool(snack_pattern is not None)

    if lane == LANE_OCCASION:
        if GROUP_CHIPS in groups_a or GROUP_CHIPS in groups_b:
            return False
        if sem_pair == frozenset({SEM_BEVERAGE, SEM_DESSERT}):
            return True
        if sem_pair == frozenset({SEM_DESSERT, SEM_DAIRY}):
            return True
        if _is_occasion_biscuit_pair(groups_a, groups_b):
            return True
        if sem_pair == frozenset({SEM_BEVERAGE, SEM_DAIRY}) and _is_tea_coffee_milk_pair(groups_a, groups_b):
            return True
        return False

    return True


def _oriented_pair_key(anchor: int, complement: int) -> tuple[int, int]:
    return int(anchor), int(complement)


def _snack_pattern_key(groups_a: set[str], groups_b: set[str]) -> str | None:
    union = set(groups_a) | set(groups_b)
    if GROUP_BREAD_CARB in union:
        return None
    if (
        GROUP_PRODUCE in union
        or GROUP_RICE_GRAINS in union
        or GROUP_NOODLES_PASTA in union
        or GROUP_PROTEIN in union
    ):
        if not (
            GROUP_DATES in union
            and (
                GROUP_CREAM in union
                or GROUP_CREAM_CHEESE in union
                or GROUP_CHEESE in union
            )
        ) and not (GROUP_NUTS in union and (GROUP_SODA in union or GROUP_TEA in union or GROUP_COFFEE in union or GROUP_JUICE in union)):
            return None

    def _has(group: str, left: set[str], right: set[str]) -> bool:
        return group in left and group in right

    # chips/crisps + soda/cola/soft drink
    if (
        (GROUP_CHIPS in groups_a and GROUP_SODA in groups_b)
        or (GROUP_CHIPS in groups_b and GROUP_SODA in groups_a)
    ):
        return "drink_snack"
    # cookies/biscuits + tea or coffee
    if (
        (GROUP_COOKIES in groups_a and (GROUP_TEA in groups_b or GROUP_COFFEE in groups_b))
        or (GROUP_COOKIES in groups_b and (GROUP_TEA in groups_a or GROUP_COFFEE in groups_a))
        or (GROUP_CRACKERS in groups_a and (GROUP_TEA in groups_b or GROUP_COFFEE in groups_b))
        or (GROUP_CRACKERS in groups_b and (GROUP_TEA in groups_a or GROUP_COFFEE in groups_a))
    ):
        return "tea_snack"
    # cookies/biscuits + milk
    if (
        ((GROUP_COOKIES in groups_a or GROUP_CRACKERS in groups_a) and GROUP_MILK in groups_b)
        or ((GROUP_COOKIES in groups_b or GROUP_CRACKERS in groups_b) and GROUP_MILK in groups_a)
    ):
        return "cookie_milk"
    # chocolate/candy + milk
    if (
        ((GROUP_CHOCOLATE in groups_a or GROUP_CANDY in groups_a) and GROUP_MILK in groups_b)
        or ((GROUP_CHOCOLATE in groups_b or GROUP_CANDY in groups_b) and GROUP_MILK in groups_a)
    ):
        return "sweet_milk"
    # wafer/cookie + chocolate
    if (
        (GROUP_COOKIES in groups_a and GROUP_CHOCOLATE in groups_b)
        or (GROUP_COOKIES in groups_b and GROUP_CHOCOLATE in groups_a)
    ):
        return "wafer_chocolate"
    # dates + cream/cream cheese/cheese
    if (
        (GROUP_DATES in groups_a and (GROUP_CREAM in groups_b or GROUP_CREAM_CHEESE in groups_b or GROUP_CHEESE in groups_b))
        or (GROUP_DATES in groups_b and (GROUP_CREAM in groups_a or GROUP_CREAM_CHEESE in groups_a or GROUP_CHEESE in groups_a))
    ):
        return "dates_cream"
    # nuts + drink
    if (
        (GROUP_NUTS in groups_a and (GROUP_SODA in groups_b or GROUP_TEA in groups_b or GROUP_COFFEE in groups_b or GROUP_JUICE in groups_b))
        or (GROUP_NUTS in groups_b and (GROUP_SODA in groups_a or GROUP_TEA in groups_a or GROUP_COFFEE in groups_a or GROUP_JUICE in groups_a))
    ):
        return "nuts_drink"
    # cheese + chips/crackers (never bread)
    if (
        ((GROUP_CHEESE in groups_a or GROUP_CREAM_CHEESE in groups_a) and (GROUP_CHIPS in groups_b or GROUP_CRACKERS in groups_b))
        or ((GROUP_CHEESE in groups_b or GROUP_CREAM_CHEESE in groups_b) and (GROUP_CHIPS in groups_a or GROUP_CRACKERS in groups_a))
    ):
        return "cheese_snack"
    return None


def _is_snack_anchor_allowed(groups: set[str]) -> bool:
    if groups & SNACK_ANCHOR_BLOCKED_GROUPS:
        return False
    return bool(groups & SNACK_ANCHOR_ALLOWED_GROUPS)


def _pair_theme(
    groups_a: set[str],
    groups_b: set[str],
    lane: str,
    snack_pattern: str | None,
) -> str:
    union = set(groups_a) | set(groups_b)
    if snack_pattern:
        return snack_pattern
    if GROUP_BREAD_CARB in union:
        return "bread_meal"
    if GROUP_RICE_GRAINS in union and GROUP_PROTEIN in union:
        return "rice_protein"
    if GROUP_NOODLES_PASTA in union and GROUP_PROTEIN in union:
        return "noodle_protein"
    if union and union <= {GROUP_SPICES}:
        return "spice_bundle"
    if lane == LANE_MEAL:
        return "meal_generic"
    if lane == LANE_SNACK:
        return "snack_generic"
    if lane == LANE_OCCASION:
        return "occasion_generic"
    return "nonfood_generic"


def _primary_group(groups: set[str]) -> str:
    for group in GROUP_PRIMARY_ORDER:
        if group in groups:
            return group
    if groups:
        return sorted(groups)[0]
    return "none"


def _pair_fingerprint(
    anchor: int,
    complement: int,
    context: PersonalizationContext,
    anchor_groups: set[str],
    comp_groups: set[str],
    row: pd.Series | None = None,
) -> tuple[str, str]:
    fam_a = _product_family(anchor, context, row=row, side="a") or "na"
    fam_b = _product_family(complement, context, row=row, side="b") or "nb"
    fam_pair = tuple(sorted((fam_a, fam_b)))
    group_pair = tuple(sorted((_primary_group(anchor_groups), _primary_group(comp_groups))))
    return (f"{fam_pair[0]}::{fam_pair[1]}", f"{group_pair[0]}::{group_pair[1]}")


def _nonfood_group_for_pid(
    pid: int,
    context: PersonalizationContext,
    row: pd.Series | None = None,
    side: str | None = None,
) -> str | None:
    groups = _group_labels_for_pid(pid, context, row=row, side=side)
    for label, mapped in NONFOOD_GROUP_MAP.items():
        if label in groups:
            return mapped
    group_name = _nonfood_group(
        _product_name(pid, context, row=row, side=side),
        _product_category(pid, context, row=row, side=side),
        _product_family(pid, context, row=row, side=side),
    )
    if group_name:
        if "clean" in group_name or "laundry" in group_name:
            return "cleaning"
        if "care" in group_name or "hair" in group_name:
            return "hair"
        if "body" in group_name:
            return "body"
        if "paper" in group_name or "tissue" in group_name:
            return "tissue"
        return "other"
    return None


def _passes_lane_negative_rules(
    anchor_pid: int,
    cand_pid: int,
    lane: str,
    context: PersonalizationContext,
    recipe_compat_score: float,
    *,
    anchor_groups: set[str] | None = None,
    cand_groups: set[str] | None = None,
    pair_row: pd.Series | None = None,
) -> bool:
    groups_a = set(anchor_groups or _group_labels_for_pid(anchor_pid, context, row=pair_row, side="a"))
    groups_b = set(cand_groups or _group_labels_for_pid(cand_pid, context, row=pair_row, side="b"))
    union = groups_a | groups_b
    name_a = _product_name(anchor_pid, context, row=pair_row, side="a")
    name_b = _product_name(cand_pid, context, row=pair_row, side="b")
    cat_a = _product_category(anchor_pid, context, row=pair_row, side="a")
    cat_b = _product_category(cand_pid, context, row=pair_row, side="b")
    family_a = _product_family(anchor_pid, context, row=pair_row, side="a")
    family_b = _product_family(cand_pid, context, row=pair_row, side="b")

    if lane in FOOD_LANE_ORDER:
        if (
            (GROUP_PRODUCE in groups_a and GROUP_CHIPS in groups_b)
            or (GROUP_PRODUCE in groups_b and GROUP_CHIPS in groups_a)
        ):
            return False
        if lane in {LANE_MEAL, LANE_OCCASION}:
            produce_noodles = (
                (GROUP_PRODUCE in groups_a and GROUP_NOODLES_PASTA in groups_b)
                or (GROUP_PRODUCE in groups_b and GROUP_NOODLES_PASTA in groups_a)
            )
            if produce_noodles and float(recipe_compat_score) < 0.35:
                return False
        if lane == LANE_MEAL:
            if GROUP_PRODUCE in groups_a and GROUP_PRODUCE in groups_b and float(recipe_compat_score) < 0.45:
                return False
            if union and union <= {GROUP_SPICES}:
                return True
        if lane == LANE_OCCASION:
            fat_cheese_combo = (
                _is_fat_or_oil_item(name_a, cat_a, family_a) and _is_cheese_spread_item(name_b, cat_b, family_b)
            ) or (
                _is_fat_or_oil_item(name_b, cat_b, family_b) and _is_cheese_spread_item(name_a, cat_a, family_a)
            )
            if fat_cheese_combo:
                return False
    if lane == LANE_SNACK:
        if GROUP_PRODUCE in union or GROUP_RICE_GRAINS in union or GROUP_NOODLES_PASTA in union or GROUP_PROTEIN in union:
            return False
    return True


def _contains_hint(text: str, hints: frozenset[str]) -> bool:
    norm = _normalise_text(text)
    return any(hint in norm for hint in hints)


def _lane_hint_strength(pid: int, context: PersonalizationContext, lane: str) -> float:
    name = str(context.product_name_by_id.get(pid, ""))
    category = str(context.category_by_id.get(pid, ""))
    family = str(context.product_family_by_id.get(pid, ""))
    if lane == LANE_NONFOOD:
        return float(
            0.45 * int(_contains_hint(name, LANE_NONFOOD_NAME_HINTS))
            + 0.30 * int(_contains_hint(category, LANE_NONFOOD_CATEGORY_HINTS))
            + 0.25 * int(_contains_hint(family, LANE_NONFOOD_FAMILY_HINTS))
        )
    if lane == LANE_SNACK:
        return float(
            0.45 * int(_contains_hint(name, LANE_SNACK_NAME_HINTS))
            + 0.30 * int(_contains_hint(category, LANE_SNACK_CATEGORY_HINTS))
            + 0.25 * int(_contains_hint(family, LANE_SNACK_FAMILY_HINTS))
        )
    if lane == LANE_OCCASION:
        return float(
            0.45 * int(_contains_hint(name, LANE_OCCASION_NAME_HINTS))
            + 0.30 * int(_contains_hint(category, LANE_OCCASION_CATEGORY_HINTS))
            + 0.25 * int(_contains_hint(family, LANE_OCCASION_FAMILY_HINTS))
        )
    return float(
        0.45 * int(_contains_hint(name, LANE_MEAL_NAME_HINTS))
        + 0.30 * int(_contains_hint(category, LANE_MEAL_CATEGORY_HINTS))
        + 0.25 * int(_contains_hint(family, LANE_MEAL_FAMILY_HINTS))
    )


def _classify_anchor_lane(pid: int, context: PersonalizationContext) -> str:
    if _is_nonfood_product(int(pid), context):
        return LANE_NONFOOD
    scores = {
        LANE_MEAL: _lane_hint_strength(pid, context, LANE_MEAL),
        LANE_SNACK: _lane_hint_strength(pid, context, LANE_SNACK),
        LANE_OCCASION: _lane_hint_strength(pid, context, LANE_OCCASION),
    }
    if scores[LANE_OCCASION] >= max(scores[LANE_MEAL], scores[LANE_SNACK]) and scores[LANE_OCCASION] > 0:
        return LANE_OCCASION
    if scores[LANE_SNACK] >= scores[LANE_MEAL] and scores[LANE_SNACK] > 0:
        return LANE_SNACK
    return LANE_MEAL


def _history_lane_ratios(profile: PersonProfile, context: PersonalizationContext) -> tuple[float, float, float]:
    history_ids = [int(pid) for pid in profile.history_product_ids if int(pid) > 0 and int(pid) not in context.non_food_ids]
    if not history_ids:
        return 0.0, 0.0, 0.0
    lane_counts = {LANE_MEAL: 0, LANE_SNACK: 0, LANE_OCCASION: 0}
    for pid in history_ids:
        lane = _classify_anchor_lane(pid, context)
        if lane not in lane_counts:
            continue
        lane_counts[lane] += 1
    total = float(len(history_ids))
    return (
        lane_counts[LANE_MEAL] / total,
        lane_counts[LANE_SNACK] / total,
        lane_counts[LANE_OCCASION] / total,
    )


def _top10_lane_quotas(meal_ratio: float, snack_ratio: float) -> dict[str, int]:
    if meal_ratio >= 0.60:
        return {LANE_MEAL: 8, LANE_SNACK: 5, LANE_OCCASION: 5}
    if snack_ratio >= 0.60:
        return {LANE_MEAL: 7, LANE_SNACK: 6, LANE_OCCASION: 5}
    return {LANE_MEAL: 8, LANE_SNACK: 5, LANE_OCCASION: 5}


def _is_same_variant(name_a: str, name_b: str) -> bool:
    set_a = _token_set(name_a)
    set_b = _token_set(name_b)
    if not set_a or not set_b:
        return False
    inter = len(set_a & set_b)
    if inter == 0:
        return False
    ratio = inter / max(len(set_a), len(set_b))
    return ratio >= 0.8 and inter >= 2


def _recipe_compatibility(anchor: int, complement: int, context: PersonalizationContext) -> float:
    ingredient_a = str(context.ingredient_by_id.get(int(anchor), "")).strip().lower()
    ingredient_b = str(context.ingredient_by_id.get(int(complement), "")).strip().lower()
    if not ingredient_a or not ingredient_b:
        return 0.0
    recipes_a = set(context.ingredient_recipe_lookup.get(ingredient_a, ()))
    recipes_b = set(context.ingredient_recipe_lookup.get(ingredient_b, ()))
    if not recipes_a or not recipes_b:
        return 0.0
    union = recipes_a | recipes_b
    if not union:
        return 0.0
    return len(recipes_a & recipes_b) / len(union)


def _known_prior_bonus(anchor_name: str, complement_name: str) -> float:
    anchor_tokens = _token_set(anchor_name)
    complement_tokens = _token_set(complement_name)
    if not anchor_tokens or not complement_tokens:
        return 0.0
    for anchor_key, complement_hints in KNOWN_COMPLEMENT_PRIORS.items():
        anchor_key_tokens = _token_set(anchor_key)
        if not anchor_key_tokens or not (anchor_key_tokens & anchor_tokens):
            continue
        for hint in complement_hints:
            if _token_set(hint) & complement_tokens:
                return 1.0
    return 0.0


def _is_utility_like_candidate(name: str, category: str) -> bool:
    tokens = _token_set(name)
    if tokens & UTILITY_TOKENS:
        return True
    return str(category).strip().lower() in UTILITY_CATEGORIES


def _is_staple_product(name: str, family: str = "", category: str = "") -> bool:
    norm_name = _normalise_text(name)
    name_tokens = {token for token in norm_name.split() if token}
    if name_tokens & STAPLE_NAME_HINTS:
        return True
    norm_family = _normalise_text(family)
    norm_category = _normalise_text(category)
    family_hints = {"staple", "grain", "grains", "sugar", "oil", "oils", "salt", "flour", "rice", "spices"}
    return any(hint in norm_family for hint in family_hints) or any(hint in norm_category for hint in family_hints)


def _is_food_lane(lane: str) -> bool:
    return lane in FOOD_LANE_ORDER


def _fails_sweet_savory_pair(anchor_name: str, anchor_family: str, complement_name: str, complement_family: str) -> bool:
    a_savory = any(h in _normalise_text(anchor_name) for h in SAVORY_BASE_HINTS) or any(
        h in _normalise_text(anchor_family) for h in SAVORY_BASE_HINTS
    )
    b_savory = any(h in _normalise_text(complement_name) for h in SAVORY_BASE_HINTS) or any(
        h in _normalise_text(complement_family) for h in SAVORY_BASE_HINTS
    )
    a_sweet = any(h in _normalise_text(anchor_name) for h in PROCESSED_SWEET_HINTS) or any(
        h in _normalise_text(anchor_family) for h in PROCESSED_SWEET_HINTS
    )
    b_sweet = any(h in _normalise_text(complement_name) for h in PROCESSED_SWEET_HINTS) or any(
        h in _normalise_text(complement_family) for h in PROCESSED_SWEET_HINTS
    )
    return bool((a_savory and b_sweet) or (b_savory and a_sweet))


def _matches_pair_hints(name_a: str, name_b: str, left_hint: str, right_hint: str) -> bool:
    text_a = _normalise_text(name_a)
    text_b = _normalise_text(name_b)
    return (left_hint in text_a and right_hint in text_b) or (left_hint in text_b and right_hint in text_a)


def _suspicious_default_signature(anchor_name: str, comp_name: str, lane: str) -> str | None:
    checks: dict[str, tuple[tuple[str, str, str], ...]] = {
        LANE_MEAL: (
            ("meal_rice_tomato_paste", "rice", "tomato paste"),
            ("meal_rice_chicken", "rice", "chicken"),
            ("meal_tuna_tomato_paste", "tuna", "tomato paste"),
        ),
        LANE_SNACK: (
            ("snack_chips_cheese", "chips", "cheese"),
            ("snack_chocolate_milk", "chocolate", "milk"),
            ("snack_dessert_condensed", "dessert", "condensed"),
        ),
        LANE_OCCASION: (
            ("occasion_tea_biscuit", "tea", "biscuit"),
            ("occasion_coffee_biscuit", "coffee", "biscuit"),
            ("occasion_milk_biscuit", "milk", "biscuit"),
            ("occasion_dates_cream", "dates", "cream"),
            ("occasion_condensed_biscuit", "condensed", "biscuit"),
        ),
    }
    for signature, left_hint, right_hint in checks.get(lane, ()): 
        if _matches_pair_hints(anchor_name, comp_name, left_hint, right_hint):
            return signature
    return None


def _suspicious_default_evidence_ok(
    *,
    lane: str,
    anchor_name: str,
    anchor_cat: str,
    anchor_family: str,
    comp_name: str,
    comp_cat: str,
    comp_family: str,
    cp_score: float,
    recipe_compat: float,
    prior_bonus: float,
    pair_count: int,
) -> bool:
    signature = _suspicious_default_signature(anchor_name, comp_name, lane)
    if signature is None:
        return True

    roles_a = semantics.infer_product_roles(anchor_name, anchor_cat, anchor_family)
    roles_b = semantics.infer_product_roles(comp_name, comp_cat, comp_family)
    all_roles = set(roles_a | roles_b)
    serving_milk = bool({semantics.ROLE_MILK_FRESH, semantics.ROLE_MILK_EVAP} & all_roles)
    serving_cream = semantics.ROLE_CREAM_TABLE in all_roles
    blocked_milk = bool({semantics.ROLE_MILK_COND, semantics.ROLE_MILK_POWDER, semantics.ROLE_MILK_BABY} & all_roles)
    biscuit_like = semantics.ROLE_BISCUIT in all_roles
    snack_cheese = semantics.ROLE_CHEESE_SNACK in all_roles
    prep_cheese = bool({semantics.ROLE_CHEESE_SPREAD, semantics.ROLE_CHEESE_COOKING} & all_roles)
    date_paste_pair = "paste" in _normalise_text(anchor_name) or "paste" in _normalise_text(comp_name)

    if signature == "meal_rice_tomato_paste":
        return float(cp_score) >= 42.0 and int(pair_count) >= 16 and float(recipe_compat) >= 0.18
    if signature == "meal_rice_chicken":
        return float(cp_score) >= 35.0 and int(pair_count) >= 14 and float(recipe_compat) >= 0.16
    if signature == "meal_tuna_tomato_paste":
        return float(cp_score) >= 45.0 and int(pair_count) >= 18 and float(recipe_compat) >= 0.18
    if signature == "snack_chips_cheese":
        return snack_cheese and not prep_cheese and float(cp_score) >= 30.0 and int(pair_count) >= 12
    if signature == "snack_chocolate_milk":
        if biscuit_like:
            return serving_milk and not blocked_milk and float(cp_score) >= 24.0 and int(pair_count) >= 10
        return serving_milk and not blocked_milk and float(cp_score) >= 22.0 and int(pair_count) >= 8 and float(recipe_compat) >= 0.12
    if signature == "snack_dessert_condensed":
        return float(cp_score) >= 28.0 and int(pair_count) >= 10 and float(recipe_compat) >= 0.14
    if signature in {"occasion_tea_biscuit", "occasion_coffee_biscuit"}:
        return float(cp_score) >= 34.0 and int(pair_count) >= 12
    if signature == "occasion_milk_biscuit":
        return serving_milk and not blocked_milk and float(cp_score) >= 45.0 and int(pair_count) >= 20 and float(recipe_compat) >= 0.10
    if signature == "occasion_dates_cream":
        return not date_paste_pair and (serving_cream or serving_milk) and not prep_cheese and float(cp_score) >= 24.0 and int(pair_count) >= 8 and (float(prior_bonus) > 0.0 or float(recipe_compat) >= 0.10)
    if signature == "occasion_condensed_biscuit":
        return False
    return True


def _passes_choice_score_floor(lane: str, origin: str, score_value: float) -> bool:
    lane_floors = LANE_ORIGIN_SCORE_FLOORS.get(lane, {})
    source_group = _source_group_from_source(origin)
    floor = float(lane_floors.get(source_group, lane_floors.get(origin, 0.0)))
    return float(score_value) >= floor


def _template_match_strength(anchor_name: str, complement_name: str, lane: str) -> float:
    patterns = KNOWN_LANE_PATTERNS.get(lane, ())
    a_tokens = _token_set(anchor_name)
    b_tokens = _token_set(complement_name)
    best = 0.0
    for left, right in patterns:
        left_tokens = _token_set(left)
        right_tokens = _token_set(right)
        if not left_tokens or not right_tokens:
            continue
        if (left_tokens & a_tokens and right_tokens & b_tokens) or (left_tokens & b_tokens and right_tokens & a_tokens):
            left_hit = max(len(left_tokens & a_tokens), len(left_tokens & b_tokens)) / max(len(left_tokens), 1)
            right_hit = max(len(right_tokens & a_tokens), len(right_tokens & b_tokens)) / max(len(right_tokens), 1)
            best = max(best, min(1.0, 0.5 * left_hit + 0.5 * right_hit))
    return float(best)


def _category_complement_strength(anchor_cat: str, comp_cat: str, lane: str) -> float:
    cat_a = _normalise_text(anchor_cat)
    cat_b = _normalise_text(comp_cat)
    if not cat_a or not cat_b:
        return 0.0
    pair = (cat_a, cat_b)
    pair_rev = (cat_b, cat_a)
    if lane == LANE_MEAL:
        return 1.0 if (pair in MEAL_CATEGORY_COMPLEMENTS or pair_rev in MEAL_CATEGORY_COMPLEMENTS) else 0.0
    if lane == LANE_SNACK:
        return 1.0 if (pair in SNACK_CATEGORY_COMPLEMENTS or pair_rev in SNACK_CATEGORY_COMPLEMENTS) else 0.0
    if lane == LANE_OCCASION:
        return 1.0 if (pair in OCCASION_CATEGORY_COMPLEMENTS or pair_rev in OCCASION_CATEGORY_COMPLEMENTS) else 0.0
    if lane == LANE_NONFOOD:
        return 1.0 if (pair in NONFOOD_CATEGORY_COMPLEMENTS or pair_rev in NONFOOD_CATEGORY_COMPLEMENTS) else 0.0
    return 0.0


def _passes_known_lane_pattern(anchor_name: str, complement_name: str, lane: str) -> bool:
    return _template_match_strength(anchor_name, complement_name, lane) >= 0.6


def _passes_category_complement_rule(anchor: int, complement: int, lane: str, context: PersonalizationContext) -> bool:
    cat_a = _normalise_text(context.category_by_id.get(int(anchor), ""))
    cat_b = _normalise_text(context.category_by_id.get(int(complement), ""))
    return _category_complement_strength(cat_a, cat_b, lane) >= 0.6


def _is_bad_pair(anchor_name: str, comp_name: str, anchor_cat: str, comp_cat: str, lane: str) -> bool:
    a_text = f"{_normalise_text(anchor_name)} {_normalise_text(anchor_cat)}"
    b_text = f"{_normalise_text(comp_name)} {_normalise_text(comp_cat)}"
    if _is_food_lane(lane):
        for left, right in BAD_PAIR_PATTERNS:
            if (left in a_text and right in b_text) or (left in b_text and right in a_text):
                return True
        if "pasta" in a_text and any(x in b_text for x in ("cream dessert", "cake", "candy")):
            return True
    return False


def _contains_nonfood_text(*parts: str) -> bool:
    text = " ".join(_normalise_text(part) for part in parts if str(part).strip())
    return any(hint in text for hint in NONFOOD_TEXT_HINTS)


def _staple_guardrail_allows(
    anchor_name: str,
    anchor_cat: str,
    anchor_family: str,
    comp_name: str,
    comp_cat: str,
    comp_family: str,
    lane: str,
) -> bool:
    if not _is_food_lane(lane):
        return True
    if not _is_staple_product(anchor_name, anchor_family, anchor_cat):
        return True
    anchor_text = f"{_normalise_text(anchor_name)} {_normalise_text(anchor_family)} {_normalise_text(anchor_cat)}"
    comp_text = f"{_normalise_text(comp_name)} {_normalise_text(comp_family)} {_normalise_text(comp_cat)}"
    allowed_hints: set[str] = set()
    for staple_key, hints in STAPLE_ALLOWED_COMPLEMENT_HINTS.items():
        if staple_key in anchor_text:
            allowed_hints.update(hints)
    if not allowed_hints:
        return True
    return any(hint in comp_text for hint in allowed_hints)


def _nonfood_pair_close_enough(
    group_a: str,
    group_b: str,
    cat_a: str,
    cat_b: str,
    family_a: str,
    family_b: str,
) -> bool:
    if group_a and group_b and group_a == group_b:
        return True
    cat_a_norm = _normalise_text(cat_a)
    cat_b_norm = _normalise_text(cat_b)
    if cat_a_norm and cat_b_norm and cat_a_norm == cat_b_norm:
        return True
    fam_a_norm = _normalise_text(family_a)
    fam_b_norm = _normalise_text(family_b)
    if fam_a_norm and fam_b_norm and fam_a_norm == fam_b_norm:
        return True
    return False


def _classify_product_groups(name: str, category: str, family: str) -> set[str]:
    groups = _group_labels_from_text(name, category, family)
    # Backward-compatible coarse labels used by existing rules.
    if groups & {GROUP_CHIPS, GROUP_CRACKERS, GROUP_COOKIES, GROUP_CHOCOLATE, GROUP_CANDY, GROUP_NUTS}:
        groups.add(GROUP_SNACKS)
    if GROUP_NOODLES_PASTA in groups:
        groups.add(GROUP_NOODLES)
    if groups & {GROUP_BREAD_CARB, GROUP_RICE_GRAINS}:
        groups.add(GROUP_CARBS)
    if groups & {GROUP_CREAM, GROUP_CREAM_CHEESE, GROUP_CHEESE, GROUP_MILK}:
        groups.add(GROUP_DAIRY)
    if groups & {GROUP_TEA, GROUP_COFFEE, GROUP_SODA, GROUP_JUICE}:
        groups.add(GROUP_BEVERAGES)
    if groups & {GROUP_COOKIES, GROUP_CHOCOLATE, GROUP_CANDY}:
        groups.add(GROUP_SWEETS)
    return groups


def _text_has_any(text: str, hints: frozenset[str]) -> bool:
    norm = _normalise_text(text)
    return any(hint in norm for hint in hints)


def _is_fruit_like(groups: set[str], text: str) -> bool:
    if GROUP_DATES in groups:
        return False
    if GROUP_PRODUCE in groups and _text_has_any(text, FRUIT_HINTS):
        return True
    return bool(_text_has_any(text, FRUIT_HINTS))


def _anchor_allowed_for_lane_legacy(
    pid: int,
    lane: str,
    context: PersonalizationContext,
    row: pd.Series | None = None,
) -> bool:
    if int(pid) <= 0:
        return False
    side = "a"
    name = _product_name(int(pid), context, row=row, side=side)
    category = _product_category(int(pid), context, row=row, side=side)
    family = _product_family(int(pid), context, row=row, side=side)
    text = f"{_normalise_text(name)} {_normalise_text(category)} {_normalise_text(family)}"
    groups = _classify_product_groups(name, category, family)
    sem = _product_semantic_group(int(pid), context, row=row, side=side)

    if lane in FOOD_LANE_ORDER:
        if _is_nonfood_product(int(pid), context, row=row, side=side):
            return False
        if _is_packaging_or_utility_item(int(pid), context, row=row, side=side):
            return False
    if lane == LANE_NONFOOD:
        return _is_nonfood_product(int(pid), context, row=row, side=side)

    is_dates_like = GROUP_DATES in groups or "dates" in text
    is_tomato_sauce_like = "tomato paste" in text or "sauce" in text or "ketchup" in text
    is_fat_or_oil = _is_fat_or_oil_item(name, category, family)

    if lane == LANE_MEAL:
        if _text_has_any(text, MEAL_ANCHOR_HARD_BLOCK_HINTS):
            return False
        if "dessert cream" in text or ("cream" in text and "dessert" in text):
            return False
        if _is_fruit_like(groups, text):
            return False
        if groups & {GROUP_CHIPS, GROUP_SNACKS, GROUP_CANDY, GROUP_CHOCOLATE, GROUP_COOKIES, GROUP_CRACKERS}:
            return False
        if groups and groups <= {GROUP_SPICES}:
            return False
        if sem == SEM_SPICES:
            return False
        if is_fat_or_oil and not is_tomato_sauce_like and sem not in {SEM_PROTEIN, SEM_SAUCE}:
            return False
        return True

    if lane == LANE_SNACK:
        if groups & {GROUP_RICE_GRAINS, GROUP_CARBS, GROUP_SPICES, GROUP_PRODUCE, GROUP_PROTEIN, GROUP_NOODLES_PASTA}:
            return False
        if _text_has_any(text, PANTRY_HINTS):
            return False
        return bool((groups & SNACK_ANCHOR_ALLOWED_GROUPS) or (groups & {GROUP_SWEETS, GROUP_MILK, GROUP_DAIRY}))

    if lane == LANE_OCCASION:
        if _text_has_any(text, OCCASION_ANCHOR_HARD_BLOCK_HINTS):
            return False
        if is_fat_or_oil:
            return False
        if groups & {GROUP_PROTEIN, GROUP_RICE_GRAINS, GROUP_NOODLES_PASTA, GROUP_SPICES}:
            return False
        if GROUP_PRODUCE in groups and not is_dates_like:
            return False
        if _is_fruit_like(groups, text) and not is_dates_like:
            return False
        if groups & {GROUP_CHIPS, GROUP_SNACKS}:
            return False
        allowed_occasion_groups = {
            GROUP_TEA,
            GROUP_COFFEE,
            GROUP_COOKIES,
            GROUP_CRACKERS,
            GROUP_DATES,
            GROUP_MILK,
            GROUP_DAIRY,
            GROUP_CREAM,
            GROUP_CREAM_CHEESE,
            GROUP_CHEESE,
            GROUP_SWEETS,
            GROUP_BEVERAGES,
        }
        if groups & {GROUP_SODA, GROUP_JUICE} and not is_dates_like:
            return False
        return bool(groups & allowed_occasion_groups)

    return True


def _anchor_allowed_for_lane(
    pid: int,
    lane: str,
    context: PersonalizationContext,
    row: pd.Series | None = None,
) -> bool:
    if not USE_NEW_BUNDLE_SEMANTICS:
        return _anchor_allowed_for_lane_legacy(pid, lane, context, row=row)
    if int(pid) <= 0:
        return False
    side = "a"
    name, category, family, text = _semantic_product_text(int(pid), context, row=row, side=side)
    roles = semantics.infer_product_roles(name, category, family)
    lane_for_eval = lane if lane in {LANE_MEAL, LANE_SNACK, LANE_OCCASION, LANE_NONFOOD} else LANE_MEAL
    if ENABLE_INTERNAL_STAPLES and lane_for_eval == LANE_MEAL:
        lane_for_eval = semantics.LANE_MEAL
    allowed, _fit, _reason = semantics.anchor_lane_eligibility(lane_for_eval, roles, text)
    if lane in FOOD_LANE_ORDER and (semantics.ROLE_NONFOOD in roles or _is_packaging_or_utility_item(int(pid), context, row=row, side=side)):
        return False
    return bool(allowed)


def _hard_invalid_pair_for_lane_legacy(
    anchor_pid: int,
    complement_pid: int,
    lane: str,
    context: PersonalizationContext,
    pair_row: pd.Series | None = None,
    anchor_groups: set[str] | None = None,
    comp_groups: set[str] | None = None,
) -> bool:
    groups_a = set(anchor_groups or _group_labels_for_pid(anchor_pid, context, row=pair_row, side="a"))
    groups_b = set(comp_groups or _group_labels_for_pid(complement_pid, context, row=pair_row, side="b"))
    name_a = _product_name(anchor_pid, context, row=pair_row, side="a")
    name_b = _product_name(complement_pid, context, row=pair_row, side="b")
    cat_a = _product_category(anchor_pid, context, row=pair_row, side="a")
    cat_b = _product_category(complement_pid, context, row=pair_row, side="b")
    family_a = _product_family(anchor_pid, context, row=pair_row, side="a")
    family_b = _product_family(complement_pid, context, row=pair_row, side="b")
    text_a = f"{_normalise_text(name_a)} {_normalise_text(cat_a)} {_normalise_text(family_a)}"
    text_b = f"{_normalise_text(name_b)} {_normalise_text(cat_b)} {_normalise_text(family_b)}"
    sem_a = _product_semantic_group(anchor_pid, context, row=pair_row, side="a")
    sem_b = _product_semantic_group(complement_pid, context, row=pair_row, side="b")
    sem_pair = _sem_pair(sem_a, sem_b)

    a_nonfood = _is_nonfood_product(anchor_pid, context, row=pair_row, side="a")
    b_nonfood = _is_nonfood_product(complement_pid, context, row=pair_row, side="b")
    if lane in FOOD_LANE_ORDER:
        if a_nonfood or b_nonfood:
            return True
        if _is_packaging_or_utility_item(anchor_pid, context, row=pair_row, side="a"):
            return True
        if _is_packaging_or_utility_item(complement_pid, context, row=pair_row, side="b"):
            return True

        dessert_with_meat = (
            (sem_a == SEM_DESSERT and _text_has_any(text_b, SAVORY_PROTEIN_HINTS))
            or (sem_b == SEM_DESSERT and _text_has_any(text_a, SAVORY_PROTEIN_HINTS))
        )
        if dessert_with_meat:
            return True

        fruit_with_chicken_tuna = (
            (_is_fruit_like(groups_a, text_a) and _text_has_any(text_b, frozenset({"chicken", "tuna"})))
            or (_is_fruit_like(groups_b, text_b) and _text_has_any(text_a, frozenset({"chicken", "tuna"})))
        )
        if fruit_with_chicken_tuna:
            return True

    if lane == LANE_MEAL:
        if sem_pair <= {SEM_SPICES}:
            return True
        if sem_pair <= {SEM_BEVERAGE}:
            return True
        if (SEM_PROTEIN in sem_pair and (_is_fruit_like(groups_a, text_a) or _is_fruit_like(groups_b, text_b))):
            return True
        if SEM_DESSERT in sem_pair and sem_pair & SAVORY_SEMANTIC_GROUPS:
            return True
        snack_starch = (
            ((GROUP_CHIPS in groups_a or GROUP_SNACKS in groups_a) and (GROUP_RICE_GRAINS in groups_b or GROUP_NOODLES_PASTA in groups_b or GROUP_BREAD_CARB in groups_b))
            or ((GROUP_CHIPS in groups_b or GROUP_SNACKS in groups_b) and (GROUP_RICE_GRAINS in groups_a or GROUP_NOODLES_PASTA in groups_a or GROUP_BREAD_CARB in groups_a))
        )
        if snack_starch:
            return True
        pantry_a = _text_has_any(text_a, PANTRY_HINTS)
        pantry_b = _text_has_any(text_b, PANTRY_HINTS)
        if pantry_a and pantry_b and not (SEM_SAUCE in sem_pair or SEM_PROTEIN in sem_pair):
            return True
        if ("tuna" in text_a and "flour" in text_b) or ("tuna" in text_b and "flour" in text_a):
            return True
        if ("eggs" in text_a and "topokki" in text_b and "cream" in text_b) or ("eggs" in text_b and "topokki" in text_a and "cream" in text_a):
            return True
        if ("oats" in text_a and "chicken" in text_b) or ("oats" in text_b and "chicken" in text_a):
            return True
        if ("tomato paste" in text_a and "samosa" in text_b) or ("tomato paste" in text_b and "samosa" in text_a):
            return True

    if lane == LANE_OCCASION:
        if "water" in text_a or "water" in text_b:
            return True
        if ("tea" in text_a or "coffee" in text_a) and ("samosa" in text_b or "chips" in text_b):
            return True
        if ("tea" in text_b or "coffee" in text_b) and ("samosa" in text_a or "chips" in text_a):
            return True
        if (("cream" in text_a or "cooking cream" in text_a) and ("vimto" in text_b or "syrup" in text_b)) or (
            ("cream" in text_b or "cooking cream" in text_b) and ("vimto" in text_a or "syrup" in text_a)
        ):
            return True
        if groups_a & {GROUP_CHIPS, GROUP_PROTEIN, GROUP_RICE_GRAINS, GROUP_NOODLES_PASTA, GROUP_SPICES}:
            return True
        if groups_b & {GROUP_CHIPS, GROUP_PROTEIN, GROUP_RICE_GRAINS, GROUP_NOODLES_PASTA, GROUP_SPICES}:
            return True

    return False


def _hard_invalid_pair_for_lane(
    anchor_pid: int,
    complement_pid: int,
    lane: str,
    context: PersonalizationContext,
    pair_row: pd.Series | None = None,
    anchor_groups: set[str] | None = None,
    comp_groups: set[str] | None = None,
) -> bool:
    del anchor_groups, comp_groups
    if not USE_NEW_BUNDLE_SEMANTICS:
        return _hard_invalid_pair_for_lane_legacy(
            anchor_pid,
            complement_pid,
            lane,
            context,
            pair_row=pair_row,
            anchor_groups=None,
            comp_groups=None,
        )
    lane_eval = lane if lane in {LANE_MEAL, LANE_SNACK, LANE_OCCASION, LANE_NONFOOD} else LANE_MEAL
    sem = _semantic_pair_snapshot(int(anchor_pid), int(complement_pid), lane_eval, context, pair_row=pair_row)
    return bool(sem.hard_invalid)


def _semantic_penalty_for_lane_legacy(
    anchor_pid: int,
    complement_pid: int,
    lane: str,
    context: PersonalizationContext,
    *,
    cp_score: float = 0.0,
    recipe_compat: float = 0.0,
    pair_row: pd.Series | None = None,
    anchor_groups: set[str] | None = None,
    comp_groups: set[str] | None = None,
) -> float:
    groups_a = set(anchor_groups or _group_labels_for_pid(anchor_pid, context, row=pair_row, side="a"))
    groups_b = set(comp_groups or _group_labels_for_pid(complement_pid, context, row=pair_row, side="b"))
    name_a = _product_name(anchor_pid, context, row=pair_row, side="a")
    name_b = _product_name(complement_pid, context, row=pair_row, side="b")
    text_a = _normalise_text(name_a)
    text_b = _normalise_text(name_b)
    sem_a = _product_semantic_group(anchor_pid, context, row=pair_row, side="a")
    sem_b = _product_semantic_group(complement_pid, context, row=pair_row, side="b")
    sem_pair = _sem_pair(sem_a, sem_b)

    cp = float(cp_score)
    recipe = float(recipe_compat)
    penalty = 0.0

    if sem_pair == frozenset({SEM_DESSERT, SEM_DAIRY}) and (cp < 32.0 and recipe < 0.18):
        penalty += 0.10
    if lane in {LANE_SNACK, LANE_OCCASION}:
        tea_sweet_weak = (
            (GROUP_TEA in groups_a or GROUP_COFFEE in groups_a)
            and (GROUP_SWEETS in groups_b or sem_b == SEM_DESSERT)
            and GROUP_COOKIES not in groups_b
            and GROUP_CRACKERS not in groups_b
        ) or (
            (GROUP_TEA in groups_b or GROUP_COFFEE in groups_b)
            and (GROUP_SWEETS in groups_a or sem_a == SEM_DESSERT)
            and GROUP_COOKIES not in groups_a
            and GROUP_CRACKERS not in groups_a
        )
        if tea_sweet_weak and cp < 30.0 and recipe < 0.16:
            penalty += 0.12

    condensed_syrup = (("condensed" in text_a and ("syrup" in text_b or "vimto" in text_b)) or ("condensed" in text_b and ("syrup" in text_a or "vimto" in text_a)))
    if condensed_syrup:
        penalty += 0.30

    cheese_chips = (
        ((GROUP_CHEESE in groups_a or GROUP_CREAM_CHEESE in groups_a) and (GROUP_CHIPS in groups_b))
        or ((GROUP_CHEESE in groups_b or GROUP_CREAM_CHEESE in groups_b) and (GROUP_CHIPS in groups_a))
    )
    if cheese_chips and cp < 30.0 and recipe < 0.18:
        penalty += 0.10

    oats_milky = (("oats" in text_a and (GROUP_MILK in groups_b or GROUP_TEA in groups_b or GROUP_COFFEE in groups_b)) or ("oats" in text_b and (GROUP_MILK in groups_a or GROUP_TEA in groups_a or GROUP_COFFEE in groups_a)))
    if lane in {LANE_SNACK, LANE_OCCASION} and oats_milky and cp < 34.0:
        penalty += 0.14

    return float(min(0.60, max(0.0, penalty)))


def _semantic_penalty_for_lane(
    anchor_pid: int,
    complement_pid: int,
    lane: str,
    context: PersonalizationContext,
    *,
    cp_score: float = 0.0,
    recipe_compat: float = 0.0,
    pair_row: pd.Series | None = None,
    anchor_groups: set[str] | None = None,
    comp_groups: set[str] | None = None,
) -> float:
    if not USE_NEW_BUNDLE_SEMANTICS:
        return _semantic_penalty_for_lane_legacy(
            anchor_pid,
            complement_pid,
            lane,
            context,
            cp_score=cp_score,
            recipe_compat=recipe_compat,
            pair_row=pair_row,
            anchor_groups=anchor_groups,
            comp_groups=comp_groups,
        )
    sem = _semantic_pair_snapshot(int(anchor_pid), int(complement_pid), lane, context, pair_row=pair_row)
    name_a, cat_a, fam_a, text_a = _semantic_product_text(int(anchor_pid), context, row=pair_row, side="a")
    name_b, cat_b, fam_b, text_b = _semantic_product_text(int(complement_pid), context, row=pair_row, side="b")
    del name_a, cat_a, fam_a, name_b, cat_b, fam_b, cp_score, recipe_compat, anchor_groups, comp_groups
    return float(
        semantics.semantic_soft_penalty(
            lane=lane if lane in {LANE_MEAL, LANE_SNACK, LANE_OCCASION, LANE_NONFOOD} else LANE_MEAL,
            relation=sem.relation,
            strength=sem.strength,
            roles_a=frozenset(sem.roles_a),
            roles_b=frozenset(sem.roles_b),
            text_a=text_a,
            text_b=text_b,
        )
    )


def _meal_guardrail_reject_reason(
    anchor_groups: set[str],
    comp_groups: set[str],
    anchor_name: str,
    comp_name: str,
) -> str | None:
    if GROUP_PRODUCE in anchor_groups and (GROUP_CHIPS in comp_groups or GROUP_SNACKS in comp_groups):
        return MEAL_REJECT_PRODUCE_SNACK_KEY
    if GROUP_PRODUCE in anchor_groups and (GROUP_NOODLES_PASTA in comp_groups or GROUP_NOODLES in comp_groups):
        return MEAL_REJECT_PRODUCE_NOODLES_KEY
    comp_name_norm = _normalise_text(comp_name)
    if GROUP_PRODUCE in anchor_groups and ("indomie" in comp_name_norm or "instant noodle" in comp_name_norm):
        return MEAL_REJECT_PRODUCE_NOODLES_KEY
    return None


def _is_low_premium_anchor(name: str, family: str) -> bool:
    tokens = _token_set(name)
    if tokens & LOW_PREMIUM_STAPLES:
        return True
    fam = _normalise_text(family)
    return any(token in fam for token in LOW_PREMIUM_STAPLES)


def _weak_semantic_evidence_ok(
    *,
    lane: str,
    cp_score: float,
    recipe_compat: float,
    prior_bonus: float,
    pair_count: int,
    template_strength: float,
    category_strength: float,
) -> bool:
    cp_threshold = float(LANE_CP_THRESHOLDS.get(lane, STRICT_COPURCHASE_MIN))
    pair_count_threshold = int(LANE_PAIR_COUNT_THRESHOLDS.get(lane, STRICT_PAIR_COUNT_MIN))
    recipe_threshold = float(LANE_RECIPE_THRESHOLDS.get(lane, STRICT_RECIPE_COMPAT_MIN))
    cp_ok = float(cp_score) >= cp_threshold and int(pair_count) >= pair_count_threshold
    recipe_ok = float(recipe_compat) >= recipe_threshold
    template_ok = float(template_strength) >= 0.6
    category_ok = float(category_strength) >= 0.6
    prior_ok = float(prior_bonus) > 0.0

    if lane == LANE_MEAL:
        return bool(cp_ok and (recipe_ok or prior_ok or (template_ok and category_ok)))
    if lane == LANE_SNACK:
        return bool(cp_ok and template_ok and (recipe_ok or int(pair_count) >= 12))
    if lane == LANE_OCCASION:
        return bool(cp_ok and (prior_ok or (template_ok and recipe_ok)))
    return bool(cp_ok or recipe_ok)


def _passes_complement_gate(
    anchor: int,
    complement: int,
    context: PersonalizationContext,
    cp_score: float,
    recipe_compat: float,
    prior_bonus: float,
    lane: str = LANE_MEAL,
    pair_count: int = 0,
    pair_row: pd.Series | None = None,
    reject_counters: dict[str, int] | None = None,
    analysis: PairAnalysis | None = None,
    serving_telemetry: ServingTelemetry | None = None,
) -> bool:
    if complement <= 0:
        return False
    pair = analysis or _pair_analysis(int(anchor), int(complement), lane, context, pair_row=pair_row)
    anchor_name = pair.anchor_name
    comp_name = pair.complement_name
    anchor_family = pair.anchor_family
    comp_family = pair.complement_family
    anchor_cat = pair.anchor_category
    comp_cat = pair.complement_category
    anchor_groups = pair.anchor_groups
    comp_groups = pair.complement_groups
    anchor_nonfood = pair.anchor_nonfood
    comp_nonfood = pair.complement_nonfood
    template_strength = _template_match_strength(anchor_name, comp_name, lane)
    category_strength = _category_complement_strength(anchor_cat, comp_cat, lane)
    pattern_ok = template_strength >= 0.6
    category_ok = category_strength >= 0.6
    prior_ok = float(prior_bonus) > 0.0

    if _is_bad_pair(anchor_name, comp_name, anchor_cat, comp_cat, lane):
        return False
    if lane in FOOD_LANE_ORDER and _contains_nonfood_text(anchor_name, comp_name, anchor_cat, comp_cat):
        return False
    if _fails_sweet_savory_pair(anchor_name, anchor_family, comp_name, comp_family) and float(cp_score) < STRICT_COPURCHASE_MIN:
        return False
    if not _suspicious_default_evidence_ok(
        lane=lane,
        anchor_name=anchor_name,
        anchor_cat=anchor_cat,
        anchor_family=anchor_family,
        comp_name=comp_name,
        comp_cat=comp_cat,
        comp_family=comp_family,
        cp_score=float(cp_score),
        recipe_compat=float(recipe_compat),
        prior_bonus=float(prior_bonus),
        pair_count=int(pair_count),
    ):
        return False

    if lane == LANE_NONFOOD:
        if not (anchor_nonfood and comp_nonfood):
            return False
        group_a = _nonfood_group_for_pid(int(anchor), context, row=pair_row, side="a")
        group_b = _nonfood_group_for_pid(int(complement), context, row=pair_row, side="b")
        if not group_a or not group_b:
            return False
        return bool(group_a == group_b)

    if pair.anchor_packaging:
        return False
    if pair.complement_packaging:
        return False
    if anchor_nonfood or comp_nonfood:
        return False
    if _hard_invalid_pair_for_lane(
        int(anchor),
        int(complement),
        lane,
        context,
        pair_row=pair_row,
        anchor_groups=anchor_groups,
        comp_groups=comp_groups,
    ):
        _record_serving_telemetry(serving_telemetry, lane, "rejected_hard_invalid")
        return False
    if USE_NEW_BUNDLE_SEMANTICS:
        sem = pair.semantic
        if sem.hard_invalid:
            _record_serving_telemetry(serving_telemetry, lane, "rejected_hard_invalid")
            return False
        if STRICT_SEMANTIC_FILTERING and not sem.lane_allowed:
            _record_serving_telemetry(serving_telemetry, lane, "rejected_lane_disallow")
            return False
        if str(sem.strength) == semantics.STRENGTH_TRASH:
            return False
        if not pair.visible_ok:
            _record_serving_telemetry(serving_telemetry, lane, "rejected_visible_expression")
            return False
        if lane == LANE_SNACK and _snack_pattern_key(anchor_groups, comp_groups) is None:
            return False
        if _is_food_lane(lane) and str(sem.strength) == semantics.STRENGTH_WEAK:
            _record_serving_telemetry(serving_telemetry, lane, "rejected_weak_strength")
            return False
        return True
    feedback_override = _feedback_pair_override(int(anchor), int(complement))
    if feedback_override and lane in FOOD_LANE_ORDER:
        return True
    if lane == LANE_MEAL:
        reject_reason = _meal_guardrail_reject_reason(anchor_groups, comp_groups, anchor_name, comp_name)
        if reject_reason:
            if reject_counters is not None:
                reject_counters[reject_reason] = int(reject_counters.get(reject_reason, 0)) + 1
            return False
    if not _semantic_lane_compatible(
        int(anchor),
        int(complement),
        lane,
        context,
        pair_row=pair_row,
        anchor_groups=anchor_groups,
        comp_groups=comp_groups,
    ):
        _record_serving_telemetry(serving_telemetry, lane, "rejected_lane_disallow")
        return False
    if not _passes_lane_negative_rules(
        int(anchor),
        int(complement),
        lane,
        context,
        float(recipe_compat),
        anchor_groups=anchor_groups,
        cand_groups=comp_groups,
        pair_row=pair_row,
    ):
        return False
    if lane == LANE_SNACK and _snack_pattern_key(anchor_groups, comp_groups) is None:
        return False
    if not _staple_guardrail_allows(
        anchor_name,
        anchor_cat,
        anchor_family,
        comp_name,
        comp_cat,
        comp_family,
        lane,
    ):
        return False

    cp_threshold = float(LANE_CP_THRESHOLDS.get(lane, STRICT_COPURCHASE_MIN))
    pair_count_threshold = int(LANE_PAIR_COUNT_THRESHOLDS.get(lane, STRICT_PAIR_COUNT_MIN))
    recipe_threshold = float(LANE_RECIPE_THRESHOLDS.get(lane, STRICT_RECIPE_COMPAT_MIN))
    recipe_ok = float(recipe_compat) >= recipe_threshold
    cp_ok = float(cp_score) >= cp_threshold and int(pair_count) >= pair_count_threshold

    if lane == LANE_MEAL:
        compatible = recipe_ok or category_ok or pattern_ok or cp_ok or prior_ok
    elif lane == LANE_SNACK:
        compatible = pattern_ok or category_ok or cp_ok or recipe_ok or prior_ok
    else:
        compatible = pattern_ok or category_ok or cp_ok or prior_ok or recipe_ok
    if not compatible:
        return False

    if _is_utility_like_candidate(comp_name, comp_cat):
        weak_evidence = float(cp_score) <= UTILITY_WEAK_CP_MAX and float(recipe_compat) <= UTILITY_WEAK_RECIPE_MAX and float(
            prior_bonus
        ) <= 0.0
        if lane == LANE_OCCASION:
            if not (pattern_ok or category_ok or prior_ok):
                return False
        elif weak_evidence:
            return False
    return True


def _profile_brand_preference_by_family(profile: PersonProfile, context: PersonalizationContext) -> dict[str, str]:
    counts: dict[str, dict[str, int]] = {}
    for pid in profile.history_product_ids:
        pid_int = int(pid)
        fam = str(context.product_family_by_id.get(pid_int, "")).strip().lower()
        brand = str(context.product_brand_by_id.get(pid_int, "")).strip().lower()
        if not fam or not brand:
            continue
        fam_counts = counts.setdefault(fam, {})
        fam_counts[brand] = int(fam_counts.get(brand, 0)) + int(profile.history_counts.get(pid_int, 1))
    preferred: dict[str, str] = {}
    for fam, fam_counts in counts.items():
        preferred[fam] = max(fam_counts.items(), key=lambda kv: kv[1])[0]
    return preferred


def _passes_pair_filters(
    anchor: int,
    complement: int,
    history_ids: set[int],
    context: PersonalizationContext,
    lane: str = LANE_MEAL,
    pair_row: pd.Series | None = None,
    analysis: PairAnalysis | None = None,
) -> bool:
    if anchor <= 0 or complement <= 0 or anchor == complement:
        return False
    pair = analysis or _pair_analysis(int(anchor), int(complement), lane, context, pair_row=pair_row)
    anchor_domain = _product_domain(int(anchor), context, row=pair_row, side="a")
    complement_domain = _product_domain(int(complement), context, row=pair_row, side="b")
    if anchor_domain != complement_domain:
        return False
    if anchor_domain == "appliance" and lane != LANE_NONFOOD:
        return False
    anchor_nonfood = pair.anchor_nonfood
    comp_nonfood = pair.complement_nonfood
    if _is_food_lane(lane):
        if pair.anchor_packaging:
            return False
        if pair.complement_packaging:
            return False
        if anchor_nonfood or comp_nonfood:
            return False
    else:
        if not (anchor_nonfood and comp_nonfood):
            return False

    family_a = pair.anchor_family
    family_b = pair.complement_family
    if _is_food_lane(lane) and family_a and family_b and family_a == family_b:
        return False

    name_a = pair.anchor_name
    name_b = pair.complement_name
    if _is_same_variant(name_a, name_b):
        return False
    if _is_food_lane(lane) and _contains_nonfood_text(name_a, name_b):
        return False
    if _hard_invalid_pair_for_lane(
        int(anchor),
        int(complement),
        lane,
        context,
        pair_row=pair_row,
    ):
        return False
    if USE_NEW_BUNDLE_SEMANTICS:
        sem = pair.semantic
        if sem.hard_invalid:
            return False
        if STRICT_SEMANTIC_FILTERING and not sem.lane_allowed:
            return False
        if str(sem.strength) == semantics.STRENGTH_TRASH:
            return False
        if not pair.visible_ok:
            return False
        if lane == LANE_SNACK:
            if _snack_pattern_key(pair.anchor_groups, pair.complement_groups) is None:
                return False
        if _is_food_lane(lane):
            return True
        group_a = _nonfood_group_for_pid(anchor, context, row=pair_row, side="a")
        group_b = _nonfood_group_for_pid(complement, context, row=pair_row, side="b")
        return bool(group_a and group_b and group_a == group_b)
    feedback_override = _feedback_pair_override(int(anchor), int(complement))
    if feedback_override and _is_food_lane(lane):
        return True

    if not _semantic_lane_compatible(
        int(anchor),
        int(complement),
        lane,
        context,
        pair_row=pair_row,
    ):
        return False
    if _is_food_lane(lane):
        return True
    group_a = _nonfood_group_for_pid(anchor, context, row=pair_row, side="a")
    group_b = _nonfood_group_for_pid(complement, context, row=pair_row, side="b")
    if not group_a or not group_b:
        return False
    if group_a != group_b:
        return False
    return True


def _score_personal_candidate(
    anchor: int,
    complement: int,
    cp_score: float,
    source: str,
    context: PersonalizationContext,
    feedback_lookup: dict[tuple[str, str], float],
    history_ids: set[int],
    preferred_brand_by_family: dict[str, str],
    rng: random.Random,
) -> tuple[float, float, float, float, bool, float, float]:
    cp_norm = max(0.0, min(1.0, float(cp_score) / 100.0))
    recipe_compat = _recipe_compatibility(anchor, complement, context)
    anchor_name = str(context.product_name_by_id.get(anchor, ""))
    complement_name = str(context.product_name_by_id.get(complement, ""))
    prior_bonus = _known_prior_bonus(
        anchor_name,
        complement_name,
    )
    feedback_multiplier, feedback_conflict = pair_feedback_multiplier(anchor_name, complement_name, feedback_lookup)
    top_bundle_bonus = 1.0 if source == "top_bundle" else 0.0
    history_bonus = 1.0 if int(complement) in history_ids else 0.0
    comp_family = str(context.product_family_by_id.get(int(complement), "")).strip().lower()
    comp_brand = str(context.product_brand_by_id.get(int(complement), "")).strip().lower()
    preferred_brand = str(preferred_brand_by_family.get(comp_family, "")).strip().lower()
    brand_signal = 0.0
    if preferred_brand and comp_brand:
        if comp_brand == preferred_brand:
            brand_signal = 1.0
        else:
            brand_signal = -1.0
    jitter = 0.0
    score = (
        cp_norm * 0.34
        + recipe_compat * 0.28
        + top_bundle_bonus * 0.20
        + prior_bonus * 0.10
        + history_bonus * HISTORY_COMPLEMENT_BOOST
        + (BRAND_MATCH_BOOST if brand_signal > 0 else 0.0)
        - (BRAND_MISMATCH_PENALTY if brand_signal < 0 else 0.0)
        + jitter
    )
    score *= feedback_multiplier
    return (
        score,
        recipe_compat,
        prior_bonus,
        float(feedback_multiplier),
        bool(feedback_conflict),
        float(history_bonus),
        float(brand_signal),
    )


def _rank_anchors_by_lane(
    profile: PersonProfile,
    context: PersonalizationContext,
    rng: random.Random,
    base_dir: Path,
) -> dict[str, list[tuple[int, float]]]:
    history_ids = [int(pid) for pid in profile.history_product_ids if int(pid) > 0]
    valid = [pid for pid in history_ids if pid not in context.non_food_ids]
    ranked_by_lane: dict[str, list[tuple[int, float]]] = {lane: [] for lane in LANE_ORDER}
    if not valid:
        return ranked_by_lane

    history_counts = {pid: float(profile.history_counts.get(pid, 1)) for pid in valid}
    recipe_scores = {pid: float(context.recipe_score_by_id.get(pid, 0.0)) for pid in valid}
    freq_norm = _normalised_values(history_counts)
    recipe_norm = _normalised_values(recipe_scores)
    category_importance_lookup = _load_category_importance_lookup(base_dir)
    meal_ratio, snack_ratio, _occasion_ratio = _history_lane_ratios(profile, context)
    quotas = _top10_lane_quotas(meal_ratio=meal_ratio, snack_ratio=snack_ratio)

    staple_count = sum(
        1
        for pid in valid
        if _is_staple_product(
            str(context.product_name_by_id.get(pid, "")),
            str(context.product_family_by_id.get(pid, "")),
        )
    )
    profile_staple_ratio = float(staple_count / max(1, len(valid)))

    for pid in valid:
        lane = _classify_anchor_lane(pid, context)
        if lane not in ranked_by_lane:
            lane = LANE_MEAL
        history_score = float(freq_norm.get(pid, 0.0))
        recipe_score = float(recipe_norm.get(pid, 0.0))
        category_score = _category_importance_norm(pid, context, category_importance_lookup)
        snackiness = _lane_hint_strength(pid, context, LANE_SNACK)
        occasion_fit = _lane_hint_strength(pid, context, LANE_OCCASION)

        if lane == LANE_MEAL:
            score = 0.45 * history_score + 0.35 * recipe_score + 0.20 * category_score
        elif lane == LANE_SNACK:
            score = 0.60 * history_score + 0.30 * snackiness + 0.10 * category_score
        else:
            score = 0.70 * history_score + 0.30 * occasion_fit + 0.05 * recipe_score

        if lane != LANE_OCCASION and _is_staple_product(
            str(context.product_name_by_id.get(pid, "")),
            str(context.product_family_by_id.get(pid, "")),
        ):
            if profile_staple_ratio >= STAPLE_HEAVY_PROFILE_THRESHOLD:
                score *= STAPLE_ANCHOR_PENALTY_STAPLE_HEAVY
            else:
                score *= STAPLE_ANCHOR_PENALTY

        ranked_by_lane[lane].append((pid, float(score)))

    for lane in LANE_ORDER:
        ranked_by_lane[lane].sort(key=_anchor_rank_sort_key)
        ranked_by_lane[lane] = ranked_by_lane[lane][: max(1, quotas.get(lane, 1))]

    selected_ids: set[int] = set()
    for lane in LANE_ORDER:
        selected_ids.update(pid for pid, _ in ranked_by_lane[lane])

    if len(selected_ids) < TOP10_ANCHOR_SIZE:
        overflow: list[tuple[int, float, str]] = []
        for pid in valid:
            lane = _classify_anchor_lane(pid, context)
            if lane not in ranked_by_lane:
                lane = LANE_MEAL
            for p, score in ranked_by_lane[lane]:
                if p == pid:
                    overflow.append((pid, score, lane))
                    break
            else:
                # Recompute a lightweight score for overflow fill.
                history_score = float(freq_norm.get(pid, 0.0))
                recipe_score = float(recipe_norm.get(pid, 0.0))
                category_score = _category_importance_norm(pid, context, category_importance_lookup)
                overflow.append((pid, 0.55 * history_score + 0.25 * recipe_score + 0.20 * category_score, lane))
        overflow.sort(key=lambda x: (-float(x[1]), int(x[0]), str(x[2])))
        for pid, score, lane in overflow:
            if pid in selected_ids:
                continue
            ranked_by_lane[lane].append((pid, score))
            selected_ids.add(pid)
            if len(selected_ids) >= TOP10_ANCHOR_SIZE:
                break

    return ranked_by_lane


def _rank_anchors(profile: PersonProfile, context: PersonalizationContext, rng: random.Random) -> list[int]:
    ranked_by_lane = _rank_anchors_by_lane(profile, context, rng, get_paths().project_root)
    merged: list[tuple[int, float]] = []
    seen: set[int] = set()
    for lane in LANE_ORDER:
        for pid, score in ranked_by_lane.get(lane, ()):
            if pid in seen:
                continue
            seen.add(pid)
            merged.append((pid, score))
    merged.sort(key=_anchor_rank_sort_key)
    return [pid for pid, _ in merged]


def _weighted_choice_pid(candidates: list[tuple[int, float]], rng: random.Random) -> int | None:
    del rng
    if not candidates:
        return None
    top = sorted(candidates, key=_anchor_rank_sort_key)
    return int(top[0][0])


def _pick_three_lane_anchors(
    lane_ranked: dict[str, list[tuple[int, float]]],
    rng: random.Random,
) -> dict[str, int] | None:
    selected: dict[str, int] = {}
    used: set[int] = set()
    all_candidates: list[tuple[int, float]] = []
    for lane in LANE_ORDER:
        all_candidates.extend(lane_ranked.get(lane, ()))
    all_candidates.sort(key=_anchor_rank_sort_key)

    for lane in LANE_ORDER:
        lane_candidates = sorted(
            [(pid, score) for pid, score in lane_ranked.get(lane, ()) if pid not in used],
            key=_anchor_rank_sort_key,
        )
        if not lane_candidates:
            lane_candidates = [(pid, score) for pid, score in all_candidates if pid not in used]
        picked = _weighted_choice_pid(lane_candidates, rng)
        if picked is None:
            return None
        selected[lane] = picked
        used.add(picked)
    return selected


def _apply_anchor_overuse_penalty(
    score: float,
    anchor: int,
    context: PersonalizationContext,
    global_anchor_counts: dict[int, int],
) -> float:
    usage = int(global_anchor_counts.get(int(anchor), 0))
    if usage <= 0:
        return float(score)
    anchor_name = str(context.product_name_by_id.get(int(anchor), ""))
    anchor_family = str(context.product_family_by_id.get(int(anchor), ""))
    if not _is_low_premium_anchor(anchor_name, anchor_family):
        return float(score)
    if usage == 1:
        return float(score) * 0.85
    return float(score) * 0.65


def _premium_complement_score(
    *,
    lane: str,
    base_final_score: float,
    cp_score: float,
    recipe_compat: float,
    template_strength: float,
    category_strength: float,
    prior_bonus: float,
    weird_pair: bool,
    family_reuse: int,
    low_premium_anchor_overuse: int,
    source: str,
    history_bonus: float,
    brand_signal: float,
) -> float:
    base_norm = max(0.0, min(1.0, float(base_final_score) / 100.0))
    cp_norm = max(0.0, min(1.0, float(cp_score) / 100.0))
    recipe_norm = max(0.0, min(1.0, float(recipe_compat)))
    template_norm = max(0.0, min(1.0, float(template_strength)))
    category_norm = max(0.0, min(1.0, float(category_strength)))
    prior_norm = max(0.0, min(1.0, float(prior_bonus)))

    recipe_weight = float(LANE_RECIPE_WEIGHTS.get(lane, 0.05))
    cp_weight = float(LANE_CP_WEIGHTS.get(lane, 0.05))
    weird_pair_penalty = 0.25 if weird_pair else 0.0
    family_penalty = min(0.45, 0.25 * max(0, family_reuse))
    overuse_penalty = min(0.24, 0.12 * max(0, low_premium_anchor_overuse))

    top_bundle_bonus = 0.20 if source == "top_bundle" else 0.0
    history_boost = history_bonus * HISTORY_COMPLEMENT_BOOST
    brand_boost = BRAND_MATCH_BOOST if brand_signal > 0 else 0.0
    brand_penalty = BRAND_MISMATCH_PENALTY if brand_signal < 0 else 0.0

    score = (
        0.30 * base_norm
        + 0.28 * template_norm
        + 0.18 * category_norm
        + recipe_weight * recipe_norm
        + cp_weight * cp_norm
        + 0.06 * prior_norm
        + top_bundle_bonus
        + history_boost
        + brand_boost
        - brand_penalty
        - weird_pair_penalty
        - family_penalty
        - overuse_penalty
    )
    return float(score)


def _pick_candidate_for_anchor(
    profile: PersonProfile,
    anchor: int,
    lane: str,
    context: PersonalizationContext,
    top_bundle_rows_by_anchor: dict[int, list[pd.Series]],
    bundle_lookup: dict[tuple[int, int], pd.Series],
    used_pair_keys: set[tuple[int, int]],
    feedback_lookup: dict[tuple[str, str], float],
    rng: random.Random,
    used_complements: set[int],
    used_complement_families: dict[str, int],
    global_anchor_lane_counts: dict[tuple[str, int], int],
    global_anchor_counts: dict[int, int],
    reject_counters: dict[str, int] | None = None,
    blocked_product_ids: set[int] | None = None,
    blocked_themes: set[str] | None = None,
    blocked_pair_fingerprints: set[tuple[str, str]] | None = None,
    blocked_groups: set[str] | None = None,
    allow_anchor_overflow: bool = False,
    allowed_complements: set[int] | None = None,
    serving_telemetry: ServingTelemetry | None = None,
) -> tuple[dict[str, object] | None, tuple[int, int] | None, float, int]:
    history_ids = {int(pid) for pid in profile.history_product_ids}
    preferred_brand_by_family = _profile_brand_preference_by_family(profile, context)
    duplicate_pair_blocked = 0
    blocked_ids = {int(pid) for pid in (blocked_product_ids or set()) if int(pid) > 0}
    allowed_ids = {int(pid) for pid in (allowed_complements or set()) if int(pid) > 0}
    blocked_theme_set = set(blocked_themes or ())
    blocked_fingerprint_set = set(blocked_pair_fingerprints or ())
    blocked_group_set = set(blocked_groups or ())
    top_bundle_scan_limit = _top_bundle_scan_limit(lane)
    cap_key = (str(lane), int(anchor))
    if int(global_anchor_lane_counts.get(cap_key, 0)) >= MAX_SAME_ANCHOR_PER_PAGE and not allow_anchor_overflow:
        return None, None, 0.0, duplicate_pair_blocked
    if int(anchor) in blocked_ids:
        return None, None, 0.0, duplicate_pair_blocked
    if not _anchor_allowed_for_lane(int(anchor), lane, context):
        return None, None, 0.0, duplicate_pair_blocked

    def _evaluate_candidate(
        complement: int,
        cp_score: float,
        source: str,
        bundle_row: pd.Series | None,
    ) -> dict[str, object] | None:
        nonlocal duplicate_pair_blocked
        if allowed_ids and int(complement) not in allowed_ids:
            return None
        if int(complement) in blocked_ids:
            return None
        key = _oriented_pair_key(anchor, complement)
        if key in used_pair_keys:
            duplicate_pair_blocked += 1
            return None
        if complement in used_complements:
            return None
        pair = _pair_analysis(int(anchor), int(complement), lane, context, pair_row=bundle_row)
        if not _passes_pair_filters(anchor, complement, history_ids, context, lane=lane, pair_row=bundle_row, analysis=pair):
            return None
        (
            _personal_score,
            recipe_compat,
            prior_bonus,
            feedback_multiplier,
            feedback_conflict,
            history_bonus,
            brand_signal,
        ) = _score_personal_candidate(
            anchor=anchor,
            complement=complement,
            cp_score=float(cp_score),
            source=source,
            context=context,
            feedback_lookup=feedback_lookup,
            history_ids=history_ids,
            preferred_brand_by_family=preferred_brand_by_family,
            rng=rng,
        )
        pair_count = int(
            _safe_int(
                bundle_row.get("pair_count", bundle_row.get("co_purchase_count", 0)) if bundle_row is not None else 0,
                default=0,
            )
        )
        if not _passes_complement_gate(
            anchor=anchor,
            complement=complement,
            context=context,
            cp_score=float(cp_score),
            recipe_compat=recipe_compat,
            prior_bonus=prior_bonus,
            lane=lane,
            pair_count=pair_count,
            pair_row=bundle_row,
            reject_counters=reject_counters,
            analysis=pair,
            serving_telemetry=serving_telemetry,
        ):
            return None
        semantic_view = pair.semantic
        if USE_NEW_BUNDLE_SEMANTICS:
            if semantic_view.hard_invalid:
                return None
            if STRICT_SEMANTIC_FILTERING and not semantic_view.lane_allowed:
                return None
            if str(semantic_view.strength) == semantics.STRENGTH_TRASH:
                return None
            if ENABLE_INTERNAL_STAPLES and str(semantic_view.internal_lane_fit) == semantics.LANE_STAPLES and lane in {
                LANE_SNACK,
                LANE_OCCASION,
            }:
                return None
        anchor_name = pair.anchor_name
        comp_name = pair.complement_name
        anchor_cat = pair.anchor_category
        comp_cat = pair.complement_category
        anchor_family = pair.anchor_family
        comp_family = pair.complement_family or "other"
        anchor_groups = pair.anchor_groups
        comp_groups = pair.complement_groups
        if lane == LANE_SNACK and not _is_snack_anchor_allowed(anchor_groups):
            return None
        snack_pattern = _snack_pattern_key(anchor_groups, comp_groups) if lane == LANE_SNACK else None
        if lane == LANE_SNACK and snack_pattern is None:
            return None
        if not USE_NEW_BUNDLE_SEMANTICS and not _passes_lane_negative_rules(
            int(anchor),
            int(complement),
            lane,
            context,
            float(recipe_compat),
            anchor_groups=anchor_groups,
            cand_groups=comp_groups,
            pair_row=bundle_row,
        ):
            return None
        theme = _pair_theme(anchor_groups, comp_groups, lane, snack_pattern)
        pair_fingerprint = _pair_fingerprint(
            anchor,
            complement,
            context,
            anchor_groups=anchor_groups,
            comp_groups=comp_groups,
            row=bundle_row,
        )
        if _is_food_lane(lane):
            if theme and theme in blocked_theme_set:
                _record_serving_telemetry(serving_telemetry, lane, "rejected_theme_block")
                return None
            if pair_fingerprint in blocked_fingerprint_set:
                _record_serving_telemetry(serving_telemetry, lane, "rejected_pair_fingerprint")
                return None
            overlap = len({g for g in (anchor_groups | comp_groups) & blocked_group_set if g not in GROUP_OVERLAP_IGNORE})
            if overlap >= 2:
                return None
        family_reuse = int(used_complement_families.get(comp_family, 0))
        template_strength = _template_match_strength(anchor_name, comp_name, lane)
        category_strength = _category_complement_strength(anchor_cat, comp_cat, lane)
        weird_pair = _is_bad_pair(anchor_name, comp_name, anchor_cat, comp_cat, lane)
        base_final_score = _safe_float(
            bundle_row.get("final_score", bundle_row.get("new_final_score", 0.0)) if bundle_row is not None else 0.0,
            default=0.0,
        )
        anchor_overuse = int(max(0, int(global_anchor_counts.get(int(anchor), 0)) - 1))
        personal_score = _premium_complement_score(
            lane=lane,
            base_final_score=float(base_final_score),
            cp_score=float(cp_score),
            recipe_compat=float(recipe_compat),
            template_strength=float(template_strength),
            category_strength=float(category_strength),
            prior_bonus=float(prior_bonus),
            weird_pair=bool(weird_pair),
            family_reuse=int(family_reuse),
            low_premium_anchor_overuse=int(anchor_overuse),
            source=source,
            history_bonus=float(history_bonus),
            brand_signal=float(brand_signal),
        )
        if USE_NEW_BUNDLE_SEMANTICS:
            personal_score += float(semantics.semantic_score_prior(str(semantic_view.strength)))
            personal_score += float(semantic_view.lane_fit_score)
            if ENABLE_INTERNAL_STAPLES and str(semantic_view.internal_lane_fit) == semantics.LANE_STAPLES and lane == LANE_MEAL:
                personal_score -= 0.12
        personal_score *= float(feedback_multiplier)
        personal_score = _apply_anchor_overuse_penalty(float(personal_score), anchor, context, global_anchor_counts)
        personal_score -= _semantic_penalty_for_lane(
            int(anchor),
            int(complement),
            lane,
            context,
            cp_score=float(cp_score),
            recipe_compat=float(recipe_compat),
            pair_row=bundle_row,
            anchor_groups=anchor_groups,
            comp_groups=comp_groups,
        )
        feedback_pair_boost = _feedback_pair_boost(int(anchor), int(complement))
        feedback_pair_penalty = _feedback_pair_penalty(int(anchor), int(complement))
        feedback_pair_class = _feedback_pair_class(int(anchor), int(complement))
        if lane in FOOD_LANE_ORDER and feedback_pair_class == "trash":
            _record_serving_telemetry(serving_telemetry, lane, "rejected_feedback_trash")
            return None
        if feedback_pair_boost > 0.0:
            personal_score += float(feedback_pair_boost)
        if feedback_pair_penalty > 0.0:
            personal_score -= float(feedback_pair_penalty)
        feedback_class_applied = str(feedback_pair_class)
        if feedback_class_applied == "strong":
            personal_score += float(FEEDBACK_STRONG_BOOST)
        elif feedback_class_applied == "staple":
            personal_score += 0.02
        elif feedback_class_applied == "weak":
            personal_score -= float(FEEDBACK_WEAK_PENALTY)
        elif feedback_class_applied == "trash":
            personal_score -= float(FEEDBACK_TRASH_PENALTY)
        feedback_override_applied = _feedback_pair_override(int(anchor), int(complement))
        return {
            "anchor": int(anchor),
            "complement": int(complement),
            "cp_score": float(cp_score),
            "bundle_row": bundle_row,
            "key": key,
            "source": source,
            "personal_score": float(personal_score),
            "recipe_compat": float(recipe_compat),
            "prior_bonus": float(prior_bonus),
            "feedback_multiplier": float(feedback_multiplier),
            "feedback_conflict": int(feedback_conflict),
            "history_bonus": float(history_bonus),
            "brand_signal": float(brand_signal),
            "lane": str(lane),
            "pair_count": int(pair_count),
            "base_final_score": float(base_final_score),
            "template_strength": float(template_strength),
            "category_strength": float(category_strength),
            "snack_pattern": str(snack_pattern or ""),
            "theme": str(theme or ""),
            "pair_fingerprint": pair_fingerprint,
            "group_union": sorted(anchor_groups | comp_groups),
            "feedback_pair_boost": float(feedback_pair_boost),
            "feedback_pair_penalty": float(feedback_pair_penalty),
            "feedback_boost_applied": bool(feedback_pair_boost > 0.0),
            "feedback_penalty_applied": bool(feedback_pair_penalty > 0.0),
            "feedback_override_applied": bool(feedback_override_applied),
            "feedback_class": str(feedback_pair_class),
            "semantic_roles_a": list(semantic_view.roles_a),
            "semantic_roles_b": list(semantic_view.roles_b),
            "pair_relation": str(semantic_view.relation),
            "pair_strength": str(semantic_view.strength),
            "lane_fit_score": float(semantic_view.lane_fit_score),
            "internal_lane_fit": str(semantic_view.internal_lane_fit),
            "semantic_reject_reason": str(semantic_view.hard_invalid_reason or semantic_view.lane_reason or ""),
            "feedback_class_applied": str(feedback_class_applied),
            "feedback_multiplier_applied": float(feedback_multiplier),
            "semantic_engine_version": SEMANTIC_ENGINE_VERSION,
        }

    candidates: list[dict[str, object]] = []
    for row in top_bundle_rows_by_anchor.get(anchor, ())[:top_bundle_scan_limit]:
        a = _safe_int(row.get("product_a"), default=-1)
        b = _safe_int(row.get("product_b"), default=-1)
        if a <= 0 or b <= 0:
            continue
        complement = b if a == anchor else a if b == anchor else -1
        if complement <= 0:
            continue
        cp_score = _safe_float(row.get("purchase_score", row.get("copurchase_score", 0.0)), default=0.0)
        evaluated = _evaluate_candidate(
            complement=int(complement),
            cp_score=float(cp_score),
            source="top_bundle",
            bundle_row=row,
        )
        if evaluated is not None:
            candidates.append(evaluated)
        if len(candidates) >= MAX_CANDIDATES_PER_PROFILE:
            break

    for neighbor, cp_score in list(context.neighbors.get(anchor, ()))[:MAX_COPURCHASE_FALLBACK]:
        complement = int(neighbor)
        evaluated = _evaluate_candidate(
            complement=complement,
            cp_score=float(cp_score),
            source="copurchase_fallback",
            bundle_row=bundle_lookup.get(_pair_key(anchor, complement)),
        )
        if evaluated is not None:
            candidates.append(evaluated)
        if len(candidates) >= MAX_CANDIDATES_PER_PROFILE:
            break

    if allowed_ids:
        neighbor_cp_scores = {int(pid): float(score) for pid, score in context.neighbors.get(int(anchor), ())}
        seen_allowed = {int(_safe_int(candidate.get("complement"), default=-1)) for candidate in candidates}
        for complement in sorted(int(pid) for pid in allowed_ids if int(pid) > 0 and int(pid) not in seen_allowed):
            row = bundle_lookup.get(_pair_key(anchor, int(complement)))
            cp_score = _safe_float(
                row.get("purchase_score", row.get("copurchase_score", 0.0))
                if row is not None
                else neighbor_cp_scores.get(int(complement), 0.0),
                default=0.0,
            )
            if cp_score <= 0.0:
                continue
            evaluated = _evaluate_candidate(
                complement=int(complement),
                cp_score=float(cp_score),
                source="copurchase_fallback",
                bundle_row=row,
            )
            if evaluated is not None:
                candidates.append(evaluated)
            if len(candidates) >= MAX_CANDIDATES_PER_PROFILE:
                break

    if not candidates:
        return None, None, 0.0, duplicate_pair_blocked

    candidates.sort(
        key=_candidate_rank_key
    )
    top_candidates = candidates[: min(PREMIUM_TOP_N, len(candidates))]
    choice = top_candidates[0]
    return choice, _oriented_pair_key(anchor, int(choice["complement"])), float(choice["personal_score"]), duplicate_pair_blocked


def _curated_entries_for_lane(lane: str) -> tuple[dict[str, object], ...]:
    lane_name = str(lane).strip().lower()
    if lane_name in FOOD_LANE_ORDER:
        entries = [entry for entry in TOP_100_CURATED_FOOD_BUNDLES if str(entry.get("lane", "")).strip().lower() == lane_name]
        return tuple(
            sorted(
                entries,
                key=lambda item: (int(_safe_int(item.get("priority", 0), default=0)), str(item.get("id", ""))),
            )
        )
    if lane_name == LANE_NONFOOD:
        return tuple(
            sorted(
                CURATED_CLEANING_FALLBACK_BUNDLES,
                key=lambda item: (int(_safe_int(item.get("priority", 0), default=0)), str(item.get("id", ""))),
            )
        )
    return ()


def _fallback_candidates_for_lane(
    history_ids: set[int],
    lane: str,
    context: PersonalizationContext,
    top_bundle_rows_by_anchor: dict[int, list[pd.Series]],
    bundle_lookup: dict[tuple[int, int], pd.Series],
) -> list[tuple[int, int, pd.Series | None, str]]:
    lane_name = str(lane).strip().lower()
    templates = _curated_entries_for_lane(lane_name)
    if lane_name in FOOD_LANE_ORDER:
        templates = templates[:MAX_CURATED_FALLBACK_TEMPLATES_PER_LANE]
    if not templates:
        return []

    history_pool = [int(pid) for pid in sorted(history_ids) if int(pid) > 0]
    if not history_pool:
        return []
    history_set = set(history_pool)

    source_rank = {"top_bundle": 0, "bundle_lookup": 1, "copurchase": 2}
    evidence_pairs: dict[tuple[int, int], tuple[pd.Series | None, float, str]] = {}

    def _add_evidence(anchor: int, complement: int, row: pd.Series | None, cp_score: float, source: str) -> None:
        if int(anchor) <= 0 or int(complement) <= 0 or int(anchor) == int(complement):
            return
        key = (int(anchor), int(complement))
        current = evidence_pairs.get(key)
        candidate_rank = int(source_rank.get(str(source), 9))
        if current is not None:
            current_rank = int(source_rank.get(str(current[2]), 9))
            if candidate_rank > current_rank:
                return
            if candidate_rank == current_rank and float(cp_score) <= float(current[1]):
                return
        evidence_pairs[key] = (row, float(cp_score), str(source))

    top_bundle_scan_limit = _top_bundle_scan_limit(lane_name)
    for pid in history_pool:
        for row in top_bundle_rows_by_anchor.get(int(pid), ())[:top_bundle_scan_limit]:
            a = _safe_int(row.get("product_a"), default=-1)
            b = _safe_int(row.get("product_b"), default=-1)
            if a <= 0 or b <= 0 or a == b:
                continue
            cp_score = _safe_float(row.get("purchase_score", row.get("copurchase_score", 0.0)), default=0.0)
            _add_evidence(int(a), int(b), row, float(cp_score), "top_bundle")
            _add_evidence(int(b), int(a), row, float(cp_score), "top_bundle")

    for (a, b), row in bundle_lookup.items():
        if int(a) <= 0 or int(b) <= 0 or int(a) == int(b):
            continue
        if int(a) not in history_set and int(b) not in history_set:
            continue
        cp_score = _safe_float(row.get("purchase_score", row.get("copurchase_score", 0.0)), default=0.0)
        _add_evidence(int(a), int(b), row, float(cp_score), "bundle_lookup")
        _add_evidence(int(b), int(a), row, float(cp_score), "bundle_lookup")

    for pid in history_pool:
        for neighbor, cp_score in list(context.neighbors.get(int(pid), ()))[:MAX_COPURCHASE_FALLBACK]:
            row = bundle_lookup.get(_pair_key(int(pid), int(neighbor)))
            _add_evidence(int(pid), int(neighbor), row, float(cp_score), "copurchase")
            if int(neighbor) in history_set:
                _add_evidence(int(neighbor), int(pid), row, float(cp_score), "copurchase")

    if not evidence_pairs:
        return []

    evidence_items = list(evidence_pairs.items())
    pid_text_cache: dict[int, str] = {}
    for (a, b), (_row, _cp_score, _source) in evidence_items:
        if int(a) not in pid_text_cache:
            pid_text_cache[int(a)] = _product_text_for_pid(int(a), context)
        if int(b) not in pid_text_cache:
            pid_text_cache[int(b)] = _product_text_for_pid(int(b), context)

    pair_pass_cache: dict[tuple[int, int], bool] = {}

    def _pair_passes_fallback_filters(anchor: int, complement: int, row: pd.Series | None, cp_score: float) -> bool:
        key = (int(anchor), int(complement))
        cached = pair_pass_cache.get(key)
        if cached is not None:
            return bool(cached)
        analysis = _pair_analysis(int(anchor), int(complement), lane_name, context, pair_row=row)
        if lane_name in FOOD_LANE_ORDER and not _anchor_allowed_for_lane(int(anchor), lane_name, context, row=row):
            pair_pass_cache[key] = False
            return False
        if not _passes_pair_filters(
            int(anchor),
            int(complement),
            history_set,
            context,
            lane=lane_name,
            pair_row=row,
            analysis=analysis,
        ):
            pair_pass_cache[key] = False
            return False
        recipe_compat = _recipe_compatibility(int(anchor), int(complement), context)
        prior_bonus = _known_prior_bonus(analysis.anchor_name, analysis.complement_name)
        pair_count = int(
            _safe_int(row.get("pair_count", row.get("co_purchase_count", 0)) if row is not None else 0, default=0)
        )
        if not _passes_complement_gate(
            anchor=int(anchor),
            complement=int(complement),
            context=context,
            cp_score=float(cp_score),
            recipe_compat=float(recipe_compat),
            prior_bonus=float(prior_bonus),
            lane=lane_name,
            pair_count=int(pair_count),
            pair_row=row,
            analysis=analysis,
        ):
            pair_pass_cache[key] = False
            return False
        if lane_name in FOOD_LANE_ORDER and str(analysis.semantic.strength) == semantics.STRENGTH_WEAK:
            pair_pass_cache[key] = False
            return False
        pair_pass_cache[key] = True
        return True

    ranked: list[tuple[tuple[int, int, int, int, float, int, int], int, int, pd.Series | None, str]] = []
    seen_pair_keys: set[tuple[int, int]] = set()

    for template in templates:
        template_id = str(template.get("id", "")).strip() or "unknown"
        template_priority = int(_safe_int(template.get("priority", 0), default=0))
        anchor_hint = str(template.get("anchor_hint", "")).strip().lower()
        complement_hint = str(template.get("complement_hint", "")).strip().lower()
        if not anchor_hint or not complement_hint:
            continue
        anchor_tokens = _token_set(anchor_hint)
        complement_tokens = _token_set(complement_hint)
        fixed_anchor = int(_safe_int(template.get("anchor_id", -1), default=-1))
        fixed_complement = int(_safe_int(template.get("complement_id", -1), default=-1))

        candidates: list[tuple[tuple[int, int, int, int, float, int, int], int, int, pd.Series | None, str]] = []

        def _evaluate_candidate(anchor: int, complement: int, match_rank: int) -> None:
            evidence = evidence_pairs.get((int(anchor), int(complement)))
            if evidence is None:
                return
            row, cp_score, evidence_source = evidence
            if not _pair_passes_fallback_filters(int(anchor), int(complement), row, float(cp_score)):
                return
            source_group = str(template.get("source_group", "fallback")).strip().lower()
            if source_group == "fallback_cleaning":
                source = f"fallback_cleaning:{lane_name}:{template_id}"
            else:
                source = f"fallback:{lane_name}:{template_id}"
            rank_key = (
                int(template_priority),
                int(int(anchor) not in history_set),
                int(match_rank),
                int(source_rank.get(evidence_source, 9)),
                -float(cp_score),
                int(anchor),
                int(complement),
            )
            candidates.append((rank_key, int(anchor), int(complement), row, source))

        if fixed_anchor > 0 and fixed_complement > 0:
            _evaluate_candidate(int(fixed_anchor), int(fixed_complement), match_rank=0)
            _evaluate_candidate(int(fixed_complement), int(fixed_anchor), match_rank=0)

        for (a, b), (_row, _cp_score, _evidence_source) in evidence_items:
            if fixed_anchor > 0 and fixed_complement > 0:
                continue
            text_a = pid_text_cache.get(int(a), "")
            text_b = pid_text_cache.get(int(b), "")
            direct = bool(anchor_tokens) and bool(complement_tokens) and any(tok in text_a for tok in anchor_tokens) and any(
                tok in text_b for tok in complement_tokens
            )
            inverse = bool(anchor_tokens) and bool(complement_tokens) and any(tok in text_b for tok in anchor_tokens) and any(
                tok in text_a for tok in complement_tokens
            )
            if direct:
                _evaluate_candidate(int(a), int(b), match_rank=1)
            if inverse:
                _evaluate_candidate(int(b), int(a), match_rank=2)

        candidates.sort(key=lambda item: item[0])
        for _rank_key, anchor, complement, row, source in candidates:
            pair_key = _pair_key(int(anchor), int(complement))
            if pair_key in seen_pair_keys:
                continue
            seen_pair_keys.add(pair_key)
            ranked.append((_rank_key, int(anchor), int(complement), row, source))
            break

    ranked.sort(key=lambda item: item[0])
    return [(int(anchor), int(complement), row, str(source)) for _rk, anchor, complement, row, source in ranked]


def _swap_pair_fields(display: dict[str, object]) -> None:
    swap_cols = [
        ("product_a", "product_b"),
        ("product_a_name", "product_b_name"),
        ("product_a_price", "product_b_price"),
        ("price_a_sar", "price_b_sar"),
        ("price_after_discount_a", "price_after_discount_b"),
        ("price_after_a_sar", "price_after_b_sar"),
        ("discount_a", "discount_b"),
        ("discount_pred_a", "discount_pred_b"),
        ("product_a_picture", "product_b_picture"),
        ("category_a", "category_b"),
        ("product_family_a", "product_family_b"),
    ]
    for left, right in swap_cols:
        display[left], display[right] = display.get(right), display.get(left)
    free = str(display.get("free_product", "")).strip().lower()
    if free == "product_a":
        display["free_product"] = "product_b"
    elif free == "product_b":
        display["free_product"] = "product_a"


def _force_item2_free(display: dict[str, object]) -> None:
    display["free_product"] = "product_b"
    if "price_after_discount_b" in display:
        display["price_after_discount_b"] = 0.0
    if "price_after_b_sar" in display:
        display["price_after_b_sar"] = "0.00"
    else:
        display["price_after_b_sar"] = "0.00"


def _price_for_swap(display: dict[str, object], side: str, context: PersonalizationContext) -> float | None:
    for key in (
        f"product_{side}_price",
        f"price_after_discount_{side}",
        f"price_after_{side}_sar",
        f"price_{side}_sar",
    ):
        if key not in display:
            continue
        value = _safe_float(display.get(key), default=0.0)
        if value > 0:
            return float(value)
    del context
    return None


def _swap_preserves_lane_semantics(
    display: dict[str, object],
    lane: str,
    context: PersonalizationContext,
) -> bool:
    candidate = dict(display)
    _swap_pair_fields(candidate)
    a_id = _safe_int(candidate.get("product_a"), default=-1)
    b_id = _safe_int(candidate.get("product_b"), default=-1)
    if a_id <= 0 or b_id <= 0:
        return False
    a_name = str(candidate.get("product_a_name", context.product_name_by_id.get(a_id, "")))
    b_name = str(candidate.get("product_b_name", context.product_name_by_id.get(b_id, "")))
    a_cat = _product_category(a_id, context)
    b_cat = _product_category(b_id, context)
    a_family = _product_family(a_id, context)
    b_family = _product_family(b_id, context)
    a_nonfood = _is_nonfood_product(a_id, context)
    b_nonfood = _is_nonfood_product(b_id, context)
    if lane in FOOD_LANE_ORDER and (a_nonfood or b_nonfood):
        return False
    if lane == LANE_NONFOOD and not (a_nonfood and b_nonfood):
        return False
    if lane == LANE_NONFOOD:
        group_a = _nonfood_group(a_name, a_cat, a_family)
        group_b = _nonfood_group(b_name, b_cat, b_family)
        if not _nonfood_pair_close_enough(group_a, group_b, a_cat, b_cat, a_family, b_family):
            return False
    if USE_NEW_BUNDLE_SEMANTICS:
        sem = _semantic_pair_snapshot(a_id, b_id, lane, context)
        if sem.hard_invalid or (STRICT_SEMANTIC_FILTERING and not sem.lane_allowed):
            return False
        if str(sem.strength) == semantics.STRENGTH_TRASH:
            return False
        if ENABLE_INTERNAL_STAPLES and lane in {LANE_SNACK, LANE_OCCASION} and str(sem.internal_lane_fit) == semantics.LANE_STAPLES:
            return False
        return True
    if lane == LANE_MEAL:
        a_groups = _classify_product_groups(a_name, a_cat, a_family)
        b_groups = _classify_product_groups(b_name, b_cat, b_family)
        if _meal_guardrail_reject_reason(a_groups, b_groups, a_name, b_name):
            return False
    if lane == LANE_SNACK:
        a_groups = _classify_product_groups(a_name, a_cat, a_family)
        b_groups = _classify_product_groups(b_name, b_cat, b_family)
        if not _is_snack_anchor_allowed(a_groups):
            return False
        if _snack_pattern_key(a_groups, b_groups) is None:
            return False
    if not _semantic_lane_compatible(a_id, b_id, lane, context, anchor_groups=_group_labels_for_pid(a_id, context), comp_groups=_group_labels_for_pid(b_id, context)):
        return False
    if _is_bad_pair(a_name, b_name, a_cat, b_cat, lane):
        return False
    return True


def _maybe_swap_to_make_free_item_cheaper(
    display: dict[str, object],
    lane: str,
    context: PersonalizationContext,
) -> None:
    display["swapped"] = False
    price_a = _price_for_swap(display, "a", context)
    price_b = _price_for_swap(display, "b", context)
    if price_a is None or price_b is None:
        _force_item2_free(display)
        return
    if float(price_b) <= float(price_a):
        _force_item2_free(display)
        return
    if not _swap_preserves_lane_semantics(display, lane, context):
        _force_item2_free(display)
        return
    _swap_pair_fields(display)
    display["swapped"] = True
    display["swap_reason"] = "make_free_item_cheaper"
    _force_item2_free(display)


def _display_dict_for_choice(choice: dict[str, object], context: PersonalizationContext) -> dict[str, object]:
    anchor = int(choice["anchor"])
    complement = int(choice["complement"])
    bundle_row = choice.get("bundle_row")
    if isinstance(bundle_row, pd.Series):
        display = _build_display_dict_from_row(bundle_row)
    else:
        display = _build_constrained_pair_record(anchor=anchor, complement=complement, context=context)

    a = _safe_int(display.get("product_a"), default=-1)
    b = _safe_int(display.get("product_b"), default=-1)
    if a == complement and b == anchor:
        _swap_pair_fields(display)
    elif a != anchor:
        display["product_a"] = anchor
        display["product_b"] = complement
        display["product_a_name"] = str(context.product_name_by_id.get(anchor, display.get("product_a_name", f"Product {anchor}")))
        display["product_b_name"] = str(context.product_name_by_id.get(complement, display.get("product_b_name", f"Product {complement}")))

    price_a = float(context.product_price_by_id.get(anchor, _safe_float(display.get("product_a_price"), default=0.0)))
    price_b = float(context.product_price_by_id.get(complement, _safe_float(display.get("product_b_price"), default=0.0)))
    display["product_a_price"] = round(price_a, 2)
    display["product_b_price"] = round(price_b, 2)
    display["price_a_sar"] = f"{price_a:,.2f}"
    display["price_b_sar"] = f"{price_b:,.2f}"
    _force_item2_free(display)
    return display


def _rng_for_profile(run_id: str | None, profile_id: str, rng_salt: str | None = None) -> random.Random:
    seed_text = f"{run_id or 'no_run'}::{profile_id or 'profile'}::{rng_salt or ''}"
    digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:16]
    seed_int = int(digest, 16)
    return random.Random(seed_int)


def _profile_seed_key(profile: PersonProfile) -> str:
    source = str(profile.source).strip().lower() or "unknown"
    orders = ",".join(str(int(oid)) for oid in sorted(int(oid) for oid in profile.order_ids))
    history = ",".join(str(int(pid)) for pid in sorted(int(pid) for pid in profile.history_product_ids))
    return f"{source}|{orders}|{history}"


def _source_priority_rank(source: str) -> int:
    source_group = _source_group_from_source(source)
    source_priority_map = {
        "top_bundle": 0,
        "copurchase_fallback": 1,
        "fallback_food": 2,
        "fallback_cleaning": 3,
        "other": 4,
    }
    return int(source_priority_map.get(source_group, 4))


def _candidate_pair_key(candidate: dict[str, object]) -> tuple[int, int]:
    return _pair_key(
        int(_safe_int(candidate.get("anchor"), default=-1)),
        int(_safe_int(candidate.get("complement"), default=-1)),
    )


def _candidate_effective_score(
    candidate: dict[str, object],
    context: PersonalizationContext,
    global_pair_exposure: dict[tuple[int, int], int],
    global_template_exposure: dict[str, int],
    global_fallback_motif_exposure: dict[str, int] | None = None,
    global_motif_family_exposure: dict[str, int] | None = None,
    global_family_pattern_exposure: dict[str, int] | None = None,
    global_bundle_shape_exposure: dict[str, int] | None = None,
) -> float:
    score = float(candidate.get("pool_score", candidate.get("personal_score", 0.0)))
    source_group = _source_group_from_source(str(candidate.get("source", "")))
    pair_key = _candidate_pair_key(candidate)
    template_signature = _template_signature(candidate, context)
    theme = str(candidate.get("theme", "")).strip().lower()
    lane = str(candidate.get("lane", "")).strip().lower() or LANE_MEAL
    pair_exposure = int(global_pair_exposure.get(pair_key, 0))
    template_exposure = int(global_template_exposure.get(template_signature, 0))
    score -= _progressive_exposure_penalty(
        pair_exposure,
        EXPOSURE_PAIR_PENALTY,
        threshold=EXPOSURE_SURGE_THRESHOLD_PAIR,
    )
    score -= _progressive_exposure_penalty(
        template_exposure,
        EXPOSURE_TEMPLATE_SIGNATURE_PENALTY,
        threshold=EXPOSURE_SURGE_THRESHOLD_TEMPLATE,
    )
    motif_family_signature = _candidate_motif_family_signature(candidate, context)
    family_pattern_signature = _candidate_family_pattern_signature(candidate, context)
    bundle_shape_signature = _candidate_bundle_shape_signature(candidate, context)
    motif_family_exposure_map = global_motif_family_exposure or {}
    family_pattern_exposure_map = global_family_pattern_exposure or {}
    bundle_shape_exposure_map = global_bundle_shape_exposure or {}
    motif_family_exposure = int(motif_family_exposure_map.get(motif_family_signature, 0))
    family_pattern_exposure = int(family_pattern_exposure_map.get(family_pattern_signature, 0))
    bundle_shape_exposure = int(bundle_shape_exposure_map.get(bundle_shape_signature, 0))
    dominant_family = _is_dominant_shopper_family_signature(motif_family_signature)
    utilitarian_family = _is_utilitarian_shopper_family_signature(motif_family_signature)
    score += float(SHOPPER_FAMILY_BASE_ADJUSTMENT.get(motif_family_signature, 0.0))
    meal_dominant_candidate = False
    score -= _progressive_exposure_penalty(
        motif_family_exposure,
        EXPOSURE_MOTIF_FAMILY_PENALTY,
        threshold=EXPOSURE_SURGE_THRESHOLD_MOTIF,
    )
    if lane == LANE_MEAL and dominant_family and motif_family_exposure >= 2:
        score -= float(DOMINANT_MEAL_FAMILY_STRONG_DECAY_STEP) * float(
            (motif_family_exposure - 1) ** DOMINANT_MEAL_FAMILY_STRONG_DECAY_POWER
        )
    elif dominant_family and motif_family_exposure >= 3:
        score -= 0.58 * float((motif_family_exposure - 2) ** 1.2)
    if lane == LANE_OCCASION and motif_family_signature in SHOPPER_FAMILY_OCCASION_DOMINANT and motif_family_exposure >= 2:
        score -= 0.52 * float((motif_family_exposure - 1) ** 1.12)
    if utilitarian_family and motif_family_exposure >= 1:
        score -= float(UTILITARIAN_FAMILY_EXTRA_DECAY_STEP) * float(motif_family_exposure)
    if motif_family_signature in SHOPPER_FAMILY_SNACK_DOMINANT and motif_family_exposure >= 2:
        score -= 0.95 * float((motif_family_exposure - 1) ** 1.15)
    score -= _progressive_exposure_penalty(
        family_pattern_exposure,
        EXPOSURE_FAMILY_PATTERN_PENALTY,
        threshold=EXPOSURE_SURGE_THRESHOLD_FAMILY,
    )
    score -= _progressive_exposure_penalty(
        bundle_shape_exposure,
        EXPOSURE_BUNDLE_SHAPE_PENALTY,
        threshold=EXPOSURE_SURGE_THRESHOLD_SHAPE,
    )
    if lane == LANE_MEAL:
        (
            _anchor,
            _complement,
            name_a,
            name_b,
            cat_a,
            cat_b,
            fam_a,
            fam_b,
            _pair_row,
        ) = _candidate_pair_fields(candidate, context)
        text_a = semantics.normalize_product_text(name_a, cat_a, fam_a)
        text_b = semantics.normalize_product_text(name_b, cat_b, fam_b)
        meal_dominant_candidate = bool(
            _is_meal_dominant_motif_signature(motif_family_signature) or _is_meal_dominant_pair_text(text_a, text_b)
        )
        if meal_dominant_candidate:
            dominant_exposure = max(0, max(motif_family_exposure, family_pattern_exposure, bundle_shape_exposure))
            if dominant_exposure > 0:
                score -= _progressive_exposure_penalty(
                    dominant_exposure,
                    EXPOSURE_MEAL_DOMINANT_MOTIF_PENALTY,
                    threshold=EXPOSURE_SURGE_THRESHOLD_MOTIF,
                )
                if dominant_exposure >= 2:
                    score -= 0.30 * float(dominant_exposure - 1)
            over_saturation = max(0, dominant_exposure - int(EXPOSURE_MEAL_FAMILY_SATURATION_THRESHOLD) + 1)
            if over_saturation > 0:
                score -= float(EXPOSURE_MEAL_FAMILY_SATURATION_PENALTY) * float(over_saturation ** 1.25)
                pattern_over = max(0, family_pattern_exposure - int(EXPOSURE_MEAL_FAMILY_SATURATION_THRESHOLD))
                shape_over = max(0, bundle_shape_exposure - int(EXPOSURE_MEAL_FAMILY_SATURATION_THRESHOLD))
                if pattern_over > 0:
                    score -= float(EXPOSURE_MEAL_PATTERN_SATURATION_PENALTY) * float(pattern_over)
                if shape_over > 0:
                    score -= float(EXPOSURE_MEAL_SHAPE_SATURATION_PENALTY) * float(shape_over)
                if source_group != "top_bundle":
                    score -= float(EXPOSURE_MEAL_DOMINANT_FALLBACK_EXTRA) * float(over_saturation)
            hard_over = max(0, motif_family_exposure - int(MEAL_DOMINANT_HARD_DECAY_START))
            if hard_over > 0:
                score -= float(MEAL_DOMINANT_HARD_DECAY_STEP) * float(hard_over)
            motif_sig = str(motif_family_signature).strip().lower()
            if motif_sig.endswith(":rice_chicken_meal") and motif_family_exposure > 1:
                score -= float(MEAL_CHICKEN_DEFAULT_EXTRA_DECAY) * float(motif_family_exposure - 1)
            if motif_sig.endswith(":rice_meat_meal") and motif_family_exposure > 2:
                score -= float(MEAL_RICE_MEAT_EXTRA_DECAY) * float(motif_family_exposure - 2)
    if lane == LANE_SNACK:
        snack_over = max(0, motif_family_exposure - int(EXPOSURE_SNACK_FAMILY_SATURATION_THRESHOLD) + 1)
        if snack_over > 0:
            score -= float(EXPOSURE_SNACK_FAMILY_SATURATION_PENALTY) * float(snack_over ** 1.1)
            if motif_family_signature in SHOPPER_FAMILY_SNACK_DOMINANT:
                score -= float(EXPOSURE_SNACK_DOMINANT_FAMILY_EXTRA) * float(snack_over)
    motif_exposure = global_fallback_motif_exposure or {}
    fallback_motif_key = _controlled_fallback_motif_key(candidate, context)
    if source_group == "fallback_food" and fallback_motif_key:
        score -= EXPOSURE_FALLBACK_MOTIF_PENALTY * float(motif_exposure.get(fallback_motif_key, 0))
    source_bonus = {
        "top_bundle": 0.18,
        "copurchase_fallback": 0.10,
        "fallback_food": 0.04,
        "fallback_cleaning": -0.02,
        "other": 0.0,
    }
    score += float(source_bonus.get(source_group, 0.0))
    strength = str(candidate.get("pair_strength", "")).strip()
    if strength in {semantics.STRENGTH_STRONG, semantics.STRENGTH_STAPLE}:
        rarity_room = max(
            0.0,
            2.0 - float(motif_family_exposure) - 0.55 * float(bundle_shape_exposure),
        )
        score += min(
            RARER_STRONG_ALTERNATIVE_BONUS_CAP,
            RARER_STRONG_ALTERNATIVE_BONUS_STEP * rarity_room,
        )
        family_rarity_room = max(
            0.0,
            3.0 - float(motif_family_exposure) - 0.55 * float(family_pattern_exposure) - 0.45 * float(bundle_shape_exposure),
        )
        score += min(
            RARER_STRONG_FAMILY_BONUS_CAP,
            RARER_STRONG_FAMILY_BONUS_STEP * family_rarity_room,
        )
        if lane == LANE_MEAL and meal_dominant_candidate and motif_family_exposure >= int(EXPOSURE_MEAL_FAMILY_SATURATION_THRESHOLD):
            score -= 0.18 * float(motif_family_exposure - int(EXPOSURE_MEAL_FAMILY_SATURATION_THRESHOLD) + 1)
    if theme.endswith("_generic"):
        score -= GENERIC_THEME_PENALTY
    if strength == semantics.STRENGTH_WEAK:
        score -= max(WEAK_CANDIDATE_PENALTY, 0.45)
    score += _human_preference_score_adjustment(candidate, context)
    return float(score)


def _candidate_is_strong_finalist(candidate: dict[str, object]) -> bool:
    lane = str(candidate.get("lane", "")).strip().lower()
    strength = str(candidate.get("pair_strength", "")).strip()
    cp_score = float(candidate.get("cp_score", 0.0))
    pair_count = int(_safe_int(candidate.get("pair_count"), default=0))
    recipe = float(candidate.get("recipe_compat", 0.0))
    template_strength = float(candidate.get("template_strength", 0.0))
    category_strength = float(candidate.get("category_strength", 0.0))
    lane_fit = float(candidate.get("lane_fit_score", 0.0))
    prior_bonus = float(candidate.get("prior_bonus", 0.0))
    source = str(candidate.get("source", "")).strip().lower()

    if strength == semantics.STRENGTH_TRASH:
        return False
    if lane == LANE_MEAL:
        if strength == semantics.STRENGTH_STRONG:
            return True
        return bool(
            strength == semantics.STRENGTH_STAPLE
            and cp_score >= 20.0
            and pair_count >= 8
            and (recipe >= 0.14 or template_strength >= 0.6 or category_strength >= 0.6)
        )
    if lane == LANE_SNACK:
        if strength == semantics.STRENGTH_STRONG:
            return True
        return bool(
            strength == semantics.STRENGTH_STAPLE
            and template_strength >= 0.75
            and cp_score >= 22.0
            and pair_count >= 8
            and lane_fit >= 0.7
        )
    if lane == LANE_OCCASION:
        if strength == semantics.STRENGTH_STRONG:
            return True
        return bool(
            strength == semantics.STRENGTH_STAPLE
            and cp_score >= 24.0
            and pair_count >= 8
            and lane_fit >= 0.8
            and (template_strength >= 0.75 or prior_bonus > 0.0 or category_strength >= 0.6)
        )
    return False


def _candidate_is_fill_finalist(candidate: dict[str, object]) -> bool:
    return _candidate_is_strong_finalist(candidate)


def _choose_candidate_first_food_trio(
    candidate_pool: dict[tuple[str, int, int, str], dict[str, object]],
    context: PersonalizationContext,
    global_pair_exposure: dict[tuple[int, int], int],
    global_template_exposure: dict[str, int],
    global_fallback_motif_exposure: dict[str, int] | None = None,
    global_motif_family_exposure: dict[str, int] | None = None,
    global_family_pattern_exposure: dict[str, int] | None = None,
    global_bundle_shape_exposure: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    def _lane_candidates(filter_fn) -> dict[str, list[dict[str, object]]]:
        lane_out: dict[str, list[dict[str, object]]] = {lane: [] for lane in FOOD_LANE_ORDER}
        for candidate in candidate_pool.values():
            lane = str(candidate.get("lane", "")).strip().lower()
            if lane not in FOOD_LANE_ORDER:
                continue
            if not filter_fn(candidate):
                continue
            enriched = dict(candidate)
            enriched["effective_score"] = _candidate_effective_score(
                enriched,
                context,
                global_pair_exposure=global_pair_exposure,
                global_template_exposure=global_template_exposure,
                global_fallback_motif_exposure=global_fallback_motif_exposure,
                global_motif_family_exposure=global_motif_family_exposure,
                global_family_pattern_exposure=global_family_pattern_exposure,
                global_bundle_shape_exposure=global_bundle_shape_exposure,
            )
            lane_out[lane].append(enriched)
        for lane in FOOD_LANE_ORDER:
            lane_out[lane].sort(
                key=lambda cand: (
                    -float(cand.get("effective_score", 0.0)),
                    _source_priority_rank(str(cand.get("source", ""))),
                    int(_safe_int(cand.get("complement"), default=-1)),
                    int(_safe_int(cand.get("anchor"), default=-1)),
                )
            )
            lane_out[lane] = lane_out[lane][:TOP_TRIO_CANDIDATES_PER_LANE]
        return lane_out

    def _best_combo(filter_fn) -> list[dict[str, object]]:
        lane_candidates = _lane_candidates(filter_fn)
        if any(not lane_candidates[lane] for lane in FOOD_LANE_ORDER):
            return []
        best_score = float("-inf")
        best_combo: list[dict[str, object]] = []
        for meal_choice in lane_candidates[LANE_MEAL]:
            for snack_choice in lane_candidates[LANE_SNACK]:
                for occasion_choice in lane_candidates[LANE_OCCASION]:
                    combo = [meal_choice, snack_choice, occasion_choice]
                    anchors = [int(_safe_int(item.get("anchor"), default=-1)) for item in combo]
                    if len({anchor for anchor in anchors if anchor > 0}) != 3:
                        continue
                    pair_keys = [_candidate_pair_key(item) for item in combo]
                    if len(set(pair_keys)) != 3:
                        continue
                    template_signatures = [_template_signature(item, context) for item in combo]
                    score = sum(float(item.get("effective_score", 0.0)) for item in combo)
                    duplicate_sig_penalty = len(template_signatures) - len(set(template_signatures))
                    if duplicate_sig_penalty > 0:
                        score -= 0.10 * float(duplicate_sig_penalty)
                    combo_rank = (
                        float(score),
                        -sum(_source_priority_rank(str(item.get("source", ""))) for item in combo),
                        -sum(int(_safe_int(item.get("pair_count"), default=0)) for item in combo),
                    )
                    if combo_rank > (best_score, float("-inf"), float("-inf")):
                        best_score = combo_rank[0]
                        best_combo = combo
        return best_combo

    combo = _best_combo(_candidate_is_strong_finalist)
    if combo:
        return combo
    combo = _best_combo(_candidate_is_fill_finalist)
    if combo:
        return combo

    used_anchors: set[int] = set()
    used_pairs: set[tuple[int, int]] = set()
    selected: list[dict[str, object]] = []
    for lane in FOOD_LANE_ORDER:
        lane_candidates: list[dict[str, object]] = []
        for candidate in candidate_pool.values():
            if str(candidate.get("lane", "")).strip().lower() != lane:
                continue
            enriched = dict(candidate)
            enriched["effective_score"] = _candidate_effective_score(
                enriched,
                context,
                global_pair_exposure=global_pair_exposure,
                global_template_exposure=global_template_exposure,
                global_fallback_motif_exposure=global_fallback_motif_exposure,
                global_motif_family_exposure=global_motif_family_exposure,
                global_family_pattern_exposure=global_family_pattern_exposure,
                global_bundle_shape_exposure=global_bundle_shape_exposure,
            )
            lane_candidates.append(enriched)
        lane_candidates.sort(
            key=lambda cand: (
                -float(cand.get("effective_score", 0.0)),
                _source_priority_rank(str(cand.get("source", ""))),
                int(_safe_int(cand.get("complement"), default=-1)),
                int(_safe_int(cand.get("anchor"), default=-1)),
            )
        )
        picked = None
        for candidate in lane_candidates:
            anchor_id = int(_safe_int(candidate.get("anchor"), default=-1))
            pair_key = _candidate_pair_key(candidate)
            if anchor_id <= 0 or anchor_id in used_anchors or pair_key in used_pairs:
                continue
            picked = candidate
            break
        if picked is None:
            return []
        selected.append(picked)
        used_anchors.add(int(_safe_int(picked.get("anchor"), default=-1)))
        used_pairs.add(_candidate_pair_key(picked))
    return selected


def _normalise_float_lookup(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    lo = min(float(v) for v in values.values())
    hi = max(float(v) for v in values.values())
    span = hi - lo
    if span <= 1e-9:
        return {str(k): 1.0 for k in values}
    out: dict[str, float] = {}
    for key, value in values.items():
        out[str(key)] = float((float(value) - lo) / span)
    return out


def _profile_product_interest(profile: PersonProfile) -> dict[int, float]:
    counts = {int(pid): float(count) for pid, count in profile.history_counts.items() if int(pid) > 0 and float(count) > 0.0}
    if not counts:
        counts = {int(pid): 1.0 for pid in profile.history_product_ids if int(pid) > 0}
    return {int(pid): float(score) for pid, score in _normalised_values(counts).items()}


def _profile_family_interest(profile: PersonProfile, context: PersonalizationContext) -> dict[str, float]:
    counts: dict[str, float] = {}
    for pid in profile.history_product_ids:
        pid_int = int(pid)
        if pid_int <= 0:
            continue
        family = _normalise_text(context.product_family_by_id.get(pid_int, ""))
        if not family:
            continue
        weight = float(profile.history_counts.get(pid_int, 1))
        counts[family] = float(counts.get(family, 0.0)) + max(1.0, weight)
    return _normalise_float_lookup(counts)


def _profile_category_affinity(profile: PersonProfile, context: PersonalizationContext) -> dict[str, float]:
    counts: dict[str, float] = {}
    for pid in profile.history_product_ids:
        pid_int = int(pid)
        if pid_int <= 0:
            continue
        category = _normalise_text(context.category_by_id.get(pid_int, ""))
        if not category:
            continue
        weight = float(profile.history_counts.get(pid_int, 1))
        counts[category] = float(counts.get(category, 0.0)) + max(1.0, weight)
    return _normalise_float_lookup(counts)


def _profile_shopper_family_interest(profile: PersonProfile, context: PersonalizationContext) -> dict[str, float]:
    history_ids = [int(pid) for pid in profile.history_product_ids if int(pid) > 0]
    if not history_ids:
        return {}
    weighted_texts: list[tuple[str, float]] = []
    for pid in history_ids:
        name = _product_name(pid, context)
        category = _product_category(pid, context)
        family = _product_family(pid, context)
        text = semantics.normalize_product_text(name, category, family)
        if not text:
            continue
        weight = float(max(1.0, profile.history_counts.get(pid, 1)))
        weighted_texts.append((text, weight))
    if not weighted_texts:
        return {}

    def _presence(hints: frozenset[str]) -> float:
        score = 0.0
        for text, weight in weighted_texts:
            if _text_has_any_hint(text, hints):
                score += float(weight)
        return float(score)

    chicken = _presence(HUMAN_HINTS_CHICKEN)
    minced_meat = _presence(HUMAN_HINTS_MINCED_MEAT)
    nuggets = _presence(HUMAN_HINTS_NUGGETS)
    eggs = _presence(HUMAN_HINTS_EGGS)
    rice = _presence(HUMAN_HINTS_RICE)
    meat = _presence(HUMAN_HINTS_MEAT | HUMAN_HINTS_FISH)
    bread = _presence(HUMAN_HINTS_BREAD)
    tortilla = _presence(HUMAN_HINTS_TORTILLA)
    labneh = _presence(HUMAN_HINTS_LABNEH)
    chips = _presence(HUMAN_HINTS_CHIPS | MOTIF_HINTS_CRUNCHY_SNACK)
    biscuits = _presence(HUMAN_HINTS_BISCUITS)
    chocolate = _presence(HUMAN_HINTS_CHOCOLATE)
    milk = _presence(HUMAN_HINTS_MILK | HUMAN_HINTS_EVAP_MILK | HUMAN_HINTS_CONDENSED_MILK)
    tea = _presence(HUMAN_HINTS_TEA)
    coffee = _presence(HUMAN_HINTS_COFFEE)
    dates = _presence(HUMAN_HINTS_DATES)

    family_interest_raw: dict[str, float] = {
        "meal:chicken_wrap_meal": min(chicken, tortilla + 0.6 * bread),
        "meal:minced_meat_wrap_meal": min(minced_meat, tortilla + 0.7 * bread),
        "meal:nuggets_bread_fastmeal": min(nuggets, bread),
        "meal:egg_breakfast_meal": min(eggs, bread),
        "meal:labneh_bread_meal": min(labneh, bread),
        "meal:rice_chicken_meal": min(rice, chicken),
        "meal:rice_meat_meal": min(rice, meat),
        "meal:rice_egg_meal": min(rice, eggs),
        "snack:biscuit_milk_tea_snack": min(biscuits, max(milk, tea, coffee)),
        "snack:wafer_chocolate_snack": min(biscuits, chocolate),
        "snack:labneh_crunchy_snack": min(labneh, chips),
        "snack:nutella_snack_pair": min(_presence(HUMAN_HINTS_NUTELLA), max(bread, biscuits)),
        "occasion:tea_milk_drink": min(tea, milk),
        "occasion:coffee_milk_drink": min(coffee, milk),
        "occasion:dates_milk_treat": min(dates, milk),
        "occasion:dates_cream_treat": min(dates, _presence(HUMAN_HINTS_CREAM_TOKEN)),
        "occasion:dates_dairy_treat": min(dates, milk),
    }
    normalised = _normalise_float_lookup({k: v for k, v in family_interest_raw.items() if v > 0.0})
    for key in ("meal:rice_meat_meal", "meal:rice_chicken_meal", "meal:protein_grain_meal", "meal:protein_starch_generic_meal"):
        if key in normalised:
            normalised[key] = float(normalised[key] * 0.72)
    return normalised


def _profile_recency_order(profile: PersonProfile) -> dict[int, float]:
    ordered = [int(pid) for pid in profile.history_product_ids if int(pid) > 0]
    if not ordered:
        return {}
    out: dict[int, float] = {}
    total = float(len(ordered))
    for idx, pid in enumerate(ordered):
        out[int(pid)] = float((idx + 1.0) / total)
    return out


def _profile_lane_intent_map(profile: PersonProfile, context: PersonalizationContext) -> dict[str, float]:
    meal, snack, occasion = _history_lane_ratios(profile, context)
    return {
        LANE_MEAL: float(meal),
        LANE_SNACK: float(snack),
        LANE_OCCASION: float(occasion),
        LANE_NONFOOD: float(sum(1 for pid in profile.history_product_ids if int(pid) in context.non_food_ids) / max(1, len(profile.history_product_ids))),
    }


def _profile_recent_windows(profile: PersonProfile) -> tuple[set[int], set[int]]:
    ordered = [int(pid) for pid in profile.history_product_ids if int(pid) > 0]
    if not ordered:
        return set(), set()
    recent_7_size = max(1, min(3, len(ordered)))
    recent_30_size = max(recent_7_size, min(10, len(ordered)))
    return set(ordered[-recent_7_size:]), set(ordered[-recent_30_size:])


def _profile_recent_category_interest(profile: PersonProfile, context: PersonalizationContext) -> dict[str, float]:
    recent_7, recent_30 = _profile_recent_windows(profile)
    counts: dict[str, float] = {}
    for pid in recent_30:
        category = _normalise_text(context.category_by_id.get(int(pid), ""))
        if not category:
            continue
        counts[category] = float(counts.get(category, 0.0)) + 1.0
    for pid in recent_7:
        category = _normalise_text(context.category_by_id.get(int(pid), ""))
        if not category:
            continue
        counts[category] = float(counts.get(category, 0.0)) + 1.5
    return counts


def _profile_recent_brand_affinity(profile: PersonProfile, context: PersonalizationContext) -> dict[str, float]:
    recent_7, recent_30 = _profile_recent_windows(profile)
    counts: dict[str, float] = {}
    for pid in recent_30:
        brand = _normalise_text(context.product_brand_by_id.get(int(pid), ""))
        if not brand:
            continue
        counts[brand] = float(counts.get(brand, 0.0)) + 1.0
    for pid in recent_7:
        brand = _normalise_text(context.product_brand_by_id.get(int(pid), ""))
        if not brand:
            continue
        counts[brand] = float(counts.get(brand, 0.0)) + 1.5
    return counts


def _candidate_personalization_boost(
    candidate: dict[str, object],
    profile: PersonProfile,
    context: PersonalizationContext,
    recency_map: dict[int, float],
    lane_intent_map: dict[str, float],
    recent_category_interest: dict[str, float],
    recent_brand_affinity: dict[str, float],
    profile_product_interest: dict[int, float],
    profile_family_interest: dict[str, float],
    profile_category_affinity: dict[str, float],
    profile_shopper_family_interest: dict[str, float],
) -> float:
    anchor = int(_safe_int(candidate.get("anchor"), default=-1))
    complement = int(_safe_int(candidate.get("complement"), default=-1))
    lane = str(candidate.get("lane", "")).strip().lower()
    recent_7, recent_30 = _profile_recent_windows(profile)
    anchor_recency = float(recency_map.get(anchor, 0.0))
    complement_recency = float(recency_map.get(complement, 0.0))
    count_total = max(1, max(profile.history_counts.values()) if profile.history_counts else 1)
    anchor_count = float(profile.history_counts.get(anchor, 0) / count_total)
    complement_count = float(profile.history_counts.get(complement, 0) / count_total)
    recent_7_component = 0.16 if (anchor in recent_7 or complement in recent_7) else 0.0
    recent_30_component = 0.08 if (anchor in recent_30 or complement in recent_30) else 0.0
    recency_component = RECENCY_BOOST_WEIGHT * max(anchor_recency, complement_recency)
    count_component = COUNT_BOOST_WEIGHT * (0.65 * anchor_count + 0.35 * complement_count)
    lane_component = LANE_INTENT_BOOST_WEIGHT * float(lane_intent_map.get(lane, 0.0))
    cat_a = _normalise_text(context.category_by_id.get(anchor, ""))
    cat_b = _normalise_text(context.category_by_id.get(complement, ""))
    category_component = 0.02 * max(float(recent_category_interest.get(cat_a, 0.0)), float(recent_category_interest.get(cat_b, 0.0)))
    category_affinity_component = PERSONAL_CATEGORY_AFFINITY_WEIGHT * max(
        float(profile_category_affinity.get(cat_a, 0.0)),
        float(profile_category_affinity.get(cat_b, 0.0)),
    )
    brand_a = _normalise_text(context.product_brand_by_id.get(anchor, ""))
    brand_b = _normalise_text(context.product_brand_by_id.get(complement, ""))
    recent_brand_component = 0.015 * max(float(recent_brand_affinity.get(brand_a, 0.0)), float(recent_brand_affinity.get(brand_b, 0.0)))
    fam_a = _normalise_text(context.product_family_by_id.get(anchor, ""))
    fam_b = _normalise_text(context.product_family_by_id.get(complement, ""))
    family_component = PERSONAL_FAMILY_AFFINITY_WEIGHT * max(
        float(profile_family_interest.get(fam_a, 0.0)),
        float(profile_family_interest.get(fam_b, 0.0)),
    )
    anchor_history_component = PERSONAL_ANCHOR_HISTORY_BOOST if float(profile_product_interest.get(anchor, 0.0)) > 0.0 else 0.0
    complement_history_component = (
        PERSONAL_COMPLEMENT_HISTORY_BOOST if float(profile_product_interest.get(complement, 0.0)) > 0.0 else 0.0
    )
    product_repeat_component = PERSONAL_PRODUCT_REPEAT_WEIGHT * max(
        float(profile_product_interest.get(anchor, 0.0)),
        float(profile_product_interest.get(complement, 0.0)),
    )
    strong_anchor_match_component = 0.0
    if float(profile_product_interest.get(anchor, 0.0)) >= 0.66:
        strong_anchor_match_component = PERSONAL_STRONG_ANCHOR_MATCH_BONUS
    lane_score = float(lane_intent_map.get(lane, 0.0))
    lane_intents_food = {
        LANE_MEAL: float(lane_intent_map.get(LANE_MEAL, 0.0)),
        LANE_SNACK: float(lane_intent_map.get(LANE_SNACK, 0.0)),
        LANE_OCCASION: float(lane_intent_map.get(LANE_OCCASION, 0.0)),
    }
    best_lane = max(lane_intents_food, key=lane_intents_food.get)
    best_lane_score = float(lane_intents_food.get(best_lane, 0.0))
    anchor_name = str(context.product_name_by_id.get(anchor, ""))
    complement_name = str(context.product_name_by_id.get(complement, ""))
    anchor_text = _normalise_text(anchor_name)
    complement_text = _normalise_text(complement_name)
    motif_family_signature = _candidate_motif_family_signature(candidate, context)
    meal_dominant_family = bool(lane == LANE_MEAL and _is_meal_dominant_motif_signature(motif_family_signature))
    utilitarian_family = _is_utilitarian_shopper_family_signature(motif_family_signature)
    dominant_family = _is_dominant_shopper_family_signature(motif_family_signature)
    wrap_style_pair = _pair_matches_hints(
        anchor_text,
        complement_text,
        HUMAN_HINTS_CHICKEN | HUMAN_HINTS_MINCED_MEAT | HUMAN_HINTS_EGGS,
        HUMAN_HINTS_BREAD | HUMAN_HINTS_TORTILLA,
    )
    snack_or_occasion_pattern = bool(
        str(candidate.get("snack_pattern", "")).strip()
        or _pair_matches_hints(anchor_text, complement_text, HUMAN_HINTS_BISCUITS, HUMAN_HINTS_MILK | HUMAN_HINTS_TEA | HUMAN_HINTS_COFFEE)
        or _pair_matches_hints(anchor_text, complement_text, HUMAN_HINTS_DATES, HUMAN_HINTS_EVAP_MILK | HUMAN_HINTS_CONDENSED_MILK)
    )
    max_family_interest = max(
        float(profile_family_interest.get(fam_a, 0.0)),
        float(profile_family_interest.get(fam_b, 0.0)),
    )
    lane_alignment_component = 0.0
    if lane in FOOD_LANE_ORDER and lane_score >= 0.40:
        lane_alignment_component += 0.05
    if lane in {LANE_SNACK, LANE_OCCASION} and lane == best_lane and lane_score >= 0.35:
        lane_alignment_component += 0.06
    if lane in FOOD_LANE_ORDER and lane != best_lane and (best_lane_score - lane_score) >= 0.22:
        lane_alignment_component -= 0.08
    if lane == LANE_MEAL and best_lane in {LANE_SNACK, LANE_OCCASION} and best_lane_score >= 0.45:
        lane_alignment_component -= 0.10
    if meal_dominant_family and best_lane in {LANE_SNACK, LANE_OCCASION} and best_lane_score >= 0.40:
        lane_alignment_component -= PERSONAL_ESCAPE_GENERIC_MEAL_PENALTY
    if meal_dominant_family and lane_score < 0.45 and max_family_interest < 0.45:
        lane_alignment_component -= PERSONAL_ESCAPE_GENERIC_MEAL_AFFINITY_PENALTY
    theme = str(candidate.get("theme", "")).strip().lower()
    if lane == LANE_MEAL and theme.endswith("_generic") and best_lane in {LANE_SNACK, LANE_OCCASION}:
        lane_alignment_component -= 0.06
    if lane == LANE_SNACK and str(candidate.get("snack_pattern", "")).strip():
        lane_alignment_component += 0.06
    if lane in {LANE_SNACK, LANE_OCCASION} and lane == best_lane and lane_score >= 0.42:
        lane_alignment_component += PERSONAL_NONMEAL_HISTORY_ESCAPE_BONUS
    if lane in {LANE_SNACK, LANE_OCCASION} and snack_or_occasion_pattern and best_lane in {LANE_SNACK, LANE_OCCASION}:
        lane_alignment_component += PERSONAL_SNACK_OCCASION_PATTERN_BONUS
    if lane == LANE_MEAL and wrap_style_pair and max_family_interest >= 0.35:
        lane_alignment_component += PERSONAL_WRAP_STYLE_HISTORY_BONUS
    shopper_family_alignment_component = PERSONAL_SHOPPER_FAMILY_ALIGNMENT_WEIGHT * float(
        profile_shopper_family_interest.get(motif_family_signature, 0.0)
    )
    if utilitarian_family and float(profile_shopper_family_interest.get(motif_family_signature, 0.0)) < 0.35:
        shopper_family_alignment_component -= 0.14
    if lane == LANE_MEAL and dominant_family and best_lane in {LANE_SNACK, LANE_OCCASION} and best_lane_score >= 0.38:
        shopper_family_alignment_component -= PERSONAL_SHOPPER_FAMILY_ESCAPE_PENALTY
    if lane in {LANE_SNACK, LANE_OCCASION} and best_lane in {LANE_SNACK, LANE_OCCASION} and lane == best_lane:
        shopper_family_alignment_component += 0.05 * float(
            profile_shopper_family_interest.get(motif_family_signature, 0.0)
        )
    brand_component = 0.0
    if float(candidate.get("brand_signal", 0.0)) > 0:
        brand_component += 0.02
    return float(
        recency_component
        + recent_7_component
        + recent_30_component
        + count_component
        + lane_component
        + category_component
        + category_affinity_component
        + family_component
        + anchor_history_component
        + complement_history_component
        + product_repeat_component
        + strong_anchor_match_component
        + lane_alignment_component
        + shopper_family_alignment_component
        + recent_brand_component
        + brand_component
    )


def _is_household_bundle(bundle: dict[str, object], context: PersonalizationContext) -> bool:
    lane = str(bundle.get("lane", "")).strip().lower()
    if lane == LANE_NONFOOD:
        return True
    a_id = int(_safe_int(bundle.get("anchor"), default=-1))
    b_id = int(_safe_int(bundle.get("complement"), default=-1))
    if a_id <= 0 or b_id <= 0:
        return False
    return bool(_is_nonfood_product(a_id, context) or _is_nonfood_product(b_id, context))


def _template_signature(choice: dict[str, object], context: PersonalizationContext) -> str:
    anchor = int(choice.get("anchor", -1))
    complement = int(choice.get("complement", -1))
    fam_a = str(context.product_family_by_id.get(anchor, "")).strip().lower() or "na"
    fam_b = str(context.product_family_by_id.get(complement, "")).strip().lower() or "nb"
    name_a = str(context.product_name_by_id.get(anchor, ""))
    name_b = str(context.product_name_by_id.get(complement, ""))
    a_tokens = sorted(_token_set(name_a))[:2]
    b_tokens = sorted(_token_set(name_b))[:2]
    lexical = "|".join(a_tokens + b_tokens) or "none"
    return f"{fam_a}::{fam_b}::{lexical}"


def _write_person_quality_artifact(
    base_dir: Path | None,
    recommendations: list[dict[str, object]],
    context: PersonalizationContext,
    run_id: str | None = None,
    global_duplicate_pair_count_blocked: int = 0,
    anchor_reuse_counts: dict[int, int] | None = None,
    guardrail_reject_counts: dict[str, int] | None = None,
    serving_telemetry: ServingTelemetry | None = None,
) -> None:
    if base_dir is None:
        return
    person_count = len(recommendations)
    nonfood_card_count = 0
    for rec in recommendations:
        bundles = rec.get("bundles")
        if not isinstance(bundles, list):
            continue
        if any(str(b.get("lane", "")).strip().lower() == LANE_NONFOOD for b in bundles if isinstance(b, dict)):
            nonfood_card_count += 1
    bundle_rows: list[dict[str, object]] = []
    for rec in recommendations:
        bundles = rec.get("bundles")
        if isinstance(bundles, list):
            for bundle in bundles:
                if isinstance(bundle, dict):
                    bundle_rows.append(bundle)
        elif isinstance(rec, dict):
            bundle_rows.append(rec)

    total = len(bundle_rows)
    top_bundle_count = sum(1 for r in bundle_rows if str(r.get("recommendation_origin", "")) == "top_bundle")
    copurchase_fallback_count = sum(1 for r in bundle_rows if str(r.get("recommendation_origin", "")) == "copurchase_fallback")
    food_fallback_count = sum(1 for r in bundle_rows if str(r.get("recommendation_origin", "")) == "fallback_food")
    cleaning_fallback_count = sum(1 for r in bundle_rows if str(r.get("recommendation_origin", "")) == "fallback_cleaning")
    template_fallback_count = int(food_fallback_count)
    fallback_count = int(food_fallback_count + cleaning_fallback_count)
    anchor_in_history = sum(1 for r in bundle_rows if bool(r.get("anchor_in_history", False)))
    history_matches = [float(r.get("history_match_count", 0)) for r in bundle_rows]
    overall_telemetry = serving_telemetry.overall if serving_telemetry is not None else {}

    non_food_pair_count = 0
    same_family_pair_count = 0
    staple_anchor_count = 0
    staple_pair_count = 0
    lane_counts = {LANE_MEAL: 0, LANE_SNACK: 0, LANE_OCCASION: 0, LANE_NONFOOD: 0}
    for rec in bundle_rows:
        a = _safe_int(rec.get("product_a"), default=-1)
        b = _safe_int(rec.get("product_b"), default=-1)
        if a in context.non_food_ids or b in context.non_food_ids:
            non_food_pair_count += 1
        fam_a = str(context.product_family_by_id.get(a, "")).strip().lower()
        fam_b = str(context.product_family_by_id.get(b, "")).strip().lower()
        if fam_a and fam_b and fam_a == fam_b:
            same_family_pair_count += 1
        a_name = str(context.product_name_by_id.get(a, rec.get("product_a_name", "")))
        b_name = str(context.product_name_by_id.get(b, rec.get("product_b_name", "")))
        if _is_staple_product(a_name, fam_a):
            staple_anchor_count += 1
        if _is_staple_product(a_name, fam_a) or _is_staple_product(b_name, fam_b):
            staple_pair_count += 1
        lane = str(rec.get("lane", "")).strip().lower()
        if lane in lane_counts:
            lane_counts[lane] += 1

    payload = {
        "generated_at": _utc_now_iso(),
        "run_id": str(run_id or ""),
        "recommendation_count": total,
        "anchor_in_history_rate": round((anchor_in_history / total) if total else 0.0, 4),
        "top_bundle_count": int(top_bundle_count),
        "copurchase_fallback_count": int(copurchase_fallback_count),
        "food_fallback_count": int(food_fallback_count),
        "cleaning_fallback_count": int(cleaning_fallback_count),
        "template_fallback_count": int(template_fallback_count),
        "fallback_count": int(fallback_count),
        "chose_top_bundle": int(overall_telemetry.get("chose_top_bundle", top_bundle_count)),
        "chose_copurchase_fallback": int(overall_telemetry.get("chose_copurchase_fallback", copurchase_fallback_count)),
        "chose_template_fallback": int(overall_telemetry.get("chose_template_fallback", template_fallback_count)),
        "chose_food_fallback": int(overall_telemetry.get("chose_food_fallback", food_fallback_count)),
        "chose_cleaning_fallback": int(overall_telemetry.get("chose_cleaning_fallback", cleaning_fallback_count)),
        "from_top_bundle_share": round((top_bundle_count / total) if total else 0.0, 4),
        "fallback_share": round((fallback_count / total) if total else 0.0, 4),
        "rejected_hard_invalid": int(overall_telemetry.get("rejected_hard_invalid", 0)),
        "rejected_lane_disallow": int(overall_telemetry.get("rejected_lane_disallow", 0)),
        "rejected_visible_expression": int(overall_telemetry.get("rejected_visible_expression", 0)),
        "rejected_theme_block": int(overall_telemetry.get("rejected_theme_block", 0)),
        "rejected_pair_fingerprint": int(overall_telemetry.get("rejected_pair_fingerprint", 0)),
        "rejected_family_reuse": int(overall_telemetry.get("rejected_family_reuse", 0)),
        "rejected_feedback_trash": int(overall_telemetry.get("rejected_feedback_trash", 0)),
        "rejected_score_floor": int(overall_telemetry.get("rejected_score_floor", 0)),
        "non_food_pair_count": int(non_food_pair_count),
        "same_family_pair_count": int(same_family_pair_count),
        "staple_anchor_rate": round((staple_anchor_count / total) if total else 0.0, 4),
        "staple_in_pair_rate": round((staple_pair_count / total) if total else 0.0, 4),
        "avg_history_match_count": round((sum(history_matches) / total) if total else 0.0, 4),
        "template_dup_rate": 0.0,
        "unique_anchor_count_top10": 0,
        "unique_family_count_top10": 0,
        "lane_share_meal": round((lane_counts[LANE_MEAL] / total) if total else 0.0, 4),
        "lane_share_snack": round((lane_counts[LANE_SNACK] / total) if total else 0.0, 4),
        "lane_share_occasion": round((lane_counts[LANE_OCCASION] / total) if total else 0.0, 4),
        "lane_share_nonfood": round((lane_counts[LANE_NONFOOD] / total) if total else 0.0, 4),
        "nonfood_bundle_count": int(lane_counts[LANE_NONFOOD]),
        "nonfood_card_share": round((nonfood_card_count / person_count) if person_count else 0.0, 4),
        "global_duplicate_pair_count_blocked": int(global_duplicate_pair_count_blocked),
        MEAL_REJECT_PRODUCE_SNACK_KEY: int((guardrail_reject_counts or {}).get(MEAL_REJECT_PRODUCE_SNACK_KEY, 0)),
        MEAL_REJECT_PRODUCE_NOODLES_KEY: int((guardrail_reject_counts or {}).get(MEAL_REJECT_PRODUCE_NOODLES_KEY, 0)),
        "semantic_engine_version": SEMANTIC_ENGINE_VERSION,
        "use_new_bundle_semantics": bool(USE_NEW_BUNDLE_SEMANTICS),
        "strict_semantic_filtering": bool(STRICT_SEMANTIC_FILTERING),
        "enable_internal_staples": bool(ENABLE_INTERNAL_STAPLES),
        "enable_staples_lane": bool(ENABLE_STAPLES_LANE),
        "top_bundle_scan_depth_by_lane": {lane: int(depth) for lane, depth in MAX_TOP_BUNDLE_CANDIDATES_BY_LANE.items()},
        "serving_telemetry_by_lane": {
            str(lane): {str(key): int(value) for key, value in sorted(counts.items())}
            for lane, counts in sorted((serving_telemetry.by_lane if serving_telemetry is not None else {}).items())
        },
        "max_anchor_count": 0,
        "anchor_reuse_histogram": {},
        "people_refreshed_for_run_id": str(run_id or ""),
    }
    if anchor_reuse_counts:
        payload["max_anchor_count"] = int(max(anchor_reuse_counts.values()) if anchor_reuse_counts else 0)
        payload["anchor_reuse_histogram"] = {str(int(k)): int(v) for k, v in sorted(anchor_reuse_counts.items())}
    if total > 0:
        top = bundle_rows[:10]
        anchors = [int(_safe_int(r.get("anchor_product_id"), default=-1)) for r in top]
        families = [
            str(context.product_family_by_id.get(_safe_int(r.get("complement_product_id"), default=-1), "")).strip().lower()
            for r in top
        ]
        signatures = [
            f"{str(context.product_family_by_id.get(_safe_int(r.get('anchor_product_id'), default=-1), '')).strip().lower()}::"
            f"{str(context.product_family_by_id.get(_safe_int(r.get('complement_product_id'), default=-1), '')).strip().lower()}"
            for r in top
        ]
        dup_count = len(signatures) - len(set(signatures))
        payload["template_dup_rate"] = round(float(dup_count / max(1, len(top))), 4)
        payload["unique_anchor_count_top10"] = int(len({x for x in anchors if x > 0}))
        payload["unique_family_count_top10"] = int(len({f for f in families if f}))

    out_dir = get_paths(project_root=base_dir).output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "person_reco_quality.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def build_recommendations_for_profiles(
    bundles_df: pd.DataFrame,
    profiles: list[PersonProfile],
    max_people: int = 10,
    row_to_record=None,
    base_dir: Path | None = None,
    run_id: str | None = None,
    rng_salt: str | None = None,
) -> list[dict[str, object]]:
    global _LAST_SERVING_PROFILE_METRICS
    if not profiles or max_people <= 0:
        _LAST_SERVING_PROFILE_METRICS = {}
        return []

    overall_started = time.perf_counter()
    active_base_dir = base_dir or get_paths().project_root
    bundle_data = bundles_df if bundles_df is not None else pd.DataFrame()
    bundle_lookup = _build_bundle_lookup(bundle_data)
    top_bundle_rows_by_anchor = _build_top_bundle_rows_by_anchor(bundle_data)
    context = load_personalization_context(active_base_dir)

    recommendations: list[dict[str, object]] = []
    feedback_lookup = build_pair_multiplier_lookup(active_base_dir)
    order_pool = load_order_pool(active_base_dir)
    profile_queue = list(profiles[:max_people])
    seen_signatures = {tuple(sorted(int(pid) for pid in p.history_product_ids)) for p in profile_queue}
    resample_budget = max(10, max_people * 4)
    idx = 0
    global_anchor_counts: dict[int, int] = {}
    global_anchor_lane_counts: dict[tuple[str, int], int] = {}
    global_duplicate_pair_count_blocked = 0
    gate_reject_counters: dict[str, int] = {}
    serving_telemetry = ServingTelemetry()

    global_pair_exposure: dict[tuple[int, int], int] = {}
    global_template_exposure: dict[str, int] = {}
    global_fallback_motif_exposure: dict[str, int] = {}
    global_motif_family_exposure: dict[str, int] = {}
    global_family_pattern_exposure: dict[str, int] = {}
    global_bundle_shape_exposure: dict[str, int] = {}
    profiling = _ServingProfileRecorder(enabled=_serving_profiling_enabled())
    profiled_nonfood_anchors = [
        int(pid)
        for pid in top_bundle_rows_by_anchor.keys()
        if int(pid) > 0 and _is_nonfood_product(int(pid), context)
    ]

    while idx < len(profile_queue) and len(recommendations) < max_people:
        profile_started = time.perf_counter()
        profile = profile_queue[idx]
        idx += 1
        prep_started = time.perf_counter()
        profile_seed_key = _profile_seed_key(profile)
        profile_rng = _rng_for_profile(run_id, profile_seed_key, rng_salt=rng_salt)
        nonfood_gate_rng = _rng_for_profile(run_id, profile_seed_key, rng_salt=f"{rng_salt or ''}::nonfood_gate")
        include_nonfood = bool(nonfood_gate_rng.random() < NONFOOD_INCLUDE_RATE)
        lane_ranked = _rank_anchors_by_lane(profile, context, profile_rng, active_base_dir)
        seed_anchors = _pick_three_lane_anchors(lane_ranked, profile_rng) or {}

        used_complements: set[int] = set()
        used_complement_families: dict[str, int] = {}
        used_anchors_for_person: set[int] = set()
        selected_themes: set[str] = set()
        selected_pair_fingerprints: set[tuple[str, str]] = set()
        selected_lane_groups: dict[str, set[str]] = {}
        bundles_for_person: list[dict[str, object]] = []
        used_bundle_keys_for_person: set[tuple[int, int]] = set()

        lane_candidate_ids: dict[str, list[int]] = {}
        all_ranked_ids: list[int] = []
        for lane in FOOD_LANE_ORDER:
            for pid, _score in lane_ranked.get(lane, ()):
                if pid not in all_ranked_ids:
                    all_ranked_ids.append(pid)
        for lane in FOOD_LANE_ORDER:
            primary = [pid for pid, _score in lane_ranked.get(lane, ())]
            fallback = [pid for pid in all_ranked_ids if pid not in primary]
            lane_candidate_ids[lane] = primary + fallback
            if seed_anchors.get(lane) is not None:
                first = int(seed_anchors[lane])
                lane_candidate_ids[lane] = [first] + [pid for pid in lane_candidate_ids[lane] if pid != first]

        nonfood_anchors: list[int] = []
        for pid in profile.history_product_ids:
            pid_int = int(pid)
            if pid_int <= 0 or not _is_nonfood_product(pid_int, context):
                continue
            if pid_int not in nonfood_anchors:
                nonfood_anchors.append(pid_int)
        for pid in profiled_nonfood_anchors:
            if len(nonfood_anchors) >= TOP10_ANCHOR_SIZE:
                break
            if int(pid) in nonfood_anchors:
                continue
            nonfood_anchors.append(int(pid))
        if not nonfood_anchors:
            for pid in context.non_food_ids:
                if len(nonfood_anchors) >= TOP10_ANCHOR_SIZE:
                    break
                pid_int = int(pid)
                if pid_int > 0 and pid_int not in nonfood_anchors:
                    nonfood_anchors.append(pid_int)
        lane_candidate_ids[LANE_NONFOOD] = nonfood_anchors

        candidate_lanes = [LANE_MEAL, LANE_SNACK, LANE_OCCASION]
        profiling.add_stage("profile_preparation", time.perf_counter() - prep_started)

        def _finalize_choice(
            lane_name: str,
            choice_payload: dict[str, object],
            score_value: float,
            *,
            local_dup_counter: int,
        ) -> tuple[dict[str, object] | None, int]:
            selected_anchor_id = _safe_int(choice_payload.get("anchor"), default=-1)
            selected_complement_id = _safe_int(choice_payload.get("complement"), default=-1)
            display_dict = _display_dict_for_choice(choice_payload, context=context)
            _maybe_swap_to_make_free_item_cheaper(display_dict, lane=lane_name, context=context)
            if row_to_record is not None:
                bundle_rec_local = row_to_record(pd.Series(display_dict))
            else:
                bundle_rec_local = display_dict

            history_set_local = {int(pid) for pid in profile.history_product_ids}
            a_id = _safe_int(bundle_rec_local.get("product_a"), default=-1)
            b_id = _safe_int(bundle_rec_local.get("product_b"), default=-1)
            fam_a = str(context.product_family_by_id.get(a_id, "")).strip().lower()
            fam_b = str(context.product_family_by_id.get(b_id, "")).strip().lower()
            a_nonfood = _is_nonfood_product(a_id, context)
            b_nonfood = _is_nonfood_product(b_id, context)
            if lane_name in FOOD_LANE_ORDER and (a_nonfood or b_nonfood):
                return None, local_dup_counter
            if lane_name == LANE_NONFOOD and not (a_nonfood and b_nonfood):
                return None, local_dup_counter
            if lane_name in FOOD_LANE_ORDER and fam_a and fam_b and fam_a == fam_b:
                _record_serving_telemetry(serving_telemetry, lane_name, "rejected_family_reuse")
                return None, local_dup_counter
            display_pair_key = _pair_key(a_id, b_id)
            if display_pair_key in used_bundle_keys_for_person:
                return None, int(local_dup_counter) + 1

            bundle_rec_local["lane"] = lane_name
            bundle_rec_local["lane_label"] = LANE_LABELS.get(lane_name, lane_name.upper())
            bundle_rec_local["profile_id"] = profile.profile_id
            bundle_rec_local["source"] = profile.source
            bundle_rec_local["run_id"] = str(run_id or "")
            bundle_rec_local["source_order_ids"] = list(profile.order_ids)
            bundle_rec_local["order_count"] = len(profile.order_ids)
            bundle_rec_local["history_items"] = list(profile.history_items)
            bundle_rec_local["anchor_product_id"] = int(selected_anchor_id if selected_anchor_id > 0 else a_id)
            bundle_rec_local["complement_product_id"] = int(selected_complement_id if selected_complement_id > 0 else b_id)
            origin_raw = str(choice_payload.get("source", "copurchase_fallback"))
            if not _passes_choice_score_floor(lane_name, origin_raw, float(score_value)):
                _record_serving_telemetry(serving_telemetry, lane_name, "rejected_score_floor")
                return None, local_dup_counter
            origin_group = _source_group_from_source(origin_raw)
            if origin_group in {"top_bundle", "copurchase_fallback", "fallback_food", "fallback_cleaning"}:
                origin = origin_group
            else:
                origin = origin_raw
            bundle_rec_local["recommendation_origin"] = origin
            bundle_rec_local["recommendation_origin_raw"] = origin_raw
            if origin == "top_bundle":
                origin_label = "Top-bundle match"
            elif origin == "copurchase_fallback":
                origin_label = "Copurchase fallback"
            elif origin == "fallback_cleaning":
                origin_label = "Cleaning fallback"
            else:
                origin_label = "Curated fallback"
            bundle_rec_local["recommendation_origin_label"] = origin_label
            bundle_rec_local["hybrid_reco_score"] = round(float(score_value), 3)
            bundle_rec_local["confidence_score"] = _confidence_from_score(float(score_value))
            bundle_rec_local["free_product"] = "product_b"

            name_a = str(bundle_rec_local.get("product_a_name", f"Product {a_id}")).strip()
            name_b = str(bundle_rec_local.get("product_b_name", f"Product {b_id}")).strip()
            bundle_rec_local["product_a_name"] = name_a
            bundle_rec_local["product_b_name"] = name_b
            bundle_rec_local["chosen_bundle_names"] = [x for x in (name_a, name_b) if x]
            matched = int(a_id in history_set_local) + int(b_id in history_set_local)
            bundle_rec_local["history_match_count"] = matched
            logical_anchor_id = int(bundle_rec_local.get("anchor_product_id", a_id))
            bundle_rec_local["anchor_in_history"] = bool(logical_anchor_id in history_set_local)

            reasons: list[str] = [f"{LANE_LABELS.get(lane_name, lane_name.upper())} lane"]
            if bool(bundle_rec_local.get("anchor_in_history", False)):
                reasons.insert(0, "Anchor from history")
            else:
                reasons.insert(0, "Anchor selected for lane fit")
            if origin == "top_bundle":
                reasons.append("Top-bundle match")
            elif origin == "copurchase_fallback":
                reasons.append("Copurchase fallback")
            elif origin == "fallback_cleaning":
                reasons.append("Cleaning fallback")
            else:
                reasons.append("Curated fallback")
            if float(choice_payload.get("recipe_compat", 0.0)) >= 0.18:
                reasons.append("Recipe-compatible")
            if float(choice_payload.get("prior_bonus", 0.0)) > 0.0:
                reasons.append("Known complement prior")
            if float(choice_payload.get("brand_signal", 0.0)) > 0.0:
                reasons.append("Matches preferred brand")
            if float(choice_payload.get("feedback_multiplier", 1.0)) < 1.0:
                reasons.append("Adjusted by feedback")
            if bool(choice_payload.get("feedback_boost_applied", False)):
                reasons.append("Feedback boost applied")
            if bool(choice_payload.get("feedback_penalty_applied", False)):
                reasons.append("Feedback penalty applied")
            if bool(choice_payload.get("feedback_override_applied", False)):
                reasons.append("Feedback override applied")
            feedback_class = str(choice_payload.get("feedback_class", "")).strip().lower()
            if feedback_class:
                reasons.append(f"Feedback class: {feedback_class}")
            bundle_rec_local["recommendation_reasons"] = reasons
            bundle_rec_local["feedback_boost_applied"] = bool(choice_payload.get("feedback_boost_applied", False))
            bundle_rec_local["feedback_penalty_applied"] = bool(choice_payload.get("feedback_penalty_applied", False))
            bundle_rec_local["feedback_override_applied"] = bool(choice_payload.get("feedback_override_applied", False))
            bundle_rec_local["feedback_class"] = feedback_class
            bundle_rec_local["feedback_class_applied"] = str(choice_payload.get("feedback_class_applied", feedback_class))
            bundle_rec_local["feedback_multiplier_applied"] = float(choice_payload.get("feedback_multiplier_applied", 1.0))
            bundle_rec_local["semantic_roles_a"] = list(choice_payload.get("semantic_roles_a", []))
            bundle_rec_local["semantic_roles_b"] = list(choice_payload.get("semantic_roles_b", []))
            bundle_rec_local["pair_relation"] = str(choice_payload.get("pair_relation", ""))
            bundle_rec_local["pair_strength"] = str(choice_payload.get("pair_strength", ""))
            bundle_rec_local["lane_fit_score"] = float(choice_payload.get("lane_fit_score", 0.0))
            bundle_rec_local["internal_lane_fit"] = str(choice_payload.get("internal_lane_fit", ""))
            bundle_rec_local["semantic_reject_reason"] = str(choice_payload.get("semantic_reject_reason", ""))
            bundle_rec_local["semantic_engine_version"] = str(choice_payload.get("semantic_engine_version", SEMANTIC_ENGINE_VERSION))

            snack_pattern = str(choice_payload.get("snack_pattern", "")).strip()
            if snack_pattern:
                bundle_rec_local["snack_pattern"] = snack_pattern
            theme = str(choice_payload.get("theme", "")).strip()
            if theme:
                bundle_rec_local["bundle_theme"] = theme
                selected_themes.add(theme)
            raw_fingerprint = choice_payload.get("pair_fingerprint")
            if isinstance(raw_fingerprint, tuple) and len(raw_fingerprint) == 2:
                fp0 = str(raw_fingerprint[0])
                fp1 = str(raw_fingerprint[1])
                selected_pair_fingerprints.add((fp0, fp1))
            group_union_raw = choice_payload.get("group_union")
            if isinstance(group_union_raw, list):
                selected_lane_groups[lane_name] = {str(item) for item in group_union_raw if str(item).strip()}
            comp_family = str(context.product_family_by_id.get(int(b_id), "")).strip().lower() or "other"
            used_bundle_keys_for_person.add(display_pair_key)
            used_anchors_for_person.add(int(logical_anchor_id))
            used_complements.add(int(b_id))
            used_complement_families[comp_family] = int(used_complement_families.get(comp_family, 0)) + 1
            global_anchor_counts[int(logical_anchor_id)] = int(global_anchor_counts.get(int(logical_anchor_id), 0)) + 1
            lane_key_local = (lane_name, int(logical_anchor_id))
            global_anchor_lane_counts[lane_key_local] = int(global_anchor_lane_counts.get(lane_key_local, 0)) + 1
            if origin == "top_bundle":
                _record_serving_telemetry(serving_telemetry, lane_name, "chose_top_bundle")
            elif origin == "copurchase_fallback":
                _record_serving_telemetry(serving_telemetry, lane_name, "chose_copurchase_fallback")
            elif origin == "fallback_cleaning":
                _record_serving_telemetry(serving_telemetry, lane_name, "chose_cleaning_fallback")
            elif origin.startswith("fallback_"):
                _record_serving_telemetry(serving_telemetry, lane_name, "chose_food_fallback")
                _record_serving_telemetry(serving_telemetry, lane_name, "chose_template_fallback")
            else:
                _record_serving_telemetry(serving_telemetry, lane_name, "chose_other")
            return bundle_rec_local, int(local_dup_counter)

        recency_map = _profile_recency_order(profile)
        lane_intent_map = _profile_lane_intent_map(profile, context)
        recent_category_interest = _profile_recent_category_interest(profile, context)
        recent_brand_affinity = _profile_recent_brand_affinity(profile, context)
        profile_product_interest = _profile_product_interest(profile)
        profile_family_interest = _profile_family_interest(profile, context)
        profile_category_affinity = _profile_category_affinity(profile, context)
        profile_shopper_family_interest = _profile_shopper_family_interest(profile, context)
        candidate_pool: dict[tuple[str, int, int, str], dict[str, object]] = {}
        local_duplicate_blocked = 0
        candidate_started = time.perf_counter()

        def _register_candidate(lane_name: str, choice: dict[str, object], score_value: float) -> None:
            anchor_id = int(_safe_int(choice.get("anchor"), default=-1))
            complement_id = int(_safe_int(choice.get("complement"), default=-1))
            if anchor_id <= 0 or complement_id <= 0 or anchor_id == complement_id:
                return
            row = choice.get("bundle_row")
            pair_row = row if isinstance(row, pd.Series) else None
            final_quality_reject = _final_human_quality_reject_reason(
                anchor_id,
                complement_id,
                str(lane_name),
                context,
                pair_row=pair_row,
            )
            if final_quality_reject:
                _record_serving_telemetry(serving_telemetry, lane_name, "rejected_final_human_quality")
                return
            source = str(choice.get("source", "copurchase_fallback"))
            key = (str(lane_name), int(anchor_id), int(complement_id), source)
            candidate = dict(choice)
            candidate["lane"] = str(lane_name)
            candidate["base_score"] = float(score_value)
            candidate["personalization_boost"] = float(
                _candidate_personalization_boost(
                    candidate,
                    profile,
                    context,
                    recency_map=recency_map,
                    lane_intent_map=lane_intent_map,
                    recent_category_interest=recent_category_interest,
                    recent_brand_affinity=recent_brand_affinity,
                    profile_product_interest=profile_product_interest,
                    profile_family_interest=profile_family_interest,
                    profile_category_affinity=profile_category_affinity,
                    profile_shopper_family_interest=profile_shopper_family_interest,
                )
            )
            candidate["pool_score"] = float(candidate["base_score"]) + float(candidate["personalization_boost"])
            candidate["motif_family_signature"] = _candidate_motif_family_signature(candidate, context)
            candidate["family_pattern_signature"] = _candidate_family_pattern_signature(candidate, context)
            candidate["bundle_shape_signature"] = _candidate_bundle_shape_signature(candidate, context)
            fallback_quality_reject = _fallback_quality_reject_reason(candidate, context)
            if fallback_quality_reject:
                _record_serving_telemetry(serving_telemetry, lane_name, "rejected_fallback_quality")
                return
            candidate["fallback_motif_key"] = _controlled_fallback_motif_key(candidate, context)
            existing = candidate_pool.get(key)
            if existing is None or float(candidate["pool_score"]) > float(existing.get("pool_score", -1e9)):
                candidate_pool[key] = candidate

        for lane in candidate_lanes:
            if lane == LANE_NONFOOD and not include_nonfood:
                continue
            for anchor in lane_candidate_ids.get(lane, ()):
                if not _anchor_allowed_for_lane(int(anchor), lane, context):
                    continue
                choice, _choice_key, personal_score, duplicate_blocked = _pick_candidate_for_anchor(
                    profile=profile,
                    anchor=int(anchor),
                    lane=lane,
                    context=context,
                    top_bundle_rows_by_anchor=top_bundle_rows_by_anchor,
                    bundle_lookup=bundle_lookup,
                    used_pair_keys=set(),
                    feedback_lookup=feedback_lookup,
                    rng=profile_rng,
                    used_complements=set(),
                    used_complement_families={},
                    global_anchor_lane_counts=global_anchor_lane_counts,
                    global_anchor_counts=global_anchor_counts,
                    reject_counters=gate_reject_counters,
                    allow_anchor_overflow=True,
                    serving_telemetry=serving_telemetry,
                )
                local_duplicate_blocked += int(duplicate_blocked)
                if choice is None:
                    continue
                _register_candidate(lane, choice, float(personal_score))

            history_ids_for_lane = {
                int(pid)
                for pid in profile.history_product_ids
                if int(pid) > 0 and _anchor_allowed_for_lane(int(pid), lane, context)
            }
            history_ids_for_lane.update(int(pid) for pid in lane_candidate_ids.get(lane, ())[:TOP10_ANCHOR_SIZE] if int(pid) > 0)
            for fallback_anchor, fallback_complement, _fallback_row, fallback_source in _fallback_candidates_for_lane(
                history_ids_for_lane,
                lane,
                context,
                top_bundle_rows_by_anchor,
                bundle_lookup,
            ):
                choice, _choice_key, personal_score, duplicate_blocked = _pick_candidate_for_anchor(
                    profile=profile,
                    anchor=int(fallback_anchor),
                    lane=lane,
                    context=context,
                    top_bundle_rows_by_anchor=top_bundle_rows_by_anchor,
                    bundle_lookup=bundle_lookup,
                    used_pair_keys=set(),
                    feedback_lookup=feedback_lookup,
                    rng=profile_rng,
                    used_complements=set(),
                    used_complement_families={},
                    global_anchor_lane_counts=global_anchor_lane_counts,
                    global_anchor_counts=global_anchor_counts,
                    reject_counters=gate_reject_counters,
                    allow_anchor_overflow=True,
                    allowed_complements={int(fallback_complement)},
                    serving_telemetry=serving_telemetry,
                )
                local_duplicate_blocked += int(duplicate_blocked)
                if choice is None:
                    continue
                choice["source"] = str(fallback_source)
                _register_candidate(lane, choice, float(personal_score))

        profiling.add_stage("candidate_generation", time.perf_counter() - candidate_started)
        global_duplicate_pair_count_blocked += int(local_duplicate_blocked)
        ordered_candidates = list(candidate_pool.values())
        selected_choices: list[dict[str, object]] = []
        selected_anchors: set[int] = set()
        selected_pairs: set[tuple[int, int]] = set()
        cleaning_count = 0
        selected_fallback_motif_counts: dict[str, int] = {}

        scoring_started = time.perf_counter()
        enriched_candidates: list[dict[str, object]] = []
        for candidate in ordered_candidates:
            enriched = dict(candidate)
            effective_score = _candidate_effective_score(
                enriched,
                context,
                global_pair_exposure=global_pair_exposure,
                global_template_exposure=global_template_exposure,
                global_fallback_motif_exposure=global_fallback_motif_exposure,
                global_motif_family_exposure=global_motif_family_exposure,
                global_family_pattern_exposure=global_family_pattern_exposure,
                global_bundle_shape_exposure=global_bundle_shape_exposure,
            )
            motif_signature = str(enriched.get("motif_family_signature", "")).strip().lower()
            family_exposure_count = int(global_motif_family_exposure.get(motif_signature, 0)) if motif_signature else 0
            family_rank_penalty = min(
                FAMILY_OVERUSE_RANK_PENALTY_CAP,
                FAMILY_OVERUSE_RANK_PENALTY_STEP * float(max(0, family_exposure_count - 1)),
            )
            enriched["effective_score"] = float(effective_score)
            enriched["family_exposure_count"] = int(family_exposure_count)
            # Explicit close-score preference: when candidates are similarly strong, prefer rarer motif families.
            enriched["effective_score_rank"] = float(effective_score) - float(family_rank_penalty)
            enriched_candidates.append(enriched)

        strong_food_candidates = [
            cand
            for cand in enriched_candidates
            if str(cand.get("lane", "")).strip().lower() in FOOD_LANE_ORDER
            and not _is_household_bundle(cand, context)
            and _candidate_is_strong_finalist(cand)
        ]
        best_strong_score = max((float(cand.get("effective_score", 0.0)) for cand in strong_food_candidates), default=float("-inf"))
        best_rare_strong_score = max(
            (
                float(cand.get("effective_score", 0.0))
                for cand in strong_food_candidates
                if int(_safe_int(cand.get("family_exposure_count"), default=0)) < FAMILY_CLOSE_SCORE_OVERUSE_THRESHOLD
            ),
            default=float("-inf"),
        )
        for cand in strong_food_candidates:
            lane_name = str(cand.get("lane", "")).strip().lower()
            motif_signature = str(cand.get("motif_family_signature", "")).strip().lower()
            family_exposure_count = int(_safe_int(cand.get("family_exposure_count"), default=0))
            effective_score = float(cand.get("effective_score", 0.0))
            rank_score = float(cand.get("effective_score_rank", effective_score))
            if family_exposure_count >= 2 and _is_dominant_shopper_family_signature(motif_signature):
                rank_score -= 0.34 * float((family_exposure_count - 1) ** 1.1)
                if lane_name == LANE_MEAL:
                    rank_score -= 0.20 * float(family_exposure_count - 1)
            if family_exposure_count >= 1 and _is_utilitarian_shopper_family_signature(motif_signature):
                rank_score -= 0.28 * float(family_exposure_count)

            close_to_rare = (
                best_rare_strong_score > float("-inf")
                and best_rare_strong_score + FAMILY_RARITY_CLOSE_SCORE_MARGIN >= effective_score
            )
            if family_exposure_count >= FAMILY_CLOSE_SCORE_OVERUSE_THRESHOLD and close_to_rare:
                overuse_steps = family_exposure_count - FAMILY_CLOSE_SCORE_OVERUSE_THRESHOLD + 1
                rank_score -= 0.30 * float(overuse_steps)
                if _is_dominant_shopper_family_signature(motif_signature):
                    rank_score -= 0.20 * float(overuse_steps)
                if lane_name == LANE_MEAL and _is_meal_dominant_motif_signature(motif_signature):
                    rank_score -= MEAL_DOMINANT_CLOSE_SCORE_EXTRA_PENALTY * float(overuse_steps)
                if _is_utilitarian_shopper_family_signature(motif_signature):
                    rank_score -= 0.22 * float(overuse_steps)

            if family_exposure_count < FAMILY_CLOSE_SCORE_OVERUSE_THRESHOLD and best_strong_score > float("-inf"):
                close_gap = float(best_strong_score - effective_score)
                if 0.0 <= close_gap <= FAMILY_RARITY_CLOSE_SCORE_MARGIN:
                    rarity_boost = FAMILY_RARITY_CLOSE_SCORE_BONUS * (
                        1.0 - (close_gap / max(FAMILY_RARITY_CLOSE_SCORE_MARGIN, 1e-9))
                    )
                    if _is_dominant_shopper_family_signature(motif_signature):
                        rarity_boost *= 0.65
                    if lane_name == LANE_MEAL and _is_utilitarian_shopper_family_signature(motif_signature):
                        rarity_boost *= 0.55
                    rank_score += float(max(0.0, rarity_boost))

            cand["effective_score_rank"] = float(rank_score)

        food_candidates = [
            cand
            for cand in enriched_candidates
            if str(cand.get("lane", "")).strip().lower() in FOOD_LANE_ORDER
            and not _is_household_bundle(cand, context)
            and _candidate_is_strong_finalist(cand)
        ]
        food_candidates.sort(
            key=lambda cand: (
                -float(cand.get("effective_score_rank", cand.get("effective_score", 0.0))),
                int(_safe_int(cand.get("family_exposure_count"), default=0)),
                -float(cand.get("effective_score", 0.0)),
                _source_priority_rank(str(cand.get("source", ""))),
                int(_safe_int(cand.get("pair_count"), default=0)) * -1,
                int(_safe_int(cand.get("complement"), default=-1)),
                int(_safe_int(cand.get("anchor"), default=-1)),
            )
        )
        profiling.add_stage("scoring_ranking", time.perf_counter() - scoring_started)

        def _try_select(candidate: dict[str, object], *, allow_cleaning: bool) -> bool:
            nonlocal cleaning_count
            anchor_id = int(_safe_int(candidate.get("anchor"), default=-1))
            complement_id = int(_safe_int(candidate.get("complement"), default=-1))
            if anchor_id <= 0 or complement_id <= 0 or anchor_id == complement_id:
                return False
            lane_name = str(candidate.get("lane", "")).strip().lower() or LANE_MEAL
            row = candidate.get("bundle_row")
            pair_row = row if isinstance(row, pd.Series) else None
            final_quality_reject = _final_human_quality_reject_reason(
                anchor_id,
                complement_id,
                lane_name,
                context,
                pair_row=pair_row,
            )
            if final_quality_reject:
                _record_serving_telemetry(serving_telemetry, lane_name, "rejected_final_human_quality")
                return False
            pair_key = _pair_key(anchor_id, complement_id)
            if anchor_id in selected_anchors or pair_key in selected_pairs:
                return False
            is_cleaning = _is_household_bundle(candidate, context)
            if is_cleaning and not allow_cleaning:
                return False
            if is_cleaning and cleaning_count >= MAX_CLEANING_BUNDLES_PER_PERSON:
                return False
            source_group = _source_group_from_source(str(candidate.get("source", "")))
            fallback_motif = str(candidate.get("fallback_motif_key", "")).strip()
            if source_group == "fallback_food" and fallback_motif in CONTROLLED_FALLBACK_MOTIFS:
                current_motif_count = int(selected_fallback_motif_counts.get(fallback_motif, 0))
                if current_motif_count >= FALLBACK_MOTIF_REPEAT_CAP_PER_PERSON:
                    _record_serving_telemetry(serving_telemetry, lane_name, "rejected_fallback_motif_repeat")
                    return False
                if fallback_motif in EVAP_REPETITIVE_FALLBACK_MOTIFS:
                    evap_count = sum(
                        int(selected_fallback_motif_counts.get(motif_key, 0))
                        for motif_key in EVAP_REPETITIVE_FALLBACK_MOTIFS
                    )
                    if evap_count >= FALLBACK_EVAP_MOTIF_CAP_PER_PERSON:
                        _record_serving_telemetry(serving_telemetry, lane_name, "rejected_fallback_evap_motif_cap")
                        return False
            selected_choices.append(candidate)
            selected_anchors.add(anchor_id)
            selected_pairs.add(pair_key)
            if is_cleaning:
                cleaning_count += 1
            if source_group == "fallback_food" and fallback_motif in CONTROLLED_FALLBACK_MOTIFS:
                selected_fallback_motif_counts[fallback_motif] = int(selected_fallback_motif_counts.get(fallback_motif, 0)) + 1
            return True

        selection_started = time.perf_counter()
        # Mild lane-balance nudge: if a strong non-meal option is close in score, take one early.
        if len(selected_choices) < MAX_BUNDLES_PER_PERSON:
            top_meal_score = max(
                (
                    float(cand.get("effective_score", 0.0))
                    for cand in food_candidates
                    if str(cand.get("lane", "")).strip().lower() == LANE_MEAL
                ),
                default=float("-inf"),
            )
            non_meal_candidates = [
                cand
                for cand in food_candidates
                if str(cand.get("lane", "")).strip().lower() in {LANE_SNACK, LANE_OCCASION}
            ]
            for candidate in non_meal_candidates:
                candidate_lane = str(candidate.get("lane", "")).strip().lower()
                if float(lane_intent_map.get(candidate_lane, 0.0)) < 0.20:
                    continue
                candidate_score = float(candidate.get("effective_score", 0.0))
                if top_meal_score > float("-inf") and candidate_score + NON_MEAL_COVERAGE_MARGIN < top_meal_score:
                    continue
                if _try_select(candidate, allow_cleaning=False):
                    break

        for candidate in food_candidates:
            if _try_select(candidate, allow_cleaning=False) and len(selected_choices) >= MAX_BUNDLES_PER_PERSON:
                break

        if len(selected_choices) < MAX_BUNDLES_PER_PERSON:
            cleaning_history_ids = {
                int(pid)
                for pid in profile.history_product_ids
                if int(pid) > 0 and _is_nonfood_product(int(pid), context)
            }
            cleaning_history_ids.update(int(pid) for pid in lane_candidate_ids.get(LANE_NONFOOD, ())[:TOP10_ANCHOR_SIZE] if int(pid) > 0)
            cleaning_candidates: list[dict[str, object]] = []
            for fallback_anchor, fallback_complement, _fallback_row, fallback_source in _fallback_candidates_for_lane(
                cleaning_history_ids,
                LANE_NONFOOD,
                context,
                top_bundle_rows_by_anchor,
                bundle_lookup,
            ):
                choice, _choice_key, personal_score, duplicate_blocked = _pick_candidate_for_anchor(
                    profile=profile,
                    anchor=int(fallback_anchor),
                    lane=LANE_NONFOOD,
                    context=context,
                    top_bundle_rows_by_anchor=top_bundle_rows_by_anchor,
                    bundle_lookup=bundle_lookup,
                    used_pair_keys=set(),
                    feedback_lookup=feedback_lookup,
                    rng=profile_rng,
                    used_complements=set(),
                    used_complement_families={},
                    global_anchor_lane_counts=global_anchor_lane_counts,
                    global_anchor_counts=global_anchor_counts,
                    reject_counters=gate_reject_counters,
                    allow_anchor_overflow=True,
                    allowed_complements={int(fallback_complement)},
                    serving_telemetry=serving_telemetry,
                )
                local_duplicate_blocked += int(duplicate_blocked)
                if choice is None:
                    continue
                enriched = dict(choice)
                enriched["source"] = str(fallback_source)
                enriched["lane"] = LANE_NONFOOD
                enriched["base_score"] = float(personal_score)
                enriched["pool_score"] = float(personal_score)
                enriched["effective_score"] = _candidate_effective_score(
                    enriched,
                    context,
                    global_pair_exposure=global_pair_exposure,
                    global_template_exposure=global_template_exposure,
                    global_fallback_motif_exposure=global_fallback_motif_exposure,
                    global_motif_family_exposure=global_motif_family_exposure,
                    global_family_pattern_exposure=global_family_pattern_exposure,
                    global_bundle_shape_exposure=global_bundle_shape_exposure,
                )
                cleaning_candidates.append(enriched)
            cleaning_candidates.sort(
                key=lambda cand: (
                    -float(cand.get("effective_score", 0.0)),
                    _source_priority_rank(str(cand.get("source", ""))),
                    int(_safe_int(cand.get("complement"), default=-1)),
                    int(_safe_int(cand.get("anchor"), default=-1)),
                )
            )
            for candidate in cleaning_candidates:
                if _try_select(candidate, allow_cleaning=True):
                    break

        profiling.add_stage("selection_fill", time.perf_counter() - selection_started)
        finalize_started = time.perf_counter()
        for chosen in selected_choices:
            lane = str(chosen.get("lane", LANE_MEAL)).strip().lower()
            bundle_rec, local_duplicate_blocked = _finalize_choice(
                lane,
                chosen,
                float(chosen.get("base_score", chosen.get("personal_score", 0.0))),
                local_dup_counter=0,
            )
            if bundle_rec is None:
                continue
            pair_exposure_key = _pair_key(
                int(_safe_int(bundle_rec.get("anchor_product_id"), default=-1)),
                int(_safe_int(bundle_rec.get("complement_product_id"), default=-1)),
            )
            global_pair_exposure[pair_exposure_key] = int(global_pair_exposure.get(pair_exposure_key, 0)) + 1
            template_signature = _template_signature(chosen, context)
            global_template_exposure[template_signature] = int(global_template_exposure.get(template_signature, 0)) + 1
            motif_key = _controlled_fallback_motif_key(chosen, context)
            if motif_key in CONTROLLED_FALLBACK_MOTIFS:
                global_fallback_motif_exposure[motif_key] = int(global_fallback_motif_exposure.get(motif_key, 0)) + 1
            motif_family_signature = _candidate_motif_family_signature(chosen, context)
            if motif_family_signature:
                global_motif_family_exposure[motif_family_signature] = int(
                    global_motif_family_exposure.get(motif_family_signature, 0)
                ) + 1
            family_pattern_signature = _candidate_family_pattern_signature(chosen, context)
            if family_pattern_signature:
                global_family_pattern_exposure[family_pattern_signature] = int(
                    global_family_pattern_exposure.get(family_pattern_signature, 0)
                ) + 1
            bundle_shape_signature = _candidate_bundle_shape_signature(chosen, context)
            if bundle_shape_signature:
                global_bundle_shape_exposure[bundle_shape_signature] = int(
                    global_bundle_shape_exposure.get(bundle_shape_signature, 0)
                ) + 1
            bundles_for_person.append(bundle_rec)
            if len(bundles_for_person) >= MAX_BUNDLES_PER_PERSON:
                break

        profiling.add_stage("output_assembly", time.perf_counter() - finalize_started)
        bundles_for_person = bundles_for_person[:MAX_BUNDLES_PER_PERSON]

        if len(bundles_for_person) < MAX_BUNDLES_PER_PERSON and str(profile.source).strip().lower() == "random":
            replaced = False
            _record_serving_telemetry(serving_telemetry, "global", "resample_attempted")
            while resample_budget > 0 and not replaced:
                resample_budget -= 1
                resample_rng = _rng_for_profile(
                    run_id,
                    profile_seed_key,
                    rng_salt=f"{rng_salt or ''}::resample::{resample_budget}",
                )
                candidate_profile = build_random_profile(order_pool, rng=resample_rng)
                if candidate_profile is None:
                    continue
                signature = tuple(sorted(int(pid) for pid in candidate_profile.history_product_ids if int(pid) > 0))
                if not signature or signature in seen_signatures:
                    _record_serving_telemetry(serving_telemetry, "global", "resample_duplicate_signature")
                    continue
                seen_signatures.add(signature)
                profile_queue.append(candidate_profile)
                _record_serving_telemetry(serving_telemetry, "global", "resample_success")
                replaced = True
            if replaced:
                profile_total_sec = time.perf_counter() - profile_started
                profiling.add_stage("profile_total", profile_total_sec)
                profiling.add_profile(
                    {
                        "profile_id": str(profile.profile_id),
                        "source": str(profile.source),
                        "status": "resampled",
                        "bundle_count": int(len(bundles_for_person)),
                        "duration_sec": float(profile_total_sec),
                    }
                )
                continue
            _record_serving_telemetry(serving_telemetry, "global", "resample_exhausted")

        person_label = f"Person {len(recommendations) + 1}"
        selected_food_lanes = {str(b.get("lane", "")) for b in bundles_for_person if str(b.get("lane", "")) in FOOD_LANE_ORDER}
        missing_food_lanes = [lane_name for lane_name in FOOD_LANE_ORDER if lane_name not in selected_food_lanes]
        person_recommendation: dict[str, object] = {
            "person_label": person_label,
            "profile_id": profile.profile_id,
            "source": profile.source,
            "run_id": str(run_id or ""),
            "source_order_ids": list(profile.order_ids),
            "order_count": len(profile.order_ids),
            "history_items": list(profile.history_items),
            "history_match_count": int(sum(int(b.get("history_match_count", 0)) for b in bundles_for_person)),
            "missing_food_lanes": missing_food_lanes,
            "bundles": bundles_for_person,
        }
        recommendations.append(person_recommendation)
        profile_total_sec = time.perf_counter() - profile_started
        profiling.add_stage("profile_total", profile_total_sec)
        profiling.add_profile(
            {
                "profile_id": str(profile.profile_id),
                "source": str(profile.source),
                "status": "served",
                "bundle_count": int(len(bundles_for_person)),
                "duration_sec": float(profile_total_sec),
            }
        )

    _write_person_quality_artifact(
        active_base_dir,
        recommendations,
        context=context,
        run_id=run_id,
        global_duplicate_pair_count_blocked=global_duplicate_pair_count_blocked,
        anchor_reuse_counts=global_anchor_counts,
        guardrail_reject_counts=gate_reject_counters,
        serving_telemetry=serving_telemetry,
    )
    total_runtime_sec = time.perf_counter() - overall_started
    profile_durations = [float(item.get("duration_sec", 0.0)) for item in profiling.per_profile if float(item.get("duration_sec", 0.0)) > 0]
    metrics_payload = {
        "enabled": bool(profiling.enabled),
        "run_id": str(run_id or ""),
        "profile_count": int(len(profile_durations)),
        "total_runtime_sec": float(total_runtime_sec),
        "avg_latency_sec": float((sum(profile_durations) / len(profile_durations)) if profile_durations else 0.0),
        "p50_latency_sec": float(_percentile(profile_durations, 0.50)),
        "p90_latency_sec": float(_percentile(profile_durations, 0.90)),
        "p95_latency_sec": float(_percentile(profile_durations, 0.95)),
        "stage_totals_sec": {str(k): float(v) for k, v in sorted(profiling.stage_totals.items())},
        "per_profile": list(profiling.per_profile),
    }
    _LAST_SERVING_PROFILE_METRICS = metrics_payload
    if profiling.enabled:
        print(
            "[qeu-serving-profile] "
            f"profiles={metrics_payload['profile_count']} "
            f"total_sec={metrics_payload['total_runtime_sec']:.3f} "
            f"avg_sec={metrics_payload['avg_latency_sec']:.3f} "
            f"p50_sec={metrics_payload['p50_latency_sec']:.3f} "
            f"p90_sec={metrics_payload['p90_latency_sec']:.3f} "
            f"p95_sec={metrics_payload['p95_latency_sec']:.3f}"
        )
        if profiling.stage_totals:
            stage_parts = ", ".join(
                f"{stage}={value:.3f}s"
                for stage, value in sorted(profiling.stage_totals.items(), key=lambda item: (-item[1], item[0]))
            )
            print(f"[qeu-serving-profile] stage_totals: {stage_parts}")
    return recommendations
