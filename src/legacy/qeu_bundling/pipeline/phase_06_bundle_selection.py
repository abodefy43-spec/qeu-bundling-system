"""Phase 6: Bundle selection with recipe-first anchors and cheap complements."""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.product_families import assign_family, load_families

NUMERIC_FEATURES = [
    "product_a_price",
    "product_b_price",
    "recipe_score_a",
    "recipe_score_b",
    "purchase_score",
    "embedding_score",
    "shared_categories_count",
    "shared_category_score",
    "category_match",
    "is_campaign_pair",
]
CATEGORICAL_FEATURES = ["category_a", "category_b", "importance_a", "importance_b"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

SHARED_CATEGORY_WEIGHT = 0.15
RECIPE_WEIGHT = 0.15
COPURCHASE_WEIGHT = 0.15
EMBEDDING_WEIGHT = 0.05
VERSATILITY_WEIGHT = 0.03
FREQUENCY_WEIGHT = 0.07
CAMPAIGN_WEIGHT = 0.10
MODEL_SCORE_WEIGHT = 0.30

USE_MODEL_FOR_RANKING = True
ENABLE_SHADOW_COMPARISON = True

MIN_ORDERS_PER_PRODUCT = 50
TOP_PRODUCTS_FOR_PAIRING = 4000
TOP_N_BUNDLES = 2000
MAX_ANCHORS = 1500
MAX_COPURCHASE_NEIGHBORS = 80
MAX_TAG_COMPLEMENTS = 120
MAX_CANDIDATES_PER_ANCHOR = 160
PERSON_CANDIDATE_ARTIFACT = "person_candidate_pairs.csv"
PAIR_SCORING_BREAKDOWN_ARTIFACT = "pair_scoring_breakdown.csv"
SUSPICIOUS_AUDIT_ARTIFACT = "suspicious_pairs_audit.csv"
RETENTION_METRICS_ARTIFACT = "phase06_retention_metrics.json"
RETAINED_CANDIDATE_CAP = 12000
ANCHOR_RETENTION_PASS_A = 4

MIN_SHARED_CATEGORIES = 2
MAX_EMBEDDING_SIMILARITY = 55.0
CHEAP_RATIO_MAX = 0.50
ADAPTIVE_PRICE_MAX = 0.80
ADAPTIVE_PRICE_RATIO_SHARE_MAX = 0.15
EMBEDDING_TOO_FAR = 5.0
EMBEDDING_BAND_LOW = 15.0
EMBEDDING_BAND_HIGH = 35.0
EMBEDDING_BAND_CENTER = 25.0

MAX_BUNDLES_PER_THEME = 25
MAX_APPEARANCES_PER_PRODUCT = 12
MAX_TEMPLATE_REPEATS_SOFT = 2
VEGETABLE_PENALTY_ONE = 0.85
VEGETABLE_PENALTY_BOTH = 0.60
RAMADAN_BOOST_BASE = 1.08
RAMADAN_BOOST_DISH = 1.15
LEXICAL_ONLY_PENALTY = 0.70
NON_FOOD_TAG = "non_food"
GENERIC_TAGS = frozenset(
    {
        "ingredient",
        "saudi",
        "dish",
        "qeu_category",
        "saudi_specialties",
        "saudi_staples",
        "frequent_purchase",
        "ramadan",
    }
)
SAUDI_DISH_TAGS = frozenset(
    {
        "kabsa",
        "jareesh",
        "saleeg",
        "samboosa",
        "quaker_soup",
        "vimto",
        "qamar_al_din",
        "laban",
        "saudi_coffee",
        "chocolate_biscuit_dessert",
    }
)
PARENT_GROUP_BY_FAMILY = {
    "milk_dairy": "cow_produce",
    "cheese": "cow_produce",
    "ghee_fats": "cow_produce",
    "chips_snacks": "snack",
    "rice_centric": "grain",
}
CATEGORY_PARENT_FALLBACK = {
    "dairy": "cow_produce",
    "snacks": "snack",
    "beverages": "beverage",
    "grains": "grain",
}
SAME_PARENT_PENALTY = 0.70
CROSS_PARENT_BONUS = 1.05

ANCHOR_RECIPE_WEIGHT = 0.60
ANCHOR_FREQUENCY_WEIGHT = 0.25
ANCHOR_CAMPAIGN_WEIGHT = 0.15
COMPLEMENT_PURCHASE_WEIGHT = 0.60
COMPLEMENT_TAG_WEIGHT = 0.25
COMPLEMENT_EMBEDDING_WEIGHT = 0.15
NEW_ANCHOR_WEIGHT = 0.45
NEW_COMPLEMENT_WEIGHT = 0.45
NEW_CAMPAIGN_WEIGHT = 0.10

UTILITY_CATEGORIES = frozenset({"spices", "herbs"})
UTILITY_TOKENS = frozenset({"salt", "seasonings", "seasoning", "spices", "fresh_herbs", "mint"})
READY_TO_CONSUME_CATEGORIES = frozenset({"fruits", "dairy", "beverages", "snacks"})
UTILITY_PENALTY_MIN = 0.75
UTILITY_PENALTY_MAX = 0.92
ADAPTIVE_PRICE_MIN_NEW_RAW = 42.0
MAX_DEAL_BOOST_SHARE = 0.05

SAVORY_BASE_HINTS = frozenset(
    {
        "tomato paste",
        "paste",
        "bouillon",
        "stock",
        "seasoning",
        "seasoning mix",
        "cube",
        "cooking oil",
        "vegetable oil",
        "sauce base",
    }
)
SWEET_PROCESSED_HINTS = frozenset(
    {
        "jelly",
        "jam",
        "candy",
        "dessert",
        "chocolate",
        "biscuit",
        "cookie",
        "marshmallow",
        "wafer",
        "sweet snack",
    }
)
SWEET_SAVORY_OVERRIDE_MIN_PAIR_COUNT = 3
SWEET_SAVORY_OVERRIDE_MIN_CP_SCORE = 0.15
RECIPE_ONLY_MIN_SCORE_NORM = 0.25
RECIPE_STAPLES = frozenset(
    {
        "salt",
        "sugar",
        "oil",
        "oils",
        "water",
        "flour",
        "rice",
        "tomato",
        "tomatoes",
        "onion",
        "onions",
        "garlic",
    }
)

PREFERENCE_NEGATIVE_MIN_MULT = 0.78
PREFERENCE_NEGATIVE_MAX_MULT = 0.92
PREFERENCE_POSITIVE_MIN_MULT = 1.05
PREFERENCE_POSITIVE_MAX_MULT = 1.12

PAIR_PENALTY_RULES_FILE = "pair_penalty_rules.json"

SIZE_WORDS = {
    "kg",
    "ÙƒÙŠÙ„Ùˆ",
    "ÙƒØ¬Ù…",
    "Ø¬Ù…",
    "g",
    "gram",
    "grams",
    "l",
    "liter",
    "litre",
    "ml",
    "pack",
    "pcs",
    "pc",
    "Ù‚Ø·Ø¹Ø©",
    "Ù‚Ø·Ø¹",
    "Ø­Ø¨Ø§Øª",
    "Ø­Ø¨Ø©",
    "x",
}


def _data_dir() -> Path:
    return get_paths().data_processed_dir


def _candidates_dir() -> Path:
    path = get_paths().data_processed_candidates_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _diagnostics_dir() -> Path:
    path = get_paths().data_processed_diagnostics_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def _reference_dir() -> Path:
    return get_paths().data_reference_dir


def _output_dir() -> Path:
    return get_paths().output_dir


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalise_name(value: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", str(value).lower()).strip()


def _core_tokens(name: str) -> list[str]:
    tokens = _normalise_name(name).split()
    core = [t for t in tokens if t not in SIZE_WORDS and not t.isdigit()]
    return core if core else tokens


def _is_same_product_variant(name_a: str, name_b: str) -> bool:
    core_a = _core_tokens(name_a)
    core_b = _core_tokens(name_b)
    if not core_a or not core_b:
        return False
    text_a, text_b = " ".join(core_a), " ".join(core_b)
    if text_a == text_b:
        return True
    set_a, set_b = set(core_a), set(core_b)
    overlap = len(set_a & set_b)
    if overlap == 0:
        return False
    ratio = overlap / max(len(set_a), len(set_b))
    return (ratio >= 0.8 and overlap >= 2) or ((text_a in text_b or text_b in text_a) and ratio >= 0.66 and overlap >= 2)


def _parse_tags(value: str | float | int | None, fallback: str = "other") -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {fallback}
    text = str(value).strip()
    if not text:
        return {fallback}
    tags = {x for x in text.split("|") if x}
    return tags if tags else {fallback}


def _specific_tags(tags: set[str]) -> set[str]:
    return {t for t in tags if t and t not in GENERIC_TAGS and t != NON_FOOD_TAG}


def _embedding_band_score(embedding_score: float) -> float:
    sim = float(embedding_score)
    if sim >= MAX_EMBEDDING_SIMILARITY or sim <= EMBEDDING_TOO_FAR:
        return 0.0
    if sim < EMBEDDING_BAND_LOW:
        return _clip((sim - EMBEDDING_TOO_FAR) / (EMBEDDING_BAND_LOW - EMBEDDING_TOO_FAR) * 50.0, 0.0, 50.0)
    if sim <= EMBEDDING_BAND_HIGH:
        width = max(EMBEDDING_BAND_HIGH - EMBEDDING_BAND_CENTER, 1.0)
        return _clip(100.0 - abs(sim - EMBEDDING_BAND_CENTER) * (50.0 / width), 50.0, 100.0)
    return _clip((MAX_EMBEDDING_SIMILARITY - sim) / (MAX_EMBEDDING_SIMILARITY - EMBEDDING_BAND_HIGH) * 50.0, 0.0, 50.0)


def _legacy_embedding_score(embedding_score: float) -> float:
    peak = 30.0
    if embedding_score <= peak:
        return float(embedding_score)
    return float(_clip(peak - (embedding_score - peak) * 0.6, 0.0, 100.0))


def _passes_cheap_constraint(price_a: float, price_b: float, max_ratio: float = CHEAP_RATIO_MAX) -> bool:
    if price_a <= 0 or price_b <= 0:
        return False
    return (price_b / price_a) <= max_ratio


def _price_ratio(price_a: float, price_b: float) -> float:
    if price_a <= 0 or price_b <= 0:
        return float("inf")
    return float(price_b / price_a)


def _adaptive_price_multiplier(ratio: float) -> float:
    if ratio <= CHEAP_RATIO_MAX:
        return 1.0
    if ratio >= ADAPTIVE_PRICE_MAX:
        return 0.75
    span = max(ADAPTIVE_PRICE_MAX - CHEAP_RATIO_MAX, 1e-6)
    progress = (ratio - CHEAP_RATIO_MAX) / span
    return float(_clip(1.0 - progress * 0.25, 0.75, 1.0))


def _passes_adaptive_price_gate(
    price_a: float,
    price_b: float,
    purchase_score: float,
    purchase_p95: float,
    positive_preference_strength: float,
    utility_penalty_multiplier: float,
    base_new_raw: float,
) -> tuple[bool, bool, float]:
    ratio = _price_ratio(price_a, price_b)
    if ratio <= CHEAP_RATIO_MAX:
        return True, False, CHEAP_RATIO_MAX
    if ratio > ADAPTIVE_PRICE_MAX:
        return False, False, CHEAP_RATIO_MAX
    strong_purchase = purchase_score >= purchase_p95
    strong_preference = positive_preference_strength >= 0.5
    utility_ok = utility_penalty_multiplier >= 0.9
    score_ok = base_new_raw >= ADAPTIVE_PRICE_MIN_NEW_RAW
    allowed = (strong_purchase or strong_preference) and utility_ok and score_ok
    return allowed, allowed, ADAPTIVE_PRICE_MAX if allowed else CHEAP_RATIO_MAX


def _derive_parent_group(product_family: str, category: str) -> str:
    fam = str(product_family).strip()
    if fam:
        return PARENT_GROUP_BY_FAMILY.get(fam, fam)
    cat = str(category).strip().lower()
    return CATEGORY_PARENT_FALLBACK.get(cat, cat or "other")


def _parent_multiplier(parent_a: str, parent_b: str) -> float:
    return SAME_PARENT_PENALTY if parent_a and parent_b and parent_a == parent_b else CROSS_PARENT_BONUS


def _build_tag_idf_lookup(tag_lookup: dict[int, set[str]], product_ids: list[int]) -> dict[str, float]:
    doc_count: dict[str, int] = {}
    n_docs = max(len(product_ids), 1)
    for pid in product_ids:
        tags = {t for t in tag_lookup.get(pid, {"other"}) if t and t != NON_FOOD_TAG}
        for tag in tags:
            doc_count[tag] = doc_count.get(tag, 0) + 1
    idf: dict[str, float] = {}
    for tag, df in doc_count.items():
        idf[tag] = float(np.log((1.0 + n_docs) / (1.0 + df)) + 1.0)
    return idf


def _tag_idf_overlap_score(tags_a: set[str], tags_b: set[str], tag_idf: dict[str, float]) -> tuple[float, set[str]]:
    shared = {t for t in (tags_a & tags_b) if t and t != NON_FOOD_TAG}
    if not shared:
        return 0.0, set()
    specific_shared = {t for t in shared if t not in GENERIC_TAGS}
    scored_tags = specific_shared if specific_shared else shared
    weights = [float(tag_idf.get(tag, 1.0)) for tag in scored_tags]
    if not weights:
        return 0.0, specific_shared
    mean_idf = float(np.mean(weights))
    overlap_score = _clip(mean_idf * 20.0 + min(len(scored_tags), 8) * 8.0, 0.0, 100.0)
    return float(overlap_score), specific_shared


def _is_ready_to_consume_anchor(category_a: str, tags_a: set[str], name_a: str) -> bool:
    cat = str(category_a).strip().lower()
    if cat in READY_TO_CONSUME_CATEGORIES:
        return True
    norm_name = _normalise_name(name_a)
    snack_tokens = {"dates", "yogurt", "juice", "drink", "chips", "chocolate"}
    if any(token in norm_name.split() for token in snack_tokens):
        return True
    return "smoothies" in tags_a or "desserts" in tags_a


def _is_utility_complement(category_b: str, tags_b: set[str], name_b: str) -> bool:
    cat = str(category_b).strip().lower()
    if cat in UTILITY_CATEGORIES:
        return True
    if any(t in UTILITY_TOKENS for t in tags_b):
        return True
    name_tokens = set(_normalise_name(name_b).split())
    return bool(name_tokens & UTILITY_TOKENS)


def _utility_penalty_multiplier(
    category_a: str,
    category_b: str,
    tags_a: set[str],
    tags_b: set[str],
    name_a: str,
    name_b: str,
    purchase_score: float,
    purchase_p90: float,
) -> tuple[float, bool]:
    ready_anchor = _is_ready_to_consume_anchor(category_a, tags_a, name_a)
    utility_comp = _is_utility_complement(category_b, tags_b, name_b)
    if not (ready_anchor and utility_comp):
        return 1.0, False
    denom = max(purchase_p90, 1e-6)
    strength = _clip(1.0 - (purchase_score / denom), 0.0, 1.0)
    penalty = UTILITY_PENALTY_MAX - strength * (UTILITY_PENALTY_MAX - UTILITY_PENALTY_MIN)
    return float(_clip(penalty, UTILITY_PENALTY_MIN, UTILITY_PENALTY_MAX)), True


def _load_feedback_preferences(path: Path) -> dict[str, list[dict[str, object]]]:
    empty = {"positive_patterns": [], "negative_patterns": []}
    if not path.exists():
        return empty
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return empty
    if not isinstance(payload, dict):
        return empty

    def _normalise_pattern(raw: object) -> dict[str, object] | None:
        if not isinstance(raw, dict):
            return None
        anchor_terms = [_normalise_name(x) for x in raw.get("anchor_terms", []) if _normalise_name(x)]
        complement_terms = [_normalise_name(x) for x in raw.get("complement_terms", []) if _normalise_name(x)]
        if not anchor_terms or not complement_terms:
            return None
        weight = _clip(float(raw.get("weight", 1.0)), 0.1, 2.0)
        return {"anchor_terms": anchor_terms, "complement_terms": complement_terms, "weight": weight}

    out = {"positive_patterns": [], "negative_patterns": []}
    for key in ("positive_patterns", "negative_patterns"):
        raw_patterns = payload.get(key, [])
        if not isinstance(raw_patterns, list):
            continue
        for p in raw_patterns:
            norm = _normalise_pattern(p)
            if norm is not None:
                out[key].append(norm)
    return out


def _pattern_match_strength(name_a: str, name_b: str, pattern: dict[str, object]) -> float:
    norm_a = _normalise_name(name_a)
    norm_b = _normalise_name(name_b)
    anchor_terms = pattern.get("anchor_terms", [])
    complement_terms = pattern.get("complement_terms", [])
    if not isinstance(anchor_terms, list) or not isinstance(complement_terms, list):
        return 0.0
    anchor_hits = sum(1 for t in anchor_terms if t in norm_a)
    complement_hits = sum(1 for t in complement_terms if t in norm_b)
    if anchor_hits == 0 or complement_hits == 0:
        return 0.0
    a_score = anchor_hits / max(len(anchor_terms), 1)
    b_score = complement_hits / max(len(complement_terms), 1)
    weight = float(pattern.get("weight", 1.0))
    return float(_clip((a_score * 0.5 + b_score * 0.5) * weight, 0.0, 1.0))


def _preference_bias_scores(
    name_a: str,
    name_b: str,
    positive_patterns: list[dict[str, object]],
    negative_patterns: list[dict[str, object]],
) -> tuple[float, float, float, float]:
    positive_strength = max((_pattern_match_strength(name_a, name_b, p) for p in positive_patterns), default=0.0)
    negative_strength = max((_pattern_match_strength(name_a, name_b, p) for p in negative_patterns), default=0.0)

    if negative_strength > 0:
        multiplier = PREFERENCE_NEGATIVE_MAX_MULT - negative_strength * (
            PREFERENCE_NEGATIVE_MAX_MULT - PREFERENCE_NEGATIVE_MIN_MULT
        )
    elif positive_strength > 0:
        multiplier = PREFERENCE_POSITIVE_MIN_MULT + positive_strength * (
            PREFERENCE_POSITIVE_MAX_MULT - PREFERENCE_POSITIVE_MIN_MULT
        )
    else:
        multiplier = 1.0

    preference_bias_score = _clip(positive_strength * 10.0 - negative_strength * 15.0, -15.0, 10.0)
    preference_match_strength = max(positive_strength, negative_strength)
    return (
        float(preference_bias_score),
        float(preference_match_strength),
        float(multiplier),
        float(positive_strength),
    )


def _load_pair_penalty_rules(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    rows = payload.get("rules", []) if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return []
    out: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip() or "unnamed_rule"
        anchor_terms = [_normalise_name(x) for x in row.get("anchor_terms", []) if _normalise_name(x)]
        complement_terms = [_normalise_name(x) for x in row.get("complement_terms", []) if _normalise_name(x)]
        if not anchor_terms or not complement_terms:
            continue
        multiplier = _clip(float(row.get("multiplier", 0.88)), 0.55, 0.98)
        reason = str(row.get("reason", "")).strip() or "suspicious_pair_pattern"
        out.append(
            {
                "name": name,
                "anchor_terms": anchor_terms,
                "complement_terms": complement_terms,
                "multiplier": float(multiplier),
                "reason": reason,
            }
        )
    return out


def _match_rule_strength(name_a: str, name_b: str, rule: dict[str, object]) -> float:
    norm_a = _normalise_name(name_a)
    norm_b = _normalise_name(name_b)
    anchor_terms = rule.get("anchor_terms", [])
    complement_terms = rule.get("complement_terms", [])
    if not isinstance(anchor_terms, list) or not isinstance(complement_terms, list):
        return 0.0
    anchor_hits = sum(1 for term in anchor_terms if term in norm_a)
    complement_hits = sum(1 for term in complement_terms if term in norm_b)
    if anchor_hits <= 0 or complement_hits <= 0:
        return 0.0
    a_score = anchor_hits / max(len(anchor_terms), 1)
    b_score = complement_hits / max(len(complement_terms), 1)
    return float(_clip((a_score + b_score) * 0.5, 0.0, 1.0))


def _staple_flags(name_a: str, name_b: str) -> str:
    staples = {"rice", "oil", "salt", "water", "sugar"}
    tokens = set(_normalise_name(name_a).split()) | set(_normalise_name(name_b).split())
    return "|".join(sorted(tokens & staples))


def _is_savory_base(name: str, family: str) -> bool:
    norm_name = _normalise_name(name)
    norm_family = _normalise_name(family)
    if any(hint in norm_name for hint in SAVORY_BASE_HINTS):
        return True
    family_hints = {"condiment", "spice", "seasoning", "oil", "sauce", "paste", "bouillon", "stock"}
    return any(hint in norm_family for hint in family_hints)


def _is_processed_sweet(name: str, family: str) -> bool:
    norm_name = _normalise_name(name)
    norm_family = _normalise_name(family)
    if any(hint in norm_name for hint in SWEET_PROCESSED_HINTS):
        return True
    family_hints = {"dessert", "confection", "candy", "chocolate", "biscuit", "cookie", "jelly", "jam", "sweet"}
    return any(hint in norm_family for hint in family_hints)


def _recipe_only_gate(
    copurchase_score: float,
    known_prior_flag: int,
    recipe_score_norm: float,
    ingredient_a: str,
    ingredient_b: str,
) -> tuple[bool, int, list[str], str, float]:
    tokens = [t for t in [str(ingredient_a).strip().lower(), str(ingredient_b).strip().lower()] if t]
    only_staples_overlap = int(bool(tokens) and all(token in RECIPE_STAPLES for token in tokens))
    effective_recipe_score = float(recipe_score_norm)
    if only_staples_overlap:
        effective_recipe_score = 0.0
    if float(copurchase_score) == 0.0 and int(known_prior_flag) == 0:
        if effective_recipe_score < RECIPE_ONLY_MIN_SCORE_NORM:
            return False, only_staples_overlap, tokens, "blocked_recipe_only_weak", effective_recipe_score
        return True, only_staples_overlap, tokens, "recipe_non_staple", effective_recipe_score
    if float(copurchase_score) > 0.0:
        return True, only_staples_overlap, tokens, "copurchase", effective_recipe_score
    return True, only_staples_overlap, tokens, "known_prior", effective_recipe_score


def _sweet_savory_block_status(
    name_a: str,
    family_a: str,
    name_b: str,
    family_b: str,
    pair_count: int,
    copurchase_score: float,
) -> tuple[bool, bool]:
    blocked = (
        (_is_savory_base(name_a, family_a) and _is_processed_sweet(name_b, family_b))
        or (_is_savory_base(name_b, family_b) and _is_processed_sweet(name_a, family_a))
    )
    override = (
        int(pair_count) >= SWEET_SAVORY_OVERRIDE_MIN_PAIR_COUNT
        or float(copurchase_score) >= SWEET_SAVORY_OVERRIDE_MIN_CP_SCORE
    )
    return bool(blocked), bool(override)


def _pair_penalty(
    name_a: str,
    name_b: str,
    rules: list[dict[str, object]],
) -> tuple[float, str, str, float, str]:
    best_multiplier = 1.0
    best_rule = ""
    best_reason = ""
    best_strength = 0.0
    for rule in rules:
        strength = _match_rule_strength(name_a, name_b, rule)
        if strength <= 0:
            continue
        base_multiplier = float(rule.get("multiplier", 0.9))
        # Blend by strength so weak partial matches get lighter penalties.
        multiplier = 1.0 - (1.0 - base_multiplier) * strength
        if multiplier < best_multiplier:
            best_multiplier = multiplier
            best_rule = str(rule.get("name", "")).strip()
            best_reason = str(rule.get("reason", "")).strip()
            best_strength = strength
    flags = _staple_flags(name_a, name_b)
    return float(best_multiplier), best_rule, best_reason, float(best_strength), flags


def _load_theme_tokens(config_path: Path) -> set[str]:
    if not config_path.exists():
        return set()
    try:
        with config_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return set()
    raw_tokens = payload.get("tokens", []) if isinstance(payload, dict) else []
    return {_normalise_name(str(token)) for token in raw_tokens if _normalise_name(str(token))}


def _extract_themes(name_a: str, name_b: str, tokens: set[str]) -> set[str]:
    if not tokens:
        return set()
    joined = f" {_normalise_name(name_a)} {_normalise_name(name_b)} "
    return {token for token in tokens if f" {token} " in joined}


def _load_models():
    try:
        out = _output_dir()
        with (out / "free_item_model.pkl").open("rb") as f:
            clf = pickle.load(f)
        with (out / "discount_model.pkl").open("rb") as f:
            reg = pickle.load(f)
        with (out / "preprocessor.pkl").open("rb") as f:
            preprocessor = pickle.load(f)
        return clf, reg, preprocessor
    except (FileNotFoundError, OSError):
        return None, None, None


def _score_with_model(pairs_df: pd.DataFrame, chunk_size: int = 200000) -> pd.DataFrame:
    clf, reg, preprocessor = _load_models()
    if clf is None or reg is None or preprocessor is None:
        pairs_df["model_discount_pred"] = 0.0
        pairs_df["model_score"] = 0.0
        return pairs_df
    print("  Applying ML model for bundle scoring...")
    n = len(pairs_df)
    model_discount_pred = np.zeros(n, dtype=np.float32)
    model_score = np.zeros(n, dtype=np.float32)
    free_item_pred = np.zeros(n, dtype=np.int8)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = pairs_df.iloc[start:end].copy()
        for col in CATEGORICAL_FEATURES:
            chunk[col] = chunk[col].fillna("other").astype(str)
        for col in NUMERIC_FEATURES:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0)
        X = preprocessor.transform(chunk[ALL_FEATURES])
        pred_free = clf.predict(X)
        discount_pred = reg.predict(X)
        if discount_pred.ndim == 1:
            discount_pred = np.column_stack([discount_pred, discount_pred])
        disc_mean = ((np.clip(discount_pred[:, 0], 0, 30) + np.clip(discount_pred[:, 1], 0, 30)) / 2.0).astype(np.float32)
        free_item_pred[start:end] = pred_free.astype(np.int8)
        model_discount_pred[start:end] = disc_mean
        model_score[start:end] = (disc_mean * 0.7 + (1 - pred_free) * 15).astype(np.float32)
    pairs_df["free_item_pred"] = free_item_pred
    pairs_df["model_discount_pred"] = np.round(model_discount_pred, 2)
    pairs_df["model_score"] = np.round(model_score, 2)
    print(f"  Model scoring complete: avg predicted discount = {pairs_df['model_discount_pred'].mean():.1f}%")
    return pairs_df


def _load_similarity_for_top_products(base: Path, product_ids: list[int], embeddings: np.ndarray, emb_map: dict[int, int]) -> np.ndarray:
    sim_path = base / "top_product_similarity.npz"
    ids_path = base / "top_product_ids.json"
    if sim_path.exists() and ids_path.exists():
        with ids_path.open("r", encoding="utf-8") as f:
            stored_ids = [int(x) for x in json.load(f)]
        if stored_ids[: len(product_ids)] == product_ids:
            sim_sparse = sparse.load_npz(sim_path)
            return sim_sparse[: len(product_ids), : len(product_ids)].toarray()
    emb_idx = [emb_map.get(pid, -1) for pid in product_ids]
    emb_mat = np.vstack([embeddings[idx] if idx >= 0 else np.zeros(embeddings.shape[1], dtype=np.float32) for idx in emb_idx])
    return np.clip(np.matmul(emb_mat, emb_mat.T), -1.0, 1.0)


def _feedback_conflict_count(df: pd.DataFrame, negative_patterns: list[dict[str, object]], top_n: int = 20) -> int:
    if df.empty or not negative_patterns:
        return 0
    sample = df.head(top_n)
    count = 0
    for _, row in sample.iterrows():
        name_a = str(row.get("product_a_name", ""))
        name_b = str(row.get("product_b_name", ""))
        conflict = max((_pattern_match_strength(name_a, name_b, p) for p in negative_patterns), default=0.0)
        if conflict > 0:
            count += 1
    return count


def _candidate_metrics(df: pd.DataFrame, negative_patterns: list[dict[str, object]] | None = None) -> dict[str, float]:
    if df.empty:
        return {
            "same_family_count": 0.0,
            "same_parent_rate": 0.0,
            "non_food_count": 0.0,
            "avg_price_ratio_b_to_a": 0.0,
            "ratio_over_0_50_share": 0.0,
            "utility_pair_count_top10": 0.0,
            "feedback_conflict_count_top20": 0.0,
            "top10_diversity": 0.0,
        }
    same_family = int((df["product_family_a"].astype(str).ne("") & df["product_family_b"].astype(str).ne("") & df["product_family_a"].astype(str).eq(df["product_family_b"].astype(str))).sum())
    same_parent_rate = float((df["parent_group_a"].astype(str).eq(df["parent_group_b"].astype(str)) & df["parent_group_a"].astype(str).ne("")).mean())
    non_food = int((df["is_non_food_a"].astype(bool) | df["is_non_food_b"].astype(bool)).sum())
    ratios = pd.to_numeric(df["price_ratio_b_to_a"], errors="coerce").fillna(0.0)
    avg_ratio = float(ratios.mean())
    ratio_over = float((ratios > CHEAP_RATIO_MAX).mean())
    util_col = pd.to_numeric(df.get("utility_penalty_multiplier", pd.Series([1.0] * len(df))), errors="coerce").fillna(1.0)
    utility_top10 = int((util_col.head(10) < 1.0).sum())
    feedback_conflicts = float(_feedback_conflict_count(df, negative_patterns or [], top_n=20))
    top10_ids = set(df.head(10)["product_a"].astype(int)).union(set(df.head(10)["product_b"].astype(int)))
    return {
        "same_family_count": float(same_family),
        "same_parent_rate": same_parent_rate,
        "non_food_count": float(non_food),
        "avg_price_ratio_b_to_a": avg_ratio,
        "ratio_over_0_50_share": ratio_over,
        "utility_pair_count_top10": float(utility_top10),
        "feedback_conflict_count_top20": feedback_conflicts,
        "top10_diversity": float(len(top10_ids)),
    }


def _retain_anchor_aware_candidates(
    df: pd.DataFrame,
    score_col: str,
    total_cap: int = RETAINED_CANDIDATE_CAP,
    per_anchor_cap: int = ANCHOR_RETENTION_PASS_A,
) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty:
        return df.copy(), {
            "raw_rows": 0,
            "retained_rows": 0,
            "unique_anchors_raw": 0,
            "unique_anchors_retained": 0,
            "pass_a_selected_rows": 0,
            "anchors_with_at_least_2": 0,
            "anchors_with_at_least_4": 0,
        }

    working = df.copy()
    working["_retention_input_order"] = np.arange(len(working), dtype=np.int64)
    sort_cols = [score_col]
    ascending = [False]
    for col in (
        "product_a",
        "product_b",
        "anchor_score",
        "complement_score",
        "specific_shared_count",
        "product_a_name",
        "product_b_name",
        "specific_shared_tags",
        "shared_categories",
        "_retention_input_order",
    ):
        if col not in working.columns:
            continue
        sort_cols.append(col)
        ascending.append(col not in {"anchor_score", "complement_score", "specific_shared_count"})

    working = working.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)
    anchor_rank = working.groupby("product_a").cumcount()
    pass_a_mask = anchor_rank < int(per_anchor_cap)
    pass_a_indices = working.index[pass_a_mask].tolist()
    pass_a_selected_rows = min(len(pass_a_indices), int(total_cap))

    if pass_a_selected_rows >= int(total_cap):
        selected_indices = pass_a_indices[: int(total_cap)]
    else:
        remaining_indices = working.index[~pass_a_mask].tolist()[: max(0, int(total_cap) - pass_a_selected_rows)]
        selected_indices = pass_a_indices + remaining_indices

    retained = working.loc[selected_indices].copy().reset_index(drop=True)
    retained.drop(columns=["_retention_input_order"], inplace=True, errors="ignore")
    retained_counts = retained["product_a"].value_counts()
    metrics = {
        "raw_rows": int(len(df)),
        "retained_rows": int(len(retained)),
        "unique_anchors_raw": int(df["product_a"].nunique()),
        "unique_anchors_retained": int(retained["product_a"].nunique()),
        "pass_a_selected_rows": int(pass_a_selected_rows),
        "anchors_with_at_least_2": int((retained_counts >= 2).sum()),
        "anchors_with_at_least_4": int((retained_counts >= 4).sum()),
    }
    return retained, metrics


def _shuffle_candidate_pool(df: pd.DataFrame, run_seed: int | None, max_rows: int = 1200) -> pd.DataFrame:
    if df.empty:
        return df
    pool = df.head(min(max_rows, len(df)))
    if run_seed is None:
        return pool.sample(frac=1)
    return pool.sample(frac=1, random_state=int(run_seed))


def select_bundles(
    data_dir: Path | None = None,
    top_n: int = TOP_N_BUNDLES,
    min_orders: int = MIN_ORDERS_PER_PRODUCT,
    top_products: int = TOP_PRODUCTS_FOR_PAIRING,
    run_seed: int | None = None,
) -> pd.DataFrame:
    base = data_dir or _data_dir()
    required = ["filtered_orders.pkl", "copurchase_scores.csv", "product_recipe_scores.csv", "product_categories.csv", "product_embeddings.npy", "embedding_mapping.json"]
    missing = [name for name in required if not (base / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required processed artifacts in {base}: {', '.join(missing)}. Run `python -m qeu_bundling.cli run full` first.")

    orders = pd.read_pickle(base / "filtered_orders.pkl")
    copurchase = pd.read_csv(base / "copurchase_scores.csv")
    recipe_df = pd.read_csv(base / "product_recipe_scores.csv")
    cat_df = pd.read_csv(base / "product_categories.csv")
    embeddings = np.load(base / "product_embeddings.npy")
    with (base / "embedding_mapping.json").open("r", encoding="utf-8") as f:
        emb_map = {int(k): int(v) for k, v in json.load(f).items()}

    eligible = (
        orders.groupby("product_id")["order_id"]
        .nunique()
        .reset_index()
        .rename(columns={"order_id": "order_count"})
    )
    eligible = eligible[eligible["order_count"] >= min_orders].sort_values("order_count", ascending=False).head(top_products).reset_index(drop=True)
    product_ids = eligible["product_id"].astype(int).tolist()
    eligible_set = set(product_ids)
    order_rank = {int(pid): int(i) for i, pid in enumerate(product_ids)}
    print(f"  Eligible products (>={min_orders} orders): {len(eligible_set):,} (top {top_products})")

    recipe_lookup = dict(zip(recipe_df["product_id"].astype(int), recipe_df["recipe_score"].astype(float)))
    ingredient_col = "final_matched_ingredient" if "final_matched_ingredient" in recipe_df.columns else "matched_ingredient"
    ingredient_lookup = {
        int(pid): str(ingredient).strip().lower()
        for pid, ingredient in zip(recipe_df["product_id"].astype(int), recipe_df.get(ingredient_col, pd.Series([""] * len(recipe_df))).fillna(""))
    }
    importance_lookup = dict(zip(recipe_df["product_id"].astype(int), recipe_df["saudi_importance"].astype(str)))
    cat_lookup = dict(zip(cat_df["product_id"].astype(int), cat_df["category"].astype(str)))
    count_col = cat_df["category_count"] if "category_count" in cat_df.columns else pd.Series([0.0] * len(cat_df), index=cat_df.index)
    freq_col = cat_df["frequency_score"] if "frequency_score" in cat_df.columns else pd.Series([0.0] * len(cat_df), index=cat_df.index)
    count_lookup = dict(zip(cat_df["product_id"].astype(int), pd.to_numeric(count_col, errors="coerce").fillna(0).astype(float)))
    frequency_lookup = dict(zip(cat_df["product_id"].astype(int), pd.to_numeric(freq_col, errors="coerce").fillna(0).astype(float)))
    family_lookup = dict(zip(cat_df["product_id"].astype(int), cat_df.get("product_family", pd.Series([""] * len(cat_df))).fillna("").astype(str)))
    tag_lookup = {int(pid): _parse_tags(tags, fallback=str(cat_lookup.get(int(pid), "other"))) for pid, tags in zip(cat_df["product_id"].astype(int), cat_df.get("category_tags", pd.Series(["other"] * len(cat_df))))}
    non_food_lookup = {int(pid): (NON_FOOD_TAG in tags) for pid, tags in tag_lookup.items()}

    name_lookup = orders.drop_duplicates("product_id").set_index("product_id")["product_name"].to_dict()
    price_lookup = orders.groupby("product_id")["unit_price"].median().to_dict()

    try:
        families = load_families(_reference_dir() / "product_families.json")
        if families:
            for pid in product_ids:
                if family_lookup.get(pid):
                    continue
                fam = assign_family(str(name_lookup.get(pid, "")), families)
                if fam:
                    family_lookup[pid] = fam
    except (OSError, TypeError, ValueError):
        pass

    campaign_lookup: dict[tuple[int, int], float] = {}
    campaign_product_raw: dict[int, float] = {}
    for fname in ("campaign_bundle_pairs.csv", "campaign_pairs.csv"):
        path = base / fname
        if not path.exists():
            continue
        dfc = pd.read_csv(path)
        if dfc.empty:
            continue
        occ = pd.to_numeric(dfc.get("occurrences", pd.Series([1.0] * len(dfc))), errors="coerce").fillna(1.0)
        max_occ = float(occ.max()) if float(occ.max()) > 0 else 1.0
        for a, b, o in zip(dfc["product_a"], dfc["product_b"], occ):
            aa, bb = int(a), int(b)
            score = float(o) / max_occ * 100.0
            campaign_lookup[(aa, bb)] = max(campaign_lookup.get((aa, bb), 0.0), score)
            campaign_lookup[(bb, aa)] = max(campaign_lookup.get((bb, aa), 0.0), score)
            campaign_product_raw[aa] = campaign_product_raw.get(aa, 0.0) + score
            campaign_product_raw[bb] = campaign_product_raw.get(bb, 0.0) + score
    campaign_product_score = {}
    if campaign_product_raw:
        mx = max(campaign_product_raw.values())
        denom = mx if mx > 0 else 1.0
        campaign_product_score = {pid: v / denom * 100.0 for pid, v in campaign_product_raw.items()}

    cp = copurchase.copy()
    cp["product_a"] = cp["product_a"].astype(int)
    cp["product_b"] = cp["product_b"].astype(int)
    cp["score"] = pd.to_numeric(cp["score"], errors="coerce").fillna(0.0)
    cp["pair_count"] = pd.to_numeric(cp.get("pair_count", pd.Series([0.0] * len(cp))), errors="coerce").fillna(0.0)
    cp = cp[cp["product_a"].isin(eligible_set) & cp["product_b"].isin(eligible_set)].copy()
    purchase_lookup: dict[tuple[int, int], float] = {}
    pair_count_lookup: dict[tuple[int, int], int] = {}
    purchase_neighbors: dict[int, list[tuple[int, float]]] = {}
    for a, b, s, pair_count in zip(cp["product_a"], cp["product_b"], cp["score"], cp["pair_count"]):
        aa, bb = int(a), int(b)
        key = (aa, bb) if aa <= bb else (bb, aa)
        purchase_lookup[key] = max(purchase_lookup.get(key, 0.0), float(s))
        pair_count_lookup[key] = max(pair_count_lookup.get(key, 0), int(pair_count))
        purchase_neighbors.setdefault(aa, []).append((bb, float(s)))
        purchase_neighbors.setdefault(bb, []).append((aa, float(s)))
    for pid in list(purchase_neighbors):
        purchase_neighbors[pid] = sorted(purchase_neighbors[pid], key=lambda x: x[1], reverse=True)

    tag_to_products: dict[str, list[int]] = {}
    for pid in product_ids:
        for tag in _specific_tags(tag_lookup.get(pid, {"other"})):
            tag_to_products.setdefault(tag, []).append(pid)

    sim_matrix = _load_similarity_for_top_products(base, product_ids, embeddings, emb_map)
    sim_idx = {pid: idx for idx, pid in enumerate(product_ids)}

    anchor_scores: list[tuple[int, float]] = []
    for pid in product_ids:
        if non_food_lookup.get(pid, False):
            continue
        score = (
            float(recipe_lookup.get(pid, 0.0)) * ANCHOR_RECIPE_WEIGHT
            + float(frequency_lookup.get(pid, 0.0)) * ANCHOR_FREQUENCY_WEIGHT
            + float(campaign_product_score.get(pid, 0.0)) * ANCHOR_CAMPAIGN_WEIGHT
        )
        anchor_scores.append((pid, score))
    anchor_scores.sort(key=lambda x: x[1], reverse=True)
    anchor_ids = [pid for pid, _ in anchor_scores[: min(MAX_ANCHORS, len(anchor_scores))]]

    feedback = _load_feedback_preferences(_reference_dir() / "bundle_feedback_preferences.json")
    positive_patterns = feedback.get("positive_patterns", [])
    negative_patterns = feedback.get("negative_patterns", [])
    pair_penalty_rules = _load_pair_penalty_rules(_reference_dir() / PAIR_PENALTY_RULES_FILE)
    tag_idf_lookup = _build_tag_idf_lookup(tag_lookup, product_ids)
    purchase_p90 = float(cp["score"].quantile(0.90)) if not cp.empty else 0.0
    purchase_p95 = float(cp["score"].quantile(0.95)) if not cp.empty else purchase_p90
    suspicious_rows: list[dict[str, object]] = []
    anchor_score_lookup = {
        pid: (
            float(recipe_lookup.get(pid, 0.0)) * ANCHOR_RECIPE_WEIGHT
            + float(frequency_lookup.get(pid, 0.0)) * ANCHOR_FREQUENCY_WEIGHT
            + float(campaign_product_score.get(pid, 0.0)) * ANCHOR_CAMPAIGN_WEIGHT
        )
        for pid in product_ids
    }

    def evaluate_oriented_pair(aid: int, bid: int, orientation_swapped: bool) -> dict[str, object] | None:
        name_a, name_b = str(name_lookup.get(aid, "")), str(name_lookup.get(bid, ""))
        if _is_same_product_variant(name_a, name_b):
            return None
        fam_a, fam_b = str(family_lookup.get(aid, "")), str(family_lookup.get(bid, ""))
        if fam_a and fam_b and fam_a == fam_b:
            return None
        if non_food_lookup.get(aid, False) or non_food_lookup.get(bid, False):
            return None

        a_price = float(price_lookup.get(aid, 0.0))
        b_price = float(price_lookup.get(bid, 0.0))
        if a_price <= 0 or b_price <= 0:
            return None

        i, j = sim_idx.get(aid), sim_idx.get(bid)
        emb = float(_clip(sim_matrix[i, j] * 100.0, 0.0, 100.0)) if i is not None and j is not None else 0.0
        if emb > MAX_EMBEDDING_SIMILARITY:
            return None

        pk = (aid, bid) if aid <= bid else (bid, aid)
        purchase_score = float(purchase_lookup.get(pk, 0.0))
        pair_count = int(pair_count_lookup.get(pk, 0))
        sweet_savory_blocked, sweet_savory_override = _sweet_savory_block_status(
            name_a=name_a,
            family_a=fam_a,
            name_b=name_b,
            family_b=fam_b,
            pair_count=pair_count,
            copurchase_score=purchase_score,
        )
        if sweet_savory_blocked and not sweet_savory_override:
            suspicious_rows.append(
                {
                    "anchor_product_id": int(aid),
                    "complement_product_id": int(bid),
                    "anchor_name": name_a,
                    "complement_name": name_b,
                    "rule_id": "sweet_savory_cooking_base_block",
                    "rule_type": "hard_block",
                    "reason": "savory_base_with_processed_sweet",
                    "override_condition_met": 0,
                    "pair_count": int(pair_count),
                    "copurchase_score": round(float(purchase_score), 6),
                }
            )
            return None
        if sweet_savory_blocked and sweet_savory_override:
            suspicious_rows.append(
                {
                    "anchor_product_id": int(aid),
                    "complement_product_id": int(bid),
                    "anchor_name": name_a,
                    "complement_name": name_b,
                    "rule_id": "sweet_savory_cooking_base_block",
                    "rule_type": "hard_block_override",
                    "reason": "override_by_strong_copurchase",
                    "override_condition_met": 1,
                    "pair_count": int(pair_count),
                    "copurchase_score": round(float(purchase_score), 6),
                }
            )

        tags_a = tag_lookup.get(aid, {"other"})
        tags_b = tag_lookup.get(bid, {"other"})
        shared = tags_a & tags_b
        if len(shared) < MIN_SHARED_CATEGORIES:
            return None

        tag_idf_score, specific_shared = _tag_idf_overlap_score(tags_a, tags_b, tag_idf_lookup)
        specific_count = len(specific_shared)
        shared_score = float(_clip(specific_count * 12.5, 0.0, 100.0))
        campaign_score = float(campaign_lookup.get((aid, bid), 0.0))
        cat_a, cat_b = str(cat_lookup.get(aid, "other")), str(cat_lookup.get(bid, "other"))
        cross_category = int(cat_a != cat_b)
        rec_a, rec_b = float(recipe_lookup.get(aid, 0.0)), float(recipe_lookup.get(bid, 0.0))
        rec_avg = (rec_a + rec_b) / 2.0
        recipe_score_norm = float(_clip(rec_avg / 100.0, 0.0, 1.0))
        ing_a = str(ingredient_lookup.get(aid, "")).strip().lower()
        ing_b = str(ingredient_lookup.get(bid, "")).strip().lower()
        known_prior_strength = max((_pattern_match_strength(name_a, name_b, p) for p in positive_patterns), default=0.0)
        known_prior_flag = int(known_prior_strength > 0.0)
        recipe_gate_pass, only_staples_overlap, recipe_overlap_tokens, gate_pass_reason, effective_recipe_score_norm = _recipe_only_gate(
            copurchase_score=purchase_score,
            known_prior_flag=known_prior_flag,
            recipe_score_norm=recipe_score_norm,
            ingredient_a=ing_a,
            ingredient_b=ing_b,
        )
        if not recipe_gate_pass:
            suspicious_rows.append(
                {
                    "anchor_product_id": int(aid),
                    "complement_product_id": int(bid),
                    "anchor_name": name_a,
                    "complement_name": name_b,
                    "rule_id": "recipe_only_weak_block",
                    "rule_type": "hard_block",
                    "reason": "blocked_recipe_only_weak",
                    "override_condition_met": 0,
                    "pair_count": int(pair_count),
                    "copurchase_score": round(float(purchase_score), 6),
                    "recipe_score_norm": round(float(effective_recipe_score_norm), 6),
                    "recipe_overlap_tokens": "|".join(recipe_overlap_tokens),
                    "only_staples_overlap": int(only_staples_overlap),
                }
            )
            return None
        freq_avg = (float(frequency_lookup.get(aid, 0.0)) + float(frequency_lookup.get(bid, 0.0))) / 2.0
        vers = float(
            _clip(
                ((float(count_lookup.get(aid, 0.0)) + float(count_lookup.get(bid, 0.0))) / 2.0) * 5.0,
                0.0,
                100.0,
            )
        )
        anchor_score = float(anchor_score_lookup.get(aid, 0.0))
        emb_band = _embedding_band_score(emb)
        complement = _clip(
            purchase_score * COMPLEMENT_PURCHASE_WEIGHT
            + tag_idf_score * COMPLEMENT_TAG_WEIGHT
            + emb_band * COMPLEMENT_EMBEDDING_WEIGHT
            + (5.0 if cross_category else 0.0),
            0.0,
            100.0,
        )
        has_ramadan = int("ramadan" in shared)
        has_saudi_dish = int(len(shared & SAUDI_DISH_TAGS) > 0)
        ramadan_boost = RAMADAN_BOOST_DISH if has_ramadan and has_saudi_dish else RAMADAN_BOOST_BASE if has_ramadan else 1.0
        veg_penalty = (
            VEGETABLE_PENALTY_BOTH
            if cat_a == "vegetables" and cat_b == "vegetables"
            else VEGETABLE_PENALTY_ONE
            if cat_a == "vegetables" or cat_b == "vegetables"
            else 1.0
        )
        lexical_only = purchase_score == 0 and campaign_score == 0 and specific_count <= 1 and tag_idf_score < 25.0
        utility_penalty, utility_pair_flag = _utility_penalty_multiplier(
            cat_a,
            cat_b,
            tags_a,
            tags_b,
            name_a,
            name_b,
            purchase_score,
            purchase_p90,
        )

        old_raw = (
            shared_score * SHARED_CATEGORY_WEIGHT
            + rec_avg * RECIPE_WEIGHT
            + purchase_score * COPURCHASE_WEIGHT
            + _legacy_embedding_score(emb) * EMBEDDING_WEIGHT
            + vers * VERSATILITY_WEIGHT
            + freq_avg * FREQUENCY_WEIGHT
            + campaign_score * CAMPAIGN_WEIGHT
        )
        new_raw = anchor_score * NEW_ANCHOR_WEIGHT + complement * NEW_COMPLEMENT_WEIGHT + campaign_score * NEW_CAMPAIGN_WEIGHT
        old_raw *= ramadan_boost * veg_penalty
        new_raw *= ramadan_boost * veg_penalty
        if lexical_only:
            old_raw *= LEXICAL_ONLY_PENALTY
            new_raw *= LEXICAL_ONLY_PENALTY
        parent_a = _derive_parent_group(fam_a, cat_a)
        parent_b = _derive_parent_group(fam_b, cat_b)
        parent_mult = _parent_multiplier(parent_a, parent_b)
        new_raw *= parent_mult
        new_raw *= utility_penalty
        old_raw *= utility_penalty

        preference_bias_score, preference_match_strength, pref_mult, positive_strength = _preference_bias_scores(
            name_a, name_b, positive_patterns, negative_patterns
        )
        new_raw *= pref_mult
        old_raw *= pref_mult

        pair_penalty_multiplier, pair_penalty_rule, pair_penalty_reason, pair_penalty_strength, staple_flags = _pair_penalty(
            name_a,
            name_b,
            pair_penalty_rules,
        )
        raw_before_penalty = float(new_raw)
        if pair_penalty_multiplier < 1.0:
            new_raw *= pair_penalty_multiplier
            old_raw *= pair_penalty_multiplier

        base_new_raw = float(new_raw)
        allowed_price, adaptive_price_used, effective_price_cap = _passes_adaptive_price_gate(
            a_price,
            b_price,
            purchase_score,
            purchase_p95,
            positive_strength,
            utility_penalty,
            base_new_raw,
        )
        if not allowed_price:
            return None
        ratio = _price_ratio(a_price, b_price)
        price_penalty = _adaptive_price_multiplier(ratio)
        old_raw *= price_penalty
        new_raw *= price_penalty
        deal_signal = float(_clip((CHEAP_RATIO_MAX - ratio) / max(CHEAP_RATIO_MAX, 1e-6), 0.0, 1.0))
        deal_boost_applied = float(_clip(deal_signal * MAX_DEAL_BOOST_SHARE, 0.0, MAX_DEAL_BOOST_SHARE))
        old_raw *= 1.0 + deal_boost_applied
        new_raw *= 1.0 + deal_boost_applied

        if pair_penalty_multiplier < 1.0:
            suspicious_rows.append(
                {
                    "anchor_product_id": int(aid),
                    "complement_product_id": int(bid),
                    "anchor_name": name_a,
                    "complement_name": name_b,
                    "rule_id": str(pair_penalty_rule or "pair_penalty_rule"),
                    "rule_type": "soft_penalty",
                    "reason": str(pair_penalty_reason or "rule_penalty"),
                    "override_condition_met": 0,
                    "pair_count": int(pair_count),
                    "copurchase_score": round(float(purchase_score), 6),
                    "anchor_family": fam_a,
                    "complement_family": fam_b,
                    "anchor_category": cat_a,
                    "complement_category": cat_b,
                    "recipe_score_anchor": round(float(rec_a), 4),
                    "recipe_score_complement": round(float(rec_b), 4),
                    "purchase_score": round(float(purchase_score), 4),
                    "recipe_compat_signal": round(float(rec_avg), 4),
                    "complement_gate_reason": pair_penalty_reason or "rule_penalty",
                    "dominant_ingredient_flags": staple_flags,
                    "penalty_rule": pair_penalty_rule,
                    "penalty_match_strength": round(float(pair_penalty_strength), 4),
                    "penalty_multiplier": round(float(pair_penalty_multiplier), 4),
                    "raw_score_before_penalty": round(float(raw_before_penalty), 4),
                    "raw_score_after_penalty": round(float(base_new_raw), 4),
                    "final_score_after_penalty": round(float(new_raw), 4),
                }
            )

        return {
            "product_a": int(aid),
            "product_b": int(bid),
            "product_a_name": name_a,
            "product_b_name": name_b,
            "product_a_price": round(a_price, 2),
            "product_b_price": round(b_price, 2),
            "price_ratio_b_to_a": round(ratio, 4),
            "recipe_score_a": rec_a,
            "recipe_score_b": rec_b,
            "recipe_score": rec_avg,
            "recipe_score_norm": round(float(effective_recipe_score_norm), 6),
            "recipe_compat_score": round(float(recipe_score_norm), 6),
            "purchase_score": purchase_score,
            "pair_count": int(pair_count),
            "known_prior_flag": int(known_prior_flag),
            "gate_pass_reason": gate_pass_reason,
            "recipe_overlap_tokens": "|".join(recipe_overlap_tokens),
            "only_staples_overlap": int(only_staples_overlap),
            "embedding_score": emb,
            "shared_categories_count": int(len(shared)),
            "shared_category_score": shared_score,
            "specific_shared_count": int(specific_count),
            "specific_shared_tags": "|".join(sorted(specific_shared)),
            "tag_idf_overlap_score": round(float(tag_idf_score), 4),
            "category_a": cat_a,
            "category_b": cat_b,
            "importance_a": str(importance_lookup.get(aid, "low")),
            "importance_b": str(importance_lookup.get(bid, "low")),
            "category_match": int(cat_a == cat_b),
            "is_campaign_pair": int(campaign_score > 0),
            "campaign_score": campaign_score,
            "frequency_score": freq_avg,
            "cross_category": cross_category,
            "product_family_a": fam_a,
            "product_family_b": fam_b,
            "parent_group_a": parent_a,
            "parent_group_b": parent_b,
            "is_non_food_a": False,
            "is_non_food_b": False,
            "shared_categories": "|".join(sorted(shared)),
            "anchor_score": round(float(anchor_score), 4),
            "complement_score": round(float(complement), 4),
            "has_ramadan_tag": int(has_ramadan),
            "has_saudi_dish_tag": int(has_saudi_dish),
            "ramadan_boost": ramadan_boost,
            "utility_penalty_multiplier": round(float(utility_penalty), 4),
            "utility_pair_flag": int(utility_pair_flag),
            "preference_bias_score": round(float(preference_bias_score), 4),
            "preference_match_strength": round(float(preference_match_strength), 4),
            "pair_penalty_rule": pair_penalty_rule,
            "pair_penalty_multiplier": round(float(pair_penalty_multiplier), 4),
            "deal_signal": round(float(deal_signal), 6),
            "deal_boost_applied": round(float(deal_boost_applied), 6),
            "weak_evidence_free_blocked": 0,
            "adaptive_price_used": int(adaptive_price_used),
            "effective_price_cap": round(float(effective_price_cap), 4),
            "orientation_swapped": int(orientation_swapped),
            "old_raw_score": round(float(old_raw), 4),
            "new_raw_score": round(float(new_raw), 4),
        }

    rows: list[dict[str, object]] = []
    seen_pairs: set[tuple[int, int]] = set()
    for anchor_id in anchor_ids:
        if float(price_lookup.get(anchor_id, 0.0)) <= 0:
            continue
        neighbors = [pid for pid, _ in purchase_neighbors.get(anchor_id, [])[:MAX_COPURCHASE_NEIGHBORS]]
        tag_pool: list[int] = []
        for tag in _specific_tags(tag_lookup.get(anchor_id, {"other"})):
            tag_pool.extend(tag_to_products.get(tag, []))
        tag_unique: list[int] = []
        tag_seen: set[int] = set()
        for pid in sorted(tag_pool, key=lambda x: order_rank.get(x, 10**9)):
            if pid == anchor_id or pid in tag_seen:
                continue
            tag_seen.add(pid)
            tag_unique.append(pid)
            if len(tag_unique) >= MAX_TAG_COMPLEMENTS:
                break
        candidates: list[int] = []
        c_seen: set[int] = set()
        for pid in neighbors + tag_unique:
            if pid in c_seen or pid == anchor_id or pid not in eligible_set:
                continue
            c_seen.add(pid)
            candidates.append(pid)
            if len(candidates) >= MAX_CANDIDATES_PER_ANCHOR:
                break

        for comp_id in candidates:
            pair_key = (min(anchor_id, comp_id), max(anchor_id, comp_id))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            best_row: dict[str, object] | None = None
            for aid, bid, swapped in ((anchor_id, comp_id, False), (comp_id, anchor_id, True)):
                oriented = evaluate_oriented_pair(aid, bid, swapped)
                if oriented is None:
                    continue
                if best_row is None:
                    best_row = oriented
                    continue
                if (float(oriented["new_raw_score"]), float(oriented["anchor_score"]), float(oriented["complement_score"])) > (
                    float(best_row["new_raw_score"]),
                    float(best_row["anchor_score"]),
                    float(best_row["complement_score"]),
                ):
                    best_row = oriented
            if best_row is not None:
                rows.append(best_row)

    pairs = pd.DataFrame(rows)
    print(f"  Candidate pairs generated (anchor->complement): {len(pairs):,}")
    if pairs.empty:
        raise ValueError("No candidate bundles after filters; relax constraints or check input artifacts.")

    if USE_MODEL_FOR_RANKING:
        pairs = _score_with_model(pairs)
        pairs["old_final_score"] = (pairs["old_raw_score"] * (1 - MODEL_SCORE_WEIGHT) + pairs["model_score"] * MODEL_SCORE_WEIGHT).round(4)
        pairs["new_final_score"] = (pairs["new_raw_score"] * (1 - MODEL_SCORE_WEIGHT) + pairs["model_score"] * MODEL_SCORE_WEIGHT).round(4)
    else:
        pairs["model_score"] = 0.0
        pairs["model_discount_pred"] = 0.0
        pairs["old_final_score"] = pairs["old_raw_score"].round(4)
        pairs["new_final_score"] = pairs["new_raw_score"].round(4)

    theme_tokens = _load_theme_tokens(_reference_dir() / "theme_tokens.json")
    ranked_cols = ["anchor_score", "complement_score", "specific_shared_count"]
    old_ranked = pairs.sort_values(["old_final_score"] + ranked_cols, ascending=False).reset_index(drop=True)
    new_ranked = pairs.sort_values(["new_final_score"] + ranked_cols, ascending=False).reset_index(drop=True)

    def pick_top(df: pd.DataFrame, score_col: str) -> tuple[pd.DataFrame, dict[str, int]]:
        out_rows = []
        product_counts: dict[int, int] = {}
        template_counts: dict[str, int] = {}
        theme_counts: dict[str, int] = {}

        df_sorted = df.sort_values(score_col, ascending=False).reset_index(drop=True)
        
        for idx, row in df_sorted.iterrows():
            pa, pb = int(row["product_a"]), int(row["product_b"])
            themes = _extract_themes(str(row.get("product_a_name", "")), str(row.get("product_b_name", "")), theme_tokens)
            fam_a = str(row.get("product_family_a", "")).strip().lower() or "other_a"
            fam_b = str(row.get("product_family_b", "")).strip().lower() or "other_b"
            shared_sig = str(row.get("specific_shared_tags", row.get("shared_categories", ""))).split("|")
            shared_key = "|".join(sorted(x for x in shared_sig if x)[:2]) or "none"
            template_key = f"{fam_a}::{fam_b}::{shared_key}"

            p_count = max(product_counts.get(pa, 0), product_counts.get(pb, 0))
            t_count = template_counts.get(template_key, 0)
            th_count = max([theme_counts.get(t, 0) for t in themes] + [0])

            penalty = 0.0
            if p_count >= 2:
                penalty += 0.02 * (p_count - 1)
            if t_count >= 1:
                penalty += 0.05 * t_count
            if th_count >= 2:
                penalty += 0.03 * (th_count - 1)

            # Cap the penalty so we don't zero out good bundles, but enough to push duplicates down
            penalty = min(0.30, penalty)
            
            row_dict = row.to_dict()
            row_dict["diversity_penalty"] = round(penalty, 4)
            row_dict["penalized_score"] = round(float(row[score_col]) * (1.0 - penalty), 4)
            out_rows.append(row_dict)

            product_counts[pa] = product_counts.get(pa, 0) + 1
            product_counts[pb] = product_counts.get(pb, 0) + 1
            template_counts[template_key] = template_counts.get(template_key, 0) + 1
            for t in themes:
                theme_counts[t] = theme_counts.get(t, 0) + 1

        out = pd.DataFrame(out_rows)
        out, retention_metrics = _retain_anchor_aware_candidates(out, score_col="penalized_score")
        out["final_score"] = out["penalized_score"]
        # Also map back to the expected score col so shadow compare works
        out[score_col] = out["penalized_score"]
        return out, retention_metrics

    old_top, old_retention = pick_top(old_ranked, "old_final_score")
    new_top, new_retention = pick_top(new_ranked, "new_final_score")

    selected_strategy = "new"
    candidates_dir = _candidates_dir()
    diagnostics_dir = _diagnostics_dir()
    if ENABLE_SHADOW_COMPARISON:
        old_shadow_path = candidates_dir / "person_candidates_shadow_old.csv"
        new_shadow_path = candidates_dir / "person_candidates_shadow_new.csv"
        old_top.to_csv(old_shadow_path, index=False, encoding="utf-8-sig")
        new_top.to_csv(new_shadow_path, index=False, encoding="utf-8-sig")
        # Compatibility mirror for older readers.
        old_top.to_csv(base / "person_candidates_shadow_old.csv", index=False, encoding="utf-8-sig")
        new_top.to_csv(base / "person_candidates_shadow_new.csv", index=False, encoding="utf-8-sig")
        old_m = {**_candidate_metrics(old_top), **old_retention}
        new_m = {**_candidate_metrics(new_top), **new_retention}
        old_pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(old_top["product_a"], old_top["product_b"])}
        new_pairs = {tuple(sorted((int(a), int(b)))) for a, b in zip(new_top["product_a"], new_top["product_b"])}
        overlap = float(len(old_pairs & new_pairs) / max(1, min(len(old_pairs), len(new_pairs))))
        criteria = {
            "same_family_zero": new_m["same_family_count"] == 0,
            "non_food_zero": new_m["non_food_count"] == 0,
            "avg_price_ratio_le_0_50": new_m["avg_price_ratio_b_to_a"] <= CHEAP_RATIO_MAX,
            "same_parent_reduced": new_m["same_parent_rate"] < old_m["same_parent_rate"],
            "top10_diversity_not_degraded": new_m["top10_diversity"] >= old_m["top10_diversity"],
        }
        cutover_passed = all(criteria.values())
        selected_strategy = "new" if cutover_passed else "old"
        with (_output_dir() / "bundle_strategy_comparison.json").open("w", encoding="utf-8") as f:
            json.dump({"old": old_m, "new": new_m, "overlap_candidates": overlap, "criteria": criteria, "cutover_passed": cutover_passed}, f, ensure_ascii=False, indent=2)
        print(f"  Shadow comparison complete. Cutover passed: {cutover_passed}")

    top = new_top.copy() if selected_strategy == "new" else old_top.copy()
    selected_retention = new_retention if selected_strategy == "new" else old_retention
    top["shadow_old_score"] = top["old_final_score"]
    top["shadow_new_score"] = top["new_final_score"]
    top["shadow_selected"] = selected_strategy
    top["final_score"] = np.where(selected_strategy == "new", top["new_final_score"], top["old_final_score"]).round(4)

    out_path = candidates_dir / PERSON_CANDIDATE_ARTIFACT
    top.to_csv(out_path, index=False, encoding="utf-8-sig")
    # Compatibility copies for older paths.
    top.to_csv(base / PERSON_CANDIDATE_ARTIFACT, index=False, encoding="utf-8-sig")
    legacy_path = base / "top_bundles.csv"
    top.to_csv(legacy_path, index=False, encoding="utf-8-sig")

    retention_payload = {
        "selected_strategy": selected_strategy,
        "old": old_retention,
        "new": new_retention,
        "selected": selected_retention,
    }
    retention_path = diagnostics_dir / RETENTION_METRICS_ARTIFACT
    retention_path.write_text(json.dumps(retention_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / RETENTION_METRICS_ARTIFACT).write_text(json.dumps(retention_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    scoring_breakdown_path = diagnostics_dir / PAIR_SCORING_BREAKDOWN_ARTIFACT
    keep_cols = [
        "product_a",
        "product_b",
        "product_a_name",
        "product_b_name",
        "product_family_a",
        "product_family_b",
        "category_a",
        "category_b",
        "purchase_score",
        "pair_count",
        "recipe_score_a",
        "recipe_score_b",
        "recipe_score_norm",
        "recipe_compat_score",
        "known_prior_flag",
        "gate_pass_reason",
        "recipe_overlap_tokens",
        "only_staples_overlap",
        "anchor_score",
        "complement_score",
        "new_raw_score",
        "new_final_score",
        "final_score",
        "price_ratio_b_to_a",
        "deal_signal",
        "deal_boost_applied",
        "weak_evidence_free_blocked",
        "pair_penalty_rule",
        "pair_penalty_multiplier",
    ]
    pairs_keep = [c for c in keep_cols if c in pairs.columns]
    pairs[pairs_keep].to_csv(scoring_breakdown_path, index=False, encoding="utf-8-sig")
    # Compatibility mirror.
    pairs[pairs_keep].to_csv(base / PAIR_SCORING_BREAKDOWN_ARTIFACT, index=False, encoding="utf-8-sig")

    suspicious_path = diagnostics_dir / SUSPICIOUS_AUDIT_ARTIFACT
    suspicious_df = pd.DataFrame(suspicious_rows)
    if suspicious_df.empty:
        suspicious_df = pd.DataFrame(
            columns=[
                "anchor_product_id",
                "complement_product_id",
                "anchor_name",
                "complement_name",
                "rule_id",
                "rule_type",
                "reason",
                "override_condition_met",
                "pair_count",
                "copurchase_score",
                "anchor_family",
                "complement_family",
                "anchor_category",
                "complement_category",
                "recipe_score_anchor",
                "recipe_score_complement",
                "recipe_score_norm",
                "recipe_overlap_tokens",
                "only_staples_overlap",
                "purchase_score",
                "recipe_compat_signal",
                "complement_gate_reason",
                "dominant_ingredient_flags",
                "penalty_rule",
                "penalty_match_strength",
                "penalty_multiplier",
                "raw_score_before_penalty",
                "raw_score_after_penalty",
                "final_score_after_penalty",
            ]
        )
    suspicious_df.to_csv(suspicious_path, index=False, encoding="utf-8-sig")
    suspicious_df.to_csv(base / SUSPICIOUS_AUDIT_ARTIFACT, index=False, encoding="utf-8-sig")
    print(f"  Selected {len(top)} internal candidate bundles ({selected_strategy} strategy) -> {out_path}")
    return top


def load_top_bundles(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or _data_dir()
    path = _candidates_dir() / PERSON_CANDIDATE_ARTIFACT
    if not path.exists():
        path = base / PERSON_CANDIDATE_ARTIFACT
    return pd.read_csv(path)


def run(run_seed: int | None = None):
    return select_bundles(run_seed=run_seed)


if __name__ == "__main__":
    print("Phase 6: Selecting top bundles ...")
    bundles = select_bundles()
    cols = ["product_a_name", "product_b_name", "anchor_score", "complement_score", "price_ratio_b_to_a", "shared_categories_count", "final_score"]
    print(f"\n  Top 15 bundles:\n{bundles[cols].head(15).to_string()}")
    print("Phase 6 complete.")
