"""Phase 5: Recipe scoring - map products to ingredients and compute scores."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.product_families import assign_family, load_families
from qeu_bundling.pipeline.phase_01_load_data import load_recipe_data, save_recipe_artifacts
from qeu_bundling.core.schema_validation import (
    NumericRangeRule,
    require_columns,
    require_not_empty,
    require_no_nulls,
    validate_numeric_ranges,
)

NON_FOOD_KEYWORDS = frozenset(
    {
        "shampoo",
        "conditioner",
        "hair mask",
        "hair balm",
        "hair cream",
        "hair serum",
        "lotion",
        "body wash",
        "face wash",
        "soap",
        "toothpaste",
        "deodorant",
        "cosmetic",
        "beauty",
        "makeup",
        "skincare",
    }
)
LOW_SIGNAL_INGREDIENTS = frozenset({"salt", "water", "sugar", "rice", "oil"})
STAPLE_INGREDIENTS = frozenset({"rice", "onions", "tomatoes", "salt", "sugar", "oil", "water"})
TOKEN_CLASS = "a-z0-9\u0600-\u06FF"
PROCESSED_SWEET_FAMILY_HINTS = frozenset(
    {
        "jelly",
        "jam",
        "candy",
        "confectionery",
        "chocolate",
        "biscuit",
        "cookie",
        "dessert",
        "sweet",
        "marshmallow",
        "cake",
        "wafer",
    }
)
PROCESSED_SWEET_NAME_HINTS = frozenset(
    {
        "jelly",
        "jam",
        "candy",
        "dessert",
        "biscuit",
        "cookie",
        "chocolate",
        "marshmallow",
        "wafer",
    }
)
FLAVOR_STOPWORDS = frozenset(
    {
        "lemon",
        "orange",
        "basil",
        "mint",
        "strawberry",
        "vanilla",
        "mango",
        "apple",
        "grape",
        "berry",
        "peach",
        "pineapple",
        "pomegranate",
        "watermelon",
        "cola",
    }
)
EXPLICIT_SWEET_INGREDIENT_ALLOWLIST = frozenset(
    {
        "honey",
        "dates",
        "cocoa",
        "chocolate",
        "milk",
        "sesame",
        "nuts",
        "pistachio",
        "almond",
        "walnut",
        "tahini",
    }
)


def _data_dir() -> Path:
    return get_paths().data_processed_dir


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", str(text).lower()).strip()


def _tokenize(norm_name: str) -> list[str]:
    return [t for t in str(norm_name).split() if t]


def _is_non_food_product(name: str) -> bool:
    norm = _normalise(name)
    return any(token in norm for token in NON_FOOD_KEYWORDS)


def _is_processed_sweet(name: str, family: str) -> bool:
    norm_name = _normalise(name)
    fam = _normalise(family)
    if any(hint in fam for hint in PROCESSED_SWEET_FAMILY_HINTS):
        return True
    return any(hint in norm_name for hint in PROCESSED_SWEET_NAME_HINTS)


def _filter_flavor_tokens(tokens: list[str], is_processed_sweet: bool) -> list[str]:
    if not is_processed_sweet:
        return tokens
    return [token for token in tokens if token not in FLAVOR_STOPWORDS]


def _build_family_lookup(products: pd.DataFrame) -> dict[int, str]:
    reference_path = get_paths().data_reference_dir / "product_families.json"
    if not reference_path.exists():
        return {}
    try:
        families = load_families(reference_path)
    except (OSError, ValueError, TypeError):
        return {}
    if not families:
        return {}
    lookup: dict[int, str] = {}
    for row in products.itertuples(index=False):
        pid = int(getattr(row, "product_id"))
        name = str(getattr(row, "product_name", ""))
        lookup[pid] = assign_family(name, families)
    return lookup


def _load_recipe_artifacts(base: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    alias_path = base / "ingredient_alias_index.csv"
    intensity_path = base / "ingredient_recipe_intensity.csv"

    if not alias_path.exists() or not intensity_path.exists():
        recipe_data = load_recipe_data()
        save_recipe_artifacts(recipe_data, data_dir=base)

    alias_df = pd.read_csv(alias_path, encoding="utf-8-sig")
    intensity_df = pd.read_csv(intensity_path, encoding="utf-8-sig")
    return alias_df, intensity_df


def _build_alias_map(
    alias_df: pd.DataFrame,
    intensity_df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """alias -> enriched ingredient metadata with intensity score."""
    intensity_lookup = {
        str(r["ingredient_key"]): {
            "intensity_score": float(r.get("intensity_score", 0.0)),
            "saudi_importance": str(r.get("saudi_importance", "low")).lower(),
            "recipe_count": int(float(r.get("recipe_count", 0)) if pd.notna(r.get("recipe_count", 0)) else 0),
            "qeu_relevance": float(r.get("qeu_relevance", 0.0)),
        }
        for _, r in intensity_df.iterrows()
    }

    alias_map: dict[str, dict[str, Any]] = {}
    for row in alias_df.itertuples(index=False):
        alias = _normalise(getattr(row, "alias", ""))
        ingredient_key = str(getattr(row, "ingredient_key", "")).strip()
        if not alias or not ingredient_key:
            continue
        stats = intensity_lookup.get(
            ingredient_key,
            {
                "intensity_score": float(getattr(row, "recipe_intensity_score", 0.0)),
                "saudi_importance": str(getattr(row, "saudi_importance", "low")).lower(),
                "recipe_count": int(float(getattr(row, "recipe_count", 0) or 0)),
                "qeu_relevance": float(getattr(row, "qeu_relevance", 0.0)),
            },
        )
        entry = {
            "ingredient_key": ingredient_key,
            "recipe_score": float(stats["intensity_score"]),
            "saudi_importance": str(stats["saudi_importance"]),
            "recipe_count": int(stats["recipe_count"]),
            "qeu_relevance": float(stats["qeu_relevance"]),
            "token_count": int(len(alias.split())),
            "alias_len": int(len(alias)),
        }
        current = alias_map.get(alias)
        if current is None or entry["recipe_score"] > current["recipe_score"]:
            alias_map[alias] = entry
    return alias_map


def _compile_alias_pattern(alias: str) -> re.Pattern[str]:
    parts = [re.escape(p) for p in alias.split() if p]
    phrase = r"\s+".join(parts)
    pattern = rf"(?<![{TOKEN_CLASS}]){phrase}(?![{TOKEN_CLASS}])"
    return re.compile(pattern)


def _candidate_score(alias: str, info: dict[str, Any], has_stronger_specific: bool) -> float:
    base = float(info.get("recipe_score", 0.0))
    token_bonus = min(int(info.get("token_count", 1)), 4) * 2.0
    alias_bonus = min(int(info.get("alias_len", len(alias))), 30) * 0.12
    score = base + token_bonus + alias_bonus

    ingredient = str(info.get("ingredient_key", "")).strip().lower()
    if ingredient in LOW_SIGNAL_INGREDIENTS and has_stronger_specific:
        score *= 0.72
    return score


def _coverage_penalty(ingredient_share: float, ingredient_key: str) -> float:
    share = max(0.0, float(ingredient_share))
    if share <= 0.015:
        penalty = 1.0
    else:
        penalty = 1.0 - min(0.35, (share - 0.015) * 8.0)

    ingredient_norm = str(ingredient_key).strip().lower()
    if ingredient_norm in STAPLE_INGREDIENTS:
        penalty *= 0.88
    return max(0.60, min(1.0, penalty))


def compute_recipe_scores(data_dir: Path | None = None) -> pd.DataFrame:
    """Score each unique product on recipe relevance (0-100)."""
    base = data_dir or _data_dir()
    orders = pd.read_pickle(base / "filtered_orders.pkl")
    require_columns(orders, ["product_id", "product_name"], artifact_name="phase_05.filtered_orders")
    require_not_empty(orders, artifact_name="phase_05.filtered_orders")

    products = (
        orders[["product_id", "product_name"]]
        .drop_duplicates(subset=["product_id"])
        .dropna(subset=["product_name"])
        .reset_index(drop=True)
    )
    products["product_id"] = products["product_id"].astype(int)

    alias_df, intensity_df = _load_recipe_artifacts(base)
    alias_map = _build_alias_map(alias_df, intensity_df)
    aliases = sorted(alias_map.keys(), key=lambda a: (len(a.split()), len(a)), reverse=True)
    alias_patterns = {alias: _compile_alias_pattern(alias) for alias in aliases}
    family_lookup = _build_family_lookup(products)

    raw_scores: list[float] = []
    importances: list[str] = []
    matched_ingredients: list[str] = []
    matched_aliases: list[str] = []
    matched_recipe_counts: list[int] = []
    matched_qeu_relevance: list[float] = []
    raw_tokens_output: list[str] = []
    filtered_tokens_output: list[str] = []
    final_matched_output: list[str] = []
    match_reason_output: list[str] = []

    for row in products.itertuples(index=False):
        pid = int(getattr(row, "product_id"))
        name = str(getattr(row, "product_name", ""))
        if _is_non_food_product(str(name)):
            raw_scores.append(0.0)
            importances.append("low")
            matched_ingredients.append("")
            matched_aliases.append("")
            matched_recipe_counts.append(0)
            matched_qeu_relevance.append(0.0)
            raw_tokens_output.append("")
            filtered_tokens_output.append("")
            final_matched_output.append("")
            match_reason_output.append("non_food_product")
            continue

        norm_name = _normalise(name)
        raw_tokens = _tokenize(norm_name)
        family = str(family_lookup.get(pid, ""))
        is_sweet = _is_processed_sweet(name, family)
        filtered_tokens = _filter_flavor_tokens(raw_tokens, is_sweet)
        filtered_norm = " ".join(filtered_tokens)
        if not norm_name:
            raw_scores.append(0.0)
            importances.append("low")
            matched_ingredients.append("")
            matched_aliases.append("")
            matched_recipe_counts.append(0)
            matched_qeu_relevance.append(0.0)
            raw_tokens_output.append("|".join(raw_tokens))
            filtered_tokens_output.append("|".join(filtered_tokens))
            final_matched_output.append("")
            match_reason_output.append("empty_name")
            continue

        matched_candidates: list[tuple[str, dict[str, Any]]] = []
        for alias in aliases:
            rx = alias_patterns[alias]
            if rx.search(filtered_norm):
                info = alias_map[alias]
                ingredient_key = str(info.get("ingredient_key", "")).strip().lower()
                if (
                    is_sweet
                    and ingredient_key in FLAVOR_STOPWORDS
                    and ingredient_key not in EXPLICIT_SWEET_INGREDIENT_ALLOWLIST
                ):
                    continue
                matched_candidates.append((alias, info))

        if not matched_candidates:
            raw_scores.append(0.0)
            importances.append("low")
            matched_ingredients.append("")
            matched_aliases.append("")
            matched_recipe_counts.append(0)
            matched_qeu_relevance.append(0.0)
            raw_tokens_output.append("|".join(raw_tokens))
            filtered_tokens_output.append("|".join(filtered_tokens))
            final_matched_output.append("")
            if is_sweet and raw_tokens != filtered_tokens:
                match_reason_output.append("processed_sweet_flavor_filtered")
            else:
                match_reason_output.append("no_alias_match")
            continue

        has_stronger_specific = any(
            str(info.get("ingredient_key", "")).strip().lower() not in LOW_SIGNAL_INGREDIENTS
            and float(info.get("recipe_score", 0.0)) > 12.0
            for _, info in matched_candidates
        )

        best_alias = ""
        best_info: dict[str, Any] = {}
        best_score = -1.0
        for alias, info in matched_candidates:
            score = _candidate_score(alias, info, has_stronger_specific=has_stronger_specific)
            if score > best_score:
                best_score = score
                best_alias = alias
                best_info = info

        raw_scores.append(min(100.0, float(best_info.get("recipe_score", 0.0))))
        importances.append(str(best_info.get("saudi_importance", "low")))
        matched_ingredients.append(str(best_info.get("ingredient_key", "")))
        matched_aliases.append(best_alias)
        matched_recipe_counts.append(int(best_info.get("recipe_count", 0)))
        matched_qeu_relevance.append(float(best_info.get("qeu_relevance", 0.0)))
        raw_tokens_output.append("|".join(raw_tokens))
        filtered_tokens_output.append("|".join(filtered_tokens))
        final_matched_output.append(str(best_info.get("ingredient_key", "")))
        match_reason_output.append("best_alias_match")

    products["recipe_score_raw"] = [round(float(v), 4) for v in raw_scores]
    products["saudi_importance"] = importances
    products["matched_ingredient"] = matched_ingredients
    products["matched_alias"] = matched_aliases
    products["matched_recipe_count"] = matched_recipe_counts
    products["matched_qeu_relevance"] = [round(float(v), 4) for v in matched_qeu_relevance]
    products["raw_tokens"] = raw_tokens_output
    products["filtered_tokens"] = filtered_tokens_output
    products["final_matched_ingredient"] = final_matched_output
    products["match_reason"] = match_reason_output

    matched_mask = products["matched_ingredient"].astype(str).str.len() > 0
    matched_total = int(matched_mask.sum())
    ingredient_counts = products.loc[matched_mask, "matched_ingredient"].value_counts().to_dict()

    penalties: list[float] = []
    final_scores: list[float] = []
    for _, row in products.iterrows():
        ingredient = str(row.get("matched_ingredient", ""))
        raw_score = float(row.get("recipe_score_raw", 0.0))
        if not ingredient or matched_total <= 0:
            penalties.append(1.0)
            final_scores.append(0.0 if raw_score <= 0 else raw_score)
            continue

        share = float(ingredient_counts.get(ingredient, 0)) / float(matched_total)
        penalty = _coverage_penalty(share, ingredient)
        penalties.append(round(penalty, 4))
        final_scores.append(round(min(100.0, raw_score * penalty), 4))

    products["coverage_penalty"] = penalties
    products["recipe_score"] = final_scores
    require_columns(
        products,
        ["product_id", "product_name", "recipe_score", "matched_ingredient"],
        artifact_name="phase_05.product_recipe_scores",
    )
    require_no_nulls(products, ["product_id", "product_name"], artifact_name="phase_05.product_recipe_scores")
    validate_numeric_ranges(
        products,
        rules={
            "recipe_score_raw": NumericRangeRule(minimum=0.0, maximum=100.0, allow_null=False),
            "coverage_penalty": NumericRangeRule(minimum=0.0, maximum=1.0, allow_null=False),
            "recipe_score": NumericRangeRule(minimum=0.0, maximum=100.0, allow_null=False),
        },
        artifact_name="phase_05.product_recipe_scores",
    )

    out_path = base / "product_recipe_scores.csv"
    products.to_csv(out_path, index=False, encoding="utf-8-sig")
    scored = products[products["recipe_score"] > 0]
    print(f"  Scored {len(scored):,}/{len(products):,} products with recipe_score > 0")
    print(f"  Saved -> {out_path}")
    return products


def load_recipe_scores(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or _data_dir()
    return pd.read_csv(base / "product_recipe_scores.csv")


def run():
    return compute_recipe_scores()


if __name__ == "__main__":
    print("Phase 5: Computing recipe scores ...")
    df = compute_recipe_scores()
    top = df.nlargest(
        15,
        "recipe_score",
    )[["product_id", "product_name", "recipe_score", "saudi_importance", "matched_ingredient", "matched_recipe_count"]]
    top_preview = top.to_string()
    safe_preview = top_preview.encode("ascii", errors="replace").decode("ascii")
    print(f"  Top 15 recipe-scored products:\n{safe_preview}")
    print("Phase 5 complete.")
