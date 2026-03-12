"""Phase 5: Recipe scoring - map products to ingredients and compute scores."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from qeu_bundling.config.paths import get_paths
from qeu_bundling.pipeline.phase_01_load_data import load_recipe_data, save_recipe_artifacts

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

LOW_SIGNAL_INGREDIENTS = frozenset({"salt", "water", "sugar"})
STRONG_FOOD_HINTS = frozenset(
    {
        "tuna",
        "fish",
        "sardine",
        "salmon",
        "chicken",
        "beef",
        "lamb",
        "yogurt",
        "cheese",
        "cream",
        "banana",
        "dates",
        "chocolate",
        "milk",
        "rice",
        "pasta",
    }
)


def _data_dir() -> Path:
    return get_paths().data_processed_dir


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", str(text).lower()).strip()


def _is_non_food_product(name: str) -> bool:
    norm = _normalise(name)
    return any(token in norm for token in NON_FOOD_KEYWORDS)


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
        }
        current = alias_map.get(alias)
        if current is None or entry["recipe_score"] > current["recipe_score"]:
            alias_map[alias] = entry
    return alias_map


def _boundary_hit(norm_name: str, alias: str) -> bool:
    return f" {alias} " in f" {norm_name} "


def _score_alias_match(norm_name: str, alias: str, info: dict[str, Any]) -> float:
    recipe_score = float(info.get("recipe_score", 0.0))
    qeu_relevance = float(info.get("qeu_relevance", 0.0))
    recipe_count = int(info.get("recipe_count", 0))
    tokens = alias.split()

    score = recipe_score * 0.7 + qeu_relevance * 0.2 + min(recipe_count * 0.15, 10.0)
    score += min(len(alias), 20) * 0.5
    if _boundary_hit(norm_name, alias):
        score += 15.0
    if len(tokens) > 1:
        score += min((len(tokens) - 1) * 4.0, 12.0)

    ingredient = str(info.get("ingredient_key", "")).lower().strip()
    if ingredient in LOW_SIGNAL_INGREDIENTS:
        words = set(norm_name.split())
        if any(h in words for h in STRONG_FOOD_HINTS if h not in LOW_SIGNAL_INGREDIENTS):
            score -= 30.0
    return score


def _best_alias_match(norm_name: str, alias_map: dict[str, dict[str, Any]], sorted_keywords: list[str]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score = float("-inf")
    for kw in sorted_keywords:
        if kw not in norm_name:
            continue
        info = alias_map[kw]
        score = _score_alias_match(norm_name, kw, info)
        if score > best_score:
            best_score = score
            best = info
    return best


def compute_recipe_scores(data_dir: Path | None = None) -> pd.DataFrame:
    """Score each unique product on recipe relevance (0-100)."""
    base = data_dir or _data_dir()
    orders = pd.read_pickle(base / "filtered_orders.pkl")

    products = (
        orders[["product_id", "product_name"]]
        .drop_duplicates(subset=["product_id"])
        .dropna(subset=["product_name"])
        .reset_index(drop=True)
    )
    products["product_id"] = products["product_id"].astype(int)

    alias_df, intensity_df = _load_recipe_artifacts(base)
    alias_map = _build_alias_map(alias_df, intensity_df)
    sorted_keywords = sorted(alias_map.keys(), key=len, reverse=True)

    scores: list[float] = []
    importances: list[str] = []
    matched_ingredients: list[str] = []
    matched_recipe_counts: list[int] = []
    matched_qeu_relevance: list[float] = []

    for name in products["product_name"]:
        if _is_non_food_product(str(name)):
            scores.append(0.0)
            importances.append("low")
            matched_ingredients.append("")
            matched_recipe_counts.append(0)
            matched_qeu_relevance.append(0.0)
            continue

        norm = _normalise(name)
        best = _best_alias_match(norm, alias_map, sorted_keywords)
        if best is None:
            scores.append(0.0)
            importances.append("low")
            matched_ingredients.append("")
            matched_recipe_counts.append(0)
            matched_qeu_relevance.append(0.0)
            continue
        scores.append(min(100.0, float(best["recipe_score"])))
        importances.append(str(best["saudi_importance"]))
        matched_ingredients.append(str(best["ingredient_key"]))
        matched_recipe_counts.append(int(best.get("recipe_count", 0)))
        matched_qeu_relevance.append(float(best.get("qeu_relevance", 0.0)))

    products["recipe_score"] = [round(float(v), 4) for v in scores]
    products["saudi_importance"] = importances
    products["matched_ingredient"] = matched_ingredients
    products["matched_recipe_count"] = matched_recipe_counts
    products["matched_qeu_relevance"] = [round(float(v), 4) for v in matched_qeu_relevance]

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
