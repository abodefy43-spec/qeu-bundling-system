"""Phase 5: Recipe scoring — map products to ingredients and compute scores."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

SAUDI_IMPORTANCE_MULTIPLIER: dict[str, float] = {
    "critical": 1.0,
    "very_high": 0.9,
    "high": 0.8,
    "medium": 0.6,
    "low": 0.4,
}

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


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", str(text).lower()).strip()


def _is_non_food_product(name: str) -> bool:
    norm = _normalise(name)
    return any(token in norm for token in NON_FOOD_KEYWORDS)


def _build_alias_map(recipe_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """keyword/alias -> {ingredient_key, qeu_relevance, saudi_importance}."""
    alias_map: dict[str, dict[str, Any]] = {}
    for key, info in recipe_data.get("ingredients", {}).items():
        info = info or {}
        entry = {
            "ingredient_key": key,
            "qeu_relevance": float(info.get("qeu_relevance", 0)),
            "saudi_importance": str(info.get("saudi_importance", "medium")).lower(),
        }
        for token in _alias_tokens(key, info):
            if token and (token not in alias_map or entry["qeu_relevance"] > alias_map[token]["qeu_relevance"]):
                alias_map[token] = entry
    return alias_map


def _alias_tokens(key: str, info: dict[str, Any]) -> list[str]:
    tokens = [key.lower(), _normalise(key)]
    if info.get("en"):
        tokens.append(info["en"].lower())
        tokens.append(_normalise(info["en"]))
    if info.get("ar"):
        tokens.append(_normalise(info["ar"]))
    for pt in info.get("product_types", []):
        tokens.append(pt.lower().replace("_", " "))
    return tokens


def compute_recipe_scores(data_dir: Path | None = None) -> pd.DataFrame:
    """Score each unique product on recipe relevance (0-100).

    Score = qeu_relevance * saudi_importance_multiplier
    """
    base = data_dir or _data_dir()
    orders = pd.read_pickle(base / "filtered_orders.pkl")

    products = (
        orders[["product_id", "product_name"]]
        .drop_duplicates(subset=["product_id"])
        .dropna(subset=["product_name"])
        .reset_index(drop=True)
    )
    products["product_id"] = products["product_id"].astype(int)

    with (base / "recipe_data.json").open("r", encoding="utf-8") as f:
        recipe_data = json.load(f)
    alias_map = _build_alias_map(recipe_data)
    sorted_keywords = sorted(alias_map.keys(), key=len, reverse=True)

    scores: list[float] = []
    importances: list[str] = []
    matched_ingredients: list[str] = []

    for name in products["product_name"]:
        if _is_non_food_product(str(name)):
            scores.append(0.0)
            importances.append("low")
            matched_ingredients.append("")
            continue

        norm = _normalise(name)
        found = False
        for kw in sorted_keywords:
            if kw in norm:
                info = alias_map[kw]
                mult = SAUDI_IMPORTANCE_MULTIPLIER.get(info["saudi_importance"], 0.6)
                scores.append(min(100.0, info["qeu_relevance"] * mult))
                importances.append(info["saudi_importance"])
                matched_ingredients.append(info["ingredient_key"])
                found = True
                break
        if not found:
            scores.append(0.0)
            importances.append("low")
            matched_ingredients.append("")

    products["recipe_score"] = scores
    products["saudi_importance"] = importances
    products["matched_ingredient"] = matched_ingredients

    out_path = base / "product_recipe_scores.csv"
    products.to_csv(out_path, index=False, encoding="utf-8-sig")
    scored = products[products["recipe_score"] > 0]
    print(f"  Scored {len(scored):,}/{len(products):,} products with recipe_score > 0")
    print(f"  Saved -> {out_path}")
    return products


def load_recipe_scores(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or _data_dir()
    return pd.read_csv(base / "product_recipe_scores.csv")


if __name__ == "__main__":
    print("Phase 5: Computing recipe scores ...")
    df = compute_recipe_scores()
    top = df.nlargest(15, "recipe_score")[["product_id", "product_name", "recipe_score", "saudi_importance", "matched_ingredient"]]
    print(f"  Top 15 recipe-scored products:\n{top.to_string()}")
    print("Phase 5 complete.")
