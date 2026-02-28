"""Phase 4: Category assignment with multi-category tags for shared-category scoring."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from product_families import assign_family, load_families

IMPORTANCE_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
    "very_high": 3,
    "critical": 4,
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


def _clean_tag(value: str) -> str:
    return _normalise(value).replace(" ", "_")


def _is_non_food_product(name: str) -> bool:
    norm = _normalise(name)
    return any(token in norm for token in NON_FOOD_KEYWORDS)


def _upgrade_importance(current: str, incoming: str) -> str:
    if IMPORTANCE_RANK.get(incoming, 0) > IMPORTANCE_RANK.get(current, 0):
        return incoming
    return current


def _add_keyword(
    kw_map: dict[str, dict[str, Any]],
    keyword: str,
    category: str,
    subcategory: str,
    importance: str,
    tags: set[str],
) -> None:
    kw = _normalise(keyword)
    if not kw or len(kw) < 2:
        return

    entry = kw_map.setdefault(
        kw,
        {
            "category": "other",
            "subcategory": "",
            "importance": "low",
            "tags": set(),
        },
    )
    if entry["category"] == "other" and category:
        entry["category"] = category
    if not entry["subcategory"] and subcategory:
        entry["subcategory"] = subcategory
    entry["importance"] = _upgrade_importance(entry["importance"], importance)
    entry["tags"].update(t for t in tags if t and t != "other")
    if category:
        entry["tags"].add(_clean_tag(category))


def _build_keyword_map(recipe_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build keyword map with primary category + multi-category tags."""
    kw_map: dict[str, dict[str, Any]] = {}

    for key, info in recipe_data.get("ingredients", {}).items():
        info = info or {}
        category = str(info.get("category", "other")).lower()
        subcategory = str(info.get("subcategory", "")).lower()
        importance = str(info.get("saudi_importance", "medium")).lower()

        tags = {_clean_tag(category), _clean_tag(subcategory), "ingredient"}
        if importance in {"critical", "very_high"}:
            tags.add("saudi_staples")
        seasonal = info.get("seasonal_relevance", {}) or {}
        if any("ramadan" in str(k).lower() and float(v) >= 70 for k, v in seasonal.items()):
            tags.add("ramadan")
        for recipe_name in info.get("recipes", []):
            tags.add(_clean_tag(recipe_name))

        _add_keyword(kw_map, key, category, subcategory, importance, tags)
        _add_keyword(kw_map, str(info.get("en", "")), category, subcategory, importance, tags)
        _add_keyword(kw_map, str(info.get("ar", "")), category, subcategory, importance, tags)
        for pt in info.get("product_types", []):
            _add_keyword(kw_map, str(pt), category, subcategory, importance, tags)

    for cat_key, cat_info in recipe_data.get("categories", {}).items():
        cat_info = cat_info or {}
        category = str(cat_key).lower()
        importance = str(cat_info.get("ramadan_importance", "medium")).lower()
        base_tags = {_clean_tag(category), "qeu_category"}

        _add_keyword(kw_map, category, category, "", importance, base_tags)
        for sub in cat_info.get("subcategories", []):
            _add_keyword(
                kw_map,
                str(sub),
                category,
                str(sub).lower(),
                importance,
                base_tags | {_clean_tag(str(sub))},
            )
        for qc in cat_info.get("qeu_categories", []):
            _add_keyword(
                kw_map,
                str(qc),
                category,
                "",
                importance,
                base_tags | {_clean_tag(str(qc))},
            )

    for dish_key, dish in recipe_data.get("saudi_ramadan_dishes", {}).items():
        dish = dish or {}
        dish_tag = _clean_tag(str(dish_key))
        dish_tags = {"ramadan", "saudi", "saudi_specialties", dish_tag, "dish"}
        _add_keyword(kw_map, str(dish_key), "saudi_specialties", str(dish_key), "critical", dish_tags)
        _add_keyword(
            kw_map,
            str(dish.get("ar", "")),
            "saudi_specialties",
            str(dish_key),
            "critical",
            dish_tags,
        )
        for ingredient in dish.get("key_ingredients", []):
            _add_keyword(
                kw_map,
                str(ingredient),
                "saudi_specialties",
                str(dish_key),
                "critical",
                dish_tags,
            )

    return kw_map


def assign_categories(data_dir: Path | None = None) -> pd.DataFrame:
    """Assign primary category + multi-category tags to every unique product."""
    base = data_dir or _data_dir()
    orders = pd.read_pickle(base / "filtered_orders.pkl")

    products = (
        orders[["product_id", "product_name"]]
        .drop_duplicates(subset=["product_id"])
        .dropna(subset=["product_name"])
        .reset_index(drop=True)
    )
    products["product_id"] = products["product_id"].astype(int)

    order_counts = orders.groupby("product_id")["order_id"].nunique()
    p90 = float(np.percentile(order_counts.values, 90)) if len(order_counts) else 0.0
    p99 = float(np.percentile(order_counts.values, 99)) if len(order_counts) else 0.0
    min_count = float(order_counts.min()) if len(order_counts) else 0.0
    max_count = float(order_counts.max()) if len(order_counts) else 0.0

    with (base / "recipe_data.json").open("r", encoding="utf-8") as f:
        recipe_data = json.load(f)
    family_rules = load_families(base / "product_families.json")
    family_ids = {str(f.get("id", "")).strip() for f in family_rules if f.get("id")}
    kw_map = _build_keyword_map(recipe_data)
    sorted_keywords = sorted(kw_map.keys(), key=len, reverse=True)

    categories: list[str] = []
    subcategories: list[str] = []
    importances: list[str] = []
    category_tags: list[str] = []
    category_count: list[int] = []
    product_families: list[str] = []
    frequency_scores: list[float] = []

    for row in products.itertuples(index=False):
        name = str(row.product_name)
        pid = int(row.product_id)
        norm = _normalise(name)
        is_non_food = _is_non_food_product(name)

        matched_entries: list[dict[str, Any]] = []
        matched_tags: set[str] = set()
        if is_non_food:
            # Non-food products can contain food words ("rice water shampoo").
            # Keep them out of recipe/category intelligence to avoid false bundles.
            cat, sub, imp = "other", "", "low"
            matched_tags.update({"other", "non_food"})
        else:
            for kw in sorted_keywords:
                if kw in norm:
                    info = kw_map[kw]
                    matched_entries.append(info)
                    matched_tags.update(info["tags"])

            if matched_entries:
                primary = max(
                    matched_entries,
                    key=lambda x: (
                        IMPORTANCE_RANK.get(str(x.get("importance", "low")), 0),
                        len(str(x.get("category", ""))),
                    ),
                )
                cat = str(primary.get("category", "other"))
                sub = str(primary.get("subcategory", ""))
                imp = str(primary.get("importance", "low"))
            else:
                cat, sub, imp = "other", "", "low"
                matched_tags.add("other")

        count = int(order_counts.get(pid, 0))
        if count >= p90:
            matched_tags.add("frequent_purchase")
        if count >= p99:
            matched_tags.add("top_seller")
        if max_count > min_count:
            freq_score = ((count - min_count) / (max_count - min_count)) * 100.0
        else:
            freq_score = 0.0
        matched_tags.add(_clean_tag(cat))
        if sub:
            matched_tags.add(_clean_tag(sub))

        tags_sorted = sorted(t for t in matched_tags if t)
        family_id = assign_family(name, family_rules) if not is_non_food else ""
        if not family_id and not is_non_food:
            for tag in matched_tags:
                if tag in family_ids:
                    family_id = tag
                    break
        categories.append(cat)
        subcategories.append(sub)
        importances.append(imp)
        category_tags.append("|".join(tags_sorted))
        category_count.append(len(tags_sorted))
        product_families.append(family_id)
        frequency_scores.append(round(float(np.clip(freq_score, 0.0, 100.0)), 2))

    products["category"] = categories
    products["subcategory"] = subcategories
    products["importance_level"] = importances
    products["category_tags"] = category_tags
    products["category_count"] = category_count
    products["product_family"] = product_families
    products["frequency_score"] = frequency_scores

    out_path = base / "product_categories.csv"
    products.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  Assigned categories to {len(products):,} products -> {out_path}")
    print(f"  Category distribution:\n{products['category'].value_counts().to_string()}")
    return products


def load_product_categories(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or _data_dir()
    return pd.read_csv(base / "product_categories.csv")


if __name__ == "__main__":
    print("Phase 4: Assigning product categories ...")
    assign_categories()
    print("Phase 4 complete.")
