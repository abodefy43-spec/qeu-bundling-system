"""Data loading and normalization for the compatible products engine."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from data.paths import get_project_paths

TOKEN_RE = re.compile(r"[a-z0-9\u0600-\u06ff]+")
TOKEN_ALIASES = {
    "biscuits": "biscuit",
    "cookies": "cookie",
    "dates": "date",
    "patties": "patty",
    "sauces": "sauce",
    "slices": "slice",
    "spices": "spice",
    "tomatoes": "tomato",
    "buns": "bun",
}
STOPWORDS = {
    "a",
    "al",
    "and",
    "box",
    "flavor",
    "flavoured",
    "for",
    "free",
    "g",
    "gm",
    "grams",
    "kg",
    "kilo",
    "l",
    "liter",
    "liters",
    "ml",
    "of",
    "pack",
    "pieces",
    "plus",
    "the",
    "with",
    "x",
}
GENERIC_TAGS = {
    "appetizers",
    "beverages",
    "dairy",
    "desserts",
    "dish",
    "frequent_purchase",
    "fruits",
    "grains",
    "ingredient",
    "other",
    "protein",
    "qeu_category",
    "ramadan",
    "saudi",
    "saudi_specialties",
    "saudi_staples",
    "snacks",
    "top_seller",
    "vegetables",
}
NONFOOD_TOKENS = {
    "battery",
    "cleaner",
    "detergent",
    "detergents",
    "diaper",
    "diapers",
    "dishwashing",
    "fork",
    "garbage",
    "plate",
    "plates",
    "shampoo",
    "soap",
    "spoon",
    "tissue",
    "toilet",
    "toothpaste",
    "wipe",
    "wipes",
}
GENERIC_NAME_TOKENS = {
    "authentic",
    "classic",
    "fresh",
    "large",
    "local",
    "mix",
    "mixed",
    "pack",
    "premium",
    "small",
    "whole",
}


@dataclass(frozen=True)
class ProductRecord:
    product_id: int
    product_name: str
    category: str
    subcategory: str
    product_family: str
    frequency_score: float
    recipe_score: float
    matched_ingredient: str
    category_tags: frozenset[str]
    name_tokens: frozenset[str]
    dominant_tokens: frozenset[str]


@dataclass(frozen=True)
class PairSignal:
    candidate_id: int
    pair_count: int
    score: float


@dataclass(frozen=True)
class PairPenaltyRule:
    name: str
    anchor_terms: frozenset[str]
    complement_terms: frozenset[str]
    multiplier: float
    reason: str


@dataclass(frozen=True)
class CompatibleProductsData:
    project_root: Path
    products: dict[int, ProductRecord]
    copurchase_index: dict[int, tuple[PairSignal, ...]]
    ingredient_recipe_index: dict[str, frozenset[str]]
    category_importance: dict[str, float]
    pair_penalty_rules: tuple[PairPenaltyRule, ...]
    missing_sources: tuple[str, ...]


def _canonical_token(token: str) -> str:
    normalized = str(token or "").strip().lower()
    if not normalized or normalized.isdigit():
        return ""
    return TOKEN_ALIASES.get(normalized, normalized)


def normalize_text(value: object) -> str:
    text = str(value or "").strip().lower()
    return text.replace("×", "x")


def tokenize_text(value: object) -> frozenset[str]:
    tokens: set[str] = set()
    for raw_token in TOKEN_RE.findall(normalize_text(value)):
        token = _canonical_token(raw_token)
        if not token or token in STOPWORDS:
            continue
        tokens.add(token)
    return frozenset(tokens)


def _parse_float(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _parse_int(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _split_tags(raw_tags: object) -> frozenset[str]:
    values = {
        normalize_text(token)
        for token in str(raw_tags or "").split("|")
        if normalize_text(token)
    }
    return frozenset(values)


def _dominant_tokens(name_tokens: frozenset[str]) -> frozenset[str]:
    return frozenset(token for token in name_tokens if token not in GENERIC_NAME_TOKENS)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_product_records(paths_root: Path, missing_sources: list[str]) -> dict[int, ProductRecord]:
    paths = get_project_paths(project_root=paths_root)
    categories_path = paths.processed_dir / "product_categories.csv"
    recipe_scores_path = paths.processed_dir / "product_recipe_scores.csv"
    if not categories_path.exists():
        missing_sources.append(str(categories_path.relative_to(paths_root)))
        return {}

    recipe_scores: dict[int, tuple[str, float]] = {}
    if recipe_scores_path.exists():
        for row in _read_csv_rows(recipe_scores_path):
            product_id = _parse_int(row.get("product_id"))
            if product_id <= 0:
                continue
            recipe_scores[product_id] = (
                normalize_text(row.get("matched_ingredient")),
                _parse_float(row.get("recipe_score")),
            )
    else:
        missing_sources.append(str(recipe_scores_path.relative_to(paths_root)))

    products: dict[int, ProductRecord] = {}
    for row in _read_csv_rows(categories_path):
        product_id = _parse_int(row.get("product_id"))
        if product_id <= 0:
            continue
        name = str(row.get("product_name") or "").strip()
        name_tokens = tokenize_text(name)
        matched_ingredient, recipe_score = recipe_scores.get(product_id, ("", 0.0))
        products[product_id] = ProductRecord(
            product_id=product_id,
            product_name=name,
            category=normalize_text(row.get("category")),
            subcategory=normalize_text(row.get("subcategory")),
            product_family=normalize_text(row.get("product_family")),
            frequency_score=_parse_float(row.get("frequency_score")),
            recipe_score=recipe_score,
            matched_ingredient=matched_ingredient,
            category_tags=_split_tags(row.get("category_tags")),
            name_tokens=name_tokens,
            dominant_tokens=_dominant_tokens(name_tokens),
        )
    return products


def _load_copurchase_index(paths_root: Path, missing_sources: list[str]) -> dict[int, tuple[PairSignal, ...]]:
    paths = get_project_paths(project_root=paths_root)
    copurchase_path = paths.processed_dir / "copurchase_scores.csv"
    if not copurchase_path.exists():
        missing_sources.append(str(copurchase_path.relative_to(paths_root)))
        return {}

    adjacency: dict[int, list[PairSignal]] = defaultdict(list)
    for row in _read_csv_rows(copurchase_path):
        product_a = _parse_int(row.get("product_a"))
        product_b = _parse_int(row.get("product_b"))
        if product_a <= 0 or product_b <= 0:
            continue
        pair_count = _parse_int(row.get("pair_count"))
        score = _parse_float(row.get("score"))
        adjacency[product_a].append(PairSignal(candidate_id=product_b, pair_count=pair_count, score=score))
        adjacency[product_b].append(PairSignal(candidate_id=product_a, pair_count=pair_count, score=score))

    return {
        product_id: tuple(
            sorted(
                signals,
                key=lambda item: (-item.pair_count, -item.score, item.candidate_id),
            )
        )
        for product_id, signals in adjacency.items()
    }


def _load_ingredient_recipe_index(paths_root: Path, missing_sources: list[str]) -> dict[str, frozenset[str]]:
    paths = get_project_paths(project_root=paths_root)
    recipe_data_path = paths.reference_dir / "recipe_data.json"
    if not recipe_data_path.exists():
        missing_sources.append(str(recipe_data_path.relative_to(paths_root)))
        return {}

    payload = json.loads(recipe_data_path.read_text(encoding="utf-8-sig"))
    ingredients = payload.get("ingredients", {})
    index: dict[str, frozenset[str]] = {}
    for ingredient, meta in ingredients.items():
        key = normalize_text(ingredient)
        recipes = meta.get("recipes") if isinstance(meta, dict) else ()
        values = frozenset(normalize_text(recipe) for recipe in recipes if normalize_text(recipe))
        if key and values:
            index[key] = values
    return index


def _load_category_importance(paths_root: Path, missing_sources: list[str]) -> dict[str, float]:
    paths = get_project_paths(project_root=paths_root)
    importance_path = paths.reference_dir / "category_importance.csv"
    if not importance_path.exists():
        missing_sources.append(str(importance_path.relative_to(paths_root)))
        return {}

    lookup: dict[str, float] = {}
    for row in _read_csv_rows(importance_path):
        key = normalize_text(row.get("category"))
        if key:
            lookup[key] = _parse_float(row.get("final_score"))
    return lookup


def _load_pair_penalty_rules(paths_root: Path, missing_sources: list[str]) -> tuple[PairPenaltyRule, ...]:
    paths = get_project_paths(project_root=paths_root)
    rules_path = paths.reference_dir / "pair_penalty_rules.json"
    if not rules_path.exists():
        missing_sources.append(str(rules_path.relative_to(paths_root)))
        return ()

    payload = json.loads(rules_path.read_text(encoding="utf-8-sig"))
    rules = []
    for row in payload.get("rules", []):
        rules.append(
            PairPenaltyRule(
                name=str(row.get("name") or "").strip(),
                anchor_terms=frozenset(tokenize_text(" ".join(row.get("anchor_terms", [])))),
                complement_terms=frozenset(tokenize_text(" ".join(row.get("complement_terms", [])))),
                multiplier=max(0.1, _parse_float(row.get("multiplier")) or 1.0),
                reason=str(row.get("reason") or "").strip(),
            )
        )
    return tuple(rules)


@lru_cache(maxsize=8)
def load_compatible_products_data(project_root: str | Path | None = None) -> CompatibleProductsData:
    resolved_root = get_project_paths(project_root=project_root).root
    missing_sources: list[str] = []
    products = _load_product_records(resolved_root, missing_sources)
    copurchase_index = _load_copurchase_index(resolved_root, missing_sources)
    ingredient_recipe_index = _load_ingredient_recipe_index(resolved_root, missing_sources)
    category_importance = _load_category_importance(resolved_root, missing_sources)
    pair_penalty_rules = _load_pair_penalty_rules(resolved_root, missing_sources)
    return CompatibleProductsData(
        project_root=resolved_root,
        products=products,
        copurchase_index=copurchase_index,
        ingredient_recipe_index=ingredient_recipe_index,
        category_importance=category_importance,
        pair_penalty_rules=pair_penalty_rules,
        missing_sources=tuple(sorted(set(missing_sources))),
    )
