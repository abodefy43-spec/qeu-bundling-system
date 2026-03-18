"""Order-level data loading for the frequently bought together engine."""

from __future__ import annotations

import csv
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
    "sauces": "sauce",
    "slices": "slice",
    "tomatoes": "tomato",
}
STOPWORDS = {
    "a",
    "al",
    "and",
    "box",
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
GENERIC_NAME_TOKENS = {
    "authentic",
    "classic",
    "copy",
    "fresh",
    "large",
    "mix",
    "mixed",
    "premium",
    "small",
    "whole",
}


@dataclass(frozen=True)
class FBTProductRecord:
    product_id: int
    product_name: str
    category: str
    subcategory: str
    product_family: str
    matched_ingredient: str
    name_tokens: frozenset[str]
    dominant_tokens: frozenset[str]


@dataclass(frozen=True)
class FBTPairRecord:
    candidate_id: int
    cooccurrence_count: int


@dataclass(frozen=True)
class FBTData:
    project_root: Path
    total_orders: int
    product_order_counts: dict[int, int]
    products: dict[int, FBTProductRecord]
    pair_index: dict[int, tuple[FBTPairRecord, ...]]
    missing_sources: tuple[str, ...]


def _parse_int(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _normalize_text(value: object) -> str:
    return str(value or "").strip().lower().replace("×", "x")


def _tokenize_text(value: object) -> frozenset[str]:
    tokens: set[str] = set()
    for raw_token in TOKEN_RE.findall(_normalize_text(value)):
        token = TOKEN_ALIASES.get(raw_token, raw_token)
        if token and token not in STOPWORDS and not token.isdigit():
            tokens.add(token)
    return frozenset(tokens)


def _dominant_tokens(tokens: frozenset[str]) -> frozenset[str]:
    return frozenset(token for token in tokens if token not in GENERIC_NAME_TOKENS)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_products(paths_root: Path, missing_sources: list[str]) -> dict[int, FBTProductRecord]:
    paths = get_project_paths(project_root=paths_root)
    categories_path = paths.processed_dir / "product_categories.csv"
    recipe_scores_path = paths.processed_dir / "product_recipe_scores.csv"
    if not categories_path.exists():
        missing_sources.append(str(categories_path.relative_to(paths_root)))
        return {}

    matched_ingredients: dict[int, str] = {}
    if recipe_scores_path.exists():
        for row in _read_csv_rows(recipe_scores_path):
            product_id = _parse_int(row.get("product_id"))
            if product_id <= 0:
                continue
            matched_ingredients[product_id] = _normalize_text(row.get("matched_ingredient"))
    else:
        missing_sources.append(str(recipe_scores_path.relative_to(paths_root)))

    products: dict[int, FBTProductRecord] = {}
    for row in _read_csv_rows(categories_path):
        product_id = _parse_int(row.get("product_id"))
        if product_id <= 0:
            continue
        name = str(row.get("product_name") or "").strip()
        name_tokens = _tokenize_text(name)
        products[product_id] = FBTProductRecord(
            product_id=product_id,
            product_name=name,
            category=_normalize_text(row.get("category")),
            subcategory=_normalize_text(row.get("subcategory")),
            product_family=_normalize_text(row.get("product_family")),
            matched_ingredient=matched_ingredients.get(product_id, ""),
            name_tokens=name_tokens,
            dominant_tokens=_dominant_tokens(name_tokens),
        )
    return products


def _load_pair_index(paths_root: Path, missing_sources: list[str]) -> dict[int, tuple[FBTPairRecord, ...]]:
    paths = get_project_paths(project_root=paths_root)
    pair_path = paths.processed_dir / "copurchase_scores.csv"
    if not pair_path.exists():
        missing_sources.append(str(pair_path.relative_to(paths_root)))
        return {}

    adjacency: dict[int, list[FBTPairRecord]] = defaultdict(list)
    for row in _read_csv_rows(pair_path):
        product_a = _parse_int(row.get("product_a"))
        product_b = _parse_int(row.get("product_b"))
        if product_a <= 0 or product_b <= 0 or product_a == product_b:
            continue
        pair_count = _parse_int(row.get("pair_count"))
        if pair_count <= 0:
            continue
        adjacency[product_a].append(FBTPairRecord(candidate_id=product_b, cooccurrence_count=pair_count))
        adjacency[product_b].append(FBTPairRecord(candidate_id=product_a, cooccurrence_count=pair_count))

    return {
        product_id: tuple(
            sorted(
                rows,
                key=lambda item: (-item.cooccurrence_count, item.candidate_id),
            )
        )
        for product_id, rows in adjacency.items()
    }


def _load_order_counts(paths_root: Path, missing_sources: list[str]) -> tuple[int, dict[int, int]]:
    paths = get_project_paths(project_root=paths_root)
    order_items_path = paths.raw_dir / "order_items.csv"
    if not order_items_path.exists():
        missing_sources.append(str(order_items_path.relative_to(paths_root)))
        return 0, {}

    order_ids: set[str] = set()
    product_order_counts: dict[int, int] = defaultdict(int)
    with order_items_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            order_id = str(row.get("order_id") or "").strip()
            product_id = _parse_int(row.get("product_id"))
            if not order_id or product_id <= 0:
                continue
            order_ids.add(order_id)
            product_order_counts[product_id] += 1

    total_orders = len(order_ids)
    if total_orders <= 0:
        return 0, {}
    return total_orders, {
        product_id: min(count, total_orders)
        for product_id, count in product_order_counts.items()
    }


@lru_cache(maxsize=8)
def load_fbt_data(project_root: str | Path | None = None) -> FBTData:
    resolved_root = get_project_paths(project_root=project_root).root
    missing_sources: list[str] = []
    products = _load_products(resolved_root, missing_sources)
    pair_index = _load_pair_index(resolved_root, missing_sources)
    total_orders, product_order_counts = _load_order_counts(resolved_root, missing_sources)
    return FBTData(
        project_root=resolved_root,
        total_orders=total_orders,
        product_order_counts=product_order_counts,
        products=products,
        pair_index=pair_index,
        missing_sources=tuple(sorted(set(missing_sources))),
    )
