"""Data loading for personalized bundle serving."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from data.paths import get_project_paths

TOKEN_RE = re.compile(r"[a-z0-9\u0600-\u06ff]+")
TOKEN_ALIASES = {
    "biscuits": "biscuit",
    "cookies": "cookie",
    "dates": "date",
    "liters": "liter",
    "pieces": "piece",
    "tomatoes": "tomato",
}
STOPWORDS = {
    "a",
    "al",
    "and",
    "box",
    "for",
    "g",
    "gm",
    "grams",
    "kg",
    "kilo",
    "l",
    "liter",
    "ml",
    "of",
    "pack",
    "piece",
    "pieces",
    "plus",
    "the",
    "with",
    "x",
}
GENERIC_NAME_TOKENS = {
    "authentic",
    "classic",
    "fresh",
    "large",
    "mixed",
    "premium",
    "small",
    "whole",
}


@dataclass(frozen=True)
class BundleProductRecord:
    product_id: int
    product_name: str
    category: str
    subcategory: str
    product_family: str
    matched_ingredient: str
    name_tokens: frozenset[str]
    dominant_tokens: frozenset[str]


@dataclass(frozen=True)
class BundleArtifactMeta:
    bundle_id: str
    seen_count: int
    last_bundle_price: float | None
    last_seen_at: str | None


@dataclass(frozen=True)
class BundleArtifactRecord:
    pair_key: tuple[int, int]
    ordered_product_ids: tuple[int, int]
    item_names: tuple[str, str]
    source_name: str
    source_family: str
    bundle_price: float | None
    quality_score: float
    category_key: str
    anchor_category: str
    complement_category: str
    bundle_id: str
    seen_count: int
    rank: int
    signals: Mapping[str, object]


@dataclass(frozen=True)
class BundleData:
    project_root: Path
    products: dict[int, BundleProductRecord]
    bundle_meta_lookup: dict[tuple[int, int], BundleArtifactMeta]
    user_bundle_index: dict[str, tuple[BundleArtifactRecord, ...]]
    fallback_bundles: tuple[BundleArtifactRecord, ...]
    fallback_index: dict[int, tuple[BundleArtifactRecord, ...]]
    top_bundles: tuple[BundleArtifactRecord, ...]
    top_bundle_index: dict[int, tuple[BundleArtifactRecord, ...]]
    pair_quality_lookup: dict[tuple[int, int], BundleArtifactRecord]
    missing_sources: tuple[str, ...]


def _parse_int(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _parse_float(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


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


def _pair_key(item_1_id: int, item_2_id: int) -> tuple[int, int]:
    left = int(item_1_id)
    right = int(item_2_id)
    return (left, right) if left <= right else (right, left)


def _load_products(paths_root: Path, missing_sources: list[str]) -> dict[int, BundleProductRecord]:
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

    products: dict[int, BundleProductRecord] = {}
    for row in _read_csv_rows(categories_path):
        product_id = _parse_int(row.get("product_id"))
        if product_id <= 0:
            continue
        product_name = str(row.get("product_name") or "").strip()
        tokens = _tokenize_text(product_name)
        products[product_id] = BundleProductRecord(
            product_id=product_id,
            product_name=product_name,
            category=_normalize_text(row.get("category")),
            subcategory=_normalize_text(row.get("subcategory")),
            product_family=_normalize_text(row.get("product_family")),
            matched_ingredient=matched_ingredients.get(product_id, ""),
            name_tokens=tokens,
            dominant_tokens=_dominant_tokens(tokens),
        )
    return products


def _load_bundle_meta_lookup(paths_root: Path, missing_sources: list[str]) -> dict[tuple[int, int], BundleArtifactMeta]:
    bundle_ids_path = paths_root / "output" / "bundle_ids.csv"
    if not bundle_ids_path.exists():
        missing_sources.append(str(bundle_ids_path.relative_to(paths_root)))
        return {}

    lookup: dict[tuple[int, int], BundleArtifactMeta] = {}
    for row in _read_csv_rows(bundle_ids_path):
        item_1_id = _parse_int(row.get("item_1_id"))
        item_2_id = _parse_int(row.get("item_2_id"))
        if item_1_id <= 0 or item_2_id <= 0 or item_1_id == item_2_id:
            continue
        pair_key = _pair_key(item_1_id, item_2_id)
        lookup[pair_key] = BundleArtifactMeta(
            bundle_id=str(row.get("bundle_id") or "").strip(),
            seen_count=max(0, _parse_int(row.get("seen_count"))),
            last_bundle_price=round(_parse_float(row.get("last_bundle_price")), 2) or None,
            last_seen_at=str(row.get("last_seen_at") or "").strip() or None,
        )
    return lookup


def _resolve_product_name(product_id: int, products: dict[int, BundleProductRecord], fallback: str = "") -> str:
    product = products.get(int(product_id))
    if product is not None and product.product_name:
        return product.product_name
    text = str(fallback or "").strip()
    if text:
        return text
    return f"Product {int(product_id)}"


def _resolve_category_key(
    item_1_id: int,
    item_2_id: int,
    products: dict[int, BundleProductRecord],
    fallback: str = "",
) -> tuple[str, str, str]:
    left = products.get(int(item_1_id))
    right = products.get(int(item_2_id))
    anchor_category = left.category if left is not None else ""
    complement_category = right.category if right is not None else ""
    category_key = str(fallback or "").strip().lower()
    if not category_key and anchor_category and complement_category:
        category_key = "|".join(sorted((anchor_category, complement_category)))
    return category_key, anchor_category, complement_category


def _build_artifact_record(
    *,
    item_1_id: int,
    item_2_id: int,
    item_1_name: str,
    item_2_name: str,
    source_name: str,
    source_family: str,
    bundle_price: float | None,
    quality_score: float,
    category_key: str,
    anchor_category: str,
    complement_category: str,
    bundle_id_lookup: dict[tuple[int, int], BundleArtifactMeta],
    rank: int,
    signals: Mapping[str, object],
) -> BundleArtifactRecord | None:
    if item_1_id <= 0 or item_2_id <= 0 or item_1_id == item_2_id:
        return None
    pair_key = _pair_key(item_1_id, item_2_id)
    meta = bundle_id_lookup.get(pair_key)
    return BundleArtifactRecord(
        pair_key=pair_key,
        ordered_product_ids=(int(item_1_id), int(item_2_id)),
        item_names=(str(item_1_name).strip(), str(item_2_name).strip()),
        source_name=source_name,
        source_family=source_family,
        bundle_price=None if bundle_price is None or bundle_price <= 0.0 else round(float(bundle_price), 2),
        quality_score=round(float(max(0.0, quality_score)), 6),
        category_key=str(category_key or "").strip().lower(),
        anchor_category=str(anchor_category or "").strip().lower(),
        complement_category=str(complement_category or "").strip().lower(),
        bundle_id="" if meta is None else meta.bundle_id,
        seen_count=0 if meta is None else int(meta.seen_count),
        rank=int(rank),
        signals=dict(signals),
    )


def _load_fallback_bundles(
    paths_root: Path,
    *,
    products: dict[int, BundleProductRecord],
    bundle_id_lookup: dict[tuple[int, int], BundleArtifactMeta],
    missing_sources: list[str],
) -> tuple[BundleArtifactRecord, ...]:
    path = paths_root / "output" / "fallback_bundle_bank.json"
    if not path.exists():
        missing_sources.append(str(path.relative_to(paths_root)))
        return ()

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("bundles", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        return ()

    records: list[BundleArtifactRecord] = []
    for rank, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        item_1_id = _parse_int(row.get("item_1_id"))
        item_2_id = _parse_int(row.get("item_2_id"))
        category_key, anchor_category, complement_category = _resolve_category_key(
            item_1_id,
            item_2_id,
            products,
            fallback=str(row.get("category_key") or ""),
        )
        record = _build_artifact_record(
            item_1_id=item_1_id,
            item_2_id=item_2_id,
            item_1_name=_resolve_product_name(item_1_id, products),
            item_2_name=_resolve_product_name(item_2_id, products),
            source_name="legacy_fallback_bundle",
            source_family="legacy_bundle",
            bundle_price=_parse_float(row.get("bundle_price")),
            quality_score=_parse_float(row.get("quality_score")),
            category_key=category_key,
            anchor_category=str(row.get("anchor_category") or anchor_category),
            complement_category=str(row.get("complement_category") or complement_category),
            bundle_id_lookup=bundle_id_lookup,
            rank=rank,
            signals={
                "quality_score": round(_parse_float(row.get("quality_score")), 6),
                "category_key": category_key,
            },
        )
        if record is not None:
            records.append(record)
    return tuple(records)


def _load_top_bundles(
    paths_root: Path,
    *,
    products: dict[int, BundleProductRecord],
    bundle_id_lookup: dict[tuple[int, int], BundleArtifactMeta],
    missing_sources: list[str],
) -> tuple[BundleArtifactRecord, ...]:
    path = get_project_paths(project_root=paths_root).processed_dir / "top_bundles.csv"
    if not path.exists():
        missing_sources.append(str(path.relative_to(paths_root)))
        return ()

    records: list[BundleArtifactRecord] = []
    for rank, row in enumerate(_read_csv_rows(path)):
        item_1_id = _parse_int(row.get("product_a"))
        item_2_id = _parse_int(row.get("product_b"))
        category_key, anchor_category, complement_category = _resolve_category_key(
            item_1_id,
            item_2_id,
            products,
            fallback="|".join(
                part for part in (
                    _normalize_text(row.get("category_a")),
                    _normalize_text(row.get("category_b")),
                )
                if part
            ),
        )
        estimated_price = _parse_float(row.get("product_a_price")) + _parse_float(row.get("product_b_price"))
        quality_score = _parse_float(row.get("new_final_score")) or _parse_float(row.get("final_score"))
        record = _build_artifact_record(
            item_1_id=item_1_id,
            item_2_id=item_2_id,
            item_1_name=_resolve_product_name(item_1_id, products, fallback=str(row.get("product_a_name") or "")),
            item_2_name=_resolve_product_name(item_2_id, products, fallback=str(row.get("product_b_name") or "")),
            source_name="legacy_curated_bundle",
            source_family="legacy_bundle",
            bundle_price=estimated_price,
            quality_score=quality_score,
            category_key=category_key,
            anchor_category=anchor_category or _normalize_text(row.get("category_a")),
            complement_category=complement_category or _normalize_text(row.get("category_b")),
            bundle_id_lookup=bundle_id_lookup,
            rank=rank,
            signals={
                "quality_score": round(quality_score, 6),
                "pair_count": _parse_int(row.get("pair_count")),
                "recipe_compat_score": round(
                    _parse_float(row.get("recipe_compat_score")) or _parse_float(row.get("recipe_score_norm")),
                    6,
                ),
                "known_prior_flag": _parse_int(row.get("known_prior_flag")),
                "pair_penalty_multiplier": round(_parse_float(row.get("pair_penalty_multiplier")) or 1.0, 6),
                "utility_penalty_multiplier": round(_parse_float(row.get("utility_penalty_multiplier")) or 1.0, 6),
                "category_key": category_key,
            },
        )
        if record is not None:
            records.append(record)
    return tuple(records)


def _best_pair_quality(
    top_bundles: tuple[BundleArtifactRecord, ...],
    fallback_bundles: tuple[BundleArtifactRecord, ...],
) -> dict[tuple[int, int], BundleArtifactRecord]:
    lookup: dict[tuple[int, int], BundleArtifactRecord] = {}
    for record in (*top_bundles, *fallback_bundles):
        existing = lookup.get(record.pair_key)
        if existing is None or (
            record.quality_score,
            record.seen_count,
            -record.rank,
        ) > (
            existing.quality_score,
            existing.seen_count,
            -existing.rank,
        ):
            lookup[record.pair_key] = record
    return lookup


def _load_user_bundle_index(
    paths_root: Path,
    *,
    products: dict[int, BundleProductRecord],
    bundle_id_lookup: dict[tuple[int, int], BundleArtifactMeta],
    pair_quality_lookup: dict[tuple[int, int], BundleArtifactRecord],
    missing_sources: list[str],
) -> dict[str, tuple[BundleArtifactRecord, ...]]:
    path = paths_root / "output" / "final_recommendations_by_user.json"
    if not path.exists():
        missing_sources.append(str(path.relative_to(paths_root)))
        return {}

    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = payload.get("recommendations_by_user", {}) if isinstance(payload, dict) else {}
    if not isinstance(mapping, dict):
        return {}

    index: dict[str, tuple[BundleArtifactRecord, ...]] = {}
    for user_id in sorted(mapping):
        rows = mapping.get(user_id)
        if not isinstance(rows, list):
            continue
        records: list[BundleArtifactRecord] = []
        for rank, row in enumerate(rows):
            if not isinstance(row, dict):
                continue
            item_1_id = _parse_int(row.get("item_1_id"))
            item_2_id = _parse_int(row.get("item_2_id"))
            pair_key = _pair_key(item_1_id, item_2_id)
            reference = pair_quality_lookup.get(pair_key)
            category_key, anchor_category, complement_category = _resolve_category_key(
                item_1_id,
                item_2_id,
                products,
                fallback="" if reference is None else reference.category_key,
            )
            record = _build_artifact_record(
                item_1_id=item_1_id,
                item_2_id=item_2_id,
                item_1_name=_resolve_product_name(
                    item_1_id,
                    products,
                    fallback="" if reference is None else reference.item_names[0],
                ),
                item_2_name=_resolve_product_name(
                    item_2_id,
                    products,
                    fallback="" if reference is None else reference.item_names[1],
                ),
                source_name="legacy_user_bundle",
                source_family="legacy_bundle",
                bundle_price=_parse_float(row.get("bundle_price"))
                or (None if reference is None else reference.bundle_price or 0.0),
                quality_score=0.0 if reference is None else reference.quality_score,
                category_key=category_key,
                anchor_category="" if reference is None else reference.anchor_category or anchor_category,
                complement_category="" if reference is None else reference.complement_category or complement_category,
                bundle_id_lookup=bundle_id_lookup,
                rank=rank,
                signals={
                    "quality_score": 0.0 if reference is None else reference.quality_score,
                    "seen_count": 0 if reference is None else reference.seen_count,
                    "category_key": category_key,
                },
            )
            if record is not None:
                records.append(record)
        if records:
            index[str(user_id).strip()] = tuple(records)
    return index


def _index_by_product(records: tuple[BundleArtifactRecord, ...]) -> dict[int, tuple[BundleArtifactRecord, ...]]:
    index: dict[int, list[BundleArtifactRecord]] = defaultdict(list)
    for record in records:
        for product_id in set(record.ordered_product_ids):
            index[int(product_id)].append(record)
    return {
        product_id: tuple(rows)
        for product_id, rows in index.items()
    }


@lru_cache(maxsize=8)
def load_bundle_data(project_root: str | Path | None = None) -> BundleData:
    resolved_root = get_project_paths(project_root=project_root).root
    missing_sources: list[str] = []
    products = _load_products(resolved_root, missing_sources)
    bundle_id_lookup = _load_bundle_meta_lookup(resolved_root, missing_sources)
    fallback_bundles = _load_fallback_bundles(
        resolved_root,
        products=products,
        bundle_id_lookup=bundle_id_lookup,
        missing_sources=missing_sources,
    )
    top_bundles = _load_top_bundles(
        resolved_root,
        products=products,
        bundle_id_lookup=bundle_id_lookup,
        missing_sources=missing_sources,
    )
    pair_quality_lookup = _best_pair_quality(top_bundles, fallback_bundles)
    user_bundle_index = _load_user_bundle_index(
        resolved_root,
        products=products,
        bundle_id_lookup=bundle_id_lookup,
        pair_quality_lookup=pair_quality_lookup,
        missing_sources=missing_sources,
    )
    return BundleData(
        project_root=resolved_root,
        products=products,
        bundle_meta_lookup=bundle_id_lookup,
        user_bundle_index=user_bundle_index,
        fallback_bundles=fallback_bundles,
        fallback_index=_index_by_product(fallback_bundles),
        top_bundles=top_bundles,
        top_bundle_index=_index_by_product(top_bundles),
        pair_quality_lookup=pair_quality_lookup,
        missing_sources=tuple(sorted(set(missing_sources))),
    )
