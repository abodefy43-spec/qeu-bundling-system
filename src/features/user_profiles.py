"""Deterministic user behavior profiles for personalized bundle ranking."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Mapping

import pandas as pd

from data.paths import get_project_paths


FEATURE_VERSION = "user_profiles_v1"
PROFILE_SOURCE = "data/processed/filtered_orders.pkl"
MAX_PROFILE_PRODUCTS = 24
MAX_PROFILE_CATEGORIES = 12
MAX_PROFILE_ARCHETYPES = 10
MAX_RECENT_PRODUCTS = 10
MAX_RECENT_CATEGORIES = 6
MAX_ARCHETYPE_PRODUCTS = 10
RECENCY_HALFLIFE_DAYS = 14.0


@dataclass(frozen=True)
class ProfileCatalogRecord:
    product_id: int
    product_name: str
    category: str
    subcategory: str
    product_family: str
    matched_ingredient: str


@dataclass(frozen=True)
class UserProfileFeatures:
    user_id: str
    known_user: bool
    source: str
    interaction_count: int = 0
    unique_products: int = 0
    last_interaction_at: str | None = None
    observed_product_ids: frozenset[int] = frozenset()
    product_affinity: Mapping[int, float] = field(default_factory=dict)
    recent_product_affinity: Mapping[int, float] = field(default_factory=dict)
    category_affinity: Mapping[str, float] = field(default_factory=dict)
    recent_category_affinity: Mapping[str, float] = field(default_factory=dict)
    subcategory_affinity: Mapping[str, float] = field(default_factory=dict)
    product_family_affinity: Mapping[str, float] = field(default_factory=dict)
    archetype_affinity: Mapping[str, float] = field(default_factory=dict)
    fatigued_product_ids: frozenset[int] = frozenset()
    fatigued_categories: frozenset[str] = frozenset()
    recent_product_ids: tuple[int, ...] = ()
    recent_category_keys: tuple[str, ...] = ()

    def product_score(self, product_id: int) -> float:
        return float(self.product_affinity.get(int(product_id), 0.0))

    def recent_product_score(self, product_id: int) -> float:
        return float(self.recent_product_affinity.get(int(product_id), 0.0))

    def category_score(self, category: str) -> float:
        return float(self.category_affinity.get(_normalize_text(category), 0.0))

    def recent_category_score(self, category: str) -> float:
        return float(self.recent_category_affinity.get(_normalize_text(category), 0.0))

    def subcategory_score(self, subcategory: str) -> float:
        return float(self.subcategory_affinity.get(_normalize_text(subcategory), 0.0))

    def family_score(self, family: str) -> float:
        return float(self.product_family_affinity.get(_normalize_text(family), 0.0))

    def archetype_score(self, archetype_key: str) -> float:
        return float(self.archetype_affinity.get(_normalize_text(archetype_key), 0.0))

    def has_product(self, product_id: int) -> bool:
        return int(product_id) in self.observed_product_ids

    def has_pair(self, pair_key: tuple[int, int]) -> bool:
        left, right = pair_key
        return int(left) in self.observed_product_ids and int(right) in self.observed_product_ids

    def as_summary(self) -> dict[str, object]:
        return {
            "known_user": self.known_user,
            "source": self.source,
            "interaction_count": int(self.interaction_count),
            "unique_products": int(self.unique_products),
            "last_interaction_at": self.last_interaction_at,
            "recent_product_ids": [str(product_id) for product_id in self.recent_product_ids],
            "recent_category_keys": list(self.recent_category_keys),
        }


@dataclass(frozen=True)
class UserProfileStore:
    project_root: Path
    source_path: str
    feature_version: str
    profile_count: int
    catalog_size: int
    max_event_ts: str | None
    profiles: Mapping[str, UserProfileFeatures] = field(default_factory=dict)
    missing_sources: tuple[str, ...] = ()

    def get_profile(self, user_id: str | int | None) -> UserProfileFeatures:
        normalized = str(user_id or "").strip()
        if not normalized:
            return empty_user_profile()
        profile = self.profiles.get(normalized)
        if profile is not None:
            return profile
        return empty_user_profile(user_id=normalized, source=self.source_path)

    def as_summary(self) -> dict[str, object]:
        return {
            "feature_version": self.feature_version,
            "source_path": self.source_path,
            "profile_count": self.profile_count,
            "catalog_size": self.catalog_size,
            "max_event_ts": self.max_event_ts,
            "missing_sources": list(self.missing_sources),
        }


def empty_user_profile(*, user_id: str = "", source: str = PROFILE_SOURCE) -> UserProfileFeatures:
    return UserProfileFeatures(
        user_id=str(user_id or ""),
        known_user=False,
        source=source,
    )


def _normalize_text(value: object) -> str:
    return str(value or "").strip().lower()


def _parse_int(value: object) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(float(text))
    except ValueError:
        return 0


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _load_catalog(project_root: Path, missing_sources: list[str]) -> dict[int, ProfileCatalogRecord]:
    paths = get_project_paths(project_root=project_root)
    categories_path = paths.processed_dir / "product_categories.csv"
    recipe_scores_path = paths.processed_dir / "product_recipe_scores.csv"
    if not categories_path.exists():
        missing_sources.append(str(categories_path.relative_to(project_root)))
        return {}

    matched_ingredients: dict[int, str] = {}
    if recipe_scores_path.exists():
        for row in _read_csv_rows(recipe_scores_path):
            product_id = _parse_int(row.get("product_id"))
            if product_id <= 0:
                continue
            matched_ingredients[product_id] = _normalize_text(row.get("matched_ingredient"))
    else:
        missing_sources.append(str(recipe_scores_path.relative_to(project_root)))

    catalog: dict[int, ProfileCatalogRecord] = {}
    for row in _read_csv_rows(categories_path):
        product_id = _parse_int(row.get("product_id"))
        if product_id <= 0:
            continue
        catalog[product_id] = ProfileCatalogRecord(
            product_id=product_id,
            product_name=str(row.get("product_name") or "").strip(),
            category=_normalize_text(row.get("category")),
            subcategory=_normalize_text(row.get("subcategory")),
            product_family=_normalize_text(row.get("product_family")),
            matched_ingredient=matched_ingredients.get(product_id, ""),
        )
    return catalog


def _mapping_top_n(mapping: Mapping[object, float], limit: int) -> dict[object, float]:
    ranked = sorted(
        (
            (key, round(float(value), 6))
            for key, value in mapping.items()
            if key not in {None, ""} and float(value) > 0.0
        ),
        key=lambda item: (-item[1], str(item[0])),
    )
    return dict(ranked[:limit])


def _recency_weight(age_days: float) -> float:
    bounded = max(0.0, float(age_days))
    return 0.55 + (0.45 * math.exp(-(bounded / RECENCY_HALFLIFE_DAYS)))


def _category_key(left_category: str, right_category: str) -> str:
    left = _normalize_text(left_category)
    right = _normalize_text(right_category)
    parts = sorted(part for part in (left, right) if part)
    return "|".join(parts)


def _profile_from_frame(
    *,
    user_id: str,
    group: pd.DataFrame,
    catalog: Mapping[int, ProfileCatalogRecord],
    max_event_ts: pd.Timestamp,
) -> UserProfileFeatures:
    product_stats: dict[int, dict[str, object]] = {}
    for row in group.itertuples(index=False):
        product_id = int(row.product_id)
        created_at = row.created_at
        quantity = max(1.0, float(row.quantity))
        entry = product_stats.setdefault(
            product_id,
            {
                "quantity": 0.0,
                "line_count": 0,
                "last_ts": created_at,
            },
        )
        entry["quantity"] = float(entry["quantity"]) + quantity
        entry["line_count"] = int(entry["line_count"]) + 1
        if created_at > entry["last_ts"]:
            entry["last_ts"] = created_at

    product_affinity_raw: dict[int, float] = {}
    recent_product_affinity_raw: dict[int, float] = {}
    category_affinity_raw: Counter[str] = Counter()
    recent_category_affinity_raw: Counter[str] = Counter()
    subcategory_affinity_raw: Counter[str] = Counter()
    family_affinity_raw: Counter[str] = Counter()
    category_unique_products: defaultdict[str, set[int]] = defaultdict(set)
    observed_product_ids: set[int] = set()
    fatigued_product_ids: set[int] = set()

    for product_id, stats in product_stats.items():
        catalog_record = catalog.get(product_id)
        if catalog_record is None:
            continue
        observed_product_ids.add(product_id)
        quantity_total = float(stats["quantity"])
        line_count = int(stats["line_count"])
        last_ts = stats["last_ts"]
        age_days = max(0.0, float((max_event_ts - last_ts).total_seconds()) / 86400.0)
        base_weight = 1.0 + math.log1p(quantity_total) + (0.3 * max(0, line_count - 1))
        recent_weight = base_weight * _recency_weight(age_days)
        product_affinity_raw[product_id] = round(base_weight, 6)
        recent_product_affinity_raw[product_id] = round(recent_weight, 6)

        if quantity_total >= 2.0 or line_count >= 2:
            fatigued_product_ids.add(product_id)

        if catalog_record.category:
            category_affinity_raw[catalog_record.category] += base_weight
            recent_category_affinity_raw[catalog_record.category] += recent_weight
            category_unique_products[catalog_record.category].add(product_id)
        if catalog_record.subcategory:
            subcategory_affinity_raw[catalog_record.subcategory] += base_weight
        if catalog_record.product_family:
            family_affinity_raw[catalog_record.product_family] += base_weight

    product_affinity = _mapping_top_n(product_affinity_raw, MAX_PROFILE_PRODUCTS)
    recent_product_affinity = _mapping_top_n(recent_product_affinity_raw, MAX_PROFILE_PRODUCTS)
    category_affinity = _mapping_top_n(category_affinity_raw, MAX_PROFILE_CATEGORIES)
    recent_category_affinity = _mapping_top_n(recent_category_affinity_raw, MAX_PROFILE_CATEGORIES)
    subcategory_affinity = _mapping_top_n(subcategory_affinity_raw, MAX_PROFILE_CATEGORIES)
    product_family_affinity = _mapping_top_n(family_affinity_raw, MAX_PROFILE_CATEGORIES)

    ranked_product_ids = [
        int(product_id)
        for product_id, _score in sorted(
            recent_product_affinity_raw.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]
    archetype_scores: Counter[str] = Counter()
    for left_id, right_id in combinations(ranked_product_ids[:MAX_ARCHETYPE_PRODUCTS], 2):
        left = catalog.get(int(left_id))
        right = catalog.get(int(right_id))
        if left is None or right is None:
            continue
        left_key = left.category or left.subcategory
        right_key = right.category or right.subcategory
        archetype_key = _category_key(left_key, right_key)
        if not archetype_key:
            continue
        pair_weight = min(
            float(recent_product_affinity_raw.get(int(left_id), 0.0)),
            float(recent_product_affinity_raw.get(int(right_id), 0.0)),
        )
        if left_key == right_key:
            pair_weight *= 0.65
        archetype_scores[archetype_key] += pair_weight
    archetype_affinity = _mapping_top_n(archetype_scores, MAX_PROFILE_ARCHETYPES)

    total_category_weight = sum(float(value) for value in category_affinity_raw.values()) or 1.0
    fatigued_categories = {
        category
        for category, score in category_affinity_raw.items()
        if len(category_unique_products.get(category, set())) >= 3 or (float(score) / total_category_weight) >= 0.45
    }

    recent_product_ids = tuple(ranked_product_ids[:MAX_RECENT_PRODUCTS])
    recent_category_keys = tuple(
        key
        for key, _score in sorted(
            recent_category_affinity_raw.items(),
            key=lambda item: (-item[1], item[0]),
        )[:MAX_RECENT_CATEGORIES]
    )
    last_interaction_at = group["created_at"].max()
    return UserProfileFeatures(
        user_id=user_id,
        known_user=True,
        source=PROFILE_SOURCE,
        interaction_count=int(len(group)),
        unique_products=len(observed_product_ids),
        last_interaction_at=last_interaction_at.isoformat() if pd.notna(last_interaction_at) else None,
        observed_product_ids=frozenset(sorted(observed_product_ids)),
        product_affinity=product_affinity,
        recent_product_affinity=recent_product_affinity,
        category_affinity=category_affinity,
        recent_category_affinity=recent_category_affinity,
        subcategory_affinity=subcategory_affinity,
        product_family_affinity=product_family_affinity,
        archetype_affinity=archetype_affinity,
        fatigued_product_ids=frozenset(sorted(fatigued_product_ids)),
        fatigued_categories=frozenset(sorted(fatigued_categories)),
        recent_product_ids=recent_product_ids,
        recent_category_keys=recent_category_keys,
    )


@lru_cache(maxsize=8)
def load_user_profile_store(project_root: str | Path | None = None) -> UserProfileStore:
    resolved_root = get_project_paths(project_root=project_root).root
    paths = get_project_paths(project_root=resolved_root)
    missing_sources: list[str] = []
    catalog = _load_catalog(resolved_root, missing_sources)

    orders_path = paths.processed_dir / "filtered_orders.pkl"
    if not orders_path.exists():
        missing_sources.append(str(orders_path.relative_to(resolved_root)))
        return UserProfileStore(
            project_root=resolved_root,
            source_path=PROFILE_SOURCE,
            feature_version=FEATURE_VERSION,
            profile_count=0,
            catalog_size=len(catalog),
            max_event_ts=None,
            profiles={},
            missing_sources=tuple(sorted(set(missing_sources))),
        )

    orders = pd.read_pickle(orders_path)
    required_columns = {"order_id", "product_id", "created_at"}
    if not required_columns.issubset(set(orders.columns)):
        missing_sources.append("filtered_orders.pkl missing required columns")
        return UserProfileStore(
            project_root=resolved_root,
            source_path=PROFILE_SOURCE,
            feature_version=FEATURE_VERSION,
            profile_count=0,
            catalog_size=len(catalog),
            max_event_ts=None,
            profiles={},
            missing_sources=tuple(sorted(set(missing_sources))),
        )

    frame = orders.loc[:, [column for column in ("order_id", "product_id", "quantity", "created_at") if column in orders.columns]].copy()
    if "quantity" not in frame.columns:
        frame["quantity"] = 1.0
    frame["created_at"] = pd.to_datetime(frame["created_at"], utc=True, errors="coerce")
    frame["quantity"] = pd.to_numeric(frame["quantity"], errors="coerce").fillna(1.0).clip(lower=1.0)
    frame["product_id"] = pd.to_numeric(frame["product_id"], errors="coerce")
    frame["order_id"] = frame["order_id"].astype(str).str.strip()
    frame = frame.dropna(subset=["created_at", "product_id"])
    frame["product_id"] = frame["product_id"].astype(int)
    frame = frame[(frame["order_id"] != "") & (frame["product_id"] > 0)]
    if frame.empty:
        return UserProfileStore(
            project_root=resolved_root,
            source_path=PROFILE_SOURCE,
            feature_version=FEATURE_VERSION,
            profile_count=0,
            catalog_size=len(catalog),
            max_event_ts=None,
            profiles={},
            missing_sources=tuple(sorted(set(missing_sources))),
        )

    max_event_ts = frame["created_at"].max()
    profiles: dict[str, UserProfileFeatures] = {}
    for user_id, group in frame.sort_values(["order_id", "created_at", "product_id"]).groupby("order_id", sort=True):
        profile = _profile_from_frame(
            user_id=str(user_id),
            group=group,
            catalog=catalog,
            max_event_ts=max_event_ts,
        )
        if profile.known_user:
            profiles[profile.user_id] = profile

    return UserProfileStore(
        project_root=resolved_root,
        source_path=PROFILE_SOURCE,
        feature_version=FEATURE_VERSION,
        profile_count=len(profiles),
        catalog_size=len(catalog),
        max_event_ts=max_event_ts.isoformat() if pd.notna(max_event_ts) else None,
        profiles=profiles,
        missing_sources=tuple(sorted(set(missing_sources))),
    )
