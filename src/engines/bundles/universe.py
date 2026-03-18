"""Offline materialization of the reusable bundle universe."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import pandas as pd

from data.paths import ensure_project_layout, get_project_paths
from engines.bundles.data import BundleArtifactMeta, load_bundle_data
from engines.bundles.engine import (
    AggregatedBundleCandidate,
    PersonalizedBundlesEngine,
    SOURCE_PRIORITY,
)
from engines.compatible.data import load_compatible_products_data
from engines.fbt.data import load_fbt_data
from features.user_profiles import empty_user_profile
from utils.runtime import timestamp_utc, write_json


BUNDLE_UNIVERSE_FILENAME = "bundle_universe.parquet"
BUNDLE_UNIVERSE_REPORT_FILENAME = "bundle_universe_report.json"
BUNDLE_UNIVERSE_VERSION = "bundle_universe_v1"
DEFAULT_TARGET_SIZE = 100_000
DEFAULT_PER_ROOT_LIMIT = 18
STRICT_FALLBACK_QUALITY_FLOOR = 55.0


@dataclass(frozen=True)
class BundleUniverseMaterializationResult:
    artifact_path: Path
    report_path: Path
    report: dict[str, object]


@dataclass(frozen=True)
class BundleUniverseRecord:
    pair_key: tuple[int, int]
    ordered_product_ids: tuple[int, int]
    bundle_id: str
    item_names: tuple[str, str]
    item_categories: tuple[str, str]
    item_subcategories: tuple[str, str]
    category_pair: str
    archetype: str
    source_names: tuple[str, ...]
    source_families: tuple[str, ...]
    quality_score: float
    selection_score: float
    quality_band: str
    quality_rank: int
    source_quality_score: float
    consensus_bonus: float
    validity_score: float
    freshness_score: float
    genericity_penalty: float
    category_popularity_penalty: float
    root_support_count: int
    distinct_source_count: int
    generic_utility_count: int
    duplicate_variant_flag: bool
    has_live_support: bool
    has_curated_support: bool
    is_valid: bool
    bundle_seen_count: int
    last_seen_at: str | None
    evidence_signals: Mapping[str, object]
    source_details: Mapping[str, Mapping[str, object]]
    freshness_metadata: Mapping[str, object]


@dataclass(frozen=True)
class BundleUniverseStore:
    project_root: Path
    artifact_path: Path
    available: bool
    record_count: int
    universe_version: str
    materialized_at: str | None
    records: tuple[BundleUniverseRecord, ...]
    by_product_id: Mapping[int, tuple[int, ...]]
    by_category: Mapping[str, tuple[int, ...]]
    by_category_pair: Mapping[str, tuple[int, ...]]
    by_archetype: Mapping[str, tuple[int, ...]]
    by_quality_band: Mapping[str, tuple[int, ...]]
    by_source_family: Mapping[str, tuple[int, ...]]
    missing_sources: tuple[str, ...] = ()

    def as_summary(self) -> dict[str, object]:
        return {
            "available": self.available,
            "artifact_path": str(self.artifact_path),
            "record_count": self.record_count,
            "universe_version": self.universe_version,
            "materialized_at": self.materialized_at,
            "missing_sources": list(self.missing_sources),
        }


@dataclass(frozen=True)
class BundleUniverseCandidate:
    pair_key: tuple[int, int]
    ordered_product_ids: tuple[int, int]
    bundle_id: str
    item_1_name: str
    item_2_name: str
    item_1_category: str
    item_2_category: str
    item_1_subcategory: str
    item_2_subcategory: str
    category_pair: str
    archetype: str
    source_names: tuple[str, ...]
    source_families: tuple[str, ...]
    quality_score: float
    selection_score: float
    quality_band: str
    source_quality_score: float
    consensus_bonus: float
    validity_score: float
    freshness_score: float
    genericity_penalty: float
    category_popularity_penalty: float
    root_support_count: int
    distinct_source_count: int
    generic_utility_count: int
    duplicate_variant_flag: bool
    has_live_support: bool
    has_curated_support: bool
    bundle_seen_count: int
    last_seen_at: str | None
    evidence_signals: dict[str, object]
    source_details: dict[str, object]
    freshness_metadata: dict[str, object]

    def as_record(self, rank: int, materialized_at: str) -> dict[str, object]:
        return {
            "bundle_id": self.bundle_id,
            "item_1_id": int(self.ordered_product_ids[0]),
            "item_2_id": int(self.ordered_product_ids[1]),
            "item_1_name": self.item_1_name,
            "item_2_name": self.item_2_name,
            "item_1_category": self.item_1_category,
            "item_2_category": self.item_2_category,
            "item_1_subcategory": self.item_1_subcategory,
            "item_2_subcategory": self.item_2_subcategory,
            "category_pair": self.category_pair,
            "archetype": self.archetype,
            "source_names": "|".join(self.source_names),
            "source_families": "|".join(self.source_families),
            "quality_score": round(self.quality_score, 6),
            "selection_score": round(self.selection_score, 6),
            "quality_band": self.quality_band,
            "quality_rank": int(rank),
            "source_quality_score": round(self.source_quality_score, 6),
            "consensus_bonus": round(self.consensus_bonus, 6),
            "validity_score": round(self.validity_score, 6),
            "freshness_score": round(self.freshness_score, 6),
            "genericity_penalty": round(self.genericity_penalty, 6),
            "category_popularity_penalty": round(self.category_popularity_penalty, 6),
            "root_support_count": int(self.root_support_count),
            "source_count": int(self.distinct_source_count),
            "generic_utility_count": int(self.generic_utility_count),
            "duplicate_variant_flag": bool(self.duplicate_variant_flag),
            "has_live_support": bool(self.has_live_support),
            "has_curated_support": bool(self.has_curated_support),
            "is_valid": True,
            "bundle_seen_count": int(self.bundle_seen_count),
            "last_seen_at": self.last_seen_at,
            "evidence_signals": json.dumps(self.evidence_signals, ensure_ascii=True, sort_keys=True),
            "source_details": json.dumps(self.source_details, ensure_ascii=True, sort_keys=True),
            "freshness_metadata": json.dumps(self.freshness_metadata, ensure_ascii=True, sort_keys=True),
            "materialized_at": materialized_at,
            "universe_version": BUNDLE_UNIVERSE_VERSION,
        }


def _bundle_id_for_pair(pair_key: tuple[int, int], meta: BundleArtifactMeta | None) -> str:
    if meta is not None and meta.bundle_id:
        return meta.bundle_id
    return f"UV-{int(pair_key[0])}-{int(pair_key[1])}"


def _root_priority(
    root_id: int,
    *,
    profiled_roots: set[int],
    fbt_roots: set[int],
    curated_roots: set[int],
    fallback_roots: set[int],
) -> tuple[int, int]:
    score = 0
    if root_id in profiled_roots:
        score += 4
    if root_id in fbt_roots:
        score += 5
    if root_id in curated_roots:
        score += 3
    if root_id in fallback_roots:
        score += 1
    return (-score, int(root_id))


def _quality_band(rank: int, total: int) -> str:
    if total <= 0:
        return "tail"
    percentile = rank / max(1, total)
    if percentile <= 0.05:
        return "elite"
    if percentile <= 0.2:
        return "high"
    if percentile <= 0.5:
        return "medium"
    return "tail"


class BundleUniverseBuilder:
    def __init__(self, project_root: str | Path | None = None) -> None:
        self._project_root = get_project_paths(project_root=project_root).root
        self._engine = PersonalizedBundlesEngine(project_root=self._project_root)
        self._bundle_data = load_bundle_data(project_root=self._project_root)

    def build(
        self,
        *,
        target_size: int = DEFAULT_TARGET_SIZE,
        per_root_limit: int = DEFAULT_PER_ROOT_LIMIT,
        root_limit: int | None = None,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        target = max(1, int(target_size))
        per_root = max(4, int(per_root_limit))
        root_ids = self._root_ids(root_limit=root_limit)
        materialized_at = timestamp_utc()

        aggregated: dict[tuple[int, int], AggregatedBundleCandidate] = {}
        source_counts: Counter[str] = Counter()

        for record in self._bundle_data.top_bundles:
            candidate = self._engine._artifact_to_source_candidate(record, None)
            self._engine._merge_source_candidate(aggregated, candidate, None)
            source_counts[candidate.source_name] += 1
        for record in self._bundle_data.fallback_bundles:
            if float(record.quality_score) < STRICT_FALLBACK_QUALITY_FLOOR:
                continue
            candidate = self._engine._artifact_to_source_candidate(record, None)
            self._engine._merge_source_candidate(aggregated, candidate, None)
            source_counts[candidate.source_name] += 1

        for root_id in root_ids:
            for candidate in self._engine._compatible_candidates(root_id=root_id, limit=per_root, data=self._bundle_data):
                self._engine._merge_source_candidate(aggregated, candidate, None)
                source_counts[candidate.source_name] += 1
            for candidate in self._engine._fbt_candidates(root_id=root_id, limit=per_root, data=self._bundle_data):
                self._engine._merge_source_candidate(aggregated, candidate, None)
                source_counts[candidate.source_name] += 1

        prefiltered = self._prefilter_candidates(aggregated)
        selected = self._select_diverse_candidates(prefiltered, target=target)
        records = [
            candidate.as_record(rank=index, materialized_at=materialized_at)
            for index, candidate in enumerate(selected, start=1)
        ]
        frame = pd.DataFrame.from_records(records)
        if not frame.empty:
            frame = frame.sort_values(
                by=["quality_rank", "selection_score", "bundle_id"],
                ascending=[True, False, True],
            ).reset_index(drop=True)

        report = {
            "version": BUNDLE_UNIVERSE_VERSION,
            "materialized_at": materialized_at,
            "target_size": target,
            "selected_count": len(selected),
            "candidate_count_before_selection": len(prefiltered),
            "aggregated_pair_count": len(aggregated),
            "root_count": len(root_ids),
            "per_root_limit": per_root,
            "artifact_path": str(get_project_paths(project_root=self._project_root).features_dir / BUNDLE_UNIVERSE_FILENAME),
            "source_counts": dict(sorted(source_counts.items())),
            "selected_source_family_counts": dict(
                sorted(Counter("|".join(candidate.source_families) for candidate in selected).items())
            ),
            "selected_category_pair_counts": dict(
                sorted(Counter(candidate.category_pair for candidate in selected).items())
            ),
            "selected_quality_bands": dict(
                sorted(Counter(candidate.quality_band for candidate in selected).items())
            ),
            "missing_sources": list(self._bundle_data.missing_sources),
            "data_sources": {
                "compatible_engine": "compatible_products",
                "fbt_engine": "frequently_bought_together",
                "curated_pairs": "data/processed/top_bundles.csv",
                "fallback_pairs": "output/fallback_bundle_bank.json",
                "bundle_ids": "output/bundle_ids.csv",
            },
        }
        return frame, report

    def _root_ids(self, *, root_limit: int | None) -> list[int]:
        compatible_data = load_compatible_products_data(project_root=self._project_root)
        fbt_data = load_fbt_data(project_root=self._project_root)
        profiled_roots = {
            int(product_id)
            for product_id, product in compatible_data.products.items()
            if self._engine._compatible_engine._profiles_for_root(product)
        }
        fbt_roots = {int(product_id) for product_id in fbt_data.pair_index}
        curated_roots = {int(product_id) for product_id in self._bundle_data.top_bundle_index}
        fallback_roots = {int(product_id) for product_id in self._bundle_data.fallback_index}
        roots = sorted(
            profiled_roots | fbt_roots | curated_roots | fallback_roots,
            key=lambda root_id: _root_priority(
                root_id,
                profiled_roots=profiled_roots,
                fbt_roots=fbt_roots,
                curated_roots=curated_roots,
                fallback_roots=fallback_roots,
            ),
        )
        if root_limit is not None:
            return roots[: max(1, int(root_limit))]
        return roots

    def _prefilter_candidates(
        self,
        aggregated: dict[tuple[int, int], AggregatedBundleCandidate],
    ) -> list[BundleUniverseCandidate]:
        empty_profile = empty_user_profile()
        prefiltered: list[dict[str, object]] = []
        category_pair_counts: Counter[str] = Counter()
        archetype_counts: Counter[str] = Counter()
        item_counts: Counter[int] = Counter()

        for aggregated_candidate in aggregated.values():
            items = self._engine._resolve_items(aggregated_candidate, self._bundle_data)
            if items is None:
                continue
            if self._engine._reject_candidate(items, aggregated_candidate, None):
                continue

            sources = tuple(
                source_name
                for source_name, _ in sorted(
                    aggregated_candidate.source_map.items(),
                    key=lambda item: (SOURCE_PRIORITY.get(item[0], 99), item[0]),
                )
            )
            source_candidates = tuple(aggregated_candidate.source_map[source_name] for source_name in sources)
            source_families = tuple(dict.fromkeys(source.source_family for source in source_candidates))
            source_quality_score = round(sum(self._engine._source_component(source) for source in source_candidates), 6)
            consensus_bonus = round(self._engine._consensus_bonus(source_candidates), 6)
            validity_score = round(
                self._engine._validity_score(items=items, source_candidates=source_candidates, requested_root_id=None),
                6,
            )
            freshness_score = round(
                self._engine._freshness_score(
                    pair_key=aggregated_candidate.pair_key,
                    source_candidates=source_candidates,
                    requested_root_id=None,
                    user_profile=empty_profile,
                ),
                6,
            )
            base_generic_penalty = round(self._engine._generic_penalty(items, source_candidates), 6)
            quality_score = round(
                source_quality_score + consensus_bonus + validity_score + freshness_score - base_generic_penalty,
                6,
            )
            if quality_score < 12.0:
                continue

            category_pair = "|".join(sorted({item.category for item in items if item.category}))
            archetype = self._engine._candidate_archetype_key(items) or category_pair

            duplicate_variant_flag = self._engine._looks_duplicate_variant(items[0], items[1])
            generic_utility_count = self._engine._generic_utility_count(items)
            root_support_count = len({source.source_root_id for source in source_candidates if source.source_root_id is not None})
            has_live_support = any(source.source_family in {"compatible", "fbt"} for source in source_candidates)
            has_curated_support = any(source.source_name in {"legacy_curated_bundle", "legacy_fallback_bundle"} for source in source_candidates)
            meta = self._bundle_data.bundle_meta_lookup.get(aggregated_candidate.pair_key)
            prefiltered.append(
                {
                    "pair_key": aggregated_candidate.pair_key,
                    "ordered_product_ids": aggregated_candidate.ordered_product_ids,
                    "items": items,
                    "sources": sources,
                    "source_families": source_families,
                    "source_candidates": source_candidates,
                    "source_quality_score": source_quality_score,
                    "consensus_bonus": consensus_bonus,
                    "validity_score": validity_score,
                    "freshness_score": freshness_score,
                    "base_generic_penalty": base_generic_penalty,
                    "quality_score": quality_score,
                    "category_pair": category_pair,
                    "archetype": archetype,
                    "duplicate_variant_flag": duplicate_variant_flag,
                    "generic_utility_count": generic_utility_count,
                    "root_support_count": root_support_count,
                    "has_live_support": has_live_support,
                    "has_curated_support": has_curated_support,
                    "meta": meta,
                    "bundle_id": _bundle_id_for_pair(aggregated_candidate.pair_key, meta),
                }
            )
            category_pair_counts[category_pair] += 1
            archetype_counts[archetype] += 1
            item_counts[int(aggregated_candidate.ordered_product_ids[0])] += 1
            item_counts[int(aggregated_candidate.ordered_product_ids[1])] += 1

        candidates: list[BundleUniverseCandidate] = []
        for row in prefiltered:
            category_pair = str(row["category_pair"])
            archetype = str(row["archetype"])
            category_popularity_penalty = min(1.75, math.log1p(category_pair_counts[category_pair]) * 0.22)
            archetype_penalty = min(1.25, math.log1p(archetype_counts[archetype]) * 0.16)
            item_penalty = min(
                1.5,
                sum(math.log1p(item_counts[int(product_id)]) for product_id in row["ordered_product_ids"]) * 0.05,
            )
            fallback_only_penalty = (
                0.6
                if tuple(row["source_families"]) == ("legacy_bundle",) and not bool(row["has_live_support"])
                else 0.0
            )
            genericity_penalty = round(
                float(row["base_generic_penalty"])
                + category_popularity_penalty
                + archetype_penalty
                + item_penalty
                + fallback_only_penalty,
                6,
            )
            selection_score = round(float(row["quality_score"]) - genericity_penalty, 6)
            if selection_score < 10.5:
                continue

            source_candidates = tuple(row["source_candidates"])
            source_details = {
                source.source_name: {
                    "source_family": source.source_family,
                    "source_score": round(source.source_score, 6),
                    "signals": dict(source.signals),
                }
                for source in source_candidates
            }
            items = tuple(row["items"])
            meta = row["meta"]
            evidence_signals = {
                "evidence": list(self._engine._build_evidence(source_candidates)),
                "compatible_rule_score_max": max(
                    (float(source.signals.get("rule_score", 0.0)) for source in source_candidates if source.source_name == "compatible_products"),
                    default=0.0,
                ),
                "compatible_use_case_score_max": max(
                    (float(source.signals.get("use_case_score", 0.0)) for source in source_candidates if source.source_name == "compatible_products"),
                    default=0.0,
                ),
                "fbt_cooccurrence_count_max": max(
                    (int(float(source.signals.get("cooccurrence_count", 0.0))) for source in source_candidates if source.source_name == "frequently_bought_together"),
                    default=0,
                ),
                "fbt_lift_max": max(
                    (float(source.signals.get("lift", 0.0)) for source in source_candidates if source.source_name == "frequently_bought_together"),
                    default=0.0,
                ),
                "legacy_quality_score_max": max(
                    (
                        float(source.signals.get("quality_score", source.source_score))
                        for source in source_candidates
                        if source.source_name.startswith("legacy_")
                    ),
                    default=0.0,
                ),
            }
            freshness_metadata = {
                "bundle_seen_count": 0 if meta is None else int(meta.seen_count),
                "last_seen_at": None if meta is None else meta.last_seen_at,
                "has_existing_bundle_id": bool(meta is not None and meta.bundle_id),
            }
            candidates.append(
                BundleUniverseCandidate(
                    pair_key=tuple(row["pair_key"]),
                    ordered_product_ids=tuple(row["ordered_product_ids"]),
                    bundle_id=str(row["bundle_id"]),
                    item_1_name=items[0].product_name,
                    item_2_name=items[1].product_name,
                    item_1_category=items[0].category,
                    item_2_category=items[1].category,
                    item_1_subcategory=items[0].subcategory,
                    item_2_subcategory=items[1].subcategory,
                    category_pair=category_pair,
                    archetype=archetype,
                    source_names=tuple(row["sources"]),
                    source_families=tuple(row["source_families"]),
                    quality_score=float(row["quality_score"]),
                    selection_score=selection_score,
                    quality_band="tail",
                    source_quality_score=float(row["source_quality_score"]),
                    consensus_bonus=float(row["consensus_bonus"]),
                    validity_score=float(row["validity_score"]),
                    freshness_score=float(row["freshness_score"]),
                    genericity_penalty=genericity_penalty,
                    category_popularity_penalty=round(category_popularity_penalty + archetype_penalty, 6),
                    root_support_count=int(row["root_support_count"]),
                    distinct_source_count=len(tuple(row["sources"])),
                    generic_utility_count=int(row["generic_utility_count"]),
                    duplicate_variant_flag=bool(row["duplicate_variant_flag"]),
                    has_live_support=bool(row["has_live_support"]),
                    has_curated_support=bool(row["has_curated_support"]),
                    bundle_seen_count=0 if meta is None else int(meta.seen_count),
                    last_seen_at=None if meta is None else meta.last_seen_at,
                    evidence_signals=evidence_signals,
                    source_details=source_details,
                    freshness_metadata=freshness_metadata,
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.selection_score,
                -candidate.quality_score,
                -candidate.root_support_count,
                candidate.ordered_product_ids[0],
                candidate.ordered_product_ids[1],
            )
        )
        total = len(candidates)
        return [
            BundleUniverseCandidate(
                **{
                    **candidate.__dict__,
                    "quality_band": _quality_band(index, total),
                }
            )
            for index, candidate in enumerate(candidates, start=1)
        ]

    def _select_diverse_candidates(
        self,
        candidates: list[BundleUniverseCandidate],
        *,
        target: int,
    ) -> list[BundleUniverseCandidate]:
        effective_target = min(max(1, int(target)), len(candidates))
        selected: list[BundleUniverseCandidate] = []
        seen_pairs: set[tuple[int, int]] = set()
        category_counts: Counter[str] = Counter()
        archetype_counts: Counter[str] = Counter()
        item_counts: Counter[int] = Counter()
        fallback_only_count = 0

        passes = (
            {
                "max_per_category_pair": max(2, int(effective_target * 0.06)),
                "max_per_archetype": max(2, int(effective_target * 0.05)),
                "max_per_item": max(2, int(effective_target * 0.004)),
                "max_fallback_only": max(1, int(effective_target * 0.05)),
            },
            {
                "max_per_category_pair": max(3, int(effective_target * 0.09)),
                "max_per_archetype": max(3, int(effective_target * 0.07)),
                "max_per_item": max(3, int(effective_target * 0.006)),
                "max_fallback_only": max(2, int(effective_target * 0.08)),
            },
            {
                "max_per_category_pair": max(4, int(effective_target * 0.12)),
                "max_per_archetype": max(4, int(effective_target * 0.09)),
                "max_per_item": max(4, int(effective_target * 0.008)),
                "max_fallback_only": max(3, int(effective_target * 0.1)),
            },
            {
                "max_per_category_pair": max(5, int(effective_target * 0.15)),
                "max_per_archetype": max(5, int(effective_target * 0.11)),
                "max_per_item": max(5, int(effective_target * 0.01)),
                "max_fallback_only": max(4, int(effective_target * 0.12)),
            },
        )

        for cap in passes:
            if len(selected) >= effective_target:
                break
            for candidate in candidates:
                if len(selected) >= effective_target:
                    break
                if candidate.pair_key in seen_pairs:
                    continue
                if category_counts[candidate.category_pair] >= cap["max_per_category_pair"]:
                    continue
                if archetype_counts[candidate.archetype] >= cap["max_per_archetype"]:
                    continue
                if any(item_counts[int(product_id)] >= cap["max_per_item"] for product_id in candidate.ordered_product_ids):
                    continue
                fallback_only = candidate.source_families == ("legacy_bundle",) and not candidate.has_live_support
                if fallback_only and fallback_only_count >= cap["max_fallback_only"]:
                    continue

                selected.append(candidate)
                seen_pairs.add(candidate.pair_key)
                category_counts[candidate.category_pair] += 1
                archetype_counts[candidate.archetype] += 1
                for product_id in candidate.ordered_product_ids:
                    item_counts[int(product_id)] += 1
                if fallback_only:
                    fallback_only_count += 1

        return selected[:effective_target]


def materialize_bundle_universe(
    *,
    project_root: str | Path | None = None,
    target_size: int = DEFAULT_TARGET_SIZE,
    per_root_limit: int = DEFAULT_PER_ROOT_LIMIT,
    root_limit: int | None = None,
) -> BundleUniverseMaterializationResult:
    paths = ensure_project_layout(get_project_paths(project_root=project_root))
    builder = BundleUniverseBuilder(project_root=paths.root)
    frame, report = builder.build(
        target_size=target_size,
        per_root_limit=per_root_limit,
        root_limit=root_limit,
    )
    artifact_path = paths.features_dir / BUNDLE_UNIVERSE_FILENAME
    report_path = paths.reports_dir / BUNDLE_UNIVERSE_REPORT_FILENAME
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(artifact_path, index=False)
    write_json(report_path, report)
    return BundleUniverseMaterializationResult(
        artifact_path=artifact_path,
        report_path=report_path,
        report=report,
    )


def load_bundle_universe(project_root: str | Path | None = None) -> pd.DataFrame:
    paths = ensure_project_layout(get_project_paths(project_root=project_root))
    artifact_path = paths.features_dir / BUNDLE_UNIVERSE_FILENAME
    if not artifact_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(artifact_path)


def _parse_pipe_values(value: object) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    values = [part.strip() for part in text.split("|") if part.strip()]
    return tuple(dict.fromkeys(values))


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes"}


def _parse_json_object(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return {str(key): payload for key, payload in value.items()}
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if isinstance(payload, dict):
        return {str(key): payload for key, payload in payload.items()}
    return {}


def _tuple_index(counter: dict[object, list[int]]) -> dict[object, tuple[int, ...]]:
    return {key: tuple(values) for key, values in counter.items()}


@lru_cache(maxsize=8)
def _load_bundle_universe_store_cached(project_root_key: str, artifact_mtime_ns: int) -> BundleUniverseStore:
    project_root = Path(project_root_key).resolve()
    paths = ensure_project_layout(get_project_paths(project_root=project_root))
    artifact_path = paths.features_dir / BUNDLE_UNIVERSE_FILENAME
    if not artifact_path.exists():
        return BundleUniverseStore(
            project_root=project_root,
            artifact_path=artifact_path,
            available=False,
            record_count=0,
            universe_version=BUNDLE_UNIVERSE_VERSION,
            materialized_at=None,
            records=(),
            by_product_id={},
            by_category={},
            by_category_pair={},
            by_archetype={},
            by_quality_band={},
            by_source_family={},
            missing_sources=(str(artifact_path.relative_to(project_root)),),
        )

    frame = pd.read_parquet(artifact_path)
    records: list[BundleUniverseRecord] = []
    by_product_id: dict[int, list[int]] = {}
    by_category: dict[str, list[int]] = {}
    by_category_pair: dict[str, list[int]] = {}
    by_archetype: dict[str, list[int]] = {}
    by_quality_band: dict[str, list[int]] = {}
    by_source_family: dict[str, list[int]] = {}
    materialized_at: str | None = None
    universe_version = BUNDLE_UNIVERSE_VERSION

    for row in frame.itertuples(index=False):
        pair_key = (int(row.item_1_id), int(row.item_2_id))
        record = BundleUniverseRecord(
            pair_key=pair_key,
            ordered_product_ids=pair_key,
            bundle_id=str(row.bundle_id),
            item_names=(str(row.item_1_name or ""), str(row.item_2_name or "")),
            item_categories=(str(row.item_1_category or ""), str(row.item_2_category or "")),
            item_subcategories=(str(row.item_1_subcategory or ""), str(row.item_2_subcategory or "")),
            category_pair=str(row.category_pair or ""),
            archetype=str(row.archetype or ""),
            source_names=_parse_pipe_values(row.source_names),
            source_families=_parse_pipe_values(row.source_families),
            quality_score=float(row.quality_score or 0.0),
            selection_score=float(row.selection_score or 0.0),
            quality_band=str(row.quality_band or "tail"),
            quality_rank=int(row.quality_rank or 0),
            source_quality_score=float(row.source_quality_score or 0.0),
            consensus_bonus=float(row.consensus_bonus or 0.0),
            validity_score=float(row.validity_score or 0.0),
            freshness_score=float(row.freshness_score or 0.0),
            genericity_penalty=float(row.genericity_penalty or 0.0),
            category_popularity_penalty=float(row.category_popularity_penalty or 0.0),
            root_support_count=int(row.root_support_count or 0),
            distinct_source_count=int(row.source_count or 0),
            generic_utility_count=int(row.generic_utility_count or 0),
            duplicate_variant_flag=_parse_bool(row.duplicate_variant_flag),
            has_live_support=_parse_bool(row.has_live_support),
            has_curated_support=_parse_bool(row.has_curated_support),
            is_valid=_parse_bool(row.is_valid),
            bundle_seen_count=int(row.bundle_seen_count or 0),
            last_seen_at=str(row.last_seen_at).strip() or None,
            evidence_signals=_parse_json_object(row.evidence_signals),
            source_details={
                key: value
                for key, value in _parse_json_object(row.source_details).items()
                if isinstance(value, dict)
            },
            freshness_metadata=_parse_json_object(row.freshness_metadata),
        )
        records.append(record)
        index = len(records) - 1

        for product_id in record.ordered_product_ids:
            by_product_id.setdefault(int(product_id), []).append(index)
        for category in dict.fromkeys(part for part in record.item_categories if str(part).strip()):
            by_category.setdefault(str(category), []).append(index)
        if record.category_pair:
            by_category_pair.setdefault(record.category_pair, []).append(index)
        if record.archetype:
            by_archetype.setdefault(record.archetype, []).append(index)
        if record.quality_band:
            by_quality_band.setdefault(record.quality_band, []).append(index)
        for source_family in record.source_families:
            by_source_family.setdefault(str(source_family), []).append(index)

        row_materialized_at = str(getattr(row, "materialized_at", "") or "").strip() or None
        if materialized_at is None and row_materialized_at:
            materialized_at = row_materialized_at
        row_version = str(getattr(row, "universe_version", "") or "").strip()
        if row_version:
            universe_version = row_version

    return BundleUniverseStore(
        project_root=project_root,
        artifact_path=artifact_path,
        available=True,
        record_count=len(records),
        universe_version=universe_version,
        materialized_at=materialized_at,
        records=tuple(records),
        by_product_id=_tuple_index(by_product_id),
        by_category=_tuple_index(by_category),
        by_category_pair=_tuple_index(by_category_pair),
        by_archetype=_tuple_index(by_archetype),
        by_quality_band=_tuple_index(by_quality_band),
        by_source_family=_tuple_index(by_source_family),
        missing_sources=(),
    )


def load_bundle_universe_store(project_root: str | Path | None = None) -> BundleUniverseStore:
    paths = get_project_paths(project_root=project_root)
    artifact_path = paths.features_dir / BUNDLE_UNIVERSE_FILENAME
    artifact_mtime_ns = artifact_path.stat().st_mtime_ns if artifact_path.exists() else -1
    return _load_bundle_universe_store_cached(str(paths.root), artifact_mtime_ns)
