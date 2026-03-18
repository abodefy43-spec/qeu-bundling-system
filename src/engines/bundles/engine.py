"""Production personalized bundle engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from data.paths import get_project_paths
from engines.bundles.data import BundleArtifactRecord, BundleData, BundleProductRecord, load_bundle_data
from engines.compatible.engine import CompatibleProductsEngine
from engines.fbt.engine import FrequentlyBoughtTogetherEngine
from features.user_profiles import UserProfileFeatures, load_user_profile_store
from shared.contracts import EngineDescriptor, EngineRequest, EngineResponse, RecommendationCandidate


ENGINE_DESCRIPTOR = EngineDescriptor(
    name="personalized_bundles",
    description="Bundle-serving engine that aggregates compatible, FBT, and legacy bundle candidates for a user.",
    required_inputs=("user_id",),
    output_description="Top personalized bundles ranked from multiple candidate sources.",
)

SOURCE_PRIORITY = {
    "compatible_products": 0,
    "frequently_bought_together": 1,
    "legacy_user_bundle": 2,
    "legacy_curated_bundle": 3,
    "legacy_fallback_bundle": 4,
}
SOURCE_BASE_WEIGHTS = {
    "compatible_products": 9.5,
    "frequently_bought_together": 10.5,
    "legacy_user_bundle": 8.5,
    "legacy_curated_bundle": 9.0,
    "legacy_fallback_bundle": 6.0,
}
QUALITY_BAND_RETRIEVAL_BONUS = {
    "elite": 6.0,
    "high": 4.0,
    "medium": 2.0,
    "tail": 0.75,
}
GENERIC_UTILITY_TOKENS = {
    "bag",
    "bags",
    "bleach",
    "cleaner",
    "cleaning",
    "detergent",
    "garbage",
    "napkin",
    "soap",
    "tissue",
    "tissues",
    "trash",
    "water",
}
NONFOOD_CATEGORIES = {
    "cleaning",
    "household",
    "paper_goods",
    "personal care",
    "personal_care",
}


@dataclass(frozen=True)
class BundleSourceCandidate:
    pair_key: tuple[int, int]
    ordered_product_ids: tuple[int, int]
    source_name: str
    source_family: str
    source_score: float
    bundle_price: float | None
    bundle_id: str
    source_root_id: int | None
    item_names: tuple[str, str]
    category_key: str
    anchor_category: str
    complement_category: str
    signals: Mapping[str, object]
    reasons: tuple[str, ...] = ()


@dataclass
class AggregatedBundleCandidate:
    pair_key: tuple[int, int]
    ordered_product_ids: tuple[int, int]
    source_map: dict[str, BundleSourceCandidate] = field(default_factory=dict)
    bundle_id: str = ""
    bundle_price: float | None = None
    candidate_metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RankedBundleCandidate:
    pair_key: tuple[int, int]
    ordered_product_ids: tuple[int, int]
    items: tuple[BundleProductRecord, BundleProductRecord]
    bundle_id: str
    bundle_price: float | None
    final_score: float
    source_quality_score: float
    consensus_bonus: float
    user_fit_score: float
    profile_signals: Mapping[str, float]
    validity_score: float
    freshness_score: float
    generic_penalty: float
    repetition_penalty: float
    fatigue_penalty: float
    exact_user_pair: bool
    sources: tuple[str, ...]
    source_families: tuple[str, ...]
    source_details: Mapping[str, Mapping[str, object]]
    evidence: tuple[str, ...]
    candidate_metadata: Mapping[str, object]


class PersonalizedBundlesEngine:
    descriptor = ENGINE_DESCRIPTOR

    def __init__(
        self,
        project_root: str | Path | None = None,
        *,
        compatible_engine: CompatibleProductsEngine | None = None,
        fbt_engine: FrequentlyBoughtTogetherEngine | None = None,
    ) -> None:
        self._project_root = Path(project_root).resolve() if project_root is not None else None
        self._compatible_engine = compatible_engine or CompatibleProductsEngine(project_root=self._project_root)
        self._fbt_engine = fbt_engine or FrequentlyBoughtTogetherEngine(project_root=self._project_root)

    def recommend(self, request: EngineRequest) -> EngineResponse:
        user_id = self._request_user_id(request)
        if not user_id:
            return EngineResponse(
                engine=self.descriptor.name,
                status="invalid_request",
                message="user_id is required for personalized_bundles.",
                metadata={"received_request": request.as_dict()},
            )

        data = load_bundle_data(project_root=self._resolved_project_root())
        requested_root_id: int | None = None
        if request.primary_product_id:
            try:
                requested_root_id = int(str(request.primary_product_id).strip())
            except ValueError:
                return EngineResponse(
                    engine=self.descriptor.name,
                    status="invalid_request",
                    message="root_product_id must be numeric when provided.",
                    metadata={"received_request": request.as_dict()},
                )

        if requested_root_id is not None and not self._root_exists(requested_root_id, data):
            return EngineResponse(
                engine=self.descriptor.name,
                status="not_found",
                message=f"Unknown root product id: {requested_root_id}",
                metadata={"root_product_id": str(requested_root_id)},
            )

        profile_store = load_user_profile_store(project_root=self._resolved_project_root())
        user_profile = profile_store.get_profile(user_id)
        user_bundle_records = data.user_bundle_index.get(user_id, ())
        history_product_ids = self._history_product_ids(
            request=request,
            user_bundle_records=user_bundle_records,
            user_profile=user_profile,
        )
        root_ids = self._resolve_root_ids(
            user_id=user_id,
            requested_root_id=requested_root_id,
            history_product_ids=history_product_ids,
            user_bundle_records=user_bundle_records,
            data=data,
        )
        exact_user_pairs = {record.pair_key for record in user_bundle_records}
        aggregated: dict[tuple[int, int], AggregatedBundleCandidate] = {}
        source_counts = {
            "bundle_universe": 0,
            "legacy": 0,
            "compatible": 0,
            "fbt": 0,
        }
        universe_aggregated, universe_source_counts, retrieval = self._retrieve_from_bundle_universe(
            user_id=user_id,
            requested_root_id=requested_root_id,
            request_limit=request.limit,
            root_ids=root_ids,
            history_product_ids=history_product_ids,
            user_profile=user_profile,
            data=data,
        )
        for candidate in universe_aggregated.values():
            self._merge_aggregated_candidate(aggregated, candidate, requested_root_id)
        for key, value in universe_source_counts.items():
            source_counts[key] = source_counts.get(key, 0) + int(value)

        ranked = self._rank_candidates(
            aggregated=aggregated,
            requested_root_id=requested_root_id,
            user_profile=user_profile,
            exact_user_pairs=exact_user_pairs,
            data=data,
        )
        selected = self._select_diverse_bundles(
            ranked=ranked,
            limit=request.limit,
            requested_root_id=requested_root_id,
        )

        fallback_reason = ""
        if len(selected) < int(request.limit):
            fallback_reason = "insufficient_universe_candidates" if retrieval["bundle_universe_available"] else "bundle_universe_unavailable"
            dynamic_aggregated, dynamic_counts = self._assemble_dynamic_candidates(
                user_id=user_id,
                requested_root_id=requested_root_id,
                root_ids=root_ids,
                limit=max(8, int(request.limit) * 4),
                data=data,
            )
            for candidate in dynamic_aggregated.values():
                self._merge_aggregated_candidate(aggregated, candidate, requested_root_id)
            for key, value in dynamic_counts.items():
                source_counts[key] = source_counts.get(key, 0) + int(value)
            if dynamic_aggregated:
                ranked = self._rank_candidates(
                    aggregated=aggregated,
                    requested_root_id=requested_root_id,
                    user_profile=user_profile,
                    exact_user_pairs=exact_user_pairs,
                    data=data,
                )
                selected = self._select_diverse_bundles(
                    ranked=ranked,
                    limit=request.limit,
                    requested_root_id=requested_root_id,
                )

        retrieval = dict(retrieval)
        if fallback_reason and source_counts["legacy"] + source_counts["compatible"] + source_counts["fbt"] > 0:
            retrieval["fallback_used"] = True
            retrieval["fallback_reason"] = fallback_reason
            retrieval["mode"] = (
                "dynamic_fallback"
                if not retrieval["bundle_universe_available"]
                else "bundle_universe+dynamic_fallback"
            )
        elif retrieval["bundle_universe_available"]:
            retrieval["mode"] = "bundle_universe"

        if not aggregated:
            if requested_root_id is None and not data.user_bundle_index and not data.fallback_bundles and not data.top_bundles:
                return EngineResponse(
                    engine=self.descriptor.name,
                    status="not_ready",
                    message="Bundle candidate sources are unavailable.",
                    metadata={
                        "missing_sources": list(data.missing_sources),
                        "profile_store": profile_store.as_summary(),
                        "profile": user_profile.as_summary(),
                        "retrieval": retrieval,
                    },
                )
            return EngineResponse(
                engine=self.descriptor.name,
                status="ok",
                message=f"No valid bundles found for user {user_id}.",
                metadata={
                    "user_id": user_id,
                    "root_product_id": None if requested_root_id is None else str(requested_root_id),
                    "requested_count": request.limit,
                    "returned_count": 0,
                    "root_candidates": [str(root_id) for root_id in root_ids],
                    "missing_sources": list(data.missing_sources),
                    "profile_store": profile_store.as_summary(),
                    "profile": user_profile.as_summary(),
                    "retrieval": retrieval,
                },
            )

        items = tuple(self._to_response_item(bundle, user_id, requested_root_id) for bundle in selected)
        return EngineResponse(
            engine=self.descriptor.name,
            status="ok",
            message=f"Ranked {len(items)} personalized bundles for user {user_id}.",
            items=items,
            metadata={
                "user_id": user_id,
                "root_product_id": None if requested_root_id is None else str(requested_root_id),
                "requested_count": request.limit,
                "returned_count": len(items),
                "root_candidates": [str(root_id) for root_id in root_ids],
                "candidate_source_counts": source_counts,
                "profile_store": profile_store.as_summary(),
                "profile": user_profile.as_summary(),
                "retrieval": retrieval,
                "data_sources": {
                    "bundle_universe": "data/processed/features/bundle_universe.parquet",
                    "compatible_engine": "compatible_products",
                    "fbt_engine": "frequently_bought_together",
                    "legacy_user_bundles": "output/final_recommendations_by_user.json",
                    "legacy_fallback_bundles": "output/fallback_bundle_bank.json",
                    "legacy_curated_pairs": "data/processed/top_bundles.csv",
                    "bundle_ids": "output/bundle_ids.csv",
                    "user_profiles": profile_store.source_path,
                },
                "missing_sources": list(data.missing_sources),
                "scoring_version": "personalized_bundles_v3",
            },
        )

    def _resolved_project_root(self) -> Path | None:
        if self._project_root is not None:
            return self._project_root
        return get_project_paths().root

    def _request_user_id(self, request: EngineRequest) -> str:
        return str(request.customer_id or "").strip()

    def _root_exists(self, root_id: int, data: BundleData) -> bool:
        return bool(
            root_id in data.products
            or root_id in data.top_bundle_index
            or root_id in data.fallback_index
        )

    def _history_product_ids(
        self,
        *,
        request: EngineRequest,
        user_bundle_records: tuple[BundleArtifactRecord, ...],
        user_profile: UserProfileFeatures,
    ) -> tuple[int, ...]:
        history_ids: list[int] = []
        raw_history = request.context.get("history_product_ids") if isinstance(request.context, dict) else None
        if isinstance(raw_history, (list, tuple, set)):
            for value in raw_history:
                try:
                    product_id = int(str(value).strip())
                except ValueError:
                    continue
                if product_id > 0:
                    history_ids.append(product_id)
        if history_ids:
            return tuple(dict.fromkeys(history_ids))

        if user_profile.known_user and user_profile.recent_product_ids:
            return tuple(int(product_id) for product_id in user_profile.recent_product_ids if int(product_id) > 0)

        derived: list[int] = []
        for record in user_bundle_records[:6]:
            derived.extend(record.ordered_product_ids)
        return tuple(dict.fromkeys(product_id for product_id in derived if product_id > 0))

    def _resolve_root_ids(
        self,
        *,
        user_id: str,
        requested_root_id: int | None,
        history_product_ids: tuple[int, ...],
        user_bundle_records: tuple[BundleArtifactRecord, ...],
        data: BundleData,
    ) -> tuple[int, ...]:
        roots: list[int] = []
        if requested_root_id is not None:
            roots.append(requested_root_id)

        for product_id in history_product_ids:
            if product_id in data.products:
                roots.append(product_id)

        if requested_root_id is None:
            for record in user_bundle_records[:4]:
                anchor_id = int(record.ordered_product_ids[0])
                if anchor_id in data.products:
                    roots.append(anchor_id)

        if not roots:
            fallback_pool = data.fallback_bundles or data.top_bundles
            if fallback_pool:
                start = self._stable_rotation_offset(user_id, len(fallback_pool))
                for offset in range(min(4, len(fallback_pool))):
                    record = fallback_pool[(start + offset) % len(fallback_pool)]
                    anchor_id = int(record.ordered_product_ids[0])
                    if anchor_id in data.products:
                        roots.append(anchor_id)

        deduped: list[int] = []
        for product_id in roots:
            if product_id not in deduped:
                deduped.append(product_id)
        return tuple(deduped[:3])

    def _stable_rotation_offset(self, value: str, size: int) -> int:
        if size <= 0:
            return 0
        total = 0
        for index, char in enumerate(str(value)):
            total += (index + 1) * ord(char)
        return total % size

    def _assemble_dynamic_candidates(
        self,
        *,
        user_id: str,
        requested_root_id: int | None,
        root_ids: tuple[int, ...],
        limit: int,
        data: BundleData,
    ) -> tuple[dict[tuple[int, int], AggregatedBundleCandidate], dict[str, int]]:
        aggregated: dict[tuple[int, int], AggregatedBundleCandidate] = {}
        source_counts = {
            "legacy": 0,
            "compatible": 0,
            "fbt": 0,
        }

        for candidate in self._legacy_candidates_for_request(
            user_id=user_id,
            requested_root_id=requested_root_id,
            root_ids=root_ids,
            limit=limit,
            data=data,
        ):
            self._merge_source_candidate(aggregated, candidate, requested_root_id)
            source_counts["legacy"] += 1

        for root_id in root_ids:
            for candidate in self._compatible_candidates(root_id=root_id, limit=limit, data=data):
                self._merge_source_candidate(aggregated, candidate, requested_root_id)
                source_counts["compatible"] += 1
            for candidate in self._fbt_candidates(root_id=root_id, limit=limit, data=data):
                self._merge_source_candidate(aggregated, candidate, requested_root_id)
                source_counts["fbt"] += 1

        return aggregated, source_counts

    def _bundle_universe_slice_limit(self, *, request_limit: int, requested_root_id: int | None) -> int:
        multiplier = 14 if requested_root_id is not None else 18
        return min(240, max(24, int(request_limit) * multiplier))

    def _retrieve_from_bundle_universe(
        self,
        *,
        user_id: str,
        requested_root_id: int | None,
        request_limit: int,
        root_ids: tuple[int, ...],
        history_product_ids: tuple[int, ...],
        user_profile: UserProfileFeatures,
        data: BundleData,
    ) -> tuple[dict[tuple[int, int], AggregatedBundleCandidate], dict[str, int], dict[str, object]]:
        from engines.bundles.universe import load_bundle_universe_store

        store = load_bundle_universe_store(project_root=self._resolved_project_root())
        slice_limit = self._bundle_universe_slice_limit(
            request_limit=request_limit,
            requested_root_id=requested_root_id,
        )
        retrieval_metadata = {
            "mode": "bundle_universe" if store.available else "dynamic_fallback",
            "bundle_universe_available": store.available,
            "artifact_path": str(store.artifact_path),
            "record_count": store.record_count,
            "slice_limit": slice_limit,
            "retrieved_record_count": 0,
            "retrieved_pair_count": 0,
            "fallback_used": False,
            "fallback_reason": "",
            "hooks": {
                "root_product_ids": [],
                "profile_product_ids": [],
                "category_keys": [],
                "archetypes": [],
                "source_families": ["compatible", "fbt"],
                "quality_bands": ["elite", "high", "medium"],
            },
        }
        if not store.available or not store.records:
            return {}, {"bundle_universe": 0}, retrieval_metadata

        root_product_ids: list[int] = []
        for product_id in ((requested_root_id,) if requested_root_id is not None else ()):
            if product_id is not None and product_id in data.products and product_id not in root_product_ids:
                root_product_ids.append(int(product_id))
        for product_id in root_ids:
            if product_id in data.products and product_id not in root_product_ids:
                root_product_ids.append(int(product_id))

        profile_product_ids: list[int] = []
        for source_ids in (history_product_ids, user_profile.recent_product_ids, tuple(user_profile.product_affinity.keys())):
            for product_id in source_ids:
                product_id = int(product_id)
                if product_id <= 0 or product_id not in data.products or product_id in root_product_ids:
                    continue
                if product_id in profile_product_ids:
                    continue
                profile_product_ids.append(product_id)
                if len(profile_product_ids) >= 10:
                    break
            if len(profile_product_ids) >= 10:
                break

        category_keys: list[str] = []
        for root_id in root_product_ids:
            root_product = data.products.get(root_id)
            if root_product is None:
                continue
            for key in (root_product.category, root_product.subcategory):
                if key and key not in category_keys:
                    category_keys.append(key)
        for source_keys in (user_profile.recent_category_keys, tuple(user_profile.category_affinity.keys())):
            for key in source_keys:
                normalized_key = str(key or "").strip().lower()
                if not normalized_key or normalized_key in category_keys:
                    continue
                category_keys.append(normalized_key)
                if len(category_keys) >= 8:
                    break
            if len(category_keys) >= 8:
                break

        archetype_keys: list[str] = []
        for key in user_profile.archetype_affinity.keys():
            normalized_key = str(key or "").strip().lower()
            if normalized_key and normalized_key not in archetype_keys:
                archetype_keys.append(normalized_key)
            if len(archetype_keys) >= 8:
                break
        if requested_root_id is not None:
            root_product = data.products.get(requested_root_id)
            root_category = "" if root_product is None else (root_product.category or root_product.subcategory)
            if root_category:
                for category_key in category_keys:
                    if category_key == root_category:
                        continue
                    candidate_key = "|".join(sorted((root_category, category_key)))
                    if candidate_key and candidate_key not in archetype_keys:
                        archetype_keys.append(candidate_key)
                    if len(archetype_keys) >= 10:
                        break

        retrieval_metadata["hooks"] = {
            "root_product_ids": [str(product_id) for product_id in root_product_ids],
            "profile_product_ids": [str(product_id) for product_id in profile_product_ids],
            "category_keys": list(category_keys),
            "archetypes": list(archetype_keys),
            "source_families": ["compatible", "fbt"],
            "quality_bands": ["elite", "high", "medium"],
        }

        candidate_indices: set[int] = set()

        def add_indices(indices: tuple[int, ...], cap: int) -> None:
            for index in indices[: max(0, cap)]:
                candidate_indices.add(int(index))

        product_cap = max(18, min(slice_limit * 2, 96))
        for product_id in root_product_ids:
            add_indices(store.by_product_id.get(int(product_id), ()), product_cap)
        for product_id in profile_product_ids:
            add_indices(store.by_product_id.get(int(product_id), ()), max(10, slice_limit // 2))

        if requested_root_id is None or len(candidate_indices) < slice_limit:
            for category_key in category_keys:
                add_indices(store.by_category.get(category_key, ()), max(12, slice_limit // 2))
            for archetype_key in archetype_keys:
                add_indices(store.by_archetype.get(archetype_key, ()), max(10, slice_limit // 3))
                add_indices(store.by_category_pair.get(archetype_key, ()), max(10, slice_limit // 3))

        if len(candidate_indices) < slice_limit:
            for source_family in ("compatible", "fbt"):
                add_indices(store.by_source_family.get(source_family, ()), max(12, slice_limit // 3))
        if len(candidate_indices) < slice_limit:
            for quality_band in ("elite", "high", "medium"):
                add_indices(store.by_quality_band.get(quality_band, ()), max(12, slice_limit // 2))

        root_product_set = set(root_product_ids)
        profile_product_set = set(profile_product_ids)
        category_set = set(category_keys)
        archetype_set = set(archetype_keys)
        observed_product_ids = set(int(product_id) for product_id in user_profile.observed_product_ids)
        scored_records: list[tuple[float, object]] = []

        for index in candidate_indices:
            record = store.records[int(index)]
            if not record.is_valid:
                continue
            if requested_root_id is not None and requested_root_id not in record.pair_key:
                continue
            if any(int(product_id) not in data.products for product_id in record.ordered_product_ids):
                continue

            pair_products = set(record.ordered_product_ids)
            retrieval_score = (record.selection_score * 0.72) + (record.quality_score * 0.18)
            retrieval_score += QUALITY_BAND_RETRIEVAL_BONUS.get(record.quality_band, 0.75)

            if requested_root_id is not None and requested_root_id in pair_products:
                retrieval_score += 16.0
            else:
                retrieval_score += min(8.0, 3.4 * len(root_product_set & pair_products))

            retrieval_score += min(4.5, 1.75 * len(profile_product_set & pair_products))

            category_matches = len(category_set & {category for category in record.item_categories if category})
            retrieval_score += min(3.6, 1.15 * category_matches)

            if record.archetype and record.archetype in archetype_set:
                retrieval_score += 3.0
            if record.category_pair and record.category_pair in archetype_set:
                retrieval_score += 2.6

            if "compatible" in record.source_families:
                retrieval_score += 0.65
            if "fbt" in record.source_families:
                retrieval_score += 0.9
            if record.has_live_support:
                retrieval_score += 0.85
            if record.has_curated_support:
                retrieval_score += 0.3

            retrieval_score -= min(record.genericity_penalty * 0.16, 1.35)
            if user_profile.known_user:
                if user_profile.has_pair(record.pair_key):
                    retrieval_score -= 1.4
                fatigue_penalty = 0.0
                for product_id, category in zip(record.ordered_product_ids, record.item_categories):
                    if requested_root_id is not None and int(product_id) == requested_root_id:
                        continue
                    if int(product_id) in user_profile.fatigued_product_ids:
                        fatigue_penalty += 0.4
                    elif int(product_id) in observed_product_ids:
                        fatigue_penalty += 0.18
                    if category and category in user_profile.fatigued_categories:
                        fatigue_penalty += 0.2
                retrieval_score -= min(fatigue_penalty, 1.0)

            scored_records.append((round(retrieval_score, 6), record))

        scored_records.sort(
            key=lambda item: (
                -item[0],
                item[1].quality_rank,
                item[1].ordered_product_ids[0],
                item[1].ordered_product_ids[1],
            )
        )
        selected_records = scored_records[:slice_limit]

        aggregated: dict[tuple[int, int], AggregatedBundleCandidate] = {}
        for retrieval_score, record in selected_records:
            aggregated_candidate = self._bundle_universe_record_to_aggregated(
                record=record,
                requested_root_id=requested_root_id,
                retrieval_score=float(retrieval_score),
                data=data,
            )
            self._merge_aggregated_candidate(aggregated, aggregated_candidate, requested_root_id)

        retrieval_metadata["retrieved_record_count"] = len(candidate_indices)
        retrieval_metadata["retrieved_pair_count"] = len(aggregated)
        return aggregated, {"bundle_universe": len(aggregated)}, retrieval_metadata

    def _bundle_universe_record_to_aggregated(
        self,
        *,
        record: object,
        requested_root_id: int | None,
        retrieval_score: float,
        data: BundleData,
    ) -> AggregatedBundleCandidate:
        ordered_product_ids = self._preferred_pair_order(record.pair_key, record.ordered_product_ids, requested_root_id)
        ordered_items = tuple(data.products.get(product_id) for product_id in ordered_product_ids)
        item_names = tuple(
            item.product_name if item is not None else f"Product {product_id}"
            for product_id, item in zip(ordered_product_ids, ordered_items)
        )
        item_categories = tuple(item.category if item is not None else "" for item in ordered_items)
        bundle_meta = data.bundle_meta_lookup.get(record.pair_key)
        entry = AggregatedBundleCandidate(
            pair_key=record.pair_key,
            ordered_product_ids=ordered_product_ids,
            bundle_id=record.bundle_id,
            bundle_price=None if bundle_meta is None else bundle_meta.last_bundle_price,
            candidate_metadata={
                "retrieval_source": "bundle_universe",
                "retrieval_score": round(retrieval_score, 4),
                "quality_score": round(record.quality_score, 4),
                "selection_score": round(record.selection_score, 4),
                "quality_band": record.quality_band,
                "quality_rank": record.quality_rank,
                "category_pair": record.category_pair,
                "archetype": record.archetype,
                "genericity_penalty": round(record.genericity_penalty, 4),
                "has_live_support": bool(record.has_live_support),
                "has_curated_support": bool(record.has_curated_support),
                "bundle_seen_count": int(record.bundle_seen_count),
                "last_seen_at": record.last_seen_at,
                "evidence_signals": dict(record.evidence_signals),
                "freshness_metadata": dict(record.freshness_metadata),
            },
        )

        for source_name in record.source_names:
            detail = record.source_details.get(source_name, {})
            source_family = str(detail.get("source_family") or self._source_family_for_name(source_name)).strip()
            source_score = float(detail.get("source_score", 0.0) or 0.0)
            signals = detail.get("signals")
            entry.source_map[source_name] = BundleSourceCandidate(
                pair_key=record.pair_key,
                ordered_product_ids=ordered_product_ids,
                source_name=source_name,
                source_family=source_family,
                source_score=source_score,
                bundle_price=entry.bundle_price,
                bundle_id=record.bundle_id,
                source_root_id=requested_root_id,
                item_names=item_names,
                category_key=record.category_pair,
                anchor_category=item_categories[0],
                complement_category=item_categories[1],
                signals=dict(signals) if isinstance(signals, dict) else {},
            )
        return entry

    def _source_family_for_name(self, source_name: str) -> str:
        if source_name == "compatible_products":
            return "compatible"
        if source_name == "frequently_bought_together":
            return "fbt"
        if source_name.startswith("legacy_"):
            return "legacy_bundle"
        return "bundle"

    def _merge_aggregated_candidate(
        self,
        aggregated: dict[tuple[int, int], AggregatedBundleCandidate],
        candidate: AggregatedBundleCandidate,
        requested_root_id: int | None,
    ) -> None:
        existing = aggregated.get(candidate.pair_key)
        if existing is None:
            aggregated[candidate.pair_key] = AggregatedBundleCandidate(
                pair_key=candidate.pair_key,
                ordered_product_ids=self._preferred_pair_order(
                    candidate.pair_key,
                    candidate.ordered_product_ids,
                    requested_root_id,
                ),
                bundle_id=candidate.bundle_id,
                bundle_price=candidate.bundle_price,
                candidate_metadata=dict(candidate.candidate_metadata),
            )
            existing = aggregated[candidate.pair_key]
        elif candidate.candidate_metadata and not existing.candidate_metadata:
            existing.candidate_metadata = dict(candidate.candidate_metadata)

        for source_candidate in candidate.source_map.values():
            self._merge_source_candidate(aggregated, source_candidate, requested_root_id)

        merged = aggregated[candidate.pair_key]
        if candidate.bundle_id and not merged.bundle_id:
            merged.bundle_id = candidate.bundle_id
        if candidate.bundle_price is not None and merged.bundle_price is None:
            merged.bundle_price = candidate.bundle_price
        if not merged.candidate_metadata and candidate.candidate_metadata:
            merged.candidate_metadata = dict(candidate.candidate_metadata)

    def _rank_candidates(
        self,
        *,
        aggregated: dict[tuple[int, int], AggregatedBundleCandidate],
        requested_root_id: int | None,
        user_profile: UserProfileFeatures,
        exact_user_pairs: set[tuple[int, int]],
        data: BundleData,
    ) -> list[RankedBundleCandidate]:
        ranked: list[RankedBundleCandidate] = []
        for aggregated_candidate in aggregated.values():
            scored = self._score_candidate(
                aggregated_candidate=aggregated_candidate,
                requested_root_id=requested_root_id,
                user_profile=user_profile,
                exact_user_pairs=exact_user_pairs,
                data=data,
            )
            if scored is not None:
                ranked.append(scored)

        ranked.sort(
            key=lambda item: (
                -item.final_score,
                -len(item.source_families),
                -len(item.sources),
                item.ordered_product_ids[0],
                item.ordered_product_ids[1],
            )
        )
        return ranked

    def _legacy_candidates_for_request(
        self,
        *,
        user_id: str,
        requested_root_id: int | None,
        root_ids: tuple[int, ...],
        limit: int,
        data: BundleData,
    ) -> tuple[BundleSourceCandidate, ...]:
        candidates: list[BundleSourceCandidate] = []
        user_records = data.user_bundle_index.get(user_id, ())
        if user_records:
            for record in user_records:
                if requested_root_id is not None and requested_root_id not in record.pair_key:
                    continue
                candidates.append(self._artifact_to_source_candidate(record, requested_root_id))
                if len(candidates) >= limit:
                    break

        top_seen: set[tuple[int, int]] = set()
        for root_id in root_ids:
            for record in data.top_bundle_index.get(root_id, ()):
                if requested_root_id is not None and requested_root_id not in record.pair_key:
                    continue
                if record.pair_key in top_seen:
                    continue
                top_seen.add(record.pair_key)
                candidates.append(self._artifact_to_source_candidate(record, requested_root_id))
                if len(top_seen) >= limit:
                    break

        fallback_records: list[BundleArtifactRecord] = []
        if requested_root_id is not None:
            fallback_records.extend(data.fallback_index.get(requested_root_id, ()))
        elif data.fallback_bundles:
            start = self._stable_rotation_offset(user_id, len(data.fallback_bundles))
            for offset in range(min(limit, len(data.fallback_bundles))):
                fallback_records.append(data.fallback_bundles[(start + offset) % len(data.fallback_bundles)])

        fallback_seen: set[tuple[int, int]] = set()
        for record in fallback_records:
            if record.pair_key in fallback_seen:
                continue
            fallback_seen.add(record.pair_key)
            candidates.append(self._artifact_to_source_candidate(record, requested_root_id))
            if len(fallback_seen) >= limit:
                break

        return tuple(candidates)

    def _artifact_to_source_candidate(
        self,
        record: BundleArtifactRecord,
        requested_root_id: int | None,
    ) -> BundleSourceCandidate:
        return BundleSourceCandidate(
            pair_key=record.pair_key,
            ordered_product_ids=self._preferred_pair_order(record.pair_key, record.ordered_product_ids, requested_root_id),
            source_name=record.source_name,
            source_family=record.source_family,
            source_score=float(record.quality_score),
            bundle_price=record.bundle_price,
            bundle_id=record.bundle_id,
            source_root_id=None if requested_root_id is None else requested_root_id,
            item_names=record.item_names,
            category_key=record.category_key,
            anchor_category=record.anchor_category,
            complement_category=record.complement_category,
            signals=dict(record.signals),
        )

    def _compatible_candidates(
        self,
        *,
        root_id: int,
        limit: int,
        data: BundleData,
    ) -> tuple[BundleSourceCandidate, ...]:
        response = self._compatible_engine.recommend(EngineRequest(root_product_id=str(root_id), limit=limit))
        if response.status != "ok":
            return ()

        root_product = data.products.get(root_id)
        root_name = f"Product {root_id}" if root_product is None else root_product.product_name
        root_category = "" if root_product is None else root_product.category
        candidates: list[BundleSourceCandidate] = []
        for item in response.items:
            metadata = dict(item.metadata)
            candidate_id = int(str(metadata.get("product_id") or "0"))
            if candidate_id <= 0:
                continue
            category_key = "|".join(sorted(part for part in (root_category, str(metadata.get("category") or "").strip().lower()) if part))
            candidates.append(
                BundleSourceCandidate(
                    pair_key=self._pair_key(root_id, candidate_id),
                    ordered_product_ids=(root_id, candidate_id),
                    source_name="compatible_products",
                    source_family="compatible",
                    source_score=float(item.score or 0.0),
                    bundle_price=None,
                    bundle_id=self._bundle_id_for_pair(root_id, candidate_id, data),
                    source_root_id=root_id,
                    item_names=(root_name, str(metadata.get("product_name") or f"Product {candidate_id}")),
                    category_key=category_key,
                    anchor_category=root_category,
                    complement_category=str(metadata.get("category") or "").strip().lower(),
                    signals=dict(metadata.get("signals") or {}),
                    reasons=tuple(str(reason) for reason in metadata.get("reasons", ()) if str(reason).strip()),
                )
            )
        return tuple(candidates)

    def _fbt_candidates(
        self,
        *,
        root_id: int,
        limit: int,
        data: BundleData,
    ) -> tuple[BundleSourceCandidate, ...]:
        response = self._fbt_engine.recommend(EngineRequest(root_product_id=str(root_id), limit=limit))
        if response.status != "ok":
            return ()

        root_product = data.products.get(root_id)
        root_name = f"Product {root_id}" if root_product is None else root_product.product_name
        root_category = "" if root_product is None else root_product.category
        candidates: list[BundleSourceCandidate] = []
        for item in response.items:
            metadata = dict(item.metadata)
            candidate_id = int(str(metadata.get("product_id") or "0"))
            if candidate_id <= 0:
                continue
            category_key = "|".join(sorted(part for part in (root_category, str(metadata.get("category") or "").strip().lower()) if part))
            candidates.append(
                BundleSourceCandidate(
                    pair_key=self._pair_key(root_id, candidate_id),
                    ordered_product_ids=(root_id, candidate_id),
                    source_name="frequently_bought_together",
                    source_family="fbt",
                    source_score=float(item.score or 0.0),
                    bundle_price=None,
                    bundle_id=self._bundle_id_for_pair(root_id, candidate_id, data),
                    source_root_id=root_id,
                    item_names=(root_name, str(metadata.get("product_name") or f"Product {candidate_id}")),
                    category_key=category_key,
                    anchor_category=root_category,
                    complement_category=str(metadata.get("category") or "").strip().lower(),
                    signals=dict(metadata.get("signals") or {}),
                    reasons=tuple(str(reason) for reason in metadata.get("reasons", ()) if str(reason).strip()),
                )
            )
        return tuple(candidates)

    def _merge_source_candidate(
        self,
        aggregated: dict[tuple[int, int], AggregatedBundleCandidate],
        candidate: BundleSourceCandidate,
        requested_root_id: int | None,
    ) -> None:
        entry = aggregated.get(candidate.pair_key)
        if entry is None:
            entry = AggregatedBundleCandidate(
                pair_key=candidate.pair_key,
                ordered_product_ids=self._preferred_pair_order(
                    candidate.pair_key,
                    candidate.ordered_product_ids,
                    requested_root_id,
                ),
            )
            aggregated[candidate.pair_key] = entry

        existing = entry.source_map.get(candidate.source_name)
        if existing is None or candidate.source_score > existing.source_score:
            entry.source_map[candidate.source_name] = candidate
        if candidate.bundle_id and not entry.bundle_id:
            entry.bundle_id = candidate.bundle_id
        if candidate.bundle_price is not None:
            if entry.bundle_price is None or candidate.source_name.startswith("legacy_"):
                entry.bundle_price = candidate.bundle_price
        entry.ordered_product_ids = self._preferred_pair_order(
            entry.pair_key,
            entry.ordered_product_ids if entry.ordered_product_ids else candidate.ordered_product_ids,
            requested_root_id,
        )

    def _score_candidate(
        self,
        *,
        aggregated_candidate: AggregatedBundleCandidate,
        requested_root_id: int | None,
        user_profile: UserProfileFeatures,
        exact_user_pairs: set[tuple[int, int]],
        data: BundleData,
    ) -> RankedBundleCandidate | None:
        items = self._resolve_items(aggregated_candidate, data)
        if items is None:
            return None
        if self._reject_candidate(items, aggregated_candidate, requested_root_id):
            return None

        sources = tuple(
            source_name
            for source_name, _ in sorted(
                aggregated_candidate.source_map.items(),
                key=lambda item: (
                    SOURCE_PRIORITY.get(item[0], 99),
                    item[0],
                ),
            )
        )
        source_candidates = tuple(aggregated_candidate.source_map[source_name] for source_name in sources)
        source_families = tuple(dict.fromkeys(source.source_family for source in source_candidates))
        source_quality_score = round(sum(self._source_component(source) for source in source_candidates), 4)
        consensus_bonus = round(self._consensus_bonus(source_candidates), 4)
        exact_user_pair = aggregated_candidate.pair_key in exact_user_pairs or user_profile.has_pair(aggregated_candidate.pair_key)
        user_fit_score, profile_signals = self._user_fit_score(
            items=items,
            pair_key=aggregated_candidate.pair_key,
            source_candidates=source_candidates,
            requested_root_id=requested_root_id,
            user_profile=user_profile,
        )
        user_fit_score = round(user_fit_score, 4)
        validity_score = round(
            self._validity_score(
                items=items,
                source_candidates=source_candidates,
                requested_root_id=requested_root_id,
            ),
            4,
        )
        freshness_score = round(
            self._freshness_score(
                pair_key=aggregated_candidate.pair_key,
                source_candidates=source_candidates,
                requested_root_id=requested_root_id,
                user_profile=user_profile,
            ),
            4,
        )
        generic_penalty = round(self._generic_penalty(items, source_candidates), 4)
        repetition_penalty = round(
            self._repetition_penalty(
                source_candidates=source_candidates,
                exact_user_pair=exact_user_pair,
            ),
            4,
        )
        fatigue_penalty = round(
            self._fatigue_penalty(
                items=items,
                requested_root_id=requested_root_id,
                user_profile=user_profile,
            ),
            4,
        )
        final_score = round(
            source_quality_score
            + consensus_bonus
            + user_fit_score
            + validity_score
            + freshness_score
            - generic_penalty
            - repetition_penalty,
            4,
        )
        final_score = round(
            final_score - fatigue_penalty,
            4,
        )
        if final_score < 8.5:
            return None

        source_details = {
            source.source_name: {
                "source_family": source.source_family,
                "source_score": round(source.source_score, 4),
                "signals": dict(source.signals),
            }
            for source in source_candidates
        }
        evidence = self._build_evidence(source_candidates)
        return RankedBundleCandidate(
            pair_key=aggregated_candidate.pair_key,
            ordered_product_ids=aggregated_candidate.ordered_product_ids,
            items=items,
            bundle_id=aggregated_candidate.bundle_id,
            bundle_price=aggregated_candidate.bundle_price,
            final_score=final_score,
            source_quality_score=source_quality_score,
            consensus_bonus=consensus_bonus,
            user_fit_score=user_fit_score,
            profile_signals={key: round(float(value), 4) for key, value in profile_signals.items()},
            validity_score=validity_score,
            freshness_score=freshness_score,
            generic_penalty=generic_penalty,
            repetition_penalty=repetition_penalty,
            fatigue_penalty=fatigue_penalty,
            exact_user_pair=exact_user_pair,
            sources=sources,
            source_families=source_families,
            source_details=source_details,
            evidence=evidence,
            candidate_metadata=dict(aggregated_candidate.candidate_metadata),
        )

    def _resolve_items(
        self,
        aggregated_candidate: AggregatedBundleCandidate,
        data: BundleData,
    ) -> tuple[BundleProductRecord, BundleProductRecord] | None:
        left_id, right_id = aggregated_candidate.ordered_product_ids
        left = data.products.get(int(left_id))
        right = data.products.get(int(right_id))
        if left is None or right is None:
            return None
        return left, right

    def _reject_candidate(
        self,
        items: tuple[BundleProductRecord, BundleProductRecord],
        aggregated_candidate: AggregatedBundleCandidate,
        requested_root_id: int | None,
    ) -> bool:
        if requested_root_id is not None and requested_root_id not in aggregated_candidate.pair_key:
            return True

        source_candidates = tuple(aggregated_candidate.source_map.values())
        has_live_support = any(source.source_family in {"compatible", "fbt"} for source in source_candidates)
        has_curated_support = any(source.source_name in {"legacy_user_bundle", "legacy_curated_bundle"} for source in source_candidates)
        duplicate_variant = self._looks_duplicate_variant(items[0], items[1])
        generic_utility_count = self._generic_utility_count(items)
        all_fallback = bool(source_candidates) and all(source.source_name == "legacy_fallback_bundle" for source in source_candidates)
        highest_quality = max((source.source_score for source in source_candidates), default=0.0)

        if duplicate_variant and not has_live_support and len(source_candidates) < 3:
            return True
        if generic_utility_count >= 1 and not has_live_support and len(source_candidates) < 2:
            return True
        if all_fallback and highest_quality < 65.0 and requested_root_id is None:
            return True
        if generic_utility_count >= 2 and not has_curated_support:
            return True
        return False

    def _source_component(self, source: BundleSourceCandidate) -> float:
        base = SOURCE_BASE_WEIGHTS.get(source.source_name, 6.0)
        if source.source_name == "compatible_products":
            signals = dict(source.signals)
            return (
                base
                + (source.source_score / 4.0)
                + (float(signals.get("rule_score", 0.0)) / 6.0)
                + (float(signals.get("use_case_score", 0.0)) / 8.0)
                + min(float(signals.get("pair_count", 0.0)), 25.0) * 0.05
            )
        if source.source_name == "frequently_bought_together":
            signals = dict(source.signals)
            lift = min(float(signals.get("lift", 0.0)), 4.0)
            confidence = min(float(signals.get("confidence", 0.0)), 1.0)
            pair_count = max(0.0, float(signals.get("cooccurrence_count", 0.0)))
            return base + (source.source_score / 5.0) + (2.6 * lift) + (8.0 * confidence) + (1.1 * math.log1p(pair_count))
        seen_count = min(float(source.signals.get("seen_count", 0.0)), 5.0)
        return base + (source.source_score / 7.0) + (0.45 * seen_count)

    def _consensus_bonus(self, source_candidates: tuple[BundleSourceCandidate, ...]) -> float:
        families = {source.source_family for source in source_candidates}
        bonus = 2.5 * max(0, len(families) - 1)
        bonus += 1.0 * max(0, len(source_candidates) - len(families))
        source_names = {source.source_name for source in source_candidates}
        if {"compatible_products", "frequently_bought_together"} <= source_names:
            bonus += 1.25
        if families >= {"legacy_bundle", "compatible"} or families >= {"legacy_bundle", "fbt"}:
            bonus += 0.75
        return bonus

    def _user_fit_score(
        self,
        *,
        items: tuple[BundleProductRecord, BundleProductRecord],
        pair_key: tuple[int, int],
        source_candidates: tuple[BundleSourceCandidate, ...],
        requested_root_id: int | None,
        user_profile: UserProfileFeatures,
    ) -> tuple[float, dict[str, float]]:
        if not user_profile.known_user:
            fallback_score = 0.65 if any(source.source_name == "legacy_user_bundle" for source in source_candidates) else 0.0
            return fallback_score, {
                "product_alignment": 0.0,
                "category_alignment": 0.0,
                "recency_alignment": 0.0,
                "archetype_alignment": 0.0,
                "root_context_alignment": 0.0,
                "history_support": round(fallback_score, 4),
            }

        product_alignment = 0.0
        category_alignment = 0.0
        recency_alignment = 0.0
        for item in items:
            item_weight = 0.55 if requested_root_id is not None and item.product_id == requested_root_id else 1.0
            product_alignment += min(user_profile.product_score(item.product_id) * 0.85 * item_weight, 2.1 * item_weight)
            recency_alignment += min(user_profile.recent_product_score(item.product_id) * 0.55 * item_weight, 1.35 * item_weight)
            if item.category:
                category_alignment += min(user_profile.category_score(item.category) * 0.38 * item_weight, 1.8 * item_weight)
                category_alignment += min(user_profile.recent_category_score(item.category) * 0.25 * item_weight, 1.2 * item_weight)
            if item.subcategory:
                category_alignment += min(user_profile.subcategory_score(item.subcategory) * 0.22 * item_weight, 0.9 * item_weight)
            if item.product_family:
                category_alignment += min(user_profile.family_score(item.product_family) * 0.2 * item_weight, 0.8 * item_weight)

        archetype_alignment = 0.0
        candidate_archetype = self._candidate_archetype_key(items)
        if candidate_archetype:
            archetype_alignment = min(user_profile.archetype_score(candidate_archetype) * 0.48, 1.9)

        root_context_alignment = 0.0
        if requested_root_id is not None and requested_root_id in pair_key:
            root_item = items[0] if items[0].product_id == requested_root_id else items[1]
            root_context_alignment += 0.85
            root_context_alignment += min(user_profile.product_score(requested_root_id) * 0.25, 0.95)
            if root_item.category:
                root_context_alignment += min(user_profile.category_score(root_item.category) * 0.2, 0.8)
                root_context_alignment += min(user_profile.recent_category_score(root_item.category) * 0.15, 0.6)

        history_support = 0.0
        if any(source.source_name == "legacy_user_bundle" for source in source_candidates):
            history_support += 0.3
        history_support += min(0.25 * sum(1 for product_id in pair_key if user_profile.has_product(product_id)), 0.45)

        score = min(
            product_alignment + category_alignment + recency_alignment + archetype_alignment + root_context_alignment + history_support,
            8.0,
        )
        return score, {
            "product_alignment": round(product_alignment, 4),
            "category_alignment": round(category_alignment, 4),
            "recency_alignment": round(recency_alignment, 4),
            "archetype_alignment": round(archetype_alignment, 4),
            "root_context_alignment": round(root_context_alignment, 4),
            "history_support": round(history_support, 4),
        }

    def _validity_score(
        self,
        *,
        items: tuple[BundleProductRecord, BundleProductRecord],
        source_candidates: tuple[BundleSourceCandidate, ...],
        requested_root_id: int | None,
    ) -> float:
        left, right = items
        score = 4.0
        if left.category and right.category and left.category != right.category:
            score += 1.4
        if requested_root_id is not None and requested_root_id in {left.product_id, right.product_id}:
            score += 1.0
        if any(source.source_name == "legacy_curated_bundle" for source in source_candidates):
            score += 0.8
        if any(source.source_name == "compatible_products" for source in source_candidates) and any(
            source.source_name == "frequently_bought_together" for source in source_candidates
        ):
            score += 0.9
        if self._looks_duplicate_variant(left, right):
            score -= 2.5
        score -= 0.75 * float(self._generic_utility_count(items))
        return score

    def _freshness_score(
        self,
        *,
        pair_key: tuple[int, int],
        source_candidates: tuple[BundleSourceCandidate, ...],
        requested_root_id: int | None,
        user_profile: UserProfileFeatures,
    ) -> float:
        score = 1.8
        if len(source_candidates) >= 3:
            score += 0.6
        if user_profile.known_user:
            novel_items = [product_id for product_id in pair_key if product_id not in user_profile.observed_product_ids]
            if requested_root_id is not None:
                novel_items = [product_id for product_id in novel_items if product_id != requested_root_id]
            if novel_items:
                score += min(0.3 * len(novel_items), 0.6)
            elif all(product_id in user_profile.observed_product_ids for product_id in pair_key):
                score -= 1.4
        if any(source.source_name == "legacy_fallback_bundle" for source in source_candidates) and len(source_candidates) == 1:
            score -= 0.8
        return score

    def _generic_penalty(
        self,
        items: tuple[BundleProductRecord, BundleProductRecord],
        source_candidates: tuple[BundleSourceCandidate, ...],
    ) -> float:
        penalty = 1.25 * float(self._generic_utility_count(items))
        if all(source.source_name == "legacy_fallback_bundle" for source in source_candidates):
            penalty += 0.75
        return penalty

    def _repetition_penalty(
        self,
        *,
        source_candidates: tuple[BundleSourceCandidate, ...],
        exact_user_pair: bool,
    ) -> float:
        penalty = 0.0
        if exact_user_pair:
            penalty += 3.0
        if any(source.source_name == "legacy_user_bundle" for source in source_candidates) and len(source_candidates) == 1:
            penalty += 0.9
        return penalty

    def _fatigue_penalty(
        self,
        *,
        items: tuple[BundleProductRecord, BundleProductRecord],
        requested_root_id: int | None,
        user_profile: UserProfileFeatures,
    ) -> float:
        if not user_profile.known_user:
            return 0.0

        penalty = 0.0
        for item in items:
            if requested_root_id is not None and item.product_id == requested_root_id:
                continue
            if item.product_id in user_profile.fatigued_product_ids:
                penalty += 0.25
            elif item.product_id in user_profile.observed_product_ids:
                penalty += 0.18
            if item.category and item.category in user_profile.fatigued_categories:
                penalty += 0.16
        return min(penalty, 0.95)

    def _candidate_archetype_key(self, items: tuple[BundleProductRecord, BundleProductRecord]) -> str:
        categories = [item.category or item.subcategory for item in items if item.category or item.subcategory]
        if len(categories) != 2:
            return ""
        return "|".join(sorted(categories))

    def _looks_duplicate_variant(self, left: BundleProductRecord, right: BundleProductRecord) -> bool:
        if left.product_family and right.product_family and left.product_family == right.product_family:
            return True
        if left.matched_ingredient and right.matched_ingredient and left.matched_ingredient == right.matched_ingredient:
            return True
        if left.category and right.category and left.category == right.category and left.subcategory and right.subcategory and left.subcategory == right.subcategory:
            if left.dominant_tokens & right.dominant_tokens:
                return True
        return False

    def _looks_generic_utility(self, product: BundleProductRecord) -> bool:
        if product.category in NONFOOD_CATEGORIES or product.subcategory in NONFOOD_CATEGORIES:
            return True
        if product.name_tokens & GENERIC_UTILITY_TOKENS:
            return True
        return bool(product.subcategory in {"water", "bags", "tissues"})

    def _generic_utility_count(self, items: tuple[BundleProductRecord, BundleProductRecord]) -> int:
        return sum(1 for item in items if self._looks_generic_utility(item))

    def _select_diverse_bundles(
        self,
        *,
        ranked: list[RankedBundleCandidate],
        limit: int,
        requested_root_id: int | None,
    ) -> list[RankedBundleCandidate]:
        target = max(1, int(limit))
        selected: list[RankedBundleCandidate] = []
        pair_seen: set[tuple[int, int]] = set()
        seen_items: set[int] = set()
        seen_non_root_items: set[int] = set()
        seen_categories: set[str] = set()

        for enforce_item_diversity, enforce_category_diversity in (
            (True, True),
            (True, False),
            (False, False),
        ):
            if len(selected) >= target:
                break
            for candidate in ranked:
                if len(selected) >= target:
                    break
                if candidate.pair_key in pair_seen:
                    continue
                candidate_ids = list(candidate.ordered_product_ids)
                if requested_root_id is not None:
                    non_root_ids = [product_id for product_id in candidate_ids if product_id != requested_root_id]
                    if enforce_item_diversity and any(product_id in seen_non_root_items for product_id in non_root_ids):
                        continue
                    complement_category = self._complement_category(candidate, requested_root_id)
                    if enforce_category_diversity and complement_category and complement_category in seen_categories:
                        continue
                else:
                    if enforce_item_diversity and any(product_id in seen_items for product_id in candidate_ids):
                        continue
                    complement_category = self._bundle_category_key(candidate)
                    if enforce_category_diversity and complement_category and complement_category in seen_categories:
                        continue

                selected.append(candidate)
                pair_seen.add(candidate.pair_key)
                seen_items.update(candidate_ids)
                if requested_root_id is not None:
                    for product_id in candidate_ids:
                        if product_id != requested_root_id:
                            seen_non_root_items.add(product_id)
                if complement_category:
                    seen_categories.add(complement_category)
        return selected[:target]

    def _complement_category(self, candidate: RankedBundleCandidate, requested_root_id: int) -> str:
        for item in candidate.items:
            if item.product_id != requested_root_id:
                return item.category or item.subcategory
        return ""

    def _bundle_category_key(self, candidate: RankedBundleCandidate) -> str:
        categories = sorted({item.category for item in candidate.items if item.category})
        return "|".join(categories)

    def _to_response_item(
        self,
        bundle: RankedBundleCandidate,
        user_id: str,
        requested_root_id: int | None,
    ) -> RecommendationCandidate:
        bundle_items = [
            {
                "product_id": str(item.product_id),
                "product_name": item.product_name,
                "category": item.category,
                "subcategory": item.subcategory,
                "product_family": item.product_family,
            }
            for item in bundle.items
        ]
        return RecommendationCandidate(
            product_ids=tuple(str(product_id) for product_id in bundle.ordered_product_ids),
            score=bundle.final_score,
            metadata={
                "user_id": user_id,
                "root_product_id": None if requested_root_id is None else str(requested_root_id),
                "bundle_id": bundle.bundle_id,
                "bundle_price": bundle.bundle_price,
                "retrieval_source": str(bundle.candidate_metadata.get("retrieval_source", "dynamic")),
                "candidate_metadata": dict(bundle.candidate_metadata),
                "bundle_items": bundle_items,
                "sources": list(bundle.sources),
                "source_families": list(bundle.source_families),
                "source_details": dict(bundle.source_details),
                "signals": {
                    "source_quality_score": bundle.source_quality_score,
                    "consensus_bonus": bundle.consensus_bonus,
                    "user_fit_score": bundle.user_fit_score,
                    "profile_signals": dict(bundle.profile_signals),
                    "validity_score": bundle.validity_score,
                    "freshness_score": bundle.freshness_score,
                    "generic_penalty": bundle.generic_penalty,
                    "repetition_penalty": bundle.repetition_penalty,
                    "fatigue_penalty": bundle.fatigue_penalty,
                    "exact_user_pair": bundle.exact_user_pair,
                },
                "evidence": list(bundle.evidence),
            },
        )

    def _build_evidence(self, source_candidates: tuple[BundleSourceCandidate, ...]) -> tuple[str, ...]:
        evidence: list[str] = []
        for source in source_candidates:
            if source.source_name == "frequently_bought_together":
                evidence.append(
                    f"fbt:pair_count={int(float(source.signals.get('cooccurrence_count', 0.0)))}:lift={float(source.signals.get('lift', 0.0)):.2f}"
                )
            elif source.source_name == "compatible_products":
                evidence.append(
                    f"compatible:rule={float(source.signals.get('rule_score', 0.0)):.2f}:use_case={float(source.signals.get('use_case_score', 0.0)):.2f}"
                )
            else:
                evidence.append(
                    f"{source.source_name}:quality={float(source.signals.get('quality_score', source.source_score)):.2f}"
                )
        return tuple(dict.fromkeys(evidence))

    def _pair_key(self, item_1_id: int, item_2_id: int) -> tuple[int, int]:
        left = int(item_1_id)
        right = int(item_2_id)
        return (left, right) if left <= right else (right, left)

    def _bundle_id_for_pair(self, item_1_id: int, item_2_id: int, data: BundleData) -> str:
        meta = data.bundle_meta_lookup.get(self._pair_key(item_1_id, item_2_id))
        return "" if meta is None else meta.bundle_id

    def _preferred_pair_order(
        self,
        pair_key: tuple[int, int],
        preferred_order: tuple[int, int],
        requested_root_id: int | None,
    ) -> tuple[int, int]:
        if requested_root_id is not None and requested_root_id in pair_key:
            other = pair_key[0] if pair_key[1] == requested_root_id else pair_key[1]
            return int(requested_root_id), int(other)
        if set(preferred_order) == set(pair_key):
            return int(preferred_order[0]), int(preferred_order[1])
        return int(pair_key[0]), int(pair_key[1])
