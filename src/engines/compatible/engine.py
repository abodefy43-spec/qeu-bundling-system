"""Production compatible-products engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from data.paths import get_project_paths
from engines.compatible.data import (
    GENERIC_TAGS,
    NONFOOD_TOKENS,
    PairPenaltyRule,
    PairSignal,
    ProductRecord,
    load_compatible_products_data,
)
from engines.compatible.rules import COMPATIBILITY_PROFILES, CompatibilityProfile
from shared.contracts import EngineDescriptor, EngineRequest, EngineResponse, RecommendationCandidate


ENGINE_DESCRIPTOR = EngineDescriptor(
    name="compatible_products",
    description="Compatibility engine for products that naturally go well together.",
    required_inputs=("root_product_id",),
    output_description="Top companion products that complement the requested root product.",
)


@dataclass(frozen=True)
class CandidateScore:
    product: ProductRecord
    final_score: float
    rule_score: float
    use_case_score: float
    copurchase_score: float
    quality_score: float
    penalty_multiplier: float
    pair_count: int
    pair_score: float
    reasons: tuple[str, ...]


class CompatibleProductsEngine:
    def __init__(self, project_root: str | Path | None = None) -> None:
        self._project_root = Path(project_root).resolve() if project_root is not None else None

    descriptor = ENGINE_DESCRIPTOR

    def recommend(self, request: EngineRequest) -> EngineResponse:
        root_product_id = request.primary_product_id
        if not root_product_id:
            return EngineResponse(
                engine=self.descriptor.name,
                status="invalid_request",
                message="root_product_id is required for compatible_products.",
                metadata={"received_request": request.as_dict()},
            )

        try:
            root_id = int(str(root_product_id).strip())
        except ValueError:
            return EngineResponse(
                engine=self.descriptor.name,
                status="invalid_request",
                message="root_product_id must be numeric.",
                metadata={"received_request": request.as_dict()},
            )

        data = load_compatible_products_data(project_root=self._resolved_project_root())
        root_product = data.products.get(root_id)
        if root_product is None:
            return EngineResponse(
                engine=self.descriptor.name,
                status="not_found",
                message=f"Unknown root product id: {root_id}",
                metadata={
                    "root_product_id": str(root_id),
                    "missing_sources": list(data.missing_sources),
                },
            )

        scored = self._rank_candidates(root_product=root_product, count=request.limit, data_root=self._resolved_project_root())
        items = tuple(self._candidate_to_response_item(root_product, candidate) for candidate in scored[: request.limit])
        return EngineResponse(
            engine=self.descriptor.name,
            status="ok",
            message=f"Ranked {len(items)} compatible products for root product {root_id}.",
            items=items,
            metadata={
                "root_product_id": str(root_product.product_id),
                "root_product_name": root_product.product_name,
                "requested_count": request.limit,
                "returned_count": len(items),
                "data_sources": {
                    "product_catalog": "data/processed/product_categories.csv",
                    "copurchase_pairs": "data/processed/copurchase_scores.csv",
                    "recipe_scores": "data/processed/product_recipe_scores.csv",
                    "recipe_reference": "data/reference/recipe_data.json",
                    "pair_penalties": "data/reference/pair_penalty_rules.json",
                },
                "missing_sources": list(data.missing_sources),
                "scoring_version": "compatible_products_v1",
            },
        )

    def _resolved_project_root(self) -> Path | None:
        if self._project_root is not None:
            return self._project_root
        return get_project_paths().root

    def _rank_candidates(self, root_product: ProductRecord, count: int, data_root: Path | None) -> list[CandidateScore]:
        data = load_compatible_products_data(project_root=data_root)
        profiles = self._profiles_for_root(root_product)
        pair_lookup = {
            signal.candidate_id: signal
            for signal in data.copurchase_index.get(root_product.product_id, ())
        }
        candidate_ids = set(pair_lookup)
        if profiles:
            for product_id, candidate in data.products.items():
                if product_id == root_product.product_id:
                    continue
                if self._matches_any_profile_candidate(candidate, profiles):
                    candidate_ids.add(product_id)
        else:
            root_specific_tags = self._specific_tags(root_product)
            for product_id, candidate in data.products.items():
                if product_id == root_product.product_id:
                    continue
                if len(root_specific_tags & self._specific_tags(candidate)) >= 2:
                    candidate_ids.add(product_id)

        scored: list[CandidateScore] = []
        for candidate_id in candidate_ids:
            candidate = data.products.get(candidate_id)
            if candidate is None:
                continue
            score = self._score_candidate(
                root_product=root_product,
                candidate=candidate,
                pair_signal=pair_lookup.get(candidate_id),
                profiles=profiles,
                data_root=data_root,
            )
            if score is not None:
                scored.append(score)

        return sorted(
            scored,
            key=lambda item: (-item.final_score, -item.pair_count, -item.copurchase_score, item.product.product_id),
        )[: max(count * 3, count)]

    def _profiles_for_root(self, root_product: ProductRecord) -> tuple[CompatibilityProfile, ...]:
        matched = []
        for profile in COMPATIBILITY_PROFILES:
            if (
                root_product.name_tokens & profile.trigger_tokens
                or root_product.category_tags & profile.trigger_tags
                or ({root_product.matched_ingredient} & profile.trigger_ingredients if root_product.matched_ingredient else set())
                or ({root_product.subcategory} & profile.trigger_subcategories if root_product.subcategory else set())
                or ({root_product.product_family} & profile.trigger_families if root_product.product_family else set())
            ):
                matched.append(profile)
        return tuple(matched)

    def _matches_any_profile_candidate(
        self,
        candidate: ProductRecord,
        profiles: tuple[CompatibilityProfile, ...],
    ) -> bool:
        for profile in profiles:
            if (
                candidate.name_tokens & profile.positive_tokens
                or candidate.category_tags & profile.positive_tags
                or (candidate.category and candidate.category in profile.positive_categories)
                or (candidate.subcategory and candidate.subcategory in profile.positive_subcategories)
                or (candidate.product_family and candidate.product_family in profile.positive_families)
                or (candidate.matched_ingredient and candidate.matched_ingredient in profile.positive_ingredients)
            ):
                return True
        return False

    def _score_candidate(
        self,
        root_product: ProductRecord,
        candidate: ProductRecord,
        pair_signal: PairSignal | None,
        profiles: tuple[CompatibilityProfile, ...],
        data_root: Path | None,
    ) -> CandidateScore | None:
        if candidate.product_id == root_product.product_id:
            return None
        if self._looks_nonfood(candidate):
            return None

        explicit_match = False
        rule_score = 0.0
        reasons: list[str] = []
        for profile in profiles:
            score, is_explicit, profile_reasons = self._profile_score(profile, candidate)
            if score > rule_score:
                rule_score = score
            if is_explicit:
                explicit_match = True
            reasons.extend(profile_reasons)

        use_case_score, use_case_reasons = self._use_case_score(root_product, candidate, explicit_match, data_root)
        reasons.extend(use_case_reasons)
        copurchase_score = self._copurchase_score(pair_signal)
        if pair_signal is not None:
            reasons.append(f"pair_support:{pair_signal.pair_count}")
        quality_score = self._quality_score(candidate, data_root)

        if self._reject_same_family(root_product, candidate, explicit_match):
            return None
        if self._reject_same_subcategory(root_product, candidate, explicit_match):
            return None
        if self._reject_same_ingredient(root_product, candidate, explicit_match):
            return None
        if self._reject_weak_same_category(root_product, candidate, explicit_match, pair_signal):
            return None

        if not explicit_match and use_case_score <= 0.0 and not self._strong_pair_signal(pair_signal):
            return None

        if rule_score <= 0.0 and use_case_score <= 0.0 and copurchase_score < 4.0:
            return None

        raw_score = rule_score + use_case_score + copurchase_score + quality_score
        if root_product.category != candidate.category:
            raw_score += 2.0

        penalty_multiplier, penalty_reasons = self._penalty_multiplier(root_product, candidate, data_root)
        reasons.extend(penalty_reasons)
        final_score = raw_score * penalty_multiplier
        if final_score < 8.0:
            return None

        return CandidateScore(
            product=candidate,
            final_score=round(final_score, 4),
            rule_score=round(rule_score, 4),
            use_case_score=round(use_case_score, 4),
            copurchase_score=round(copurchase_score, 4),
            quality_score=round(quality_score, 4),
            penalty_multiplier=round(penalty_multiplier, 4),
            pair_count=0 if pair_signal is None else int(pair_signal.pair_count),
            pair_score=0.0 if pair_signal is None else float(pair_signal.score),
            reasons=tuple(dict.fromkeys(reason for reason in reasons if reason)),
        )

    def _profile_score(
        self,
        profile: CompatibilityProfile,
        candidate: ProductRecord,
    ) -> tuple[float, bool, list[str]]:
        score = 0.0
        reasons: list[str] = []
        token_hits = sorted(candidate.name_tokens & profile.positive_tokens)
        tag_hits = sorted(candidate.category_tags & profile.positive_tags)
        category_hit = candidate.category in profile.positive_categories if candidate.category else False
        subcategory_hit = candidate.subcategory in profile.positive_subcategories if candidate.subcategory else False
        family_hit = candidate.product_family in profile.positive_families if candidate.product_family else False
        ingredient_hit = candidate.matched_ingredient in profile.positive_ingredients if candidate.matched_ingredient else False

        if token_hits:
            score += min(18.0, 8.0 * len(token_hits))
            reasons.append(f"profile:{profile.name}:token:{','.join(token_hits[:3])}")
        priority_token_hits = sorted(set(token_hits) & set(profile.priority_tokens))
        if priority_token_hits:
            score += min(8.0, 4.0 * len(priority_token_hits))
            reasons.append(f"profile:{profile.name}:priority:{','.join(priority_token_hits[:3])}")
        if tag_hits:
            score += min(12.0, 4.0 * len(tag_hits))
            reasons.append(f"profile:{profile.name}:tag:{','.join(tag_hits[:3])}")
        if category_hit:
            score += 3.0
        if subcategory_hit:
            score += 5.0
        if family_hit:
            score += 5.0
        if ingredient_hit:
            score += 5.0

        explicit_match = bool(
            token_hits
            or tag_hits
            or subcategory_hit
            or family_hit
            or ingredient_hit
        )
        return score, explicit_match, reasons

    def _use_case_score(
        self,
        root_product: ProductRecord,
        candidate: ProductRecord,
        explicit_match: bool,
        data_root: Path | None,
    ) -> tuple[float, list[str]]:
        data = load_compatible_products_data(project_root=data_root)
        reasons: list[str] = []
        score = 0.0

        shared_tags = sorted(self._specific_tags(root_product) & self._specific_tags(candidate))
        if shared_tags:
            score += min(12.0, 3.5 * len(shared_tags))
            reasons.append(f"shared_tags:{','.join(shared_tags[:3])}")

        if root_product.matched_ingredient and candidate.matched_ingredient:
            root_recipes = data.ingredient_recipe_index.get(root_product.matched_ingredient, frozenset())
            candidate_recipes = data.ingredient_recipe_index.get(candidate.matched_ingredient, frozenset())
            overlap = sorted(root_recipes & candidate_recipes)
            if overlap and root_product.matched_ingredient != candidate.matched_ingredient:
                score += min(10.0, 2.0 * len(overlap))
                reasons.append(f"recipe_overlap:{','.join(overlap[:3])}")
            elif overlap and explicit_match:
                score += min(4.0, 1.0 * len(overlap))

        root_overlap = root_product.dominant_tokens & candidate.dominant_tokens
        if root_overlap and not explicit_match:
            score -= min(10.0, 3.0 * len(root_overlap))
        return score, reasons

    def _copurchase_score(self, pair_signal: PairSignal | None) -> float:
        if pair_signal is None:
            return 0.0
        pair_component = 1.6 * math.log1p(max(0, pair_signal.pair_count))
        score_component = 2.0 * min(max(pair_signal.score, 0.0), 3.0)
        return min(12.0, pair_component + score_component)

    def _quality_score(self, candidate: ProductRecord, data_root: Path | None) -> float:
        data = load_compatible_products_data(project_root=data_root)
        importance = 0.0
        for key in (candidate.matched_ingredient, candidate.subcategory, candidate.category):
            if key and key in data.category_importance:
                importance = max(importance, data.category_importance[key])
        score = min(6.0, (candidate.recipe_score / 25.0) + (candidate.frequency_score / 30.0) + (importance / 18.0))
        return max(0.0, score)

    def _reject_same_family(
        self,
        root_product: ProductRecord,
        candidate: ProductRecord,
        explicit_match: bool,
    ) -> bool:
        return bool(
            root_product.product_family
            and candidate.product_family
            and root_product.product_family == candidate.product_family
            and not explicit_match
        )

    def _reject_same_subcategory(
        self,
        root_product: ProductRecord,
        candidate: ProductRecord,
        explicit_match: bool,
    ) -> bool:
        return bool(
            root_product.subcategory
            and candidate.subcategory
            and root_product.subcategory == candidate.subcategory
            and not explicit_match
        )

    def _reject_same_ingredient(
        self,
        root_product: ProductRecord,
        candidate: ProductRecord,
        explicit_match: bool,
    ) -> bool:
        return bool(
            root_product.matched_ingredient
            and candidate.matched_ingredient
            and root_product.matched_ingredient == candidate.matched_ingredient
            and not explicit_match
        )

    def _reject_weak_same_category(
        self,
        root_product: ProductRecord,
        candidate: ProductRecord,
        explicit_match: bool,
        pair_signal: PairSignal | None,
    ) -> bool:
        if root_product.category != candidate.category:
            return False
        if explicit_match:
            return False
        return not self._strong_pair_signal(pair_signal)

    def _strong_pair_signal(self, pair_signal: PairSignal | None) -> bool:
        if pair_signal is None:
            return False
        return bool(pair_signal.pair_count >= 25 or pair_signal.score >= 3.0)

    def _penalty_multiplier(
        self,
        root_product: ProductRecord,
        candidate: ProductRecord,
        data_root: Path | None,
    ) -> tuple[float, list[str]]:
        data = load_compatible_products_data(project_root=data_root)
        root_terms = set(root_product.name_tokens) | set(root_product.category_tags)
        candidate_terms = set(candidate.name_tokens) | set(candidate.category_tags)
        multiplier = 1.0
        reasons: list[str] = []
        for rule in data.pair_penalty_rules:
            if self._matches_pair_penalty_rule(rule, root_terms, candidate_terms):
                multiplier *= rule.multiplier
                reasons.append(f"penalty:{rule.name}")
        return max(0.3, multiplier), reasons

    def _matches_pair_penalty_rule(
        self,
        rule: PairPenaltyRule,
        root_terms: set[str],
        candidate_terms: set[str],
    ) -> bool:
        if not rule.anchor_terms or not rule.complement_terms:
            return False
        return bool(
            (root_terms & set(rule.anchor_terms) and candidate_terms & set(rule.complement_terms))
            or (candidate_terms & set(rule.anchor_terms) and root_terms & set(rule.complement_terms))
        )

    def _looks_nonfood(self, candidate: ProductRecord) -> bool:
        candidate_tokens = set(candidate.name_tokens) | set(candidate.category_tags)
        return bool(candidate_tokens & NONFOOD_TOKENS)

    def _specific_tags(self, product: ProductRecord) -> set[str]:
        return {tag for tag in product.category_tags if tag not in GENERIC_TAGS}

    def _candidate_to_response_item(
        self,
        root_product: ProductRecord,
        candidate: CandidateScore,
    ) -> RecommendationCandidate:
        return RecommendationCandidate(
            product_ids=(str(candidate.product.product_id),),
            score=round(candidate.final_score, 4),
            metadata={
                "product_id": str(candidate.product.product_id),
                "product_name": candidate.product.product_name,
                "category": candidate.product.category,
                "subcategory": candidate.product.subcategory,
                "product_family": candidate.product.product_family,
                "matched_ingredient": candidate.product.matched_ingredient,
                "root_product_id": str(root_product.product_id),
                "signals": {
                    "rule_score": candidate.rule_score,
                    "use_case_score": candidate.use_case_score,
                    "copurchase_score": candidate.copurchase_score,
                    "quality_score": candidate.quality_score,
                    "pair_count": candidate.pair_count,
                    "pair_score": candidate.pair_score,
                    "penalty_multiplier": candidate.penalty_multiplier,
                },
                "reasons": list(candidate.reasons),
            },
        )
