"""Production frequently-bought-together engine."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from data.paths import get_project_paths
from engines.fbt.data import FBTPairRecord, FBTProductRecord, load_fbt_data
from shared.contracts import EngineDescriptor, EngineRequest, EngineResponse, RecommendationCandidate


ENGINE_DESCRIPTOR = EngineDescriptor(
    name="frequently_bought_together",
    description="Order-level co-purchase engine for products empirically bought together.",
    required_inputs=("root_product_id",),
    output_description="Products repeatedly bought in the same baskets as the requested root product.",
)


@dataclass(frozen=True)
class FBTCandidateScore:
    product: FBTProductRecord
    final_score: float
    cooccurrence_count: int
    support: float
    confidence: float
    candidate_support: float
    lift: float
    pmi: float
    jaccard: float
    root_order_count: int
    candidate_order_count: int
    reasons: tuple[str, ...]


class FrequentlyBoughtTogetherEngine:
    def __init__(self, project_root: str | Path | None = None) -> None:
        self._project_root = Path(project_root).resolve() if project_root is not None else None

    descriptor = ENGINE_DESCRIPTOR

    def recommend(self, request: EngineRequest) -> EngineResponse:
        root_product_id = request.primary_product_id
        if not root_product_id:
            return EngineResponse(
                engine=self.descriptor.name,
                status="invalid_request",
                message="root_product_id is required for frequently_bought_together.",
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

        data = load_fbt_data(project_root=self._resolved_project_root())
        if data.total_orders <= 0 or not data.pair_index:
            return EngineResponse(
                engine=self.descriptor.name,
                status="not_ready",
                message="FBT data sources are unavailable or empty.",
                metadata={"missing_sources": list(data.missing_sources)},
            )

        root_product = data.products.get(root_id)
        root_order_count = data.product_order_counts.get(root_id, 0)
        if root_product is None or root_order_count <= 0:
            return EngineResponse(
                engine=self.descriptor.name,
                status="not_found",
                message=f"Unknown root product id: {root_id}",
                metadata={
                    "root_product_id": str(root_id),
                    "missing_sources": list(data.missing_sources),
                },
            )

        ranked = self._rank_candidates(root_product=root_product, root_order_count=root_order_count)
        items = tuple(self._to_candidate(root_product, candidate) for candidate in ranked[: request.limit])
        return EngineResponse(
            engine=self.descriptor.name,
            status="ok",
            message=f"Ranked {len(items)} FBT products for root product {root_id}.",
            items=items,
            metadata={
                "root_product_id": str(root_product.product_id),
                "root_product_name": root_product.product_name,
                "requested_count": request.limit,
                "returned_count": len(items),
                "total_orders": data.total_orders,
                "root_order_count": root_order_count,
                "data_sources": {
                    "order_items": "data/raw/order_items.csv",
                    "pair_counts": "data/processed/copurchase_scores.csv",
                    "product_catalog": "data/processed/product_categories.csv",
                },
                "missing_sources": list(data.missing_sources),
                "scoring_version": "fbt_v1",
            },
        )

    def _resolved_project_root(self) -> Path | None:
        if self._project_root is not None:
            return self._project_root
        return get_project_paths().root

    def _rank_candidates(self, root_product: FBTProductRecord, root_order_count: int) -> list[FBTCandidateScore]:
        data = load_fbt_data(project_root=self._resolved_project_root())
        scored: list[FBTCandidateScore] = []
        for pair in data.pair_index.get(root_product.product_id, ()):
            candidate = data.products.get(pair.candidate_id)
            if candidate is None:
                continue
            candidate_score = self._score_candidate(
                root_product=root_product,
                root_order_count=root_order_count,
                candidate=candidate,
                pair=pair,
            )
            if candidate_score is not None:
                scored.append(candidate_score)
        return sorted(
            scored,
            key=lambda item: (-item.final_score, -item.lift, -item.cooccurrence_count, item.product.product_id),
        )

    def _score_candidate(
        self,
        root_product: FBTProductRecord,
        root_order_count: int,
        candidate: FBTProductRecord,
        pair: FBTPairRecord,
    ) -> FBTCandidateScore | None:
        data = load_fbt_data(project_root=self._resolved_project_root())
        if candidate.product_id == root_product.product_id:
            return None

        candidate_order_count = data.product_order_counts.get(candidate.product_id, 0)
        if candidate_order_count <= 0 or pair.cooccurrence_count <= 0:
            return None

        total_orders = max(1, data.total_orders)
        cooccurrence_count = int(pair.cooccurrence_count)
        support = cooccurrence_count / total_orders
        confidence = cooccurrence_count / max(1, root_order_count)
        candidate_support = candidate_order_count / total_orders
        if candidate_support <= 0:
            return None
        lift = confidence / candidate_support
        if lift <= 0:
            return None
        pmi = math.log(lift, 2)
        union_orders = max(1, root_order_count + candidate_order_count - cooccurrence_count)
        jaccard = cooccurrence_count / union_orders

        if self._looks_duplicate_variant(
            root_product=root_product,
            root_order_count=root_order_count,
            candidate=candidate,
            candidate_order_count=candidate_order_count,
            lift=lift,
        ):
            return None

        if not self._passes_minimum_signal(
            cooccurrence_count=cooccurrence_count,
            confidence=confidence,
            candidate_support=candidate_support,
            lift=lift,
            jaccard=jaccard,
            root_order_count=root_order_count,
        ):
            return None

        stability = 3.5 * math.log1p(cooccurrence_count)
        confidence_score = 40.0 * confidence
        lift_score = 20.0 * max(0.0, min(lift, 4.0) - 1.0)
        jaccard_score = 25.0 * jaccard
        pmi_score = 8.0 * max(0.0, min(pmi, 2.5))
        final_score = stability + confidence_score + lift_score + jaccard_score + pmi_score

        reasons = [
            f"pair_count:{cooccurrence_count}",
            f"confidence:{confidence:.4f}",
            f"lift:{lift:.4f}",
        ]
        return FBTCandidateScore(
            product=candidate,
            final_score=round(final_score, 4),
            cooccurrence_count=cooccurrence_count,
            support=round(support, 6),
            confidence=round(confidence, 6),
            candidate_support=round(candidate_support, 6),
            lift=round(lift, 6),
            pmi=round(pmi, 6),
            jaccard=round(jaccard, 6),
            root_order_count=root_order_count,
            candidate_order_count=candidate_order_count,
            reasons=tuple(reasons),
        )

    def _passes_minimum_signal(
        self,
        *,
        cooccurrence_count: int,
        confidence: float,
        candidate_support: float,
        lift: float,
        jaccard: float,
        root_order_count: int,
    ) -> bool:
        if root_order_count < 10:
            min_pair_count = 2
        elif root_order_count < 100:
            min_pair_count = 3
        else:
            min_pair_count = 5

        if cooccurrence_count < min_pair_count:
            return False
        if lift < 1.05:
            return False
        if confidence < 0.02:
            return False
        if cooccurrence_count < 4 and lift < 1.2:
            return False
        if cooccurrence_count < 4 and jaccard < 0.015:
            return False
        if cooccurrence_count < 3 and candidate_support >= 0.25 and lift < 2.0:
            return False
        return True

    def _looks_duplicate_variant(
        self,
        *,
        root_product: FBTProductRecord,
        root_order_count: int,
        candidate: FBTProductRecord,
        candidate_order_count: int,
        lift: float,
    ) -> bool:
        if root_product.product_family and candidate.product_family and root_product.product_family == candidate.product_family:
            return True
        if (
            root_product.category
            and candidate.category
            and root_product.category == candidate.category
            and root_product.subcategory
            and candidate.subcategory
            and root_product.subcategory == candidate.subcategory
        ):
            if root_product.matched_ingredient and candidate.matched_ingredient and root_product.matched_ingredient == candidate.matched_ingredient:
                return True
            shared_dominant_tokens = root_product.dominant_tokens & candidate.dominant_tokens
            if shared_dominant_tokens and candidate_order_count >= root_order_count and lift < 1.6:
                return True
        return False

    def _to_candidate(self, root_product: FBTProductRecord, candidate: FBTCandidateScore) -> RecommendationCandidate:
        return RecommendationCandidate(
            product_ids=(str(candidate.product.product_id),),
            score=round(candidate.final_score, 4),
            metadata={
                "product_id": str(candidate.product.product_id),
                "product_name": candidate.product.product_name,
                "category": candidate.product.category,
                "subcategory": candidate.product.subcategory,
                "product_family": candidate.product.product_family,
                "root_product_id": str(root_product.product_id),
                "signals": {
                    "cooccurrence_count": candidate.cooccurrence_count,
                    "support": candidate.support,
                    "confidence": candidate.confidence,
                    "candidate_support": candidate.candidate_support,
                    "lift": candidate.lift,
                    "pmi": candidate.pmi,
                    "jaccard": candidate.jaccard,
                    "root_order_count": candidate.root_order_count,
                    "candidate_order_count": candidate.candidate_order_count,
                },
                "reasons": list(candidate.reasons),
            },
        )
