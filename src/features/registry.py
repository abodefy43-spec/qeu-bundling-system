"""Planned feature sets for the three engine families."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    owner_engine: str
    source: str
    status: str
    description: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "owner_engine": self.owner_engine,
            "source": self.source,
            "status": self.status,
            "description": self.description,
        }


FEATURE_SPECS = (
    FeatureSpec(
        name="catalog_attributes",
        owner_engine="compatible_products",
        source="data/reference",
        status="planned",
        description="Catalog normalization, product families, and compatibility rule inputs.",
    ),
    FeatureSpec(
        name="transaction_history",
        owner_engine="frequently_bought_together",
        source="data/raw",
        status="planned",
        description="Basket-level transaction aggregates for co-purchase modeling.",
    ),
    FeatureSpec(
        name="customer_profiles",
        owner_engine="personalized_bundles",
        source="data/processed/filtered_orders.pkl",
        status="active",
        description="Deterministic user/order behavior features for personalized bundle ranking.",
    ),
    FeatureSpec(
        name="bundle_universe",
        owner_engine="personalized_bundles",
        source="data/processed/features/bundle_universe.parquet",
        status="active",
        description="Offline materialized bundle pool for later user-level reranking.",
    ),
)


def list_feature_specs() -> list[FeatureSpec]:
    return list(FEATURE_SPECS)
