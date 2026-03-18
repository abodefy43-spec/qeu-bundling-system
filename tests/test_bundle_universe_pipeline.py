from pathlib import Path

import pandas as pd

from pipelines.bundle_universe import build_bundle_universe


def test_bundle_universe_pipeline_writes_curated_parquet_artifact(bundle_project_root: Path):
    result = build_bundle_universe(
        project_root=bundle_project_root,
        target_size=5,
        per_root_limit=4,
    )

    assert result.artifact_path.exists()
    assert result.report_path.exists()

    frame = pd.read_parquet(result.artifact_path)
    assert len(frame) == result.report["selected_count"]
    assert frame["bundle_id"].is_unique
    assert not frame[["item_1_id", "item_2_id"]].duplicated().any()
    assert {
        "bundle_id",
        "item_1_id",
        "item_2_id",
        "source_names",
        "source_families",
        "quality_score",
        "category_pair",
        "archetype",
        "genericity_penalty",
        "evidence_signals",
        "freshness_metadata",
        "quality_band",
    }.issubset(frame.columns)

    pair_keys = {(int(row.item_1_id), int(row.item_2_id)) for row in frame.itertuples(index=False)}
    assert (100, 210) in pair_keys
    assert (300, 310) in pair_keys
    assert (300, 340) not in pair_keys


def test_bundle_universe_pipeline_enforces_diversity_and_preserves_sources(bundle_project_root: Path):
    result = build_bundle_universe(
        project_root=bundle_project_root,
        target_size=4,
        per_root_limit=4,
    )

    frame = pd.read_parquet(result.artifact_path)
    assert frame["category_pair"].nunique() >= 3

    coffee_milk = frame[(frame["item_1_id"] == 100) & (frame["item_2_id"] == 210)].iloc[0]
    assert "compatible_products" in coffee_milk["source_names"]
    assert "frequently_bought_together" in coffee_milk["source_names"]
    assert "legacy_curated_bundle" in coffee_milk["source_names"]
    assert bool(coffee_milk["has_live_support"]) is True
    assert bool(coffee_milk["is_valid"]) is True
