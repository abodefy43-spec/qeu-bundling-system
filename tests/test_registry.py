from engines.bundles.engine import PersonalizedBundlesEngine
from engines.fbt.engine import FrequentlyBoughtTogetherEngine
from engines.compatible.engine import CompatibleProductsEngine
from shared.contracts import EngineRequest
from shared.registry import execute_engine, list_engine_descriptors


def test_registry_lists_three_engines():
    descriptors = list_engine_descriptors()

    assert [descriptor.name for descriptor in descriptors] == [
        "compatible_products",
        "frequently_bought_together",
        "personalized_bundles",
    ]


def test_personalized_bundles_engine_is_registered_and_returns_aggregated_results(monkeypatch, bundle_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root))

    response = execute_engine(
        "personalized_bundles",
        EngineRequest.from_payload({"user_id": "101", "root_product_id": "100", "count": 2}),
    )

    assert response.status == "ok"
    assert response.engine == "personalized_bundles"
    assert response.metadata["user_id"] == "101"
    assert response.items[0].product_ids == ("100", "200")
    assert set(response.items[0].metadata["sources"]) == {
        "compatible_products",
        "frequently_bought_together",
        "legacy_user_bundle",
        "legacy_curated_bundle",
    }


def test_compatible_engine_is_registered_and_returns_results(monkeypatch, compatible_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(compatible_project_root))

    response = execute_engine(
        "compatible_products",
        EngineRequest(root_product_id="100", limit=3),
    )

    assert response.status == "ok"
    assert response.engine == "compatible_products"
    assert {item.metadata["product_name"] for item in response.items} == {
        "Coffee Creamer",
        "Whole Milk",
        "Butter Biscuits",
    }


def test_compatible_engine_suppresses_substitutes_and_weak_pairs(monkeypatch, compatible_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(compatible_project_root))
    engine = CompatibleProductsEngine()

    coffee_response = engine.recommend(EngineRequest(root_product_id="100", limit=5))
    coffee_names = [item.metadata["product_name"] for item in coffee_response.items]
    assert "Instant Coffee Classic" not in coffee_names
    assert "Chicken Breast" not in coffee_names

    dates_response = engine.recommend(EngineRequest(root_product_id="300", limit=5))
    dates_names = [item.metadata["product_name"] for item in dates_response.items]
    assert dates_names[:2] == ["Liquid Tahini", "Cream Cheese Spread"]
    assert "Medjool Dates Box" not in dates_names
    assert "Bottled Water" not in dates_names


def test_fbt_engine_is_registered_and_returns_results(monkeypatch, fbt_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(fbt_project_root))

    response = execute_engine(
        "frequently_bought_together",
        EngineRequest(root_product_id="100", limit=3),
    )

    assert response.status == "ok"
    assert response.engine == "frequently_bought_together"
    assert [item.metadata["product_name"] for item in response.items] == [
        "Coffee Creamer",
        "Whole Milk",
        "Butter Biscuits",
    ]
    assert all(item.metadata["signals"]["lift"] > 1.0 for item in response.items)


def test_fbt_engine_suppresses_popularity_artifacts_and_duplicate_variants(monkeypatch, fbt_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(fbt_project_root))
    engine = FrequentlyBoughtTogetherEngine()

    coffee_response = engine.recommend(EngineRequest(root_product_id="100", limit=5))
    coffee_names = [item.metadata["product_name"] for item in coffee_response.items]
    assert "Instant Coffee Classic" not in coffee_names
    assert "Chicken Breast" not in coffee_names

    dates_response = engine.recommend(EngineRequest(root_product_id="300", limit=5))
    dates_names = [item.metadata["product_name"] for item in dates_response.items]
    assert dates_names == ["Liquid Tahini", "Cream Cheese Spread"]
    assert "Medjool Dates Box" not in dates_names
    assert "Bottled Water" not in dates_names


def test_personalized_bundles_filters_weak_legacy_pairs_and_keeps_diverse_results(monkeypatch, bundle_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root))
    engine = PersonalizedBundlesEngine()

    response = engine.recommend(EngineRequest.from_payload({"user_id": "101", "root_product_id": "100", "count": 3}))

    pair_keys = [tuple(int(product_id) for product_id in item.product_ids) for item in response.items]
    assert pair_keys[0] == (100, 200)
    assert (100, 230) not in pair_keys
    assert set(pair_keys) == {(100, 200), (100, 210), (100, 220)}
    non_root_ids = [pair[1] for pair in pair_keys]
    assert len(non_root_ids) == len(set(non_root_ids))


def test_personalized_bundles_reranks_candidates_using_profile_fit(monkeypatch, bundle_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root))
    engine = PersonalizedBundlesEngine()

    response = engine.recommend(EngineRequest.from_payload({"user_id": "202", "root_product_id": "100", "count": 2}))

    assert response.status == "ok"
    assert response.items[0].product_ids == ("100", "210")
    assert response.items[0].metadata["signals"]["profile_signals"]["category_alignment"] > 0.0


def test_personalized_bundles_engine_serves_from_bundle_universe_when_available(
    monkeypatch,
    bundle_project_root_with_universe,
):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root_with_universe))

    response = execute_engine(
        "personalized_bundles",
        EngineRequest.from_payload({"user_id": "101", "root_product_id": "100", "count": 2}),
    )

    assert response.status == "ok"
    assert response.metadata["retrieval"]["mode"] == "bundle_universe"
    assert response.metadata["retrieval"]["bundle_universe_available"] is True
    assert "100" in response.metadata["retrieval"]["hooks"]["root_product_ids"]
    assert response.metadata["retrieval"]["retrieved_pair_count"] >= len(response.items)
    assert all("100" in item.product_ids for item in response.items)
    assert all(item.metadata["retrieval_source"] == "bundle_universe" for item in response.items)


def test_personalized_bundles_universe_reranking_uses_profile_fit(monkeypatch, bundle_project_root_with_universe):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root_with_universe))
    engine = PersonalizedBundlesEngine()

    response = engine.recommend(EngineRequest.from_payload({"user_id": "202", "root_product_id": "100", "count": 2}))

    assert response.status == "ok"
    assert response.metadata["retrieval"]["mode"] == "bundle_universe"
    assert response.items[0].product_ids == ("100", "210")
    assert response.items[0].metadata["candidate_metadata"]["quality_band"] in {"elite", "high", "medium", "tail"}
    assert response.items[0].metadata["signals"]["profile_signals"]["category_alignment"] > 0.0
