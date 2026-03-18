from api.app import create_app


def test_healthz_returns_ok():
    client = create_app().test_client()
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok", "phase": "phase0_cleanup"}


def test_engine_recommendation_endpoint_returns_ranked_compatible_products(monkeypatch, compatible_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(compatible_project_root))
    client = create_app().test_client()
    response = client.post(
        "/engines/compatible_products/recommendations",
        json={"root_product_id": "100", "count": 3},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["engine"] == "compatible_products"
    assert payload["status"] == "ok"
    assert payload["metadata"]["root_product_id"] == "100"
    assert len(payload["items"]) == 3
    returned_names = {item["metadata"]["product_name"] for item in payload["items"]}
    assert returned_names == {"Coffee Creamer", "Whole Milk", "Butter Biscuits"}
    assert all(item["metadata"]["signals"]["pair_count"] >= 7 for item in payload["items"])


def test_engine_recommendation_endpoint_returns_404_for_unknown_root(monkeypatch, compatible_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(compatible_project_root))
    client = create_app().test_client()
    response = client.post(
        "/engines/compatible_products/recommendations",
        json={"root_product_id": "999"},
    )

    assert response.status_code == 404
    assert response.get_json()["status"] == "not_found"


def test_unknown_engine_returns_404():
    client = create_app().test_client()
    response = client.post("/engines/unknown/recommendations", json={})

    assert response.status_code == 404


def test_fbt_endpoint_returns_ranked_order_associations(monkeypatch, fbt_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(fbt_project_root))
    client = create_app().test_client()
    response = client.post(
        "/engines/frequently_bought_together/recommendations",
        json={"root_product_id": "100", "count": 3},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["engine"] == "frequently_bought_together"
    assert payload["status"] == "ok"
    assert payload["metadata"]["root_product_id"] == "100"
    assert [item["metadata"]["product_name"] for item in payload["items"]] == [
        "Coffee Creamer",
        "Whole Milk",
        "Butter Biscuits",
    ]
    first_signals = payload["items"][0]["metadata"]["signals"]
    assert first_signals["cooccurrence_count"] == 4
    assert round(first_signals["confidence"], 2) == 0.8
    assert first_signals["lift"] > 1.0


def test_fbt_endpoint_returns_404_for_unknown_root(monkeypatch, fbt_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(fbt_project_root))
    client = create_app().test_client()
    response = client.post(
        "/engines/frequently_bought_together/recommendations",
        json={"root_product_id": "999"},
    )

    assert response.status_code == 404
    assert response.get_json()["status"] == "not_found"


def test_personalized_bundles_endpoint_returns_structured_bundles(monkeypatch, bundle_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root))
    client = create_app().test_client()
    response = client.post(
        "/engines/personalized_bundles/recommendations",
        json={"user_id": "101", "root_product_id": "100", "count": 2},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["engine"] == "personalized_bundles"
    assert payload["status"] == "ok"
    assert payload["metadata"]["user_id"] == "101"
    assert payload["metadata"]["profile"]["known_user"] is True
    assert payload["metadata"]["profile_store"]["source_path"] == "data/processed/filtered_orders.pkl"
    assert payload["metadata"]["retrieval"]["mode"] == "dynamic_fallback"
    assert payload["metadata"]["retrieval"]["fallback_used"] is True
    assert payload["items"][0]["product_ids"] == ["100", "200"]
    assert payload["items"][0]["metadata"]["bundle_items"][0]["product_name"] == "Arabic Coffee Blend"
    assert set(payload["items"][0]["metadata"]["sources"]) == {
        "compatible_products",
        "frequently_bought_together",
        "legacy_user_bundle",
        "legacy_curated_bundle",
    }
    assert "profile_signals" in payload["items"][0]["metadata"]["signals"]


def test_personalized_bundles_endpoint_is_stable_for_missing_user_profile(monkeypatch, bundle_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root))
    client = create_app().test_client()
    response = client.post(
        "/engines/personalized_bundles/recommendations",
        json={"user_id": "999", "root_product_id": "100", "count": 2},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["engine"] == "personalized_bundles"
    assert payload["status"] == "ok"
    assert payload["metadata"]["profile"]["known_user"] is False
    assert len(payload["items"]) >= 1
    assert "signals" in payload["items"][0]["metadata"]


def test_personalized_bundles_endpoint_serves_from_bundle_universe(monkeypatch, bundle_project_root_with_universe):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root_with_universe))
    client = create_app().test_client()
    response = client.post(
        "/engines/personalized_bundles/recommendations",
        json={"user_id": "202", "root_product_id": "100", "count": 2},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["metadata"]["retrieval"]["mode"] == "bundle_universe"
    assert payload["metadata"]["retrieval"]["bundle_universe_available"] is True
    assert payload["metadata"]["retrieval"]["retrieved_record_count"] >= payload["metadata"]["returned_count"]
    assert payload["items"][0]["metadata"]["retrieval_source"] == "bundle_universe"
    assert "candidate_metadata" in payload["items"][0]["metadata"]
    assert "evidence_signals" in payload["items"][0]["metadata"]["candidate_metadata"]


def test_personalized_bundles_endpoint_falls_back_safely_without_bundle_universe(monkeypatch, bundle_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root))
    client = create_app().test_client()
    response = client.post(
        "/engines/personalized_bundles/recommendations",
        json={"user_id": "999", "root_product_id": "100", "count": 2},
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "ok"
    assert payload["metadata"]["retrieval"]["mode"] == "dynamic_fallback"
    assert payload["metadata"]["retrieval"]["fallback_used"] is True
    assert payload["metadata"]["retrieval"]["bundle_universe_available"] is False
    assert len(payload["items"]) >= 1


def test_personalized_bundles_endpoint_requires_user_id(monkeypatch, bundle_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root))
    client = create_app().test_client()
    response = client.post(
        "/engines/personalized_bundles/recommendations",
        json={"root_product_id": "100", "count": 2},
    )

    assert response.status_code == 400
    assert response.get_json()["status"] == "invalid_request"
