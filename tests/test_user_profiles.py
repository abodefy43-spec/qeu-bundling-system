from features.user_profiles import load_user_profile_store


def test_user_profile_store_derives_affinity_and_archetypes(monkeypatch, bundle_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root))

    store = load_user_profile_store(project_root=bundle_project_root)
    profile = store.get_profile("101")

    assert profile.known_user is True
    assert profile.product_score(200) > profile.product_score(220)
    assert profile.category_score("beverages") > 0.0
    assert profile.archetype_score("beverages|snacks") > 0.0
    assert 200 in profile.fatigued_product_ids


def test_user_profile_store_returns_empty_profile_for_missing_user(monkeypatch, bundle_project_root):
    monkeypatch.setenv("QEU_PROJECT_ROOT", str(bundle_project_root))

    store = load_user_profile_store(project_root=bundle_project_root)
    profile = store.get_profile("999")

    assert profile.known_user is False
    assert profile.product_affinity == {}
    assert profile.recent_product_ids == ()
