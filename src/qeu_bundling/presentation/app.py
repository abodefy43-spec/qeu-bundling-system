"""Flask presentation app for QEU bundle outputs."""

from __future__ import annotations

import copy
import hashlib
import os
import random
import secrets
import threading
import time

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.feedback_memory import append_feedback_row
from qeu_bundling.core.run_manifest import read_latest_manifest
from qeu_bundling.presentation.bundle_view import load_bundle_view, row_to_record
from qeu_bundling.presentation.dashboard_i18n import (
    dashboard_ui_text_ar,
    translate_recommendations_for_dashboard,
)
from qeu_bundling.presentation.person_predictions import (
    PersonRecommendationState,
    build_default_profiles,
    build_manual_profile,
    build_recommendations_for_profiles,
    load_order_pool,
    load_product_matcher,
)
from qeu_bundling.presentation.run_service import get_status, start_pipeline_run

PATHS = get_paths()
BASE_DIR = PATHS.project_root

MAX_PERSON_COUNT = 20
PERSON_STATE_TTL_SECONDS = 3 * 60 * 60
SESSION_PERSON_KEY = "qeu_person_state_key"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = str(os.environ.get(name, "") or "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except ValueError:
        return int(default)


LOCAL_FAST_MODE = _env_flag("QEU_LOCAL_FAST_MODE", default=False)
DEFAULT_PERSON_COUNT = max(
    1,
    min(
        MAX_PERSON_COUNT,
        _env_int("QEU_DASHBOARD_DEFAULT_PERSON_COUNT", 5 if LOCAL_FAST_MODE else 10),
    ),
)

_PERSON_STATES: dict[str, PersonRecommendationState] = {}
_DEFAULT_PROFILE_CACHE: dict[str, list] = {}
_DEFAULT_RECOMMENDATION_CACHE: dict[str, list[dict[str, object]]] = {}
_STARTUP_PREWARM: dict[str, object] = {
    "enabled": bool(LOCAL_FAST_MODE),
    "ok": True,
    "ready": not LOCAL_FAST_MODE,
    "run_id": "",
    "profile_count": 0,
    "recommendation_count": 0,
    "duration_sec": 0.0,
    "error": "",
}
_PERSON_LOCK = threading.Lock()

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)


def _now_ts() -> float:
    return time.time()


def _latest_run_id() -> str:
    payload = read_latest_manifest(base_dir=BASE_DIR)
    return str(payload.get("run_id", "") or "").strip()


def _seeded_rng(run_id: str) -> random.Random:
    if not run_id:
        return random.Random()
    digest = hashlib.sha1(run_id.encode("utf-8")).hexdigest()[:16]
    return random.Random(int(digest, 16))


def _session_seeded_rng(run_id: str, session_key: str, extra: str = "") -> random.Random:
    seed_text = f"{run_id or 'no_run'}::{session_key or 'session'}::{extra or ''}"
    digest = hashlib.sha1(seed_text.encode("utf-8")).hexdigest()[:16]
    return random.Random(int(digest, 16))


def _get_session_state_key() -> str:
    current = str(session.get(SESSION_PERSON_KEY, "") or "").strip()
    if current:
        return current
    generated = secrets.token_urlsafe(12)
    session[SESSION_PERSON_KEY] = generated
    return generated


def _prune_expired_states(now_ts: float) -> None:
    expired = [
        key
        for key, state in _PERSON_STATES.items()
        if (now_ts - float(state.last_updated or 0.0)) > PERSON_STATE_TTL_SECONDS
    ]
    for key in expired:
        _PERSON_STATES.pop(key, None)


def _cap_profiles(profiles):
    if len(profiles) <= MAX_PERSON_COUNT:
        return profiles
    return profiles[-MAX_PERSON_COUNT:]


def _cap_state(state: PersonRecommendationState) -> PersonRecommendationState:
    """Cap profiles and keep recommendation alignment with profile order."""
    if len(state.profiles) <= MAX_PERSON_COUNT:
        if len(state.recommendations) > len(state.profiles):
            state.recommendations = state.recommendations[: len(state.profiles)]
        return state

    overflow = len(state.profiles) - MAX_PERSON_COUNT
    state.profiles = state.profiles[overflow:]
    if state.recommendations:
        state.recommendations = state.recommendations[overflow:]
        if len(state.recommendations) > len(state.profiles):
            state.recommendations = state.recommendations[: len(state.profiles)]
    return state


def _clone_profiles(profiles):
    return list(profiles or [])


def _clone_recommendations(recommendations: list[dict[str, object]] | None) -> list[dict[str, object]]:
    return copy.deepcopy(list(recommendations or []))


def _default_profiles_for_run(run_id: str):
    cached = _DEFAULT_PROFILE_CACHE.get(run_id)
    if cached is not None:
        return _clone_profiles(cached)
    order_pool = load_order_pool(BASE_DIR)
    profiles = _cap_profiles(
        build_default_profiles(order_pool, count=DEFAULT_PERSON_COUNT, rng=_seeded_rng(run_id))
    )
    _DEFAULT_PROFILE_CACHE[run_id] = _clone_profiles(profiles)
    return _clone_profiles(profiles)


def _cache_default_recommendations(run_id: str, state: PersonRecommendationState, recommendations) -> None:
    if not run_id:
        return
    if len(state.profiles) != DEFAULT_PERSON_COUNT:
        return
    if len(recommendations or []) != len(state.profiles):
        return
    if not all(getattr(profile, "source", "") == "random" for profile in state.profiles):
        return
    _DEFAULT_PROFILE_CACHE[run_id] = _clone_profiles(state.profiles)
    _DEFAULT_RECOMMENDATION_CACHE[run_id] = _clone_recommendations(recommendations)


def _store_default_cache(run_id: str, profiles, recommendations) -> None:
    if not run_id:
        return
    _DEFAULT_PROFILE_CACHE[run_id] = _clone_profiles(profiles)
    _DEFAULT_RECOMMENDATION_CACHE[run_id] = _clone_recommendations(recommendations)


def prewarm_local_dashboard_defaults(run_id: str | None = None) -> dict[str, object]:
    payload = {
        "enabled": bool(LOCAL_FAST_MODE),
        "ok": True,
        "ready": not LOCAL_FAST_MODE,
        "run_id": str(run_id or "").strip(),
        "profile_count": 0,
        "recommendation_count": 0,
        "duration_sec": 0.0,
        "error": "",
    }
    if not LOCAL_FAST_MODE:
        with _PERSON_LOCK:
            _STARTUP_PREWARM.clear()
            _STARTUP_PREWARM.update(payload)
        return dict(payload)

    effective_run_id = str(run_id or _latest_run_id() or "").strip()
    started = time.perf_counter()
    try:
        profiles = _cap_profiles(_default_profiles_for_run(effective_run_id))
        recommendations = _clone_recommendations(_DEFAULT_RECOMMENDATION_CACHE.get(effective_run_id))
        if len(recommendations) != len(profiles):
            data = load_bundle_view(BASE_DIR)
            if data.bundles_df is not None and not data.bundles_df.empty and profiles:
                recommendations = build_recommendations_for_profiles(
                    bundles_df=data.bundles_df,
                    profiles=profiles,
                    max_people=len(profiles),
                    row_to_record=row_to_record,
                    base_dir=BASE_DIR,
                    run_id=effective_run_id,
                )
                if recommendations and not _recommendations_already_translated(recommendations):
                    recommendations = translate_recommendations_for_dashboard(
                        recommendations,
                        cache_path=PATHS.output_dir / "dashboard_en_ar_cache.json",
                    )
            else:
                recommendations = []
        _store_default_cache(effective_run_id, profiles, recommendations)
        payload.update(
            {
                "ready": True,
                "run_id": effective_run_id,
                "profile_count": len(profiles),
                "recommendation_count": len(recommendations),
                "duration_sec": round(float(time.perf_counter() - started), 3),
            }
        )
        print(
            "Local dashboard prewarm ready: "
            f"profiles={payload['profile_count']} recommendations={payload['recommendation_count']} "
            f"duration={payload['duration_sec']:.3f}s"
        )
    except Exception as exc:
        payload.update(
            {
                "ok": False,
                "ready": False,
                "run_id": effective_run_id,
                "duration_sec": round(float(time.perf_counter() - started), 3),
                "error": str(exc),
            }
        )
        print(f"Local dashboard prewarm failed: {exc}")
    with _PERSON_LOCK:
        _STARTUP_PREWARM.clear()
        _STARTUP_PREWARM.update(payload)
    return dict(payload)


def _get_or_seed_state(session_key: str, run_id: str = "") -> PersonRecommendationState:
    now_ts = _now_ts()
    with _PERSON_LOCK:
        _prune_expired_states(now_ts)
        existing = _PERSON_STATES.get(session_key)
        if existing is not None:
            if run_id and existing.run_id != run_id:
                existing.profiles = _default_profiles_for_run(run_id)
                existing.recommendations = _clone_recommendations(_DEFAULT_RECOMMENDATION_CACHE.get(run_id))
                existing.run_id = run_id
            existing.last_updated = now_ts
            return existing

    profiles = _default_profiles_for_run(run_id)
    state = PersonRecommendationState(
        profiles=_cap_profiles(profiles),
        recommendations=_clone_recommendations(_DEFAULT_RECOMMENDATION_CACHE.get(run_id)),
        run_id=run_id,
        last_updated=now_ts,
    )

    with _PERSON_LOCK:
        current = _PERSON_STATES.get(session_key)
        if current is not None:
            if run_id and current.run_id != run_id:
                current.profiles = _default_profiles_for_run(run_id)
                current.recommendations = _clone_recommendations(_DEFAULT_RECOMMENDATION_CACHE.get(run_id))
                current.run_id = run_id
            current.last_updated = now_ts
            return current
        _PERSON_STATES[session_key] = state
        return state


def _set_state_profiles(session_key: str, profiles, run_id: str = "") -> PersonRecommendationState:
    now_ts = _now_ts()
    with _PERSON_LOCK:
        _prune_expired_states(now_ts)
        state = _PERSON_STATES.get(session_key)
        if state is None:
            state = PersonRecommendationState(profiles=[], recommendations=[], run_id=run_id, last_updated=now_ts)
            _PERSON_STATES[session_key] = state
        state.profiles = list(profiles)
        state.recommendations = []
        if run_id:
            state.run_id = run_id
        _cap_state(state)
        state.last_updated = now_ts
        return state


def _append_profile(session_key: str, profile) -> PersonRecommendationState:
    now_ts = _now_ts()
    with _PERSON_LOCK:
        _prune_expired_states(now_ts)
        state = _PERSON_STATES.get(session_key)
        if state is None:
            state = PersonRecommendationState(profiles=[], recommendations=[], last_updated=now_ts)
            _PERSON_STATES[session_key] = state
        state.profiles.append(profile)
        _cap_state(state)
        state.last_updated = now_ts
        return state


def _append_random_profiles(session_key: str, run_id: str, count: int) -> tuple[PersonRecommendationState, int]:
    state = _get_or_seed_state(session_key, run_id=run_id)
    available_slots = max(0, MAX_PERSON_COUNT - len(state.profiles))
    target = min(max(0, int(count)), available_slots)
    if target <= 0:
        return state, 0

    order_pool = load_order_pool(BASE_DIR)
    rng = _session_seeded_rng(run_id, session_key, extra=f"bulk:{len(state.profiles)}:{target}")
    new_profiles = build_default_profiles(order_pool, count=target, rng=rng)
    added = 0
    for profile in new_profiles:
        _append_profile(session_key, profile)
        added += 1
        if added >= target:
            break
    refreshed = _get_or_seed_state(session_key, run_id=run_id)
    return refreshed, added


def _set_state_recommendations(session_key: str, recommendations) -> PersonRecommendationState | None:
    now_ts = _now_ts()
    with _PERSON_LOCK:
        _prune_expired_states(now_ts)
        state = _PERSON_STATES.get(session_key)
        if state is None:
            return None
        state.recommendations = list(recommendations)
        if len(state.recommendations) > len(state.profiles):
            state.recommendations = state.recommendations[: len(state.profiles)]
        state.last_updated = now_ts
        return state


def _recommendations_already_translated(recommendations: list[dict[str, object]]) -> bool:
    if not recommendations:
        return True
    for rec in recommendations:
        if not isinstance(rec, dict):
            return False
        bundles = rec.get("bundles", [])
        if not isinstance(bundles, list):
            return False
        for bundle in bundles:
            if not isinstance(bundle, dict):
                return False
            if "product_a_name_ar" not in bundle or "product_b_name_ar" not in bundle:
                return False
    return True


def _build_cleaning_fallback_bundle(fill_lane: str) -> dict[str, object]:
    return {
        "lane": str(fill_lane),
        "lane_label": str(fill_lane).upper(),
        "product_a": -100001,
        "product_b": -100002,
        "product_a_name": "Multi-purpose cleaner",
        "product_b_name": "Paper towels",
        "anchor_product_id": -100001,
        "complement_product_id": -100002,
        "recommendation_origin": "cleaning_fallback",
        "recommendation_origin_label": "Cleaning fallback",
        "confidence_score": 64.0,
        "hybrid_reco_score": 0.64,
        "free_product": "product_b",
        "history_match_count": 0,
        "anchor_in_history": False,
        "chosen_bundle_names": ["Multi-purpose cleaner", "Paper towels"],
        "recommendation_reasons": ["Cleaning fallback"],
        "bundle_items": [
            {
                "name": "Multi-purpose cleaner",
                "price_sar": "14.90",
                "price_after_sar": "14.90",
                "discount": "",
                "is_free": False,
                "image_url": "",
            },
            {
                "name": "Paper towels",
                "price_sar": "9.90",
                "price_after_sar": "0.00",
                "discount": "100%",
                "is_free": True,
                "image_url": "",
            },
        ],
    }


def _apply_cleaning_display_fallback(recommendations: list[dict[str, object]]) -> list[dict[str, object]]:
    """Preserve serving output without injecting dashboard-only cleaning bundles."""
    return list(recommendations or [])


@app.get("/")
def dashboard():
    status = get_status()
    session_key = _get_session_state_key()
    current_run_id = _latest_run_id()
    state = _get_or_seed_state(session_key, run_id=current_run_id)

    data = load_bundle_view(BASE_DIR)
    recommendations = list(state.recommendations)
    if data.bundles_df is not None:
        target_count = max(DEFAULT_PERSON_COUNT, len(state.profiles))
        if len(recommendations) > len(state.profiles):
            recommendations = recommendations[: len(state.profiles)]

        # Keep existing cards fixed; only generate for newly added profiles.
        if len(recommendations) < len(state.profiles):
            missing_profiles = state.profiles[len(recommendations) :]
            generated = build_recommendations_for_profiles(
                bundles_df=data.bundles_df,
                profiles=missing_profiles,
                max_people=len(missing_profiles),
                row_to_record=row_to_record,
                base_dir=BASE_DIR,
                run_id=state.run_id,
            )
            recommendations = recommendations + generated

        if len(recommendations) < target_count and len(recommendations) < len(state.profiles):
            # Safety fallback for partial generation failures.
            missing_profiles = state.profiles[len(recommendations) :]
            generated = build_recommendations_for_profiles(
                bundles_df=data.bundles_df,
                profiles=missing_profiles,
                max_people=len(missing_profiles),
                row_to_record=row_to_record,
                base_dir=BASE_DIR,
                run_id=state.run_id,
            )
            recommendations = recommendations + generated

        recommendations = recommendations[: len(state.profiles)]
        _set_state_recommendations(session_key, recommendations)

    if _recommendations_already_translated(recommendations):
        translated_recommendations = recommendations
    else:
        translation_cache_path = PATHS.output_dir / "dashboard_en_ar_cache.json"
        translated_recommendations = translate_recommendations_for_dashboard(
            recommendations,
            cache_path=translation_cache_path,
        )
        _set_state_recommendations(session_key, translated_recommendations)
    with _PERSON_LOCK:
        _cache_default_recommendations(current_run_id, state, translated_recommendations)

    return render_template(
        "dashboard.html",
        page_title="Executive Dashboard",
        kpis=data.kpis,
        top10_rows=data.top10_rows,
        person_recommendations=translated_recommendations,
        data_warning=data.data_warning,
        run_status=status,
        people_count=len(state.profiles),
        ui_locale="ar",
        ui_dir="rtl",
        ui_text=dashboard_ui_text_ar(),
    )


@app.get("/healthz")
def healthz():
    with _PERSON_LOCK:
        payload = dict(_STARTUP_PREWARM)
    payload.setdefault("enabled", bool(LOCAL_FAST_MODE))
    payload.setdefault("default_person_count", int(DEFAULT_PERSON_COUNT))
    payload["default_person_count"] = int(DEFAULT_PERSON_COUNT)
    payload["latest_run_id"] = _latest_run_id()
    status_code = 200 if bool(payload.get("ok", True)) else 503
    return jsonify(payload), status_code


@app.get("/bundles")
def all_bundles():
    flash("People-only mode enabled.", "info")
    return redirect(url_for("dashboard"))


@app.post("/refresh")
def refresh():
    ok, message = start_pipeline_run(BASE_DIR)
    flash(message, "success" if ok else "warning")
    return redirect(url_for("dashboard"))


@app.post("/people/add-random")
def add_random_person():
    session_key = _get_session_state_key()
    run_id = _latest_run_id()
    _state, added = _append_random_profiles(session_key, run_id, count=1)
    if added <= 0:
        flash("Could not sample random person data from processed orders.", "warning")
        return redirect(url_for("dashboard"))

    flash("Added 1 random prediction.", "success")
    return redirect(url_for("dashboard"))


@app.post("/people/add-five")
def add_five_predictions():
    session_key = _get_session_state_key()
    run_id = _latest_run_id()
    state, added = _append_random_profiles(session_key, run_id, count=5)
    if state is None:
        state = _get_or_seed_state(session_key, run_id=run_id)
    if added <= 0:
        flash(f"Already at the maximum of {MAX_PERSON_COUNT} predictions.", "info")
    elif len(state.profiles) >= MAX_PERSON_COUNT and added < 5:
        flash(f"Added {added} predictions and reached the max of {MAX_PERSON_COUNT}.", "success")
    else:
        flash(f"Added {added} predictions.", "success")
    return redirect(url_for("dashboard"))


@app.post("/people/add-manual")
def add_manual_person():
    session_key = _get_session_state_key()
    run_id = _latest_run_id()
    _get_or_seed_state(session_key, run_id=run_id)

    orders_text = request.form.get("orders_text", "")
    matcher = load_product_matcher(BASE_DIR)
    result = build_manual_profile(orders_text, matcher)

    if result.profile is None:
        message = result.warnings[0] if result.warnings else "Manual input could not be parsed."
        flash(message, "warning")
        return redirect(url_for("dashboard"))

    _append_profile(session_key, result.profile)
    flash(
        f"Added manual person with {result.matched_count} matched item(s) across {len(result.profile.order_ids)} order line(s).",
        "success",
    )
    if result.warnings:
        preview = "; ".join(result.warnings[:3])
        extra = len(result.warnings) - 3
        if extra > 0:
            preview = f"{preview}; +{extra} more unmatched entries"
        flash(preview, "warning")
    return redirect(url_for("dashboard"))


@app.post("/people/shuffle")
def shuffle_people():
    session_key = _get_session_state_key()
    run_id = _latest_run_id()
    state = _get_or_seed_state(session_key, run_id=run_id)
    target_count = len(state.profiles) if state.profiles else DEFAULT_PERSON_COUNT

    order_pool = load_order_pool(BASE_DIR)
    profiles = build_default_profiles(order_pool, count=target_count, rng=_seeded_rng(run_id))
    updated = _set_state_profiles(session_key, profiles, run_id=run_id)

    if updated.profiles:
        flash(f"Shuffled {len(updated.profiles)} people and regenerated predictions.", "success")
    else:
        flash("Shuffle produced no profiles. Check processed order data.", "warning")
    return redirect(url_for("dashboard"))


@app.post("/people/regenerate")
def regenerate_person():
    session_key = _get_session_state_key()
    run_id = _latest_run_id()
    state = _get_or_seed_state(session_key, run_id=run_id)
    profile_id = str(request.form.get("profile_id", "") or "").strip()
    if not profile_id:
        flash("Missing profile id for regeneration.", "warning")
        return redirect(url_for("dashboard"))

    target_idx = -1
    for idx, profile in enumerate(state.profiles):
        if profile.profile_id == profile_id:
            target_idx = idx
            break
    if target_idx < 0:
        flash("Could not locate that profile.", "warning")
        return redirect(url_for("dashboard"))

    data = load_bundle_view(BASE_DIR)
    if data.bundles_df is None or data.bundles_df.empty:
        flash("No bundles available to regenerate recommendation.", "warning")
        return redirect(url_for("dashboard"))

    target_profile = state.profiles[target_idx]
    generated = build_recommendations_for_profiles(
        bundles_df=data.bundles_df,
        profiles=[target_profile],
        max_people=1,
        row_to_record=row_to_record,
        base_dir=BASE_DIR,
        run_id=state.run_id,
        rng_salt=secrets.token_hex(4),
    )
    if not generated:
        flash("Could not regenerate this recommendation right now.", "warning")
        return redirect(url_for("dashboard"))

    recs = list(state.recommendations)
    while len(recs) < len(state.profiles):
        recs.append({})
    recs[target_idx] = generated[0]
    _set_state_recommendations(session_key, recs)
    flash("Regenerated recommendation for one person.", "success")
    return redirect(url_for("dashboard"))


@app.post("/people/feedback")
def people_feedback():
    profile_id = str(request.form.get("profile_id", "") or "").strip()
    feedback_type = str(request.form.get("feedback_type", "") or "").strip().lower()
    anchor_product_id = request.form.get("anchor_product_id", "0")
    complement_product_id = request.form.get("complement_product_id", "0")
    product_a_name = str(request.form.get("product_a_name", "") or "")
    product_b_name = str(request.form.get("product_b_name", "") or "")
    recommendation_origin = str(request.form.get("recommendation_origin", "") or "")
    source = str(request.form.get("source", "") or "")

    try:
        append_feedback_row(
            profile_id=profile_id,
            anchor_product_id=int(float(anchor_product_id)),
            complement_product_id=int(float(complement_product_id)),
            product_a_name=product_a_name,
            product_b_name=product_b_name,
            feedback_type=feedback_type,
            recommendation_origin=recommendation_origin,
            source=source,
            base_dir=BASE_DIR,
        )
    except Exception as exc:
        flash(f"Could not save feedback: {exc}", "warning")
        return redirect(url_for("dashboard"))

    flash("Feedback saved. Future rankings will adapt.", "success")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
