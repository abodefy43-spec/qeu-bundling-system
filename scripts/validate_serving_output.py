#!/usr/bin/env python3
"""Run serving validation on a deterministic profile sample.

This script executes the real serving path:
`build_recommendations_for_profiles()` and writes:
- raw recommendations
- aggregate metrics report
- optional comparison vs a previous raw artifact
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.run_manifest import read_latest_manifest
from qeu_bundling.presentation.bundle_view import load_bundle_view, row_to_record
from qeu_bundling.presentation.person_predictions import (
    PersonProfile,
    _build_user_intent_profile,
    _bundle_primary_intent,
    _top_intent,
    build_default_profiles,
    build_recommendations_for_profiles,
    get_last_serving_profile_metrics,
    load_personalization_context,
    load_order_pool,
)


def _tier_from_origin(origin: str) -> str:
    norm = str(origin or "").strip().lower()
    if norm == "top_bundle":
        return "tier1_personalized"
    if norm == "copurchase_fallback":
        return "tier2_safe_fallback"
    if norm in {"fallback_food", "fallback_cleaning"} or norm.startswith("fallback_"):
        return "tier3_curated_fallback"
    return "other"


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, str):
        value = value.replace(",", "").strip()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if out != out:  # NaN
        return float(default)
    return float(out)


def _is_forbidden_pair(name_a: str, name_b: str) -> bool:
    text = f"{name_a} {name_b}".lower()
    seafood = any(token in text for token in ("tuna", "seafood"))
    dairy_beverage = any(token in text for token in ("milk", "yogurt", "yoghurt", "chocolate milk"))
    dessert = any(token in text for token in ("biscuit", "cookie", "chocolate", "dessert", "candy", "wafer"))
    noodles = any(token in text for token in ("noodle", "indomie", "ramen", "mi sidap", "cup noodles"))
    bread = any(token in text for token in ("bread", "toast", "tortilla"))
    sweets = any(token in text for token in ("chocolate", "candy", "dessert", "sweet"))
    if seafood and (dairy_beverage or dessert):
        return True
    if noodles and (bread or sweets):
        return True
    return False


def _pair_key(bundle: dict[str, Any]) -> tuple[int, int]:
    a = int(bundle.get("product_a", -1))
    b = int(bundle.get("product_b", -1))
    return tuple(sorted((a, b)))


def _bundle_layout(rec: dict[str, Any]) -> list[tuple[int, int, str, str]]:
    return [
        (
            int(bundle.get("anchor_product_id", -1)),
            int(bundle.get("complement_product_id", -1)),
            str(bundle.get("lane", "")),
            str(bundle.get("recommendation_origin", "")),
        )
        for bundle in rec.get("bundles", [])
        if isinstance(bundle, dict)
    ]


def _profile_signature_from_order_ids(order_ids: list[int]) -> tuple[int, ...]:
    return tuple(sorted(int(x) for x in order_ids if int(x) > 0))


def _profile_from_recommendation(
    rec: dict[str, Any],
    profile_lookup: dict[tuple[int, ...], PersonProfile],
    order_pool,
) -> PersonProfile:
    order_ids = [int(x) for x in rec.get("source_order_ids", []) if int(x) > 0]
    signature = _profile_signature_from_order_ids(order_ids)
    existing = profile_lookup.get(signature)
    if existing is not None:
        return existing

    history_counts: dict[int, int] = {}
    for oid in order_ids:
        for pid in order_pool.order_product_ids.get(int(oid), ()):
            pid_int = int(pid)
            history_counts[pid_int] = int(history_counts.get(pid_int, 0)) + 1
    history_ids = sorted(history_counts.keys())
    return PersonProfile(
        profile_id=str(rec.get("profile_id", "")),
        source=str(rec.get("source", "unknown")),
        order_ids=order_ids,
        history_product_ids=history_ids,
        history_items=[str(x) for x in rec.get("history_items", []) if str(x).strip()],
        created_at="",
        history_counts={int(k): int(v) for k, v in history_counts.items()},
    )


def _intent_misclassification_reason(intent: str, name_a: str, name_b: str, lane: str) -> str | None:
    text = f"{name_a} {name_b}".lower()
    lane_norm = str(lane).strip().lower()
    is_food = any(token in text for token in ("rice", "chicken", "tuna", "milk", "tea", "coffee", "biscuit", "chocolate"))
    has_cleaning = any(token in text for token in ("detergent", "softener", "dish", "disinfectant", "bleach", "wipes", "cleaner"))
    if intent == "cleaning" and is_food:
        return "cleaning_with_food_tokens"
    if lane_norm == "nonfood" and intent != "cleaning":
        return "nonfood_not_classified_cleaning"
    if lane_norm != "nonfood" and intent == "cleaning":
        return "food_lane_classified_cleaning"
    if intent == "meal_base" and any(token in text for token in ("caramel", "custard", "dessert")):
        return "meal_base_with_dessert_tokens"
    if intent == "snack" and any(token in text for token in ("detergent", "softener", "disinfectant")):
        return "snack_with_cleaning_tokens"
    if intent == "drink_pairing" and not any(token in text for token in ("tea", "coffee", "milk", "juice", "soda", "drink")):
        return "drink_pairing_without_drink_token"
    if has_cleaning and lane_norm != "nonfood":
        return "cleaning_token_in_food_lane"
    return None


def _build_summary(
    recommendations: list[dict[str, Any]],
    profiles: list[PersonProfile],
    context,
    order_pool,
    metrics: dict[str, Any],
    elapsed_sec: float,
    sample_size: int,
) -> dict[str, Any]:
    profile_lookup: dict[tuple[int, ...], PersonProfile] = {
        _profile_signature_from_order_ids([int(x) for x in profile.order_ids]): profile
        for profile in profiles
    }
    coverage = Counter()
    tier_counts = Counter()
    tier_mix = Counter()
    intent_counts = Counter()
    user_top_intent_counts = Counter()
    pair_counts = Counter()
    pair_name_counts = Counter()
    anchor_counts = Counter()
    motif_counts = Counter()
    item_reuse_violations: list[str] = []
    compatibility_violations: list[dict[str, Any]] = []
    cleaning_bundles_served: list[dict[str, Any]] = []
    suspicious_intent_misclassifications: list[dict[str, Any]] = []
    per_user_top_intent: list[dict[str, Any]] = []
    per_user_selected_intents: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for rec in recommendations:
        bundles = [b for b in rec.get("bundles", []) if isinstance(b, dict)]
        coverage[int(len(bundles))] += 1
        user_profile = _profile_from_recommendation(rec, profile_lookup, order_pool)
        user_intent_profile = _build_user_intent_profile(user_profile, context)
        top_intent, top_intent_weight = _top_intent(user_intent_profile)
        user_top_intent_counts[str(top_intent)] += 1
        per_user_top_intent.append(
            {
                "profile_id": str(rec.get("profile_id", "")),
                "source_order_ids": [int(x) for x in rec.get("source_order_ids", []) if int(x) > 0],
                "top_user_intent": str(top_intent),
                "top_user_intent_weight": float(top_intent_weight),
            }
        )
        seen_items: set[int] = set()
        reused = False
        user_tiers: list[str] = []
        selected_intents: list[str] = []
        for bundle in bundles:
            origin = str(bundle.get("recommendation_origin", ""))
            tier = _tier_from_origin(origin)
            user_tiers.append(tier)
            tier_counts[tier] += 1
            pair = _pair_key(bundle)
            pair_counts[pair] += 1
            anchor_counts[int(bundle.get("anchor_product_id", -1))] += 1
            name_a = str(bundle.get("product_a_name", "")).strip()
            name_b = str(bundle.get("product_b_name", "")).strip()
            pair_name = " + ".join(sorted((name_a.lower(), name_b.lower())))
            pair_name_counts[pair_name] += 1
            motif_counts[pair_name] += 1
            if _is_forbidden_pair(name_a, name_b):
                compatibility_violations.append(
                    {
                        "profile_id": str(rec.get("profile_id", "")),
                        "pair": pair_name,
                        "origin": origin,
                        "lane": str(bundle.get("lane", "")),
                    }
                )
            a = int(bundle.get("product_a", -1))
            b = int(bundle.get("product_b", -1))
            lane = str(bundle.get("lane", "")).strip().lower()
            candidate_stub = {
                "anchor": int(bundle.get("anchor_product_id", a)),
                "complement": int(bundle.get("complement_product_id", b)),
                "lane": lane,
            }
            bundle_intent = str(_bundle_primary_intent(candidate_stub, context, lane_hint=lane))
            selected_intents.append(bundle_intent)
            intent_counts[bundle_intent] += 1
            if bundle_intent == "cleaning":
                cleaning_bundles_served.append(
                    {
                        "profile_id": str(rec.get("profile_id", "")),
                        "source_order_ids": [int(x) for x in rec.get("source_order_ids", []) if int(x) > 0],
                        "pair_names": [name_a, name_b],
                        "lane": lane,
                        "origin": origin,
                    }
                )
            misclassification_reason = _intent_misclassification_reason(bundle_intent, name_a, name_b, lane)
            if misclassification_reason:
                suspicious_intent_misclassifications.append(
                    {
                        "profile_id": str(rec.get("profile_id", "")),
                        "bundle_intent": bundle_intent,
                        "reason": misclassification_reason,
                        "pair_names": [name_a, name_b],
                        "lane": lane,
                        "origin": origin,
                    }
                )
            if a in seen_items or b in seen_items:
                reused = True
            if a > 0:
                seen_items.add(a)
            if b > 0:
                seen_items.add(b)
            sample_rows.append(
                {
                    "profile_id": str(rec.get("profile_id", "")),
                    "lane": str(bundle.get("lane", "")),
                    "origin": origin,
                    "tier": tier,
                    "bundle_intent": bundle_intent,
                    "top_user_intent": str(top_intent),
                    "pair_ids": [a, b],
                    "pair_names": [name_a, name_b],
                    "hybrid_score": _safe_float(bundle.get("hybrid_reco_score", 0.0)),
                    "paid_price": _safe_float(bundle.get("price_after_discount_a", 0.0)),
                }
            )
        if reused:
            item_reuse_violations.append(str(rec.get("profile_id", "")))
        tier_mix[",".join(sorted(user_tiers))] += 1
        per_user_selected_intents.append(
            {
                "profile_id": str(rec.get("profile_id", "")),
                "source_order_ids": [int(x) for x in rec.get("source_order_ids", []) if int(x) > 0],
                "top_user_intent": str(top_intent),
                "selected_bundle_intents": list(selected_intents),
            }
        )

    latency = {
        "elapsed_total_sec": float(elapsed_sec),
        "elapsed_avg_per_returned_profile_sec": float(elapsed_sec / max(1, len(recommendations))),
        "profile_count_requested": int(sample_size),
        "profile_count_returned": int(len(recommendations)),
        "profile_metrics": metrics,
    }
    top_pairs = [
        {"pair_ids": [int(p[0]), int(p[1])], "count": int(c)}
        for p, c in pair_counts.most_common(20)
    ]
    top_pair_names = [
        {"pair": str(p), "count": int(c)}
        for p, c in pair_name_counts.most_common(20)
    ]

    return {
        "coverage": {
            "bundle_count_distribution": {str(k): int(v) for k, v in sorted(coverage.items())},
            "exactly_3_rate": float(coverage.get(3, 0) / max(1, len(recommendations))),
        },
        "tier_usage": {
            "bundle_tier_counts": {str(k): int(v) for k, v in sorted(tier_counts.items())},
            "per_user_tier_mix_counts": {str(k): int(v) for k, v in sorted(tier_mix.items())},
        },
        "intent_personalization": {
            "bundle_intent_counts": {str(k): int(v) for k, v in sorted(intent_counts.items())},
            "user_top_intent_counts": {str(k): int(v) for k, v in sorted(user_top_intent_counts.items())},
            "per_user_top_intent": per_user_top_intent,
            "per_user_selected_intents": per_user_selected_intents,
            "cleaning_bundles_served_count": int(len(cleaning_bundles_served)),
            "cleaning_bundles_served": cleaning_bundles_served,
            "suspicious_intent_misclassifications_count": int(len(suspicious_intent_misclassifications)),
            "suspicious_intent_misclassifications": suspicious_intent_misclassifications,
        },
        "hard_rule_checks": {
            "item_reuse_violation_count": int(len(item_reuse_violations)),
            "item_reuse_violations": item_reuse_violations,
            "compatibility_violation_count": int(len(compatibility_violations)),
            "compatibility_violations": compatibility_violations,
        },
        "repetition": {
            "top_pairs_by_id": top_pairs,
            "top_pairs_by_name": top_pair_names,
            "top_anchors": [
                {"anchor_id": int(anchor), "count": int(count)}
                for anchor, count in anchor_counts.most_common(20)
            ],
            "top_motifs": [
                {"motif": str(motif), "count": int(count)}
                for motif, count in motif_counts.most_common(20)
            ],
        },
        "latency": latency,
        "sample_rows": sample_rows,
    }


def _comparison_payload(
    current_recommendations: list[dict[str, Any]],
    compare_path: Path,
) -> dict[str, Any]:
    if not compare_path.exists():
        return {"compare_found": False, "changed_profiles": -1, "missing_profiles": -1}
    previous = json.loads(compare_path.read_text(encoding="utf-8"))
    prev_recs = previous.get("recommendations", [])
    if not isinstance(prev_recs, list):
        return {"compare_found": False, "changed_profiles": -1, "missing_profiles": -1}

    def key_for(rec: dict[str, Any]) -> tuple[int, ...]:
        return tuple(sorted(int(x) for x in rec.get("source_order_ids", [])))

    current_by_key = {key_for(rec): rec for rec in current_recommendations if isinstance(rec, dict)}
    previous_by_key = {key_for(rec): rec for rec in prev_recs if isinstance(rec, dict)}
    all_keys = sorted(set(current_by_key) | set(previous_by_key))
    changed = 0
    missing = 0
    changed_examples: list[dict[str, Any]] = []
    changed_bundle_count_by_user: list[dict[str, Any]] = []
    for key in all_keys:
        cur = current_by_key.get(key)
        prev = previous_by_key.get(key)
        if cur is None or prev is None:
            missing += 1
            continue
        prev_bundle_count = int(len([b for b in prev.get("bundles", []) if isinstance(b, dict)]))
        cur_bundle_count = int(len([b for b in cur.get("bundles", []) if isinstance(b, dict)]))
        if prev_bundle_count != cur_bundle_count:
            changed_bundle_count_by_user.append(
                {
                    "source_order_ids": list(key),
                    "before_bundle_count": prev_bundle_count,
                    "after_bundle_count": cur_bundle_count,
                }
            )
        cur_layout = _bundle_layout(cur)
        prev_layout = _bundle_layout(prev)
        if cur_layout != prev_layout:
            changed += 1
            if len(changed_examples) < 10:
                changed_examples.append(
                    {
                        "source_order_ids": list(key),
                        "before_bundle_count": prev_bundle_count,
                        "after_bundle_count": cur_bundle_count,
                        "previous": prev_layout,
                        "current": cur_layout,
                    }
                )
    return {
        "compare_found": True,
        "changed_profiles": int(changed),
        "missing_profiles": int(missing),
        "changed_bundle_count_by_user": changed_bundle_count_by_user,
        "changed_examples": changed_examples,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate serving output on deterministic profile samples.")
    parser.add_argument("--sample-size", type=int, default=25, help="Number of profiles to sample.")
    parser.add_argument("--seed", type=int, default=20260315, help="Deterministic sampling seed.")
    parser.add_argument("--label", type=str, default="serving_validation", help="Artifact label.")
    parser.add_argument("--run-id", type=str, default="", help="Run id forwarded to serving.")
    parser.add_argument("--rng-salt", type=str, default="validation", help="rng_salt forwarded to serving.")
    parser.add_argument("--sample-users", type=int, default=30, help="Rows to keep in human-readable sample.")
    parser.add_argument(
        "--compare-path",
        type=str,
        default="",
        help="Optional prior raw JSON artifact to compare against.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    base_dir = get_paths().project_root
    output_dir = base_dir / "output" / "serving_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(args.run_id or "").strip()
    if not run_id:
        manifest = read_latest_manifest(base_dir=base_dir)
        run_id = str(manifest.get("run_id", "") or "validation_run")

    view = load_bundle_view(base_dir)
    bundles_df = view.bundles_df
    order_pool = load_order_pool(base_dir)
    context = load_personalization_context(base_dir)
    profiles = build_default_profiles(order_pool, count=max(1, int(args.sample_size)), rng=random.Random(int(args.seed)))

    os.environ["QEU_SERVING_PROFILE"] = "1"
    started = time.perf_counter()
    recommendations = build_recommendations_for_profiles(
        bundles_df=bundles_df,
        profiles=profiles,
        max_people=len(profiles),
        row_to_record=row_to_record,
        base_dir=base_dir,
        run_id=run_id,
        rng_salt=str(args.rng_salt),
    )
    elapsed = time.perf_counter() - started
    metrics = get_last_serving_profile_metrics()

    summary = _build_summary(
        recommendations=recommendations,
        profiles=profiles,
        context=context,
        order_pool=order_pool,
        metrics=metrics,
        elapsed_sec=elapsed,
        sample_size=len(profiles),
    )
    summary["comparison"] = _comparison_payload(recommendations, Path(args.compare_path).resolve()) if args.compare_path else {}

    raw_payload = {
        "label": str(args.label),
        "seed": int(args.seed),
        "run_id": run_id,
        "rng_salt": str(args.rng_salt),
        "sample_size": int(len(profiles)),
        "profiles": [profile.__dict__ for profile in profiles],
        "recommendations": recommendations,
        "metrics": metrics,
    }

    stem = f"{args.label}_sample{len(profiles)}_seed{args.seed}"
    raw_path = output_dir / f"{stem}_raw.json"
    summary_path = output_dir / f"{stem}_summary.json"
    sample_path = output_dir / f"{stem}_sample.json"
    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_rows = summary.get("sample_rows", [])
    sample_path.write_text(
        json.dumps(sample_rows[: max(1, int(args.sample_users))], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"base_dir={base_dir}")
    print(f"raw_output={raw_path}")
    print(f"summary_output={summary_path}")
    print(f"sample_output={sample_path}")
    print(
        "coverage="
        + json.dumps(summary.get("coverage", {}).get("bundle_count_distribution", {}), ensure_ascii=False, sort_keys=True)
    )
    tier_counts = summary.get("tier_usage", {}).get("bundle_tier_counts", {})
    print("tier_counts=" + json.dumps(tier_counts, ensure_ascii=False, sort_keys=True))
    intent_summary = summary.get("intent_personalization", {})
    print(
        "intent_counts="
        + json.dumps(intent_summary.get("bundle_intent_counts", {}), ensure_ascii=False, sort_keys=True)
    )
    print(f"cleaning_bundles_served={int(intent_summary.get('cleaning_bundles_served_count', 0))}")
    print(
        "suspicious_intent_misclassifications="
        + str(int(intent_summary.get("suspicious_intent_misclassifications_count", 0)))
    )
    latency = summary.get("latency", {})
    print(
        "latency="
        + json.dumps(
            {
                "elapsed_total_sec": latency.get("elapsed_total_sec"),
                "elapsed_avg_sec": latency.get("elapsed_avg_per_returned_profile_sec"),
                "p50_sec": metrics.get("p50_latency_sec"),
                "p90_sec": metrics.get("p90_latency_sec"),
                "p95_sec": metrics.get("p95_latency_sec"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
