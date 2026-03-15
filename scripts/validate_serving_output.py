#!/usr/bin/env python3
"""Run real serving-path validation for person bundle outputs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qeu_bundling.core.pricing import FIXED_MARGIN_DISCOUNT_PCT, margin_discounted_sale_price
from qeu_bundling.presentation.bundle_view import load_bundle_view, row_to_record
from qeu_bundling.presentation.person_predictions import (
    _consumption_pair_reject_reason,
    build_default_profiles,
    build_recommendations_for_profiles,
    load_order_pool,
    load_personalization_context,
)

FOOD_LANES = {"meal", "snack", "occasion"}
NONFOOD_LANE = "nonfood"
TIER1 = "tier1_personalized"
TIER2 = "tier2_safe_fallback"
TIER3 = "tier3_emergency_curated"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pair_label(name_a: str, name_b: str) -> str:
    names = sorted([str(name_a).strip(), str(name_b).strip()], key=lambda x: x.lower())
    return " + ".join([n for n in names if n])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate serving outputs via real final path.")
    parser.add_argument("--sample-size", type=int, default=150, help="Number of profiles to evaluate.")
    parser.add_argument("--sample-users", type=int, default=30, help="Number of users to include in readable sample.")
    parser.add_argument("--seed", type=int, default=20260315, help="Deterministic seed for profile generation.")
    parser.add_argument("--run-id", default="serving_validation", help="Run id to pass into serving path.")
    parser.add_argument(
        "--output-dir",
        default="output/review/serving_validation",
        help="Directory for generated validation artifacts.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    base_dir = Path(__file__).resolve().parents[1]
    output_dir = (base_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_view = load_bundle_view(base_dir)
    if bundle_view.bundles_df is None or bundle_view.bundles_df.empty:
        raise RuntimeError("No bundle candidate data found in real serving path input.")

    order_pool = load_order_pool(base_dir)
    profiles = build_default_profiles(
        order_pool,
        count=max(1, int(args.sample_size)),
        rng=random.Random(int(args.seed)),
    )
    if not profiles:
        raise RuntimeError("Could not build profiles from order pool.")

    recommendations = build_recommendations_for_profiles(
        bundles_df=bundle_view.bundles_df,
        profiles=profiles,
        max_people=len(profiles),
        row_to_record=row_to_record,
        base_dir=base_dir,
        run_id=str(args.run_id),
    )
    context = load_personalization_context(base_dir)
    non_food_ids = {int(pid) for pid in context.non_food_ids}

    coverage = Counter()
    mode_counts = Counter()
    tier_bundle_counts = Counter()
    pair_counts = Counter()
    pair_users: dict[tuple[int, int], set[str]] = defaultdict(set)
    pair_names: dict[tuple[int, int], str] = {}
    item_reuse_violations: list[dict[str, Any]] = []
    compatibility_violations: list[dict[str, Any]] = []
    pricing_violations: list[dict[str, Any]] = []
    suspicious_bundles: list[dict[str, Any]] = []
    tier3_pair_counts = Counter()
    tier3_pair_users: dict[tuple[int, int], set[str]] = defaultdict(set)
    details_rows: list[dict[str, Any]] = []

    users_all_3_tier1 = 0
    users_mixed_tiers = 0
    users_requiring_tier3 = 0

    recommendation_by_profile = {
        str(rec.get("profile_id", "")): rec
        for rec in recommendations
        if isinstance(rec, dict)
    }
    for profile in profiles:
        profile_id = str(profile.profile_id)
        rec = recommendation_by_profile.get(profile_id, {})
        bundles = rec.get("bundles", []) if isinstance(rec, dict) else []
        bundles = bundles if isinstance(bundles, list) else []
        coverage[len(bundles)] += 1
        mode_counts[str(rec.get("recommendation_mode", ""))] += 1

        tiers = []
        item_ids: list[int] = []
        for index, bundle in enumerate(bundles):
            if not isinstance(bundle, dict):
                continue
            a = _safe_int(bundle.get("product_a"), default=-1)
            b = _safe_int(bundle.get("product_b"), default=-1)
            if a > 0:
                item_ids.append(a)
            if b > 0:
                item_ids.append(b)
            name_a = str(bundle.get("product_a_name", "")).strip()
            name_b = str(bundle.get("product_b_name", "")).strip()
            lane = str(bundle.get("lane", "")).strip().lower()
            tier = str(bundle.get("serving_tier", "")).strip().lower()
            origin = str(bundle.get("recommendation_origin", "")).strip().lower()
            tiers.append(tier)
            tier_bundle_counts[tier] += 1
            pair_key = tuple(sorted((a, b)))
            if a > 0 and b > 0:
                pair_counts[pair_key] += 1
                pair_users[pair_key].add(profile_id)
                pair_names.setdefault(pair_key, _pair_label(name_a, name_b))
                if tier == TIER3:
                    tier3_pair_counts[pair_key] += 1
                    tier3_pair_users[pair_key].add(profile_id)

            compatibility_reason = _consumption_pair_reject_reason(name_a, name_b)
            if compatibility_reason:
                compatibility_violations.append(
                    {
                        "profile_id": profile_id,
                        "bundle_index": index + 1,
                        "pair": _pair_label(name_a, name_b),
                        "reason": compatibility_reason,
                    }
                )

            a_nonfood = a in non_food_ids
            b_nonfood = b in non_food_ids
            suspicious_reason = ""
            if compatibility_reason:
                suspicious_reason = f"compatibility_block:{compatibility_reason}"
            elif lane in FOOD_LANES and (a_nonfood or b_nonfood):
                suspicious_reason = "food_lane_contains_nonfood_item"
            elif lane == NONFOOD_LANE and not (a_nonfood and b_nonfood):
                suspicious_reason = "nonfood_lane_contains_food_item"
            elif not name_a or not name_b:
                suspicious_reason = "missing_product_name"
            if suspicious_reason:
                suspicious_bundles.append(
                    {
                        "profile_id": profile_id,
                        "bundle_index": index + 1,
                        "pair": _pair_label(name_a, name_b),
                        "lane": lane,
                        "tier": tier,
                        "reason": suspicious_reason,
                    }
                )

            paid_side = str(bundle.get("paid_product", "product_a")).strip().lower()
            if paid_side == "product_b":
                sale = _safe_float(bundle.get("product_b_price"), default=0.0)
                purchase = _safe_float(bundle.get("purchase_price_b"), default=sale)
                free_value = _safe_float(bundle.get("price_after_discount_a"), default=0.0)
            else:
                sale = _safe_float(bundle.get("product_a_price"), default=0.0)
                purchase = _safe_float(bundle.get("purchase_price_a"), default=sale)
                free_value = _safe_float(bundle.get("price_after_discount_b"), default=0.0)
            expected_paid = margin_discounted_sale_price(sale, purchase, FIXED_MARGIN_DISCOUNT_PCT)
            actual_paid = _safe_float(bundle.get("paid_item_final_price"), default=0.0)
            if abs(expected_paid - actual_paid) > 0.05 or abs(free_value) > 0.05:
                pricing_violations.append(
                    {
                        "profile_id": profile_id,
                        "bundle_index": index + 1,
                        "pair": _pair_label(name_a, name_b),
                        "paid_side": paid_side,
                        "expected_paid": round(expected_paid, 3),
                        "actual_paid": round(actual_paid, 3),
                        "free_side_value": round(free_value, 3),
                    }
                )

            details_rows.append(
                {
                    "profile_id": profile_id,
                    "recommendation_mode": str(rec.get("recommendation_mode", "")),
                    "bundle_index": index + 1,
                    "lane": lane,
                    "serving_tier": tier,
                    "origin": origin,
                    "product_a": a,
                    "product_a_name": name_a,
                    "product_b": b,
                    "product_b_name": name_b,
                    "pair_label": _pair_label(name_a, name_b),
                    "final_stage_score": _safe_float(bundle.get("final_stage_score"), default=0.0),
                    "final_stage_repetition_penalty": _safe_float(bundle.get("final_stage_repetition_penalty"), default=0.0),
                    "final_stage_tuna_penalty": _safe_float(bundle.get("final_stage_tuna_penalty"), default=0.0),
                    "paid_item_final_price": _safe_float(bundle.get("paid_item_final_price"), default=0.0),
                }
            )

        if len(item_ids) != len(set(item_ids)):
            repeated = sorted([item_id for item_id, count in Counter(item_ids).items() if count > 1])
            item_reuse_violations.append({"profile_id": profile_id, "reused_item_ids": repeated})

        unique_tiers = sorted({tier for tier in tiers if tier})
        if len(bundles) == 3 and unique_tiers == [TIER1]:
            users_all_3_tier1 += 1
        if len(unique_tiers) > 1:
            users_mixed_tiers += 1
        if TIER3 in unique_tiers:
            users_requiring_tier3 += 1

    total_profiles = len(profiles)
    total_bundles = sum(coverage[count] * count for count in coverage)
    tier3_bundles = int(tier_bundle_counts.get(TIER3, 0))
    repeated_pair_types = int(sum(1 for count in pair_counts.values() if count > 1))

    top_pairs = []
    for pair_key, count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))[:20]:
        top_pairs.append(
            {
                "pair_key": f"{pair_key[0]}-{pair_key[1]}",
                "pair_label": pair_names.get(pair_key, ""),
                "bundle_occurrences": int(count),
                "user_occurrences": int(len(pair_users.get(pair_key, set()))),
            }
        )

    top_tier3_pairs = []
    for pair_key, count in sorted(tier3_pair_counts.items(), key=lambda item: (-item[1], item[0]))[:20]:
        top_tier3_pairs.append(
            {
                "pair_key": f"{pair_key[0]}-{pair_key[1]}",
                "pair_label": pair_names.get(pair_key, ""),
                "bundle_occurrences": int(count),
                "user_occurrences": int(len(tier3_pair_users.get(pair_key, set()))),
            }
        )

    sample_users = []
    for rec in recommendations[: max(1, int(args.sample_users))]:
        profile_id = str(rec.get("profile_id", ""))
        mode = str(rec.get("recommendation_mode", ""))
        bundles = rec.get("bundles", []) if isinstance(rec.get("bundles", []), list) else []
        rows = []
        for idx, bundle in enumerate(bundles):
            if not isinstance(bundle, dict):
                continue
            rows.append(
                {
                    "bundle_index": idx + 1,
                    "pair": _pair_label(bundle.get("product_a_name", ""), bundle.get("product_b_name", "")),
                    "lane": str(bundle.get("lane", "")).strip().lower(),
                    "tier": str(bundle.get("serving_tier", "")).strip().lower(),
                    "origin": str(bundle.get("recommendation_origin", "")).strip().lower(),
                    "price_paid": round(_safe_float(bundle.get("paid_item_final_price"), default=0.0), 2),
                }
            )
        sample_users.append(
            {
                "profile_id": profile_id,
                "recommendation_mode": mode,
                "bundle_count": len(rows),
                "bundles": rows,
            }
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(args.seed),
        "run_id": str(args.run_id),
        "sample_size_requested": int(args.sample_size),
        "sample_size_actual": int(total_profiles),
        "coverage": {
            "users_with_3_bundles": int(coverage.get(3, 0)),
            "users_with_2_bundles": int(coverage.get(2, 0)),
            "users_with_1_bundle": int(coverage.get(1, 0)),
            "users_with_0_bundles": int(coverage.get(0, 0)),
        },
        "recommendation_mode_counts": {str(k): int(v) for k, v in sorted(mode_counts.items())},
        "tier_usage": {
            "bundle_counts": {str(k): int(v) for k, v in sorted(tier_bundle_counts.items())},
            "users_all_3_tier1_personalized": int(users_all_3_tier1),
            "users_with_mixed_tiers": int(users_mixed_tiers),
            "users_requiring_tier3": int(users_requiring_tier3),
            "tier3_bundle_share": round(float(tier3_bundles / max(1, total_bundles)), 4),
        },
        "hard_rule_compliance": {
            "item_reuse_violation_count": int(len(item_reuse_violations)),
            "compatibility_violation_count": int(len(compatibility_violations)),
            "pricing_violation_count": int(len(pricing_violations)),
        },
        "repetition": {
            "total_bundles": int(total_bundles),
            "unique_pair_types": int(len(pair_counts)),
            "repeated_pair_types": int(repeated_pair_types),
            "top_repeated_pairs": top_pairs,
        },
        "tier3_quality": {
            "tier3_bundle_count": int(tier3_bundles),
            "tier3_user_count": int(users_requiring_tier3),
            "tier3_suspicious_bundle_count": int(
                sum(1 for row in suspicious_bundles if str(row.get("tier", "")) == TIER3)
            ),
            "top_tier3_pairs": top_tier3_pairs,
        },
        "suspicious_bundle_count": int(len(suspicious_bundles)),
    }

    summary_path = output_dir / "summary.json"
    details_path = output_dir / "bundle_details.csv"
    top_pairs_path = output_dir / "top_repeated_pairs.csv"
    sample_path = output_dir / "sample_users.json"
    violations_path = output_dir / "violations.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sample_path.write_text(json.dumps(sample_users, ensure_ascii=False, indent=2), encoding="utf-8")
    violations_path.write_text(
        json.dumps(
            {
                "item_reuse_violations": item_reuse_violations,
                "compatibility_violations": compatibility_violations,
                "pricing_violations": pricing_violations,
                "suspicious_bundles": suspicious_bundles,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with details_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "profile_id",
                "recommendation_mode",
                "bundle_index",
                "lane",
                "serving_tier",
                "origin",
                "product_a",
                "product_a_name",
                "product_b",
                "product_b_name",
                "pair_label",
                "final_stage_score",
                "final_stage_repetition_penalty",
                "final_stage_tuna_penalty",
                "paid_item_final_price",
            ],
        )
        writer.writeheader()
        writer.writerows(details_rows)

    with top_pairs_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["pair_key", "pair_label", "bundle_occurrences", "user_occurrences"],
        )
        writer.writeheader()
        writer.writerows(top_pairs)

    print(f"Summary: {summary_path}")
    print(f"Details: {details_path}")
    print(f"Top repeated pairs: {top_pairs_path}")
    print(f"Sample users: {sample_path}")
    print(f"Violations: {violations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
