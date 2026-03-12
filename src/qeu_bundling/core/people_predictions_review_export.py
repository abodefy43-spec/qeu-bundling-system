"""People predictions review export (100-person food-lane audit).

What this generates:
- A deterministic quality-review export for people-based predictions.
- Each exported person has exactly three food lane slots: MEAL, SNACK, OCCASION.
- NONFOOD is disabled for this review run.

Where it writes:
- output/review/people_predictions_review_100.json
- output/review/people_predictions_review_100_bundles.csv
- output/review/people_predictions_review_100_summary.md

How to run:
- python -m qeu_bundling.core.people_predictions_review_export
- python -m qeu_bundling.core.people_predictions_review_export --seed 20260307 --target-people 100 --run-id review_people_20260307
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from unittest.mock import patch

import pandas as pd

from qeu_bundling.config.paths import ensure_layout, get_paths
from qeu_bundling.presentation.bundle_view import load_bundle_view, row_to_record
from qeu_bundling.presentation.person_predictions import (
    LANE_MEAL,
    LANE_OCCASION,
    LANE_SNACK,
    OrderPool,
    PersonProfile,
    build_random_profile,
    build_recommendations_for_profiles,
    load_order_pool,
)

FOOD_LANES: tuple[str, str, str] = (LANE_MEAL, LANE_SNACK, LANE_OCCASION)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _profile_signature(profile: PersonProfile) -> tuple[int, ...]:
    return tuple(sorted(int(pid) for pid in profile.history_product_ids if int(pid) > 0))


def _stable_seed(run_id: str, seed: int) -> int:
    digest = hashlib.sha1(f"{run_id}::{seed}".encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def _build_profile_from_orders(
    *,
    profile_id: str,
    source: str,
    order_ids: Iterable[int],
    order_pool: OrderPool,
) -> PersonProfile | None:
    order_ids_clean = sorted({int(oid) for oid in order_ids if int(oid) in order_pool.order_product_ids})
    if not order_ids_clean:
        return None
    history_ids: set[int] = set()
    history_counts: dict[int, int] = {}
    history_items: list[str] = []
    seen_names: set[str] = set()
    for oid in order_ids_clean:
        for pid in order_pool.order_product_ids.get(oid, ()):
            pid_int = int(pid)
            history_ids.add(pid_int)
            history_counts[pid_int] = int(history_counts.get(pid_int, 0)) + 1
        for name in order_pool.order_product_names.get(oid, ()):
            text = str(name).strip()
            if text and text not in seen_names:
                seen_names.add(text)
                history_items.append(text)
    if not history_ids:
        return None
    return PersonProfile(
        profile_id=profile_id,
        source=source,
        order_ids=order_ids_clean,
        history_product_ids=sorted(history_ids),
        history_items=history_items,
        created_at="review_seeded",
        history_counts={int(k): int(v) for k, v in sorted(history_counts.items())},
    )


def _build_distinct_profiles(
    *,
    order_pool: OrderPool,
    target_people: int,
    seed: int,
    run_id: str,
) -> list[PersonProfile]:
    rng = random.Random(_stable_seed(run_id, seed))
    profiles: list[PersonProfile] = []
    seen: set[tuple[int, ...]] = set()
    attempts = max(1000, target_people * 300)

    while len(profiles) < target_people and attempts > 0:
        attempts -= 1
        sampled = build_random_profile(order_pool, preferred_orders=2, fallback_orders=1, rng=rng)
        if sampled is None:
            continue
        sig = _profile_signature(sampled)
        if not sig or sig in seen:
            continue
        seen.add(sig)
        profiles.append(
            PersonProfile(
                profile_id=f"review_person_{len(profiles)+1:03d}",
                source=sampled.source,
                order_ids=list(sampled.order_ids),
                history_product_ids=list(sampled.history_product_ids),
                history_items=list(sampled.history_items),
                created_at="review_seeded",
                history_counts=dict(sampled.history_counts),
            )
        )

    # Deterministic single-order fallback if random sampling cannot reach target.
    if len(profiles) < target_people:
        for oid in order_pool.fallback_order_ids:
            if len(profiles) >= target_people:
                break
            fallback_profile = _build_profile_from_orders(
                profile_id=f"review_person_{len(profiles)+1:03d}",
                source="review_single_order",
                order_ids=[int(oid)],
                order_pool=order_pool,
            )
            if fallback_profile is None:
                continue
            sig = _profile_signature(fallback_profile)
            if not sig or sig in seen:
                continue
            seen.add(sig)
            profiles.append(fallback_profile)

    return profiles


def _lane_placeholder(lane: str) -> dict[str, object]:
    return {
        "lane": lane,
        "status": "missing",
        "anchor_product_id": None,
        "anchor_product_name": "",
        "complement_product_id": None,
        "complement_product_name": "",
        "source": "",
        "reason": "",
        "confidence_score": None,
        "hybrid_reco_score": None,
    }


def _compact_history(items: list[str], limit: int = 8) -> str:
    clean = [str(x).strip() for x in items if str(x).strip()]
    if len(clean) <= limit:
        return ", ".join(clean)
    return ", ".join(clean[:limit]) + f" ... (+{len(clean)-limit})"


def _build_review_payload(
    *,
    profiles: list[PersonProfile],
    recommendations: list[dict[str, object]],
    target_people: int,
    seed: int,
    run_id: str,
) -> dict[str, object]:
    profile_by_id = {str(p.profile_id): p for p in profiles}
    rec_by_profile_id: dict[str, dict[str, object]] = {}
    for rec in recommendations:
        if not isinstance(rec, dict):
            continue
        rid = str(rec.get("profile_id", "") or "").strip()
        if rid and rid not in rec_by_profile_id:
            rec_by_profile_id[rid] = rec
    people_rows: list[dict[str, object]] = []

    for profile in profiles:
        profile_id = str(profile.profile_id)
        rec = rec_by_profile_id.get(profile_id, {})
        bundles = rec.get("bundles", [])
        lane_map: dict[str, dict[str, object]] = {}
        matched_count_value: int | None = None
        if isinstance(bundles, list):
            for bundle in bundles:
                if not isinstance(bundle, dict):
                    continue
                if bundle.get("history_match_count") is not None:
                    try:
                        matched_count_value = int(bundle.get("history_match_count"))
                    except (TypeError, ValueError):
                        pass
                lane = str(bundle.get("lane", "")).strip().lower()
                if lane not in FOOD_LANES:
                    continue
                lane_map[lane] = {
                    "lane": lane,
                    "status": "ok",
                    "anchor_product_id": int(bundle.get("anchor_product_id", bundle.get("product_a", -1)))
                    if str(bundle.get("anchor_product_id", bundle.get("product_a", ""))).strip() != ""
                    else None,
                    "anchor_product_name": str(bundle.get("product_a_name", "") or ""),
                    "complement_product_id": int(bundle.get("complement_product_id", bundle.get("product_b", -1)))
                    if str(bundle.get("complement_product_id", bundle.get("product_b", ""))).strip() != ""
                    else None,
                    "complement_product_name": str(bundle.get("product_b_name", "") or ""),
                    "source": str(bundle.get("recommendation_origin", "") or ""),
                    "reason": str(bundle.get("recommendation_origin_label", "") or ""),
                    "confidence_score": float(bundle.get("confidence_score")) if bundle.get("confidence_score") is not None else None,
                    "hybrid_reco_score": float(bundle.get("hybrid_reco_score")) if bundle.get("hybrid_reco_score") is not None else None,
                    "reasons": list(bundle.get("recommendation_reasons", []))
                    if isinstance(bundle.get("recommendation_reasons"), list)
                    else [],
                }

        lane_rows = [lane_map.get(lane, _lane_placeholder(lane)) for lane in FOOD_LANES]
        missing_lanes = [row["lane"] for row in lane_rows if row.get("status") != "ok"]
        order_ids = list(profile.order_ids)
        history_items = list(profile.history_items)

        people_rows.append(
            {
                "profile_id": profile_id,
                "source": str(rec.get("source", profile.source)),
                "order_count": int(len(order_ids)),
                "matched_count": matched_count_value,
                "order_ids": order_ids,
                "history_item_count": int(len(history_items)),
                "history_summary": _compact_history(history_items),
                "history_items": history_items,
                "lanes": lane_rows,
                "missing_lanes": missing_lanes,
                "has_all_three_food_lanes": not missing_lanes,
            }
        )

    payload: dict[str, object] = {
        "generated_at_utc": _utc_now_iso(),
        "run_id": run_id,
        "seed": int(seed),
        "target_people": int(target_people),
        "distinct_profiles_built": int(len(profiles)),
        "recommendations_returned": int(len(recommendations)),
        "people_with_all_three_food_lanes": int(
            sum(1 for person in people_rows if bool(person.get("has_all_three_food_lanes")))
        ),
        "people": people_rows,
    }
    return payload


def _to_bundle_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for person in payload.get("people", []):
        if not isinstance(person, dict):
            continue
        base = {
            "profile_id": str(person.get("profile_id", "")),
            "source": str(person.get("source", "")),
            "order_count": int(person.get("order_count", 0)),
            "matched_count": person.get("matched_count"),
            "history_item_count": int(person.get("history_item_count", 0)),
            "history_summary": str(person.get("history_summary", "")),
        }
        for lane_info in person.get("lanes", []):
            if not isinstance(lane_info, dict):
                continue
            row = dict(base)
            row.update(
                {
                    "lane": str(lane_info.get("lane", "")),
                    "status": str(lane_info.get("status", "")),
                    "anchor_product_id": lane_info.get("anchor_product_id"),
                    "anchor_product_name": str(lane_info.get("anchor_product_name", "")),
                    "complement_product_id": lane_info.get("complement_product_id"),
                    "complement_product_name": str(lane_info.get("complement_product_name", "")),
                    "source_reason": str(lane_info.get("reason", "")),
                    "source_origin": str(lane_info.get("source", "")),
                    "confidence_score": lane_info.get("confidence_score"),
                    "hybrid_reco_score": lane_info.get("hybrid_reco_score"),
                    "reasons": "|".join(str(x) for x in lane_info.get("reasons", [])) if isinstance(lane_info.get("reasons"), list) else "",
                }
            )
            rows.append(row)
    return rows


def _to_markdown(payload: dict[str, object]) -> str:
    people = payload.get("people", [])
    ok_count = int(payload.get("people_with_all_three_food_lanes", 0))
    lines: list[str] = [
        "# People Predictions Review (100, Food Lanes Only)",
        "",
        f"- Run ID: `{payload.get('run_id', '')}`",
        f"- Seed: `{payload.get('seed', '')}`",
        f"- Target people: `{payload.get('target_people', 0)}`",
        f"- Distinct profiles built: `{payload.get('distinct_profiles_built', 0)}`",
        f"- Recommendations returned: `{payload.get('recommendations_returned', 0)}`",
        f"- People with all 3 food lanes: `{ok_count}`",
        "",
        "## Sample (first 20 people)",
        "",
    ]

    if not isinstance(people, list) or not people:
        lines.append("_No people recommendations were generated._")
        return "\n".join(lines)

    for person in people[:20]:
        if not isinstance(person, dict):
            continue
        lines.append(f"### {person.get('profile_id', '')}")
        lines.append(f"- Orders: `{person.get('order_count', 0)}`")
        if person.get("matched_count") is not None:
            lines.append(f"- Matched count: `{person.get('matched_count')}`")
        lines.append(f"- History: {person.get('history_summary', '')}")
        missing = person.get("missing_lanes", [])
        if isinstance(missing, list) and missing:
            lines.append(f"- Missing lanes: `{', '.join(str(x) for x in missing)}`")
        lane_rows = person.get("lanes", [])
        if isinstance(lane_rows, list):
            for lane in lane_rows:
                if not isinstance(lane, dict):
                    continue
                lane_name = str(lane.get("lane", "")).upper()
                if lane.get("status") != "ok":
                    lines.append(f"  - {lane_name}: MISSING")
                    continue
                lines.append(
                    f"  - {lane_name}: {lane.get('anchor_product_name', '')} + {lane.get('complement_product_name', '')}"
                    f" | source={lane.get('source', '')}"
                    f" | score={lane.get('hybrid_reco_score', '')}"
                )
        lines.append("")
    return "\n".join(lines)


def generate_people_predictions_review(
    *,
    base_dir: Path | None = None,
    target_people: int = 100,
    seed: int = 20260307,
    run_id: str = "review_people_100",
) -> dict[str, str]:
    paths = get_paths(project_root=base_dir)
    ensure_layout(paths)

    view = load_bundle_view(paths.project_root)
    bundles_df = view.bundles_df if view.bundles_df is not None else pd.DataFrame()
    if bundles_df.empty:
        raise RuntimeError("No candidate bundle data found. Run pipeline first to generate person_candidates_scored.csv.")

    order_pool = load_order_pool(paths.project_root)
    profiles = _build_distinct_profiles(order_pool=order_pool, target_people=int(target_people), seed=int(seed), run_id=str(run_id))

    base_build_random_profile = build_random_profile

    def _deterministic_resample_profile(
        order_pool_arg: OrderPool,
        preferred_orders: int = 2,
        fallback_orders: int = 1,
        rng: random.Random | None = None,
    ) -> PersonProfile | None:
        sampled = base_build_random_profile(
            order_pool_arg,
            preferred_orders=preferred_orders,
            fallback_orders=fallback_orders,
            rng=rng,
        )
        if sampled is None:
            return None
        sig = _profile_signature(sampled)
        sig_hash = hashlib.sha1(",".join(str(x) for x in sig).encode("utf-8")).hexdigest()[:12]
        return PersonProfile(
            profile_id=f"resample_{sig_hash}",
            source=sampled.source,
            order_ids=list(sampled.order_ids),
            history_product_ids=list(sampled.history_product_ids),
            history_items=list(sampled.history_items),
            created_at="review_seeded",
            history_counts=dict(sampled.history_counts),
        )

    with patch("qeu_bundling.presentation.person_predictions.NONFOOD_INCLUDE_RATE", 0.0):
        with patch(
            "qeu_bundling.presentation.person_predictions.build_random_profile",
            side_effect=_deterministic_resample_profile,
        ):
            recommendations = build_recommendations_for_profiles(
                bundles_df=bundles_df,
                profiles=profiles,
                max_people=len(profiles),
                row_to_record=row_to_record,
                base_dir=paths.project_root,
                run_id=run_id,
                rng_salt=f"review_seed_{seed}",
            )

    payload = _build_review_payload(
        profiles=profiles,
        recommendations=recommendations,
        target_people=int(target_people),
        seed=int(seed),
        run_id=str(run_id),
    )
    bundle_rows = _to_bundle_rows(payload)
    summary_md = _to_markdown(payload)

    review_dir = paths.output_review_dir
    review_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"people_predictions_review_{int(target_people)}"
    json_path = review_dir / f"{prefix}.json"
    csv_path = review_dir / f"{prefix}_bundles.csv"
    md_path = review_dir / f"{prefix}_summary.md"

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(bundle_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(summary_md, encoding="utf-8")

    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "distinct_profiles_built": str(len(profiles)),
        "recommendations_returned": str(len(recommendations)),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate people predictions review export (food lanes only).")
    parser.add_argument("--target-people", type=int, default=100, help="Target distinct people to review.")
    parser.add_argument("--seed", type=int, default=20260307, help="Deterministic seed for profile generation.")
    parser.add_argument("--run-id", type=str, default="review_people_100", help="Run id tag embedded in output payload.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = generate_people_predictions_review(
        target_people=max(1, int(args.target_people)),
        seed=int(args.seed),
        run_id=str(args.run_id).strip() or "review_people_100",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
