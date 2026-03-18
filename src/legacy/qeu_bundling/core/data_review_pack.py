"""Build a human + machine review pack for people-only recommendation outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qeu_bundling.config.paths import ensure_layout, get_paths
from qeu_bundling.core.run_manifest import read_latest_manifest, resolve_latest_artifact

STAPLE_TOKENS = frozenset({"rice", "oil", "salt", "water", "sugar"})


def _safe_read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _normalise_text(value: object) -> str:
    text = str(value or "").strip().lower()
    return " ".join(text.split())


def _staple_flags(name_a: str, name_b: str) -> str:
    tokens = set(_normalise_text(name_a).split()) | set(_normalise_text(name_b).split())
    flags = sorted(tokens & STAPLE_TOKENS)
    return "|".join(flags)


def _write_markdown(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def _build_data_inventory(paths, manifest: dict[str, object], artifact_rows: list[tuple[str, Path]]) -> str:
    lines: list[str] = [
        "# Data Inventory",
        "",
        "## Core Inputs",
        "",
    ]
    input_targets = [
        ("Raw Orders", paths.data_raw_dir / "order_items.csv"),
        ("Recipe Reference", paths.data_reference_dir / "recipe_data.json"),
        ("Families Reference", paths.data_reference_dir / "product_families.json"),
        ("Theme Tokens", paths.data_reference_dir / "theme_tokens.json"),
        ("Category Importance", paths.data_reference_dir / "category_importance.csv"),
        ("Pair Penalty Rules", paths.data_reference_dir / "pair_penalty_rules.json"),
    ]
    for label, file_path in input_targets:
        status = "exists" if file_path.exists() else "missing"
        lines.append(f"- **{label}**: `{file_path}` ({status})")

    lines.extend(["", "## Current Run Artifacts", ""])
    run_id = str(manifest.get("run_id", "") or "")
    mode = str(manifest.get("mode", "") or "")
    seed = manifest.get("seed", "")
    if run_id:
        lines.append(f"- Run ID: `{run_id}`")
    if mode:
        lines.append(f"- Mode: `{mode}`")
    if seed != "":
        lines.append(f"- Seed: `{seed}`")
    for label, file_path in artifact_rows:
        exists = file_path.exists()
        suffix = f"{file_path.stat().st_size:,} bytes" if exists else "missing"
        lines.append(f"- **{label}**: `{file_path}` ({suffix})")
    lines.append("")
    return "\n".join(lines)


def _build_data_lineage(manifest: dict[str, object]) -> str:
    lines: list[str] = [
        "# Data Lineage",
        "",
        "This report traces the latest pipeline run and phase outputs used by people recommendations.",
        "",
    ]
    run_id = str(manifest.get("run_id", "") or "")
    mode = str(manifest.get("mode", "") or "")
    started = str(manifest.get("started_at", "") or "")
    finished = str(manifest.get("finished_at", "") or "")
    if run_id:
        lines.append(f"- Run ID: `{run_id}`")
    if mode:
        lines.append(f"- Mode: `{mode}`")
    if started:
        lines.append(f"- Started: `{started}`")
    if finished:
        lines.append(f"- Finished: `{finished}`")
    lines.append("")

    phases = manifest.get("phases", [])
    if isinstance(phases, list) and phases:
        lines.append("## Phase Status")
        lines.append("")
        for row in phases:
            if not isinstance(row, dict):
                continue
            phase = str(row.get("phase", "") or "")
            status = str(row.get("status", "") or "")
            duration = row.get("duration_sec", "")
            lines.append(f"- `{phase}`: `{status}` ({duration}s)")
    else:
        lines.append("_No phase records found in manifest._")
    lines.append("")
    return "\n".join(lines)


def _build_scoring_explanation() -> str:
    return "\n".join(
        [
            "# Scoring Explanation",
            "",
            "## Item 1 (Anchor) selection",
            "- History frequency + recipe signal drive anchor ranking.",
            "- Non-food and same-family constraints stay hard-blocked.",
            "",
            "## Item 2 (Complement) selection",
            "- Strict complement gate requires strong evidence from recipe-compatibility, copurchase, or known-prior match.",
            "- Utility-like complements with weak evidence are rejected.",
            "",
            "## Score controls",
            "- Feedback multipliers slightly up/down-weight known good/bad pairs.",
            "- Brand preference can boost in-family preferred brand complements.",
            "- Pair penalty rules can down-rank suspicious pair patterns (for example, staple mismatches).",
            "",
            "## Staple dominance handling",
            "- Staples are tracked (`rice/oil/salt/water/sugar`) and surfaced in audit output.",
            "- Suspicious pair rules are soft penalties (not hard blocks), so high-evidence pairs can still survive when justified.",
            "",
        ]
    )


def generate_review_pack(base_dir: Path | None = None) -> dict[str, str]:
    paths = get_paths(project_root=base_dir)
    ensure_layout(paths)

    manifest = read_latest_manifest(base_dir=base_dir)
    person_candidates = resolve_latest_artifact(
        "person_candidates_scored",
        base_dir=base_dir,
        fallback=paths.output_dir / "person_candidates_scored.csv",
    )
    pair_candidates = resolve_latest_artifact(
        "person_candidate_pairs",
        base_dir=base_dir,
        fallback=paths.data_processed_candidates_dir / "person_candidate_pairs.csv",
    )
    suspicious_audit = resolve_latest_artifact(
        "suspicious_pairs_audit",
        base_dir=base_dir,
        fallback=paths.data_processed_diagnostics_dir / "suspicious_pairs_audit.csv",
    )
    recipe_scores = paths.data_processed_dir / "product_recipe_scores.csv"
    categories = paths.data_processed_dir / "product_categories.csv"
    feedback = paths.data_processed_dir / "person_feedback.csv"
    quality = paths.output_dir / "person_reco_quality.json"

    review_dir = paths.output_review_dir
    review_dir.mkdir(parents=True, exist_ok=True)

    person_df = _safe_read_csv(person_candidates)
    pair_df = _safe_read_csv(pair_candidates)
    suspicious_df = _safe_read_csv(suspicious_audit)
    recipe_df = _safe_read_csv(recipe_scores)
    category_df = _safe_read_csv(categories)

    artifact_rows = [
        ("People Candidates (Scored)", person_candidates or Path("")),
        ("Pair Candidates", pair_candidates or Path("")),
        ("Suspicious Pair Audit", suspicious_audit or Path("")),
        ("Recipe Scores", recipe_scores),
        ("Categories", categories),
        ("Feedback", feedback),
        ("Person Quality", quality),
    ]
    inventory_md = review_dir / "data_inventory.md"
    lineage_md = review_dir / "data_lineage.md"
    scoring_md = review_dir / "scoring_explanation.md"
    _write_markdown(inventory_md, _build_data_inventory(paths, manifest, artifact_rows))
    _write_markdown(lineage_md, _build_data_lineage(manifest))
    _write_markdown(scoring_md, _build_scoring_explanation())

    usage_rows = [
        ("filtered_orders.pkl", "order_id, product_id, product_name, unit_price", "phase_01, personalization context"),
        ("copurchase_scores.csv", "product_a, product_b, score", "phase_03, phase_06, personalization fallback"),
        ("product_recipe_scores.csv", "product_id, recipe_score, matched_ingredient", "phase_05, phase_06, personalization anchor/compat"),
        ("product_categories.csv", "product_id, category, category_tags, product_family", "phase_04, filters, family/brand logic"),
        ("person_candidate_pairs.csv", "pair-level ranking features", "phase_06 output to phase_08 + dashboard"),
        ("person_candidates_scored.csv", "discount/free-item enriched candidates", "phase_08 output used by dashboard"),
    ]
    usage_df = pd.DataFrame(usage_rows, columns=["artifact", "key_columns", "used_by"])
    usage_csv = review_dir / "column_usage_matrix.csv"
    usage_df.to_csv(usage_csv, index=False, encoding="utf-8-sig")

    product_matches_csv = review_dir / "product_ingredient_matches.csv"
    wanted_match_cols = [
        "product_id",
        "product_name",
        "matched_ingredient",
        "final_matched_ingredient",
        "recipe_score",
        "ingredient_match_quality",
        "raw_tokens",
        "filtered_tokens",
        "match_reason",
    ]
    match_cols = [c for c in wanted_match_cols if (not recipe_df.empty and c in recipe_df.columns)]
    if recipe_df.empty or not match_cols:
        pd.DataFrame(columns=wanted_match_cols).to_csv(product_matches_csv, index=False, encoding="utf-8-sig")
    else:
        recipe_df[match_cols].to_csv(product_matches_csv, index=False, encoding="utf-8-sig")
    breakdown_cols = [
        "product_a",
        "product_b",
        "product_a_name",
        "product_b_name",
        "product_family_a",
        "product_family_b",
        "category_a",
        "category_b",
        "purchase_score",
        "pair_count",
        "recipe_score_a",
        "recipe_score_b",
        "recipe_score_norm",
        "recipe_compat_score",
        "known_prior_flag",
        "gate_pass_reason",
        "recipe_overlap_tokens",
        "only_staples_overlap",
        "anchor_score",
        "complement_score",
        "deal_signal",
        "deal_boost_applied",
        "weak_evidence_free_blocked",
        "final_score",
        "new_final_score",
        "price_ratio_b_to_a",
    ]
    pair_breakdown_csv = review_dir / "pair_scoring_breakdown.csv"
    if not pair_df.empty:
        keep = [c for c in breakdown_cols if c in pair_df.columns]
        pair_df[keep].head(5000).to_csv(pair_breakdown_csv, index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame(columns=breakdown_cols).to_csv(pair_breakdown_csv, index=False, encoding="utf-8-sig")

    suspicious_csv = review_dir / "suspicious_pairs_audit.csv"
    if not suspicious_df.empty:
        out = suspicious_df.copy()
    else:
        out = pd.DataFrame(
            columns=[
                "anchor_product_id",
                "complement_product_id",
                "anchor_name",
                "complement_name",
                "rule_id",
                "rule_type",
                "reason",
                "override_condition_met",
                "pair_count",
                "copurchase_score",
                "anchor_family",
                "complement_family",
                "anchor_category",
                "complement_category",
                "recipe_score_anchor",
                "recipe_score_complement",
                "recipe_score_norm",
                "recipe_overlap_tokens",
                "only_staples_overlap",
                "purchase_score",
                "complement_gate_reason",
                "dominant_ingredient_flags",
                "penalty_rule",
                "penalty_multiplier",
                "raw_score_before_penalty",
                "raw_score_after_penalty",
            ]
        )

    if not out.empty and "dominant_ingredient_flags" not in out.columns:
        out["dominant_ingredient_flags"] = out.apply(
            lambda r: _staple_flags(r.get("anchor_name", ""), r.get("complement_name", "")),
            axis=1,
        )
    out.to_csv(suspicious_csv, index=False, encoding="utf-8-sig")

    # Keep diagnostics synchronized in processed diagnostics folder.
    diag_suspicious = paths.data_processed_diagnostics_dir / "suspicious_pairs_audit.csv"
    out.to_csv(diag_suspicious, index=False, encoding="utf-8-sig")

    summary = {
        "data_inventory": str(inventory_md),
        "data_lineage": str(lineage_md),
        "scoring_explanation": str(scoring_md),
        "column_usage_matrix": str(usage_csv),
        "product_ingredient_matches": str(product_matches_csv),
        "pair_scoring_breakdown": str(pair_breakdown_csv),
        "suspicious_pairs_audit": str(suspicious_csv),
    }
    return summary


def main() -> int:
    payload = generate_review_pack()
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
