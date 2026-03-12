"""Phase 3: Co-purchase analysis — products bought together in the same order."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import pandas as pd
from qeu_bundling.config.paths import get_paths
from qeu_bundling.core.schema_validation import (
    DataValidationError,
    NumericRangeRule,
    require_columns,
    require_not_empty,
    require_no_nulls,
    validate_numeric_ranges,
)

MIN_PAIR_COUNT = 2


def _data_dir() -> Path:
    return get_paths().data_processed_dir


def compute_copurchase_scores(
    data_dir: Path | None = None,
    min_pair_count: int = MIN_PAIR_COUNT,
    max_order_size: int = 10,
) -> pd.DataFrame:
    """Build co-purchase pair scores normalised to 0-100.

    Parameters
    ----------
    min_pair_count : int
        Minimum times a pair must co-occur to be kept.
    max_order_size : int
        Skip orders with more than this many *distinct* products
        (they generate too many spurious pairs).
    """
    base = data_dir or _data_dir()
    orders = pd.read_pickle(base / "filtered_orders.pkl")
    require_columns(orders, ["order_id", "product_id"], artifact_name="phase_03.filtered_orders")
    require_not_empty(orders, artifact_name="phase_03.filtered_orders")
    require_no_nulls(orders, ["order_id", "product_id"], artifact_name="phase_03.filtered_orders")

    grouped = (
        orders.groupby("order_id")["product_id"]
        .apply(lambda x: set(x.dropna().astype(int)))
        .reset_index()
    )
    grouped = grouped[grouped["product_id"].apply(len).between(2, max_order_size)]
    print(f"  Orders with 2-{max_order_size} distinct products: {len(grouped):,}")

    pair_counts: dict[tuple[int, int], int] = {}
    for products in grouped["product_id"]:
        for a, b in combinations(sorted(products), 2):
            key = (a, b)
            pair_counts[key] = pair_counts.get(key, 0) + 1

    pairs = pd.DataFrame(
        [(a, b, c) for (a, b), c in pair_counts.items()],
        columns=["product_a", "product_b", "pair_count"],
    )
    pairs = pairs[pairs["pair_count"] >= min_pair_count].copy()
    if pairs.empty:
        print("  WARNING: no co-purchase pairs found.")
        pairs["score"] = pd.Series(dtype=float)
    else:
        max_count = pairs["pair_count"].max()
        pairs["score"] = (pairs["pair_count"] / max_count * 100).round(2)

    pairs = pairs.sort_values("score", ascending=False).reset_index(drop=True)
    if not pairs.empty:
        validate_numeric_ranges(
            pairs,
            rules={
                "pair_count": NumericRangeRule(minimum=1, allow_null=False),
                "score": NumericRangeRule(minimum=0.0, maximum=100.0, allow_null=False),
            },
            artifact_name="phase_03.copurchase_scores",
        )

    out_path = base / "copurchase_scores.csv"
    pairs.to_csv(out_path, index=False)
    print(f"  Saved {len(pairs):,} pairs -> {out_path}")
    return pairs


def load_copurchase_scores(data_dir: Path | None = None) -> pd.DataFrame:
    base = data_dir or _data_dir()
    return pd.read_csv(base / "copurchase_scores.csv")


def run():
    return compute_copurchase_scores()


if __name__ == "__main__":
    print("Phase 3: Computing co-purchase scores ...")
    df = compute_copurchase_scores()
    print(f"  Top 10 pairs:\n{df.head(10).to_string()}")
    print("Phase 3 complete.")
