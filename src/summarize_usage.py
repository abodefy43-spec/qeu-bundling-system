import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def to_int(value: str) -> int:
    try:
        return int(str(value).replace(",", "").strip() or 0)
    except (TypeError, ValueError):
        return 0


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(rows: List[Dict[str, str]]) -> Dict[str, object]:
    included = [r for r in rows if r.get("Kind", "").strip().lower() == "included"]
    by_model = defaultdict(int)
    for row in included:
        model = row.get("Model", "unknown")
        by_model[model] += to_int(row.get("Total Tokens", 0))

    model_ranking = sorted(
        by_model.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    return {
        "models": model_ranking,
    }


def build_report(csv_path: Path, summary: Dict[str, object]) -> str:
    lines = [
        "============================================================",
        "Total Usage Tokens by Model",
        "============================================================",
        f"CSV File: {csv_path}",
        "",
        "Model,Total Tokens",
    ]

    for model, tokens in summary["models"]:
        lines.append(f"{model},{tokens:,}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize usage events CSV.")
    parser.add_argument(
        "--csv",
        default=r"d:\usage-events-2026-02-28.csv",
        help="Path to usage-events CSV file.",
    )
    parser.add_argument(
        "--save",
        default="",
        help="Optional output text file path for saving the summary.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    rows = load_rows(csv_path)
    summary = summarize(rows)
    report = build_report(csv_path, summary)
    print(report)

    if args.save:
        save_path = Path(args.save)
        os.makedirs(save_path.parent, exist_ok=True)
        save_path.write_text(report, encoding="utf-8")
        print("")
        print(f"Saved summary to: {save_path}")


if __name__ == "__main__":
    main()
