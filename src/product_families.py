"""Shared product-family taxonomy utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9\u0600-\u06FF]+", " ", str(text).lower()).strip()


def load_families(config_path: Path) -> list[dict[str, Any]]:
    """Load and normalise family definitions from JSON config."""
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    families: list[dict[str, Any]] = []
    for entry in payload.get("families", []):
        family_id = str(entry.get("id", "")).strip()
        if not family_id:
            continue
        raw_keywords = entry.get("keywords", []) or []
        keywords = sorted(
            {
                _normalise(str(k))
                for k in raw_keywords
                if _normalise(str(k))
            },
            key=len,
            reverse=True,
        )
        if not keywords:
            continue
        families.append(
            {
                "id": family_id,
                "description": str(entry.get("description", "")).strip(),
                "keywords": keywords,
            }
        )
    return families


def assign_family(product_name: str, families: list[dict[str, Any]]) -> str:
    """Assign first matching family id for a product name."""
    norm = _normalise(product_name)
    if not norm:
        return ""

    for family in families:
        for kw in family.get("keywords", []):
            if kw and kw in norm:
                return str(family.get("id", ""))
    return ""
