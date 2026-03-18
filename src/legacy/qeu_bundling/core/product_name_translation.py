"""Utilities for translating Arabic product names to English."""

from __future__ import annotations

import json
import re
from pathlib import Path

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_WHITESPACE_RE = re.compile(r"\s+")

_CACHE_BY_PATH: dict[str, dict[str, str]] = {}
_TRANSLATOR = None
_WARNED_MISSING_TRANSLATOR = False


def _normalise_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(value)).strip()


def _cache_key(cache_path: Path) -> str:
    return str(cache_path.resolve())


def _load_cache(cache_path: Path) -> dict[str, str]:
    key = _cache_key(cache_path)
    if key in _CACHE_BY_PATH:
        return _CACHE_BY_PATH[key]

    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                parsed = json.load(f)
            if isinstance(parsed, dict):
                cache = {str(k): _normalise_text(v) for k, v in parsed.items() if str(k).strip()}
            else:
                cache = {}
        except (json.JSONDecodeError, OSError):
            cache = {}
    else:
        cache = {}

    _CACHE_BY_PATH[key] = cache
    return cache


def _save_cache(cache_path: Path, cache: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _get_translator():
    global _TRANSLATOR, _WARNED_MISSING_TRANSLATOR
    if _TRANSLATOR is not None:
        return _TRANSLATOR

    try:
        from deep_translator import GoogleTranslator

        _TRANSLATOR = GoogleTranslator(source="ar", target="en")
        return _TRANSLATOR
    except Exception:
        if not _WARNED_MISSING_TRANSLATOR:
            print("  WARNING: deep-translator unavailable; Arabic names will remain unchanged.")
            _WARNED_MISSING_TRANSLATOR = True
        return None


def _is_arabic(text: str) -> bool:
    """Return True when the input contains Arabic-script characters."""
    return bool(_ARABIC_RE.search(str(text)))


def translate_arabic_to_english(text: str, cache_path: Path) -> str:
    """Translate Arabic product name to English with a persistent cache."""
    original = _normalise_text(text)
    if not original or not _is_arabic(original):
        return original

    cache = _load_cache(cache_path)
    cached = cache.get(original)
    if cached:
        return cached

    translator = _get_translator()
    if translator is None:
        return original

    try:
        translated = _normalise_text(translator.translate(original))
    except Exception:
        translated = ""

    if not translated:
        return original

    cache[original] = translated
    _save_cache(cache_path, cache)
    return translated

