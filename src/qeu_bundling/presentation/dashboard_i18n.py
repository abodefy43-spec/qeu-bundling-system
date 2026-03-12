"""Dashboard-only i18n helpers for Arabic-first presentation."""

from __future__ import annotations

from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
from pathlib import Path
import re

_WHITESPACE_RE = re.compile(r"\s+")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_NO_LETTER_RE = re.compile(r"^[\d\s\W_]+$")
_CURRENCY_TOKEN_RE = re.compile(r"\b(sar|usd|aed|qar|riyal|riyal)\b", re.IGNORECASE)

_CACHE_BY_PATH: dict[str, dict[str, str]] = {}
_TRANSLATOR = None
_WARNED_TRANSLATOR_MISSING = False
_REQUEST_TRANSLATION_BUDGET: int | None = None
_MAX_TRANSLATIONS_PER_REQUEST = 8
_TRANSLATE_TIMEOUT_SECONDS = 0.35


def _normalise_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "")).strip()


def _cache_key(cache_path: Path) -> str:
    return str(cache_path.resolve())


def _load_cache(cache_path: Path) -> dict[str, str]:
    key = _cache_key(cache_path)
    if key in _CACHE_BY_PATH:
        return _CACHE_BY_PATH[key]
    if cache_path.exists():
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, dict):
                cache = {
                    _normalise_text(str(k)): _normalise_text(str(v))
                    for k, v in payload.items()
                    if _normalise_text(str(k))
                }
            else:
                cache = {}
        except (OSError, json.JSONDecodeError):
            cache = {}
    else:
        cache = {}
    _CACHE_BY_PATH[key] = cache
    return cache


def _save_cache(cache_path: Path, cache: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as fh:
        json.dump(cache, fh, ensure_ascii=False, indent=2)


def _get_translator():
    global _TRANSLATOR, _WARNED_TRANSLATOR_MISSING
    if _TRANSLATOR is not None:
        return _TRANSLATOR
    try:
        from deep_translator import GoogleTranslator

        _TRANSLATOR = GoogleTranslator(source="en", target="ar")
        return _TRANSLATOR
    except Exception:
        if not _WARNED_TRANSLATOR_MISSING:
            print("  WARNING: deep-translator unavailable; dashboard Arabic fallback will use original text.")
            _WARNED_TRANSLATOR_MISSING = True
        return None


def _translate_with_timeout(translator, text: str) -> str:
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(translator.translate, text)
        return _normalise_text(str(future.result(timeout=_TRANSLATE_TIMEOUT_SECONDS)))
    except FutureTimeoutError:
        try:
            future.cancel()
        except Exception:
            pass
        return ""
    except Exception:
        return ""
    finally:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass


def _skip_translation(text: str) -> bool:
    cleaned = _normalise_text(text)
    if not cleaned:
        return True
    if _ARABIC_RE.search(cleaned):
        return True
    if _NO_LETTER_RE.match(cleaned):
        return True
    cleaned_no_currency = _normalise_text(_CURRENCY_TOKEN_RE.sub(" ", cleaned))
    if cleaned_no_currency and _NO_LETTER_RE.match(cleaned_no_currency):
        return True
    return False


def translate_en_to_ar(text: str, cache_path: Path) -> str:
    """Translate a dashboard display string to Arabic with persistent caching."""
    cleaned = _normalise_text(text)
    if _skip_translation(cleaned):
        return cleaned

    cache = _load_cache(cache_path)
    cached = cache.get(cleaned)
    if cached:
        return cached

    global _REQUEST_TRANSLATION_BUDGET
    if _REQUEST_TRANSLATION_BUDGET is not None and _REQUEST_TRANSLATION_BUDGET <= 0:
        return cleaned

    translator = _get_translator()
    if translator is None:
        return cleaned

    if _REQUEST_TRANSLATION_BUDGET is not None:
        _REQUEST_TRANSLATION_BUDGET -= 1
    translated = _translate_with_timeout(translator, cleaned)

    if not translated:
        return cleaned

    cache[cleaned] = translated
    _save_cache(cache_path, cache)
    return translated


def _translate_text_list(values: object, cache_path: Path) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    for item in values:
        raw = _normalise_text(str(item))
        if raw:
            out.append(translate_en_to_ar(raw, cache_path))
    return out


def translate_bundle_for_dashboard(bundle: dict, cache_path: Path) -> dict:
    """Return a translated dashboard copy of one bundle payload."""
    out = deepcopy(bundle)
    product_a_name = _normalise_text(str(out.get("product_a_name", "")))
    product_b_name = _normalise_text(str(out.get("product_b_name", "")))
    out["product_a_name_ar"] = translate_en_to_ar(product_a_name, cache_path)
    out["product_b_name_ar"] = translate_en_to_ar(product_b_name, cache_path)

    chosen_names = out.get("chosen_bundle_names", [])
    out["chosen_bundle_names_ar"] = _translate_text_list(chosen_names, cache_path)

    reasons = out.get("recommendation_reasons", [])
    out["recommendation_reasons_ar"] = _translate_text_list(reasons, cache_path)

    origin_label = _normalise_text(str(out.get("recommendation_origin_label", "")))
    out["recommendation_origin_label_ar"] = translate_en_to_ar(origin_label, cache_path)

    translated_items: list[dict] = []
    for item in out.get("bundle_items", []) if isinstance(out.get("bundle_items", []), list) else []:
        if not isinstance(item, dict):
            continue
        item_out = deepcopy(item)
        item_name = _normalise_text(str(item_out.get("name", "")))
        item_out["name_ar"] = translate_en_to_ar(item_name, cache_path)
        translated_items.append(item_out)
    out["bundle_items"] = translated_items
    return out


def translate_recommendations_for_dashboard(recos: list[dict], cache_path: Path) -> list[dict]:
    """Translate recommendation display fields for dashboard rendering only."""
    global _REQUEST_TRANSLATION_BUDGET
    _REQUEST_TRANSLATION_BUDGET = _MAX_TRANSLATIONS_PER_REQUEST
    output: list[dict] = []
    for rec in recos or []:
        if not isinstance(rec, dict):
            continue
        rec_out = deepcopy(rec)
        history_items = rec_out.get("history_items", [])
        rec_out["history_items_ar"] = _translate_text_list(history_items, cache_path)

        translated_bundles: list[dict] = []
        for bundle in rec_out.get("bundles", []) if isinstance(rec_out.get("bundles", []), list) else []:
            if not isinstance(bundle, dict):
                continue
            translated_bundles.append(translate_bundle_for_dashboard(bundle, cache_path))
        rec_out["bundles"] = translated_bundles
        output.append(rec_out)
    return output


def dashboard_ui_text_ar() -> dict[str, object]:
    """Arabic-first UI dictionary for the dashboard template."""
    return {
        "header_title_ar": "نظام تجميع منتجات QEU",
        "header_title_en": "QEU Product Bundling",
        "header_subtitle_ar": "لوحة احترافية لتوصيات الباقات والتحكم بالتحديث",
        "header_subtitle_en": "Executive bundle insights & refresh controls",
        "dashboard_nav_ar": "لوحة التحكم",
        "dashboard_nav_en": "Dashboard",
        "people_predictions_title_ar": "توصيات مخصصة حسب الأشخاص",
        "people_predictions_title_en": "People-Based Predictions",
        "personalized_predictions_ar": "توقعات الباقات المخصصة",
        "personalized_predictions_en": "personalized bundle predictions",
        "new_prediction_ar": "إضافة شخص",
        "new_prediction_en": "New Prediction",
        "add_five_predictions_ar": "إضافة ٥ أشخاص",
        "shuffle_all_ar": "تبديل الأشخاص",
        "shuffle_all_en": "Shuffle All",
        "refresh_data_ar": "تحديث التوصيات",
        "refresh_data_en": "Refresh Data",
        "refresh_running_ar": "يتم التحديث...",
        "refresh_running_en": "Refresh Running...",
        "random_person_ar": "شخص عشوائي",
        "random_person_en": "Random Person",
        "paste_orders_ar": "إدخال الطلبات",
        "paste_orders_en": "Paste Orders",
        "generate_random_person_ar": "توليد شخص عشوائي",
        "generate_random_person_en": "Generate Random Person",
        "create_person_from_input_ar": "إنشاء شخص من الإدخال",
        "create_person_from_input_en": "Create Person From Input",
        "history_label_ar": "السجل",
        "history_label_en": "History",
        "products_ar": "منتجات",
        "products_en": "products",
        "orders_ar": "طلبات",
        "orders_en": "orders",
        "matched_ar": "مطابق",
        "matched_en": "matched",
        "bundles_ar": "باقات",
        "bundles_en": "bundles",
        "bundle_label_ar": "الباقة",
        "bundle_label_en": "Bundle",
        "free_label_ar": "المجاني",
        "free_label_en": "Free",
        "confidence_ar": "ثقة",
        "confidence_en": "confidence",
        "copurchase_fallback_ar": "شراء مشترك",
        "copurchase_fallback_en": "Copurchase fallback",
        "like_ar": "إعجاب",
        "like_en": "Like",
        "dislike_ar": "عدم إعجاب",
        "dislike_en": "Dislike",
        "wrong_pair_ar": "زوج غير مناسب",
        "wrong_pair_en": "Wrong Pair",
        "too_expensive_ar": "مرتفع السعر",
        "too_expensive_en": "Too Expensive",
        "regenerate_ar": "إعادة التوليد",
        "regenerate_en": "Regenerate",
        "empty_state_ar": "لا توجد توقعات للأشخاص حتى الآن.",
        "empty_state_en": "No person-level predictions available yet.",
        "composer_note_ar": "توليد شخص جديد من الطلبات التاريخية (يحاول طلبين ثم يعود لطلب واحد).",
        "composer_note_en": "Generate a new person from historical orders (tries 2 orders, falls back to 1).",
        "manual_label_ar": "ألصق الطلبات (كل طلب في سطر، والمنتجات مفصولة بفواصل)",
        "manual_label_en": "Paste orders (one order per line, items separated by commas)",
        "manual_hint_ar": "أمثلة: tuna, mayonnaise و bread, chips. يمكن أيضًا استخدام الأرقام.",
        "manual_hint_en": "Examples: tuna, mayonnaise and bread, chips. Numeric IDs are also accepted.",
        "lane_meal_ar": "وجبة",
        "lane_meal_en": "MEAL",
        "lane_snack_ar": "سناك",
        "lane_snack_en": "SNACK",
        "lane_occasion_ar": "مناسبة",
        "lane_occasion_en": "OCCASION",
        "lane_nonfood_ar": `غير غذائي",
        "lane_nonfood_en": "NONFOOD",
        "main_product_ar": "المنتج الرئيسي",
        "main_product_en": "Main Product",
        "free_product_ar": "المنتج المجاني",
        "free_product_en": "Free Product",
    }
