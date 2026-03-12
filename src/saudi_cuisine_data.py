"""Utilities for loading and using Saudi cuisine data in bundling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DATA: dict[str, Any] | None = None


def _get_data_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "saudi_cuisine_data.json"


def load_cuisine_data() -> dict[str, Any]:
    """Load the Saudi cuisine data from JSON file."""
    global _DATA
    if _DATA is not None:
        return _DATA
    
    path = _get_data_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            _DATA = json.load(f)
    else:
        _DATA = {"saudi_dishes": {}, "food_pairings": {}, "common_ingredients": [], "category_templates": {}}
    
    return _DATA


def get_dish_ingredients(dish_name: str) -> list[str]:
    """Get ingredients for a specific Saudi dish."""
    data = load_cuisine_data()
    dish = data.get("saudi_dishes", {}).get(dish_name.lower(), {})
    return dish.get("ingredients", [])


def get_dish_tags(dish_name: str) -> list[str]:
    """Get tags for a specific Saudi dish."""
    data = load_cuisine_data()
    dish = data.get("saudi_dishes", {}).get(dish_name.lower(), {})
    return dish.get("tags", [])


def get_food_pairings() -> dict[str, list[dict]]:
    """Get all food pairing definitions."""
    data = load_cuisine_data()
    return data.get("food_pairings", {})


def get_pairing_items() -> list[tuple[str, str, str]]:
    """Get all food pairing items as (item_a, item_b, type) tuples.
    
    Returns:
        List of tuples: (product_keyword_a, product_keyword_b, pairing_type)
    """
    data = load_cuisine_data()
    pairings = data.get("food_pairings", {})
    
    items = []
    for pairing_type, pairing_data in pairings.items():
        pairs_list = pairing_data.get("pairs", [])
        for pair in pairs_list:
            items.append((
                pair.get("item_a", ""),
                pair.get("item_b", ""),
                pair.get("type", pairing_type)
            ))
    
    return items


def get_common_ingredients() -> list[str]:
    """Get list of common Saudi cuisine ingredients."""
    data = load_cuisine_data()
    return data.get("common_ingredients", [])


def get_category_templates() -> dict[str, list[str]]:
    """Get category templates for Saudi dishes."""
    data = load_cuisine_data()
    return data.get("category_templates", {})


def get_enhanced_tags_for_product(product_name: str) -> set[str]:
    """Get enhanced tags for a product based on Saudi cuisine data.
    
    This function analyzes a product name and returns matching Saudi cuisine tags.
    """
    data = load_cuisine_data()
    product_lower = product_name.lower()
    
    tags = set()
    
    for dish_name, dish_data in data.get("saudi_dishes", {}).items():
        for ingredient in dish_data.get("ingredients", []):
            if ingredient.lower() in product_lower:
                tags.update(dish_data.get("tags", []))
        
        if dish_name in product_lower:
            tags.update(dish_data.get("tags", []))
    
    for category, dishes in data.get("category_templates", {}).items():
        for dish in dishes:
            if dish.replace("_", " ") in product_lower:
                tags.add(category)
    
    pairing_items = get_pairing_items()
    for item_a, item_b, pairing_type in pairing_items:
        if item_a in product_lower or item_b in product_lower:
            tags.add(f"pairs_with_{pairing_type}")
    
    return tags


def create_enhanced_copurchase_from_pairings() -> list[dict]:
    """Create synthetic copurchase pairs from food pairing data.
    
    These can be added to the bundling system to boost scores for
    culturally relevant product combinations.
    """
    data = load_cuisine_data()
    pairings = data.get("food_pairings", {})
    
    enhanced_pairs = []
    
    pairing_type_boosts = {
        "classic": 15,
        "traditional": 20,
        "saudi_style": 18,
        "hospitality": 12,
        "classic_snack": 10,
        "dairy_complement": 12,
        "fresh_balance": 8,
        "ramadan": 25,
        "iftar": 25,
        "iftar_starter": 30,
        "iftar_main": 25
    }
    
    for pairing_name, pairing_data in pairings.items():
        boost = pairing_type_boosts.get(pairing_name, 10)
        
        for pair in pairing_data.get("pairs", []):
            item_a = pair.get("item_a", "")
            item_b = pair.get("item_b", "")
            pair_type = pair.get("type", pairing_name)
            
            enhanced_pairs.append({
                "product_keyword_a": item_a,
                "product_keyword_b": item_b,
                "pairing_type": pair_type,
                "boost_score": boost,
                "source": "saudi_cuisine_data"
            })
    
    return enhanced_pairs


if __name__ == "__main__":
    print("Saudi Cuisine Data Summary")
    print("=" * 40)
    
    data = load_cuisine_data()
    
    print(f"\nSaudi Dishes: {len(data.get('saudi_dishes', {}))}")
    for dish, info in data.get("saudi_dishes", {}).items():
        print(f"  - {dish}: {info.get('tags', [])[:3]}")
    
    print(f"\nFood Pairing Categories: {len(data.get('food_pairings', {}))}")
    for pairing_type, pairing_data in data.get("food_pairings", {}).items():
        print(f"  - {pairing_type}: {len(pairing_data.get('pairs', []))} pairs")
    
    print(f"\nCommon Ingredients: {len(data.get('common_ingredients', []))}")
    print(f"Category Templates: {len(data.get('category_templates', {}))}")
    
    print("\n" + "=" * 40)
    print("Sample Enhanced Copurchase Pairs:")
    print("-" * 40)
    enhanced = create_enhanced_copurchase_from_pairings()[:10]
    for pair in enhanced:
        print(f"  {pair['product_keyword_a']} <-> {pair['product_keyword_b']} ({pair['boost_score']})")
