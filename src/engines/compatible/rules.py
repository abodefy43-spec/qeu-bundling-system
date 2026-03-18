"""Compatibility-oriented heuristics for the compatible products engine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompatibilityProfile:
    name: str
    trigger_tokens: frozenset[str] = frozenset()
    trigger_tags: frozenset[str] = frozenset()
    trigger_ingredients: frozenset[str] = frozenset()
    trigger_subcategories: frozenset[str] = frozenset()
    trigger_families: frozenset[str] = frozenset()
    positive_tokens: frozenset[str] = frozenset()
    priority_tokens: frozenset[str] = frozenset()
    positive_tags: frozenset[str] = frozenset()
    positive_categories: frozenset[str] = frozenset()
    positive_subcategories: frozenset[str] = frozenset()
    positive_ingredients: frozenset[str] = frozenset()
    positive_families: frozenset[str] = frozenset()
    allow_same_category_tokens: frozenset[str] = frozenset()


COMPATIBILITY_PROFILES = (
    CompatibilityProfile(
        name="hot_drink_service",
        trigger_tokens=frozenset({"coffee", "tea"}),
        trigger_tags=frozenset({"coffee", "tea", "saudi_tea", "milk_tea"}),
        trigger_ingredients=frozenset({"coffee_beans", "tea_biscuits"}),
        trigger_subcategories=frozenset({"coffee", "tea"}),
        positive_tokens=frozenset({"milk", "creamer", "biscuit", "cookie", "date", "cardamom"}),
        priority_tokens=frozenset({"creamer", "biscuit", "cookie"}),
        positive_tags=frozenset({"tea_biscuits", "milk_tea"}),
        positive_categories=frozenset({"dairy", "snacks", "fruits", "beverages"}),
        positive_subcategories=frozenset({"dairy_products", "cookies"}),
        positive_ingredients=frozenset({"milk", "biscuits"}),
        allow_same_category_tokens=frozenset({"milk", "creamer", "biscuit", "cookie", "date"}),
    ),
    CompatibilityProfile(
        name="pasta_completion",
        trigger_tokens=frozenset({"pasta", "spaghetti", "penne", "macaroni", "lasagna"}),
        trigger_tags=frozenset({"pasta", "pasta_sauce", "spaghetti", "macaroni", "penne", "lasagna"}),
        trigger_ingredients=frozenset({"pasta"}),
        trigger_subcategories=frozenset({"pasta"}),
        positive_tokens=frozenset({"sauce", "tomato", "pesto", "cheese", "cream", "parmesan", "mozzarella"}),
        priority_tokens=frozenset({"sauce", "pesto", "tomato", "parmesan", "mozzarella"}),
        positive_tags=frozenset({"pasta_sauce", "sauces", "pizza"}),
        positive_categories=frozenset({"dairy", "condiments", "vegetables", "grains"}),
        positive_subcategories=frozenset({"cheese", "dairy_products", "tomatoes", "paste"}),
        positive_ingredients=frozenset({"cheese", "tomatoes", "milk"}),
        allow_same_category_tokens=frozenset({"sauce", "cheese", "cream", "tomato", "pesto"}),
    ),
    CompatibilityProfile(
        name="burger_build",
        trigger_tokens=frozenset({"burger", "bun"}),
        trigger_tags=frozenset({"sandwiches", "bread"}),
        positive_tokens=frozenset({"burger", "chicken", "beef", "cheese", "slice", "ketchup", "mayo", "mayonnaise", "patty"}),
        priority_tokens=frozenset({"burger", "patty", "slice", "cheese"}),
        positive_tags=frozenset({"sandwiches", "appetizers"}),
        positive_categories=frozenset({"protein", "dairy", "condiments", "other"}),
        positive_subcategories=frozenset({"poultry", "red_meat", "cheese", "condiments"}),
        positive_ingredients=frozenset({"chicken", "beef", "cheese"}),
        allow_same_category_tokens=frozenset({"burger", "patty", "cheese", "ketchup", "mayo", "mayonnaise"}),
    ),
    CompatibilityProfile(
        name="dates_pairing",
        trigger_tokens=frozenset({"date"}),
        trigger_tags=frozenset({"dates", "stuffed_dates", "iftar_traditional", "date_smoothie"}),
        trigger_ingredients=frozenset({"dates"}),
        trigger_subcategories=frozenset({"dried_fruits"}),
        positive_tokens=frozenset({"tahini", "cream", "cheese", "labneh", "coffee", "honey", "nut", "cardamom"}),
        priority_tokens=frozenset({"tahini", "labneh", "cheese"}),
        positive_tags=frozenset({"stuffed_dates", "halva"}),
        positive_categories=frozenset({"condiments", "dairy", "beverages", "baking"}),
        positive_subcategories=frozenset({"paste", "cheese", "sweeteners"}),
        positive_ingredients=frozenset({"tahini", "cheese", "coffee_beans", "milk"}),
        allow_same_category_tokens=frozenset(),
    ),
    CompatibilityProfile(
        name="rice_meal_completion",
        trigger_tokens=frozenset({"rice"}),
        trigger_tags=frozenset({"rice_dishes", "kabsa", "machboos", "biryani"}),
        trigger_ingredients=frozenset({"rice"}),
        trigger_subcategories=frozenset({"rice"}),
        trigger_families=frozenset({"rice_centric"}),
        positive_tokens=frozenset({"chicken", "lamb", "meat", "tomato", "paste", "spice", "cardamom", "cumin", "cinnamon", "ghee", "oil", "stock", "broth"}),
        priority_tokens=frozenset({"spice", "cardamom", "cumin", "cinnamon", "stock", "broth"}),
        positive_tags=frozenset({"kabsa", "rice_dishes", "sauces", "marinades"}),
        positive_categories=frozenset({"protein", "spices", "oils", "vegetables", "condiments"}),
        positive_subcategories=frozenset({"poultry", "red_meat", "tomatoes", "seasonings", "cooking_oils"}),
        positive_ingredients=frozenset({"chicken", "lamb", "tomatoes", "ghee"}),
        allow_same_category_tokens=frozenset({"spice"}),
    ),
)
