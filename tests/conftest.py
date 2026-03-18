import csv
import json
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def compatible_project_root(tmp_path: Path) -> Path:
    reference_dir = tmp_path / "data" / "reference"
    processed_dir = tmp_path / "data" / "processed"
    reference_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    product_rows = [
        {
            "product_id": "100",
            "product_name": "Arabic Coffee Blend",
            "category": "beverages",
            "subcategory": "coffee",
            "importance_level": "high",
            "category_tags": "beverages|coffee|qeu_category",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "20",
        },
        {
            "product_id": "200",
            "product_name": "Coffee Creamer",
            "category": "beverages",
            "subcategory": "coffee",
            "importance_level": "high",
            "category_tags": "beverages|coffee|cream|dairy|ingredient",
            "category_count": "5",
            "product_family": "",
            "frequency_score": "18",
        },
        {
            "product_id": "210",
            "product_name": "Whole Milk",
            "category": "dairy",
            "subcategory": "dairy_products",
            "importance_level": "high",
            "category_tags": "dairy|milk|drinks|ingredient",
            "category_count": "4",
            "product_family": "milk_dairy",
            "frequency_score": "19",
        },
        {
            "product_id": "220",
            "product_name": "Butter Biscuits",
            "category": "snacks",
            "subcategory": "cookies",
            "importance_level": "medium",
            "category_tags": "snacks|tea_biscuits|cookies|desserts",
            "category_count": "4",
            "product_family": "",
            "frequency_score": "17",
        },
        {
            "product_id": "230",
            "product_name": "Instant Coffee Classic",
            "category": "beverages",
            "subcategory": "coffee",
            "importance_level": "high",
            "category_tags": "beverages|coffee|qeu_category",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "16",
        },
        {
            "product_id": "240",
            "product_name": "Chicken Breast",
            "category": "protein",
            "subcategory": "poultry",
            "importance_level": "high",
            "category_tags": "protein|chicken|ingredient",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "22",
        },
        {
            "product_id": "300",
            "product_name": "Sukkari Dates",
            "category": "fruits",
            "subcategory": "dried_fruits",
            "importance_level": "high",
            "category_tags": "dates|stuffed_dates|ramadan|ingredient",
            "category_count": "4",
            "product_family": "",
            "frequency_score": "15",
        },
        {
            "product_id": "310",
            "product_name": "Liquid Tahini",
            "category": "condiments",
            "subcategory": "paste",
            "importance_level": "high",
            "category_tags": "tahini|halva|sauces|ingredient",
            "category_count": "4",
            "product_family": "",
            "frequency_score": "13",
        },
        {
            "product_id": "320",
            "product_name": "Cream Cheese Spread",
            "category": "dairy",
            "subcategory": "cheese",
            "importance_level": "high",
            "category_tags": "cheese|desserts|sandwiches|ingredient",
            "category_count": "4",
            "product_family": "cheese",
            "frequency_score": "14",
        },
        {
            "product_id": "330",
            "product_name": "Medjool Dates Box",
            "category": "fruits",
            "subcategory": "dried_fruits",
            "importance_level": "high",
            "category_tags": "dates|dried_fruits|ingredient",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "14",
        },
        {
            "product_id": "340",
            "product_name": "Bottled Water",
            "category": "beverages",
            "subcategory": "water",
            "importance_level": "medium",
            "category_tags": "beverages|water|drinks",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "18",
        },
    ]
    with (processed_dir / "product_categories.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(product_rows[0].keys()))
        writer.writeheader()
        writer.writerows(product_rows)

    recipe_rows = [
        {"product_id": "100", "matched_ingredient": "coffee_beans", "recipe_score": "30"},
        {"product_id": "200", "matched_ingredient": "", "recipe_score": "20"},
        {"product_id": "210", "matched_ingredient": "milk", "recipe_score": "24"},
        {"product_id": "220", "matched_ingredient": "biscuits", "recipe_score": "21"},
        {"product_id": "230", "matched_ingredient": "", "recipe_score": "18"},
        {"product_id": "240", "matched_ingredient": "chicken", "recipe_score": "26"},
        {"product_id": "300", "matched_ingredient": "dates", "recipe_score": "28"},
        {"product_id": "310", "matched_ingredient": "tahini", "recipe_score": "25"},
        {"product_id": "320", "matched_ingredient": "cheese", "recipe_score": "24"},
        {"product_id": "330", "matched_ingredient": "dates", "recipe_score": "27"},
        {"product_id": "340", "matched_ingredient": "water", "recipe_score": "8"},
    ]
    with (processed_dir / "product_recipe_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_id", "matched_ingredient", "recipe_score"])
        writer.writeheader()
        writer.writerows(recipe_rows)

    copurchase_rows = [
        {"product_a": "100", "product_b": "200", "pair_count": "9", "score": "1.6"},
        {"product_a": "100", "product_b": "210", "pair_count": "8", "score": "1.4"},
        {"product_a": "100", "product_b": "220", "pair_count": "7", "score": "1.2"},
        {"product_a": "100", "product_b": "230", "pair_count": "10", "score": "1.4"},
        {"product_a": "100", "product_b": "240", "pair_count": "18", "score": "0.7"},
        {"product_a": "300", "product_b": "310", "pair_count": "5", "score": "1.3"},
        {"product_a": "300", "product_b": "320", "pair_count": "4", "score": "1.1"},
        {"product_a": "300", "product_b": "330", "pair_count": "6", "score": "1.5"},
        {"product_a": "300", "product_b": "340", "pair_count": "5", "score": "1.0"},
    ]
    with (processed_dir / "copurchase_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_a", "product_b", "pair_count", "score"])
        writer.writeheader()
        writer.writerows(copurchase_rows)

    recipe_data = {
        "ingredients": {
            "coffee_beans": {"recipes": ["coffee", "saudi_coffee"]},
            "milk": {"recipes": ["coffee", "desserts"]},
            "biscuits": {"recipes": ["coffee", "tea_biscuits"]},
            "chicken": {"recipes": ["kabsa"]},
            "dates": {"recipes": ["stuffed_dates", "ramadan_snack"]},
            "tahini": {"recipes": ["stuffed_dates", "halva"]},
            "cheese": {"recipes": ["stuffed_dates", "sandwiches"]},
            "water": {"recipes": ["hydration"]},
        }
    }
    (reference_dir / "recipe_data.json").write_text(json.dumps(recipe_data), encoding="utf-8")

    category_importance_rows = [
        {"category": "milk", "final_score": "44"},
        {"category": "cheese", "final_score": "45"},
        {"category": "dates", "final_score": "49"},
        {"category": "tahini", "final_score": "40"},
        {"category": "coffee", "final_score": "25"},
    ]
    with (reference_dir / "category_importance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "final_score"])
        writer.writeheader()
        writer.writerows(category_importance_rows)

    penalty_rules = {
        "rules": [
            {
                "name": "tea_with_biscuit_generic",
                "anchor_terms": ["tea", "coffee"],
                "complement_terms": ["biscuit", "cookie"],
                "multiplier": 0.8,
                "reason": "Generic hot drink and biscuit pairings should not dominate.",
            },
            {
                "name": "dates_with_generic_cheese",
                "anchor_terms": ["date"],
                "complement_terms": ["cheese"],
                "multiplier": 0.82,
                "reason": "Dates pair better with specific serving complements than generic cheese.",
            },
        ]
    }
    (reference_dir / "pair_penalty_rules.json").write_text(json.dumps(penalty_rules), encoding="utf-8")

    return tmp_path


@pytest.fixture
def fbt_project_root(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    product_rows = [
        {
            "product_id": "100",
            "product_name": "Arabic Coffee Blend",
            "category": "beverages",
            "subcategory": "coffee",
            "importance_level": "high",
            "category_tags": "beverages|coffee|qeu_category",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "20",
        },
        {
            "product_id": "200",
            "product_name": "Coffee Creamer",
            "category": "beverages",
            "subcategory": "coffee",
            "importance_level": "high",
            "category_tags": "beverages|coffee|cream|dairy|ingredient",
            "category_count": "5",
            "product_family": "",
            "frequency_score": "18",
        },
        {
            "product_id": "210",
            "product_name": "Whole Milk",
            "category": "dairy",
            "subcategory": "dairy_products",
            "importance_level": "high",
            "category_tags": "dairy|milk|drinks|ingredient",
            "category_count": "4",
            "product_family": "milk_dairy",
            "frequency_score": "19",
        },
        {
            "product_id": "220",
            "product_name": "Butter Biscuits",
            "category": "snacks",
            "subcategory": "cookies",
            "importance_level": "medium",
            "category_tags": "snacks|tea_biscuits|cookies|desserts",
            "category_count": "4",
            "product_family": "",
            "frequency_score": "17",
        },
        {
            "product_id": "230",
            "product_name": "Instant Coffee Classic",
            "category": "beverages",
            "subcategory": "coffee",
            "importance_level": "high",
            "category_tags": "beverages|coffee|qeu_category",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "16",
        },
        {
            "product_id": "240",
            "product_name": "Chicken Breast",
            "category": "protein",
            "subcategory": "poultry",
            "importance_level": "high",
            "category_tags": "protein|chicken|ingredient",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "22",
        },
        {
            "product_id": "300",
            "product_name": "Sukkari Dates",
            "category": "fruits",
            "subcategory": "dried_fruits",
            "importance_level": "high",
            "category_tags": "dates|stuffed_dates|ramadan|ingredient",
            "category_count": "4",
            "product_family": "",
            "frequency_score": "15",
        },
        {
            "product_id": "310",
            "product_name": "Liquid Tahini",
            "category": "condiments",
            "subcategory": "paste",
            "importance_level": "high",
            "category_tags": "tahini|halva|ingredient",
            "category_count": "4",
            "product_family": "",
            "frequency_score": "13",
        },
        {
            "product_id": "320",
            "product_name": "Cream Cheese Spread",
            "category": "dairy",
            "subcategory": "cheese",
            "importance_level": "high",
            "category_tags": "cheese|desserts|sandwiches|ingredient",
            "category_count": "4",
            "product_family": "cheese",
            "frequency_score": "14",
        },
        {
            "product_id": "330",
            "product_name": "Medjool Dates Box",
            "category": "fruits",
            "subcategory": "dried_fruits",
            "importance_level": "high",
            "category_tags": "dates|dried_fruits|ingredient",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "14",
        },
        {
            "product_id": "340",
            "product_name": "Bottled Water",
            "category": "beverages",
            "subcategory": "water",
            "importance_level": "medium",
            "category_tags": "beverages|water|drinks",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "18",
        },
    ]
    with (processed_dir / "product_categories.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(product_rows[0].keys()))
        writer.writeheader()
        writer.writerows(product_rows)

    recipe_rows = [
        {"product_id": "100", "matched_ingredient": "coffee_beans", "recipe_score": "30"},
        {"product_id": "200", "matched_ingredient": "", "recipe_score": "20"},
        {"product_id": "210", "matched_ingredient": "milk", "recipe_score": "24"},
        {"product_id": "220", "matched_ingredient": "biscuits", "recipe_score": "21"},
        {"product_id": "230", "matched_ingredient": "", "recipe_score": "18"},
        {"product_id": "240", "matched_ingredient": "chicken", "recipe_score": "26"},
        {"product_id": "300", "matched_ingredient": "dates", "recipe_score": "28"},
        {"product_id": "310", "matched_ingredient": "tahini", "recipe_score": "25"},
        {"product_id": "320", "matched_ingredient": "cheese", "recipe_score": "24"},
        {"product_id": "330", "matched_ingredient": "dates", "recipe_score": "27"},
        {"product_id": "340", "matched_ingredient": "water", "recipe_score": "8"},
    ]
    with (processed_dir / "product_recipe_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_id", "matched_ingredient", "recipe_score"])
        writer.writeheader()
        writer.writerows(recipe_rows)

    copurchase_rows = [
        {"product_a": "100", "product_b": "200", "pair_count": "4", "score": "1.6"},
        {"product_a": "100", "product_b": "210", "pair_count": "3", "score": "1.4"},
        {"product_a": "100", "product_b": "220", "pair_count": "2", "score": "1.0"},
        {"product_a": "100", "product_b": "230", "pair_count": "4", "score": "1.3"},
        {"product_a": "100", "product_b": "240", "pair_count": "1", "score": "0.2"},
        {"product_a": "300", "product_b": "310", "pair_count": "3", "score": "1.4"},
        {"product_a": "300", "product_b": "320", "pair_count": "2", "score": "1.1"},
        {"product_a": "300", "product_b": "330", "pair_count": "4", "score": "1.8"},
        {"product_a": "300", "product_b": "340", "pair_count": "2", "score": "0.6"},
    ]
    with (processed_dir / "copurchase_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_a", "product_b", "pair_count", "score"])
        writer.writeheader()
        writer.writerows(copurchase_rows)

    order_rows = [
        {"order_id": "1", "product_id": "100"},
        {"order_id": "1", "product_id": "200"},
        {"order_id": "1", "product_id": "210"},
        {"order_id": "1", "product_id": "230"},
        {"order_id": "2", "product_id": "100"},
        {"order_id": "2", "product_id": "200"},
        {"order_id": "2", "product_id": "210"},
        {"order_id": "2", "product_id": "230"},
        {"order_id": "3", "product_id": "100"},
        {"order_id": "3", "product_id": "200"},
        {"order_id": "3", "product_id": "210"},
        {"order_id": "3", "product_id": "230"},
        {"order_id": "4", "product_id": "100"},
        {"order_id": "4", "product_id": "200"},
        {"order_id": "4", "product_id": "220"},
        {"order_id": "4", "product_id": "230"},
        {"order_id": "5", "product_id": "100"},
        {"order_id": "5", "product_id": "220"},
        {"order_id": "5", "product_id": "240"},
        {"order_id": "5", "product_id": "230"},
        {"order_id": "6", "product_id": "230"},
        {"order_id": "6", "product_id": "340"},
        {"order_id": "7", "product_id": "230"},
        {"order_id": "8", "product_id": "230"},
        {"order_id": "9", "product_id": "230"},
        {"order_id": "10", "product_id": "300"},
        {"order_id": "10", "product_id": "310"},
        {"order_id": "10", "product_id": "320"},
        {"order_id": "10", "product_id": "330"},
        {"order_id": "10", "product_id": "340"},
        {"order_id": "11", "product_id": "300"},
        {"order_id": "11", "product_id": "310"},
        {"order_id": "11", "product_id": "330"},
        {"order_id": "12", "product_id": "300"},
        {"order_id": "12", "product_id": "310"},
        {"order_id": "12", "product_id": "330"},
        {"order_id": "13", "product_id": "300"},
        {"order_id": "13", "product_id": "320"},
        {"order_id": "13", "product_id": "330"},
        {"order_id": "13", "product_id": "340"},
        {"order_id": "14", "product_id": "330"},
        {"order_id": "14", "product_id": "340"},
        {"order_id": "15", "product_id": "340"},
    ]
    with (raw_dir / "order_items.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order_id", "product_id"])
        writer.writeheader()
        writer.writerows(order_rows)

    return tmp_path


@pytest.fixture
def bundle_project_root(tmp_path: Path) -> Path:
    raw_dir = tmp_path / "data" / "raw"
    processed_dir = tmp_path / "data" / "processed"
    reference_dir = tmp_path / "data" / "reference"
    output_dir = tmp_path / "output"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    product_rows = [
        {
            "product_id": "100",
            "product_name": "Arabic Coffee Blend",
            "category": "beverages",
            "subcategory": "coffee",
            "importance_level": "high",
            "category_tags": "beverages|coffee|qeu_category",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "20",
        },
        {
            "product_id": "200",
            "product_name": "Coffee Creamer",
            "category": "beverages",
            "subcategory": "coffee",
            "importance_level": "high",
            "category_tags": "beverages|coffee|cream|dairy|ingredient",
            "category_count": "5",
            "product_family": "",
            "frequency_score": "18",
        },
        {
            "product_id": "210",
            "product_name": "Whole Milk",
            "category": "dairy",
            "subcategory": "dairy_products",
            "importance_level": "high",
            "category_tags": "dairy|milk|drinks|ingredient",
            "category_count": "4",
            "product_family": "milk_dairy",
            "frequency_score": "19",
        },
        {
            "product_id": "220",
            "product_name": "Butter Biscuits",
            "category": "snacks",
            "subcategory": "cookies",
            "importance_level": "medium",
            "category_tags": "snacks|tea_biscuits|cookies|desserts",
            "category_count": "4",
            "product_family": "",
            "frequency_score": "17",
        },
        {
            "product_id": "230",
            "product_name": "Instant Coffee Classic",
            "category": "beverages",
            "subcategory": "coffee",
            "importance_level": "high",
            "category_tags": "beverages|coffee|qeu_category",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "16",
        },
        {
            "product_id": "240",
            "product_name": "Chicken Breast",
            "category": "protein",
            "subcategory": "poultry",
            "importance_level": "high",
            "category_tags": "protein|chicken|ingredient",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "22",
        },
        {
            "product_id": "300",
            "product_name": "Sukkari Dates",
            "category": "fruits",
            "subcategory": "dried_fruits",
            "importance_level": "high",
            "category_tags": "dates|stuffed_dates|ramadan|ingredient",
            "category_count": "4",
            "product_family": "",
            "frequency_score": "15",
        },
        {
            "product_id": "310",
            "product_name": "Liquid Tahini",
            "category": "condiments",
            "subcategory": "paste",
            "importance_level": "high",
            "category_tags": "tahini|halva|ingredient",
            "category_count": "4",
            "product_family": "",
            "frequency_score": "13",
        },
        {
            "product_id": "320",
            "product_name": "Cream Cheese Spread",
            "category": "dairy",
            "subcategory": "cheese",
            "importance_level": "high",
            "category_tags": "cheese|desserts|sandwiches|ingredient",
            "category_count": "4",
            "product_family": "cheese",
            "frequency_score": "14",
        },
        {
            "product_id": "330",
            "product_name": "Medjool Dates Box",
            "category": "fruits",
            "subcategory": "dried_fruits",
            "importance_level": "high",
            "category_tags": "dates|dried_fruits|ingredient",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "14",
        },
        {
            "product_id": "340",
            "product_name": "Bottled Water",
            "category": "beverages",
            "subcategory": "water",
            "importance_level": "medium",
            "category_tags": "beverages|water|drinks",
            "category_count": "3",
            "product_family": "",
            "frequency_score": "18",
        },
    ]
    with (processed_dir / "product_categories.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(product_rows[0].keys()))
        writer.writeheader()
        writer.writerows(product_rows)

    recipe_rows = [
        {"product_id": "100", "matched_ingredient": "coffee_beans", "recipe_score": "30"},
        {"product_id": "200", "matched_ingredient": "", "recipe_score": "20"},
        {"product_id": "210", "matched_ingredient": "milk", "recipe_score": "24"},
        {"product_id": "220", "matched_ingredient": "biscuits", "recipe_score": "21"},
        {"product_id": "230", "matched_ingredient": "", "recipe_score": "18"},
        {"product_id": "240", "matched_ingredient": "chicken", "recipe_score": "26"},
        {"product_id": "300", "matched_ingredient": "dates", "recipe_score": "28"},
        {"product_id": "310", "matched_ingredient": "tahini", "recipe_score": "25"},
        {"product_id": "320", "matched_ingredient": "cheese", "recipe_score": "24"},
        {"product_id": "330", "matched_ingredient": "dates", "recipe_score": "27"},
        {"product_id": "340", "matched_ingredient": "water", "recipe_score": "8"},
    ]
    with (processed_dir / "product_recipe_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_id", "matched_ingredient", "recipe_score"])
        writer.writeheader()
        writer.writerows(recipe_rows)

    copurchase_rows = [
        {"product_a": "100", "product_b": "200", "pair_count": "4", "score": "1.6"},
        {"product_a": "100", "product_b": "210", "pair_count": "3", "score": "1.4"},
        {"product_a": "100", "product_b": "220", "pair_count": "2", "score": "1.0"},
        {"product_a": "100", "product_b": "230", "pair_count": "4", "score": "1.3"},
        {"product_a": "100", "product_b": "240", "pair_count": "1", "score": "0.2"},
        {"product_a": "300", "product_b": "310", "pair_count": "3", "score": "1.4"},
        {"product_a": "300", "product_b": "320", "pair_count": "2", "score": "1.1"},
        {"product_a": "300", "product_b": "330", "pair_count": "4", "score": "1.8"},
        {"product_a": "300", "product_b": "340", "pair_count": "2", "score": "0.6"},
    ]
    with (processed_dir / "copurchase_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_a", "product_b", "pair_count", "score"])
        writer.writeheader()
        writer.writerows(copurchase_rows)

    top_bundle_rows = [
        {
            "product_a": "100",
            "product_b": "200",
            "product_a_name": "Arabic Coffee Blend",
            "product_b_name": "Coffee Creamer",
            "product_a_price": "19.5",
            "product_b_price": "8.5",
            "category_a": "beverages",
            "category_b": "beverages",
            "pair_count": "9",
            "recipe_compat_score": "0.92",
            "known_prior_flag": "1",
            "pair_penalty_multiplier": "1.0",
            "utility_penalty_multiplier": "1.0",
            "new_final_score": "88.0",
            "final_score": "88.0",
        },
        {
            "product_a": "100",
            "product_b": "210",
            "product_a_name": "Arabic Coffee Blend",
            "product_b_name": "Whole Milk",
            "product_a_price": "19.5",
            "product_b_price": "7.0",
            "category_a": "beverages",
            "category_b": "dairy",
            "pair_count": "8",
            "recipe_compat_score": "0.86",
            "known_prior_flag": "0",
            "pair_penalty_multiplier": "1.0",
            "utility_penalty_multiplier": "1.0",
            "new_final_score": "82.0",
            "final_score": "82.0",
        },
        {
            "product_a": "100",
            "product_b": "220",
            "product_a_name": "Arabic Coffee Blend",
            "product_b_name": "Butter Biscuits",
            "product_a_price": "19.5",
            "product_b_price": "6.5",
            "category_a": "beverages",
            "category_b": "snacks",
            "pair_count": "7",
            "recipe_compat_score": "0.81",
            "known_prior_flag": "0",
            "pair_penalty_multiplier": "1.0",
            "utility_penalty_multiplier": "1.0",
            "new_final_score": "73.0",
            "final_score": "73.0",
        },
        {
            "product_a": "300",
            "product_b": "310",
            "product_a_name": "Sukkari Dates",
            "product_b_name": "Liquid Tahini",
            "product_a_price": "24.0",
            "product_b_price": "10.0",
            "category_a": "fruits",
            "category_b": "condiments",
            "pair_count": "5",
            "recipe_compat_score": "0.89",
            "known_prior_flag": "1",
            "pair_penalty_multiplier": "1.0",
            "utility_penalty_multiplier": "1.0",
            "new_final_score": "85.0",
            "final_score": "85.0",
        },
    ]
    with (processed_dir / "top_bundles.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(top_bundle_rows[0].keys()))
        writer.writeheader()
        writer.writerows(top_bundle_rows)

    filtered_order_rows = [
        {
            "order_id": "101",
            "product_id": "100",
            "product_name": "Arabic Coffee Blend",
            "quantity": 1,
            "created_at": "2026-02-24T10:00:00+00:00",
        },
        {
            "order_id": "101",
            "product_id": "200",
            "product_name": "Coffee Creamer",
            "quantity": 3,
            "created_at": "2026-02-24T10:00:04+00:00",
        },
        {
            "order_id": "101",
            "product_id": "220",
            "product_name": "Butter Biscuits",
            "quantity": 1,
            "created_at": "2026-02-24T10:00:08+00:00",
        },
        {
            "order_id": "202",
            "product_id": "100",
            "product_name": "Arabic Coffee Blend",
            "quantity": 1,
            "created_at": "2026-02-24T10:05:00+00:00",
        },
        {
            "order_id": "202",
            "product_id": "320",
            "product_name": "Cream Cheese Spread",
            "quantity": 4,
            "created_at": "2026-02-24T10:05:05+00:00",
        },
        {
            "order_id": "202",
            "product_id": "300",
            "product_name": "Sukkari Dates",
            "quantity": 2,
            "created_at": "2026-02-24T10:05:10+00:00",
        },
    ]
    pd.DataFrame(filtered_order_rows).to_pickle(processed_dir / "filtered_orders.pkl")

    order_rows = [
        {"order_id": "1", "product_id": "100"},
        {"order_id": "1", "product_id": "200"},
        {"order_id": "1", "product_id": "210"},
        {"order_id": "1", "product_id": "230"},
        {"order_id": "2", "product_id": "100"},
        {"order_id": "2", "product_id": "200"},
        {"order_id": "2", "product_id": "210"},
        {"order_id": "2", "product_id": "230"},
        {"order_id": "3", "product_id": "100"},
        {"order_id": "3", "product_id": "200"},
        {"order_id": "3", "product_id": "210"},
        {"order_id": "3", "product_id": "230"},
        {"order_id": "4", "product_id": "100"},
        {"order_id": "4", "product_id": "200"},
        {"order_id": "4", "product_id": "220"},
        {"order_id": "4", "product_id": "230"},
        {"order_id": "5", "product_id": "100"},
        {"order_id": "5", "product_id": "220"},
        {"order_id": "5", "product_id": "240"},
        {"order_id": "5", "product_id": "230"},
        {"order_id": "6", "product_id": "230"},
        {"order_id": "6", "product_id": "340"},
        {"order_id": "7", "product_id": "230"},
        {"order_id": "8", "product_id": "230"},
        {"order_id": "9", "product_id": "230"},
        {"order_id": "10", "product_id": "300"},
        {"order_id": "10", "product_id": "310"},
        {"order_id": "10", "product_id": "320"},
        {"order_id": "10", "product_id": "330"},
        {"order_id": "10", "product_id": "340"},
        {"order_id": "11", "product_id": "300"},
        {"order_id": "11", "product_id": "310"},
        {"order_id": "11", "product_id": "330"},
        {"order_id": "12", "product_id": "300"},
        {"order_id": "12", "product_id": "310"},
        {"order_id": "12", "product_id": "330"},
        {"order_id": "13", "product_id": "300"},
        {"order_id": "13", "product_id": "320"},
        {"order_id": "13", "product_id": "330"},
        {"order_id": "13", "product_id": "340"},
        {"order_id": "14", "product_id": "330"},
        {"order_id": "14", "product_id": "340"},
        {"order_id": "15", "product_id": "340"},
    ]
    with (raw_dir / "order_items.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["order_id", "product_id"])
        writer.writeheader()
        writer.writerows(order_rows)

    recipe_data = {
        "ingredients": {
            "coffee_beans": {"recipes": ["coffee", "saudi_coffee"]},
            "milk": {"recipes": ["coffee", "desserts"]},
            "biscuits": {"recipes": ["coffee", "tea_biscuits"]},
            "chicken": {"recipes": ["kabsa"]},
            "dates": {"recipes": ["stuffed_dates", "ramadan_snack"]},
            "tahini": {"recipes": ["stuffed_dates", "halva"]},
            "cheese": {"recipes": ["stuffed_dates", "sandwiches"]},
            "water": {"recipes": ["hydration"]},
        }
    }
    (reference_dir / "recipe_data.json").write_text(json.dumps(recipe_data), encoding="utf-8")

    category_importance_rows = [
        {"category": "milk", "final_score": "44"},
        {"category": "cheese", "final_score": "45"},
        {"category": "dates", "final_score": "49"},
        {"category": "tahini", "final_score": "40"},
        {"category": "coffee", "final_score": "25"},
    ]
    with (reference_dir / "category_importance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "final_score"])
        writer.writeheader()
        writer.writerows(category_importance_rows)

    penalty_rules = {
        "rules": [
            {
                "name": "tea_with_biscuit_generic",
                "anchor_terms": ["tea", "coffee"],
                "complement_terms": ["biscuit", "cookie"],
                "multiplier": 0.8,
                "reason": "Generic hot drink and biscuit pairings should not dominate.",
            },
            {
                "name": "dates_with_generic_cheese",
                "anchor_terms": ["date"],
                "complement_terms": ["cheese"],
                "multiplier": 0.82,
                "reason": "Dates pair better with specific serving complements than generic cheese.",
            },
        ]
    }
    (reference_dir / "pair_penalty_rules.json").write_text(json.dumps(penalty_rules), encoding="utf-8")

    final_recommendations = {
        "recommendations_by_user": {
            "101": [
                {"item_1_id": 100, "item_2_id": 200, "bundle_price": 23.5},
                {"item_1_id": 100, "item_2_id": 230, "bundle_price": 26.0},
                {"item_1_id": 300, "item_2_id": 310, "bundle_price": 28.5},
            ]
        }
    }
    (output_dir / "final_recommendations_by_user.json").write_text(
        json.dumps(final_recommendations),
        encoding="utf-8",
    )

    fallback_bundle_bank = {
        "bundles": [
            {
                "item_1_id": 100,
                "item_2_id": 210,
                "bundle_price": 22.0,
                "quality_score": 69.0,
                "category_key": "beverages|dairy",
                "anchor_category": "beverages",
                "complement_category": "dairy",
            },
            {
                "item_1_id": 300,
                "item_2_id": 340,
                "bundle_price": 18.0,
                "quality_score": 38.0,
                "category_key": "beverages|fruits",
                "anchor_category": "fruits",
                "complement_category": "beverages",
            },
        ]
    }
    (output_dir / "fallback_bundle_bank.json").write_text(json.dumps(fallback_bundle_bank), encoding="utf-8")

    bundle_id_rows = [
        {
            "bundle_id": "B00000001",
            "item_1_id": "100",
            "item_2_id": "200",
            "first_seen_at": "2026-03-17T19:49:12+00:00",
            "first_seen_run_id": "test_run",
            "last_seen_at": "2026-03-17T21:47:31+00:00",
            "last_seen_run_id": "test_run",
            "last_bundle_price": "23.50",
            "seen_count": "3",
        },
        {
            "bundle_id": "B00000002",
            "item_1_id": "100",
            "item_2_id": "210",
            "first_seen_at": "2026-03-17T19:49:12+00:00",
            "first_seen_run_id": "test_run",
            "last_seen_at": "2026-03-17T21:47:31+00:00",
            "last_seen_run_id": "test_run",
            "last_bundle_price": "22.00",
            "seen_count": "2",
        },
        {
            "bundle_id": "B00000003",
            "item_1_id": "300",
            "item_2_id": "310",
            "first_seen_at": "2026-03-17T19:49:12+00:00",
            "first_seen_run_id": "test_run",
            "last_seen_at": "2026-03-17T21:47:31+00:00",
            "last_seen_run_id": "test_run",
            "last_bundle_price": "28.50",
            "seen_count": "4",
        },
    ]
    with (output_dir / "bundle_ids.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(bundle_id_rows[0].keys()))
        writer.writeheader()
        writer.writerows(bundle_id_rows)

    return tmp_path


@pytest.fixture
def bundle_project_root_with_universe(bundle_project_root: Path) -> Path:
    from pipelines.bundle_universe import build_bundle_universe

    build_bundle_universe(
        project_root=bundle_project_root,
        target_size=6,
        per_root_limit=4,
    )
    return bundle_project_root
