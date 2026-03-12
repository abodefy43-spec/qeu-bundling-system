"""Deterministic semantic engine for person bundle selection.

This module intentionally has no runtime side effects and no randomness.
It classifies products into roles, classifies pair relations/strength, and
evaluates lane compatibility with explicit hard-invalid rules.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


ROLE_PROTEIN = "protein"
ROLE_CARB_BASE = "carb_base"
ROLE_BREAD = "bread"
ROLE_SAUCE = "sauce"
ROLE_FAT = "fat"
ROLE_DAIRY = "dairy"
ROLE_CHEESE_SNACK = "cheese_snack"
ROLE_CHEESE_COOKING = "cheese_cooking"
ROLE_CHEESE_SPREAD = "cheese_spread"
ROLE_MILK_FRESH = "milk_fresh"
ROLE_MILK_EVAP = "milk_evaporated"
ROLE_MILK_COND = "milk_condensed"
ROLE_MILK_POWDER = "milk_powder"
ROLE_MILK_BABY = "milk_baby"
ROLE_CREAM_TABLE = "cream_table"
ROLE_CREAM_COOKING = "cream_cooking"
ROLE_DESSERT_BASE = "dessert_base"
ROLE_DESSERT_MIX = "dessert_mix"
ROLE_DESSERT_READY = "dessert_ready"
ROLE_SNACK_SALTY = "snack_salty"
ROLE_SNACK_SWEET = "snack_sweet"
ROLE_BISCUIT = "biscuit"
ROLE_CHOCOLATE = "chocolate"
ROLE_COFFEE = "coffee"
ROLE_TEA = "tea"
ROLE_PRODUCE_COOKING = "produce_cooking_base"
ROLE_PRODUCE_AROMATIC = "produce_aromatic"
ROLE_PRODUCE_SALAD = "produce_salad_fresh"
ROLE_PRODUCE_HERB = "produce_herb_fresh"
ROLE_PRODUCE_FRUIT = "produce_sweet_fruit"
ROLE_PRODUCE_STARCHY = "produce_starchy"
ROLE_SEASONING = "seasoning"
ROLE_STAPLE_CORE = "staple_core"
ROLE_NONFOOD = "nonfood"


REL_EAT = "eat_together"
REL_DRINK = "drink_together"
REL_COOK = "cook_together"
REL_DESSERT = "dessert_together"
REL_STAPLE = "staple_together"
REL_HOUSEHOLD = "household_together"
REL_INVALID = "invalid"


STRENGTH_STRONG = "strong"
STRENGTH_STAPLE = "staple"
STRENGTH_WEAK = "weak_valid"
STRENGTH_TRASH = "trash"


LANE_MEAL = "meal"
LANE_SNACK = "snack"
LANE_OCCASION = "occasion"
LANE_STAPLES = "staples"
LANE_NONFOOD = "nonfood"


FOOD_LANES = frozenset({LANE_MEAL, LANE_SNACK, LANE_OCCASION, LANE_STAPLES})

ROLE_PRIORITY = (
    ROLE_NONFOOD,
    ROLE_MILK_BABY,
    ROLE_TEA,
    ROLE_COFFEE,
    ROLE_MILK_EVAP,
    ROLE_MILK_COND,
    ROLE_MILK_POWDER,
    ROLE_MILK_FRESH,
    ROLE_BISCUIT,
    ROLE_CHOCOLATE,
    ROLE_CHEESE_SNACK,
    ROLE_CHEESE_SPREAD,
    ROLE_CHEESE_COOKING,
    ROLE_SNACK_SALTY,
    ROLE_SNACK_SWEET,
    ROLE_DESSERT_READY,
    ROLE_DESSERT_MIX,
    ROLE_DESSERT_BASE,
    ROLE_PROTEIN,
    ROLE_CARB_BASE,
    ROLE_BREAD,
    ROLE_SAUCE,
    ROLE_FAT,
    ROLE_PRODUCE_AROMATIC,
    ROLE_PRODUCE_COOKING,
    ROLE_PRODUCE_STARCHY,
    ROLE_PRODUCE_HERB,
    ROLE_PRODUCE_SALAD,
    ROLE_PRODUCE_FRUIT,
    ROLE_SEASONING,
    ROLE_STAPLE_CORE,
    ROLE_DAIRY,
)


_WORD_RE = re.compile(r"[a-z0-9]+")


def normalize_product_text(name: str, category: str = "", family: str = "") -> str:
    text = f"{name or ''} {category or ''} {family or ''}".lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return " ".join(text.split())


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text))


def _has_any(text: str, hints: frozenset[str]) -> bool:
    return any(h in text for h in hints)


NONFOOD_HINTS = frozenset(
    {
        "tissue",
        "paper towel",
        "toilet paper",
        "detergent",
        "dishwashing",
        "disinfectant",
        "bleach",
        "cleaner",
        "shampoo",
        "conditioner",
        "soap",
        "body wash",
        "trash bag",
        "garbage bag",
        "foil",
        "wrap",
        "napkin",
        "moisturizing",
        "moisturiser",
        "moisturizer",
        "skin care",
        "skincare",
        "face cream",
        "hand cream",
        "body lotion",
        "underarm cream",
        "cosmetic",
        "cosmetics",
        "serum",
    }
)

MILK_BABY_HINTS = frozenset({"baby milk", "infant formula", "stage 1", "stage 2", "follow on formula"})
TEA_HINTS = frozenset({"tea", "chai"})
COFFEE_HINTS = frozenset({"coffee", "nescafe", "cappuccino", "espresso"})
EVAP_HINTS = frozenset({"evaporated milk"})
COND_HINTS = frozenset({"condensed milk"})
POWDER_MILK_HINTS = frozenset({"milk powder", "powder milk"})
FRESH_MILK_HINTS = frozenset({"fresh milk", "full cream milk", "low fat milk", "milk"})
CHEESE_COOKING_HINTS = frozenset({"mozzarella", "grated cheese", "shredded cheese"})
CHEESE_SPREAD_HINTS = frozenset({"cream cheese", "cheese spread", "labneh", "triangle cheese", "puck", "kiri"})
CHEESE_SNACK_HINTS = frozenset({"kraft cheese", "portion cheese", "cheese portions", "processed cheese"})
CREAM_COOKING_HINTS = frozenset({"cooking cream", "cream substitute"})
CREAM_TABLE_HINTS = frozenset({"fresh cream", "table cream", "qishta", "cream"})
DESSERT_READY_HINTS = frozenset({"dessert", "pudding", "custard", "caramel cream dessert", "jelly"})
DESSERT_MIX_HINTS = frozenset({"cake mix", "brownie mix", "dessert mix", "custard powder"})
DESSERT_BASE_HINTS = frozenset({"nutella", "chocolate spread", "cocoa", "sweetened condensed"})
SNACK_SALTY_HINTS = frozenset({"chips", "crisps", "nachos", "spring roll chips", "cracker"})
SNACK_SWEET_HINTS = frozenset({"wafer", "cookies", "biscuit", "cake bar", "candy", "gummy"})
BISCUIT_HINTS = frozenset({"biscuit", "biscuits", "cookie", "cookies", "cracker"})
CHOCOLATE_HINTS = frozenset({"chocolate", "nutella", "cocoa"})
SAUCE_HINTS = frozenset({"tomato paste", "ketchup", "mayo", "mayonnaise", "sauce", "hot sauce"})
FAT_HINTS = frozenset({"ghee", "butter", "margarine", "oil", "shortening", "fat"})
BREAD_HINTS = frozenset({"bread", "toast", "tortilla", "wrap", "bun", "pita", "naan"})
CARB_HINTS = frozenset({"rice", "pasta", "spaghetti", "noodles", "flour", "semolina", "oats"})
PROTEIN_HINTS = frozenset({"chicken", "beef", "lamb", "tuna", "fish", "eggs", "nuggets", "burger", "meat"})
SEASONING_HINTS = frozenset({"pepper", "cumin", "spice", "masala", "stock cube", "turmeric", "cardamom"})
AROMATIC_HINTS = frozenset({"onion", "garlic", "green onion"})
COOKING_PRODUCE_HINTS = frozenset({"tomato", "bell pepper", "capsicum", "carrot", "carrots"})
SALAD_PRODUCE_HINTS = frozenset({"cucumber", "lettuce", "radish"})
HERB_PRODUCE_HINTS = frozenset({"mint", "parsley", "coriander", "cilantro", "dill", "basil"})
SWEET_FRUIT_HINTS = frozenset({"banana", "strawberry", "orange", "apple", "grape", "pineapple", "mango"})
STARCHY_PRODUCE_HINTS = frozenset({"potato"})
STAPLE_HINTS = frozenset({"rice", "flour", "sugar", "salt", "oil", "ghee", "eggs", "tomato paste", "pasta", "oats"})
DATES_HINTS = frozenset({"dates", "khallas", "ajwa"})


def infer_product_roles(name: str, category: str, family: str) -> frozenset[str]:
    text = normalize_product_text(name, category, family)
    roles: set[str] = set()
    if _has_any(text, NONFOOD_HINTS):
        roles.add(ROLE_NONFOOD)
        return frozenset(roles)

    if _has_any(text, MILK_BABY_HINTS):
        roles.update({ROLE_MILK_BABY, ROLE_DAIRY})
    if _has_any(text, TEA_HINTS):
        roles.add(ROLE_TEA)
    if _has_any(text, COFFEE_HINTS):
        roles.add(ROLE_COFFEE)
    if _has_any(text, EVAP_HINTS):
        roles.update({ROLE_MILK_EVAP, ROLE_DAIRY})
    if _has_any(text, COND_HINTS):
        roles.update({ROLE_MILK_COND, ROLE_DAIRY, ROLE_DESSERT_BASE})
    if _has_any(text, POWDER_MILK_HINTS):
        roles.update({ROLE_MILK_POWDER, ROLE_DAIRY, ROLE_STAPLE_CORE})
    if _has_any(text, FRESH_MILK_HINTS) and not roles & {ROLE_MILK_BABY, ROLE_MILK_EVAP, ROLE_MILK_COND, ROLE_MILK_POWDER}:
        roles.update({ROLE_MILK_FRESH, ROLE_DAIRY})

    if _has_any(text, CHEESE_COOKING_HINTS):
        roles.update({ROLE_CHEESE_COOKING, ROLE_DAIRY})
    if _has_any(text, CHEESE_SPREAD_HINTS):
        roles.update({ROLE_CHEESE_SPREAD, ROLE_DAIRY})
    if _has_any(text, CHEESE_SNACK_HINTS):
        roles.update({ROLE_CHEESE_SNACK, ROLE_DAIRY})
    if "cheese" in text and not roles & {ROLE_CHEESE_COOKING, ROLE_CHEESE_SPREAD, ROLE_CHEESE_SNACK}:
        roles.update({ROLE_CHEESE_SPREAD, ROLE_DAIRY})

    if _has_any(text, CREAM_COOKING_HINTS):
        roles.update({ROLE_CREAM_COOKING, ROLE_DAIRY})
    if _has_any(text, CREAM_TABLE_HINTS) and ROLE_CREAM_COOKING not in roles:
        roles.update({ROLE_CREAM_TABLE, ROLE_DAIRY})
    if _has_any(text, DESSERT_READY_HINTS):
        roles.add(ROLE_DESSERT_READY)
    if _has_any(text, DESSERT_MIX_HINTS):
        roles.add(ROLE_DESSERT_MIX)
    if _has_any(text, DESSERT_BASE_HINTS):
        roles.add(ROLE_DESSERT_BASE)

    if _has_any(text, SNACK_SALTY_HINTS):
        roles.add(ROLE_SNACK_SALTY)
    if _has_any(text, SNACK_SWEET_HINTS):
        roles.add(ROLE_SNACK_SWEET)
    if _has_any(text, BISCUIT_HINTS):
        roles.add(ROLE_BISCUIT)
    if _has_any(text, CHOCOLATE_HINTS):
        roles.add(ROLE_CHOCOLATE)

    if _has_any(text, SAUCE_HINTS):
        roles.add(ROLE_SAUCE)
    if _has_any(text, FAT_HINTS):
        roles.add(ROLE_FAT)
    if _has_any(text, BREAD_HINTS):
        roles.add(ROLE_BREAD)
    if _has_any(text, CARB_HINTS):
        roles.add(ROLE_CARB_BASE)
    if _has_any(text, PROTEIN_HINTS):
        roles.add(ROLE_PROTEIN)
    if _has_any(text, SEASONING_HINTS):
        roles.add(ROLE_SEASONING)

    if _has_any(text, AROMATIC_HINTS):
        roles.add(ROLE_PRODUCE_AROMATIC)
    if _has_any(text, COOKING_PRODUCE_HINTS):
        roles.add(ROLE_PRODUCE_COOKING)
    if _has_any(text, SALAD_PRODUCE_HINTS):
        roles.add(ROLE_PRODUCE_SALAD)
    if _has_any(text, HERB_PRODUCE_HINTS):
        roles.add(ROLE_PRODUCE_HERB)
    if _has_any(text, SWEET_FRUIT_HINTS):
        roles.add(ROLE_PRODUCE_FRUIT)
    if _has_any(text, DATES_HINTS):
        roles.update({ROLE_DESSERT_BASE, ROLE_SNACK_SWEET, ROLE_STAPLE_CORE})
    if _has_any(text, STARCHY_PRODUCE_HINTS):
        roles.add(ROLE_PRODUCE_STARCHY)

    if _has_any(text, STAPLE_HINTS):
        roles.add(ROLE_STAPLE_CORE)
    if roles & {ROLE_MILK_FRESH, ROLE_MILK_EVAP, ROLE_MILK_COND, ROLE_MILK_POWDER, ROLE_CHEESE_COOKING, ROLE_CHEESE_SNACK, ROLE_CHEESE_SPREAD, ROLE_CREAM_COOKING, ROLE_CREAM_TABLE}:
        roles.add(ROLE_DAIRY)
    if not roles:
        # Conservative fallback: unknown food-like items become staple-core only if common pantry hints exist.
        if any(x in text for x in ("food", "grocery", "ingredient")):
            roles.add(ROLE_STAPLE_CORE)
    return frozenset(sorted(roles, key=lambda r: ROLE_PRIORITY.index(r) if r in ROLE_PRIORITY else 999))


def _contains(roles: frozenset[str], *expected: str) -> bool:
    return any(role in roles for role in expected)


def _text_has_all(text: str, *parts: str) -> bool:
    return all(part in text for part in parts)


BEVERAGE_HINTS = frozenset({"soda", "cola", "soft drink", "juice", "vimto", "energy drink", "beverage", "drink"})
DESSERT_ROLES = frozenset({ROLE_DESSERT_BASE, ROLE_DESSERT_MIX, ROLE_DESSERT_READY, ROLE_CHOCOLATE, ROLE_SNACK_SWEET})
SNACK_ROLES = frozenset({ROLE_SNACK_SALTY, ROLE_SNACK_SWEET, ROLE_BISCUIT, ROLE_CHOCOLATE, ROLE_CHEESE_SNACK, ROLE_CHEESE_SPREAD})
COOK_ROLES = frozenset(
    {
        ROLE_PROTEIN,
        ROLE_CARB_BASE,
        ROLE_BREAD,
        ROLE_SAUCE,
        ROLE_FAT,
        ROLE_PRODUCE_AROMATIC,
        ROLE_PRODUCE_COOKING,
        ROLE_SEASONING,
    }
)
SERVING_DAIRY_ROLES = frozenset({ROLE_CREAM_TABLE, ROLE_MILK_EVAP, ROLE_MILK_FRESH, ROLE_MILK_COND})
BLOCKED_OCCASION_DAIRY_ROLES = frozenset({ROLE_MILK_POWDER, ROLE_CREAM_COOKING, ROLE_CHEESE_SPREAD, ROLE_CHEESE_COOKING})

MEAL_LOW_EXPRESSION_RULES = (
    lambda text: _text_has_all(text, "rice", "salt"),
    lambda text: _text_has_all(text, "ghee", "salt"),
    lambda text: _text_has_all(text, "rice", "oil") and "tomato paste" not in text,
    lambda text: _text_has_all(text, "rice", "tomato paste"),
    lambda text: _text_has_all(text, "pasta", "pasta"),
    lambda text: _text_has_all(text, "eggs", "milk"),
    lambda text: _text_has_all(text, "oats", "chicken"),
    lambda text: _text_has_all(text, "oats", "eggs"),
    lambda text: _text_has_all(text, "oats", "tuna"),
    lambda text: _text_has_all(text, "rice", "carrot"),
    lambda text: _text_has_all(text, "rice", "carrots"),
    lambda text: _text_has_all(text, "chicken", "condensed"),
    lambda text: _text_has_all(text, "chicken", "honey"),
    lambda text: _text_has_all(text, "rice", "mustard"),
    lambda text: _text_has_all(text, "eggs", "carrot"),
    lambda text: _text_has_all(text, "eggs", "carrots"),
    lambda text: _text_has_all(text, "oil", "chips"),
    lambda text: _text_has_all(text, "tomato paste", "nuggets"),
    lambda text: _text_has_all(text, "milk", "soup powder"),
    lambda text: _text_has_all(text, "chicken", "peanut butter"),
)

OCCASION_LOW_EXPRESSION_RULES = (
    lambda text: _text_has_all(text, "tea", "sugar"),
    lambda text: _text_has_all(text, "cooking cream", "biscuit"),
    lambda text: _text_has_all(text, "condensed", "biscuit"),
    lambda text: _text_has_all(text, "cream cheese", "condensed"),
    lambda text: _text_has_all(text, "mozzarella", "condensed"),
    lambda text: _text_has_all(text, "syrup", "candy"),
    lambda text: _text_has_all(text, "milk", "vimto"),
    lambda text: _text_has_all(text, "biscuit", "cheese"),
)


@dataclass(frozen=True)
class PairFacts:
    roles_a: frozenset[str]
    roles_b: frozenset[str]
    text_a: str
    text_b: str
    text: str
    union_roles: frozenset[str]
    a_beverage: bool
    b_beverage: bool
    a_dessert: bool
    b_dessert: bool
    a_snack: bool
    b_snack: bool
    a_cook: bool
    b_cook: bool
    a_staple: bool
    b_staple: bool
    has_protein: bool
    has_carb: bool
    has_sauce: bool
    has_fat: bool
    has_cook_produce: bool
    has_serving_dairy: bool
    has_blocked_occasion_dairy: bool


def _has_role(roles: frozenset[str], expected: frozenset[str]) -> bool:
    return bool(set(roles) & set(expected))


def _is_beverage(roles: frozenset[str], text: str) -> bool:
    return _contains(roles, ROLE_TEA, ROLE_COFFEE, ROLE_MILK_FRESH, ROLE_MILK_EVAP, ROLE_MILK_COND) or any(
        hint in text for hint in BEVERAGE_HINTS
    )


def _pair_facts(roles_a: frozenset[str], roles_b: frozenset[str], text_a: str, text_b: str) -> PairFacts:
    union_roles = frozenset(set(roles_a) | set(roles_b))
    return PairFacts(
        roles_a=roles_a,
        roles_b=roles_b,
        text_a=text_a,
        text_b=text_b,
        text=f"{text_a} {text_b}",
        union_roles=union_roles,
        a_beverage=_is_beverage(roles_a, text_a),
        b_beverage=_is_beverage(roles_b, text_b),
        a_dessert=_has_role(roles_a, DESSERT_ROLES),
        b_dessert=_has_role(roles_b, DESSERT_ROLES),
        a_snack=_has_role(roles_a, SNACK_ROLES),
        b_snack=_has_role(roles_b, SNACK_ROLES),
        a_cook=_has_role(roles_a, COOK_ROLES),
        b_cook=_has_role(roles_b, COOK_ROLES),
        a_staple=ROLE_STAPLE_CORE in roles_a,
        b_staple=ROLE_STAPLE_CORE in roles_b,
        has_protein=_contains(union_roles, ROLE_PROTEIN),
        has_carb=_contains(union_roles, ROLE_CARB_BASE, ROLE_BREAD),
        has_sauce=_contains(union_roles, ROLE_SAUCE),
        has_fat=_contains(union_roles, ROLE_FAT),
        has_cook_produce=_contains(union_roles, ROLE_PRODUCE_AROMATIC, ROLE_PRODUCE_COOKING),
        has_serving_dairy=bool(set(union_roles) & set(SERVING_DAIRY_ROLES)),
        has_blocked_occasion_dairy=bool(set(union_roles) & set(BLOCKED_OCCASION_DAIRY_ROLES)),
    )


def _matches_any_rule(text: str, rules: tuple) -> bool:
    return any(rule(text) for rule in rules)


def _meal_visible_expression_ok(facts: PairFacts) -> tuple[bool, str | None]:
    if _matches_any_rule(facts.text, MEAL_LOW_EXPRESSION_RULES):
        return False, "meal_low_expression_pattern"
    if facts.has_carb and facts.has_cook_produce and not facts.has_protein and not facts.has_sauce and not facts.has_fat:
        return False, "meal_carb_produce_only"
    if facts.has_protein and (facts.has_carb or facts.has_sauce or facts.has_fat or facts.has_cook_produce):
        return True, None
    if facts.has_carb and (facts.has_sauce or facts.has_cook_produce):
        return True, None
    if _text_has_all(facts.text, "mayo", "tuna"):
        return True, None
    return False, "meal_expression_floor"


def _occasion_visible_expression_ok(facts: PairFacts) -> tuple[bool, str | None]:
    if _matches_any_rule(facts.text, OCCASION_LOW_EXPRESSION_RULES):
        return False, "occasion_low_expression_pattern"
    if ("tea" in facts.text and ("evaporated milk" in facts.text or "biscuit" in facts.text)) or (
        "coffee" in facts.text and ("evaporated milk" in facts.text or "biscuit" in facts.text)
    ):
        return True, None
    if "evaporated milk" in facts.text and "biscuit" in facts.text:
        return True, None
    if "dates" in facts.text:
        if facts.has_serving_dairy and not (facts.has_blocked_occasion_dairy and not facts.has_serving_dairy):
            return True, None
        return False, "occasion_dates_need_serving_dairy"
    if facts.has_blocked_occasion_dairy and not facts.has_serving_dairy:
        return False, "occasion_non_serving_dairy"
    if any(x in facts.text for x in ("dessert", "chocolate", "nutella", "caramel")) and facts.has_serving_dairy:
        return True, None
    return False, "occasion_expression_floor"


def classify_pair_relation(roles_a: frozenset[str], roles_b: frozenset[str], text_a: str, text_b: str) -> str:
    facts = _pair_facts(roles_a, roles_b, text_a, text_b)
    if ROLE_NONFOOD in roles_a or ROLE_NONFOOD in roles_b:
        return REL_HOUSEHOLD if ROLE_NONFOOD in roles_a and ROLE_NONFOOD in roles_b else REL_INVALID

    if (facts.a_dessert and facts.b_dessert) or (facts.a_dessert and _contains(roles_b, ROLE_DAIRY, ROLE_CREAM_TABLE, ROLE_CREAM_COOKING, ROLE_MILK_COND, ROLE_MILK_FRESH, ROLE_MILK_EVAP)) or (
        facts.b_dessert and _contains(roles_a, ROLE_DAIRY, ROLE_CREAM_TABLE, ROLE_CREAM_COOKING, ROLE_MILK_COND, ROLE_MILK_FRESH, ROLE_MILK_EVAP)
    ):
        return REL_DESSERT
    if ("dates" in text_a and _contains(roles_b, ROLE_MILK_FRESH, ROLE_MILK_EVAP, ROLE_CREAM_TABLE, ROLE_MILK_COND)) or (
        "dates" in text_b and _contains(roles_a, ROLE_MILK_FRESH, ROLE_MILK_EVAP, ROLE_CREAM_TABLE, ROLE_MILK_COND)
    ):
        return REL_DESSERT
    tea_or_coffee_a = _contains(roles_a, ROLE_TEA, ROLE_COFFEE)
    tea_or_coffee_b = _contains(roles_b, ROLE_TEA, ROLE_COFFEE)
    if (tea_or_coffee_a and "sugar" in text_b) or (tea_or_coffee_b and "sugar" in text_a):
        return REL_DRINK
    if facts.a_beverage and facts.b_beverage:
        return REL_DRINK
    if (facts.a_beverage and _contains(roles_b, ROLE_BISCUIT, ROLE_DESSERT_READY, ROLE_DESSERT_BASE)) or (
        facts.b_beverage and _contains(roles_a, ROLE_BISCUIT, ROLE_DESSERT_READY, ROLE_DESSERT_BASE)
    ):
        return REL_DRINK
    if facts.a_cook and facts.b_cook:
        return REL_COOK
    if (facts.a_snack and facts.b_snack) or (facts.a_snack and facts.b_beverage) or (facts.b_snack and facts.a_beverage):
        return REL_EAT
    if facts.a_staple and facts.b_staple:
        return REL_STAPLE
    if (facts.a_staple and facts.b_cook) or (facts.b_staple and facts.a_cook):
        return REL_STAPLE
    return REL_INVALID


def classify_pair_strength(roles_a: frozenset[str], roles_b: frozenset[str], relation: str, text_a: str, text_b: str) -> str:
    facts = _pair_facts(roles_a, roles_b, text_a, text_b)
    text = facts.text
    if relation == REL_INVALID:
        return STRENGTH_TRASH
    if ROLE_MILK_BABY in roles_a or ROLE_MILK_BABY in roles_b:
        return STRENGTH_TRASH

    strong_patterns = (
        ("tea", "evaporated milk"),
        ("coffee", "evaporated milk"),
        ("tea", "biscuit"),
        ("coffee", "biscuit"),
        ("biscuit", "fresh milk"),
        ("cookie", "fresh milk"),
        ("wafer", "chocolate"),
        ("nutella", "milk"),
        ("eggs", "ghee"),
        ("eggs", "bread"),
        ("mayo", "tuna"),
        ("kraft", "spring roll chips"),
        ("dessert", "fresh cream"),
    )
    for left, right in strong_patterns:
        if left in text and right in text:
            return STRENGTH_STRONG

    if any(x in text for x in ("milk onions", "onions milk", "chicken flakes", "cornflakes chicken", "baby milk")):
        return STRENGTH_TRASH
    if any(
        (
            _text_has_all(text, "tea", "biscuit"),
            _text_has_all(text, "coffee", "biscuit"),
            _text_has_all(text, "chocolate", "milk"),
            _text_has_all(text, "rice", "tomato paste"),
        )
    ):
        return STRENGTH_WEAK
    if any(
        (
            _text_has_all(text, "rice", "salt"),
            _text_has_all(text, "ghee", "salt"),
            _text_has_all(text, "rice", "oil"),
            _text_has_all(text, "pasta", "pasta"),
            _text_has_all(text, "eggs", "milk"),
            _text_has_all(text, "chicken", "condensed"),
            _text_has_all(text, "chicken", "honey"),
            _text_has_all(text, "oil", "chips"),
            _text_has_all(text, "tomato paste", "nuggets"),
            _text_has_all(text, "milk", "soup powder"),
            _text_has_all(text, "chicken", "peanut butter"),
            _text_has_all(text, "olive oil", "fish biscuits"),
            _text_has_all(text, "ketchup", "flour"),
            _text_has_all(text, "cream cheese", "condensed"),
            _text_has_all(text, "mozzarella", "condensed"),
            _text_has_all(text, "syrup", "candy"),
            _text_has_all(text, "milk", "vimto"),
        )
    ):
        return STRENGTH_TRASH
    if any((_text_has_all(text, "tea", "sugar"), _text_has_all(text, "biscuit", "cheese"))):
        return STRENGTH_WEAK
    if ("oats" in text and "tuna" in text) or ("topokki" in text and "eggs" in text):
        return STRENGTH_WEAK

    if relation == REL_HOUSEHOLD:
        return STRENGTH_STAPLE
    if relation == REL_DRINK:
        if _contains(roles_a, ROLE_TEA, ROLE_COFFEE) and _contains(roles_b, ROLE_MILK_EVAP, ROLE_MILK_FRESH, ROLE_BISCUIT):
            return STRENGTH_STRONG
        if _contains(roles_b, ROLE_TEA, ROLE_COFFEE) and _contains(roles_a, ROLE_MILK_EVAP, ROLE_MILK_FRESH, ROLE_BISCUIT):
            return STRENGTH_STRONG
        return STRENGTH_WEAK
    if relation == REL_DESSERT:
        if ("dates" in text_a and _contains(roles_b, ROLE_CREAM_TABLE, ROLE_MILK_FRESH, ROLE_MILK_EVAP)) or (
            "dates" in text_b and _contains(roles_a, ROLE_CREAM_TABLE, ROLE_MILK_FRESH, ROLE_MILK_EVAP)
        ):
            return STRENGTH_STRONG
        if _contains(roles_a, ROLE_CHOCOLATE, ROLE_DESSERT_BASE) and _contains(roles_b, ROLE_MILK_FRESH, ROLE_MILK_EVAP, ROLE_MILK_COND):
            return STRENGTH_STRONG
        if _contains(roles_b, ROLE_CHOCOLATE, ROLE_DESSERT_BASE) and _contains(roles_a, ROLE_MILK_FRESH, ROLE_MILK_EVAP, ROLE_MILK_COND):
            return STRENGTH_STRONG
        if _contains(roles_a, ROLE_DESSERT_READY, ROLE_DESSERT_BASE, ROLE_CHOCOLATE) and _contains(
            roles_b, ROLE_CREAM_TABLE, ROLE_MILK_FRESH, ROLE_MILK_EVAP
        ):
            return STRENGTH_STRONG
        if _contains(roles_b, ROLE_DESSERT_READY, ROLE_DESSERT_BASE, ROLE_CHOCOLATE) and _contains(
            roles_a, ROLE_CREAM_TABLE, ROLE_MILK_FRESH, ROLE_MILK_EVAP
        ):
            return STRENGTH_STRONG
        return STRENGTH_WEAK
    if relation == REL_COOK:
        if _contains(roles_a, ROLE_PROTEIN) and _contains(roles_b, ROLE_FAT, ROLE_SAUCE, ROLE_CARB_BASE, ROLE_BREAD):
            return STRENGTH_STRONG
        if _contains(roles_b, ROLE_PROTEIN) and _contains(roles_a, ROLE_FAT, ROLE_SAUCE, ROLE_CARB_BASE, ROLE_BREAD):
            return STRENGTH_STRONG
        if _contains(roles_a, ROLE_PROTEIN) and _contains(roles_b, ROLE_PRODUCE_FRUIT):
            return STRENGTH_TRASH
        if _contains(roles_b, ROLE_PROTEIN) and _contains(roles_a, ROLE_PRODUCE_FRUIT):
            return STRENGTH_TRASH
        return STRENGTH_STAPLE
    if relation == REL_EAT:
        if _contains(roles_a, ROLE_SNACK_SALTY, ROLE_SNACK_SWEET, ROLE_BISCUIT, ROLE_CHOCOLATE) and _contains(
            roles_b,
            ROLE_SNACK_SALTY,
            ROLE_SNACK_SWEET,
            ROLE_BISCUIT,
            ROLE_CHOCOLATE,
            ROLE_MILK_FRESH,
            ROLE_CHEESE_SNACK,
            ROLE_CHEESE_SPREAD,
        ):
            return STRENGTH_STRONG
        if _contains(roles_b, ROLE_SNACK_SALTY, ROLE_SNACK_SWEET, ROLE_BISCUIT, ROLE_CHOCOLATE) and _contains(
            roles_a,
            ROLE_MILK_FRESH,
            ROLE_CHEESE_SNACK,
            ROLE_CHEESE_SPREAD,
        ):
            return STRENGTH_STRONG
        return STRENGTH_WEAK
    if relation == REL_STAPLE:
        if _contains(roles_a, ROLE_STAPLE_CORE) and _contains(roles_b, ROLE_STAPLE_CORE):
            return STRENGTH_STAPLE
        return STRENGTH_WEAK
    return STRENGTH_WEAK


def is_pair_hard_invalid(
    lane: str,
    roles_a: frozenset[str],
    roles_b: frozenset[str],
    relation: str,
    strength: str,
    text_a: str,
    text_b: str,
) -> tuple[bool, str | None]:
    facts = _pair_facts(roles_a, roles_b, text_a, text_b)
    food_lane = lane in FOOD_LANES and lane != LANE_NONFOOD
    if food_lane and (ROLE_NONFOOD in roles_a or ROLE_NONFOOD in roles_b):
        return True, "nonfood_contamination"
    if lane == LANE_NONFOOD and (ROLE_NONFOOD not in roles_a or ROLE_NONFOOD not in roles_b):
        return True, "nonfood_required"

    if ROLE_MILK_BABY in roles_a or ROLE_MILK_BABY in roles_b:
        if _contains(roles_a | roles_b, ROLE_CHOCOLATE, ROLE_SNACK_SWEET, ROLE_SNACK_SALTY, ROLE_DESSERT_BASE, ROLE_DESSERT_MIX, ROLE_DESSERT_READY, ROLE_TEA, ROLE_COFFEE):
            return True, "baby_milk_incompatible"

    text = f"{text_a}::{text_b}"
    if "milk" in text and ("onion" in text or "garlic" in text):
        return True, "milk_allium"
    if "chicken" in text and "peanut butter" in text:
        return True, "chicken_peanut_butter"
    if "milk" in text and "soup powder" in text:
        return True, "milk_soup_powder"
    if "tomato paste" in text and "nuggets" in text:
        return True, "tomato_paste_nuggets"
    if "burger" in text and "rice pouch" in text:
        return True, "burger_rice_pouch"
    if "burger" in text and "ready rice" in text:
        return True, "burger_ready_rice"
    if _text_has_all(text, "olive oil", "fish biscuits"):
        return True, "olive_oil_fish_biscuits"
    if _text_has_all(text, "ketchup", "flour"):
        return True, "ketchup_flour"
    if "chicken" in text and ("flakes" in text or "cornflakes" in text):
        return True, "chicken_flakes"
    if ("tuna" in text and "flour" in text):
        return True, "tuna_flour"
    if ("eggs" in text and "topokki" in text and "cream" in text):
        return True, "eggs_topokki_cream"
    if ("tuna" in text or "chicken" in text) and any(f in text for f in ("banana", "strawberry", "orange", "apple", "grape", "pineapple")):
        return True, "protein_fruit"
    if relation == REL_INVALID:
        return True, "invalid_relation"

    if lane == LANE_MEAL:
        if _contains(roles_a, ROLE_SEASONING) and _contains(roles_b, ROLE_SEASONING):
            return True, "meal_spice_spice"
        if relation in {REL_DESSERT, REL_DRINK}:
            return True, "meal_not_serving"
        if strength == STRENGTH_TRASH:
            return True, "meal_trash"
        if _text_has_all(text, "rice", "salt"):
            return True, "meal_rice_salt"
        if _text_has_all(text, "ghee", "salt"):
            return True, "meal_ghee_salt"
        if _text_has_all(text, "rice", "oil") and "tomato paste" not in text:
            return True, "meal_rice_oil"
        if _text_has_all(text, "pasta", "pasta"):
            return True, "meal_duplicate_pasta"
        if _text_has_all(text, "eggs", "milk"):
            return True, "meal_eggs_milk"
        if _text_has_all(text, "oats", "chicken"):
            return True, "meal_oats_chicken"
        if _text_has_all(text, "oats", "eggs"):
            return True, "meal_oats_eggs"
        if _text_has_all(text, "oats", "tuna"):
            return True, "meal_oats_tuna"
        if _text_has_all(text, "rice", "tomato paste"):
            return True, "meal_rice_tomato_paste"
        if _text_has_all(text, "tomato paste", "nuggets"):
            return True, "meal_tomato_paste_nuggets"
        if "chicken" in text and ("condensed" in text or "honey" in text or "cinnamon" in text):
            return True, "meal_savory_sweet_contamination"
        if "chicken" in text and "peanut butter" in text:
            return True, "meal_chicken_peanut_butter"
        if "milk" in text and "soup powder" in text:
            return True, "meal_milk_soup_powder"
        if "burger" in text and "rice" in text and not any(token in text for token in ("sauce", "tomato", "spice", "onion", "garlic")):
            return True, "meal_burger_plain_rice"
        if _text_has_all(text, "oil", "chips"):
            return True, "meal_oil_chips"
        if _contains(roles_a, ROLE_CARB_BASE) and _contains(roles_b, ROLE_CARB_BASE):
            return True, "meal_duplicate_carb"
    if lane == LANE_SNACK:
        if _contains(roles_a | roles_b, ROLE_MILK_BABY, ROLE_PROTEIN, ROLE_PRODUCE_AROMATIC, ROLE_PRODUCE_COOKING):
            return True, "snack_cooking_heavy"
        if relation in {REL_COOK}:
            return True, "snack_cook_relation"
    if lane == LANE_OCCASION:
        if "water" in text:
            return True, "occasion_plain_water"
        if _text_has_all(text, "tea", "sugar"):
            return True, "occasion_tea_sugar"
        if ("cooking cream" in text and any(x in text for x in ("biscuit", "biscuits", "cookie", "cracker"))):
            return True, "occasion_cooking_cream_biscuit"
        if "condensed" in text and any(
            x in text
            for x in (
                "pasta sauce",
                "tomato paste",
                "cream cheese",
                "cheese spread",
                "mozzarella",
            )
        ):
            return True, "occasion_condensed_non_serving"
        if ("syrup" in text or "vimto" in text or "juice" in text) and ("milk" in text or "candy" in text):
            return True, "occasion_syrup_noise"
        if _contains(roles_a | roles_b, ROLE_SNACK_SALTY) and _contains(
            roles_a | roles_b,
            ROLE_DESSERT_BASE,
            ROLE_DESSERT_MIX,
            ROLE_DESSERT_READY,
            ROLE_CHOCOLATE,
            ROLE_SNACK_SWEET,
        ):
            return True, "occasion_snack_dessert_mismatch"
        if _contains(roles_a | roles_b, ROLE_BISCUIT) and _contains(roles_a | roles_b, ROLE_CHEESE_SPREAD, ROLE_CHEESE_COOKING):
            return True, "occasion_biscuit_cheese"
        if _contains(roles_a | roles_b, ROLE_PROTEIN, ROLE_PRODUCE_AROMATIC, ROLE_SEASONING):
            return True, "occasion_savory_prep"
        if relation == REL_COOK:
            return True, "occasion_cook_relation"
    return False, None


def lane_compatibility(
    lane: str,
    relation: str,
    strength: str,
    roles_a: frozenset[str],
    roles_b: frozenset[str],
) -> tuple[bool, float, str | None]:
    facts = _pair_facts(roles_a, roles_b, "", "")
    if strength == STRENGTH_TRASH:
        return False, 0.0, "trash_strength"
    if lane == LANE_NONFOOD:
        return (relation == REL_HOUSEHOLD), (0.95 if relation == REL_HOUSEHOLD else 0.0), ("lane_nonfood_mismatch" if relation != REL_HOUSEHOLD else None)
    if lane == LANE_MEAL:
        if relation == REL_COOK:
            if facts.has_protein and (facts.has_carb or facts.has_sauce or facts.has_fat or facts.has_cook_produce):
                return True, 0.95, None
            if facts.has_carb and (facts.has_sauce or facts.has_cook_produce):
                return True, 0.88, None
            return False, 0.0, "meal_low_expression_cook"
        if relation == REL_EAT and facts.has_protein and _contains(roles_a | roles_b, ROLE_BREAD, ROLE_SAUCE):
            return True, 0.78, None
        if relation == REL_STAPLE:
            return False, 0.0, "meal_staples_not_visible"
        return False, 0.0, "meal_relation_mismatch"
    if lane == LANE_SNACK:
        if relation == REL_EAT:
            return True, 0.92, None
        if relation == REL_DESSERT:
            return True, 0.88, None
        if relation == REL_DRINK and _contains(roles_a | roles_b, ROLE_SNACK_SALTY, ROLE_SNACK_SWEET, ROLE_BISCUIT, ROLE_CHOCOLATE):
            return True, 0.62, None
        return False, 0.0, "snack_relation_mismatch"
    if lane == LANE_OCCASION:
        if relation == REL_DRINK:
            has_tea_coffee = _contains(roles_a | roles_b, ROLE_TEA, ROLE_COFFEE)
            has_biscuit = _contains(roles_a | roles_b, ROLE_BISCUIT)
            has_evap = _contains(roles_a | roles_b, ROLE_MILK_EVAP)
            has_serving_milk = _contains(roles_a | roles_b, ROLE_MILK_EVAP, ROLE_MILK_FRESH)
            if (has_tea_coffee and (has_biscuit or has_evap or has_serving_milk)) or (has_evap and has_biscuit):
                return True, 0.95, None
            return False, 0.0, "occasion_drink_not_serving"
        if relation == REL_DESSERT:
            if _contains(roles_a | roles_b, ROLE_DESSERT_READY, ROLE_DESSERT_BASE, ROLE_CHOCOLATE) and _contains(
                roles_a | roles_b, ROLE_CREAM_TABLE, ROLE_MILK_EVAP, ROLE_MILK_FRESH, ROLE_MILK_COND
            ):
                return True, 0.88, None
            return False, 0.0, "occasion_dessert_not_serving"
        if relation == REL_EAT and _contains(roles_a | roles_b, ROLE_BISCUIT) and _contains(roles_a | roles_b, ROLE_TEA, ROLE_COFFEE):
            return True, 0.70, None
        if relation == REL_STAPLE:
            return False, 0.0, "occasion_staples_not_visible"
        return False, 0.0, "occasion_relation_mismatch"
    if lane == LANE_STAPLES:
        if relation == REL_STAPLE:
            return True, 0.95, None
        if relation == REL_COOK:
            return True, 0.65, None
        return False, 0.0, "staples_relation_mismatch"
    return False, 0.0, "unknown_lane"


def visible_lane_expression_ok(
    lane: str,
    relation: str,
    strength: str,
    roles_a: frozenset[str],
    roles_b: frozenset[str],
    text_a: str,
    text_b: str,
) -> tuple[bool, str | None]:
    if lane not in {LANE_MEAL, LANE_SNACK, LANE_OCCASION}:
        return True, None
    if strength == STRENGTH_TRASH:
        return False, "trash_strength"

    facts = _pair_facts(roles_a, roles_b, text_a, text_b)

    if lane == LANE_MEAL:
        return _meal_visible_expression_ok(facts)

    if lane == LANE_SNACK:
        if relation not in {REL_EAT, REL_DESSERT, REL_DRINK}:
            return False, "snack_relation_floor"
        if _contains(roles_a | roles_b, ROLE_PROTEIN, ROLE_PRODUCE_AROMATIC, ROLE_PRODUCE_COOKING, ROLE_MILK_BABY, ROLE_SEASONING):
            return False, "snack_cooking_heavy"
        if _contains(roles_a | roles_b, ROLE_CHEESE_COOKING, ROLE_CREAM_COOKING, ROLE_MILK_POWDER):
            return False, "snack_prep_dairy"
        if not _contains(
            roles_a | roles_b,
            ROLE_SNACK_SALTY,
            ROLE_SNACK_SWEET,
            ROLE_BISCUIT,
            ROLE_CHOCOLATE,
            ROLE_DESSERT_READY,
            ROLE_DESSERT_BASE,
            ROLE_CHEESE_SNACK,
            ROLE_CHEESE_SPREAD,
        ):
            return False, "snack_expression_floor"
        return True, None

    # occasion
    return _occasion_visible_expression_ok(facts)


def semantic_score_prior(strength: str) -> float:
    if strength == STRENGTH_STRONG:
        return 0.65
    if strength == STRENGTH_STAPLE:
        return 0.28
    if strength == STRENGTH_WEAK:
        return 0.05
    return -1.0


def semantic_soft_penalty(
    lane: str,
    relation: str,
    strength: str,
    roles_a: frozenset[str],
    roles_b: frozenset[str],
    text_a: str,
    text_b: str,
) -> float:
    penalty = 0.0
    if strength == STRENGTH_WEAK:
        penalty += 0.20
    if lane == LANE_SNACK and relation == REL_STAPLE:
        penalty += 0.28
    if lane == LANE_OCCASION and relation == REL_STAPLE:
        penalty += 0.22
    if "condensed milk" in f"{text_a} {text_b}" and ("syrup" in f"{text_a} {text_b}" or "vimto" in f"{text_a} {text_b}"):
        penalty += 0.30
    return float(min(0.8, penalty))


def anchor_lane_eligibility(lane: str, roles: frozenset[str], text: str) -> tuple[bool, float, str | None]:
    if ROLE_NONFOOD in roles and lane in FOOD_LANES:
        return False, 0.0, "anchor_nonfood_food_lane"
    if lane == LANE_NONFOOD:
        return (ROLE_NONFOOD in roles), (1.0 if ROLE_NONFOOD in roles else 0.0), ("anchor_nonfood_required" if ROLE_NONFOOD not in roles else None)

    if lane == LANE_MEAL:
        if _contains(roles, ROLE_MILK_BABY, ROLE_DESSERT_READY, ROLE_DESSERT_MIX, ROLE_DESSERT_BASE, ROLE_SNACK_SALTY, ROLE_SNACK_SWEET):
            return False, 0.0, "meal_anchor_non_meal"
        if _contains(roles, ROLE_SEASONING) and len(roles) == 1:
            return False, 0.0, "meal_anchor_spice_only"
        if _contains(roles, ROLE_PROTEIN, ROLE_CARB_BASE, ROLE_BREAD, ROLE_SAUCE, ROLE_FAT, ROLE_PRODUCE_AROMATIC, ROLE_PRODUCE_COOKING, ROLE_STAPLE_CORE):
            return True, 1.0, None
        return False, 0.0, "meal_anchor_low_fit"

    if lane == LANE_SNACK:
        if _contains(roles, ROLE_PROTEIN, ROLE_SEASONING, ROLE_PRODUCE_AROMATIC, ROLE_PRODUCE_COOKING, ROLE_MILK_BABY):
            return False, 0.0, "snack_anchor_cooking_heavy"
        if _contains(roles, ROLE_SNACK_SALTY, ROLE_SNACK_SWEET, ROLE_BISCUIT, ROLE_CHOCOLATE, ROLE_CHEESE_SNACK, ROLE_DESSERT_READY):
            return True, 1.0, None
        if _contains(roles, ROLE_TEA, ROLE_COFFEE, ROLE_MILK_FRESH, ROLE_MILK_EVAP):
            return True, 0.7, None
        return False, 0.0, "snack_anchor_low_fit"

    if lane == LANE_OCCASION:
        if _contains(roles, ROLE_PROTEIN, ROLE_PRODUCE_AROMATIC, ROLE_SEASONING, ROLE_MILK_BABY):
            return False, 0.0, "occasion_anchor_savory"
        if _contains(roles, ROLE_TEA, ROLE_COFFEE, ROLE_BISCUIT, ROLE_MILK_EVAP, ROLE_DESSERT_READY, ROLE_CREAM_TABLE, ROLE_MILK_COND):
            return True, 1.0, None
        if "dates" in text:
            return True, 0.95, None
        return False, 0.0, "occasion_anchor_low_fit"

    if lane == LANE_STAPLES:
        return (_contains(roles, ROLE_STAPLE_CORE, ROLE_CARB_BASE, ROLE_SAUCE, ROLE_FAT, ROLE_BREAD, ROLE_MILK_POWDER)), (
            1.0 if _contains(roles, ROLE_STAPLE_CORE, ROLE_CARB_BASE, ROLE_SAUCE, ROLE_FAT, ROLE_BREAD, ROLE_MILK_POWDER) else 0.0
        ), ("staples_anchor_low_fit" if not _contains(roles, ROLE_STAPLE_CORE, ROLE_CARB_BASE, ROLE_SAUCE, ROLE_FAT, ROLE_BREAD, ROLE_MILK_POWDER) else None)
    return False, 0.0, "unknown_lane"


@dataclass(frozen=True)
class BundleSemantics:
    roles_a: tuple[str, ...]
    roles_b: tuple[str, ...]
    relation: str
    strength: str
    hard_invalid: bool
    hard_invalid_reason: str | None
    lane_allowed: bool
    lane_fit_score: float
    lane_reason: str | None
    internal_lane_fit: str


def classify_bundle_semantics(
    lane: str,
    name_a: str,
    category_a: str,
    family_a: str,
    name_b: str,
    category_b: str,
    family_b: str,
) -> BundleSemantics:
    text_a = normalize_product_text(name_a, category_a, family_a)
    text_b = normalize_product_text(name_b, category_b, family_b)
    roles_a = infer_product_roles(name_a, category_a, family_a)
    roles_b = infer_product_roles(name_b, category_b, family_b)
    relation = classify_pair_relation(roles_a, roles_b, text_a, text_b)
    strength = classify_pair_strength(roles_a, roles_b, relation, text_a, text_b)
    hard_invalid, hard_reason = is_pair_hard_invalid(lane, roles_a, roles_b, relation, strength, text_a, text_b)
    lane_allowed, lane_fit, lane_reason = lane_compatibility(lane, relation, strength, roles_a, roles_b)

    best_lane = lane
    best_fit = lane_fit
    for candidate_lane in (LANE_MEAL, LANE_SNACK, LANE_OCCASION, LANE_STAPLES):
        allowed, fit, _reason = lane_compatibility(candidate_lane, relation, strength, roles_a, roles_b)
        if allowed and fit > best_fit:
            best_fit = fit
            best_lane = candidate_lane

    return BundleSemantics(
        roles_a=tuple(sorted(roles_a)),
        roles_b=tuple(sorted(roles_b)),
        relation=relation,
        strength=strength,
        hard_invalid=bool(hard_invalid),
        hard_invalid_reason=hard_reason,
        lane_allowed=bool(lane_allowed and not hard_invalid),
        lane_fit_score=float(lane_fit),
        lane_reason=lane_reason,
        internal_lane_fit=str(best_lane),
    )
