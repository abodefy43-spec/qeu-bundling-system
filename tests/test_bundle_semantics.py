from __future__ import annotations

import unittest

from qeu_bundling.presentation import bundle_semantics as bs


def _roles(name: str, category: str = "", family: str = "") -> frozenset[str]:
    return bs.infer_product_roles(name, category, family)


def _relation(name_a: str, name_b: str) -> str:
    a_text = bs.normalize_product_text(name_a, "", "")
    b_text = bs.normalize_product_text(name_b, "", "")
    return bs.classify_pair_relation(_roles(name_a), _roles(name_b), a_text, b_text)


def _strength(name_a: str, name_b: str) -> str:
    rel = _relation(name_a, name_b)
    a_text = bs.normalize_product_text(name_a, "", "")
    b_text = bs.normalize_product_text(name_b, "", "")
    return bs.classify_pair_strength(_roles(name_a), _roles(name_b), rel, a_text, b_text)


class BundleSemanticsTests(unittest.TestCase):
    def _visible(self, lane: str, name_a: str, name_b: str) -> bool:
        roles_a = _roles(name_a)
        roles_b = _roles(name_b)
        text_a = bs.normalize_product_text(name_a)
        text_b = bs.normalize_product_text(name_b)
        rel = bs.classify_pair_relation(roles_a, roles_b, text_a, text_b)
        strength = bs.classify_pair_strength(roles_a, roles_b, rel, text_a, text_b)
        ok, _reason = bs.visible_lane_expression_ok(lane, rel, strength, roles_a, roles_b, text_a, text_b)
        return bool(ok)

    def test_role_inference_core(self):
        self.assertIn(bs.ROLE_PROTEIN, _roles("eggs 30 count"))
        self.assertIn(bs.ROLE_STAPLE_CORE, _roles("eggs 30 count"))
        self.assertIn(bs.ROLE_FAT, _roles("farm ghee 800g"))
        self.assertIn(bs.ROLE_TEA, _roles("black tea bags"))
        self.assertIn(bs.ROLE_MILK_EVAP, _roles("evaporated milk 170g"))
        self.assertIn(bs.ROLE_MILK_COND, _roles("sweetened condensed milk"))
        self.assertIn(bs.ROLE_MILK_BABY, _roles("yamani baby milk stage 2"))
        self.assertIn(bs.ROLE_CHOCOLATE, _roles("nutella chocolate spread"))
        self.assertIn(bs.ROLE_SAUCE, _roles("tomato paste"))
        self.assertIn(bs.ROLE_NONFOOD, _roles("paper tissues pack"))
        self.assertIn(bs.ROLE_NONFOOD, _roles("skin soothing moisturizing cream 50 ml"))

    def test_relation_classification(self):
        self.assertEqual(_relation("tea", "evaporated milk"), bs.REL_DRINK)
        self.assertEqual(_relation("coffee", "evaporated milk"), bs.REL_DRINK)
        self.assertEqual(_relation("nutella chocolate spread", "fresh milk"), bs.REL_DESSERT)
        self.assertEqual(_relation("eggs", "ghee"), bs.REL_COOK)
        self.assertEqual(_relation("eggs", "toast bread"), bs.REL_COOK)
        self.assertIn(_relation("mayo", "tuna"), {bs.REL_EAT, bs.REL_COOK})
        self.assertIn(_relation("kraft cheese 50g", "spring roll chips"), {bs.REL_EAT, bs.REL_DESSERT})
        self.assertIn(_relation("rice", "tomato paste"), {bs.REL_STAPLE, bs.REL_COOK})

    def test_invalid_patterns(self):
        roles_a = _roles("full cream milk")
        roles_b = _roles("fresh onions")
        rel = bs.classify_pair_relation(roles_a, roles_b, bs.normalize_product_text("full cream milk"), bs.normalize_product_text("fresh onions"))
        strength = bs.classify_pair_strength(roles_a, roles_b, rel, bs.normalize_product_text("full cream milk"), bs.normalize_product_text("fresh onions"))
        hard_invalid, reason = bs.is_pair_hard_invalid(
            bs.LANE_MEAL,
            roles_a,
            roles_b,
            rel,
            strength,
            bs.normalize_product_text("full cream milk"),
            bs.normalize_product_text("fresh onions"),
        )
        self.assertTrue(hard_invalid)
        self.assertIn(reason, {"milk_allium", "invalid_relation"})

    def test_hard_invalid_olive_oil_plus_fish_biscuits(self):
        a = "olive oil"
        b = "fish biscuits"
        roles_a = _roles(a)
        roles_b = _roles(b)
        text_a = bs.normalize_product_text(a)
        text_b = bs.normalize_product_text(b)
        rel = bs.classify_pair_relation(roles_a, roles_b, text_a, text_b)
        strength = bs.classify_pair_strength(roles_a, roles_b, rel, text_a, text_b)
        hard_invalid, reason = bs.is_pair_hard_invalid(bs.LANE_MEAL, roles_a, roles_b, rel, strength, text_a, text_b)
        self.assertTrue(hard_invalid)
        self.assertEqual(reason, "olive_oil_fish_biscuits")

    def test_hard_invalid_ketchup_plus_flour(self):
        a = "ketchup"
        b = "flour"
        roles_a = _roles(a)
        roles_b = _roles(b)
        text_a = bs.normalize_product_text(a)
        text_b = bs.normalize_product_text(b)
        rel = bs.classify_pair_relation(roles_a, roles_b, text_a, text_b)
        strength = bs.classify_pair_strength(roles_a, roles_b, rel, text_a, text_b)
        hard_invalid, reason = bs.is_pair_hard_invalid(bs.LANE_MEAL, roles_a, roles_b, rel, strength, text_a, text_b)
        self.assertTrue(hard_invalid)
        self.assertEqual(reason, "ketchup_flour")

    def test_strength_classes(self):
        self.assertEqual(_strength("eggs", "ghee"), bs.STRENGTH_STRONG)
        self.assertEqual(_strength("tea", "evaporated milk"), bs.STRENGTH_STRONG)
        self.assertEqual(_strength("rice", "tomato paste"), bs.STRENGTH_WEAK)
        self.assertEqual(_strength("tea", "biscuit"), bs.STRENGTH_STRONG)
        self.assertEqual(_strength("dates", "fresh cream"), bs.STRENGTH_STRONG)
        self.assertEqual(_strength("oats", "tuna"), bs.STRENGTH_WEAK)
        self.assertEqual(_strength("baby milk", "nutella"), bs.STRENGTH_TRASH)

    def test_lane_compatibility(self):
        roles_a = _roles("tea")
        roles_b = _roles("evaporated milk")
        rel = _relation("tea", "evaporated milk")
        strength = _strength("tea", "evaporated milk")
        ok, fit, _reason = bs.lane_compatibility(bs.LANE_OCCASION, rel, strength, roles_a, roles_b)
        self.assertTrue(ok)
        self.assertGreater(fit, 0.6)

        roles_a = _roles("milk")
        roles_b = _roles("onions")
        rel = _relation("milk", "onions")
        strength = _strength("milk", "onions")
        ok, _fit, _reason = bs.lane_compatibility(bs.LANE_MEAL, rel, strength, roles_a, roles_b)
        self.assertFalse(ok)

    def test_anchor_eligibility(self):
        allow, _fit, _reason = bs.anchor_lane_eligibility(bs.LANE_MEAL, _roles("eggs"), bs.normalize_product_text("eggs"))
        self.assertTrue(allow)
        allow, _fit, _reason = bs.anchor_lane_eligibility(bs.LANE_SNACK, _roles("flour"), bs.normalize_product_text("flour"))
        self.assertFalse(allow)
        allow, _fit, _reason = bs.anchor_lane_eligibility(bs.LANE_OCCASION, _roles("evaporated milk"), bs.normalize_product_text("evaporated milk"))
        self.assertTrue(allow)
        allow, _fit, _reason = bs.anchor_lane_eligibility(bs.LANE_OCCASION, _roles("chicken breast"), bs.normalize_product_text("chicken breast"))
        self.assertFalse(allow)

    def test_visible_expression_floor_blocks_low_expression_meal_pairs(self):
        a = "rice"
        b = "salt"
        roles_a = _roles(a)
        roles_b = _roles(b)
        text_a = bs.normalize_product_text(a)
        text_b = bs.normalize_product_text(b)
        rel = bs.classify_pair_relation(roles_a, roles_b, text_a, text_b)
        strength = bs.classify_pair_strength(roles_a, roles_b, rel, text_a, text_b)
        ok, _reason = bs.visible_lane_expression_ok(bs.LANE_MEAL, rel, strength, roles_a, roles_b, text_a, text_b)
        self.assertFalse(ok)

    def test_visible_expression_floor_blocks_occasion_noise(self):
        a = "tea"
        b = "sugar"
        roles_a = _roles(a)
        roles_b = _roles(b)
        text_a = bs.normalize_product_text(a)
        text_b = bs.normalize_product_text(b)
        rel = bs.classify_pair_relation(roles_a, roles_b, text_a, text_b)
        strength = bs.classify_pair_strength(roles_a, roles_b, rel, text_a, text_b)
        ok, _reason = bs.visible_lane_expression_ok(bs.LANE_OCCASION, rel, strength, roles_a, roles_b, text_a, text_b)
        self.assertFalse(ok)

    def test_visible_expression_floor_keeps_known_good_pairs(self):
        a = "tea"
        b = "evaporated milk"
        roles_a = _roles(a)
        roles_b = _roles(b)
        text_a = bs.normalize_product_text(a)
        text_b = bs.normalize_product_text(b)
        rel = bs.classify_pair_relation(roles_a, roles_b, text_a, text_b)
        strength = bs.classify_pair_strength(roles_a, roles_b, rel, text_a, text_b)
        ok, _reason = bs.visible_lane_expression_ok(bs.LANE_OCCASION, rel, strength, roles_a, roles_b, text_a, text_b)
        self.assertTrue(ok)

    def test_keeper_visible_pairs_remain_allowed(self):
        self.assertTrue(self._visible(bs.LANE_OCCASION, "tea", "evaporated milk"))
        self.assertTrue(self._visible(bs.LANE_OCCASION, "coffee", "evaporated milk"))
        self.assertTrue(self._visible(bs.LANE_OCCASION, "tea", "biscuit"))
        self.assertTrue(self._visible(bs.LANE_OCCASION, "dates", "fresh cream"))
        self.assertTrue(self._visible(bs.LANE_OCCASION, "dates", "fresh milk"))
        self.assertTrue(self._visible(bs.LANE_OCCASION, "dessert pudding", "fresh cream"))
        self.assertTrue(self._visible(bs.LANE_MEAL, "eggs", "toast bread"))
        self.assertTrue(self._visible(bs.LANE_SNACK, "chocolate", "fresh milk"))

    def test_meal_visible_expression_blocks_weak_leak_pairs(self):
        self.assertFalse(self._visible(bs.LANE_MEAL, "oats", "eggs"))
        self.assertFalse(self._visible(bs.LANE_MEAL, "oats", "tuna"))
        self.assertFalse(self._visible(bs.LANE_MEAL, "rice", "carrots"))
        self.assertFalse(self._visible(bs.LANE_MEAL, "eggs", "fresh carrots"))
        self.assertFalse(self._visible(bs.LANE_MEAL, "basmati rice", "honey mustard sauce"))
        self.assertFalse(self._visible(bs.LANE_MEAL, "rice", "tomato paste"))
        self.assertFalse(self._visible(bs.LANE_MEAL, "bread", "oats"))
        self.assertFalse(self._visible(bs.LANE_MEAL, "chicken", "milk"))

    def test_meal_visible_expression_keeps_known_good_pairs(self):
        self.assertTrue(self._visible(bs.LANE_MEAL, "eggs", "tomatoes"))
        self.assertTrue(self._visible(bs.LANE_MEAL, "tomato", "chicken"))
        self.assertTrue(self._visible(bs.LANE_MEAL, "eggs", "ghee"))

    def test_occasion_visible_expression_requires_serving_context(self):
        self.assertFalse(self._visible(bs.LANE_OCCASION, "dessert pudding", "milk powder"))
        self.assertFalse(self._visible(bs.LANE_OCCASION, "dessert pudding", "triangle cheese"))
        self.assertFalse(self._visible(bs.LANE_OCCASION, "dessert pudding", "biscuit"))
        self.assertFalse(self._visible(bs.LANE_OCCASION, "dates", "cooking cream"))

    def test_snack_visible_expression_blocks_prep_dairy_pairs(self):
        self.assertFalse(self._visible(bs.LANE_SNACK, "potato chips", "grated mozzarella cheese"))
        self.assertFalse(self._visible(bs.LANE_SNACK, "dessert pudding", "cooking cream"))
        self.assertTrue(self._visible(bs.LANE_SNACK, "chocolate", "fresh milk"))
        self.assertTrue(self._visible(bs.LANE_SNACK, "chocolate biscuit", "fresh milk"))


if __name__ == "__main__":
    unittest.main()
