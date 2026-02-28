# QEU Product Bundling System - Prompt 3: Shared Categories Fix

## This is Prompt 3 - Fixes the Bundle Issue

Read Prompt 1 and Prompt 2 first.

---

## Problem (From Results)

Current bundles are BAD:
- Same product (different sizes) ❌
- Only share 1 category ❌
- Random combos ❌

---

## Solution: Shared Categories Priority

### What Makes a GOOD Bundle:
**Products that share MANY categories together**

Example: Cucumber + Tomato
- Both in "salads" category ✓
- Both in "vegetables" ✓
- Both in "fresh produce" ✓
- Both in "Ramadan" ✓
- Both in " Saudi" ✓
- → They share 5+ categories = GOOD BUNDLE

---

## NEW Priority Order:

```
1. Products share 5+ categories → BEST BUNDLE
2. Products share 3-4 categories → GOOD BUNDLE
3. Products share 1-2 categories → OK
4. Same product (different sizes) → BLOCK
```

---

## How to Implement

### Step 1: Define Categories for Each Product

From recipe_data.json, each product has categories:
- From `saudi_ramadan_dishes`: which dishes it belongs to
- From `categories`: grains, protein, dairy, vegetables, etc.
- From `ingredients`: recipe relevance
- From order data: purchase frequency

### Step 2: Assign Multiple Categories to Each Product

Create a category mapping:

```
product_name → [category1, category2, category3, ...]

Example:
"الخضار بطاطس" (potatoes) → ["vegetables", "ramadan", "saudi_staples", "comfort_food", "fried", "soups", "stews", "salads"]
"الخضار طماطم" (tomatoes) → ["vegetables", "ramadan", "saudi_staples", "salads", "sauces", "soups", "cooking"]
```

### Step 3: Count Shared Categories

```
shared_categories = len(set(categories_a) & set(categories_b))

Example:
potatoes categories: ["vegetables", "ramadan", "saudi_staples", "comfort_food", "fried", "soups", "stews", "salads"]
tomatoes categories: ["vegetables", "ramadan", "saudi_staples", "salads", "sauces", "soups", "cooking"]

Shared = {"vegetables", "ramadan", "saudi_staples", "salads", "soups"} = 5 categories
```

### Step 4: Scoring Based on Shared Categories

```
Category Score = shared_categories × 20 (scale to 0-100)

5 shared = 100
4 shared = 80
3 shared = 60
2 shared = 40
1 shared = 20
0 shared = 0
```

---

## Updated Scoring Formula

### NEW: 30/30/30/10 (Shared Categories / Recipe / Copurchase / Embedding)

```
Final Score = (Shared_Category_Score × 0.30) + 
              (Recipe_Score × 0.30) + 
              (CoPurchase_Score × 0.30) + 
              (Embedding_Score × 0.10)
```

### Filter Rules:

1. **BLOCK**: Same product (different sizes)
   - If product_a and product_b have similar names → REMOVE

2. **MINIMUM**: Must have at least 2 shared categories

3. **SORT**: By shared_categories descending

---

## Example Results Should Be:

| Bundle | Shared Categories | Score |
|--------|------------------|-------|
| Cucumber + Tomato | 5 (vegetables, salads, Ramadan, Saudi, fresh) | HIGH |
| Rice + Chicken | 4 (ramadan, saudi, main_dish, protein) | HIGH |
| Bread + Cheese | 3 (breakfast, sandwiches, easy_meal) | MEDIUM |
| Tomato + Flour | 1 (baking) | LOW |
| Chicken Large + Chicken Small | SAME PRODUCT → BLOCK | ❌ |

---

## Updated Implementation

### Phase 3: Category Assignment - UPDATED

```
3.1 Build Category Mapping
    - From recipe_data.json: saudi_ramadan_dishes, categories, ingredients
    - From order data: purchase frequency category
    - Create 10-15 categories per product

3.2 Assign Categories to All Products
    - Use keyword matching
    - Each product gets: [cat1, cat2, cat3, ...]

3.3 Calculate Shared Categories
    - For each pair: shared = len(set(cats_a) & set(cats_b))
    - Category_Score = shared × 20
```

### Phase 6: Bundle Selection - UPDATED

```
6.1 Generate Product Pairs
    - Use top 2000 products

6.2 Calculate Scores
    - Shared_Category_Score = shared_categories × 20
    - Recipe Score = from recipe_data
    - CoPurchase Score = from order analysis
    - Embedding Score = similarity

6.3 FINAL SCORING (NEW):
    Final = (Shared_Category × 0.30) + (Recipe × 0.30) + (CoPurchase × 0.30) + (Embedding × 0.10)

6.4 FILTER:
    - REMOVE: Same product (different sizes) - check name similarity
    - MINIMUM: Must have shared_categories ≥ 2

6.5 SORT:
    - By shared_categories descending
    - Then by Final Score
```

---

## Updated File Structure

Same as Prompt 2, but:
- `src/04_categories.py` - Now assigns MULTIPLE categories per product
- `src/06_bundle_selection.py` - Uses shared_category scoring

---

## Key Changes Summary

| Change | Before | After |
|--------|--------|-------|
| Focus | Similarity + Recipe | **Shared Categories** |
| Weights | 30/35/30 | **30/30/30/10** |
| Filter | Basic | **Block same product** |
| Min requirement | None | **2+ shared categories** |
| Sort | By final score | **By shared_categories** |

---

## Ready!

This fixes the issue. Use Prompt 1 + Prompt 2 + Prompt 3 together.
