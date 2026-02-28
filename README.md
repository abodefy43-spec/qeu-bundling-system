# QEU Product Bundling System

## Quick Start (Easiest)

Windows users can double-click `easy_run.bat` and choose:
- `1` Quick refresh (Phases 6-9) + open dashboard
- `2` Full rebuild (Phases 1-9) + open dashboard
- `3` Open dashboard only

Dashboard URL: `http://127.0.0.1:5000`

---

## Data Note (Important)

This repo excludes very large raw/derived files (GitHub size limits).

If you clone this repo, place your local dataset files back into:
- `data first/` (raw CSV files)
- `data/` (cached artifacts used by fast runs)

Then run `easy_run.bat`.

---

## Overview

This system creates intelligent "Buy X, Get Y Free/Cheap" bundle offers for QEU customers using:
- **Recipe Intelligence** - Products from Ramadan & Saudi recipes
- **Purchase Intelligence** - Products frequently bought together
- **Embedding System** - Sentence Transformers for product similarity
- **Shared Categories** - Products that share multiple categories
- **ML Model** - Predicts free item + discount amount

---

## Project Structure

```
Bundling system/
├── data/
│   ├── order_items.csv              # Original order data (1.46M rows)
│   ├── orders.csv                   # Order details
│   ├── recipe_data.json             # Scraped recipe data (120+ ingredients)
│   ├── category_importance.csv      # Category analysis
│   ├── filtered_orders.pkl          # Preprocessed data
│   ├── product_embeddings.npy       # Generated embeddings
│   ├── embedding_mapping.json       # Product→index mapping
│   └── copurchase_scores.csv        # Co-purchase analysis
├── src/
│   ├── __init__.py
│   ├── 01_load_data.py             # Data loading
│   ├── 02_embeddings.py            # Sentence Transformers
│   ├── 03_copurchase.py           # Co-purchase analysis
│   ├── 04_categories.py           # Category assignment
│   ├── 05_recipe_scoring.py      # Recipe scoring
│   ├── 06_bundle_selection.py     # Bundle selection + scoring
│   ├── 07_train_models.py         # ML model training
│   ├── 08_predict.py              # Prediction function
│   └── run_pipeline.py            # Run all phases
├── output/
│   ├── final_bundles.csv         # Final bundle offers
│   ├── free_item_model.pkl       # Free item classifier
│   ├── discount_model.pkl         # Discount regressor
│   └── preprocessor.pkl           # Feature encoder
├── cursor_prompt_1.md            # Overview + phases
├── cursor_prompt_2.md            # Fixes + optimizations
├── cursor_prompt_3.md            # Shared categories fix
├── cursor_prompt_4.md            # Same product blocking fix
├── README.md                     # This file
└── README_Cursor_Fix.md         # Fix documentation
```

---

## Data Sources

### Order Data (CSV)
- **Date Range**: Jan 8 - Feb 25, 2026 (~1.46M order items)
- **Use**: Last 30 days (Jan 26 - Feb 25, 2026) for analysis
- **Columns**: product_id, product_name, unit_price, effective_price, order_id

### Recipe Data (JSON)
- **Sources**: Ramadan 2025/2026 recipes, Saudi Arabia cuisine
- **Contents**: 120+ ingredients, 150+ recipes
- **Key Dishes**: Kabsa, Jareesh, Saleeg, Matazeez, Samboosa, Vimto, etc.
- **Saudi Consumption Data**: Rice (52kg/year), Dates (36kg), Onions (20.5kg), Tomatoes (19.56kg)

---

## How It Works

### Phase 1: Data Loading
- Load order data + recipe data
- Filter to last 30 days
- Preprocess product names

### Phase 2: Embedding System
- Use Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`)
- Generate embeddings for all products
- Calculate cosine similarity between product pairs

### Phase 3: Co-purchase Analysis
- Find products bought together in same order
- Calculate co-purchase frequency scores

### Phase 4: Category System
- Assign multiple categories to each product:
  - From recipe_data (saudi_ramadan_dishes, categories, ingredients)
  - From order data (purchase frequency)
- Example: Tomatoes → ["vegetables", "ramadan", "salads", "sauces", "soups", "cooking"]

### Phase 5: Shared Categories Scoring
- Count shared categories between product pairs
- Category Score = shared_categories × 20

### Phase 6: Bundle Selection

**CRITICAL RULES:**
1. ❌ **BLOCK**: Same product (different sizes) - MUST REMOVE
2. ✅ Must have ≥2 shared categories
3. ✅ Sort by shared categories

**Scoring Formula (30/30/30/10):**
```
Final Score = (Shared_Category × 0.30) + (Recipe × 0.30) + (CoPurchase × 0.30) + (Embedding × 0.10)
```

### Phase 7: ML Model Training
- **Classifier**: Predicts which item is free (product_a or product_b)
- **Regressor**: Predicts discount percentage
- **Features**: prices, recipe scores, purchase scores, embedding scores, categories
- **Model**: RandomForest or XGBoost

### Phase 8: Bundle Prediction
- Input: product_a, product_b
- Output: free_item, discount_amount

---

## Expected Output

### final_bundles.csv
| rank | product_a | product_b | price_a_sar | price_b_sar | free_item | discount_amount | shared_categories | final_score |
|------|-----------|-----------|-------------|-------------|-----------|-----------------|------------------|-------------|
| 1 | Tomato | Cucumber | 5.0 | 3.0 | product_b | 40% | 5 | 85 |
| 2 | Rice | Chicken | 25.0 | 15.0 | product_b | 37.5% | 4 | 80 |
| 3 | Bread | Cheese | 3.0 | 8.0 | product_b | 62.5% | 3 | 75 |

**IMPORTANT:** 
- ❌ NO same products (Tuna + Tuna = WRONG)
- ❌ NO products with same core name
- ✅ ONLY different products that share categories

---

## Common Issues & Fixes

### Issue 1: Same Products Bundled
**Problem:** Tuna + Tuna, Tea + Tea appearing  
**Fix:** See cursor_prompt_4.md - Same Product Detection

### Issue 2: Random Combos
**Problem:** Products with no shared categories  
**Fix:** See cursor_prompt_3.md - Shared Categories Priority

### Issue 3: Circular Training Data
**Problem:** Model training on self-generated labels  
**Fix:** See cursor_prompt_2.md - Use actual order discounts

---

## Key Formulas

### Same Product Detection
```python
def is_same_product(a, b):
    # Remove size words
    # Check if core names overlap
    # If >70% similar → SAME → BLOCK
```

### Shared Categories
```python
shared = len(set(cats_a) & set(cats_b))
category_score = shared × 20
```

### Final Score
```python
final = (shared_cat × 0.30) + (recipe × 0.30) + (copurchase × 0.30) + (embedding × 0.10)
```

---

## Usage

### Run Full Pipeline
```bash
cd src
python run_pipeline.py
```

### Run Individual Phases
```bash
python src/01_load_data.py
python src/02_embeddings.py
python src/03_copurchase.py
# ... etc
```

---

## Prompt History

| Prompt | Version | Description |
|--------|---------|-------------|
| cursor_prompt_1.md | v1.0 | Original overview + phases |
| cursor_prompt_2.md | v1.1 | Fixes: training data, embeddings, performance |
| cursor_prompt_3.md | v1.2 | Shared categories priority |
| cursor_prompt_4.md | v1.3 | Same product blocking (CRITICAL) |

---

## Notes

- Product names are in Arabic - uses multilingual embedding model
- No 2025 data in CSV (only Jan-Feb 2026)
- Ramadan 2026 data - prioritize seasonal items
- All scoring uses 0-100 scale
- Weights: 30/30/30/10 (Shared/Recipe/Copurchase/Embedding)

---

## Success Criteria

- ✅ 15-20 bundle offers
- ✅ NO same products in output
- ✅ Products share ≥2 categories
- ✅ ML model predicts free_item + discount
- ✅ Ready for marketing team
