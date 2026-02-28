# QEU Product Bundling System

## Executive Summary

This system creates intelligent "Buy X, Get Y Free/Cheap" bundle offers for QEU customers. It combines:

1. **Recipe Intelligence** - Products frequently used together in cooking (Ramadan recipes, Middle Eastern dishes, general staples)
2. **Purchase Intelligence** - Products frequently bought together in actual orders
3. **ML Pricing Model** - AI determines optimal bundle pricing

The free item is always the cheaper one, with no price cap (ML decides pricing).

---

## Data Sources

### Order Data (CSV Files)
- **order_items.csv** - Contains ~1.46M order items from Jan 8 - Feb 25, 2026
- **orders.csv** - Contains ~66K orders with customer info
- **Date Range**: Use last 30 days (Jan 26 - Feb 25, 2026) as primary analysis period

### Internet Data (To Scrape)
- Ramadan 2025 recipes (from cooking websites)
- Ramadan 2026 recipes (current Ramadan)
- Middle Eastern pantry staples and common ingredients
- General grocery staples (year-round items)

---

## Project Structure

```
Bundling system/
├── data/
│   ├── order_items.csv       # Your QEU order data
│   ├── orders.csv            # Your QEU orders
│   ├── recipe_data.json      # Scraped recipe ingredients
│   └── category_importance.csv # Analyzed category scores
├── src/
│   ├── scraper.py            # Recipe data scraping
│   ├── analyzer.py           # Order data analysis
│   ├── scorer.py             # Hybrid scoring engine
│   ├── pricing_model.py     # ML pricing model
│   └── generator.py         # Bundle offer generator
├── output/
│   ├── bundles.csv           # Final 15-20 bundle offers
│   └── pricing_model.pkl     # Saved ML model
├── README.md                 # This file
└── cursor_prompt.md         # Prompt for Cursor AI
```

---

## Phase 1: Recipe Intelligence (Internet Scraping)

### Sources to Scrape

**Ramadan Recipes:**
- Dates, Harira soup, Samboosa, Dolma
- Biryani, Kabsa, Jareesh, Saleeg, Matazeez
- Fattoush, Mezze (hummus, mutabbal, tabbouleh, falafel)
- Qatayef, Kunafa, Baklava
- Drinks: Vimto, Qamar Al Din, Jallab, Laban

**Middle Eastern Staples (Year-Round):**
- Grains: Rice (basmati), Bulgur, Couscous, Quinoa, Pasta, Flour
- Legumes: Lentils (brown, red, green), Chickpeas, Beans
- Oils: Olive oil, Vegetable oil, Ghee
- Spices: Cumin, Coriander, Turmeric, Cinnamon, Cardamom, Baharat, Sumac, Za'atar
- Dairy: Yogurt, Labneh, Cheese (white cheese, cream cheese)
- Proteins: Chicken, Beef, Lamb, Fish, Eggs
- Vegetables: Onions, Tomatoes, Potatoes, Carrots, Garlic
- Fruits: Lemons, Oranges, Bananas, Apples

**General Grocery Staples:**
- Bread, Tea, Coffee, Sugar, Honey
- Nuts (almonds, walnuts, pistachios)
- Canned goods, Frozen foods
- Cleaning supplies, Personal care

### Scoring System
- Each ingredient maps to product categories
- Recipe Score (0-100) = number of recipe categories a product appears in
- Higher score = more recipe-relevant = higher bundle priority

---

## Phase 2: Purchase Intelligence (Order Data Analysis)

### Analysis Steps
1. Load order_items.csv (filter to last 30 days: Jan 26 - Feb 25, 2026)
2. Group products by order_id to find co-purchased items
3. Calculate co-purchase frequency for product pairs
4. Identify top 50-100 product pairs by frequency

### Purchase Score (0-100)
- Normalize co-purchase frequency to 0-100 scale
- Products bought together more often = higher score

---

## Phase 3: Hybrid Scoring Engine

### Formula
```
Final Score = (Recipe Score × 0.6) + (Purchase Score × 0.4)
```

- **Recipe weight (60%)** - Captures seasonal/recipe relevance
- **Purchase weight (40%)** - Captures actual buying behavior

### Scoring Scale: 0-100

| Score Range | Priority | Description |
|-------------|----------|-------------|
| 80-100 | Highest | Both recipe + purchase: MUST bundle |
| 60-79 | High | Strong in one, good in other |
| 40-59 | Medium | Good potential, test bundles |
| 20-39 | Low | Consider for promotions |
| 0-19 | Skip | Not recommended |

---

## Phase 4: Bundle Generation

### Rules
1. **Free Item = Cheaper Item** - Always
2. **No Price Cap** - ML model decides optimal price
3. **Bundle Price** = Price of expensive item only
4. **Target**: 15-20 final bundles

### Output Columns (bundles.csv)
- bundle_id
- product_a (expensive)
- product_b (free/cheap)
- product_a_price
- product_b_price
- bundle_price
- savings_value
- recipe_score
- purchase_score
- final_score
- recommendation

---

## Phase 5: ML Pricing Model

### Features
- Product A price
- Product B price
- Recipe score
- Purchase score
- Category (food, beverage, household, etc.)
- Seasonal factor (Ramadan, summer, etc.)

### Model Options
- Linear Regression (simple, interpretable)
- Random Forest (handles non-linear relationships)
- Gradient Boosting (XGBoost/LightGBM)

### Training Data
- Use existing discount patterns from order_items.csv
- Target: Predict optimal bundle price that maximizes perceived value

---

## Category Importance Analysis

Create `category_importance.csv` with these columns:

| Category | Purchase_Frequency | Recipe_Importance | Seasonal_Weight | Final_Score |
|----------|-------------------|-------------------|-----------------|-------------|
| Rice | High | Very High | Ramadan | 95 |
| Dates | Very High | Very High | Ramadan | 98 |
| Yogurt | High | High | Medium | 75 |
| ... | ... | ... | ... | ... |

### Scoring Criteria
- **Purchase Frequency**: Based on order data analysis
- **Recipe Importance**: Based on scraped recipe data (Ramadan 2025 + 2026)
- **Seasonal Weight**: Ramadan = highest, normal months = baseline
- **Final Score**: Weighted average for category prioritization

---

## Cursor AI Prompt

When giving this project to Cursor AI, include:

1. **This README** - Full context
2. **recipe_data.json** - Scraped ingredient data (you'll need to scrape)
3. **order_items.csv** - Your order data
4. **cursor_prompt.md** - Specific instructions

---

## Expected Timeline

- **Day 1**: Scrape recipe data, analyze order data, generate category importance CSV
- **Day 2-3**: Build hybrid scoring engine, train ML pricing model
- **Day 4**: Generate final bundles, export CSV + ML model
- **Day 5**: Team review and refinement

---

## Success Metrics

- 15-20 ready-to-use bundle offers
- Each bundle includes: products, regular prices, bundle price, savings value
- ML model achieves reasonable pricing predictions
- Marketing team can launch within 1 day of receiving output

---

## Notes

- Ramadan 2025 recipe data from internet (scraping required)
- Order data covers Jan 8 - Feb 25, 2026 only (no 2025 data available)
- Use last 30 days (Jan 26 - Feb 25, 2026) for most relevant analysis
- All scoring uses 0-100 scale (not 0-10)
