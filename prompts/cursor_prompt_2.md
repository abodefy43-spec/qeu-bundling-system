# QEU Product Bundling System - Prompt 2: Implementation Fixes & Optimizations

## This is Prompt 2 - Builds on Prompt 1

This prompt fixes the issues identified in Prompt 1. Read Prompt 1 first, then apply these fixes.

---

## Issues Fixed in This Prompt:

| Issue | Fix |
|-------|-----|
| 1. Circular Training Data | Use actual discount patterns from order data |
| 2. Embedding Computation | Optimize for large product pairs |
| 3. Arabic Text Handling | Add preprocessing for Arabic product names |
| 4. Performance | Add batching and optimization notes |

---

## Phase 1 (from Prompt 1): Data Loading ✅
(Keep as is from Prompt 1)

---

## Phase 2: Embedding System - FIXED

```
2.1 Install Dependencies
    pip install sentence-transformers numpy pandas scikit-learn xgboost joblib

2.2 Arabic Text Preprocessing
    - Product names are in Arabic format: {"ar": "ليبتون شاي", "en": "Lipton Tea"}
    - Extract the 'en' (English) field for embedding
    - For Arabic-only: use multilingual model

2.3 Generate Embeddings - OPTIMIZED
    a) Get unique products (filter: min 20 orders)
    b) Use 'paraphrase-multilingual-MiniLM-L12-v2' model
    c) Batch processing: batch_size = 32
    d) Save in chunks to avoid memory issues
    
2.4 Calculate Similarity - FIXED
    a) For top N products by frequency only (top 2000)
    b) Use matrix multiplication for efficiency
    c) Store as sparse matrix
    d) Embedding Score = similarity × 100 (0-100)

2.5 Save
    - data/product_embeddings.npy
    - data/embedding_mapping.json
```

---

## Phase 3: Co-purchase Analysis ✅
(Keep as is from Prompt 1)

---

## Phase 4: Category System ✅
(Keep as is from Prompt 1)

---

## Phase 5: Recipe Scoring ✅
(Keep as is from Prompt 1)

---

## Phase 6: Bundle Selection - FIXED

```
6.1 Generate Product Pairs
    - Use TOP 2000 products only (by order frequency)
    - This = ~2M pairs (manageable)
    - Or sample: top 500 products = ~125K pairs

6.2 Calculate Scores
    (Same as Prompt 1)

6.3 Select Top Bundles
    (Same as Prompt 1)
```

---

## Phase 7: ML Model Training - FIXED

### FIXED: Proper Training Data Generation

Instead of circular training, we use **actual historical data**:

```
7.1 Generate Training Data from Orders
    a) Look at orders with MULTIPLE items
    b) Extract all product pairs from same order
    c) Calculate actual discount for each product:
       - discount = (base_price - effective_price) / base_price × 100
    
    Training Data:
    - product_a: higher priced product
    - product_b: lower priced product  
    - free_item_label: 0 (a is free) or 1 (b is free)
    - discount_amount: actual discount from order
    
    WHY: We use real purchases to learn what combinations work!

7.2 Features:
    - product_a_price
    - product_b_price
    - recipe_score_a (from recipe_data)
    - recipe_score_b
    - embedding_score (similarity)
    - category_a (encoded)
    - category_b (encoded)
    - saudi_importance_a
    - saudi_importance_b

7.3 Model Architecture
    Two models:
    
    a) Classifier: RandomForest/XGBoost
       - Input: features
       - Output: free_item (0 or 1)
       - Metric: Accuracy
    
    b) Regressor: RandomForest/XGBoost  
       - Input: features
       - Output: discount_percentage (0-100)
       - Metric: RMSE

7.4 Train
    - Split: 80% train, 20% test
    - Train both models
    - Evaluate with Accuracy, F1, RMSE, MAE

7.5 Save
    - output/free_item_model.pkl
    - output/discount_model.pkl
    - output/preprocessor.pkl
```

---

## Phase 8: Bundle Prediction ✅
(Same as Prompt 1)

---

## NEW Phase 9: Performance Optimization (Add This)

```
9.1 Embedding Optimization
    - Use FAISS for similarity search (faster)
    - Or: compute embeddings on-the-fly for prediction only
    
9.2 Batch Processing
    - Process products in batches of 1000
    - Save intermediate results
    
9.3 Memory Management
    - Use sparse matrices where possible
    - Delete unused DataFrames
```

---

## Updated File Structure

```
Bundling system/
├── data/
│   ├── order_items.csv
│   ├── recipe_data.json
│   ├── category_importance.csv
│   ├── filtered_orders.pkl
│   ├── product_embeddings.npy
│   ├── embedding_mapping.json
│   ├── copurchase_scores.csv
│   └── training_data.csv          # NEW: Training data
├── src/
│   ├── __init__.py
│   ├── 01_load_data.py
│   ├── 02_embeddings.py           # FIXED: Optimized
│   ├── 03_copurchase.py
│   ├── 04_categories.py
│   ├── 05_recipe_scoring.py
│   ├── 06_bundle_selection.py
│   ├── 07_train_models.py         # FIXED: Proper training
│   ├── 08_predict.py
│   ├── 09_optimize.py             # NEW: Performance
│   └── run_pipeline.py
├── output/
│   ├── final_bundles.csv
│   ├── free_item_model.pkl
│   ├── discount_model.pkl
│   └── preprocessor.pkl
├── README.md
├── cursor_prompt_1.md             # Original prompt
└── cursor_prompt_2.md             # This file
```

---

## Key Fixes Summary

| Fix | Description |
|-----|-------------|
| Training Data | Use actual order discounts, not circular self-scoring |
| Embedding Products | Limit to top 2000 products by frequency |
| Arabic Handling | Extract English names, use multilingual model |
| Performance | Add batching, FAISS option, memory management |
| Two Models | Separate classifier + regressor |

---

## Ready for Implementation!

This prompt fixes all the issues. Use this AFTER Prompt 1.
