# QEU Product Bundling System - Prompt 4: OPTIMIZED for ~10 Hours

## GOAL: Get really good results in ~10 hours max

---

## THE PROBLEM

Previous results had SAME PRODUCTS bundled together. We need:
1. NO same products
2. High-quality shared categories
3. Really good bundle predictions

---

## TIME OPTIMIZATION STRATEGY

### Target: ~10 hours max

| Phase | Time | Optimization |
|-------|------|--------------|
| Embeddings | 2-3 hours | Batch + top 4000 products |
| Similarity | 2-3 hours | Matrix multiplication |
| Co-purchase | 1-2 hours | Vectorized operations |
| Categories | 1 hour | Parallel processing |
| ML Training | 1 hour | Standard |
| Buffer | 1 hour | For issues |

---

## PHASE 1: Product Selection (CRITICAL)

### Don't use all products! Use TOP products:

```python
# Select top 4000 products by order frequency
# This reduces pairs from 50M to 8M (manageable)

top_products = get_top_products_by_frequency(n=4000)
print(f"Using {len(top_products)} products")
```

### Why 4000?
- 4000 products = 8M pairs = ~2-3 hours for similarity
- Enough variety for good bundles
- Not too slow

---

## PHASE 2: Embedding Optimization

### Use efficient model + batching:

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# Use smaller, faster model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Batch processing
batch_size = 64  # Process 64 products at a time

embeddings = model.encode(
    product_names, 
    batch_size=batch_size,
    show_progress_bar=True,
    convert_to_numpy=True
)

# Save embeddings
np.save('data/product_embeddings.npy', embeddings)
```

### Time: ~2 hours for 4000 products

---

## PHASE 3: Similarity Calculation (KEY OPTIMIZATION)

### Use matrix multiplication (FAST):

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Normalize embeddings
embeddings_normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Compute similarity matrix (FAST - one operation)
similarity_matrix = cosine_similarity(embeddings_normalized)

# Extract upper triangle (unique pairs)
upper_triangle = np.triu(similarity_matrix, k=1)

# Find top similar pairs
# Time: ~10-20 minutes instead of hours!
```

### Time: ~20 minutes!

---

## PHASE 4: Co-purchase Analysis

### Use efficient pandas operations:

```python
import pandas as pd
from collections import Counter
from itertools import combinations

# Get orders with 2-10 items only (too many = noise)
df_filtered = df[df.groupby('order_id')['product_id'].transform('count').between(2, 10)]

# Group by order
orders = df_filtered.groupby('order_id')['product_id'].apply(list)

# Count co-occurrences
counter = Counter()
for order_products in orders:
    # Sort to ensure consistent pairs
    pair = tuple(sorted(order_products))
    counter[pair] += 1

# Convert to DataFrame
copurchase_df = pd.DataFrame([
    {'product_a': p[0], 'product_b': p[1], 'count': c} 
    for p, c in counter.most_common()
])

# Normalize to 0-100
copurchase_df['score'] = copurchase_df['count'] / copurchase_df['count'].max() * 100
```

### Time: ~1-2 hours for 1.4M orders

---

## PHASE 5: Category Assignment

### Multi-category per product:

```python
# Each product gets MULTIPLE categories
# From recipe_data.json:
# - saudi_ramadan_dishes
# - categories (grains, protein, dairy, etc.)
# - ingredients
# - saudi_importance

def get_categories(product_name):
    categories = []
    product_lower = product_name.lower()
    
    # Check each category keyword
    for category, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in product_lower:
                categories.append(category)
                break
    
    return categories

# Assign to all products
product_categories = {p: get_categories(p) for p in top_products}
```

### Time: ~30 minutes

---

## PHASE 6: Same Product Detection (CRITICAL!)

### The FIX:

```python
def is_same_product(product_a, product_b):
    """Returns True if products are THE SAME (different sizes/brands)"""
    
    # Remove size/quantity words
    size_words = ['g', 'kg', 'ml', 'l', 'pack', 'pcs', 'piece', 'حبة', 'حبات', 'كيس', 'كجم', 'جم', 'مل', 'لتر']
    
    name_a = str(product_a).lower()
    name_b = str(product_b).lower()
    
    for word in size_words:
        name_a = name_a.replace(word, '')
        name_b = name_b.replace(word, '')
    
    # Remove numbers
    name_a = ''.join(c for c in name_a if not c.isdigit())
    name_b = ''.join(c for c in name_b if not c.isdigit())
    
    # Clean
    name_a = ' '.join(name_a.split())
    name_b = ' '.join(name_b.split())
    
    # Check exact match after cleaning
    if name_a == name_b:
        return True
    
    # Check if one contains the other
    if len(name_a) > 3 and len(name_b) > 3:
        if name_a in name_b or name_b in name_a:
            return True
    
    # Check word overlap
    words_a = set(name_a.split())
    words_b = set(name_b.split())
    if words_a and words_b:
        overlap = len(words_a & words_b) / len(words_a | words_b)
        if overlap > 0.7:  # 70% same words
            return True
    
    return False

# APPLY THIS FILTER FIRST!
print("Filtering out same products...")
different_products = []
for i, (a, b) in enumerate(all_pairs):
    if not is_same_product(a, b):
        different_products.append((a, b))
    if i % 100000 == 0:
        print(f"Processed {i} pairs...")

print(f"Removed {len(all_pairs) - len(different_products)} same-product pairs")
```

---

## PHASE 7: Shared Categories Scoring

```python
def calculate_shared_categories(product_a, product_b):
    cats_a = set(product_categories.get(product_a, []))
    cats_b = set(product_categories.get(product_b, []))
    shared = len(cats_a & cats_b)
    return shared * 20  # 5 shared = 100

# Calculate for all pairs
for pair in different_products:
    shared = calculate_shared_categories(pair[0], pair[1])
    if shared >= 40:  # At least 2 shared categories
        bundle_scores.append({
            'product_a': pair[0],
            'product_b': pair[1],
            'shared_categories': shared
        })
```

---

## PHASE 8: Final Scoring

### 30/30/30/10 Formula:

```python
def calculate_final_score(shared_cat_score, recipe_score, copurchase_score, embedding_score):
    return (
        shared_cat_score * 0.30 +
        recipe_score * 0.30 +
        copurchase_score * 0.30 +
        embedding_score * 0.10
    )

# Sort by final score
bundles.sort(key=lambda x: x['final_score'], reverse=True)

# Take top 100
final_bundles = bundles[:100]
```

---

## PHASE 9: FINAL VALIDATION (MUST DO!)

```python
# VALIDATE - Make absolutely sure no same products

print("=== FINAL VALIDATION ===")
errors = 0
for idx, bundle in enumerate(final_bundles):
    if is_same_product(bundle['product_a'], bundle['product_b']):
        print(f"ERROR row {idx}: {bundle['product_a']} + {bundle['product_b']}")
        errors += 1
        # REMOVE THIS BUNDLE!
        final_bundles.remove(bundle)

if errors == 0:
    print("✅ PASSED: No same products found!")
else:
    print(f"❌ FAILED: Found {errors} same-product bundles")
```

---

## OUTPUT FORMAT

Save as `output/final_bundles.csv`:

```csv
rank,product_a,product_b,price_a_sar,price_b_sar,free_item,discount_amount,shared_categories,final_score
1,tomato product,cucumber product,5.0,3.0,product_b,40,5,85
2,rice product,chicken product,25.0,15.0,product_b,37.5,4,80
```

---

## TIME ESTIMATE

| Phase | Estimated Time |
|-------|---------------|
| Product selection (top 4000) | 10 min |
| Embeddings generation | 2 hours |
| Similarity calculation | 20 min |
| Co-purchase analysis | 1.5 hours |
| Category assignment | 30 min |
| Same product filter | 30 min |
| Scoring + sorting | 20 min |
| ML model training | 1 hour |
| Validation + save | 10 min |
| **TOTAL** | **~7 hours** |

Plus buffer for issues = **~10 hours max**

---

## EXPECTED RESULTS

After running, you should get:
- ✅ 100 high-quality bundles
- ✅ NO same products
- ✅ All bundles have ≥2 shared categories
- ✅ Sorted by shared categories + final score

---

## IF TAKING TOO LONG

Reduce product count:
- 5000 products = 12M pairs = ~6 hours similarity
- 3000 products = 4.5M pairs = ~3 hours similarity

```python
# Adjust this line:
top_products = get_top_products_by_frequency(n=3000)  # Faster
# or
top_products = get_top_products_by_frequency(n=5000)  # More thorough
```

---

## READY TO RUN

This optimization should give you really good results in ~10 hours!
