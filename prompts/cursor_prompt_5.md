# QEU Product Bundling System - Prompt 5: Bug Fixes & Improvements

## GOAL: Fix critical bugs and improve code quality

---

## CRITICAL ISSUES (FIX IMMEDIATELY)

### Issue 1: Non-Reproducible Results - Random Noise in Scores

**File:** `src/06_bundle_selection.py`  
**Lines:** 360-361

**Problem:** Random noise is added to final scores on every run, making output inconsistent and non-reproducible.

```python
# CURRENT CODE (WRONG):
rng = np.random.default_rng(int(time.time()))
noise = rng.uniform(0, 15.0, size=len(pairs))
pairs["final_score"] = pairs["final_score"] + noise
```

**Fix:** Remove the random noise injection, or use a fixed seed for reproducibility:

```python
# OPTION A: Remove noise entirely (RECOMMENDED)
# Simply delete lines 359-361

# OPTION B: Use fixed seed for reproducibility
rng = np.random.default_rng(42)  # Fixed seed for consistent results
noise = rng.uniform(0, 0.1, size=len(pairs))  # Much smaller noise
pairs["final_score"] = pairs["final_score"] + noise
```

**Why:** The noise (0-15 points) can drastically change bundle rankings between runs, making it impossible to validate improvements or compare results.

---

### Issue 2: Insecure Secret Key Fallback

**File:** `app.py`  
**Line:** 16

**Problem:** Flask app has a hardcoded fallback secret key.

```python
# CURRENT CODE (INSECURE):
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "qeu-presentation-secret")
```

**Fix:**

```python
# OPTION A: Require env var (RECOMMENDED for production)
import os
secret_key = os.environ.get("FLASK_SECRET_KEY")
if not secret_key:
    raise ValueError("FLASK_SECRET_KEY environment variable must be set")
app.config["SECRET_KEY"] = secret_key

# OPTION B: Generate random key for development only
import secrets
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)
```

---

### Issue 3: Duplicate Model Files Wasting Storage

**File:** `src/07_train_models.py`  
**Lines:** 297-305

**Problem:** Models are saved twice with different names, wasting ~10MB.

```python
# CURRENT CODE (WASTEFUL):
with (out / "free_item_model.pkl").open("wb") as f:
    pickle.dump(clf, f)
with (out / "discount_model.pkl").open("wb") as f:
    pickle.dump(reg, f)
# Backward-compatible names used by previous phase-8 code.
with (out / "bundle_free_item_model.pkl").open("wb") as f:
    pickle.dump(clf, f)
with (out / "bundle_discount_model.pkl").open("wb") as f:
    pickle.dump(reg, f)
```

**Fix:** Keep only one set, or check what Phase 8 actually uses:

```python
# OPTION A: Keep only standard names (RECOMMENDED)
with (out / "free_item_model.pkl").open("wb") as f:
    pickle.dump(clf, f)
with (out / "discount_model.pkl").open("wb") as f:
    pickle.dump(reg, f)
with (out / "preprocessor.pkl").open("wb") as f:
    pickle.dump(preprocessor, f)

# If Phase 8 needs old names, create symlinks or aliases instead
# OR update Phase 8 to use new names
```

---

## MINOR IMPROVEMENTS

### Issue 4: Missing .gitignore

**Create:** `.gitignore` in project root

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
.venv/

# Data files (keep in repo if needed)
*.pkl
*.npy
*.npz
*.csv
*.json

# Output
output/
*.log

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

---

### Issue 5: Similar Products in Bundles (Brand Differentiation)

**File:** `src/06_bundle_selection.py`

**Problem:** Bundle #6 pairs two different tomato paste brands together - may want to prevent this.

```python
# CURRENT OUTPUT:
# جورجينا معجون طماطم 800 جم + Saudi Tomato Paste
# Both are tomato paste from different brands
```

**Fix:** Add brand/family differentiation logic:

```python
def _is_same_brand(name_a: str, name_b: str) -> bool:
    """Check if products are from the same brand."""
    brand_keywords = [
        "almarai", "al salam", "sadia", "juhayna", "luna", 
        "nido", "saudia", "kdd", "nadec", "alba", "milar",
        # Add more brand names as needed
    ]
    name_a_lower = name_a.lower()
    name_b_lower = name_b.lower()
    
    for brand in brand_keywords:
        if brand in name_a_lower and brand in name_b_lower:
            return True
    return False


# In bundle selection, add filter:
# same_brand_mask = pairs.apply(
#     lambda r: _is_same_brand(r['product_a_name'], r['product_b_name']),
#     axis=1
# )
# filtered = filtered[~same_brand_mask]
```

---

### Issue 6: Output File Cleanup

**Problem:** Multiple similar output files causing confusion:
- `final_bundles.csv`
- `results_100_final.csv`
- `results_final.csv`
- `top_10_bundles.csv`

**Fix:** Document each file's purpose or consolidate:

```markdown
## Output Files Explained

| File | Description |
|------|-------------|
| `final_bundles.csv` | Main output - top 100 bundles for marketing |
| `top_10_bundles.csv` | Human-readable top 10 for presentations |
| `results_final.csv` | Full results with all scores |
| `results_100_final.csv` | Alternative format (legacy) |
```

Or consolidate into one file with a `rank` column.

---

### Issue 7: Data Folder Confusion

**Problem:** Raw data in `data first/` but processed in `data/`

**Fix:** Add a README in `data first/` explaining the difference:

```markdown
# Data First Directory

This folder contains ORIGINAL raw data files from the source system.

- `order_items.csv` - Original order items (630MB)
- `orders.csv` - Order details
- etc.

DO NOT EDIT THESE FILES - They are the source of truth.

Processed data goes in the parent `data/` directory.
```

---

## VERIFICATION CHECKLIST

After applying fixes, verify:

- [ ] Running pipeline produces same output twice (reproducibility)
- [ ] No duplicate model files in output/
- [ ] Flask app fails if FLASK_SECRET_KEY not set
- [ ] .gitignore excludes appropriate files
- [ ] Bundle #6 tomato paste issue addressed (optional)

---

## FILES TO MODIFY

| File | Changes |
|------|---------|
| `src/06_bundle_selection.py` | Remove/add back random noise |
| `app.py` | Fix secret key handling |
| `src/07_train_models.py` | Remove duplicate saves |
| `.gitignore` | Create new file |
| `output/README.md` | Document output files (optional) |
| `data first/README.md` | Explain raw data (optional) |

---

## TESTING

After fixes, run:

```bash
# Test reproducibility
cd src
python run_pipeline.py
cp ../output/final_bundles.csv /tmp/run1.csv
python run_pipeline.py
diff ../output/final_bundles.csv /tmp/run1.csv
# Should be identical or very similar (minor floating point diffs OK)

# Check for duplicates
ls -la output/*.pkl
# Should show only 3 files, not 5
```

---

## NOTES

- Issue #1 (random noise) is the MOST CRITICAL - it makes all results non-reproducible
- Issue #2 (secret key) is important for production deployment
- Issues #3-#7 are quality improvements

Priority order: 1 > 2 > 3 > 4 > 5 > 6 > 7
