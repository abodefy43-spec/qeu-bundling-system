"""Remove a product from all data files."""

import pandas as pd
from pathlib import Path

PRODUCT_NAME = "Rawahil Creamy Sella Basmati Rice - 5kg"
PRODUCT_ID = 2219

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"


def remove_from_orders():
    """Remove all order lines with this product."""
    path = DATA_DIR / "filtered_orders.pkl"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    
    df = pd.read_pickle(path)
    before = len(df)
    df = df[df["product_id"] != PRODUCT_ID]
    after = len(df)
    df.to_pickle(path)
    print(f"  filtered_orders.pkl: removed {before - after:,} lines (product_id={PRODUCT_ID})")


def remove_from_categories():
    """Remove product from category file."""
    path = DATA_DIR / "product_categories.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["product_id"] != PRODUCT_ID]
    after = len(df)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  product_categories.csv: removed {before - after:,} rows")


def remove_from_recipe_scores():
    """Remove product from recipe scores."""
    path = DATA_DIR / "product_recipe_scores.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["product_id"] != PRODUCT_ID]
    after = len(df)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  product_recipe_scores.csv: removed {before - after:,} rows")


def remove_from_copurchase():
    """Remove any copurchase pairs involving this product."""
    path = DATA_DIR / "copurchase_scores.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    
    df = pd.read_csv(path)
    before = len(df)
    df = df[(df["product_a"] != PRODUCT_ID) & (df["product_b"] != PRODUCT_ID)]
    after = len(df)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  copurchase_scores.csv: removed {before - after:,} rows")


def remove_from_pictures():
    """Remove product picture."""
    path = DATA_DIR / "product_pictures.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    
    df = pd.read_csv(path)
    before = len(df)
    df = df[df["product_id"] != PRODUCT_ID]
    after = len(df)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  product_pictures.csv: removed {before - after:,} rows")


def remove_from_top_bundles():
    """Remove bundles containing this product."""
    path = DATA_DIR / "top_bundles.csv"
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return
    
    df = pd.read_csv(path)
    before = len(df)
    df = df[(df["product_a"] != PRODUCT_ID) & (df["product_b"] != PRODUCT_ID)]
    after = len(df)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  top_bundles.csv: removed {before - after:,} bundles")


def remove_from_output_files():
    """Remove from final output files."""
    for filename in ["final_bundles.csv", "top_10_bundles.csv"]:
        path = OUTPUT_DIR / filename
        if not path.exists():
            print(f"  SKIP: {filename} not found")
            continue
        
        df = pd.read_csv(path)
        before = len(df)
        df = df[(df["product_a"] != PRODUCT_ID) & (df["product_b"] != PRODUCT_ID)]
        if "product_c" in df.columns:
            df = df[df["product_c"] != PRODUCT_ID]
        after = len(df)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  {filename}: removed {before - after:,} bundles")


def main():
    print(f"Removing product: {PRODUCT_NAME} (ID: {PRODUCT_ID})")
    print("=" * 50)
    
    print("\n[data/] files:")
    remove_from_orders()
    remove_from_categories()
    remove_from_recipe_scores()
    remove_from_copurchase()
    remove_from_pictures()
    remove_from_top_bundles()
    
    print("\n[output/] files:")
    remove_from_output_files()
    
    print("\n" + "=" * 50)
    print("Done! Run the pipeline again to regenerate bundles.")


if __name__ == "__main__":
    main()
