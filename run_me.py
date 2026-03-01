"""Interactive bundle review script.

Run this to review bundles and provide feedback on their quality.
"""

import csv
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLES_PATH = PROJECT_ROOT / "output" / "final_bundles.csv"
FEEDBACK_PATH = PROJECT_ROOT / "data" / "bundle_feedback.csv"


def load_bundles():
    if not BUNDLES_PATH.exists():
        print(f"ERROR: No bundles found at {BUNDLES_PATH}")
        print("Run the pipeline first: python src/run_pipeline.py")
        return []
    
    df = pd.read_csv(BUNDLES_PATH)
    print(f"Loaded {len(df)} bundles from {BUNDLES_PATH.name}")
    return df


def load_existing_feedback():
    if not FEEDBACK_PATH.exists():
        return []
    
    with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def save_feedback(feedback_list):
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if not feedback_list:
        return
    
    fieldnames = list(feedback_list[0].keys())
    
    write_header = not FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(feedback_list)


def display_bundle(idx, row):
    print("\n" + "=" * 60)
    print(f"BUNDLE #{idx + 1}")
    print("=" * 60)
    
    product_a = row.get("product_a_name", "Unknown")
    product_b = row.get("product_b_name", "Unknown")
    price_a = row.get("product_a_price", 0)
    price_b = row.get("product_b_price", 0)
    free_item = row.get("free_product", "product_a")
    discount_a = row.get("discount_pred_a", 0)
    discount_b = row.get("discount_pred_b", 0)
    
    print(f"\nProduct A: {product_a}")
    print(f"  Price: SAR {price_a:.2f}  |  Discount: {discount_a:.1f}%")
    if free_item == "product_a":
        print(f"  >>> FREE <<<")
    
    print(f"\nProduct B: {product_b}")
    print(f"  Price: SAR {price_b:.2f}  |  Discount: {discount_b:.1f}%")
    if free_item == "product_b":
        print(f"  >>> FREE <<<")
    
    if row.get("product_c_name"):
        product_c = row.get("product_c_name", "")
        price_c = row.get("product_c_price", 0)
        print(f"\nProduct C: {product_c}")
        print(f"  Price: SAR {price_c:.2f}")
        if row.get("is_triple_bundle"):
            print("  >>> 3-ITEM BUNDLE <<<")
    
    print(f"\nCategories: {row.get('category_a', '')} + {row.get('category_b', '')}")
    print(f"Shared categories: {row.get('shared_categories_count', 'N/A')}")
    print(f"Final score: {row.get('final_score', 'N/A'):.2f}")


def get_user_response():
    print("\n" + "-" * 40)
    print("Is this a GOOD bundle? (y/n)")
    print("  y = Yes, good bundle")
    print("  n = No, bad bundle") 
    print("  s = Skip this bundle")
    print("  q = Quit review")
    print("-" * 40)
    
    while True:
        response = input("\nYour choice (y/n/s/q): ").strip().lower()
        
        if response in ['y', 'n', 's', 'q']:
            return response
        
        print("Invalid input. Please enter y, n, s, or q")


def get_reason():
    print("\nWhy is this bundle bad? (select one or type custom reason)")
    print("  1 = Too similar products (same brand/size)")
    print("  2 = Wrong category combination")
    print("  3 = Not complementary items")
    print("  4 = Prices don't make sense")
    print("  5 = Doesn't make sense for Saudi cuisine")
    print("  6 = Other (type custom reason)")
    
    while True:
        response = input("\nReason (1-6): ").strip()
        
        reasons_map = {
            "1": "too_similar",
            "2": "wrong_category", 
            "3": "not_complementary",
            "4": "bad_pricing",
            "5": "not_suitable",
            "6": "other"
        }
        
        if response in reasons_map:
            if response == "6":
                custom = input("Enter your reason: ").strip()
                return custom if custom else "other"
            return reasons_map[response]
        
        print("Invalid input. Please enter 1-6")


def show_statistics(feedback_list):
    if not feedback_list:
        print("\nNo feedback recorded yet.")
        return
    
    df = pd.DataFrame(feedback_list)
    
    total = len(df)
    good = len(df[df['is_good'] == 'true'])
    bad = total - good
    
    print("\n" + "=" * 40)
    print("REVIEW STATISTICS")
    print("=" * 40)
    print(f"Total reviewed: {total}")
    print(f"Good bundles: {good} ({good/total*100:.1f}%)")
    print(f"Bad bundles: {bad} ({bad/total*100:.1f}%)")
    
    if bad > 0:
        print("\nReasons for bad bundles:")
        reasons = df[df['is_good'] == 'false']['reason'].value_counts()
        for reason, count in reasons.items():
            print(f"  - {reason}: {count}")
    
    print(f"\nFeedback saved to: {FEEDBACK_PATH}")


def main():
    print("=" * 60)
    print("BUNDLE REVIEW TOOL")
    print("=" * 60)
    
    bundles = load_bundles()
    if not bundles:
        return
    
    existing = load_existing_feedback()
    reviewed_indices = set(int(f['bundle_index']) for f in existing)
    
    print(f"\nPreviously reviewed: {len(reviewed_indices)} bundles")
    
    feedback_list = []
    
    for idx, row in bundles.iterrows():
        if idx in reviewed_indices:
            continue
        
        display_bundle(idx, row)
        response = get_user_response()
        
        if response == 'q':
            print("\nQuitting review...")
            break
        
        if response == 's':
            print("Skipped.")
            continue
        
        is_good = (response == 'y')
        
        if not is_good:
            reason = get_reason()
        else:
            reason = ""
        
        feedback_list.append({
            'bundle_index': idx,
            'product_a': row.get('product_a_name', ''),
            'product_b': row.get('product_b_name', ''),
            'product_a_id': row.get('product_a', ''),
            'product_b_id': row.get('product_b', ''),
            'is_good': 'true' if is_good else 'false',
            'reason': reason,
            'final_score': str(row.get('final_score', '')),
            'shared_categories': str(row.get('shared_categories', '')),
        })
        
        if is_good:
            print("\n>>> Marked as GOOD!")
        else:
            print(f"\n>>> Marked as BAD: {reason}")
    
    if feedback_list:
        save_feedback(feedback_list)
        print(f"\nSaved {len(feedback_list)} new feedback entries.")
    
    show_statistics(existing + feedback_list)


if __name__ == "__main__":
    main()
