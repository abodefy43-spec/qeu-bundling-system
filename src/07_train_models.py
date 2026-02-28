"""Phase 7: ML model training — use actual historical order discount patterns."""

from __future__ import annotations

import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import OrdinalEncoder


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _output_dir() -> Path:
    d = Path(__file__).resolve().parents[1] / "output"
    d.mkdir(exist_ok=True)
    return d


IMPORTANCE_ORDER = ["low", "medium", "high", "very_high", "critical"]

NUMERIC_FEATURES = [
    "product_a_price",
    "product_b_price",
    "recipe_score_a",
    "recipe_score_b",
    "purchase_score",
    "embedding_score",
    "shared_categories_count",
    "shared_category_score",
    "category_match",
]
CATEGORICAL_FEATURES = [
    "category_a",
    "category_b",
    "importance_a",
    "importance_b",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MAX_TRAINING_PAIRS = 250000


def _prepare_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    """Build training data from real multi-item orders and actual discounts."""
    base = data_dir or _data_dir()
    orders = pd.read_pickle(base / "filtered_orders.pkl").copy()
    recipe = pd.read_csv(base / "product_recipe_scores.csv")
    categories = pd.read_csv(base / "product_categories.csv")
    top_bundles = pd.read_csv(base / "top_bundles.csv")

    for col in ["unit_price", "base_price", "effective_price"]:
        orders[col] = pd.to_numeric(orders[col], errors="coerce")
    orders["line_discount_pct"] = np.where(
        orders["base_price"] > 0,
        (orders["base_price"] - orders["effective_price"]) / orders["base_price"] * 100.0,
        0.0,
    )
    orders["line_discount_pct"] = orders["line_discount_pct"].clip(0, 100).fillna(0)

    recipe_lookup = recipe.set_index("product_id")[["recipe_score", "saudi_importance"]].to_dict("index")
    cat_lookup = categories.set_index("product_id")["category"].to_dict()
    pair_feature_lookup: dict[tuple[int, int], dict[str, float]] = {}
    for row in top_bundles.itertuples(index=False):
        a_id = int(getattr(row, "product_a"))
        b_id = int(getattr(row, "product_b"))
        features = {
            "embedding_score": float(getattr(row, "embedding_score", 0.0)),
            "purchase_score": float(getattr(row, "purchase_score", 0.0)),
            "shared_categories_count": float(getattr(row, "shared_categories_count", 0.0)),
            "shared_category_score": float(getattr(row, "shared_category_score", 0.0)),
        }
        pair_feature_lookup[(a_id, b_id)] = features
        pair_feature_lookup[(b_id, a_id)] = features

    rows: list[dict] = []
    grouped = orders.groupby("order_id")
    processed_orders = 0
    for _, grp in grouped:
        grp = grp.dropna(subset=["product_id", "unit_price"]).copy()
        if grp["product_id"].nunique() < 2:
            continue
        # Collapse repeated lines of same product within an order.
        prod = (
            grp.groupby("product_id", as_index=False)
            .agg(
                unit_price=("unit_price", "median"),
                line_discount_pct=("line_discount_pct", "mean"),
            )
        )
        if len(prod) > 20:
            continue
        tuples = list(prod.itertuples(index=False))
        for a_row, b_row in combinations(tuples, 2):
            a_id = int(a_row.product_id)
            b_id = int(b_row.product_id)
            a_price = float(a_row.unit_price)
            b_price = float(b_row.unit_price)
            a_disc = float(a_row.line_discount_pct)
            b_disc = float(b_row.line_discount_pct)

            # product_a should be higher priced, product_b lower priced.
            if b_price > a_price:
                a_id, b_id = b_id, a_id
                a_price, b_price = b_price, a_price
                a_disc, b_disc = b_disc, a_disc

            # Infer free-item label from observed discount intensity.
            if abs(a_disc - b_disc) < 1e-6:
                free_label = int(b_price <= a_price)
            else:
                free_label = int(b_disc >= a_disc)
            discount_amount = max(a_disc, b_disc)

            rec_a = recipe_lookup.get(a_id, {})
            rec_b = recipe_lookup.get(b_id, {})
            category_a = str(cat_lookup.get(a_id, "other"))
            category_b = str(cat_lookup.get(b_id, "other"))
            pair_features = pair_feature_lookup.get((a_id, b_id), {})
            emb_score = float(pair_features.get("embedding_score", 0.0))
            purchase_score = float(pair_features.get("purchase_score", 0.0))
            shared_count = int(pair_features.get("shared_categories_count", 0.0))
            shared_score = float(pair_features.get("shared_category_score", min(shared_count * 20, 100)))

            rows.append(
                {
                    "product_a": a_id,
                    "product_b": b_id,
                    "product_a_price": a_price,
                    "product_b_price": b_price,
                    "recipe_score_a": float(rec_a.get("recipe_score", 0.0)),
                    "recipe_score_b": float(rec_b.get("recipe_score", 0.0)),
                    "embedding_score": float(emb_score),
                    "purchase_score": float(purchase_score),
                    "shared_categories_count": int(shared_count),
                    "shared_category_score": float(np.clip(shared_score, 0.0, 100.0)),
                    "category_a": category_a,
                    "category_b": category_b,
                    "importance_a": str(rec_a.get("saudi_importance", "low")),
                    "importance_b": str(rec_b.get("saudi_importance", "low")),
                    "category_match": int(category_a == category_b),
                    "free_item": int(free_label),
                    "discount_amount": float(np.clip(discount_amount, 0, 100)),
                    "discount_a": float(np.clip(a_disc, 0, 100)),
                    "discount_b": float(np.clip(b_disc, 0, 100)),
                }
            )
            if len(rows) >= MAX_TRAINING_PAIRS:
                break
        processed_orders += 1
        if processed_orders % 5000 == 0:
            print(f"  Processed orders: {processed_orders:,} | training rows: {len(rows):,}")
        if len(rows) >= MAX_TRAINING_PAIRS:
            print(f"  Reached max training pair cap: {MAX_TRAINING_PAIRS:,}")
            break

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["product_a", "product_b", "product_a_price", "product_b_price"]
    )
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("other").astype(str)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["discount_amount"] = pd.to_numeric(df["discount_amount"], errors="coerce").fillna(0).clip(0, 100)
    df["discount_a"] = pd.to_numeric(df["discount_a"], errors="coerce").fillna(0).clip(0, 100)
    df["discount_b"] = pd.to_numeric(df["discount_b"], errors="coerce").fillna(0).clip(0, 100)
    df.to_csv(base / "training_data.csv", index=False, encoding="utf-8-sig")
    return df


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def train_models(data_dir: Path | None = None):
    """Train the free-item classifier and discount regressor."""
    base = data_dir or _data_dir()
    out = _output_dir()
    df = _prepare_dataset(base)
    if df.empty:
        raise ValueError("training_data.csv is empty. Check filtered orders and discount fields.")
    print(f"  Training data: {len(df)} rows")

    X = df[ALL_FEATURES]
    y_cls = df["free_item"]
    y_reg = df[["discount_a", "discount_b"]].values

    preprocessor = _build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X_transformed, y_cls, y_reg, test_size=0.2, random_state=42
    )

    # --- Train both RandomForest and XGBoost, pick best ---
    candidates_clf = {
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1, eval_metric="logloss", verbosity=0,
        ),
    }
    base_regressors = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbosity=0,
        ),
    }
    candidates_reg = {
        name: MultiOutputRegressor(reg, n_jobs=-1)
        for name, reg in base_regressors.items()
    }

    comparison: dict[str, dict] = {}

    best_clf_name, best_clf, best_clf_f1 = "", None, -1.0
    for name, model in candidates_clf.items():
        print(f"  Training {name} classifier ...")
        model.fit(X_train, y_cls_train)
        pred = model.predict(X_test)
        a = accuracy_score(y_cls_test, pred)
        f = f1_score(y_cls_test, pred, zero_division=0)
        p = precision_score(y_cls_test, pred, zero_division=0)
        r = recall_score(y_cls_test, pred, zero_division=0)
        print(f"    {name}: Accuracy={a:.4f}  F1={f:.4f}  Precision={p:.4f}  Recall={r:.4f}")
        comparison[f"classifier_{name}"] = {"accuracy": a, "f1": f, "precision": p, "recall": r}
        if f > best_clf_f1:
            best_clf_name, best_clf, best_clf_f1 = name, model, f

    best_reg_name, best_reg, best_reg_rmse = "", None, float("inf")
    for name, model in candidates_reg.items():
        print(f"  Training {name} regressor ...")
        model.fit(X_train, y_reg_train)
        pred = model.predict(X_test)
        rm = float(np.sqrt(mean_squared_error(y_reg_test, pred)))
        ma = float(mean_absolute_error(y_reg_test, pred))
        r2v = float(r2_score(y_reg_test, pred))
        print(f"    {name}: RMSE={rm:.4f}  MAE={ma:.4f}  R2={r2v:.4f}")
        comparison[f"regressor_{name}"] = {"rmse": rm, "mae": ma, "r2": r2v}
        if rm < best_reg_rmse:
            best_reg_name, best_reg, best_reg_rmse = name, model, rm

    print(f"\n  >> Best classifier: {best_clf_name}")
    print(f"  >> Best regressor:  {best_reg_name}\n")

    clf = best_clf
    reg = best_reg
    y_cls_pred = clf.predict(X_test)
    y_reg_pred = reg.predict(X_test)
    acc = accuracy_score(y_cls_test, y_cls_pred)
    f1 = f1_score(y_cls_test, y_cls_pred, zero_division=0)
    precision = precision_score(y_cls_test, y_cls_pred, zero_division=0)
    recall = recall_score(y_cls_test, y_cls_pred, zero_division=0)
    cls_cm = confusion_matrix(y_cls_test, y_cls_pred)
    rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_reg_pred)))
    mae = float(mean_absolute_error(y_reg_test, y_reg_pred))
    r2 = float(r2_score(y_reg_test, y_reg_pred))

    cls_cv = cross_val_score(
        clone(clf),
        X_transformed,
        y_cls,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
    )
    reg_cv = cross_val_score(
        clone(reg),
        X_transformed,
        y_reg,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
    )
    reg_cv_rmse = (-reg_cv).astype(float)

    # --- Save ---
    with (out / "free_item_model.pkl").open("wb") as f:
        pickle.dump(clf, f)
    with (out / "discount_model.pkl").open("wb") as f:
        pickle.dump(reg, f)
    with (out / "preprocessor.pkl").open("wb") as f:
        pickle.dump(preprocessor, f)

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "best_models": {
            "classifier": best_clf_name,
            "regressor": best_reg_name,
        },
        "model_comparison": {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in comparison.items()},
        "classification": {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": cls_cm.tolist(),
            "cross_val_accuracy_mean": float(cls_cv.mean()),
            "cross_val_accuracy_std": float(cls_cv.std()),
            "cross_val_accuracy_scores": [float(x) for x in cls_cv],
            "report": classification_report(
                y_cls_test, y_cls_pred, zero_division=0, output_dict=True
            ),
        },
        "regression": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2),
            "cross_val_rmse_mean": float(reg_cv_rmse.mean()),
            "cross_val_rmse_std": float(reg_cv_rmse.std()),
            "cross_val_rmse_scores": [float(x) for x in reg_cv_rmse],
        },
        "note": (
            "Labels are derived from historical order-line discounts in multi-item orders. "
            "Free-item label is inferred from relative discount intensity when explicit bundle labels are absent."
        ),
    }
    with (out / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    report_lines = [
        "QEU Bundle Model Evaluation Report",
        "=================================",
        "",
        f"Dataset rows: {metrics['dataset']['rows']}",
        f"Train/Test split: {metrics['dataset']['train_rows']} / {metrics['dataset']['test_rows']}",
        "",
        "Classifier (free_item)",
        "----------------------",
        f"Accuracy:  {acc:.4f}",
        f"F1 score:  {f1:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
        f"CV Accuracy (5-fold): {cls_cv.mean():.4f} ± {cls_cv.std():.4f}",
        f"Confusion Matrix [[TN, FP], [FN, TP]]: {cls_cm.tolist()}",
        "",
        "Regressor (discount_amount)",
        "---------------------------",
        f"RMSE: {rmse:.4f}",
        f"MAE:  {mae:.4f}",
        f"R2:   {r2:.4f}",
        f"CV RMSE (5-fold): {reg_cv_rmse.mean():.4f} ± {reg_cv_rmse.std():.4f}",
        "",
        "Important note:",
        metrics["note"],
        "",
        "Model comparison:",
        *(f"  {k}: {v}" for k, v in comparison.items()),
        f"Best classifier: {best_clf_name}",
        f"Best regressor:  {best_reg_name}",
        "",
        "Copy-paste summary:",
        (
            f"Classifier ({best_clf_name}) -> Acc {acc:.4f}, F1 {f1:.4f}, Precision {precision:.4f}, Recall {recall:.4f}; "
            f"Regressor ({best_reg_name}) -> RMSE {rmse:.4f}, MAE {mae:.4f}, R2 {r2:.4f}."
        ),
    ]
    with (out / "model_evaluation_report.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"  Saved models -> {out}")
    print(f"  Saved metrics -> {out / 'model_metrics.json'}")
    print(f"  Saved report  -> {out / 'model_evaluation_report.txt'}")
    return clf, reg, preprocessor


if __name__ == "__main__":
    print("Phase 7: Training ML models ...")
    train_models()
    print("Phase 7 complete.")
