"""Phase 7: ML model training - use actual historical order discount patterns."""

from __future__ import annotations

import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
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
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

from qeu_bundling.config.paths import get_paths


def _data_dir() -> Path:
    return get_paths().data_processed_dir


def _output_dir() -> Path:
    out = get_paths().output_dir
    out.mkdir(exist_ok=True)
    return out


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
    "is_campaign_pair",
]
CATEGORICAL_FEATURES = [
    "category_a",
    "category_b",
    "importance_a",
    "importance_b",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MAX_TRAINING_PAIRS = 250000
GENERIC_TAGS = frozenset(
    {
        "ingredient",
        "saudi",
        "dish",
        "qeu_category",
        "saudi_specialties",
        "saudi_staples",
        "frequent_purchase",
        "ramadan",
    }
)


def _parse_tags(value: str | float | int | None, fallback: str = "other") -> set[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {fallback}
    text = str(value).strip()
    if not text:
        return {fallback}
    tags = {x for x in text.split("|") if x}
    return tags if tags else {fallback}


def _prepare_dataset(data_dir: Path | None = None) -> pd.DataFrame:
    """Build training data from real multi-item orders and historical discounts."""
    base = data_dir or _data_dir()

    orders = pd.read_pickle(base / "filtered_orders.pkl").copy()
    recipe = pd.read_csv(base / "product_recipe_scores.csv")
    categories = pd.read_csv(base / "product_categories.csv")
    copurchase = pd.read_csv(base / "copurchase_scores.csv")
    embeddings = np.load(base / "product_embeddings.npy")
    with (base / "embedding_mapping.json").open("r", encoding="utf-8") as f:
        emb_map = {int(k): int(v) for k, v in json.load(f).items()}

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

    if "category_tags" in categories.columns:
        tag_col = categories["category_tags"]
    else:
        tag_col = pd.Series([""] * len(categories), index=categories.index)
    tag_lookup = {
        int(pid): _parse_tags(tags, fallback=str(cat_lookup.get(int(pid), "other")))
        for pid, tags in zip(categories["product_id"].astype(int), tag_col)
    }

    purchase_lookup: dict[tuple[int, int], float] = {}
    for row in copurchase.itertuples(index=False):
        a_id = int(getattr(row, "product_a"))
        b_id = int(getattr(row, "product_b"))
        score = float(getattr(row, "score", 0.0))
        key = (a_id, b_id) if a_id <= b_id else (b_id, a_id)
        purchase_lookup[key] = max(score, purchase_lookup.get(key, 0.0))

    pair_feature_lookup: dict[tuple[int, int], dict[str, float]] = {}

    def _pair_features(a_id: int, b_id: int) -> dict[str, float]:
        key = (a_id, b_id) if a_id <= b_id else (b_id, a_id)
        cached = pair_feature_lookup.get(key)
        if cached is not None:
            return cached

        idx_a = emb_map.get(a_id)
        idx_b = emb_map.get(b_id)
        emb_score = 0.0
        if idx_a is not None and idx_b is not None:
            emb_score = float(np.clip(np.dot(embeddings[idx_a], embeddings[idx_b]) * 100.0, 0.0, 100.0))

        tags_a = tag_lookup.get(a_id, {str(cat_lookup.get(a_id, "other"))})
        tags_b = tag_lookup.get(b_id, {str(cat_lookup.get(b_id, "other"))})
        shared_count = len(tags_a & tags_b)
        specific_shared = len((tags_a - GENERIC_TAGS) & (tags_b - GENERIC_TAGS))
        shared_score = float(np.clip(specific_shared * 12.5, 0.0, 100.0))

        computed = {
            "embedding_score": emb_score,
            "purchase_score": float(purchase_lookup.get(key, 0.0)),
            "shared_categories_count": float(shared_count),
            "shared_category_score": shared_score,
        }
        pair_feature_lookup[key] = computed
        return computed

    rows: list[dict[str, float | int | str]] = []
    grouped = orders.groupby("order_id")
    processed_orders = 0
    for _, grp in grouped:
        grp = grp.dropna(subset=["product_id", "unit_price"]).copy()
        if grp["product_id"].nunique() < 2:
            continue

        # Collapse repeated lines of the same product within one order.
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

            # product_a should be the higher-priced item.
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
            pair_features = _pair_features(a_id, b_id)

            rows.append(
                {
                    "product_a": a_id,
                    "product_b": b_id,
                    "product_a_price": a_price,
                    "product_b_price": b_price,
                    "recipe_score_a": float(rec_a.get("recipe_score", 0.0)),
                    "recipe_score_b": float(rec_b.get("recipe_score", 0.0)),
                    "embedding_score": float(pair_features["embedding_score"]),
                    "purchase_score": float(pair_features["purchase_score"]),
                    "shared_categories_count": int(pair_features["shared_categories_count"]),
                    "shared_category_score": float(pair_features["shared_category_score"]),
                    "category_a": category_a,
                    "category_b": category_b,
                    "importance_a": str(rec_a.get("saudi_importance", "low")),
                    "importance_b": str(rec_b.get("saudi_importance", "low")),
                    "category_match": int(category_a == category_b),
                    "is_campaign_pair": 0,
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

    # Override labels with campaign ground-truth when available.
    campaign_path = base / "campaign_pairs.csv"
    if campaign_path.exists():
        cp = pd.read_csv(campaign_path)
        if not cp.empty:
            cp["product_a"] = cp["product_a"].astype(int)
            cp["product_b"] = cp["product_b"].astype(int)
            gt = (
                cp.groupby(["product_a", "product_b"])
                .agg(
                    free_item=("free_item", lambda x: x.mode().iloc[0]),
                    occurrences=("occurrences", "sum"),
                )
                .reset_index()
            )
            gt_lookup = {
                (int(r["product_a"]), int(r["product_b"])): int(r["free_item"])
                for _, r in gt.iterrows()
            }
            overridden = 0
            for idx, row in df.iterrows():
                key = (int(row["product_a"]), int(row["product_b"]))
                rev_key = (key[1], key[0])
                if key in gt_lookup:
                    df.at[idx, "free_item"] = gt_lookup[key]
                    df.at[idx, "is_campaign_pair"] = 1
                    overridden += 1
                elif rev_key in gt_lookup:
                    df.at[idx, "free_item"] = 1 - gt_lookup[rev_key]
                    df.at[idx, "is_campaign_pair"] = 1
                    overridden += 1
            print(f"  Campaign ground-truth labels applied to {overridden:,} training rows")

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
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _classifier_candidates(random_state: int = 42) -> dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            verbosity=0,
        ),
    }


def _regressor_candidates(random_state: int = 42) -> dict[str, MultiOutputRegressor]:
    base_regressors = {
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            random_state=random_state,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
        ),
    }
    return {name: MultiOutputRegressor(reg, n_jobs=-1) for name, reg in base_regressors.items()}


def _build_quality_labels(df: pd.DataFrame) -> pd.Series:
    """Pseudo labels for bundle relevance quality (0/1)."""
    purchase = pd.to_numeric(df.get("purchase_score", 0.0), errors="coerce").fillna(0.0)
    recipe_avg = (
        pd.to_numeric(df.get("recipe_score_a", 0.0), errors="coerce").fillna(0.0)
        + pd.to_numeric(df.get("recipe_score_b", 0.0), errors="coerce").fillna(0.0)
    ) / 2.0
    shared = pd.to_numeric(df.get("shared_category_score", 0.0), errors="coerce").fillna(0.0)
    is_campaign = pd.to_numeric(df.get("is_campaign_pair", 0), errors="coerce").fillna(0).astype(int)

    strong_positive = (is_campaign == 1) | ((purchase >= 65.0) & (recipe_avg >= 30.0) & (shared >= 20.0))
    strong_negative = (purchase <= 18.0) & (recipe_avg <= 18.0) & (shared <= 12.0)

    labels = pd.Series(0, index=df.index, dtype="int64")
    labels.loc[strong_positive] = 1
    # keep uncertain rows as 0 for conservative relevance model
    labels.loc[strong_negative] = 0

    # fallback safety: if one class collapses, use quantile split.
    if labels.nunique() < 2:
        combo = purchase * 0.5 + recipe_avg * 0.3 + shared * 0.2
        cutoff = float(combo.quantile(0.70))
        labels = (combo >= cutoff).astype(int)
    return labels


def train_models(data_dir: Path | None = None):
    """Train the free-item classifier and discount regressor."""
    base = data_dir or _data_dir()
    out = _output_dir()
    df = _prepare_dataset(base)
    if df.empty:
        raise ValueError("training_data.csv is empty. Check filtered orders and discount fields.")
    print(f"  Training data: {len(df)} rows")

    X = df[ALL_FEATURES]
    y_cls = df["free_item"].astype(int)
    y_reg = df[["discount_a", "discount_b"]].values
    y_quality = _build_quality_labels(df).astype(int)

    preprocessor = _build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)

    split_kwargs: dict[str, object] = {"test_size": 0.2, "random_state": 42}
    if y_cls.nunique() > 1:
        split_kwargs["stratify"] = y_cls
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test, y_quality_train, y_quality_test = train_test_split(
        X_transformed, y_cls, y_reg, y_quality, **split_kwargs
    )
    cls_sample_weight = compute_sample_weight(class_weight="balanced", y=y_cls_train)
    quality_sample_weight = compute_sample_weight(class_weight="balanced", y=y_quality_train)

    candidates_clf = _classifier_candidates(random_state=42)
    candidates_reg = _regressor_candidates(random_state=42)

    comparison: dict[str, dict[str, float]] = {}

    best_clf_name = ""
    best_clf = None
    best_clf_f1_macro = -1.0
    for name, model in candidates_clf.items():
        print(f"  Training {name} classifier ...")
        model.fit(X_train, y_cls_train, sample_weight=cls_sample_weight)
        pred = model.predict(X_test)
        acc_val = accuracy_score(y_cls_test, pred)
        f1_val = f1_score(y_cls_test, pred, zero_division=0)
        f1_macro_val = f1_score(y_cls_test, pred, average="macro", zero_division=0)
        balanced_acc_val = balanced_accuracy_score(y_cls_test, pred)
        precision_val = precision_score(y_cls_test, pred, zero_division=0)
        recall_val = recall_score(y_cls_test, pred, zero_division=0)
        print(
            f"    {name}: Accuracy={acc_val:.4f}  F1={f1_val:.4f}  F1_macro={f1_macro_val:.4f}  "
            f"BalancedAcc={balanced_acc_val:.4f}  Precision={precision_val:.4f}  Recall={recall_val:.4f}"
        )
        comparison[f"classifier_{name}"] = {
            "accuracy": float(acc_val),
            "f1": float(f1_val),
            "f1_macro": float(f1_macro_val),
            "balanced_accuracy": float(balanced_acc_val),
            "precision": float(precision_val),
            "recall": float(recall_val),
        }
        if f1_macro_val > best_clf_f1_macro:
            best_clf_name = name
            best_clf = model
            best_clf_f1_macro = f1_macro_val

    best_reg_name = ""
    best_reg = None
    best_reg_rmse = float("inf")
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
            best_reg_name = name
            best_reg = model
            best_reg_rmse = rm

    best_quality_name = ""
    best_quality = None
    best_quality_ap = -1.0
    for name, model in _classifier_candidates(random_state=1337).items():
        print(f"  Training {name} bundle-quality classifier ...")
        model.fit(X_train, y_quality_train, sample_weight=quality_sample_weight)
        pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = pred
        ap = average_precision_score(y_quality_test, prob)
        f1q = f1_score(y_quality_test, pred, zero_division=0)
        print(f"    {name}: PR-AUC={ap:.4f}  F1={f1q:.4f}")
        comparison[f"quality_{name}"] = {
            "pr_auc": float(ap),
            "f1": float(f1q),
        }
        if ap > best_quality_ap:
            best_quality_ap = float(ap)
            best_quality_name = name
            best_quality = model

    print(f"\n  >> Best classifier: {best_clf_name}")
    print(f"  >> Best regressor:  {best_reg_name}\n")
    print(f"  >> Best quality model: {best_quality_name}\n")

    clf = best_clf
    reg = best_reg
    quality_model = best_quality
    if clf is None or reg is None or quality_model is None:
        raise RuntimeError("Model selection failed; classifier/regressor/quality model missing.")

    y_cls_pred = clf.predict(X_test)
    y_reg_pred = reg.predict(X_test)
    acc = accuracy_score(y_cls_test, y_cls_pred)
    f1 = f1_score(y_cls_test, y_cls_pred, zero_division=0)
    f1_macro = f1_score(y_cls_test, y_cls_pred, average="macro", zero_division=0)
    balanced_acc = balanced_accuracy_score(y_cls_test, y_cls_pred)
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
        scoring="f1_macro",
    )
    reg_cv = cross_val_score(
        clone(reg),
        X_transformed,
        y_reg,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_root_mean_squared_error",
    )
    reg_cv_rmse = (-reg_cv).astype(float)
    quality_cv = cross_val_score(
        clone(quality_model),
        X_transformed,
        y_quality,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="average_precision",
    )

    with (out / "free_item_model.pkl").open("wb") as f:
        pickle.dump(clf, f)
    with (out / "discount_model.pkl").open("wb") as f:
        pickle.dump(reg, f)
    with (out / "bundle_quality_model.pkl").open("wb") as f:
        pickle.dump(quality_model, f)
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
            "bundle_quality": best_quality_name,
        },
        "model_comparison": {k: {mk: round(mv, 4) for mk, mv in v.items()} for k, v in comparison.items()},
        "classification": {
            "accuracy": float(acc),
            "f1": float(f1),
            "f1_macro": float(f1_macro),
            "balanced_accuracy": float(balanced_acc),
            "precision": float(precision),
            "recall": float(recall),
            "confusion_matrix": cls_cm.tolist(),
            "cross_val_f1_macro_mean": float(cls_cv.mean()),
            "cross_val_f1_macro_std": float(cls_cv.std()),
            "cross_val_f1_macro_scores": [float(x) for x in cls_cv],
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
        "bundle_quality": {
            "pr_auc": float(best_quality_ap),
            "cross_val_pr_auc_mean": float(quality_cv.mean()),
            "cross_val_pr_auc_std": float(quality_cv.std()),
            "cross_val_pr_auc_scores": [float(x) for x in quality_cv],
        },
        "note": (
            "Labels are derived from historical order-line discounts in multi-item orders. "
            "Free-item label is inferred from relative discount intensity when explicit bundle labels are absent. "
            "Pair features are sourced from full processed artifacts (co-purchase, embeddings, and category tags). "
            "Bundle-quality labels are pseudo-labeled from campaign + strong compatibility evidence."
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
        f"F1 macro:  {f1_macro:.4f}",
        f"Balanced Accuracy: {balanced_acc:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall:    {recall:.4f}",
        f"CV F1 macro (5-fold): {cls_cv.mean():.4f} +/- {cls_cv.std():.4f}",
        f"Confusion Matrix [[TN, FP], [FN, TP]]: {cls_cm.tolist()}",
        "",
        "Regressor (discount_a, discount_b)",
        "-------------------------------",
        f"RMSE: {rmse:.4f}",
        f"MAE:  {mae:.4f}",
        f"R2:   {r2:.4f}",
        f"CV RMSE (5-fold): {reg_cv_rmse.mean():.4f} +/- {reg_cv_rmse.std():.4f}",
        "",
        "Important note:",
        metrics["note"],
        "",
        "Model comparison:",
        *(f"  {k}: {v}" for k, v in comparison.items()),
        f"Best classifier: {best_clf_name}",
        f"Best regressor:  {best_reg_name}",
        f"Best bundle-quality model: {best_quality_name}",
        "",
        "Copy-paste summary:",
        (
            f"Classifier ({best_clf_name}) -> Acc {acc:.4f}, F1 {f1:.4f}, F1_macro {f1_macro:.4f}, "
            f"Precision {precision:.4f}, Recall {recall:.4f}; "
            f"Regressor ({best_reg_name}) -> RMSE {rmse:.4f}, MAE {mae:.4f}, R2 {r2:.4f}; "
            f"Bundle quality ({best_quality_name}) -> PR-AUC {best_quality_ap:.4f}."
        ),
    ]
    with (out / "model_evaluation_report.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print(f"  Saved models -> {out}")
    print(f"  Saved metrics -> {out / 'model_metrics.json'}")
    print(f"  Saved report  -> {out / 'model_evaluation_report.txt'}")
    return clf, reg, preprocessor


def run():
    return train_models()


if __name__ == "__main__":
    print("Phase 7: Training ML models ...")
    train_models()
    print("Phase 7 complete.")
