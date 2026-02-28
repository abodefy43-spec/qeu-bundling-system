"""Phase 2: Product name embeddings using Sentence Transformers."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sentence_transformers import SentenceTransformer

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 64
MIN_ORDERS_FOR_EMBEDDING = 20
TOP_N_SIMILARITY = 4000


def _data_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "data"


def _load_unique_products(data_dir: Path | None = None) -> pd.DataFrame:
    """Return unique products with frequency, filtered by minimum order count."""
    base = data_dir or _data_dir()
    orders = pd.read_pickle(base / "filtered_orders.pkl")
    order_counts = (
        orders.groupby("product_id")["order_id"]
        .nunique()
        .reset_index(name="order_count")
    )
    products = (
        orders[["product_id", "product_name"]].dropna(subset=["product_name"])
        .drop_duplicates(subset=["product_id"])
        .merge(order_counts, on="product_id", how="left")
        .query("order_count >= @MIN_ORDERS_FOR_EMBEDDING")
        .sort_values("order_count", ascending=False)
        .reset_index(drop=True)
    )
    products["product_id"] = products["product_id"].astype(int)
    products["product_name"] = products["product_name"].astype(str).apply(_clean_product_name)
    return products


def _clean_product_name(value: str) -> str:
    """Extract EN name when raw JSON appears; keep multilingual fallback."""
    if not isinstance(value, str):
        return str(value)
    text = value.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            en = str(parsed.get("en", "")).strip()
            ar = str(parsed.get("ar", "")).strip()
            return en if en else ar
    except json.JSONDecodeError:
        pass
    return re.sub(r"\s+", " ", text).strip()


def generate_embeddings(
    data_dir: Path | None = None,
    batch_size: int = BATCH_SIZE,
) -> tuple[np.ndarray, dict[int, int]]:
    """Embed all unique product names and save to disk.

    Returns
    -------
    embeddings : np.ndarray of shape (n_products, dim)
    mapping    : dict mapping product_id -> row index in the embeddings array
    """
    base = data_dir or _data_dir()
    products = _load_unique_products(base)
    names = products["product_name"].fillna("").astype(str).tolist()
    product_ids = products["product_id"].tolist()

    print(f"  Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"  Encoding {len(names):,} product names (batch_size={batch_size}) ...")
    # Encode in chunks to keep memory bounded for larger datasets.
    chunks: list[np.ndarray] = []
    for start in range(0, len(names), 1000):
        end = min(start + 1000, len(names))
        emb_chunk = model.encode(
            names[start:end],
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        chunks.append(np.asarray(emb_chunk, dtype=np.float32))
    embeddings = np.vstack(chunks) if chunks else np.empty((0, 384), dtype=np.float32)

    mapping: dict[int, int] = {pid: idx for idx, pid in enumerate(product_ids)}

    emb_path = base / "product_embeddings.npy"
    map_path = base / "embedding_mapping.json"
    np.save(emb_path, embeddings)
    with map_path.open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in mapping.items()}, f)

    _save_top_similarity_matrix(
        products=products,
        embeddings=embeddings,
        base=base,
        top_n=min(TOP_N_SIMILARITY, len(products)),
    )

    print(f"  Saved embeddings   -> {emb_path}  shape={embeddings.shape}")
    print(f"  Saved mapping      -> {map_path}  ({len(mapping):,} products)")
    return embeddings, mapping


def _save_top_similarity_matrix(
    products: pd.DataFrame,
    embeddings: np.ndarray,
    base: Path,
    top_n: int,
) -> None:
    """Save sparse similarity matrix for top-N frequent products."""
    if top_n <= 1 or embeddings.size == 0:
        return
    top = products.head(top_n).copy()
    top_indices = top.index.to_numpy()
    top_embeddings = embeddings[top_indices]
    sim = np.matmul(top_embeddings, top_embeddings.T)
    sim = np.clip(sim, -1.0, 1.0).astype(np.float32)
    sim_sparse = sparse.csr_matrix(sim)
    sparse.save_npz(base / "top_product_similarity.npz", sim_sparse)
    with (base / "top_product_ids.json").open("w", encoding="utf-8") as f:
        json.dump([int(x) for x in top["product_id"].tolist()], f)
    print(f"  Saved sparse similarity -> {base / 'top_product_similarity.npz'}  (top {top_n})")


def load_embeddings(data_dir: Path | None = None) -> tuple[np.ndarray, dict[int, int]]:
    """Load previously saved embeddings and mapping."""
    base = data_dir or _data_dir()
    embeddings = np.load(base / "product_embeddings.npy")
    with (base / "embedding_mapping.json").open("r", encoding="utf-8") as f:
        raw = json.load(f)
    mapping = {int(k): v for k, v in raw.items()}
    return embeddings, mapping


def embedding_similarity(
    embeddings: np.ndarray,
    mapping: dict[int, int],
    pid_a: int,
    pid_b: int,
) -> float:
    """Cosine similarity between two products (already L2-normalised)."""
    idx_a = mapping.get(pid_a)
    idx_b = mapping.get(pid_b)
    if idx_a is None or idx_b is None:
        return 0.0
    return float(np.dot(embeddings[idx_a], embeddings[idx_b]))


if __name__ == "__main__":
    print("Phase 2: Generating product embeddings ...")
    embs, mapping = generate_embeddings()
    print("Phase 2 complete.")
