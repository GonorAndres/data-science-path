"""Offline evaluation: Recall@K, NDCG@K, coverage. Uses saved embeddings."""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .data import encode_ids, leave_last_out, load_ratings


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
K = 20


def recall_at_k(ranked: list[int], relevant: int, k: int = K) -> float:
    return 1.0 if relevant in ranked[:k] else 0.0


def ndcg_at_k(ranked: list[int], relevant: int, k: int = K) -> float:
    if relevant not in ranked[:k]:
        return 0.0
    pos = ranked.index(relevant)
    return 1.0 / math.log2(pos + 2)


def score_for_user(u_idx: int, user_emb, item_emb) -> np.ndarray:
    return item_emb @ user_emb[u_idx]


def evaluate_embedding_model(
    name: str,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    train,
    test,
    user_map: dict,
    item_map: dict,
) -> dict:
    recalls, ndcgs = [], []
    seen_items = set()
    users_seen = train.groupby("u_idx")["i_idx"].agg(set).to_dict()

    test_encoded = test[test["userId"].isin(user_map.keys()) & test["movieId"].isin(item_map.keys())]
    for row in test_encoded.itertuples(index=False):
        u = user_map[row.userId]
        i = item_map[row.movieId]
        scores = score_for_user(u, user_emb, item_emb)
        seen = users_seen.get(u, set())
        for j in seen:
            scores[j] = -np.inf
        top = np.argpartition(-scores, K)[:K]
        top = top[np.argsort(-scores[top])].tolist()
        seen_items.update(top)
        recalls.append(recall_at_k(top, i))
        ndcgs.append(ndcg_at_k(top, i))
    return {
        "model": name,
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_k": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "coverage": len(seen_items) / max(item_emb.shape[0], 1),
        "n_users": len(recalls),
    }


def run() -> None:
    ratings = load_ratings()
    train, test = leave_last_out(ratings)
    train, user_map, item_map = encode_ids(train)

    results = []
    for name, uf, vf in [
        ("als", "als_user_factors.npy", "als_item_factors.npy"),
        ("two-tower", "tt_user_emb.npy", "tt_item_emb.npy"),
    ]:
        u_path = MODEL_DIR / uf
        i_path = MODEL_DIR / vf
        if not (u_path.exists() and i_path.exists()):
            print(f"{name}: embeddings not found, skip")
            continue
        u_emb = np.load(u_path)
        i_emb = np.load(i_path)
        r = evaluate_embedding_model(name, u_emb, i_emb, train, test, user_map, item_map)
        print(r)
        results.append(r)

    if results:
        pd.DataFrame(results).to_csv(MODEL_DIR / "metrics.csv", index=False)


if __name__ == "__main__":
    run()
