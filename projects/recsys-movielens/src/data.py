"""Load MovieLens 25M, build leave-last-out splits, encode IDs."""

from pathlib import Path

import pandas as pd
import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml-25m"


def load_ratings() -> pd.DataFrame:
    path = DATA_DIR / "ratings.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run data_download.py first.")
    ratings = pd.read_csv(path)
    return ratings


def leave_last_out(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ratings = ratings.sort_values(["userId", "timestamp"])
    last_idx = ratings.groupby("userId")["timestamp"].transform("max") == ratings["timestamp"]
    train = ratings[~last_idx]
    test = ratings[last_idx].drop_duplicates("userId", keep="last")
    return train, test


def encode_ids(
    df: pd.DataFrame,
    user_map: dict | None = None,
    item_map: dict | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    if user_map is None:
        user_ids = df["userId"].unique()
        user_map = {u: i for i, u in enumerate(sorted(user_ids))}
    if item_map is None:
        item_ids = df["movieId"].unique()
        item_map = {m: i for i, m in enumerate(sorted(item_ids))}
    df = df.copy()
    df["u_idx"] = df["userId"].map(user_map)
    df["i_idx"] = df["movieId"].map(item_map)
    return df, user_map, item_map


def sample_negatives(
    train: pd.DataFrame,
    n_items: int,
    k: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positives = set(zip(train["u_idx"], train["i_idx"]))
    rows = []
    for u, _ in train[["u_idx", "i_idx"]].itertuples(index=False):
        count = 0
        while count < k:
            j = int(rng.integers(0, n_items))
            if (u, j) not in positives:
                rows.append((u, j, 0))
                count += 1
    neg_df = pd.DataFrame(rows, columns=["u_idx", "i_idx", "label"])
    pos_df = train[["u_idx", "i_idx"]].assign(label=1)
    return pd.concat([pos_df, neg_df], ignore_index=True)
