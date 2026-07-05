"""Load and split the Criteo Uplift dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load() -> pd.DataFrame:
    candidates = sorted(DATA_DIR.glob("criteo*"))
    if not candidates:
        raise FileNotFoundError(f"no criteo file in {DATA_DIR}")
    path = candidates[0]
    if path.suffix in {".csv", ".gz"}:
        df = pd.read_csv(path)
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"unknown extension: {path.suffix}")
    return df


def split(df: pd.DataFrame, test_size: float = 0.2, seed: int = 0):
    y = df["conversion"] if "conversion" in df.columns else df["visit"]
    t = df["treatment"]
    features = [c for c in df.columns if c not in {"conversion", "visit", "treatment", "exposure"}]
    x = df[features]
    return train_test_split(x, y, t, test_size=test_size, random_state=seed, stratify=t)
