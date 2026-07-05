"""Seasonal-naive baseline: y_hat(t) = y(t-7).

Produces forecasts for the last 28 days of the available data and evaluates WAPE.
"""

from pathlib import Path

import pandas as pd

from .data import load_prepared
from .evaluate import wape


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def run() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    df = load_prepared()
    df = df.sort_values(["id", "date"])
    horizon = 28
    max_date = df["date"].max()
    split_date = max_date - pd.Timedelta(days=horizon)

    test = df[df["date"] > split_date].copy()

    test["pred"] = test.groupby("id")["sales"].shift(7)
    test = test.dropna(subset=["pred"])

    overall_wape = wape(test["sales"].values, test["pred"].values)
    print(f"seasonal-naive WAPE: {overall_wape:.4f}")

    test.to_parquet(MODEL_DIR / "baseline_predictions.parquet", index=False)


if __name__ == "__main__":
    run()
