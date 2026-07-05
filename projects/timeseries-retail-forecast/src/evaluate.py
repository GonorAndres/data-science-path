"""Forecast evaluation metrics."""

import numpy as np
import pandas as pd


def wape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true).sum()
    if denom == 0:
        return float("nan")
    return float(np.abs(y_true - y_pred).sum() / denom)


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.abs(y_true[mask] - y_pred[mask]) / np.abs(y_true[mask])).mean())


def bias(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_pred - y_true).mean())


def per_level_metrics(df: pd.DataFrame, level: str) -> pd.DataFrame:
    g = df.groupby(level)
    rows = []
    for key, sub in g:
        rows.append({
            level: key,
            "n": len(sub),
            "wape": wape(sub["sales"], sub["pred"]),
            "bias": bias(sub["sales"], sub["pred"]),
        })
    return pd.DataFrame(rows).sort_values("wape")


if __name__ == "__main__":
    from pathlib import Path
    for name in ["baseline_predictions", "gbm_predictions"]:
        path = Path(__file__).resolve().parent.parent / "models" / f"{name}.parquet"
        if not path.exists():
            print(f"{path.name}: not found, skip")
            continue
        df = pd.read_parquet(path)
        w = wape(df["sales"], df["pred"])
        b = bias(df["sales"], df["pred"])
        print(f"{name}: WAPE={w:.4f}  bias={b:.2f}")
