"""Compute Qini curve and uplift @ K for all three learners."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load, split


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def qini_curve(y, t, uplift):
    n = len(y)
    order = np.argsort(-uplift)
    y_sorted = y[order]
    t_sorted = t[order]

    cum_t = np.cumsum(t_sorted)
    cum_c = np.cumsum(1 - t_sorted)
    cum_y_t = np.cumsum(y_sorted * t_sorted)
    cum_y_c = np.cumsum(y_sorted * (1 - t_sorted))

    total_t = cum_t[-1]
    total_c = cum_c[-1]

    q = cum_y_t - cum_y_c * (total_t / total_c if total_c > 0 else 0)
    k_pct = np.arange(1, n + 1) / n
    return k_pct, q


def uplift_at_k(y, t, uplift, k: float = 0.1) -> float:
    n = len(y)
    cutoff = int(n * k)
    order = np.argsort(-uplift)[:cutoff]
    y_top = y[order]
    t_top = t[order]
    if t_top.sum() == 0 or (1 - t_top).sum() == 0:
        return float("nan")
    return float(y_top[t_top == 1].mean() - y_top[t_top == 0].mean())


def evaluate() -> None:
    df = load()
    _, x_test, _, y_test, _, t_test = split(df)
    y = y_test.values
    t = t_test.values

    results = []
    for name in ("t", "s", "r"):
        path = MODEL_DIR / f"uplift_{name}.npy"
        if not path.exists():
            continue
        uplift = np.load(path)
        k_pct, q = qini_curve(y, t, uplift)
        qini_coef = float(np.trapezoid(q, k_pct))
        results.append({
            "learner": name,
            "qini": qini_coef,
            "uplift_at_10pct": uplift_at_k(y, t, uplift, 0.1),
            "uplift_at_30pct": uplift_at_k(y, t, uplift, 0.3),
        })
    df_res = pd.DataFrame(results)
    df_res.to_csv(MODEL_DIR / "eval.csv", index=False)
    print(df_res)


if __name__ == "__main__":
    evaluate()
