"""Simulate an A/B test: targeted (top-K by uplift) vs random vs never."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import load, split


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def simulate(k: float = 0.1, n_boot: int = 500, seed: int = 0) -> None:
    df = load()
    _, x_test, _, y_test, _, t_test = split(df)
    y = y_test.values
    t = t_test.values
    uplift_path = MODEL_DIR / "uplift_r.npy"
    if not uplift_path.exists():
        print("run `python -m src.r_learner` first")
        return
    uplift = np.load(uplift_path)

    n = len(y)
    rng = np.random.default_rng(seed)
    cutoff = int(n * k)

    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yi, ti, ui = y[idx], t[idx], uplift[idx]

        top_idx = np.argsort(-ui)[:cutoff]
        yt = yi[top_idx]
        tt = ti[top_idx]
        mask_t = tt == 1
        mask_c = tt == 0
        if mask_t.sum() == 0 or mask_c.sum() == 0:
            continue

        targeted_lift = yt[mask_t].mean() - yt[mask_c].mean()
        random_lift = yi[ti == 1].mean() - yi[ti == 0].mean()
        boot_diffs.append(targeted_lift - random_lift)

    boot = np.array(boot_diffs)
    mean = boot.mean()
    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    print(f"Targeted top {int(k*100)} percent vs random: mean lift delta {mean:.4f}")
    print(f"95 percent CI: [{ci_low:.4f}, {ci_high:.4f}]")

    pd.DataFrame({"mean": [mean], "ci_low": [ci_low], "ci_high": [ci_high]}).to_csv(
        MODEL_DIR / "ab_simulation.csv", index=False
    )


if __name__ == "__main__":
    simulate()
