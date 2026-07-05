"""Propensity-score weighting on synthetic data.

Ground truth: X1 causes both T and Y. Naive regression of Y on T overestimates
the effect. IPW corrects it.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression


def simulate(n: int = 5000, seed: int = 0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, size=n)
    x2 = rng.normal(0, 1, size=n)
    t_prop = 1 / (1 + np.exp(-(0.5 * x1 - 0.2 * x2)))
    t = (rng.uniform(size=n) < t_prop).astype(int)
    y = 2.0 * t + 1.5 * x1 + 0.3 * x2 + rng.normal(0, 1, size=n)
    return x1, x2, t, y


def main() -> None:
    x1, x2, t, y = simulate()

    naive = y[t == 1].mean() - y[t == 0].mean()
    print(f"naive difference in means: {naive:.3f}  (true ATE = 2.0)")

    X = np.column_stack([x1, x2])
    ps_model = LogisticRegression(max_iter=1000).fit(X, t)
    ps = ps_model.predict_proba(X)[:, 1]
    ps = np.clip(ps, 0.02, 0.98)

    w = t / ps + (1 - t) / (1 - ps)
    y_t = (y[t == 1] * (w[t == 1])).sum() / (w[t == 1]).sum()
    y_c = (y[t == 0] * (w[t == 0])).sum() / (w[t == 0]).sum()
    ipw = y_t - y_c
    print(f"IPW estimate: {ipw:.3f}  (true ATE = 2.0)")


if __name__ == "__main__":
    main()
