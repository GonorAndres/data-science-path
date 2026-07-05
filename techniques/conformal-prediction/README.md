---
tags: [uncertainty-quantification, prediction-intervals, calibration, actuarial]
status: draft
created: 2026-04-25
updated: 2026-04-25
---

# Conformal Prediction

## What it is

A family of distribution-free uncertainty-quantification methods that wrap any predictor and produce prediction sets (regression intervals or classification sets) with a finite-sample coverage guarantee. The guarantee requires only that the data are exchangeable; no parametric model, no asymptotics.

## Why it matters

Point predictions are not enough in regulated settings. Insurance pricing, credit decisions, and reserve estimates must be reported with uncertainty. Conformal prediction gives that uncertainty in a model-agnostic way, valid in finite samples and directly auditable. For an actuarial reader, conformal intervals on claim severity or expected loss are the modern complement to GLM-based pricing.

## When to use it

- Regulatory filings that require predictive intervals on tariffs or reserves.
- Production ML systems where downstream decisions depend on uncertainty (set a safety margin proportional to the interval width).
- Validation of a black-box ML pricing model: even if the model is opaque, the interval coverage is provably correct.
- Anomaly detection: flag examples whose prediction set is too wide or whose true label falls outside the conformal set.

## How it works (intuitive)

The mental model: the calibration set tells you how wrong the model has historically been. To bound the future error with target probability $1 - \alpha$, take an empirical quantile of past errors and form a symmetric interval around the new prediction. Under exchangeability, the future error is no worse than the historical quantile, with the right coverage adjustment.

### Split conformal regression

1. Split training data into a proper training set and a calibration set of size $n$.
2. Train any regressor $\hat\mu$ on the proper training set.
3. Compute nonconformity scores on the calibration set: $S_i = |Y_i - \hat\mu(X_i)|$.
4. Take the $\lceil (n+1)(1-\alpha) \rceil / n$ quantile $\hat q$.
5. For a new $X_{n+1}$, the interval is $[\hat\mu(X_{n+1}) - \hat q, \hat\mu(X_{n+1}) + \hat q]$.

The coverage guarantee $P(Y \in C(X)) \geq 1 - \alpha$ holds in finite samples.

## Key concepts

- **Nonconformity score.** A function measuring how strange a labeled example looks under the model. Default for regression: absolute residual. Default for classification: $1 - \hat p_y$.
- **Calibration set.** Held-out data used to compute the score quantile. Typically 20% of the training data.
- **Coverage.** The guaranteed marginal probability of the true label being in the prediction set, $1 - \alpha$.
- **Locally adaptive variants.** Normalize residuals by an estimated heteroskedasticity term so intervals widen where the model is less certain.

## Python implementation

Use the `mapie` library (scikit-learn-contrib).

```python
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from mapie.regression import MapieRegressor

X, y = make_regression(n_samples=2000, n_features=10, noise=2.0, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

base = GradientBoostingRegressor(random_state=0)
mapie = MapieRegressor(base, method="plus", cv=5)
mapie.fit(X_train, y_train)

# 90% prediction interval
y_pred, y_pis = mapie.predict(X_test, alpha=0.1)
lower = y_pis[:, 0, 0]
upper = y_pis[:, 1, 0]

coverage = np.mean((y_test >= lower) & (y_test <= upper))
mean_width = np.mean(upper - lower)
print(f"Empirical coverage: {coverage:.3f} (target 0.90)")
print(f"Mean interval width: {mean_width:.2f}")
```

For classification with adaptive prediction sets:

```python
from mapie.classification import MapieClassifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)
mapie_clf = MapieClassifier(clf, method="aps", cv=5)
mapie_clf.fit(X_train, y_train)
y_pred, y_sets = mapie_clf.predict(X_test, alpha=0.1)
# y_sets is a boolean (n, n_classes) array indicating which labels are in each set.
```

## Actuarial application: claim severity intervals

Suppose you have a GLM or gradient-boosted model predicting expected claim severity. To wrap it in a conformal interval:

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from mapie.regression import MapieRegressor

claims = pd.read_parquet("claims.parquet")
features = ["age", "vehicle_group", "tenure_years", "annual_mileage"]
X = claims[features]
y = claims["severity"]

# train and calibrate
mapie = MapieRegressor(GradientBoostingRegressor(), method="base", cv="split")
mapie.fit(X, y)

# 95% predictive interval at scoring time
new_risk = pd.DataFrame([{
    "age": 35, "vehicle_group": 12, "tenure_years": 4, "annual_mileage": 15000
}])
point, intervals = mapie.predict(new_risk, alpha=0.05)
print(f"Severity point estimate: ${point[0]:,.0f}")
print(f"95% interval: [${intervals[0,0,0]:,.0f}, ${intervals[0,1,0]:,.0f}]")
```

The interval is valid under the (frequently approximate) assumption that the new risk is exchangeable with the calibration data. When portfolio drift breaks exchangeability, switch to weighted conformal with estimated likelihood ratios.

## Gotchas & lessons learned

- **Coverage is marginal, not conditional.** A 90% conformal interval covers 90% of points on average; the conditional coverage at any specific $X$ is not guaranteed.
- **Calibration-set size matters.** With fewer than 100 calibration examples, the quantile is noisy and intervals are unstable. Use CV+ or jackknife+ to pool data.
- **Exchangeability is broken by time.** Time-series and online settings need adapted variants (weighted conformal, OnlineConformal).
- **Interval widths reflect model quality.** A bad model gives wide intervals. Conformal does not improve the model; it only quantifies its honest uncertainty.
- **Do not look at the calibration set during model selection.** Doing so reintroduces the leakage that conformal is designed to avoid.

## Resources

- `docs/theory/18_calibration_and_conformal.md`: theory backing this technique.
- Angelopoulos, Bates, "A Gentle Introduction to Conformal Prediction", 2021 (arXiv:2107.07511).
- Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World*, Springer, 2005.
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman, "Distribution-Free Predictive Inference for Regression", JASA, 2018.
- `mapie` documentation: https://mapie.readthedocs.io/
- Romano, Sesia, Candes, "Classification with Valid and Adaptive Coverage" (APS), NeurIPS 2020.
