# Calibration and Conformal Prediction

A model that ranks well can still be miscalibrated: predicted probabilities may not match observed frequencies, and predicted point estimates may carry no honest measure of uncertainty. This essay covers the two complementary fixes: post-hoc calibration of probabilistic outputs, and conformal prediction for distribution-free predictive intervals.

For the actuarial reader, this is high-leverage material. Premium calculations, capital reserves, and reinsurance pricing all hinge on the upper tails of predictive distributions; a point estimate is rarely sufficient and a regulator-facing filing usually demands a defensible uncertainty quantification.

## Calibration

A binary classifier is **calibrated** if among examples assigned probability $p$, a fraction $p$ are positive. Formally,

$$
P(Y = 1 \mid \hat p(X) = p) = p \quad \text{for all } p \in [0, 1].
$$

Calibration is independent of discrimination: a model that always outputs 0.5 is perfectly calibrated on a 50-50 dataset but useless. We want both: a model that ranks well (high AUC) and outputs trustworthy probabilities.

### Reliability diagrams

The standard visual diagnostic. Bin predictions into equal-width or equal-frequency bins; for each bin plot the average predicted probability against the empirical positive rate. A perfectly calibrated model traces the diagonal; deviations show systematic over- or under-confidence.

### Expected Calibration Error (ECE)

A scalar summary:

$$
\mathrm{ECE} = \sum_{m=1}^M \frac{|B_m|}{n} \left| \mathrm{acc}(B_m) - \mathrm{conf}(B_m) \right|,
$$

where $B_m$ is the $m$-th bin, $\mathrm{acc}(B_m)$ is the empirical accuracy in the bin, and $\mathrm{conf}(B_m)$ is the average predicted probability in the bin. Lower is better. ECE is sensitive to the binning scheme; report the binning along with the number.

### Brier score and its decomposition

The Brier score is the mean squared error of probabilistic predictions:

$$
\mathrm{BS} = \frac{1}{n} \sum_i (\hat p_i - y_i)^2.
$$

It decomposes (Murphy 1973) as

$$
\mathrm{BS} = \mathrm{Reliability} - \mathrm{Resolution} + \mathrm{Uncertainty},
$$

where reliability measures calibration error, resolution measures how much predictions vary across classes (discrimination), and uncertainty is a baseline depending only on the marginal prevalence. A model with low Brier score is both calibrated and discriminative.

## Calibration methods

### Platt scaling

Platt (1999). Fit a logistic regression to map raw scores $f(x)$ to calibrated probabilities:

$$
\hat p = \sigma(a f(x) + b),
$$

with $a, b$ learned on a held-out calibration set. Two-parameter; works best when the score-probability relationship is monotonic and roughly logistic. Originally designed for SVM scores; still useful for any monotonically-related output.

### Isotonic regression

Zadrozny, Elkan (2002). Fit a non-decreasing step function from scores to probabilities, minimizing squared error. Non-parametric, more flexible than Platt. Risk: overfitting on small calibration sets, since the step function can absorb noise. Need at least a few hundred examples.

### Temperature scaling for neural networks

Guo, Pleiss, Sun, Weinberger (2017). Modern neural networks (especially deep classifiers) are systematically overconfident. The fix is a single scalar $T$ applied to logits before softmax:

$$
\hat p_k = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}.
$$

Tune $T$ on a held-out calibration set by minimizing NLL. Higher $T$ softens the softmax; lower sharpens it. One parameter, no impact on top-1 accuracy, dramatic ECE reduction in practice. Default first move for any deep classifier.

### Trade-offs

| Method | Parameters | Data requirements | Notes |
|--------|-----------|-------------------|-------|
| Platt scaling | 2 | small (50-100) | parametric, monotone fix |
| Isotonic | up to $n$ | larger (300+) | non-parametric, can overfit |
| Temperature scaling | 1 | small | NN-specific, preserves ranking |

## Conformal prediction: distribution-free intervals

Where calibration adjusts probabilities, conformal prediction produces prediction sets (regression intervals or classification sets) with a finite-sample coverage guarantee:

$$
P(Y_{n+1} \in C(X_{n+1})) \geq 1 - \alpha,
$$

assuming only that $(X_i, Y_i)$ are exchangeable. No distributional assumption, no model assumption. The guarantee is marginal (averaged over future inputs), not conditional.

### Split conformal regression

Vovk, Gammerman, Shafer (2005); Lei et al. (2018). Algorithm:

1. Split training data into a proper training set $\mathcal{D}_{\mathrm{tr}}$ and a calibration set $\mathcal{D}_{\mathrm{cal}}$ of size $n$.
2. Train any regressor $\hat\mu$ on $\mathcal{D}_{\mathrm{tr}}$.
3. Compute nonconformity scores on the calibration set: $S_i = |Y_i - \hat\mu(X_i)|$ for $i \in \mathcal{D}_{\mathrm{cal}}$.
4. Set $\hat q$ to the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile of $\{S_i\}$.
5. The prediction interval at a new $X_{n+1}$ is $[\hat\mu(X_{n+1}) - \hat q, \hat\mu(X_{n+1}) + \hat q]$.

The coverage guarantee holds in finite samples under exchangeability. The interval width is constant (does not adapt to local difficulty), which is the main limitation of vanilla split conformal.

### Locally adaptive conformal regression

Replace the absolute residual with a normalized score $S_i = |Y_i - \hat\mu(X_i)| / \hat\sigma(X_i)$, where $\hat\sigma$ is a learned heteroskedasticity estimator. Intervals widen where the model is uncertain and tighten elsewhere, recovering local adaptivity at the cost of a small efficiency loss when $\hat\sigma$ is poorly estimated.

### CV+ and jackknife+

Barber, Candes, Ramdas, Tibshirani (2021). Use cross-validation residuals instead of a single split, recovering coverage guarantees with fewer wasted samples. Slightly looser bound than split conformal but uses all data for training and calibration.

## Conformal classification

For classification, output a **set** of plausible labels with coverage $1 - \alpha$.

### Adaptive Prediction Sets (APS)

Romano, Sesia, Candes (2020). For an example $X$, sort class probabilities in decreasing order and form a cumulative sum until the total exceeds a calibration-set quantile. The result is a label set whose size adapts to the example: easy examples get a single label, hard examples get several. Coverage guarantee remains valid.

## Weighted conformal under covariate shift

Tibshirani et al. (2019). Standard conformal assumes exchangeability. Under covariate shift ($P(Y \mid X)$ unchanged but $P(X)$ changes), exchangeability fails. The fix: weight each calibration example by the likelihood ratio $w(X_i) = p_{\mathrm{test}}(X_i) / p_{\mathrm{train}}(X_i)$ and use the weighted empirical quantile. Coverage holds under correct ratio specification; in practice $w$ is estimated.

This matters in actuarial settings whenever the inforce population shifts (new agency channel, marketing campaign reaches a different demographic) and you need to keep coverage on the new distribution.

## Actuarial applications

### Predictive intervals on claim severity

Frequency-severity GLMs produce point predictions. A regulator-facing pricing filing typically wants intervals: "with 90% confidence, this risk's expected severity is between $X$ and $Y$." Wrap the GLM with split conformal:

1. Hold out 20% of the training data as a calibration set.
2. Train the GLM on the rest.
3. Compute nonconformity scores on the calibration set.
4. Apply the conformal interval at scoring time.

Result: a distribution-free interval at the chosen $\alpha$, valid in finite samples. The pricing intervals are honest in the technical sense regulators care about.

### Reserve uncertainty bands

Loss reserving is dominated by point estimates from chain-ladder, Bornhuetter-Ferguson, or GLM reserving. Stochastic reserving (Mack, ODP bootstrap) gives parametric uncertainty. Conformal prediction provides a non-parametric alternative: validate model predictions on a held-out development period and conformalize. Particularly useful for ML-based reserving (Wuthrich 2018) where parametric uncertainty quantification is harder.

### Why a point estimate is not enough

In regulatory filings (CNBV, CNSF, IFRS 17) and in capital adequacy work, the regulator routinely asks for upper-tail estimates: VaR at 99.5%, Tail VaR, conditional capital requirements. A point estimate has no answer for any of these questions. Conformal prediction and quantile regression are the modern, defensible answers.

## Implementation

Python: `mapie` (https://github.com/scikit-learn-contrib/MAPIE) implements split conformal, CV+, jackknife+, APS, and weighted conformal with sklearn-style APIs.

```python
from mapie.regression import MapieRegressor
from sklearn.ensemble import GradientBoostingRegressor

base = GradientBoostingRegressor()
mapie = MapieRegressor(base, method="plus", cv=5)
mapie.fit(X_train, y_train)

# 90% prediction interval
y_pred, y_pis = mapie.predict(X_test, alpha=0.1)
# y_pis has shape (n, 2, 1); columns are lower and upper bounds
```

For calibration, scikit-learn's `CalibratedClassifierCV` provides Platt and isotonic; temperature scaling is one-line in PyTorch.

## Linking to the portfolio

- `insurance-pricing` (root project): conformal intervals on premium predictions.
- `recsys-movielens`: conformal sets for top-K recommendation, covering true user-preferred items with target probability.
- `mlops-model-service`: monitor coverage of conformal intervals over time as a calibration-drift detector.

## References

- Angelopoulos, Bates, "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification", 2021 (free arXiv tutorial).
- Vovk, Gammerman, Shafer, *Algorithmic Learning in a Random World*, Springer, 2005.
- Lei, G'Sell, Rinaldo, Tibshirani, Wasserman, "Distribution-Free Predictive Inference for Regression", JASA, 2018.
- Barber, Candes, Ramdas, Tibshirani, "Predictive inference with the jackknife+", Annals of Statistics, 2021.
- Romano, Sesia, Candes, "Classification with Valid and Adaptive Coverage" (APS), NeurIPS 2020.
- Tibshirani, Barber, Candes, Ramdas, "Conformal Prediction Under Covariate Shift", NeurIPS 2019.
- Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks", ICML 2017.
- Platt, "Probabilistic Outputs for Support Vector Machines", in *Advances in Large Margin Classifiers*, 1999.
- Zadrozny, Elkan, "Transforming Classifier Scores into Accurate Multiclass Probability Estimates", KDD 2002.
- Murphy, "A New Vector Partition of the Probability Score" (Brier decomposition), J. Applied Meteorology, 1973.
