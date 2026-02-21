# Model Calibration

## What it is

Model calibration is about ensuring that when your model says "there is a 30% chance of a claim," roughly 30% of the policies with that prediction actually do have claims. A well-calibrated model produces probabilities that match real-world frequencies. Calibration is distinct from discrimination: a model can be excellent at ranking risks (high AUC) while being terrible at estimating actual probabilities (poor calibration). Discrimination asks "Can the model tell apart high-risk from low-risk?" Calibration asks "Are the predicted probabilities accurate in an absolute sense?"

Think of calibration like a weather forecast. A forecaster who says "30% chance of rain" is well-calibrated if, historically, it rains about 30% of the time when they make that forecast. If it actually rains 60% of the time when they say 30%, the forecaster has good discrimination (they know when rain is more likely) but poor calibration (the numbers are wrong).

## Why it matters

For actuarial and data science practitioners, calibration is arguably more important than discrimination. Here is why:

**Insurance pricing depends on accurate probabilities.** An actuary calculates expected loss as frequency times severity. If the frequency model predicts 5% when the true frequency is 10%, premiums will be systematically too low, leading to underwriting losses. Good discrimination (ranking) is necessary but not sufficient -- you need the numbers to be right.

**Regulatory and governance requirements.** Actuarial standards require that assumptions be reasonable. A model whose predicted probabilities do not match observed frequencies is failing a basic reasonableness test.

**Business decisions depend on absolute probabilities.** If a fraud detection model says "90% probability of fraud," the investigation team allocates expensive resources accordingly. If the model is poorly calibrated and "90%" actually means "40%," resources are wasted.

**Post-processing calibration can improve any model.** Even if you inherit a poorly calibrated model, you can often fix the calibration without retraining, using techniques like Platt scaling or isotonic regression.

## When to use it

- **After training any model that outputs probabilities** -- check calibration before trusting the probabilities.
- **When probabilities directly feed into business decisions** -- pricing, resource allocation, risk assessment.
- **When using tree-based models** (random forest, gradient boosting) -- these are often poorly calibrated out of the box.
- **When combining predictions from multiple models** (ensembles) -- check whether the combined output is calibrated.
- **When comparing models** -- two models might have similar AUC but very different calibration; the better-calibrated one is preferable for pricing.
- **Before model deployment** -- calibration should be part of the model validation checklist.
- **When monitoring deployed models** -- calibration can drift over time as the population changes.

## How it works (intuitive)

### The calibration curve (reliability diagram)

The calibration curve is the primary diagnostic tool. Here is how to create one:

1. **Sort observations by predicted probability** (low to high).
2. **Group into bins** (e.g., 10 bins: 0-10%, 10-20%, ..., 90-100%).
3. **For each bin, compute**:
   - The average predicted probability (x-axis).
   - The actual proportion of positives (y-axis).
4. **Plot** average predicted vs actual proportion.

A perfectly calibrated model follows the 45-degree diagonal line. Points above the diagonal mean the model underestimates (actual frequency is higher than predicted). Points below mean it overestimates.

*Analogy*: Think of testing a bathroom scale. You put a series of known weights on it -- 5 kg, 10 kg, 15 kg -- and record what it displays. If the display matches the true weight, the scale is calibrated. If it consistently reads too low, it needs calibration.

### Why tree models are often poorly calibrated

Tree-based models (random forest, gradient boosting) are among the most popular ML models, but they often have poor calibration. Understanding why helps you fix it.

**Random forests** average predictions from many trees. Each tree leaf contains the proportion of positive cases that fell into that leaf during training. But these proportions tend to be pushed away from 0 and 1. The averaging across many trees with slightly different splits produces predictions clustered in the middle of the probability range. Random forests rarely predict very low or very high probabilities, even when they should.

*Analogy*: Imagine 100 weather forecasters. Some would say "95% chance of rain" and others "50% chance." When you average them, you get something like "70%." The average smooths out the extremes, even when the extreme was correct.

**Gradient boosting** (XGBoost, LightGBM) is better than random forests but still not perfectly calibrated. The sequential nature of boosting and the learning rate can cause predicted probabilities to be too extreme or too conservative, depending on the hyperparameters.

**Logistic regression** is generally well-calibrated because it is trained by directly optimizing log-likelihood, which is a proper scoring rule that incentivizes calibration. But even logistic regression can become poorly calibrated with regularization or when the model is misspecified.

### Calibration methods

#### Platt scaling

Platt scaling fits a logistic regression on top of the model's raw outputs. Given the model's predicted probabilities (or scores) $s$:

$$P(\text{positive} | s) = \frac{1}{1 + \exp(-(A \cdot s + B))}$$

The parameters $A$ and $B$ are fit on a held-out calibration dataset by minimizing log-likelihood.

*Intuition*: Platt scaling is like putting a "correction lens" over your model's predictions. If the model systematically overestimates by a certain pattern, the logistic regression learns to correct that pattern.

**When to use**: When the calibration curve looks roughly sigmoidal (S-shaped). Works well when the miscalibration is smooth and monotonic.

**Limitation**: Platt scaling assumes the miscalibration can be corrected by a logistic transformation (two parameters). If the miscalibration is more complex (e.g., good calibration in the middle but poor at extremes), Platt scaling might not be enough.

#### Isotonic regression

Isotonic regression fits a non-decreasing step function to the relationship between predicted probabilities and observed frequencies. It makes no parametric assumptions about the shape of the correction.

*Intuition*: Isotonic regression is like drawing a staircase function through a scatter plot of (predicted, actual) pairs, with the only constraint being that the function never decreases. It can capture any monotonic miscalibration pattern.

**When to use**: When the calibration curve has a complex or non-smooth shape. When you have enough calibration data (isotonic regression needs more data than Platt scaling because it has more flexibility).

**Limitation**: With small calibration datasets, isotonic regression can overfit. It needs at least a few hundred observations in the calibration set.

#### Temperature scaling

Common for neural networks. It divides the model's logits by a temperature parameter $T$ before applying the softmax:

$$P(\text{class } k) = \frac{\exp(z_k / T)}{\sum_j \exp(z_j / T)}$$

- $T > 1$: Makes probabilities softer (closer to uniform). Use when the model is overconfident.
- $T < 1$: Makes probabilities sharper. Use when the model is underconfident.
- $T = 1$: No change.

### Measuring calibration quality

#### Brier score

The Brier score is the mean squared error of probability predictions:

$$\text{Brier} = \frac{1}{n}\sum_{i=1}^{n}(\hat{p}_i - y_i)^2$$

Where $\hat{p}_i$ is the predicted probability and $y_i \in \{0, 1\}$ is the actual outcome.

- Range: [0, 1]. Lower is better.
- Brier score = 0: Perfect predictions.
- Brier score for always predicting the base rate: $\bar{y}(1 - \bar{y})$.

The Brier score can be decomposed into three components:

$$\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$$

- **Reliability** (calibration component): Measures how close the calibration curve is to the diagonal. Lower is better.
- **Resolution**: Measures how much the model's predictions vary across bins. Higher is better (you want the model to differentiate between risks).
- **Uncertainty**: The variance of the target variable. This is a property of the data, not the model.

#### Expected Calibration Error (ECE)

ECE is the weighted average of the calibration error across bins:

$$\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{n} \left| \text{acc}(b) - \text{conf}(b) \right|$$

Where:
- $B$ is the number of bins.
- $n_b$ is the number of observations in bin $b$.
- $\text{acc}(b)$ is the actual positive rate in bin $b$.
- $\text{conf}(b)$ is the average predicted probability in bin $b$.

ECE gives you a single number summarizing calibration quality. Typical good values are below 0.05 (5 percentage points average miscalibration).

#### Maximum Calibration Error (MCE)

$$\text{MCE} = \max_{b=1}^{B} \left| \text{acc}(b) - \text{conf}(b) \right|$$

MCE reports the worst miscalibration across any bin. Useful when you care about worst-case accuracy of probabilities.

### The calibration workflow

1. **Train your model** on the training set.
2. **Evaluate discrimination** (AUC, Gini) on the validation set -- make sure the model can rank.
3. **Plot the calibration curve** on the validation set -- check if probabilities are accurate.
4. **If poorly calibrated**, apply Platt scaling or isotonic regression using a separate calibration set (or use the validation set if you have no separate calibration data).
5. **Re-evaluate calibration** after the correction.
6. **Report both discrimination and calibration** metrics.

**Critical point**: The calibration correction should be fit on data that was NOT used to train the model. Otherwise, you are calibrating to in-sample biases.

## Key concepts

### Proper scoring rules

A scoring rule is **proper** if the expected score is minimized (for loss-type scores) when the predicted probabilities equal the true probabilities. Log loss and the Brier score are both proper scoring rules. Accuracy is not.

Why this matters: If you optimize a proper scoring rule during training, you naturally incentivize calibration. This is why logistic regression (which optimizes log loss) tends to be well-calibrated.

### Calibration vs discrimination summary

| Aspect | Calibration | Discrimination |
|---|---|---|
| Question | Are the probabilities accurate? | Can the model rank risks? |
| Metric | Brier score, ECE, calibration curve | AUC, Gini, lift |
| Can be fixed post-hoc? | Yes (Platt, isotonic) | Not easily |
| Required for pricing? | Yes -- directly impacts premium accuracy | Yes -- needed for risk differentiation |

### Calibration and class imbalance

With imbalanced datasets (e.g., 2% claim rate), calibration curves can be noisy because bins at the higher probability end have very few observations. Solutions:
- Use more bins at the low end and fewer at the high end (adaptive binning).
- Use quantile-based bins (equal number of observations per bin) instead of equal-width bins.
- Look at the Brier score rather than relying solely on the visual calibration curve.

## Python implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.ensemble import GradientBoostingClassifier

# --- Train a model ---
model = GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]

# --- Calibration curve ---
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10, strategy='uniform')

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(prob_pred, prob_true, 's-', label='Model')
ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_title('Calibration Curve (Reliability Diagram)')
ax.legend()
plt.show()

# --- Brier score ---
brier = brier_score_loss(y_test, y_prob)
print(f"Brier score: {brier:.4f}")

# --- Expected Calibration Error ---
def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return ece / len(y_true)

ece = expected_calibration_error(y_test, y_prob)
print(f"ECE: {ece:.4f}")

# --- Platt scaling (logistic calibration) ---
cal_model_platt = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
cal_model_platt.fit(X_cal, y_cal)  # Fit on calibration set
y_prob_platt = cal_model_platt.predict_proba(X_test)[:, 1]

# --- Isotonic regression calibration ---
cal_model_iso = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
cal_model_iso.fit(X_cal, y_cal)
y_prob_iso = cal_model_iso.predict_proba(X_test)[:, 1]

# --- Compare calibration before and after ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, probs, title in zip(
    axes,
    [y_prob, y_prob_platt, y_prob_iso],
    ['Original', 'Platt Scaling', 'Isotonic Regression']
):
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    ax.plot(prob_pred, prob_true, 's-')
    ax.plot([0, 1], [0, 1], 'k--')
    brier = brier_score_loss(y_test, probs)
    ax.set_title(f'{title}\nBrier: {brier:.4f}')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')

plt.tight_layout()
plt.show()

# --- Calibration with cross-validation (when no separate cal set) ---
cal_model_cv = CalibratedClassifierCV(
    GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42),
    method='isotonic',
    cv=5  # Uses 5-fold CV to generate out-of-fold calibration data
)
cal_model_cv.fit(X_train, y_train)
y_prob_cv = cal_model_cv.predict_proba(X_test)[:, 1]
```

## R implementation (if applicable)

```r
library(gbm)
library(probably)  # tidymodels calibration tools

# --- Calibration curve ---
# Using base R
calibration_data <- function(y_true, y_prob, n_bins = 10) {
  bins <- cut(y_prob, breaks = seq(0, 1, length.out = n_bins + 1),
              include.lowest = TRUE)
  data.frame(
    bin = levels(bins),
    avg_predicted = tapply(y_prob, bins, mean),
    avg_actual = tapply(y_true, bins, mean),
    n = tapply(y_true, bins, length)
  )
}

cal_df <- calibration_data(y_test, y_prob)
plot(cal_df$avg_predicted, cal_df$avg_actual,
     xlab = "Mean predicted probability",
     ylab = "Fraction of positives",
     main = "Calibration Curve")
abline(0, 1, lty = 2)

# --- Platt scaling ---
platt_model <- glm(y_cal ~ y_prob_cal, family = binomial)
y_prob_platt <- predict(platt_model, newdata = data.frame(y_prob_cal = y_prob_test),
                         type = "response")

# --- Brier score ---
brier <- mean((y_prob - y_test)^2)
cat("Brier score:", round(brier, 4), "\n")
```

## Gotchas & lessons learned

1. **Good AUC does not mean good calibration**: You can have AUC = 0.90 with terrible calibration. A model that predicts 0.01 for all negatives and 0.02 for all positives has perfect AUC (it ranks everything correctly) but predicts probabilities that are all far too low. Always check both.

2. **Tree models need calibration**: Random forests and gradient boosting almost always benefit from post-hoc calibration. Do not trust their raw `predict_proba()` outputs as actual probabilities, especially if those probabilities feed into pricing or resource allocation.

3. **Use a separate calibration set**: Calibrating on the training set will overfit the calibration. Calibrating on the test set contaminates your evaluation. Ideally, split your data into train/calibration/test. If data is limited, use `CalibratedClassifierCV` with `cv=5` to generate out-of-fold calibration data.

4. **Isotonic regression needs data**: With fewer than a few hundred observations in the calibration set, isotonic regression can overfit (producing a jagged, unreliable calibration function). Platt scaling, with only two parameters, is safer for small calibration sets.

5. **Calibration can drift over time**: A model calibrated on 2023 data may become miscalibrated in 2025 if the underlying claim rate shifts. Monitor calibration regularly and recalibrate as needed. This is standard practice in actuarial reserving and pricing reviews.

6. **The number of bins matters**: Calibration curves and ECE are sensitive to the number and type of bins. Too few bins (5) may hide miscalibration. Too many bins (50) produce noisy estimates. Start with 10 equal-width bins, then try quantile-based bins for a different perspective.

7. **Do not confuse Brier score components**: The Brier score conflates calibration and discrimination. A model can have a low Brier score because it discriminates well, even if calibration is mediocre. Use the Brier score decomposition or ECE to isolate the calibration component.

8. **Calibration after threshold selection**: If you calibrate the model and then choose a classification threshold, the threshold should be chosen based on the calibrated probabilities. The calibrated probabilities have a direct interpretation (actual risk level), which makes threshold selection more intuitive.

9. **Multiclass calibration is harder**: For multiclass problems, you need to calibrate each class probability while ensuring they still sum to 1. Temperature scaling is the simplest approach. Platt scaling and isotonic regression are typically applied per-class.

10. **Log loss already incentivizes calibration**: If you optimize log loss during training (as logistic regression and neural networks typically do), the model has a natural incentive to be well-calibrated. But regularization, early stopping, and other hyperparameter choices can still cause miscalibration.

## Resources

- **scikit-learn calibration guide**: https://scikit-learn.org/stable/modules/calibration.html
- **Platt (1999)**: "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods" -- the original Platt scaling paper.
- **Niculescu-Mizil & Caruana (2005)**: "Predicting Good Probabilities With Supervised Learning" -- comprehensive comparison of calibration across model types.
- **Guo et al. (2017)**: "On Calibration of Modern Neural Networks" -- why modern neural nets are miscalibrated and how temperature scaling fixes it. https://arxiv.org/abs/1706.04599
- **Christoph Molnar -- Interpretable ML Book, Chapter on Calibration**: https://christophm.github.io/interpretable-ml-book/
- **probably R package (tidymodels)**: https://probably.tidymodels.org/ -- tools for calibration in the tidyverse ecosystem.
- **CAS Predictive Analytics seminar materials**: Often cover calibration in the context of insurance pricing models.
