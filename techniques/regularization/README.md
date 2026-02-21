# Regularization

## What it is

Regularization is a technique that constrains a model's complexity to prevent overfitting. It works by adding a penalty term to the model's loss function that discourages large or extreme parameter values. Think of it as putting a leash on your model: without it, the model is free to twist and contort itself to fit every quirk and noise in the training data; with it, the model is pulled back toward simpler, more generalizable solutions. Regularization is one of the most fundamental tools in the machine learning toolkit and is especially critical when you have many features, limited data, or correlated predictors.

## Why it matters

For actuarial and data science practitioners, regularization addresses a core tension: you want a model complex enough to capture real patterns but simple enough to generalize to new data. In insurance pricing, an unregularized GLM with 200 rating factors might fit the training data beautifully but produce wildly unstable predictions on new policies. Regularization tames this instability.

Regularization also serves as an automatic feature selection mechanism (in the case of L1/Lasso) and helps deal with multicollinearity -- a common problem when policy databases have many correlated variables (e.g., driver age, years licensed, and years claim-free). In actuarial practice, regulators expect models to be stable and justifiable. Regularization helps on both counts.

## When to use it

- **When your model is overfitting** -- training performance is much better than validation/test performance.
- **When you have many features relative to observations** -- regularization prevents the model from memorizing noise.
- **When features are correlated** -- regularization stabilizes coefficient estimates that would otherwise be wildly uncertain.
- **When you want automatic feature selection** -- L1 (Lasso) regularization drives unimportant coefficients to exactly zero.
- **When building GLMs for insurance pricing** -- regularized GLMs (glmnet) are becoming standard in actuarial pricing.
- **When you need a model that generalizes well to out-of-sample data** -- regularization directly optimizes for this.
- **When you want to control the bias-variance tradeoff** -- regularization lets you dial in the right amount of complexity.

## How it works (intuitive)

### The overfitting problem

Imagine you are trying to draw a smooth curve through 20 data points. An unregularized model (high-degree polynomial, for example) might pass through every single point -- a perfect fit. But if you collect 20 new points, the curve probably misses them badly because it was contorting to fit noise in the original data.

Regularization says: "I am going to penalize you for making the curve too wiggly." The more complex the curve (larger coefficients), the higher the penalty. The model must balance two goals: fitting the data well AND keeping the parameters small.

### The loss function with regularization

The standard loss function (e.g., sum of squared errors for regression) becomes:

$$\text{Regularized Loss} = \text{Data Loss} + \lambda \cdot \text{Penalty}$$

Where:
- **Data Loss**: How well the model fits the training data (e.g., MSE, log-likelihood).
- **Penalty**: A measure of model complexity (how big the coefficients are).
- **$\lambda$ (lambda)**: A tuning parameter that controls the strength of the penalty. $\lambda = 0$ means no regularization (back to the original problem). Large $\lambda$ means heavy regularization (coefficients shrink toward zero).

### L2 regularization (Ridge)

Ridge regression penalizes the sum of squared coefficients:

$$\text{Ridge Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p}\beta_j^2$$

*Analogy*: Imagine each coefficient is attached to the origin by a rubber band. Ridge pulls all coefficients toward zero, but never exactly to zero. It shrinks them, distributing the "work" of prediction across many features rather than relying heavily on a few.

**Geometric intuition**: The constraint region for Ridge is a circle (in 2D) or a sphere (in higher dimensions). The optimal coefficients are found where the contours of the loss function first touch this sphere. Because the sphere has no corners, the coefficients never hit exactly zero -- they just get very small.

### L1 regularization (Lasso)

Lasso (Least Absolute Shrinkage and Selection Operator) penalizes the sum of absolute values of coefficients:

$$\text{Lasso Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p}|\beta_j|$$

*Analogy*: Lasso uses a budget system. You have a fixed budget of "coefficient dollars" to spread across features. When the budget is tight, you are forced to spend it only on the most important features, leaving the rest at zero.

**Geometric intuition**: The constraint region for Lasso is a diamond (in 2D) or a cross-polytope (in higher dimensions). The diamond has sharp corners that sit on the axes. The loss contours are much more likely to touch the diamond at a corner, which means one or more coefficients are exactly zero. This is why Lasso performs automatic feature selection.

### Elastic Net

Elastic Net combines both L1 and L2 penalties:

$$\text{Elastic Net Loss} = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \left[\alpha \sum_{j=1}^{p}|\beta_j| + \frac{(1-\alpha)}{2} \sum_{j=1}^{p}\beta_j^2 \right]$$

Where $\alpha$ controls the mix:
- $\alpha = 1$: pure Lasso
- $\alpha = 0$: pure Ridge
- $0 < \alpha < 1$: a blend

*Analogy*: Elastic Net is like having both a budget constraint (L1) and rubber bands (L2). The budget forces some coefficients to zero (feature selection), while the rubber bands keep the remaining coefficients from getting too large (stability).

**Why Elastic Net matters**: When features are correlated, Lasso tends to arbitrarily pick one and zero out the others. Elastic Net keeps groups of correlated features together (either all in or all out), which is more stable and often more interpretable.

### Shrinkage and the bias-variance tradeoff

Regularization introduces bias (the coefficients are systematically smaller than their true values) in exchange for reduced variance (the coefficients are more stable across different samples). This is the fundamental bias-variance tradeoff:

- **No regularization ($\lambda = 0$)**: Low bias, high variance. Coefficients are unbiased but noisy.
- **Heavy regularization (large $\lambda$)**: High bias, low variance. Coefficients are stable but systematically shrunk.
- **Optimal $\lambda$**: The sweet spot that minimizes total prediction error (bias^2 + variance).

*Analogy*: Think of shooting arrows at a target. Without regularization, your arrows cluster tightly around the bullseye on average but scatter widely (high variance). With too much regularization, your arrows cluster tightly together but consistently miss the center (high bias). The right amount of regularization gives you the best overall accuracy.

### Regularization paths

A regularization path shows how each coefficient changes as $\lambda$ increases from 0 to a large value. For Lasso, you see coefficients gradually shrinking and then hitting zero one by one. The order in which they hit zero tells you about their relative importance.

Plotting the regularization path is a powerful diagnostic tool. It shows you:
- Which features are most important (they survive the longest as $\lambda$ increases).
- How stable the coefficients are (stable features have smooth paths).
- A natural ordering of features by importance.

### Choosing $\lambda$ with cross-validation

You do not pick $\lambda$ by hand. Instead, you use cross-validation:

1. Define a grid of $\lambda$ values (e.g., 100 values from very small to very large, usually on a log scale).
2. For each $\lambda$, run k-fold cross-validation and compute the average validation error.
3. Pick the $\lambda$ that minimizes validation error.

Two common choices:
- **lambda.min**: The $\lambda$ with the lowest cross-validation error.
- **lambda.1se**: The largest $\lambda$ within one standard error of the minimum. This gives a simpler model (more regularization) that is nearly as good. In actuarial work, where interpretability matters, lambda.1se is often preferred.

## Key concepts

### Ridge regression

$$\hat{\beta}_{\text{Ridge}} = \arg\min_{\beta} \left[ \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2 \right]$$

Closed-form solution:

$$\hat{\beta}_{\text{Ridge}} = (X^TX + \lambda I)^{-1}X^Ty$$

The term $\lambda I$ makes the matrix invertible even when $X^TX$ is singular (which happens with multicollinearity or when $p > n$). This is one of Ridge's key advantages.

### Lasso regression

$$\hat{\beta}_{\text{Lasso}} = \arg\min_{\beta} \left[ \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1 \right]$$

No closed-form solution. Solved using coordinate descent or LARS (Least Angle Regression) algorithm.

### Elastic Net

$$\hat{\beta}_{\text{EN}} = \arg\min_{\beta} \left[ \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \left( \alpha\|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2 \right) \right]$$

### Important properties

| Property | Ridge (L2) | Lasso (L1) | Elastic Net |
|---|---|---|---|
| Feature selection | No (shrinks, never zeros) | Yes (sets some to exactly zero) | Yes (from L1 component) |
| Correlated features | Keeps all, shares weight | Picks one arbitrarily | Keeps groups together |
| Computational | Closed-form solution | Iterative (coordinate descent) | Iterative |
| Number of selected features | Always p | At most n (for p > n) | Can select more than n |
| Stability with correlations | High | Low | Medium-High |

### Standardization requirement

Regularization penalizes large coefficients. If features are on different scales (e.g., age in years vs. income in dollars), the penalty will disproportionately affect the large-scale feature. **Always standardize features before applying regularization**:

$$x_j^{\text{standardized}} = \frac{x_j - \bar{x}_j}{s_{x_j}}$$

After fitting, you can transform coefficients back to the original scale for interpretation.

### Regularization in GLMs

For actuarial pricing, regularized GLMs extend the concept to Poisson, Gamma, and Tweedie distributions:

$$\hat{\beta} = \arg\min_{\beta} \left[ -\text{log-likelihood}(\beta) + \lambda \cdot \text{penalty}(\beta) \right]$$

The `glmnet` library handles this for the exponential family. In Python, `scikit-learn`'s `SGDClassifier` and `SGDRegressor` support regularized GLMs, and the `glum` package provides regularized GLMs closer to the R `glmnet` experience.

## Python implementation

```python
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

# --- Always standardize first ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# --- Ridge with cross-validated lambda ---
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5)
ridge_cv.fit(X_scaled, y_train)
print(f"Best alpha (lambda): {ridge_cv.alpha_:.4f}")

# --- Lasso with cross-validated lambda ---
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5, random_state=42)
lasso_cv.fit(X_scaled, y_train)
print(f"Best alpha: {lasso_cv.alpha_:.4f}")
print(f"Non-zero coefficients: {np.sum(lasso_cv.coef_ != 0)} out of {len(lasso_cv.coef_)}")

# --- Elastic Net with cross-validated alpha and l1_ratio ---
enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
                        alphas=np.logspace(-4, 1, 50),
                        cv=5, random_state=42)
enet_cv.fit(X_scaled, y_train)
print(f"Best alpha: {enet_cv.alpha_:.4f}, Best l1_ratio: {enet_cv.l1_ratio_:.2f}")

# --- Pipeline (recommended) ---
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', LassoCV(cv=5, random_state=42))
])
pipe.fit(X_train, y_train)

# --- Regularization path plot ---
from sklearn.linear_model import lasso_path

alphas, coefs, _ = lasso_path(X_scaled, y_train, alphas=np.logspace(-4, 1, 100))
fig, ax = plt.subplots(figsize=(10, 6))
for i in range(coefs.shape[0]):
    ax.plot(np.log10(alphas), coefs[i], label=feature_names[i] if i < 10 else None)
ax.set_xlabel('log10(alpha)')
ax.set_ylabel('Coefficient value')
ax.set_title('Lasso Regularization Path')
ax.axvline(np.log10(lasso_cv.alpha_), color='black', linestyle='--', label='CV optimal')
ax.legend(loc='best', fontsize=8)
plt.show()
```

## R implementation (if applicable)

```r
library(glmnet)

# --- Standardization happens automatically in glmnet ---

# Ridge regression (alpha = 0)
cv_ridge <- cv.glmnet(X_matrix, y, alpha = 0, nfolds = 10)
plot(cv_ridge)  # CV error vs lambda
coef(cv_ridge, s = "lambda.1se")

# Lasso regression (alpha = 1)
cv_lasso <- cv.glmnet(X_matrix, y, alpha = 1, nfolds = 10)
plot(cv_lasso)
selected <- coef(cv_lasso, s = "lambda.1se")
cat("Selected features:", sum(selected != 0) - 1, "\n")  # -1 for intercept

# Elastic Net (alpha between 0 and 1)
# Search over alpha values
alpha_grid <- seq(0.1, 0.9, by = 0.1)
cv_results <- lapply(alpha_grid, function(a) {
  cv.glmnet(X_matrix, y, alpha = a, nfolds = 10)
})
best_alpha <- alpha_grid[which.min(sapply(cv_results, function(m) min(m$cvm)))]

# Regularization path plot
fit_lasso <- glmnet(X_matrix, y, alpha = 1)
plot(fit_lasso, xvar = "lambda", label = TRUE)

# For Poisson GLM (actuarial frequency model)
cv_poisson <- cv.glmnet(X_matrix, y_counts, alpha = 0.5, family = "poisson",
                         offset = log(exposure))
```

## Gotchas & lessons learned

1. **Forgetting to standardize**: This is the number one mistake. If you do not standardize features before regularization, the penalty treats features unequally based on their scale, not their importance. A feature measured in millions will be penalized much more than one measured in single digits. Note: `glmnet` in R standardizes internally by default; scikit-learn does not -- you must do it explicitly.

2. **Confusing alpha and lambda**: In scikit-learn, the regularization strength parameter is called `alpha`. In `glmnet` (R), it is called `lambda`, while `alpha` controls the L1/L2 mix. This naming inconsistency causes endless confusion. Always check which library you are using.

3. **Lambda.min vs lambda.1se**: `lambda.min` gives the best CV error but often selects a complex model. `lambda.1se` gives a simpler model within one standard error of the best. For actuarial models where you need to justify every variable, `lambda.1se` is usually the better choice.

4. **Lasso is unstable with correlated features**: If two features are highly correlated, Lasso will randomly pick one and zero out the other. The choice can flip between runs or CV folds. If you care about which features are selected (not just prediction accuracy), use Elastic Net.

5. **Regularization does not fix bad features**: Regularization can shrink or eliminate useless features, but it cannot create signal where none exists. Good feature engineering comes first.

6. **Do not regularize the intercept**: Most implementations exclude the intercept from the penalty by default (and they should). Penalizing the intercept would shift your predictions systematically.

7. **Interpreting Lasso coefficients**: When Lasso sets a coefficient to zero, it does not mean the feature is truly unimportant -- it might just be correlated with another feature that was kept. Conversely, features that survive Lasso are genuinely useful conditional on the others.

8. **Regularization strength depends on sample size**: The optimal $\lambda$ changes with your sample size. As you get more data, you can usually afford less regularization (smaller $\lambda$). Always re-tune $\lambda$ when your data changes substantially.

9. **The grouping effect**: Ridge keeps correlated features with similar coefficients. Lasso picks one and drops the rest. Elastic Net provides a middle ground -- it tends to select groups of correlated features together. Choose based on whether you want sparsity (Lasso), stability (Ridge), or both (Elastic Net).

10. **Regularization in actuarial GLMs**: When using regularized GLMs for pricing, remember that the regularization penalizes deviation from zero (or the grand mean). This means the model shrinks rate relativities toward 1.0, which is a form of credibility weighting. Small groups with little data get more shrinkage -- exactly what actuarial credibility theory prescribes.

## Resources

- **An Introduction to Statistical Learning (ISLR), Chapter 6**: https://www.statlearning.com/ -- clear, accessible treatment of Ridge, Lasso, and Elastic Net with geometric intuition.
- **Elements of Statistical Learning (ESL), Chapter 3**: https://hastie.su.domains/ElemStatLearn/ -- more technical but definitive.
- **glmnet vignette**: https://glmnet.stanford.edu/articles/glmnet.html -- the canonical reference for regularized GLMs.
- **scikit-learn linear models**: https://scikit-learn.org/stable/modules/linear_model.html -- Ridge, Lasso, ElasticNet documentation.
- **Regularization in Actuarial Pricing (CAS)**: https://www.casact.org/ -- look for monographs on predictive modeling that discuss regularization in GLMs.
- **Tibshirani (1996), "Regression Shrinkage and Selection via the Lasso"**: The original Lasso paper.
- **Zou & Hastie (2005), "Regularization and Variable Selection via the Elastic Net"**: The original Elastic Net paper.
