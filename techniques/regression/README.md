# Regression (Linear, Ridge, Lasso, ElasticNet)

## What it is

Regression is the foundational supervised learning technique for predicting a **continuous numeric outcome** from one or more input features. Ordinary Least Squares (OLS) linear regression fits a straight line (or hyperplane) through the data by minimizing the sum of squared errors between predicted and actual values. Ridge, Lasso, and ElasticNet are **regularized** extensions that add penalty terms to the OLS objective, shrinking coefficient estimates to improve generalization and handle multicollinearity or high-dimensional data. Together, these four variants form the backbone of predictive modeling -- understanding them deeply makes every other regression-based method (GLMs, time-series models, boosting) far more intuitive.

## Why it matters

For an actuarial or data science practitioner, regression is the first tool you reach for. Pricing models, reserving projections, demand forecasts, and risk scores often start as regression problems. Regularized regression is especially important when you have many correlated predictors (e.g., dozens of policyholder features) and need a model that generalizes rather than memorizing noise. The bias-variance tradeoff that Ridge and Lasso expose is a concept you will use in every modeling decision you ever make, regardless of technique.

## When to use it

- **OLS Linear Regression**: Small-to-moderate number of features, no severe multicollinearity, when interpretability and coefficient inference matter (hypothesis testing, confidence intervals).
- **Ridge Regression**: Many correlated features that you want to keep in the model (e.g., region dummies, polynomial terms). You suspect multicollinearity is inflating coefficient variance.
- **Lasso Regression**: You have many features and suspect most are irrelevant. You want automatic feature selection (Lasso drives irrelevant coefficients exactly to zero).
- **ElasticNet**: You have many correlated features AND you want some feature selection. This combines the strengths of Ridge and Lasso.
- **General guidance**: Use regularized regression whenever `p` (features) is large relative to `n` (observations), or when you care more about prediction accuracy than individual coefficient interpretation.

## How it works (intuitive)

### OLS Linear Regression

Imagine you have a scatter plot of house prices (y-axis) versus square footage (x-axis). OLS draws the single straight line that minimizes the total squared vertical distance from every data point to the line. "Squared" means big errors get penalized disproportionately -- a prediction that is off by 10 costs 100, while being off by 2 costs only 4.

With multiple features, the "line" becomes a plane or hyperplane, but the principle is identical: find the surface that minimizes squared errors.

**Step by step:**

1. Assume the relationship is $y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p + \epsilon$.
2. Collect your data into a matrix $X$ (features) and vector $y$ (target).
3. Solve for the coefficient vector $\hat{\beta}$ that minimizes $\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$.
4. The closed-form solution is $\hat{\beta} = (X^T X)^{-1} X^T y$.

### The problem: overfitting and multicollinearity

When you have many features, OLS can overfit -- it finds coefficients that perfectly explain the training data noise but predict poorly on new data. When features are correlated (e.g., "total claim amount" and "number of claims"), the matrix $X^T X$ becomes nearly singular, and coefficient estimates become wildly unstable.

**Analogy**: Imagine fitting a polynomial through 10 data points. A degree-9 polynomial passes through every single point (zero training error) but wiggles crazily between them. That polynomial has **low bias** (it captures the pattern) but **high variance** (a slightly different sample would produce a totally different curve). Regularization deliberately introduces a small amount of bias to dramatically reduce variance.

### Ridge Regression (L2 penalty)

Ridge adds a penalty proportional to the **sum of squared coefficients**:

$$\text{Minimize } \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2$$

The hyperparameter $\lambda$ controls how strongly we penalize large coefficients. As $\lambda$ increases, coefficients shrink toward zero (but never exactly reach zero). Think of it as putting a budget on the total "magnitude" of your coefficients -- the model has to spread its explanatory power more evenly.

**Analogy**: You are packing a suitcase with a weight limit. You can bring everything, but you have to bring less of each item. Ridge does not remove any item entirely; it just reduces the amount of each.

### Lasso Regression (L1 penalty)

Lasso replaces the squared penalty with an **absolute-value penalty**:

$$\text{Minimize } \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j|$$

The geometry of the L1 penalty (a diamond shape in coefficient space) means solutions tend to land at corners, which forces some coefficients to be **exactly zero**. This is automatic feature selection.

**Analogy**: Now your suitcase has a weight limit AND you want to travel light. You decide to leave some items behind entirely (zero weight) rather than bringing tiny amounts of everything.

### ElasticNet

ElasticNet combines both penalties:

$$\text{Minimize } \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \left[ \alpha \sum_{j=1}^{p} |\beta_j| + \frac{(1-\alpha)}{2} \sum_{j=1}^{p} \beta_j^2 \right]$$

Where $\alpha \in [0, 1]$ controls the mix: $\alpha = 1$ is pure Lasso, $\alpha = 0$ is pure Ridge. ElasticNet is especially useful when features come in correlated groups -- Lasso might arbitrarily pick one feature from the group, while ElasticNet tends to keep or drop the group together.

## Key concepts

### OLS Assumptions (the CLRM conditions)

1. **Linearity**: The true relationship between features and target is linear (or can be made linear via transformations).
2. **Independence**: Observations are independent of each other.
3. **Homoscedasticity**: The variance of residuals is constant across all levels of the predictors.
4. **Normality of residuals**: Residuals are approximately normally distributed (required for valid confidence intervals and p-values, but less critical for prediction).
5. **No perfect multicollinearity**: No feature is an exact linear combination of others.
6. **Exogeneity**: $E[\epsilon | X] = 0$ -- the error term is not correlated with the predictors.

### Bias-Variance Tradeoff

| Component | What it means | Effect of regularization |
|-----------|---------------|--------------------------|
| **Bias** | Systematic error from simplifying assumptions | Increases with stronger penalty |
| **Variance** | Sensitivity of the model to the specific training sample | Decreases with stronger penalty |
| **Total Error** | Bias^2 + Variance + Irreducible Noise | Minimized at an optimal penalty strength |

The sweet spot is the value of $\lambda$ that minimizes total prediction error (typically found via cross-validation).

### Key Formulas

| Metric | Formula | Purpose |
|--------|---------|---------|
| R-squared | $R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained |
| Adjusted R-squared | $\bar{R}^2 = 1 - \frac{(1-R^2)(n-1)}{n-p-1}$ | R-squared penalized for number of features |
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Average squared prediction error |
| RMSE | $\sqrt{MSE}$ | Same units as y, easier to interpret |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Average absolute error, robust to outliers |

### Regularization Path

As you sweep $\lambda$ from 0 to infinity:
- At $\lambda = 0$: You get the OLS solution.
- At small $\lambda$: Coefficients shrink slightly. Variance drops more than bias increases. Test error improves.
- At optimal $\lambda$: Test error is minimized.
- At large $\lambda$: All coefficients approach zero. The model underfits. Bias dominates.

Plotting coefficients vs. $\lambda$ (the "regularization path") is one of the most informative diagnostic plots in regression analysis.

### Choosing Lambda via Cross-Validation

Use k-fold cross-validation (typically k=5 or k=10):

1. Split data into k folds.
2. For each candidate $\lambda$, train on k-1 folds, evaluate on the held-out fold.
3. Average the error across folds.
4. Select the $\lambda$ that minimizes mean CV error (or use the "one-standard-error rule": pick the largest $\lambda$ within one SE of the minimum, for a more parsimonious model).

## Python implementation

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import numpy as np

# IMPORTANT: Always scale features before regularized regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# OLS
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)

# Ridge with built-in cross-validation
ridge_cv = RidgeCV(alphas=np.logspace(-4, 4, 50), cv=5)
ridge_cv.fit(X_train_scaled, y_train)
print(f"Best Ridge alpha: {ridge_cv.alpha_}")

# Lasso with built-in cross-validation
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 50), cv=5, max_iter=10000)
lasso_cv.fit(X_train_scaled, y_train)
print(f"Best Lasso alpha: {lasso_cv.alpha_}")
print(f"Features selected: {np.sum(lasso_cv.coef_ != 0)} / {X_train.shape[1]}")

# ElasticNet with built-in cross-validation
enet_cv = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9], cv=5, max_iter=10000)
enet_cv.fit(X_train_scaled, y_train)
print(f"Best alpha: {enet_cv.alpha_}, Best l1_ratio: {enet_cv.l1_ratio_}")
```

**For inference (p-values, confidence intervals)**, use `statsmodels`:

```python
import statsmodels.api as sm

X_with_const = sm.add_constant(X_train)
model = sm.OLS(y_train, X_with_const).fit()
print(model.summary())  # Full inference table
```

## R implementation (if applicable)

```r
# OLS
model_ols <- lm(y ~ ., data = train_data)
summary(model_ols)

# Ridge & Lasso via glmnet
library(glmnet)
X_mat <- model.matrix(y ~ . - 1, data = train_data)
y_vec <- train_data$y

# Ridge (alpha = 0)
cv_ridge <- cv.glmnet(X_mat, y_vec, alpha = 0)
plot(cv_ridge)

# Lasso (alpha = 1)
cv_lasso <- cv.glmnet(X_mat, y_vec, alpha = 1)
plot(cv_lasso)
coef(cv_lasso, s = "lambda.1se")

# ElasticNet (alpha = 0.5)
cv_enet <- cv.glmnet(X_mat, y_vec, alpha = 0.5)
```

## Gotchas & lessons learned

1. **Always scale features before regularization.** Ridge and Lasso penalize the magnitude of coefficients. If features are on different scales (e.g., age in years vs. income in thousands), the penalty unfairly targets the larger-scale feature. Use `StandardScaler` or normalize first.

2. **Do NOT scale features for vanilla OLS inference.** If you want interpretable coefficients ("a one-unit increase in X changes Y by beta"), scaling destroys those units. Scale only for regularized regression.

3. **Lasso can be unstable with correlated features.** If two features are highly correlated, Lasso might pick one and drop the other arbitrarily. The choice can flip between runs or with slightly different data. Use ElasticNet for stability in this scenario.

4. **Check residual plots, not just R-squared.** A high R-squared does not mean your model is valid. Plot residuals vs. fitted values to check for non-linearity and heteroscedasticity. Plot residuals vs. each predictor. Use a Q-Q plot to check normality.

5. **The "one-standard-error rule" is your friend.** When using cross-validation to choose $\lambda$, the minimum-error $\lambda$ often overfits slightly. The one-SE rule picks a simpler model that performs nearly as well.

6. **Multicollinearity inflates standard errors, not bias.** OLS coefficients are still unbiased with multicollinearity, but their confidence intervals become huge, making individual coefficients uninterpretable. This is where Ridge shines.

7. **Watch out for the intercept.** By default, `scikit-learn` regularizes all coefficients except the intercept. This is correct behavior -- you generally do not want to penalize the intercept.

8. **Regularization is not a substitute for feature engineering.** If the true relationship is non-linear, no amount of regularization on a linear model will capture it. Consider polynomial features, interactions, or non-linear models.

9. **In actuarial contexts, interpretability often matters more than marginal accuracy gains.** A Lasso model that selects 8 interpretable features may be preferred over a Ridge model using all 50, even if the Ridge model has slightly lower RMSE.

10. **Always report train AND test metrics.** A large gap between training error and test error is the hallmark of overfitting, which regularization should reduce.

## Decision guide: which variant?

```
Start here
  |
  v
Do you need coefficient inference (p-values, CIs)?
  |-- YES --> Use OLS (statsmodels). Check assumptions carefully.
  |-- NO  --> Continue
  v
Do you have more features than observations (p > n)?
  |-- YES --> Use Lasso or ElasticNet (OLS cannot solve this).
  |-- NO  --> Continue
  v
Are many features likely irrelevant?
  |-- YES --> Use Lasso (automatic feature selection).
  |-- NO  --> Continue
  v
Are features correlated with each other?
  |-- YES --> Use Ridge or ElasticNet.
  |-- NO  --> Use Ridge (safe default for prediction).
```

## Resources

- **scikit-learn documentation**: [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html) -- Excellent reference with mathematical details.
- **An Introduction to Statistical Learning (ISLR)**, Chapter 6 -- The best introductory treatment of regularization and model selection. Free PDF available.
- **The Elements of Statistical Learning (ESL)**, Chapter 3 -- More rigorous treatment. Free PDF available.
- **statsmodels OLS documentation**: [statsmodels.org](https://www.statsmodels.org/stable/regression.html)
- **Hastie & Qian, glmnet vignette**: [glmnet for R](https://glmnet.stanford.edu/) -- The canonical regularized regression package.
- **3Blue1Brown, "Linear Algebra" series**: YouTube playlist that builds geometric intuition for why $(X^T X)^{-1}$ works.
- **Penn State STAT 501**: [Regression Methods](https://online.stat.psu.edu/stat501/) -- Thorough treatment of OLS assumptions and diagnostics.
