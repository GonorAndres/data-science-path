# Tweedie and Compound Poisson-Gamma Models

## What it is

The Tweedie distribution is a family of probability distributions that can model data with an exact mass of zeros combined with a continuous positive distribution for non-zero values. In insurance, this maps perfectly to claims data: most policyholders in a given period have zero claims (the probability mass at zero), while those who do have claims incur continuous positive costs (the positive distribution). The Tweedie distribution with power parameter between 1 and 2 is equivalent to a **Compound Poisson-Gamma** model, where the number of claims follows a Poisson distribution and each individual claim amount follows a Gamma distribution. This makes it the natural single-distribution solution for modeling aggregate insurance losses (total claims cost per policyholder), avoiding the need to model frequency and severity separately.

## Why it matters

- **Insurance claims are zero-inflated and continuous**: The vast majority of policyholders have zero claims in any given year, and those who do have claims incur a continuous positive amount. Standard distributions (normal, Gamma, Poisson) cannot model this data structure. Tweedie can.
- **One model instead of two**: The traditional actuarial approach models frequency (Poisson) and severity (Gamma) separately, then combines them. Tweedie provides an alternative: model the total (pure premium) directly. This simplifies the modeling pipeline while preserving the frequency-severity interpretation.
- **GLM compatibility**: Tweedie is a member of the exponential dispersion family, so it fits naturally into the Generalized Linear Model (GLM) framework. This means standard GLM tools, diagnostics, and software apply.
- **Industry adoption**: Tweedie GLMs are widely used in P&C insurance pricing, especially for personal lines (auto, home). Many regulatory filings use Tweedie-based models.
- **Actuarial exam relevance**: The compound Poisson model and its connection to aggregate loss distributions are core exam topics (SOA Exam STAM, CAS Exam MAS-I).

## When to use it

- You are modeling aggregate (total) insurance losses per policyholder or per policy, where many observations are zero and the rest are positive continuous values.
- You want a single GLM for pure premium (loss cost) instead of separate frequency and severity models.
- Your data has a point mass at zero with a right-skewed continuous distribution for positive values.
- You are building a pricing model for P&C insurance (auto, home, commercial lines) and want a parsimonious approach.
- You need to model any outcome that is zero-inflated and continuous (e.g., total rainfall, total medical costs, total energy consumption).

## How it works (intuitive)

### The frequency-severity decomposition

The traditional actuarial approach to modeling aggregate losses is:

1. **Frequency model**: How many claims will this policyholder have? Model with Poisson (or Negative Binomial).
   $$N \sim \text{Poisson}(\lambda)$$

2. **Severity model**: Given a claim occurs, how large will it be? Model with Gamma (or lognormal, Pareto).
   $$Y_j \sim \text{Gamma}(\alpha, \beta) \quad \text{for each claim } j$$

3. **Aggregate loss**: The total loss is the sum of individual claim amounts:
   $$S = \sum_{j=1}^{N} Y_j$$

If $N = 0$, then $S = 0$. If $N > 0$, then $S$ is a sum of Gamma random variables.

This $S$ follows a **Compound Poisson-Gamma** distribution. The remarkable result is that this compound distribution is exactly a Tweedie distribution with power parameter $p$ between 1 and 2.

### An analogy: the rain model

Think of daily rainfall in a region:
- On most days, it does not rain at all (zero rainfall). This is the "frequency" component: whether or not a rain event occurs.
- On days when it does rain, the amount of rainfall is a positive continuous quantity. This is the "severity" component: how much rain falls.
- Total monthly rainfall is the sum across all rainy days in the month. Some months have zero rain; others have varying positive amounts.

Tweedie models this total directly, without needing to separately model "will it rain?" and "if it rains, how much?"

### The Tweedie family

The Tweedie distribution is parameterized by:

| Parameter | Symbol | Range | Meaning |
|-----------|--------|-------|---------|
| Mean | $\mu$ | $\mu > 0$ | Expected value of the outcome |
| Dispersion | $\phi$ | $\phi > 0$ | Controls the spread/variance |
| Power | $p$ | $1 < p < 2$ for insurance | Determines the variance-mean relationship |

The **variance function** is:

$$\text{Var}(Y) = \phi \cdot \mu^p$$

This single formula encompasses many familiar distributions:

| Power $p$ | Distribution | Variance | Insurance use |
|-----------|-------------|----------|---------------|
| $p = 0$ | Normal | $\phi$ (constant) | N/A |
| $p = 1$ | Poisson | $\phi \cdot \mu$ | Claim counts |
| $1 < p < 2$ | **Compound Poisson-Gamma (Tweedie)** | $\phi \cdot \mu^p$ | **Aggregate losses (pure premium)** |
| $p = 2$ | Gamma | $\phi \cdot \mu^2$ | Claim severity |
| $p = 3$ | Inverse Gaussian | $\phi \cdot \mu^3$ | Occasionally used for severity |

The "insurance sweet spot" is $1 < p < 2$, which produces a distribution with a point mass at zero and a continuous right-skewed positive part -- exactly what insurance loss data looks like.

### Connection to compound Poisson-Gamma

When $1 < p < 2$, the Tweedie distribution is a compound Poisson-Gamma with:

$$\text{Poisson rate: } \lambda = \frac{\mu^{2-p}}{\phi(2-p)}$$

$$\text{Gamma shape: } \alpha = \frac{2-p}{p-1}$$

$$\text{Gamma scale: } \beta = \phi(p-1)\mu^{p-1}$$

This means:
- $p$ close to 1: The distribution behaves more like Poisson (many small claims). The Gamma shape parameter is large, meaning individual claim sizes are concentrated near their mean.
- $p$ close to 2: The distribution behaves more like Gamma (fewer, larger claims). The Gamma shape parameter is small, meaning individual claim sizes are highly variable.
- The probability of zero is: $P(S = 0) = P(N = 0) = e^{-\lambda}$

## Key concepts

### Tweedie GLM

In a GLM for pure premium:

$$\log(E[S_i]) = \mathbf{x}_i^T \boldsymbol{\beta}$$

$$S_i \sim \text{Tweedie}(\mu_i, \phi, p)$$

where:
- $S_i$ is the aggregate loss for policyholder $i$
- $\mathbf{x}_i$ are covariates (age, vehicle type, territory, etc.)
- $\boldsymbol{\beta}$ are the regression coefficients
- The log link is standard, ensuring positive predicted values
- $\phi$ and $p$ are estimated from the data (or $p$ can be fixed)

### Estimating the power parameter $p$

The power parameter determines the shape of the distribution. Methods for selecting $p$:

1. **Profile likelihood**: Fit the Tweedie GLM at a grid of $p$ values (e.g., 1.1, 1.2, ..., 1.9) and select the $p$ that maximizes the log-likelihood. This is the most principled approach.

2. **Variance-mean relationship**: Plot $\log(\text{Var}(Y))$ vs. $\log(\text{mean}(Y))$ across groups. The slope estimates $p$.

3. **Rule of thumb**: For auto insurance, $p$ typically falls between 1.5 and 1.8. For property insurance, it may be closer to 1.3-1.6 due to higher frequency and lower severity relative to auto.

4. **Cross-validation**: Select $p$ that minimizes out-of-sample prediction error.

### Tweedie vs. separate frequency-severity models

| Feature | Tweedie GLM | Separate Freq/Sev |
|---------|------------|-------------------|
| Number of models | 1 | 2 (or 3 with zero-inflation) |
| Exposure handling | Offset in single model | Offset in frequency model |
| Covariate effects | Joint effect on pure premium | Separate effects on frequency and severity; can differ |
| Interpretation | Effect on total loss cost | Separate effects on "how often" and "how much" |
| Interactions | Implicit (through joint model) | Can have different interactions in freq vs. sev |
| Computational cost | Lower | Higher (two models to fit, maintain, and validate) |
| Flexibility | Less (assumes freq/sev link through $p$) | More (each model is fully independent) |
| When to prefer | Simpler models, faster iteration, aggregate loss focus | Deep understanding of freq/sev drivers, complex tail modeling |

In practice, many actuaries fit both and compare. The Tweedie model is excellent for initial analysis and screening; separate models provide more insight when frequency and severity have different drivers.

### Exposure handling

In insurance, exposure varies across policies (some are in force for a full year, some for 6 months). In a Tweedie GLM, exposure enters as an **offset**:

$$\log(E[S_i]) = \log(\text{exposure}_i) + \mathbf{x}_i^T \boldsymbol{\beta}$$

This ensures the model predicts loss cost *per unit of exposure* (e.g., per car-year).

### Aggregate loss distribution and moments

For a compound Poisson-Gamma:

$$E[S] = \lambda \cdot E[Y] = \lambda \cdot \alpha \beta$$

$$\text{Var}(S) = \lambda \cdot E[Y^2] = \lambda \cdot \alpha\beta^2(\alpha + 1)$$

$$\text{Skewness}(S) = \frac{E[Y^3]}{\lambda^{1/2} (E[Y^2])^{3/2}}$$

The distribution is always right-skewed. Higher frequency ($\lambda$) makes the distribution more symmetric (by the CLT, the sum of many claims approaches normality). Lower frequency makes the zero mass larger and the distribution more extreme.

### Tweedie density

The Tweedie density does not have a simple closed form for $1 < p < 2$. It is defined as a mixture:

$$f(y; \mu, \phi, p) = \begin{cases} e^{-\lambda} & \text{if } y = 0 \\ \sum_{n=1}^{\infty} \frac{e^{-\lambda} \lambda^n}{n!} \cdot g(y; n\alpha, \beta) & \text{if } y > 0 \end{cases}$$

where $g(y; n\alpha, \beta)$ is the Gamma density with shape $n\alpha$ and scale $\beta$, and the sum is over the number of claims $n$.

In practice, likelihood evaluation uses series approximations or saddlepoint methods.

## Python implementation

**Key libraries:**
- `statsmodels.genmod.families.Tweedie`: Tweedie GLM within statsmodels.
- `sklearn.linear_model.TweedieRegressor`: Scikit-learn implementation (simpler interface, good for ML pipelines).
- `lightgbm` / `xgboost`: Both support Tweedie loss for gradient boosted trees.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import cross_val_score

# --- scikit-learn Tweedie GLM ---
# Fit a Tweedie GLM for pure premium
model = TweedieRegressor(
    power=1.6,           # power parameter (1 < p < 2 for insurance)
    alpha=0.1,           # regularization strength (0 = no regularization)
    link='log',          # log link (standard for insurance)
    max_iter=1000,
)
model.fit(X_train, y_train, sample_weight=exposure_train)
predictions = model.predict(X_test)

# Profile likelihood: select optimal power parameter
from sklearn.metrics import mean_tweedie_deviance

best_p, best_score = None, float('inf')
for p in np.arange(1.1, 2.0, 0.1):
    mdl = TweedieRegressor(power=p, alpha=0.1, link='log', max_iter=1000)
    mdl.fit(X_train, y_train, sample_weight=exposure_train)
    preds = mdl.predict(X_test)
    deviance = mean_tweedie_deviance(y_test, preds, sample_weight=exposure_test, power=p)
    print(f"p={p:.1f}, Tweedie deviance={deviance:.4f}")
    if deviance < best_score:
        best_score = deviance
        best_p = p

print(f"Best power parameter: {best_p:.1f}")
```

```python
# --- statsmodels Tweedie GLM ---
import statsmodels.api as sm
from statsmodels.genmod.families import Tweedie
from statsmodels.genmod.families.links import Log

# Tweedie GLM with statsmodels
model = sm.GLM(
    y_train,
    sm.add_constant(X_train),
    family=Tweedie(var_power=1.6, link=Log()),
    exposure=exposure_train,
)
result = model.fit()
print(result.summary())
```

```python
# --- LightGBM with Tweedie loss ---
import lightgbm as lgb

train_data = lgb.Dataset(X_train, label=y_train, weight=exposure_train)
params = {
    'objective': 'tweedie',
    'tweedie_variance_power': 1.6,
    'metric': 'tweedie',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'verbose': -1,
}
gbm = lgb.train(params, train_data, num_boost_round=500)
predictions = gbm.predict(X_test)
```

```python
# --- Compound Poisson-Gamma simulation ---
def simulate_tweedie(mu, phi, p, n_samples):
    """
    Simulate from a Tweedie (Compound Poisson-Gamma) distribution.
    """
    # Derive compound Poisson-Gamma parameters
    lam = mu**(2 - p) / (phi * (2 - p))    # Poisson rate
    alpha = (2 - p) / (p - 1)               # Gamma shape
    beta = phi * (p - 1) * mu**(p - 1)      # Gamma scale

    samples = np.zeros(n_samples)
    for i in range(n_samples):
        n_claims = np.random.poisson(lam)
        if n_claims > 0:
            claim_amounts = np.random.gamma(shape=alpha, scale=beta, size=n_claims)
            samples[i] = claim_amounts.sum()

    return samples

# Example: simulate 10,000 policy-level losses
losses = simulate_tweedie(mu=500, phi=10, p=1.6, n_samples=10_000)
print(f"Fraction of zeros: {(losses == 0).mean():.3f}")
print(f"Mean (non-zero): {losses[losses > 0].mean():.1f}")
print(f"Overall mean: {losses.mean():.1f}")
```

## R implementation (if applicable)

R has excellent support for Tweedie models:

```r
# tweedie package: density, distribution, and parameter estimation
library(tweedie)
library(statmod)

# Fit Tweedie GLM
library(stats)
model <- glm(aggregate_loss ~ age + territory + vehicle_type,
             data = claims_data,
             family = tweedie(var.power = 1.6, link.power = 0),  # log link
             weights = exposure)
summary(model)

# Profile likelihood for power parameter
library(tweedie)
p_profile <- tweedie.profile(
  aggregate_loss ~ age + territory + vehicle_type,
  data = claims_data,
  p.vec = seq(1.1, 1.9, by = 0.1),
  weights = claims_data$exposure,
  do.plot = TRUE
)
print(p_profile$p.max)  # Optimal power parameter

# cpglm: Compound Poisson GLM (alternative implementation)
library(cplm)
cpglm_fit <- cpglm(aggregate_loss ~ age + territory + vehicle_type,
                    data = claims_data,
                    link = "log")
summary(cpglm_fit)

# Bayesian Compound Poisson GLM
bcplm_fit <- bcplm(aggregate_loss ~ age + territory + vehicle_type,
                    data = claims_data,
                    link = "log",
                    n.chains = 2, n.iter = 5000)
summary(bcplm_fit)
```

## Gotchas & lessons learned

1. **Power parameter sensitivity**: Small changes in $p$ can noticeably affect predictions, especially for the probability of zero and the tail behavior. Always perform a profile likelihood analysis or cross-validation to select $p$ rather than guessing.

2. **Tweedie GLM cannot model different covariates for frequency and severity**: In a separate frequency-severity approach, you can have different predictors for each model. Tweedie forces the same predictors to affect both components jointly. If frequency and severity have very different drivers, separate models may be preferable.

3. **Log link and zeros**: The log link ensures positive predictions, which is essential for loss costs. However, the Tweedie distribution handles exact zeros in the response variable -- unlike the Gamma distribution, which requires strictly positive values.

4. **Exposure as an offset, not a weight**: In a Tweedie GLM for pure premium, exposure enters as an offset (log exposure is added to the linear predictor). Using exposure as a sample weight instead is incorrect and will produce biased estimates. However, some software implementations (scikit-learn) use weights instead of offsets, which works if you model loss per unit of exposure as the response.

5. **Regularization helps**: Insurance pricing data often has many categorical variables (territory, vehicle make/model). Without regularization, the Tweedie GLM can overfit to sparse categories. Use L1 (Lasso) or L2 (Ridge) regularization.

6. **Comparison with gradient boosted trees**: LightGBM and XGBoost both support Tweedie loss. These often outperform Tweedie GLMs in predictive accuracy but are harder to interpret and explain to regulators. Use Tweedie GLMs for regulatory filings; use boosted models for internal analysis.

7. **Deviance residuals for model checking**: Use Tweedie deviance residuals (not raw residuals) for diagnostic plots. They should be approximately normally distributed if the model is well-specified.

8. **The Tweedie distribution is not unique for zero-inflated data**: Zero-inflated models (ZIP, ZINB -- see that README) are an alternative. The key difference: Tweedie generates zeros through the compound Poisson mechanism (zero claims), while zero-inflated models add a separate zero-generating process (structural zeros). The choice depends on whether you believe the zeros have a single explanation (no claims) or multiple explanations (no claims + not at risk).

9. **Computational considerations**: Tweedie log-likelihood evaluation uses series approximations, which can be slow for very large datasets. The scikit-learn and LightGBM implementations are faster than statsmodels for large-scale problems.

10. **Dispersion parameter estimation**: In statsmodels, the dispersion parameter $\phi$ is estimated after the regression coefficients (not jointly). This is standard for quasi-likelihood estimation but can be less efficient than joint estimation. The `cplm` R package does joint estimation.

## Resources

- [Dunn & Smyth (2018) -- Generalized Linear Models with Examples in R](https://link.springer.com/book/10.1007/978-1-4419-0118-7) -- Excellent treatment of Tweedie and other GLM families.
- [Jorgensen (1997) -- The Theory of Dispersion Models](https://www.taylorfrancis.com/books/mono/10.1201/9781315136271/theory-dispersion-models-bent-j%C3%B8rgensen) -- The theoretical foundation of Tweedie distributions.
- [Smyth & Jorgensen (2002) -- Fitting Tweedie's Compound Poisson Model to Insurance Claims Data](https://www.jstor.org/) -- Practical guide for insurance applications.
- [tweedie R package](https://cran.r-project.org/web/packages/tweedie/) -- Tweedie density, distribution, and profile likelihood.
- [cplm R package](https://cran.r-project.org/web/packages/cplm/) -- Compound Poisson linear and mixed models.
- [scikit-learn TweedieRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html) -- Python implementation.
- [LightGBM Tweedie regression](https://lightgbm.readthedocs.io/en/latest/Parameters.html) -- Gradient boosting with Tweedie loss.
- [XGBoost Tweedie regression](https://xgboost.readthedocs.io/) -- Alternative gradient boosting implementation.
- [Frees, Derrig, Meyers -- Predictive Modeling Applications in Actuarial Science](https://www.cambridge.org/core/books/predictive-modeling-applications-in-actuarial-science/7828CFE4A73F59B1AC5A11DC9FF3B78C) -- Tweedie GLMs in insurance pricing context.
- SOA/CAS research papers on frequency-severity vs. aggregate loss modeling.
