# Generalized Linear Models (GLMs)

## What it is

Generalized Linear Models (GLMs) extend ordinary linear regression to handle response variables that follow distributions other than the normal (Gaussian) distribution. While OLS assumes the target is continuous and normally distributed, GLMs can model **counts** (Poisson), **positive continuous amounts** (Gamma), **binary outcomes** (Bernoulli/Binomial), and **mixed count-continuous data** (Tweedie). A GLM has three components: a **random component** (the distribution of the response), a **systematic component** (the linear predictor $\eta = X\beta$), and a **link function** that connects the two. GLMs are the **actuarial workhorse** -- virtually all insurance pricing, frequency modeling, and severity modeling is built on GLMs.

## Why it matters

GLMs are the standard in the insurance industry for a reason: they provide a principled statistical framework for modeling non-normal data with interpretable coefficients, built-in handling of exposure (offsets), and well-understood inference (deviance, chi-squared tests, confidence intervals). Regulatory bodies and actuarial standards boards expect actuaries to understand and justify GLM choices. Even if you ultimately use machine learning methods like XGBoost, you will almost certainly benchmark against a GLM, and many organizations require a GLM as the "official" model. Understanding GLMs deeply -- especially the Poisson, Gamma, and Tweedie families -- is non-negotiable for actuarial data science.

## When to use it

- **Poisson GLM**: Modeling **claim frequency** (count data), event rates, number of customer interactions. When your target is a non-negative integer and you have an exposure variable (policy-years, hours, area).
- **Gamma GLM**: Modeling **claim severity** (positive continuous amounts), average cost per claim, revenue per transaction. When your target is strictly positive and right-skewed.
- **Tweedie GLM**: Modeling **pure premium** (frequency x severity combined), aggregate losses. When your target has a mass of exact zeros and a continuous positive distribution for non-zero values.
- **Binomial/Logistic GLM**: Modeling binary outcomes (lapse, fraud, mortality). This is logistic regression framed as a GLM.
- **Negative Binomial GLM**: Count data with overdispersion (variance exceeds the mean), which Poisson cannot handle.
- **Inverse Gaussian GLM**: Positive continuous data that is more heavily right-skewed than Gamma.

## How it works (intuitive)

### The three components of a GLM

**1. Random Component**: What distribution does the response follow?

In OLS, we assume $y \sim \text{Normal}(\mu, \sigma^2)$. In a GLM, we generalize this to any distribution from the **exponential family**: Normal, Poisson, Gamma, Binomial, Inverse Gaussian, Tweedie.

**Analogy**: Think of the random component as choosing the right "shape" for the noise in your data. If you are modeling claim counts, the noise looks like a Poisson process (discrete, non-negative, variance proportional to mean). If you are modeling claim sizes, the noise looks like a Gamma distribution (positive, right-skewed).

**2. Systematic Component**: The linear predictor.

Just like in linear regression, we combine features linearly:

$$\eta = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p$$

This part looks identical to OLS. The features, interactions, and transformations all enter here.

**3. Link Function**: The bridge between the linear predictor and the mean response.

$$g(\mu) = \eta$$

The link function $g$ transforms the expected value of the response ($\mu$) into something that can be modeled linearly. It ensures that the linear predictor maps to valid values of $\mu$.

**Analogy**: Think of the link function as a translator. The linear predictor speaks the language of real numbers ($-\infty$ to $+\infty$). The response variable speaks a different language (e.g., counts can only be 0, 1, 2, ...; probabilities must be between 0 and 1). The link function translates between these two languages.

### Common link functions

| Distribution | Canonical Link | Link Function $g(\mu)$ | Inverse Link $\mu = g^{-1}(\eta)$ |
|-------------|---------------|------------------------|-----------------------------------|
| Normal | Identity | $g(\mu) = \mu$ | $\mu = \eta$ |
| Poisson | Log | $g(\mu) = \ln(\mu)$ | $\mu = e^{\eta}$ |
| Gamma | Reciprocal | $g(\mu) = 1/\mu$ | $\mu = 1/\eta$ |
| Binomial | Logit | $g(\mu) = \ln(\mu/(1-\mu))$ | $\mu = 1/(1+e^{-\eta})$ |
| Tweedie | Power | $g(\mu) = \mu^{1-p}/(1-p)$ | varies |

**Important note**: You are not forced to use the canonical link. In practice, the **log link** is used with the Gamma distribution far more often than the reciprocal link, because:
- Log link ensures predicted values are always positive.
- Coefficients have a multiplicative interpretation ($e^{\beta}$ is the multiplicative effect).
- This is industry standard in actuarial pricing.

### Fitting a GLM

GLMs are fitted using **Iteratively Reweighted Least Squares (IRLS)**, which is a form of maximum likelihood estimation:

1. Start with initial estimates for $\beta$.
2. Compute predicted values $\hat{\mu}$.
3. Compute working weights and working responses based on the link function and variance function.
4. Solve a weighted least squares problem.
5. Update $\beta$.
6. Repeat until convergence.

This is all handled internally by `statsmodels` or `glm()` in R -- you just specify the family and link.

### The Exponential Dispersion Family

The distributions used in GLMs belong to the exponential dispersion family, with density:

$$f(y; \theta, \phi) = a(y, \phi) \exp\left(\frac{y\theta - b(\theta)}{\phi}\right)$$

Where:
- $\theta$ is the **natural parameter** (related to $\mu$)
- $\phi$ is the **dispersion parameter** (controls spread)
- $b(\theta)$ determines the mean and variance

The **variance function** relates the variance to the mean:

$$\text{Var}(Y) = \phi \cdot V(\mu)$$

| Distribution | Variance Function $V(\mu)$ | Mean-Variance Relationship |
|-------------|---------------------------|---------------------------|
| Normal | $V(\mu) = 1$ | Constant variance |
| Poisson | $V(\mu) = \mu$ | Variance equals mean |
| Gamma | $V(\mu) = \mu^2$ | Variance proportional to mean squared |
| Inverse Gaussian | $V(\mu) = \mu^3$ | Variance proportional to mean cubed |
| Tweedie | $V(\mu) = \mu^p$ | Power of mean, $1 < p < 2$ |

### Actuarial Application: Frequency-Severity Modeling

The standard actuarial approach to pricing:

**Step 1: Model Frequency** (how often claims occur)
- Target: Number of claims per policy
- Distribution: Poisson (or Negative Binomial if overdispersed)
- Link: Log
- Offset: $\ln(\text{exposure})$ -- accounts for different policy durations
- Result: Predicted claim rate per unit of exposure

**Step 2: Model Severity** (how much each claim costs)
- Target: Average cost per claim (conditional on having a claim)
- Distribution: Gamma
- Link: Log
- Data: Only policies with at least one claim
- Result: Predicted average claim cost

**Step 3: Pure Premium**
$$\text{Pure Premium} = \hat{\text{Frequency}} \times \hat{\text{Severity}}$$

**Alternative: Tweedie GLM**
Model the pure premium directly in one step. The Tweedie distribution handles the mix of zeros (no claim) and positive values (claim amounts) naturally. The Tweedie power parameter $p$ controls the shape:
- $p = 1$: Poisson
- $1 < p < 2$: Compound Poisson-Gamma (the sweet spot for insurance)
- $p = 2$: Gamma

## Key concepts

### Deviance

Deviance is the GLM analog of the residual sum of squares. It measures how well the model fits relative to a saturated model (one with a parameter for every observation):

$$D = 2 \sum_{i=1}^{n} \left[ l_i(\text{saturated}) - l_i(\text{fitted}) \right]$$

Where $l_i$ is the log-likelihood contribution of observation $i$.

- **Null deviance**: Deviance of the intercept-only model.
- **Residual deviance**: Deviance of the fitted model.
- **Deviance reduction** ($= \text{Null} - \text{Residual}$): Analogous to the sum of squares explained. Large reduction = good model.

### Model Comparison

- **Likelihood Ratio Test**: Compare nested models by the difference in deviance. Under $H_0$, $D_1 - D_2 \sim \chi^2(df_1 - df_2)$.
- **AIC**: $-2 \ln L + 2k$. Balances fit and complexity. Lower is better.
- **BIC**: $-2 \ln L + k \ln n$. Penalizes complexity more than AIC.

### Offset and Exposure

In frequency modeling, policies have different exposure periods. A policy active for 6 months should be expected to have half the claims of a full-year policy, all else equal.

The offset enters the linear predictor:

$$\ln(\mu) = \beta_0 + \beta_1 x_1 + \ldots + \ln(\text{exposure})$$

This is equivalent to modeling the **rate** (claims per unit exposure) rather than the raw count:

$$\ln(\mu / \text{exposure}) = \beta_0 + \beta_1 x_1 + \ldots$$

### Residuals in GLMs

GLMs have several types of residuals:

| Residual Type | Formula | Use |
|--------------|---------|-----|
| Raw (Response) | $y_i - \hat{\mu}_i$ | Simple but scale depends on $\mu$ |
| Pearson | $(y_i - \hat{\mu}_i) / \sqrt{V(\hat{\mu}_i)}$ | Standardized by variance function |
| Deviance | $\text{sign}(y_i - \hat{\mu}_i)\sqrt{d_i}$ | Approximately normal if model is correct |

Always plot deviance residuals vs. fitted values and vs. each predictor. Look for patterns -- systematic curves indicate missing non-linearity or interactions.

## Python implementation

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd

# --- Poisson GLM for Claim Frequency ---
freq_model = smf.glm(
    'claim_count ~ age + vehicle_age + region + C(gender)',
    data=train_data,
    family=sm.families.Poisson(link=sm.families.links.Log()),
    offset=np.log(train_data['exposure'])
).fit()
print(freq_model.summary())

# Predictions (rate per unit exposure)
train_data['pred_freq'] = freq_model.predict(train_data)

# --- Gamma GLM for Claim Severity ---
claim_data = train_data[train_data['claim_count'] > 0].copy()
sev_model = smf.glm(
    'avg_claim_amount ~ age + vehicle_age + region',
    data=claim_data,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()
print(sev_model.summary())

# --- Tweedie GLM for Pure Premium ---
tweedie_model = smf.glm(
    'total_claim_amount ~ age + vehicle_age + region',
    data=train_data,
    family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.Log()),
    offset=np.log(train_data['exposure'])
).fit()
print(tweedie_model.summary())

# --- Model diagnostics ---
# Deviance residuals
deviance_resid = freq_model.resid_deviance
pearson_resid = freq_model.resid_pearson

# Dispersion estimate (for quasi-models)
pearson_chi2 = np.sum(pearson_resid**2)
dispersion = pearson_chi2 / freq_model.df_resid
print(f"Estimated dispersion: {dispersion:.4f}")
# For Poisson, dispersion should be close to 1. If >> 1, consider Negative Binomial.
```

**Using scikit-learn** (for prediction pipelines, less inference):

```python
from sklearn.linear_model import PoissonRegressor, GammaRegressor, TweedieRegressor

poisson_reg = PoissonRegressor(alpha=0.0, max_iter=1000)
poisson_reg.fit(X_train, y_train, sample_weight=exposure_train)

gamma_reg = GammaRegressor(alpha=0.0, max_iter=1000)
gamma_reg.fit(X_train_claims, y_train_claims)

tweedie_reg = TweedieRegressor(power=1.5, alpha=0.0, max_iter=1000)
tweedie_reg.fit(X_train, y_train, sample_weight=exposure_train)
```

## R implementation (if applicable)

```r
# Poisson GLM for frequency
freq_model <- glm(claim_count ~ age + vehicle_age + region + gender,
                  data = train_data,
                  family = poisson(link = "log"),
                  offset = log(exposure))
summary(freq_model)

# Gamma GLM for severity
sev_model <- glm(avg_claim_amount ~ age + vehicle_age + region,
                 data = claim_data,
                 family = Gamma(link = "log"))
summary(sev_model)

# Tweedie GLM (via tweedie or statmod package)
library(tweedie)
library(statmod)
tweedie_model <- glm(total_claim_amount ~ age + vehicle_age + region,
                     data = train_data,
                     family = tweedie(var.power = 1.5, link.power = 0))

# Estimate optimal Tweedie power parameter
tweedie_profile <- tweedie.profile(
  total_claim_amount ~ age + vehicle_age + region,
  data = train_data,
  p.vec = seq(1.1, 1.9, by = 0.1)
)
```

## Gotchas & lessons learned

1. **Always include exposure as an offset in frequency models.** If you do not account for varying exposure, you are modeling total claims instead of claim rates. A policy in force for 2 years will naturally have more claims than one in force for 3 months.

2. **Use the log link for Gamma, not the canonical reciprocal link.** The reciprocal link can produce negative predicted values and has worse convergence properties. The log link is industry standard and gives multiplicative coefficients.

3. **Check for overdispersion in Poisson models.** Compute the dispersion statistic (Pearson chi-squared / residual df). If it is substantially greater than 1, your Poisson model underestimates uncertainty. Switch to Negative Binomial or use a quasi-Poisson approach.

4. **The Tweedie power parameter matters.** Values between 1 and 2 give compound Poisson-Gamma behavior. Use `tweedie.profile()` in R or grid search in Python to estimate the optimal power.

5. **GLM coefficients with a log link are multiplicative.** A coefficient of 0.15 means that a one-unit increase in that feature multiplies the predicted value by $e^{0.15} \approx 1.16$ (a 16% increase). This is the interpretation actuaries use for rating factors.

6. **Watch out for separation in logistic GLMs.** If a predictor perfectly separates the classes, the MLE does not exist (coefficients diverge to infinity). Use Firth's penalized likelihood or Bayesian methods.

7. **Large claim amounts can dominate Gamma models.** Consider capping or Winsorizing extreme severity values, or use a separate model for large losses.

8. **GLM does not automatically handle non-linearity.** If the relationship between a feature and the response is non-linear (e.g., claim frequency vs. age is U-shaped), you must add polynomial terms, splines, or bin the variable.

9. **Feature engineering is where actuarial expertise shines.** The GLM framework is simple -- the value you add is in knowing which features to include, how to transform them, and which interactions make business sense.

10. **Always validate on holdout data.** In-sample deviance always improves as you add features. Use cross-validation or a time-based holdout to assess true predictive performance.

11. **Understand the difference between `statsmodels` and `scikit-learn` GLM interfaces.** `statsmodels` provides full inference (p-values, confidence intervals, deviance tables) and is what you want for actuarial reporting. `scikit-learn` provides a prediction-oriented interface with regularization, better for ML pipelines.

## Resources

- **McCullagh & Nelder, "Generalized Linear Models" (1989)** -- The definitive GLM textbook. Dense but authoritative.
- **Jong & Heller, "Generalized Linear Models for Insurance Data" (2008)** -- The actuarial-focused GLM book. Highly recommended.
- **Frees, "Regression Modeling with Actuarial and Financial Applications" (2010)** -- Excellent bridge between statistics and actuarial practice.
- **CAS Monograph 5**: "Generalized Linear Models for Insurance Rating" -- Free, covers frequency/severity modeling in detail.
- **statsmodels GLM documentation**: [statsmodels.org/stable/glm.html](https://www.statsmodels.org/stable/glm.html)
- **scikit-learn GLM documentation**: [Generalized Linear Regression](https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression)
- **R `glm()` documentation**: Built into base R, extensively documented.
- **SOA Exam PA (Predictive Analytics)**: Uses GLMs extensively. Study materials are practical and actuarial-focused.
