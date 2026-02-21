# Zero-Inflated Models

## What it is

Zero-inflated models are statistical models designed for count data that has more zeros than a standard count distribution (Poisson or Negative Binomial) can accommodate. The core idea is that the zeros come from **two different sources**: (1) a **structural zero** process -- observations that could never have a positive count (e.g., a policyholder with no vehicle cannot file an auto claim), and (2) a **sampling zero** process -- observations that *could* have a positive count but happened not to (e.g., a driver who owns a car but had no accidents this year). Zero-inflated models explicitly separate these two zero-generating mechanisms into a two-component mixture: a binary model for the structural zeros and a count model for the sampling process. A related approach, the **Hurdle model**, takes a different perspective: it models the probability of *any* count (zero vs. positive) separately from the distribution of positive counts.

## Why it matters

- **Excess zeros are pervasive in insurance data**: Most policyholders file zero claims in any given year. For low-frequency lines (earthquake, flood), the zero rate can exceed 99%. Standard Poisson or Negative Binomial models underestimate the probability of zero and distort all other predictions.
- **Better model fit and predictions**: Ignoring excess zeros leads to biased parameter estimates, inflated standard errors, and poor predictive performance. Zero-inflated models produce more accurate predictions of both claim frequency and the probability of no claims.
- **Two different business questions**: "What makes someone unlikely to ever file a claim?" (structural zero) and "Among those who might file, what drives the number of claims?" (count process) may have different answers. Zero-inflated models let you investigate both.
- **Insurance pricing and underwriting**: Accurate zero probability estimation matters for pricing (especially for surplus/deductible structures) and for underwriting segmentation (identifying truly low-risk policyholders vs. those who are simply lucky).
- **Broad applicability**: Beyond insurance, zero-inflated models are used in ecology (species counts with many absent sites), healthcare (utilization counts with many non-users), manufacturing (defect counts), and marketing (purchase counts).

## When to use it

- Your count data has significantly more zeros than a Poisson or Negative Binomial distribution predicts.
- You believe the zeros arise from two distinct processes: "not at risk" (structural) and "at risk but no event" (sampling).
- You want separate models for the zero-generating process and the count-generating process.
- You need accurate prediction of $P(Y = 0)$ for pricing, reserving, or risk selection.
- Standard count models (Poisson, NB) show poor fit in the zero region (observed zeros far exceed predicted zeros).
- You are modeling insurance claims where some segment of the portfolio is essentially zero-risk for certain perils.

## How it works (intuitive)

### The two-component story

Imagine you are modeling the number of flood claims per homeowner. Some homeowners live on hilltops far from any water body -- they have essentially zero flood risk (structural zeros). Others live in flood zones but may or may not experience a flood in any given year (sampling zeros or positive counts).

A zero-inflated model captures this:
1. **Binary component (zero-inflation)**: A logistic regression predicts the probability $\pi_i$ that observation $i$ is a structural zero. Think of this as "is this policyholder in the zero-risk group?"
2. **Count component**: For observations *not* in the structural zero group, a Poisson (or Negative Binomial) distribution models the count. Think of this as "given this policyholder is at risk, how many claims do they file?"

An observed zero can come from either component: the policyholder is a structural zero (probability $\pi$), or the policyholder is at risk but had zero claims by chance (probability $(1-\pi) \times P(Y=0 | \text{count model})$).

### An analogy: fishing

Imagine counting fish caught per trip:
- Some people go to a lake with no fish (structural zero). They always catch zero.
- Others go to a stocked lake (at risk of catching fish). Some catch zero by bad luck (sampling zero), others catch 1, 2, 3, etc.

A standard Poisson model sees all the zeros and concludes "average catch is low" -- which distorts the predictions for everyone. A zero-inflated model separates "went to empty lake" from "went to stocked lake but caught nothing."

### Zero-inflated vs. Hurdle

| Feature | Zero-Inflated Model | Hurdle Model |
|---------|---------------------|--------------|
| Philosophy | Two sources of zeros (structural + sampling) | Two stages: zero vs. any count, then count magnitude |
| Zero mechanism | Mixture: $P(0) = \pi + (1-\pi) \cdot P_{\text{count}}(0)$ | Binary: $P(0) = 1 - p$; all zeros from one source |
| Positive counts | From the count component (conditional on not being structural zero) | From a zero-truncated count distribution |
| Identifiability | Can have identification issues if structural and sampling zeros are indistinguishable | Generally cleaner; fewer identification issues |
| Interpretation | "Some policyholders are immune to claims" | "First decide if any claims, then how many" |
| When to prefer | You believe two distinct zero-generating processes exist | You care about modeling the zero/non-zero split separately from the count magnitude |

## Key concepts

### Zero-Inflated Poisson (ZIP)

The ZIP model has two components:

$$P(Y_i = 0) = \pi_i + (1 - \pi_i) \cdot e^{-\lambda_i}$$

$$P(Y_i = k) = (1 - \pi_i) \cdot \frac{e^{-\lambda_i} \lambda_i^k}{k!} \quad \text{for } k = 1, 2, 3, \ldots$$

where:
- $\pi_i \in [0, 1]$: Probability that observation $i$ is a structural zero.
- $\lambda_i > 0$: Poisson rate for the count component.
- Both $\pi_i$ and $\lambda_i$ can depend on covariates via link functions.

**Model specification:**

$$\text{logit}(\pi_i) = \mathbf{w}_i^T \boldsymbol{\gamma} \quad \text{(zero-inflation model)}$$

$$\log(\lambda_i) = \mathbf{x}_i^T \boldsymbol{\beta} \quad \text{(count model)}$$

The covariates $\mathbf{w}_i$ and $\mathbf{x}_i$ can be the same or different. Using different covariates allows the zero-inflation process and the count process to have different drivers.

**Moments:**

$$E[Y_i] = (1 - \pi_i) \lambda_i$$

$$\text{Var}(Y_i) = (1 - \pi_i)\lambda_i(1 + \pi_i \lambda_i)$$

Note: $\text{Var}(Y_i) > E[Y_i]$ when $\pi_i > 0$, so the ZIP model naturally produces overdispersion.

### Zero-Inflated Negative Binomial (ZINB)

When the count component itself is overdispersed (variance exceeds mean even after accounting for zero-inflation), replace Poisson with Negative Binomial:

$$P(Y_i = k) = (1 - \pi_i) \cdot \binom{k + r - 1}{k} \left(\frac{r}{r + \lambda_i}\right)^r \left(\frac{\lambda_i}{r + \lambda_i}\right)^k \quad \text{for } k = 0, 1, 2, \ldots$$

where $r$ is the Negative Binomial dispersion parameter. As $r \to \infty$, ZINB reduces to ZIP.

**When to use ZINB over ZIP:**
- There is overdispersion in the count data *beyond* what zero-inflation explains.
- The data has both excess zeros and a long right tail (some observations with many claims).
- A Poisson assumption is restrictive: variance must equal mean in Poisson, but NB allows variance > mean.

### Hurdle Models

A Hurdle model has two parts:

**Part 1: Hurdle (binary):**

$$P(Y_i = 0) = 1 - p_i$$
$$P(Y_i > 0) = p_i$$

Modeled with logistic regression: $\text{logit}(p_i) = \mathbf{w}_i^T \boldsymbol{\gamma}$

**Part 2: Truncated count (given $Y > 0$):**

$$P(Y_i = k \mid Y_i > 0) = \frac{P_{\text{count}}(k)}{1 - P_{\text{count}}(0)} \quad \text{for } k = 1, 2, 3, \ldots$$

This is a zero-truncated Poisson or zero-truncated Negative Binomial.

**Key difference from zero-inflated models**: In a Hurdle model, all zeros come from the hurdle component. In a zero-inflated model, zeros can come from either the inflation component or the count component.

### Model comparison: choosing the right approach

The decision process:

```
Does the data have excess zeros?
|
|- No --> Standard Poisson or NB
|
|- Yes --> Is there overdispersion beyond zero-inflation?
           |
           |- No  --> ZIP or Poisson Hurdle
           |
           |- Yes --> ZINB or NB Hurdle
           |
           Do you believe in two sources of zeros?
           |
           |- Yes --> Zero-inflated (ZIP or ZINB)
           |- No  --> Hurdle
```

**Diagnostic tools for excess zeros:**

1. **Compare observed vs. expected zeros**: Fit a Poisson model. If the observed number of zeros is much larger than the Poisson-predicted number, there are excess zeros.

   $$\text{Expected zeros (Poisson)} = n \cdot e^{-\hat{\lambda}}$$

2. **Vuong test**: A likelihood ratio-based test comparing ZIP vs. Poisson (or ZINB vs. NB). A significant positive test statistic favors the zero-inflated model.

3. **Score test (van den Broek)**: Tests for zero-inflation against the null of no inflation.

4. **AIC/BIC comparison**: Compare the standard count model, zero-inflated model, and hurdle model using information criteria. Lower AIC/BIC is better.

### Likelihood function for ZIP

The log-likelihood for ZIP is:

$$\ell = \sum_{y_i = 0} \log\left[\pi_i + (1 - \pi_i) e^{-\lambda_i}\right] + \sum_{y_i > 0} \left[\log(1 - \pi_i) + y_i \log \lambda_i - \lambda_i - \log(y_i!)\right]$$

Maximized via EM algorithm or direct numerical optimization.

### Marginal effects and interpretation

In a zero-inflated model, the effect of a covariate on the expected count has two channels:

$$\frac{\partial E[Y]}{\partial x} = -\lambda \frac{\partial \pi}{\partial x} + (1 - \pi) \frac{\partial \lambda}{\partial x}$$

A covariate can:
1. Decrease the probability of being a structural zero (increases $1-\pi$), raising the expected count.
2. Increase the Poisson rate $\lambda$, raising the expected count.
3. Both, or in opposing directions.

This dual-channel interpretation is one of the strengths of zero-inflated models: you can disentangle "who is at risk" from "how many events among those at risk."

## Python implementation

**Key libraries:**
- `statsmodels.discrete.count_model.ZeroInflatedPoisson`: ZIP model.
- `statsmodels.discrete.count_model.ZeroInflatedNegativeBinomialP`: ZINB model.
- `statsmodels.discrete.count_model.ZeroInflatedGeneralizedPoisson`: ZIGP model.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.discrete.count_model import (
    ZeroInflatedPoisson,
    ZeroInflatedNegativeBinomialP,
)

# --- Zero-Inflated Poisson ---
# count model covariates
X_count = sm.add_constant(data[['age', 'territory_risk', 'vehicle_value']])
# zero-inflation model covariates (can be different)
X_inflate = sm.add_constant(data[['has_garage', 'years_licensed']])

zip_model = ZeroInflatedPoisson(
    endog=data['claim_count'],
    exog=X_count,
    exog_infl=X_inflate,
    inflation='logit',
)
zip_result = zip_model.fit(method='bfgs', maxiter=1000)
print(zip_result.summary())

# Predicted values
predicted_mean = zip_result.predict(X_count, exog_infl=X_inflate)
predicted_zero_prob = zip_result.predict(X_count, exog_infl=X_inflate, which='prob-zero')
```

```python
# --- Zero-Inflated Negative Binomial ---
zinb_model = ZeroInflatedNegativeBinomialP(
    endog=data['claim_count'],
    exog=X_count,
    exog_infl=X_inflate,
    inflation='logit',
)
zinb_result = zinb_model.fit(method='bfgs', maxiter=1000)
print(zinb_result.summary())
```

```python
# --- Model comparison ---
from scipy.stats import chi2

# Vuong test: ZIP vs Poisson
poisson_model = sm.GLM(data['claim_count'], X_count, family=sm.families.Poisson())
poisson_result = poisson_model.fit()

# Compare log-likelihoods
ll_zip = zip_result.llf
ll_poisson = poisson_result.llf
print(f"ZIP log-likelihood: {ll_zip:.2f}")
print(f"Poisson log-likelihood: {ll_poisson:.2f}")

# AIC comparison
print(f"ZIP AIC: {zip_result.aic:.2f}")
print(f"Poisson AIC: {poisson_result.aic:.2f}")
print(f"ZINB AIC: {zinb_result.aic:.2f}")
```

```python
# --- Checking for excess zeros ---
def check_excess_zeros(y, model_result):
    """
    Compare observed vs expected zeros.
    """
    observed_zeros = (y == 0).sum()
    predicted_probs = model_result.predict()
    expected_zeros = np.exp(-predicted_probs).sum()  # For Poisson
    print(f"Observed zeros: {observed_zeros}")
    print(f"Expected zeros (Poisson): {expected_zeros:.0f}")
    print(f"Ratio: {observed_zeros / expected_zeros:.2f}")
```

```python
# --- Hurdle model (manual implementation) ---
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.truncated_model import TruncatedLFPoisson

# Part 1: Binary model (zero vs. positive)
y_binary = (data['claim_count'] > 0).astype(int)
logit = LogisticRegression()
logit.fit(X_hurdle, y_binary)

# Part 2: Zero-truncated Poisson for positive counts
positive_data = data[data['claim_count'] > 0]
X_pos = sm.add_constant(positive_data[['age', 'territory_risk', 'vehicle_value']])
truncated_model = TruncatedLFPoisson(
    positive_data['claim_count'],
    X_pos,
    truncation=0,
)
truncated_result = truncated_model.fit()
print(truncated_result.summary())
```

## R implementation (if applicable)

R has excellent support for zero-inflated models:

```r
# pscl package: the standard for zero-inflated models in R
library(pscl)

# Zero-Inflated Poisson
zip_fit <- zeroinfl(claim_count ~ age + territory_risk + vehicle_value |
                      has_garage + years_licensed,
                    data = claims_data,
                    dist = "poisson")
summary(zip_fit)

# The formula syntax: count_model | zero_inflation_model
# If you want the same covariates for both: claim_count ~ covariates

# Zero-Inflated Negative Binomial
zinb_fit <- zeroinfl(claim_count ~ age + territory_risk + vehicle_value |
                       has_garage + years_licensed,
                     data = claims_data,
                     dist = "negbin")
summary(zinb_fit)

# Hurdle models
hurdle_fit <- hurdle(claim_count ~ age + territory_risk + vehicle_value |
                       has_garage + years_licensed,
                     data = claims_data,
                     dist = "poisson")
summary(hurdle_fit)

# NB Hurdle
hurdle_nb_fit <- hurdle(claim_count ~ age + territory_risk + vehicle_value |
                          has_garage + years_licensed,
                        data = claims_data,
                        dist = "negbin")
summary(hurdle_nb_fit)

# Vuong test: ZIP vs Poisson
library(pscl)
vuong(zip_fit, glm(claim_count ~ age + territory_risk + vehicle_value,
                    data = claims_data, family = poisson))

# Model comparison
AIC(zip_fit, zinb_fit, hurdle_fit, hurdle_nb_fit)
BIC(zip_fit, zinb_fit, hurdle_fit, hurdle_nb_fit)

# Predicted probabilities
predict(zip_fit, type = "prob")      # P(Y = k) for each observation
predict(zip_fit, type = "zero")      # P(Y = 0) for each observation
predict(zip_fit, type = "count")     # E[Y | count component]
predict(zip_fit, type = "response")  # E[Y] overall

# glmmTMB: for zero-inflated mixed models
library(glmmTMB)
zi_mixed <- glmmTMB(claim_count ~ age + territory_risk + (1 | territory),
                     zi = ~ has_garage + years_licensed,
                     data = claims_data,
                     family = poisson)
summary(zi_mixed)
```

## Gotchas & lessons learned

1. **Not all excess zeros are "zero-inflated"**: Overdispersion alone can produce excess zeros. A Negative Binomial model (without zero-inflation) may fit just as well. Always compare NB to ZINB -- if NB fits well, the extra complexity of zero-inflation is unnecessary.

2. **Identification issues**: If the structural zeros and sampling zeros are generated by the same covariates with the same coefficients, the model has difficulty separating the two components. Using different covariates for the zero-inflation and count components helps identification. Substantive knowledge should guide the choice: what variables predict being "not at risk" vs. "at risk but with low frequency"?

3. **Convergence problems**: Zero-inflated models have more parameters and can have convergence issues, especially with sparse data or poor starting values. Try different optimization methods (`bfgs`, `nm`, `powell`). In R, `pscl::zeroinfl` is generally robust.

4. **Interpreting the zero-inflation coefficients**: A positive coefficient in the zero-inflation model means the covariate *increases* the probability of being a structural zero (i.e., *decreases* the expected count). This is counterintuitive if you are used to interpreting Poisson coefficients. Read carefully which component each coefficient belongs to.

5. **Zero-inflated vs. Tweedie**: For modeling pure premium (total loss cost), Tweedie is often preferred over a zero-inflated model because Tweedie handles zeros through the compound Poisson mechanism (zero claims) rather than a separate structural zero process. Zero-inflated models are better suited for count data (claim frequency), not aggregate losses. See the Tweedie README for comparison.

6. **Excess zeros vs. underexposure**: Before invoking zero-inflation, check whether the zeros are simply due to low exposure. A policyholder in force for 1 month has a much higher chance of zero claims than one in force for 12 months. Proper exposure handling (offset) in the Poisson model may eliminate the need for zero-inflation.

7. **Hurdle models are more robust but less interpretable**: Hurdle models avoid the identification issues of zero-inflated models because all zeros come from one source. However, they do not provide the "two types of zeros" interpretation that can be scientifically meaningful.

8. **Prediction accuracy for rare events**: Even with a well-fitted zero-inflated model, predicting individual claim occurrences (P(claim) for a specific policyholder) is inherently difficult when the base rate is very low. Focus on aggregate predictions (expected total claims for a portfolio) rather than individual classification.

9. **Model checking**: Plot the observed distribution of counts against the predicted distribution from the fitted model. A rootogram (Tukey's hanging rootogram) is particularly useful for this comparison. If the zero-inflated model still underpredicts or overpredicts zeros, something is wrong.

10. **Software defaults differ**: In `statsmodels`, the zero-inflation model is parameterized as $P(\text{structural zero})$. In some R packages, it may be parameterized as $P(\text{not zero-inflated})$. Always check the software documentation to ensure correct interpretation.

11. **Zero-inflated GLMMs**: When your data has both excess zeros and a grouping structure (policyholders within territories), use a zero-inflated mixed model. The `glmmTMB` package in R handles this elegantly. In Python, `pymc` can specify Bayesian zero-inflated mixed models.

12. **Do not inflate for the sake of inflating**: If the substantive theory does not support two zero-generating processes, a zero-inflated model is adding unnecessary complexity. The statistical test for zero-inflation (Vuong test) can be significant even when the zero-inflated model is not the "true" model, simply because it has more parameters. Use domain knowledge alongside statistical tests.

## Resources

- [Lambert (1992) -- Zero-Inflated Poisson Regression with an Application to Defects in Manufacturing](https://www.jstor.org/stable/1269547) -- The foundational ZIP paper.
- [Zeileis, Kleiber, Jackman -- Regression Models for Count Data in R](https://www.jstatsoft.org/article/view/v027i08) -- Excellent overview of count models including zero-inflated and hurdle, with R examples.
- [pscl R package documentation](https://cran.r-project.org/web/packages/pscl/) -- Zero-inflated and hurdle models in R.
- [glmmTMB R package](https://cran.r-project.org/web/packages/glmmTMB/) -- Zero-inflated mixed models.
- [statsmodels count models documentation](https://www.statsmodels.org/stable/discretemod.html) -- Python implementation of ZIP and ZINB.
- [Cameron & Trivedi -- Regression Analysis of Count Data](https://www.cambridge.org/core/books/regression-analysis-of-count-data/E89B0ED9C0F95B4FE65F1AD3140B0BEF) -- Comprehensive textbook covering all count regression models.
- [Hilbe -- Modeling Count Data](https://www.cambridge.org/core/books/modeling-count-data/CDBB47C673E3B3CBEB005E2C21A0A7EA) -- Practical guide with emphasis on overdispersion and zero-inflation.
- [Denuit, Marechal, Pitrebois, Walhin -- Actuarial Modelling of Claim Counts](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470723609) -- Actuarial perspective on count models for insurance.
- [PyMC documentation on zero-inflated models](https://www.pymc.io/) -- Bayesian zero-inflated models.
- [Vuong (1989) -- Likelihood Ratio Tests for Model Selection](https://www.jstor.org/stable/1912557) -- The Vuong test paper for comparing non-nested models.
