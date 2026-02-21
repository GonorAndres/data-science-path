# Mixed Effects Models

## What it is

Mixed effects models (also called multilevel models, hierarchical linear models, or random effects models) are regression models that contain both **fixed effects** and **random effects**. Fixed effects are the standard regression coefficients -- they describe the average relationship between predictors and the outcome across the entire population. Random effects capture variation between groups (clusters, levels) in the data. If your data has a natural grouping structure -- policyholders within territories, claims within insurers, repeated measurements on the same subject -- mixed effects models account for this structure, producing more accurate estimates and valid inference. They sit at the intersection of classical regression and Bayesian shrinkage, providing a principled way to model heterogeneity across groups.

## Why it matters

- **Insurance data is inherently hierarchical**: Policies are nested within agents, agents within territories, territories within states. Claims are nested within policyholders. Mixed effects models respect this structure.
- **Improved predictions for groups with sparse data**: A territory with only 50 policies gets a territory effect that is "shrunk" toward the overall average, preventing wild estimates from small samples. This is the same concept as credibility weighting (see Bayesian Credibility README).
- **Connection to credibility theory**: The Best Linear Unbiased Predictor (BLUP) from a random effects model is mathematically identical to the Buhlmann credibility estimate. Mixed effects models are the modern statistical formalization of what actuaries have been doing for decades.
- **Avoids the false dichotomy**: Without mixed models, you face a choice: (a) ignore group structure entirely (pool all data), or (b) fit a separate model for each group. Option (a) misses real group differences; option (b) overfits for small groups. Mixed effects models find the optimal middle ground.
- **Proper standard errors**: If you ignore clustering and fit a standard regression, your standard errors are too small (because observations within a cluster are correlated, not independent). Mixed models correct for this.

## When to use it

- Your data has a grouping structure (policyholders within territories, students within schools, patients within hospitals) and you want to model both within-group and between-group variation.
- You have **repeated measures** on the same subject (multiple policy years for the same policyholder, longitudinal data).
- You want to estimate group-specific effects (territory rating factors, agent effects) but some groups have very little data and need regularization/shrinkage.
- You are building an insurance pricing model and want territory or agent effects to be estimated with appropriate credibility weighting.
- You want to account for correlation within clusters to get correct standard errors and confidence intervals.
- You have a continuous outcome (LMM) or a discrete/count outcome (GLMM) and hierarchical data.

## How it works (intuitive)

### The core idea: partial pooling

Consider estimating average claim cost for 100 territories. You have three strategies:

1. **Complete pooling** (no territory effect): Assume all territories are the same. Use the grand mean. Simple but ignores real differences.

2. **No pooling** (fixed effects for every territory): Estimate a separate mean for each territory. Flexible but noisy -- a territory with 5 claims gets a very imprecise estimate.

3. **Partial pooling** (random effects): Estimate territory means, but "shrink" them toward the grand mean. Territories with lots of data keep estimates close to their observed mean. Territories with little data are pulled strongly toward the grand mean.

Mixed effects models implement strategy 3. The amount of shrinkage is determined automatically by the data: the model estimates both the within-territory variance (noise) and the between-territory variance (real differences), then uses their ratio to calibrate the shrinkage.

This is exactly the credibility formula:

$$\hat{\mu}_i = Z_i \cdot \bar{X}_i + (1 - Z_i) \cdot \hat{\mu}$$

where $Z_i$ depends on the sample size for territory $i$ and the ratio of within to between variance.

### An analogy: grading on a curve

Imagine grading students across many classes. Each class has a different teacher, and some teachers are stricter graders. A student who gets 80% in a strict teacher's class might be performing better than a student who gets 85% in a lenient teacher's class.

A mixed effects model adjusts for the "teacher effect" (random intercept per teacher), so the adjusted scores are more comparable. If a class is small, the model is less certain about how strict that teacher is, so the adjustment is smaller (shrinkage toward the average teacher).

### Fixed effects vs. random effects

| Feature | Fixed Effects | Random Effects |
|---------|--------------|----------------|
| What they represent | Specific, repeatable levels (e.g., male/female, specific drug doses) | Levels sampled from a larger population (e.g., specific territories out of all possible territories) |
| Inference | About these specific levels | About the population of levels |
| Number of levels | Typically few | Can be many (even hundreds) |
| Estimation | Each level gets its own parameter, no shrinkage | Levels are modeled as draws from a distribution; estimates are shrunk toward the mean |
| When data is sparse | Estimates can be very noisy | Estimates are regularized by borrowing strength from other levels |

**Rule of thumb**: If the levels are a sample from a larger population (territories, agents, policyholders), treat them as random. If the levels are exhaustive categories of scientific interest (gender, treatment vs. control), treat them as fixed.

## Key concepts

### Linear Mixed Model (LMM)

The simplest mixed model. For observation $j$ in group $i$:

$$Y_{ij} = \mathbf{X}_{ij} \boldsymbol{\beta} + \mathbf{Z}_{ij} \mathbf{b}_i + \varepsilon_{ij}$$

| Component | Meaning |
|-----------|---------|
| $\mathbf{X}_{ij} \boldsymbol{\beta}$ | Fixed effects: average relationship (like standard regression) |
| $\mathbf{Z}_{ij} \mathbf{b}_i$ | Random effects: group-specific deviations from the average |
| $\mathbf{b}_i \sim N(\mathbf{0}, \mathbf{D})$ | Random effects are drawn from a normal distribution with covariance $\mathbf{D}$ |
| $\varepsilon_{ij} \sim N(0, \sigma^2)$ | Residual error |

**Random intercept model** (the most common case):

$$Y_{ij} = \beta_0 + \beta_1 X_{ij} + b_i + \varepsilon_{ij}$$

where $b_i \sim N(0, \sigma_b^2)$ is the random intercept for group $i$.

This says: "Each group has its own intercept, but the intercepts come from a normal distribution centered at 0 (deviations from the overall intercept $\beta_0$)."

**Random intercept and slope model:**

$$Y_{ij} = \beta_0 + \beta_1 X_{ij} + b_{0i} + b_{1i} X_{ij} + \varepsilon_{ij}$$

This allows both the intercept and the slope of $X$ to vary across groups.

### Generalized Linear Mixed Model (GLMM)

Extension of LMM to non-normal outcomes via a link function, just as GLMs extend linear regression.

$$g(E[Y_{ij}]) = \mathbf{X}_{ij} \boldsymbol{\beta} + \mathbf{Z}_{ij} \mathbf{b}_i$$

| Outcome type | Link function $g$ | Distribution | Insurance example |
|-------------|-------------------|-------------|-------------------|
| Binary (0/1) | Logit | Binomial | Claim / no claim |
| Count | Log | Poisson | Number of claims |
| Continuous positive | Log | Gamma | Claim severity |
| Mixed (zeros + continuous) | Log | Tweedie | Pure premium |

### Variance components and ICC

The **Intraclass Correlation Coefficient (ICC)** measures how much of the total variance is explained by group membership:

$$\text{ICC} = \frac{\sigma_b^2}{\sigma_b^2 + \sigma^2}$$

- ICC = 0: No group effect; observations within a group are no more similar to each other than to observations in other groups.
- ICC = 1: All variation is between groups; observations within a group are identical.

In insurance: if ICC for territory is 0.15, it means 15% of the variation in loss costs is explained by territory differences. This justifies including territory as a random effect.

### Estimation methods

| Method | Full name | Notes |
|--------|-----------|-------|
| **ML** | Maximum Likelihood | Jointly estimates fixed and random effects. Tends to underestimate variance components (biased downward). |
| **REML** | Restricted Maximum Likelihood | Adjusts for the degrees of freedom used by fixed effects. Gives unbiased variance component estimates. Preferred for inference on variance components. |
| **PQL** | Penalized Quasi-Likelihood | Approximation for GLMMs. Fast but can be biased, especially for binary outcomes with small clusters. |
| **Laplace** | Laplace approximation | More accurate than PQL for GLMMs. Default in `lme4`. |
| **MCMC** | Markov Chain Monte Carlo | Fully Bayesian. Most accurate but slowest. Used when approximations are inadequate. |

### BLUP (Best Linear Unbiased Predictor)

The predicted random effect for group $i$ is called the BLUP:

$$\hat{b}_i = \mathbf{D} \mathbf{Z}_i^T (\mathbf{Z}_i \mathbf{D} \mathbf{Z}_i^T + \sigma^2 \mathbf{I})^{-1} (Y_i - \mathbf{X}_i \hat{\boldsymbol{\beta}})$$

This is the shrunken estimate of the group's deviation from the population mean. For a random intercept model with balanced data:

$$\hat{b}_i = \frac{n_i \sigma_b^2}{n_i \sigma_b^2 + \sigma^2} (\bar{Y}_i - \mathbf{X}_i \hat{\boldsymbol{\beta}})$$

The shrinkage factor $\frac{n_i \sigma_b^2}{n_i \sigma_b^2 + \sigma^2}$ is identical to the Buhlmann credibility factor $Z = n / (n + k)$ where $k = \sigma^2 / \sigma_b^2$.

### Model selection and testing

- **Likelihood Ratio Test (LRT)**: Compare nested models. For testing whether a random effect is needed, compare the model with and without the random effect. Note: testing whether $\sigma_b^2 = 0$ is on the boundary of the parameter space, so the standard chi-squared distribution does not apply. Use a mixture of chi-squared distributions (e.g., 50:50 mix of $\chi^2_0$ and $\chi^2_1$).
- **AIC / BIC**: Compare non-nested models. Use ML (not REML) when comparing models with different fixed effects.
- **Conditional vs. marginal $R^2$**: Marginal $R^2$ measures variance explained by fixed effects alone. Conditional $R^2$ measures variance explained by both fixed and random effects. Both are useful and tell different stories.

## Python implementation

**Key libraries:**
- `statsmodels.MixedLM`: Linear mixed models. Supports random intercepts, random slopes, and variance component estimation.
- `pymer4`: Python wrapper around R's `lme4`. Provides `lmer` and `glmer` syntax in Python.

```python
import statsmodels.formula.api as smf
import pandas as pd

# Random intercept model
# claim_cost ~ vehicle_age + driver_age + (1 | territory)
model = smf.mixedlm(
    "claim_cost ~ vehicle_age + driver_age",
    data=claims_data,
    groups=claims_data["territory"],
    re_formula="~1"  # random intercept only
)
result = model.fit(reml=True)
print(result.summary())

# Extract random effects (BLUPs)
random_effects = result.random_effects
for territory, effects in random_effects.items():
    print(f"Territory {territory}: intercept adjustment = {effects['Group']:.2f}")

# Random intercept and slope
model_rs = smf.mixedlm(
    "claim_cost ~ vehicle_age + driver_age",
    data=claims_data,
    groups=claims_data["territory"],
    re_formula="~vehicle_age"  # random intercept + random slope for vehicle_age
)
result_rs = model_rs.fit(reml=True)
print(result_rs.summary())
```

```python
# GLMM using pymer4 (requires R and lme4 installed)
from pymer4.models import Lmer

# Poisson GLMM for claim counts
model = Lmer(
    "claim_count ~ vehicle_age + driver_age + (1 | territory)",
    data=claims_data,
    family="poisson"
)
result = model.fit()
print(result)
```

## R implementation (if applicable)

R is the gold standard for mixed effects models:

```r
# lme4: the standard package for mixed effects
library(lme4)

# Linear Mixed Model
lmm <- lmer(claim_cost ~ vehicle_age + driver_age + (1 | territory),
            data = claims_data, REML = TRUE)
summary(lmm)

# BLUPs (random effects predictions)
ranef(lmm)

# Conditional modes with confidence intervals
dotplot(ranef(lmm, condVar = TRUE))

# GLMM: Poisson for claim counts
glmm_pois <- glmer(claim_count ~ vehicle_age + driver_age + (1 | territory),
                    data = claims_data, family = poisson)
summary(glmm_pois)

# GLMM: Binomial for claim occurrence
glmm_binom <- glmer(has_claim ~ vehicle_age + driver_age + (1 | territory),
                     data = claims_data, family = binomial)
summary(glmm_binom)

# Random intercept + random slope
lmm_rs <- lmer(claim_cost ~ vehicle_age + driver_age +
                 (1 + vehicle_age | territory),
               data = claims_data, REML = TRUE)
summary(lmm_rs)

# nlme package (alternative, supports more correlation structures)
library(nlme)
lmm_nlme <- lme(claim_cost ~ vehicle_age + driver_age,
                random = ~ 1 | territory,
                data = claims_data)
summary(lmm_nlme)

# ICC calculation
library(performance)
icc(lmm)

# Conditional and marginal R-squared
r2(lmm)
```

## Gotchas & lessons learned

1. **Singular fits**: If the random effects variance estimate hits zero (singular fit), the model is telling you there is no between-group variation (or not enough data to detect it). This is not necessarily an error, but check:
   - Is the grouping variable meaningful?
   - Do you have enough groups? (Rule of thumb: at least 5-10 groups, ideally more.)
   - Is the random effects structure too complex for the data?

2. **Number of groups matters more than group size**: Mixed models estimate the distribution of random effects. You need enough groups (at least 10-20, preferably 30+) to estimate $\sigma_b^2$ reliably. Having very large groups does not compensate for too few groups.

3. **Convergence issues in GLMMs**: GLMMs (especially with binary outcomes) can have convergence problems. Try:
   - Scaling and centering predictors.
   - Using a simpler random effects structure.
   - Switching optimizers (e.g., `bobyqa` in `lme4`).
   - Using Bayesian estimation via `brms` or `rstanarm`.

4. **Fixed vs. random is not always clear-cut**: The decision to treat a factor as fixed or random is partly philosophical and partly practical. If you have only 3-4 levels, the random effects distribution is poorly estimated, and fixed effects may be safer. If you have 50+ levels and want predictions for new levels, random effects are strongly preferred.

5. **Random effects are not regression coefficients**: The BLUPs are predictions of the random effects, not estimates in the usual sense. They are shrunken and have different uncertainty properties. Do not interpret them like fixed effect coefficients.

6. **Crossed vs. nested random effects**: Policyholders crossed with agents (a policyholder can have different agents over time) vs. policyholders nested within agents (each policyholder belongs to exactly one agent). The model specification differs, and getting it wrong produces incorrect variance estimates.

   ```r
   # Nested: students within schools
   lmer(score ~ age + (1 | school/student))
   # Crossed: items crossed with subjects
   lmer(score ~ condition + (1 | subject) + (1 | item))
   ```

7. **Do not forget to check residuals**: Mixed models have residuals at two levels: level-1 (observation level) and level-2 (group level). Check normality of both. Non-normal level-2 residuals suggest the random effects distribution assumption is wrong.

8. **Prediction for new groups**: Mixed models can predict for groups not seen in the training data by using the fixed effects alone (setting random effects to 0). This is a major advantage over fitting separate models per group.

9. **Software differences**: `lme4` (R) and `statsmodels` (Python) can give slightly different results due to different optimization algorithms and defaults (ML vs. REML, optimizer choice). Always verify by checking log-likelihoods.

10. **Reporting**: Always report both fixed effects (average relationships) and random effects (variance components, ICC). State whether ML or REML was used. Include model comparison results (LRT, AIC) if you tested multiple model structures.

## Resources

- [Gelman & Hill -- Data Analysis Using Regression and Multilevel/Hierarchical Models](https://www.cambridge.org/highereducation/books/data-analysis-using-regression-and-multilevel-hierarchical-models/32A29531C7FD730C3A68951A17C9D983) -- The best applied introduction.
- [Bates et al. -- Fitting Linear Mixed-Effects Models Using lme4](https://www.jstatsoft.org/article/view/v067i01) -- The `lme4` paper. Excellent technical reference.
- [statsmodels MixedLM documentation](https://www.statsmodels.org/stable/mixed_linear.html) -- Python implementation.
- [lme4 R package](https://cran.r-project.org/web/packages/lme4/) -- The standard R implementation.
- [nlme R package](https://cran.r-project.org/web/packages/nlme/) -- Alternative R implementation with more flexible correlation structures.
- [brms R package](https://paul-buerkner.github.io/brms/) -- Bayesian mixed models via Stan. Handles complex random effects structures easily.
- [pymer4 documentation](https://eshkim.github.io/pymer4/) -- Python wrapper for `lme4`.
- [Frees -- Regression Modeling with Actuarial and Financial Applications](https://instruction.bus.wisc.edu/jfrees/jfreesbooks/) -- Actuarial applications of mixed models.
- [Antonio & Beirlant (2007)](https://www.jstor.org/) -- Actuarial statistics and mixed models for claims data.
