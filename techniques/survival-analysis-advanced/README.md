# Advanced Survival Analysis

## What it is

Advanced survival analysis extends beyond the basic Kaplan-Meier estimator and Cox proportional hazards model to handle situations where the standard assumptions break down. This guide covers four important extensions: **Accelerated Failure Time (AFT) models** that model survival time directly rather than hazard rates, **Competing risks** for situations where multiple types of events can occur, **Time-varying covariates** for predictors that change over the observation period, and **Frailty models** that account for unobserved heterogeneity. These methods are essential in actuarial work where the data is messy, decrements are multiple, and individual-level heterogeneity matters.

## Why it matters

Basic survival analysis tools are powerful but limited. In real actuarial and insurance applications, you frequently encounter situations they cannot handle well:

- **Multiple decrements**: A life insurance policyholder can exit through death, lapse, surrender, or conversion. Each is a different type of "failure," and ignoring competing events biases your estimates.
- **Non-proportional hazards**: The Cox PH model assumes that the ratio of hazards between groups is constant over time. If the effect of a variable changes over time (e.g., the risk difference between smokers and non-smokers widens as age increases), you need an alternative.
- **Changing risk factors**: A policyholder's health status, coverage amount, or premium rate can change over the policy's lifetime. Basic models treat covariates as fixed at baseline; time-varying covariate models track these changes.
- **Hidden heterogeneity**: Even after accounting for observed covariates, individuals within a group may have different underlying risk levels due to unobserved factors (genetics, lifestyle habits, unreported health conditions). Frailty models capture this.
- **Direct time modeling**: Sometimes you want to model the expected time to event (e.g., average time to claim settlement) rather than the hazard rate. AFT models provide this directly.

Understanding these advanced methods transforms survival analysis from a descriptive tool into a flexible modeling framework for actuarial applications.

## When to use it

- **AFT models**: When you want to model survival time directly (not hazard), when the proportional hazards assumption is violated, or when you want a more interpretable model ("this factor extends/shortens expected survival time by X%").
- **Competing risks**: When subjects can experience more than one type of event (death vs. lapse vs. surrender), and you want to correctly estimate the incidence of each event type.
- **Time-varying covariates**: When predictor values change during the observation period (e.g., premium changes, health status updates, policy modifications).
- **Frailty models**: When you suspect unobserved heterogeneity (individual-level random effects), when data is clustered (policyholders in the same family, claims from the same employer group), or when you observe apparent "cure rates" or leveling off of hazard at long durations.

## How it works (intuitive)

### AFT models: Modeling time directly

**The basic idea**: While the Cox model asks "what is the instantaneous risk of failure?", the AFT model asks "how long until failure?" It models the survival time directly as a function of covariates.

**Analogy**: Think of the Cox model as a speedometer -- it tells you how fast you are approaching the event at each moment. The AFT model is more like a trip planner -- it tells you the expected total travel time. Some covariates accelerate the trip (shorter survival), others decelerate it (longer survival).

**Mathematical formulation**:

$$\log(T) = \mu + \beta_1 x_1 + \beta_2 x_2 + \ldots + \sigma \epsilon$$

Where:
- $T$ is the survival time
- $\mu$ is the intercept
- $\beta_i$ are regression coefficients
- $x_i$ are covariates
- $\sigma$ is a scale parameter
- $\epsilon$ is an error term with a specified distribution

The distribution of $\epsilon$ determines the survival time distribution:

| Error distribution | Survival time distribution | Hazard shape |
|-------------------|--------------------------|--------------|
| Extreme value (Gumbel) | Weibull | Monotone increasing or decreasing |
| Logistic | Log-logistic | Hump-shaped (increases then decreases) |
| Normal | Log-normal | Hump-shaped |
| -- | Exponential | Constant (special case of Weibull) |

**Interpretation**: The coefficients have an "acceleration factor" interpretation. If $\beta_1 = 0.5$ for a binary covariate, then $e^{0.5} \approx 1.65$, meaning having that covariate multiplies the expected survival time by 1.65 (65% longer survival). If $\beta_1 = -0.3$, then $e^{-0.3} \approx 0.74$ -- survival time is 74% of baseline (26% shorter).

**AFT vs. Cox PH**:

| Aspect | Cox PH | AFT |
|--------|--------|-----|
| Models | Hazard function | Survival time directly |
| Assumption | Proportional hazards | Accelerated failure time |
| Interpretation | Hazard ratios | Time ratios (acceleration factors) |
| Baseline function | Non-parametric (unspecified) | Parametric (must choose a distribution) |
| Flexibility | Very flexible (semi-parametric) | Less flexible (must specify distribution) |
| When PH fails | Need stratification or interaction terms | Often a natural alternative |
| Prediction | Hazard/survival curves | Direct time predictions |

### Competing risks: Multiple ways to fail

**The problem**: In standard survival analysis, if a policyholder lapses, they are "censored" for the death analysis. But lapse is not random censoring -- it may be related to the same factors that predict death. Treating dependent competing events as independent censoring biases your estimates.

**Analogy**: Imagine you are studying how long light bulbs last until they burn out. But some bulbs are removed from the socket before they burn out (competing event: removal). If the removed bulbs were the ones that were flickering (about to burn out), then treating removal as censoring makes the remaining bulbs look more durable than they really are.

**Two approaches**:

1. **Cause-specific hazard**: Model the hazard for each event type separately, treating other event types as censoring. This answers: "Among those still at risk, what is the instantaneous rate of this specific event?"

2. **Cumulative incidence function (CIF) / Fine-Gray model**: Models the subdistribution hazard, which accounts for the fact that subjects who experience a competing event are permanently removed from the risk set for the event of interest. This answers: "What is the actual probability of experiencing this event by time t, given that competing events can also occur?"

**When to use which**:
- **Cause-specific**: When you want to understand the biological/behavioral mechanism driving each event. "Among those who haven't exited yet, what drives death risk?"
- **CIF/Fine-Gray**: When you want to predict the actual probability of an event occurring. "What is the probability this policyholder will lapse within 5 years, considering they might also die or surrender?"

**Cumulative incidence function**:

$$CIF_k(t) = P(T \leq t, \text{event type} = k) = \int_0^t S(u^-) \, h_k(u) \, du$$

Where $S(u^-)$ is the overall survival function (probability of surviving without any event) and $h_k(u)$ is the cause-specific hazard for event type $k$.

Key property: $\sum_k CIF_k(\infty) = 1$ -- over infinite time, everyone experiences one of the competing events.

### Time-varying covariates: Predictors that change

**The problem**: The standard Cox model uses covariate values measured at baseline (start of observation). But in insurance:
- Premium rates change at renewal
- Health status evolves over time
- Coverage amounts are adjusted
- Economic conditions fluctuate

**The solution**: Restructure the data so that each person-period at risk is a separate row, and covariate values are updated within each period.

**Analogy**: Instead of taking a snapshot of a person at the start and using that forever, you take a time-lapse video, noting each frame's conditions.

**Data structure**: The long ("counting process") format:

| ID | Start | Stop | Event | Premium | Health_Score |
|----|-------|------|-------|---------|-------------|
| 1 | 0 | 12 | 0 | 500 | 85 |
| 1 | 12 | 24 | 0 | 550 | 80 |
| 1 | 24 | 30 | 1 | 550 | 72 |
| 2 | 0 | 18 | 0 | 400 | 90 |
| 2 | 18 | 36 | 0 | 450 | 88 |

Person 1 has three intervals: months 0-12 (premium=500, health=85), months 12-24 (premium increased to 550, health declined to 80), and months 24-30 (event occurred at month 30, health now 72).

**Implementation in Cox model**: The model is the same, but the data is in this long format. The partial likelihood is computed using the covariate values at each event time.

### Frailty models: Hidden heterogeneity

**The problem**: Even after including all available covariates, individuals in the same risk group may have different underlying risk levels. This unobserved heterogeneity has important consequences:
- The observed hazard for a group decreases over time (even if individual hazards are constant), because the frail individuals fail first, leaving a progressively selected (hardier) population.
- Covariate effects are attenuated (biased toward zero) because some of the association is absorbed by the unobserved frailty.

**Analogy**: You are studying the lifespan of cars in a junkyard. Two cars of the same make, model, and year may have very different lifespans because one was driven gently on smooth roads and the other was driven hard on rough terrain. "Driving style" is the unobserved frailty.

**Mathematical formulation**:

$$h_i(t) = z_i \cdot h_0(t) \exp(\mathbf{x}_i^T \boldsymbol{\beta})$$

Where $z_i$ is the frailty term for individual $i$ -- a random effect drawn from some distribution (usually Gamma or Log-Normal). $z_i > 1$ means higher risk than predicted by covariates; $z_i < 1$ means lower risk.

**Gamma frailty**: The Gamma distribution is the most common choice because:
- It has a simple closed-form solution for the marginal survival function
- It leads to a marginal hazard that decreases over time (matching the "selection" effect)
- Mean is fixed at 1 (no frailty on average), with variance $\theta$ controlling the degree of heterogeneity

**Shared frailty**: When data is clustered (family members, employees in same group), all members of a cluster share the same frailty value. This models within-cluster correlation.

**Actuarial connection**: Frailty models are directly related to the actuarial concept of heterogeneity in mortality. The phenomenon where observed aggregate mortality declines faster than individual mortality ("mortality deceleration") is explained by frailty -- the most vulnerable die early, leaving a selected population.

## Key concepts

### Choosing an AFT distribution

Fit multiple distributions and compare using AIC/BIC:

```
AIC = -2 * log_likelihood + 2 * n_parameters
```

Also use:
- **QQ plots**: Plot log survival times against theoretical quantiles of the candidate distribution.
- **Residual analysis**: Cox-Snell residuals should follow an Exponential(1) distribution if the model is correct.
- **Domain knowledge**: Weibull is natural for aging/wear-out processes. Log-normal and log-logistic are better when hazard first increases then decreases (e.g., post-surgery mortality).

### Multiple decrements in actuarial notation

In actuarial science, competing risks are formalized as **multiple decrement theory**:

- ${}^{(\tau)}q_x$ = probability of decrement from any cause between ages $x$ and $x+1$
- ${}^{(j)}q_x$ = probability of decrement from cause $j$ between ages $x$ and $x+1$
- $\sum_j {}^{(j)}q_x = {}^{(\tau)}q_x$

The **associated single decrement** table removes the competition: "what would the lapse rate be if death were not possible?" This is derived from cause-specific forces of decrement:

$$\mu_x^{(j)} = \text{cause-specific force of decrement for cause } j$$
$$\mu_x^{(\tau)} = \sum_j \mu_x^{(j)}$$

Under the assumption of independence between decrements (the uniform distribution of deaths assumption, or UDD):

$$q_x^{(j)'} \approx 1 - (1 - {}^{(\tau)}q_x)^{\mu_x^{(j)}/\mu_x^{(\tau)}}$$

where $q_x^{(j)'}$ is the associated single decrement probability.

### Model diagnostics

**For AFT**:
- Cox-Snell residuals plotted against the cumulative hazard should follow a 45-degree line
- Deviance residuals for identifying outliers
- QQ plots of standardized residuals against the assumed distribution

**For competing risks**:
- Compare observed CIF curves with model-predicted CIF
- Test the proportional subdistribution hazards assumption for Fine-Gray models

**For frailty models**:
- Likelihood ratio test comparing model with and without frailty term
- Estimate frailty variance $\theta$ -- if close to zero, frailty is unnecessary

## Python implementation

### AFT models with lifelines

```python
from lifelines import WeibullAFTFitter, LogNormalAFTFitter, LogLogisticAFTFitter

# Weibull AFT
waft = WeibullAFTFitter()
waft.fit(df, duration_col='duration', event_col='event')
waft.print_summary()

# Compare distributions
from lifelines import LogNormalAFTFitter, LogLogisticAFTFitter

lnaft = LogNormalAFTFitter()
lnaft.fit(df, duration_col='duration', event_col='event')

llaft = LogLogisticAFTFitter()
llaft.fit(df, duration_col='duration', event_col='event')

print(f"Weibull AIC: {waft.AIC_}")
print(f"Log-Normal AIC: {lnaft.AIC_}")
print(f"Log-Logistic AIC: {llaft.AIC_}")

# Predict median survival time for new data
waft.predict_median(new_data)

# Predict survival function for a specific individual
waft.predict_survival_function(new_data.iloc[:1])
```

### Competing risks with lifelines

```python
from lifelines import AalenJohansenFitter

# Aalen-Johansen estimator for cumulative incidence
# event_of_interest is the event code you want to estimate CIF for
# event column should contain: 0 for censored, 1 for event type 1, 2 for event type 2, etc.

ajf = AalenJohansenFitter()
ajf.fit(durations=df['duration'], event_observed=df['event'], event_of_interest=1)
ajf.plot()

# For cause-specific Cox models, simply filter events
# Cause-specific model for lapse (event=1): treat death (event=2) as censored
df_cause1 = df.copy()
df_cause1['event_cause1'] = (df['event'] == 1).astype(int)

from lifelines import CoxPHFitter
cox_cause1 = CoxPHFitter()
cox_cause1.fit(df_cause1, duration_col='duration', event_col='event_cause1')
cox_cause1.print_summary()
```

### Competing risks with scikit-survival (Fine-Gray)

```python
# For Fine-Gray subdistribution hazard models, use R's cmprsk
# or the competing risks module in scikit-survival (if available)
# R integration via rpy2:
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
pandas2ri.activate()

ro.r('''
library(cmprsk)
# Fine-Gray model
fg <- crr(ftime=data$duration, fstatus=data$event, cov1=model_matrix)
summary(fg)
''')
```

### Time-varying covariates with lifelines

```python
from lifelines import CoxPHFitter

# Data must be in long format with start/stop times
# Columns: id, start, stop, event, covariate1, covariate2, ...
cox_tv = CoxPHFitter()
cox_tv.fit(
    df_long,
    duration_col='stop',
    event_col='event',
    entry_col='start',     # This enables the counting process format
    id_col='id',
    cluster_col='id'       # Robust standard errors for within-subject correlation
)
cox_tv.print_summary()
```

### Frailty models

```python
# Shared frailty (gamma) with lifelines
from lifelines import CoxPHFitter

# Use the cluster_col parameter for shared frailty
cox_frailty = CoxPHFitter()
cox_frailty.fit(
    df,
    duration_col='duration',
    event_col='event',
    cluster_col='group_id'  # Shared frailty by group
)
cox_frailty.print_summary()

# For more control over frailty distributions, use R's survival package:
# survreg(Surv(time, event) ~ x1 + x2 + frailty(group_id, distribution="gamma"), data=df)
```

## R implementation (if applicable)

R has the strongest ecosystem for advanced survival analysis:

```r
library(survival)
library(cmprsk)
library(frailtypack)

# AFT model (Weibull)
aft_weibull <- survreg(Surv(time, event) ~ x1 + x2, data = df, dist = "weibull")
summary(aft_weibull)

# Competing risks: cumulative incidence function
cif <- cuminc(ftime = df$time, fstatus = df$event, group = df$treatment)
plot(cif)

# Fine-Gray competing risks regression
fg_model <- crr(ftime = df$time, fstatus = df$event, cov1 = model.matrix(~ x1 + x2, df))
summary(fg_model)

# Time-varying covariates
# Use the counting process format: Surv(start, stop, event)
cox_tv <- coxph(Surv(start, stop, event) ~ x1_tv + x2, data = df_long,
                id = df_long$id)
summary(cox_tv)

# Gamma frailty model
cox_frailty <- coxph(Surv(time, event) ~ x1 + x2 + frailty(group_id, distribution = "gamma"),
                     data = df)
summary(cox_frailty)
```

R's `survival` and `cmprsk` packages are the gold standard for competing risks and frailty models. For advanced AFT models, the `flexsurv` package offers additional distributions and flexibility.

## Gotchas & lessons learned

### Do not treat competing events as censoring without thinking
If policyholders who lapse are systematically different from those who do not (they almost always are), treating lapse as non-informative censoring in a mortality analysis will bias your results. Use competing risks methods or, at minimum, test the sensitivity of your results to different censoring assumptions.

### AFT distribution choice matters
Unlike Cox PH (which leaves the baseline unspecified), AFT models require you to choose a distribution. A wrong distributional assumption produces biased estimates. Always compare multiple distributions using AIC/BIC and residual diagnostics. The Weibull AFT is a good default because it includes the exponential as a special case.

### Time-varying covariates require careful data preparation
Converting from the standard format to the counting process (start, stop) format is error-prone. Verify:
- No overlapping intervals for the same subject
- Events only at interval endpoints (stop times)
- Covariate values are correct within each interval
- Total person-time equals the original duration

### Frailty variance near zero does not mean no heterogeneity
It may mean you have already captured most heterogeneity with your observed covariates, or that the frailty distribution is misspecified. Try different frailty distributions (Gamma, Log-Normal, Inverse Gaussian).

### Shared frailty requires enough clusters
If you have only 5 employer groups with 2-3 members each, a shared frailty model will not reliably estimate the frailty variance. You need a reasonable number of clusters (30+) with reasonable cluster sizes (5+).

### Competing risks change the interpretation of Kaplan-Meier
The complement of the Kaplan-Meier curve (1 - KM) overestimates the event probability when competing risks exist, because KM treats competing events as censoring and assumes they could still experience the event of interest. Use the cumulative incidence function instead for event probabilities.

### Informative censoring violates most models
If people drop out of your study for reasons related to their risk (e.g., sick patients stop renewing), your censoring is informative. Standard methods assume non-informative censoring. Consider sensitivity analyses or joint modeling of the event process and the censoring process.

### Actuarial notation vs. statistical notation
Actuaries and statisticians use different notation and sometimes different definitions for the same concepts. Be careful when translating between actuarial multiple decrement tables and statistical competing risks models. The force of decrement $\mu_x^{(j)}$ is the same as the cause-specific hazard $h_j(t)$.

## Resources

- [lifelines documentation](https://lifelines.readthedocs.io/en/latest/) -- Python library for survival analysis, including AFT models and basic competing risks
- [scikit-survival documentation](https://scikit-survival.readthedocs.io/en/stable/) -- Survival analysis compatible with scikit-learn API
- [R survival package](https://cran.r-project.org/web/packages/survival/index.html) -- The definitive implementation
- [R cmprsk package](https://cran.r-project.org/web/packages/cmprsk/) -- Competing risks regression
- [R flexsurv package](https://cran.r-project.org/web/packages/flexsurv/) -- Flexible parametric survival models
- [Klein & Moeschberger (2003) - Survival Analysis: Techniques for Censored and Truncated Data](https://link.springer.com/book/10.1007/b97377) -- Comprehensive textbook
- [Bowers et al. - Actuarial Mathematics](https://www.soa.org/) -- Multiple decrement theory from the actuarial perspective
- [Kalbfleisch & Prentice (2002) - The Statistical Analysis of Failure Time Data](https://www.wiley.com/en-us/The+Statistical+Analysis+of+Failure+Time+Data) -- Rigorous treatment of competing risks
- [Hougaard (2000) - Analysis of Multivariate Survival Data](https://link.springer.com/book/10.1007/978-1-4612-1304-8) -- Definitive reference on frailty models
- [Fine & Gray (1999) - A Proportional Hazards Model for the Subdistribution of a Competing Risk](https://doi.org/10.1080/01621459.1999.10474144) -- The Fine-Gray paper
