# Survival Analysis (Kaplan-Meier, Cox Proportional Hazards, Log-Rank Test)

## What it is

Survival analysis is a family of statistical methods for analyzing **time-to-event data** -- how long until something happens. The "event" can be death (the original medical application), but also customer churn, insurance claim, machine failure, loan default, or policy lapse. What makes survival analysis special is its ability to handle **censoring**: observations where the event has not yet occurred by the end of the study. If you track 1,000 policyholders for 3 years and 400 are still active at the end, standard methods would either ignore those 400 or incorrectly treat them as "never lapsing." Survival analysis properly accounts for this incomplete information. The three core methods are: **Kaplan-Meier** (non-parametric survival curve estimation), **log-rank test** (comparing survival between groups), and **Cox Proportional Hazards** (regression model for survival with covariates).

## Why it matters

Survival analysis is deeply connected to actuarial science -- life tables, mortality rates, and lapse studies are all survival analysis. As an actuarial practitioner, you already think in terms of hazard rates and conditional probabilities of death or decrement; survival analysis formalizes and extends this framework. In data science more broadly, it is essential for customer lifetime analysis (how long until churn?), reliability engineering, clinical trials, and any problem where you need to model duration with incomplete follow-up. Ignoring censoring leads to biased estimates -- you will systematically underestimate survival times because you only observe events for short-lived subjects.

## When to use it

- **Customer churn analysis**: How long do customers stay before canceling? Censored observations = customers still active at the analysis date.
- **Policy lapse studies**: What is the probability a policy remains in force for 5 years? How do lapse rates vary by product or demographic?
- **Mortality modeling**: Life table construction, comparing mortality across groups, actuarial experience studies.
- **Claims development**: Time from policy inception to first claim.
- **Clinical trials**: Time from treatment to recovery, relapse, or death.
- **Employee retention**: Time from hire to resignation.
- **Warranty analysis**: Time from product purchase to failure.
- **Credit risk**: Time from loan origination to default.

## How it works (intuitive)

### Core concepts

**Survival function** $S(t)$: The probability of surviving (no event) beyond time $t$.

$$S(t) = P(T > t)$$

This is a decreasing function that starts at 1 (everyone is alive at time 0) and approaches 0 (eventually everyone experiences the event).

**Hazard function** $h(t)$: The instantaneous rate of the event occurring at time $t$, given survival up to that point.

$$h(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t \mid T \ge t)}{\Delta t}$$

**Analogy**: The hazard is like the "danger level" at each moment. A bathtub-shaped hazard (high at birth, low in middle age, high in old age) describes human mortality. A constant hazard describes a light bulb (failure is equally likely at any moment). An increasing hazard describes a machine that wears out.

**Cumulative hazard** $H(t) = \int_0^t h(u) du$. Related to survival by: $S(t) = e^{-H(t)}$.

**Censoring**: The key challenge. An observation is **right-censored** when you know the subject survived at least until time $t$ but you do not know the actual event time. Reasons:
- Study ended before the event occurred.
- Subject dropped out / was lost to follow-up.
- Subject experienced a different, competing event.

### Kaplan-Meier Estimator

The Kaplan-Meier (KM) estimator is a **non-parametric** method to estimate the survival function directly from the data, properly accounting for censoring.

**Step by step:**

1. Order the observed event times: $t_1 < t_2 < \ldots < t_k$.
2. At each event time $t_j$:
   - $d_j$ = number of events at time $t_j$
   - $n_j$ = number of subjects "at risk" just before $t_j$ (still under observation, have not yet had the event or been censored)
3. The survival estimate at $t_j$ is:

$$\hat{S}(t) = \prod_{t_j \le t} \left(1 - \frac{d_j}{n_j}\right)$$

**Analogy**: Imagine a class of 100 students taking a multi-round elimination quiz. After each round, some students are eliminated (events) and some leave the room voluntarily (censored). To estimate the probability of surviving to round 5, you calculate the survival probability at each round using only the students who were still present, then multiply them together.

The KM curve is a **step function** that drops at each event time. Censored observations are typically marked with small tick marks on the curve. Wider steps between drops indicate stable periods; rapid drops indicate high-risk periods.

**Confidence intervals**: Use Greenwood's formula for the variance of $\hat{S}(t)$, which gives pointwise confidence bands around the survival curve.

### Log-Rank Test

The log-rank test compares the survival distributions of two or more groups (e.g., treated vs. control, male vs. female, product A vs. product B).

**Null hypothesis**: The survival functions are the same across groups.

**How it works:**

1. At each event time, compute the expected number of events in each group under $H_0$ (proportional to the number at risk).
2. Compare observed vs. expected events across all time points.
3. The test statistic follows a chi-squared distribution.

$$\chi^2 = \frac{\left(\sum_j (O_{1j} - E_{1j})\right)^2}{\sum_j V_j}$$

Where $O_{1j}$ is the observed events in group 1 at time $j$, $E_{1j}$ is the expected events, and $V_j$ is the variance term.

**Analogy**: The log-rank test is like comparing two groups' exam pass rates at every stage, accounting for the changing number of students in each group, and asking "is the overall pattern of elimination different?"

**Limitations**: The log-rank test tells you IF groups differ, not WHY or by how much. For that, you need the Cox model.

### Cox Proportional Hazards Model

The Cox model is **regression for survival data**. It relates covariates (features) to the hazard function:

$$h(t|X) = h_0(t) \cdot \exp(\beta_1 x_1 + \beta_2 x_2 + \ldots + \beta_p x_p)$$

Where:
- $h_0(t)$ is the **baseline hazard** -- the hazard when all covariates are zero. This is left unspecified (the model is "semi-parametric").
- $\exp(\beta_j)$ is the **hazard ratio** for a one-unit increase in $x_j$.

**Key interpretation**:
- $\exp(\beta) = 1$: No effect.
- $\exp(\beta) = 1.5$: 50% higher hazard (event happens faster).
- $\exp(\beta) = 0.7$: 30% lower hazard (event is delayed).

**Analogy**: Think of the baseline hazard as the "default speed" at which events happen. Each covariate acts as an accelerator or brake. Being a smoker ($HR = 1.8$) speeds up the clock by 80%. Taking medication ($HR = 0.6$) slows it down by 40%.

**The proportional hazards assumption**: The hazard ratio between any two individuals is **constant over time**. If smokers have 1.8x the hazard at age 30, they also have 1.8x the hazard at age 60. This is a strong assumption that must be tested.

**Fitting**: The Cox model uses **partial likelihood**, which elegantly avoids estimating $h_0(t)$. It only uses the ordering of event times and the covariates at each event time.

### Connection to Actuarial Life Tables

Actuaries have used survival analysis concepts for centuries under different names:

| Survival Analysis | Actuarial Notation | Meaning |
|---|---|---|
| $S(t)$ | ${}_tp_x$ | Probability of surviving $t$ years from age $x$ |
| $1 - S(t)$ | ${}_tq_x$ | Probability of dying within $t$ years from age $x$ |
| $h(t)$ | $\mu_x$ | Force of mortality at age $x$ |
| Hazard rate | Decrement rate | Rate of exit from a state |

The Kaplan-Meier estimator is essentially an empirical life table.

## Key concepts

### Types of Censoring

| Type | Description | Example |
|------|-------------|---------|
| Right censoring | Event time is after the observation period | Policy still active at study end |
| Left censoring | Event time is before the observation period | Disease was present before the study began |
| Interval censoring | Event time is known only to fall within an interval | Machine failed between two inspection dates |

Most survival analysis methods handle right censoring. Left and interval censoring require specialized methods.

### Hazard Ratio Interpretation

The hazard ratio (HR) from the Cox model is the most commonly reported result:

- $HR > 1$: Higher hazard, shorter survival, higher risk.
- $HR < 1$: Lower hazard, longer survival, lower risk.
- $HR = 1$: No effect.
- $95\% CI$ for HR: If it contains 1, the effect is not statistically significant.

### Concordance Index (C-index)

The C-index is the primary discrimination metric for survival models. It is the probability that, for a random pair of subjects, the one predicted to have the event first actually does.

- $C = 0.5$: Random predictions (no discrimination).
- $C = 1.0$: Perfect discrimination.
- $C > 0.7$: Generally considered acceptable in clinical/actuarial settings.

Analogous to AUC-ROC for binary classification.

### Testing the Proportional Hazards Assumption

Methods to test whether the PH assumption holds:

1. **Schoenfeld residuals**: Plot scaled Schoenfeld residuals against time for each covariate. A flat (non-trending) plot supports PH.
2. **Log-log plot**: Plot $\ln(-\ln(\hat{S}(t)))$ vs. $\ln(t)$ for each group. Parallel curves support PH.
3. **Interaction with time**: Add a covariate-time interaction term. If significant, PH is violated.

If PH is violated, consider:
- Stratified Cox model: Separate baseline hazards for different groups.
- Time-varying coefficients.
- Parametric models (Weibull, log-normal, etc.).

## Python implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import (KaplanMeierFitter, CoxPHFitter,
                       NelsonAalenFitter, WeibullFitter)
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.calibration import survival_probability_calibration

# --- Kaplan-Meier ---
kmf = KaplanMeierFitter()
kmf.fit(durations=df['duration'], event_observed=df['event'],
        label='Overall Survival')
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')

# Median survival time
print(f"Median survival: {kmf.median_survival_time_}")

# KM by group
fig, ax = plt.subplots(figsize=(10, 6))
for group in df['segment'].unique():
    mask = df['segment'] == group
    kmf.fit(df.loc[mask, 'duration'], df.loc[mask, 'event'], label=group)
    kmf.plot_survival_function(ax=ax)

# --- Log-Rank Test ---
group_a = df[df['segment'] == 'A']
group_b = df[df['segment'] == 'B']
result = logrank_test(
    group_a['duration'], group_b['duration'],
    event_observed_A=group_a['event'],
    event_observed_B=group_b['event']
)
print(f"Log-rank test p-value: {result.p_value:.4f}")

# --- Cox Proportional Hazards ---
cph = CoxPHFitter()
cph.fit(df, duration_col='duration', event_col='event',
        formula='age + gender + product_type + premium')
cph.print_summary()

# Hazard ratios
print(cph.hazard_ratios_)

# Check proportional hazards assumption
cph.check_assumptions(df, p_value_threshold=0.05, show_plots=True)

# Concordance
print(f"C-index: {cph.concordance_index_:.4f}")

# Predict survival function for a new individual
new_data = pd.DataFrame({'age': [45], 'gender': [1],
                          'product_type': [0], 'premium': [500]})
cph.predict_survival_function(new_data).plot()
```

## R implementation (if applicable)

```r
library(survival)
library(survminer)

# Kaplan-Meier
surv_obj <- Surv(time = df$duration, event = df$event)
km_fit <- survfit(surv_obj ~ segment, data = df)
ggsurvplot(km_fit, data = df, risk.table = TRUE, pval = TRUE,
           conf.int = TRUE)

# Log-rank test
survdiff(surv_obj ~ segment, data = df)

# Cox PH model
cox_model <- coxph(surv_obj ~ age + gender + product_type + premium, data = df)
summary(cox_model)

# Check PH assumption
cox.zph(cox_model)
plot(cox.zph(cox_model))

# Forest plot of hazard ratios
ggforest(cox_model, data = df)
```

## Gotchas & lessons learned

1. **Ignoring censoring biases your estimates.** If you drop censored observations or treat them as "no event," you systematically underestimate survival times and overestimate event rates. Always use proper survival methods.

2. **The proportional hazards assumption is not always valid.** Always test it. Common violations: treatment effects that wear off over time, age effects that change with duration. When PH fails, use stratification, time-varying coefficients, or parametric models.

3. **Kaplan-Meier is descriptive, not predictive.** It tells you what happened in your data but cannot incorporate covariates. For prediction, you need Cox or parametric models.

4. **Be careful with the definition of "time zero."** The clock must start at a meaningful, consistent point. For churn analysis, time zero is usually the customer's sign-up date. For mortality, it is birth or policy issue date. Inconsistent time origins lead to **immortal time bias**.

5. **Competing risks require special handling.** If a policyholder can lapse, surrender, or die, these are competing events. Standard KM and Cox treat competing events as censoring, which can bias estimates. Use the Aalen-Johansen estimator or Fine-Gray model.

6. **Small risk sets at the tail make survival estimates unreliable.** When only a few subjects remain at risk, the KM curve becomes very noisy. Report confidence intervals and note the number at risk at key time points.

7. **Tied event times are common in discrete data.** When multiple events occur at the same time, different tie-handling methods (Breslow, Efron, exact) give slightly different results. Efron's method is a good default.

8. **In actuarial contexts, distinguish between calendar time and policy time.** A policy issued in 2020 and observed in 2023 has 3 years of policy time (duration), but the calendar period matters for trend analysis.

9. **The C-index is threshold-free.** Unlike accuracy (which requires choosing a cutoff), the C-index evaluates the ranking of predicted risks. But it does not tell you about calibration -- the predicted survival probabilities might be wrong in absolute terms while the ranking is correct.

10. **Consider time-varying covariates.** If a covariate changes over time (e.g., a policyholder's BMI, a company's credit rating), the standard Cox model with baseline values is misleading. Use the time-varying covariate extension.

## Resources

- **Kleinbaum & Klein, "Survival Analysis" (3rd ed.)** -- The most accessible textbook. Clear explanations and examples.
- **lifelines documentation**: [lifelines.readthedocs.io](https://lifelines.readthedocs.io/) -- Excellent Python library for survival analysis with great tutorials.
- **Hosmer, Lemeshow & May, "Applied Survival Analysis" (2nd ed.)** -- Rigorous but practical.
- **R `survival` package vignette**: The most widely used survival analysis software.
- **survminer R package**: Beautiful survival plots with ggplot2 integration.
- **Actuarial applications**: SOA experience study resources and mortality table construction guides.
- **Hyndman, "Survival analysis in R"**: Free online course.
- **Clark, "Survival Analysis Part I"**: SOA study note -- connects survival analysis to actuarial notation.
