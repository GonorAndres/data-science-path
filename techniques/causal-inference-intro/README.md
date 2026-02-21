# Causal Inference: Introduction

## What it is

Causal inference is the science of determining whether and how much one thing *causes* another, as opposed to merely being associated with it. While most of statistics and machine learning focuses on prediction (what will happen?), causal inference focuses on intervention (what would happen *if we did something*?). The fundamental challenge is that we can never observe both outcomes for the same unit -- the "treated" outcome and the "untreated" outcome -- at the same time. This is called the **fundamental problem of causal inference**. The entire field is built around clever strategies for overcoming this limitation using assumptions, designs, and data.

## Why it matters

- **Correlation is not causation -- and the difference matters for decisions**: A model might predict that customers with long tenure have lower churn. But does long tenure *cause* lower churn, or do low-churn-propensity customers just happen to accumulate tenure? If you design a retention strategy based on confusing these, you waste money.
- **Business decisions are interventions**: Every business decision is a causal question. "If we raise prices by 5%, will we lose more customers than we gain in revenue?" Predictive models alone cannot answer this because they are trained on observational data where price was not randomly varied.
- **Insurance and actuarial relevance**: Actuaries constantly deal with causal questions. Does telematics reduce accident rates, or do safer drivers self-select into telematics programs? Do wellness programs lower claims costs, or do healthier employees enroll? Getting the causal answer right is the difference between a profitable program and a losing one.
- **Regulatory and legal requirements**: In healthcare, insurance, and public policy, causal evidence is often required to justify interventions, rate changes, or coverage decisions. "The data shows a correlation" is not sufficient for regulators.
- **Competitive advantage**: Organizations that understand causation make better decisions than those that only understand correlation. This is an increasingly recognized source of competitive advantage in data science.

## When to use it

- You want to estimate the effect of a specific intervention, treatment, or policy change.
- You suspect that a correlation in your data is driven by confounding (a third variable causing both the treatment and the outcome).
- You cannot run a randomized experiment and must rely on observational data.
- You are evaluating a past policy change and want to know if it *caused* the observed outcomes.
- You want to reason about what would happen under a counterfactual scenario ("What if we had not changed the pricing?").
- You are asked to justify a business decision with causal evidence rather than just statistical association.

## How it works (intuitive)

### The ice cream and drowning example

Here is the classic example of why correlation is not causation: ice cream sales and drowning deaths are strongly correlated. More ice cream sold = more drownings. Does ice cream cause drowning? Of course not. The **confounder** is summer weather: hot days cause both more ice cream consumption and more swimming, which leads to more drownings.

If you naively built a model predicting drowning risk from ice cream sales, it would "work" (high correlation), but banning ice cream would not reduce drownings. The model captures correlation, not causation.

Causal inference is the toolkit for distinguishing the ice cream story (confounded association) from a genuine causal relationship.

### The potential outcomes framework (Rubin Causal Model)

The most widely used framework for thinking about causation was formalized by Donald Rubin:

**For each unit $i$, there are two potential outcomes**:
- $Y_i(1)$: the outcome if unit $i$ receives the treatment.
- $Y_i(0)$: the outcome if unit $i$ does not receive the treatment.

**The individual causal effect** is the difference:

$$\tau_i = Y_i(1) - Y_i(0)$$

**The fundamental problem**: We only observe one of these outcomes for each unit. If a customer received a discount, we see $Y_i(1)$ but not $Y_i(0)$ (what would have happened without the discount). The unobserved outcome is called the **counterfactual**.

**The goal**: Estimate the **Average Treatment Effect (ATE)** over the population:

$$ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]$$

Since we cannot observe both potential outcomes for any individual, we need strategies to estimate the ATE from observed data.

### Why randomization solves the problem

In a randomized experiment, treatment is assigned by coin flip, so it is independent of everything else:

$$T \perp (Y(1), Y(0))$$

This means the treated group and the control group are, on average, identical in every way except the treatment. So:

$$E[Y | T=1] - E[Y | T=0] = E[Y(1)] - E[Y(0)] = ATE$$

Randomization eliminates confounding by design.

### When you cannot randomize: the five main strategies

When randomization is not possible, causal inference offers several strategies, each with different assumptions and data requirements:

| Strategy | Key idea | When it works |
|----------|---------|---------------|
| **Conditioning on confounders** (regression, matching, IPW) | Control for all variables that confound the treatment-outcome relationship | When you believe all confounders are observed |
| **Difference-in-differences** | Compare before/after changes between treated and untreated groups | When a treatment is applied to one group but not another, at a specific time |
| **Instrumental variables** | Find a variable that affects treatment but not the outcome (except through treatment) | When you have a valid instrument and unobserved confounders exist |
| **Regression discontinuity** | Exploit a sharp cutoff rule for treatment assignment | When treatment is assigned based on a threshold (e.g., credit score > 700) |
| **Synthetic control** | Construct a "synthetic" version of the treated unit from untreated units | When a treatment is applied to an entire unit (a state, a company) |

### Directed Acyclic Graphs (DAGs)

DAGs are visual tools for encoding causal assumptions about the world. They make your assumptions explicit and help you determine what you need to control for (and what you should *not* control for).

**Nodes** = variables. **Arrows** = direct causal effects.

**Example: Insurance telematics and claims**

```
                  [Driving Behavior]
                  /               \
                 v                 v
  [Enrolls in Telematics]     [Claims Frequency]
```

In this DAG, "Driving Behavior" is a confounder: it causes both telematics enrollment (safe drivers are more likely to enroll) and claims frequency (safe drivers have fewer claims). If you compare enrollees to non-enrollees without adjusting for driving behavior, you will overestimate the causal effect of telematics on claims.

**Three fundamental structures in DAGs**:

1. **Fork (confounder)**: $X \leftarrow Z \rightarrow Y$. Z causes both X and Y. Controlling for Z removes the spurious association.

2. **Chain (mediator)**: $X \rightarrow Z \rightarrow Y$. X causes Z, which causes Y. Controlling for Z blocks the causal pathway (do NOT control for mediators if you want the total effect).

3. **Collider**: $X \rightarrow Z \leftarrow Y$. Both X and Y cause Z. Controlling for Z *creates* a spurious association (collider bias). Do NOT control for colliders.

**The backdoor criterion**: To estimate the causal effect of $X$ on $Y$, you need to block all "backdoor paths" (non-causal paths from $X$ to $Y$) by conditioning on the appropriate set of variables. DAGs tell you exactly which variables to condition on.

## Key concepts

### Potential outcomes notation

| Symbol | Meaning |
|--------|---------|
| $Y_i(1)$ | Outcome for unit $i$ if treated |
| $Y_i(0)$ | Outcome for unit $i$ if not treated |
| $\tau_i = Y_i(1) - Y_i(0)$ | Individual treatment effect (never observed) |
| $ATE = E[\tau_i]$ | Average Treatment Effect |
| $ATT = E[\tau_i | T_i = 1]$ | Average Treatment Effect on the Treated |
| $CATE = E[\tau_i | X_i = x]$ | Conditional ATE (effect for subgroup with $X = x$) |

### Unconfoundedness (ignorability) assumption

$$(Y(1), Y(0)) \perp T | X$$

Translation: After conditioning on observed covariates $X$, treatment assignment is as good as random. This is the key assumption for matching, regression adjustment, and inverse probability weighting.

This assumption is **untestable** -- you can never verify that you have included all confounders. Sensitivity analysis is essential.

### Difference-in-Differences (DiD)

DiD is used when:
- A treatment is applied to one group (treated) but not another (control) at a specific time.
- You have data on both groups before and after the treatment.

**Setup**:

| | Before treatment | After treatment |
|---|---|---|
| Treated group | $\bar{Y}_{T, pre}$ | $\bar{Y}_{T, post}$ |
| Control group | $\bar{Y}_{C, pre}$ | $\bar{Y}_{C, post}$ |

**DiD estimator**:

$$\hat{\tau}_{DiD} = (\bar{Y}_{T, post} - \bar{Y}_{T, pre}) - (\bar{Y}_{C, post} - \bar{Y}_{C, pre})$$

In regression form:

$$Y_{it} = \alpha + \beta_1 \cdot T_i + \beta_2 \cdot Post_t + \beta_3 \cdot (T_i \times Post_t) + \epsilon_{it}$$

The coefficient $\beta_3$ is the DiD estimate of the treatment effect.

**Key assumption: Parallel trends**. In the absence of treatment, the treated and control groups would have followed the same trend over time. This is partially testable by examining pre-treatment trends.

**Insurance example**: A state mandates a new safety regulation. You compare claim rates in that state (treated) to neighboring states (control), before and after the regulation. The DiD estimate tells you how much the regulation changed claim rates, after removing the common trend.

### Instrumental Variables (IV)

IV is used when you suspect unobserved confounders but have access to a variable (the instrument, $Z$) that:
1. Affects the treatment ($Z \rightarrow T$) -- "relevance"
2. Does not affect the outcome except through the treatment ($Z \not\rightarrow Y$ except via $T$) -- "exclusion restriction"
3. Is not caused by the same confounders as the outcome -- "independence"

```
Z ---> T ---> Y
              ^
              |
         [Unobserved confounders]
              |
              v
              T
```

**Two-Stage Least Squares (2SLS)**:
1. **First stage**: Regress $T$ on $Z$ to get predicted treatment $\hat{T}$.
2. **Second stage**: Regress $Y$ on $\hat{T}$.

The second-stage coefficient estimates the **Local Average Treatment Effect (LATE)** -- the effect for units whose treatment status was changed by the instrument ("compliers").

**Classic example**: Distance to a college as an instrument for education's effect on earnings. Living closer to a college makes you more likely to attend (relevance), but distance itself does not directly affect your future earnings (exclusion restriction -- debatable).

**Insurance example**: A random premium discount coupon as an instrument for insurance take-up. The coupon affects purchase probability (relevance) but should not directly affect claims behavior (exclusion restriction).

### Regression Discontinuity Design (RDD)

RDD exploits situations where treatment is assigned based on whether a continuous "running variable" crosses a threshold.

**Setup**: Treatment is assigned when $X_i \geq c$ (e.g., customers with a risk score above 700 get a premium discount).

**Key insight**: Customers just above and just below the threshold are nearly identical -- they differ only in whether they received the treatment. So comparing outcomes for customers at $X = c + \epsilon$ vs. $X = c - \epsilon$ gives a local causal estimate.

$$\hat{\tau}_{RDD} = \lim_{x \downarrow c} E[Y | X = x] - \lim_{x \uparrow c} E[Y | X = x]$$

In practice, fit separate regressions on each side of the cutoff and compare predicted values at the threshold.

**Types**:
- **Sharp RDD**: Treatment is perfectly determined by the cutoff (everyone above gets it, no one below does).
- **Fuzzy RDD**: The cutoff changes the probability of treatment but does not determine it perfectly (analogous to IV with the cutoff as the instrument).

**Insurance example**: An insurer offers a "good driver" discount to customers with a safety score above 750. Comparing claim rates for customers just above and below 750 estimates the causal effect of the discount on claims behavior.

### Concrete examples: correlation vs. causation

| Observation | Naive interpretation | Causal reality | Why it matters |
|-------------|---------------------|----------------|----------------|
| Customers who called support have higher churn | Calling support causes churn | Unhappy customers both call support and churn (confounder: dissatisfaction) | Reducing support calls would not reduce churn |
| Insured drivers with telematics have fewer accidents | Telematics causes safer driving | Safe drivers self-select into telematics (confounder: driving behavior) | The program benefit is smaller than naive comparison suggests |
| Regions with more police have more crime | Police cause crime | High-crime areas get more police (reverse causation) | Reducing police would not reduce crime |
| Students who use tutoring score higher | Tutoring causes higher scores | Motivated students both seek tutoring and study more (confounder: motivation) | The tutoring effect is smaller than the raw comparison suggests |

## Python implementation

```python
# --- Difference-in-Differences ---
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# df has columns: outcome, treated (0/1), post (0/1)
did_model = smf.ols('outcome ~ treated * post', data=df).fit()
print(did_model.summary())
# The interaction coefficient (treated:post) is the DiD estimate

# --- Instrumental Variables (2SLS) ---
from linearmodels.iv import IV2SLS

# Y ~ endogenous_var, instrumented by instrument_var, controlling for controls
iv_model = IV2SLS(
    dependent=df['outcome'],
    exog=df[['const'] + control_vars],
    endog=df['treatment'],
    instruments=df['instrument']
).fit()
print(iv_model.summary)

# --- Regression Discontinuity ---
# pip install rdrobust (R port) or use manual implementation
from sklearn.linear_model import LinearRegression

cutoff = 700
bandwidth = 50

# Subset to bandwidth around cutoff
rdd_data = df[(df['running_var'] >= cutoff - bandwidth) &
              (df['running_var'] <= cutoff + bandwidth)].copy()
rdd_data['above_cutoff'] = (rdd_data['running_var'] >= cutoff).astype(int)
rdd_data['centered'] = rdd_data['running_var'] - cutoff

# Regression with interaction
rdd_model = smf.ols(
    'outcome ~ above_cutoff * centered',
    data=rdd_data
).fit()
print(rdd_model.summary())
# Coefficient on 'above_cutoff' is the RDD estimate at the cutoff

# --- Using DoWhy for causal inference ---
# pip install dowhy
import dowhy
from dowhy import CausalModel

model = CausalModel(
    data=df,
    treatment='treatment',
    outcome='outcome',
    common_causes=['age', 'tenure', 'risk_score'],  # confounders
    # Optional: specify the causal graph
    graph="digraph {age -> treatment; age -> outcome; "
          "tenure -> treatment; tenure -> outcome; "
          "risk_score -> treatment; risk_score -> outcome; "
          "treatment -> outcome;}"
)

# Identify the causal effect
identified = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate using different methods
estimate_matching = model.estimate_effect(
    identified,
    method_name="backdoor.propensity_score_matching"
)
estimate_ipw = model.estimate_effect(
    identified,
    method_name="backdoor.propensity_score_weighting"
)
estimate_did = model.estimate_effect(
    identified,
    method_name="backdoor.linear_regression"
)

# Refutation tests (sensitivity analysis)
refute_placebo = model.refute_estimate(
    identified, estimate_matching,
    method_name="placebo_treatment_refuter"
)
refute_subset = model.refute_estimate(
    identified, estimate_matching,
    method_name="data_subset_refuter"
)
print(refute_placebo)
print(refute_subset)
```

Key libraries:
- **DoWhy**: Microsoft's causal inference library. Supports identification, estimation, and refutation. [DoWhy docs](https://www.pywhy.org/dowhy/)
- **EconML**: Microsoft's library for heterogeneous treatment effects and CATE estimation. [EconML docs](https://econml.azurewebsites.net/)
- **causalinference**: Clean API for matching and IPW. [causalinferenceinpython.org](https://causalinferenceinpython.org/)
- **linearmodels**: Instrumental variables (IV2SLS). [linearmodels docs](https://bashtage.github.io/linearmodels/)
- **statsmodels**: OLS regression for DiD and basic causal models.
- **dagitty** (R, with Python wrappers): DAG construction and analysis.

## R implementation (if applicable)

```r
# Difference-in-Differences
did_model <- lm(outcome ~ treated * post, data = df)
summary(did_model)

# Instrumental Variables
library(AER)
iv_model <- ivreg(outcome ~ treatment + controls | instrument + controls, data = df)
summary(iv_model, diagnostics = TRUE)

# Regression Discontinuity
library(rdrobust)
rd <- rdrobust(y = df$outcome, x = df$running_var, c = cutoff)
summary(rd)
rdplot(y = df$outcome, x = df$running_var, c = cutoff)

# DAGs
library(dagitty)
dag <- dagitty('dag {
  Age -> Treatment
  Age -> Outcome
  Treatment -> Outcome
}')
adjustmentSets(dag, "Treatment", "Outcome")
```

**Python vs. R tradeoff**: R has more mature packages for specific methods (`rdrobust` for RDD, `AER` for IV, `dagitty` for DAGs, `MatchIt` for matching). Python's DoWhy provides a unified framework that covers all methods in one API, which is more convenient for exploration. For production pipelines, Python is preferred. For research and diagnostics, R often has deeper tools.

## Gotchas & lessons learned

### 1. Correlation vs. causation is not a slogan -- it is the central problem
Most practitioners understand the phrase but underestimate how pervasive confounding is. Even in "obviously causal" relationships, confounders lurk. Always ask: "What else could explain this pattern?"

### 2. No method eliminates the need for assumptions
Every causal inference method relies on untestable assumptions (no unobserved confounders for matching/regression, parallel trends for DiD, exclusion restriction for IV, continuity for RDD). Be explicit about your assumptions and conduct sensitivity analyses.

### 3. Do not control for colliders or post-treatment variables
Controlling for a collider (a variable caused by both treatment and outcome) creates a spurious association. Controlling for a variable affected by the treatment (a mediator or post-treatment variable) blocks part of the causal pathway. Draw a DAG before deciding what to control for.

**Example of collider bias**: If you control for "filed a claim" when estimating the effect of telematics on claim severity, you are conditioning on a collider (telematics and severity both affect whether a claim is filed). This biases the estimate.

### 4. External validity vs. internal validity
Many causal methods give you a clean estimate for a specific subgroup (internal validity) but do not generalize broadly (external validity). RDD gives you the effect at the cutoff. IV gives you the LATE for compliers. Be honest about what population your estimate applies to.

### 5. Difference-in-differences requires careful parallel trends checking
If the parallel trends assumption is violated (the treated and control groups were already diverging before the treatment), DiD is biased. Always plot pre-treatment trends and test for parallel trends. Consider using synthetic control methods as an alternative.

### 6. Weak instruments cause more harm than good
An instrumental variable that barely predicts the treatment (weak instrument) leads to severely biased IV estimates -- often more biased than naive OLS. The first-stage F-statistic should be above 10 (Staiger-Stock rule of thumb). Report it.

### 7. The role of domain knowledge
Causal inference is not a purely statistical exercise. Choosing confounders, drawing DAGs, selecting instruments, and judging the plausibility of assumptions all require deep domain knowledge. A causal analysis by someone who understands the business is worth ten times more than one by someone who only knows the statistical methods.

### 8. Heterogeneous treatment effects are the norm
The average treatment effect often masks important variation. A new pricing strategy might reduce churn for price-sensitive customers but increase it for quality-focused customers. Explore conditional average treatment effects (CATE) using methods like causal forests (EconML) or stratified analysis.

### 9. Reproducibility and pre-registration
Causal analyses are prone to "researcher degrees of freedom" -- different choices of confounders, bandwidths, instruments, or specifications can lead to different conclusions. Pre-registering your analysis plan (even informally, in a document before looking at the data) greatly increases credibility.

### 10. Start with the simplest method
Before reaching for sophisticated methods, check if a simple approach works:
- Can you run an A/B test? Do that first.
- Is there a natural experiment? Use DiD.
- Is there a threshold? Use RDD.
- Can you plausibly identify all confounders? Use matching or regression.
- Only resort to IV when you have a convincing instrument.

## Resources

- **Hernan & Robins**: "Causal Inference: What If" -- free textbook, the definitive reference. [Link](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
- **Cunningham (2021)**: "Causal Inference: The Mixtape" -- free, accessible, with code examples. [Link](https://mixtape.scunning.com/)
- **Huntington-Klein (2022)**: "The Effect: An Introduction to Research Design and Causality" -- excellent introductory text with R and Python code. [Link](https://theeffectbook.net/)
- **Pearl (2009)**: "Causality: Models, Reasoning, and Inference" -- the foundational book on DAGs and structural causal models.
- **Angrist & Pischke**: "Mostly Harmless Econometrics" -- the econometric perspective on causal inference (IV, DiD, RDD).
- **DoWhy documentation**: [pywhy.org/dowhy](https://www.pywhy.org/dowhy/) -- practical tutorials.
- **EconML documentation**: [econml.azurewebsites.net](https://econml.azurewebsites.net/) -- heterogeneous treatment effects.
- **dagitty**: [dagitty.net](https://www.dagitty.net/) -- online tool for drawing and analyzing DAGs.
- **Brady Neal**: "Introduction to Causal Inference from a Machine Learning Perspective" -- free course notes. [Link](https://www.bradyneal.com/causal-inference-course)
- **Coursera**: "Crash Course in Causality" by University of Pennsylvania.
