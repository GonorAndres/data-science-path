---
tags: [causal-inference, econometrics, experimentation]
status: complete
created: 2026-04-23
updated: 2026-04-25
---

# Causal Inference

## What it is

Causal inference is the science of determining whether and how much one thing *causes* another, as opposed to merely being associated with it. While most of statistics and machine learning focuses on prediction (what will happen?), causal inference focuses on intervention (what would happen *if we did something*?). The fundamental challenge is that we can never observe both outcomes for the same unit, the "treated" outcome and the "untreated" outcome, at the same time. This is the **fundamental problem of causal inference**. The entire field is built around clever strategies for overcoming this limitation using assumptions, designs, and data.

## Why it matters

- **Correlation is not causation, and the difference matters for decisions.** A model might predict that customers with long tenure have lower churn. But does long tenure *cause* lower churn, or do low-churn-propensity customers just happen to accumulate tenure? If you design a retention strategy based on confusing these, you waste money.
- **Business decisions are interventions.** Every business decision is a causal question. "If we raise prices by 5%, will we lose more customers than we gain in revenue?" Predictive models alone cannot answer this because they are trained on observational data where price was not randomly varied.
- **Insurance and actuarial relevance.** Actuaries constantly deal with causal questions. Does telematics reduce accident rates, or do safer drivers self-select into telematics programs? Do wellness programs lower claims costs, or do healthier employees enroll? Getting the causal answer right is the difference between a profitable program and a losing one.
- **Regulatory and legal requirements.** In healthcare, insurance, and public policy, causal evidence is often required to justify interventions, rate changes, or coverage decisions. "The data shows a correlation" is not sufficient for regulators.
- **Competitive advantage.** Organizations that understand causation make better decisions than those that only understand correlation.

## When to use it

- You want to estimate the effect of a specific intervention, treatment, or policy change.
- You suspect that a correlation in your data is driven by confounding (a third variable causing both the treatment and the outcome).
- You cannot run a randomized experiment and must rely on observational data.
- You are evaluating a past policy change and want to know if it *caused* the observed outcomes.
- You want to reason about what would happen under a counterfactual scenario.
- You are asked to justify a business decision with causal evidence rather than just statistical association.

## How it works (intuitive)

### The ice cream and drowning example

Ice cream sales and drowning deaths are strongly correlated. More ice cream sold = more drownings. Does ice cream cause drowning? Of course not. The **confounder** is summer weather: hot days cause both more ice cream consumption and more swimming, which leads to more drownings.

If you naively built a model predicting drowning risk from ice cream sales, it would "work" (high correlation), but banning ice cream would not reduce drownings. The model captures correlation, not causation.

### The potential outcomes framework (Rubin Causal Model)

For each unit $i$, there are two potential outcomes:

- $Y_i(1)$: the outcome if unit $i$ receives the treatment.
- $Y_i(0)$: the outcome if unit $i$ does not receive the treatment.

The individual causal effect is the difference $\tau_i = Y_i(1) - Y_i(0)$.

The fundamental problem: we only observe one of these outcomes for each unit. The unobserved one is the **counterfactual**. The goal is to estimate the **Average Treatment Effect (ATE)**:

$$ATE = E[Y(1) - Y(0)] = E[Y(1)] - E[Y(0)]$$

### Why randomization solves the problem

In a randomized experiment, treatment is assigned by coin flip, so it is independent of everything else: $T \perp (Y(1), Y(0))$. The treated and control groups are, on average, identical except for the treatment, and so:

$$E[Y \mid T=1] - E[Y \mid T=0] = E[Y(1)] - E[Y(0)] = ATE$$

Randomization eliminates confounding by design.

### When you cannot randomize: the strategies

| Strategy | Key idea | When it works |
|----------|----------|---------------|
| Conditioning on confounders (regression, matching, IPW) | Control for all variables that confound treatment-outcome | When you believe all confounders are observed |
| Difference-in-differences | Compare before/after changes between treated and control groups | A treatment is applied to one group at a specific time |
| Instrumental variables | Find a variable that affects treatment but not outcome (except through treatment) | You have a valid instrument and unobserved confounders exist |
| Regression discontinuity | Exploit a sharp cutoff rule for treatment assignment | Treatment is assigned based on a threshold |
| Synthetic control | Construct a "synthetic" version of the treated unit from untreated units | Treatment applied to an entire unit (state, company) |
| DML / Orthogonal ML | Residualize treatment and outcome on confounders, then regress | Many confounders, flexible nuisance models needed |
| Causal forests | Honest random forests for heterogeneous treatment effects | You need CATE with confidence intervals |

## DAGs: encoding causal assumptions

Directed Acyclic Graphs make assumptions explicit and tell you what to control for and what not to control for.

**Nodes** = variables. **Arrows** = direct causal effects.

Example. Insurance telematics and claims:

```
              [Driving Behavior]
              /               \
             v                 v
[Enrolls in Telematics]   [Claims Frequency]
                  \           /
                   v         v
                 [Claims Frequency]
```

"Driving Behavior" is a confounder: it causes both telematics enrollment and claims frequency. Comparing enrollees to non-enrollees without adjusting for driving behavior overestimates the causal effect of telematics on claims.

### Three fundamental structures

1. **Fork (confounder).** $X \leftarrow Z \rightarrow Y$. Z causes both X and Y. Controlling for Z removes the spurious association.
2. **Chain (mediator).** $X \rightarrow Z \rightarrow Y$. X causes Z, which causes Y. Controlling for Z blocks the causal pathway. Do NOT control for mediators if you want the total effect.
3. **Collider.** $X \rightarrow Z \leftarrow Y$. Both X and Y cause Z. Controlling for Z *creates* a spurious association (collider bias). Do NOT control for colliders.

### d-separation

Two nodes are **d-separated** by a set $S$ if every path between them is blocked. A path is blocked when:

- It contains a chain or fork with a node in $S$ on the path.
- It contains a collider whose collider node and all its descendants are not in $S$.

d-separation is the formal definition of conditional independence implied by the DAG.

### The back-door criterion

To estimate the causal effect of $X$ on $Y$, find a set $S$ that:

1. Blocks every back-door path from $X$ to $Y$ (paths that begin with an arrow into $X$).
2. Contains no descendants of $X$.

Conditioning on $S$ identifies the causal effect: $E[Y \mid \text{do}(X)] = \sum_S E[Y \mid X, S] P(S)$. The back-door criterion is the formal justification for "control for these confounders."

Worked example. Treatment $T$ (a marketing campaign), outcome $Y$ (purchase), confounders age and tenure (cause both $T$ and $Y$). The back-door set is $\{age, tenure\}$. Controlling for them via regression, matching, or IPW identifies the ATE.

### The front-door criterion

When you cannot block back-door paths because some confounders are unobserved, but there is a complete causal mediator $M$ between $X$ and $Y$ with no unobserved confounders on the $X \to M$ and $M \to Y$ paths, then:

$$E[Y \mid \text{do}(X)] = \sum_M P(M \mid X) \sum_{X'} E[Y \mid X', M] P(X')$$

Worked example. Smoking ($X$) causes cancer ($Y$); the unobserved confounder is genetic susceptibility. Tar in lungs ($M$) is a mediator caused only by smoking and unaffected by genetics. The front-door formula identifies the smoking-cancer effect using $X, M, Y$ alone, even with unobserved $U$.

The front-door criterion is rare in practice but conceptually important: it shows that causal identification does not require controlling for every confounder.

## Key concepts (potential outcomes notation)

| Symbol | Meaning |
|--------|---------|
| $Y_i(1)$ | Outcome for unit $i$ if treated |
| $Y_i(0)$ | Outcome for unit $i$ if not treated |
| $\tau_i = Y_i(1) - Y_i(0)$ | Individual treatment effect (never observed) |
| $ATE = E[\tau_i]$ | Average Treatment Effect |
| $ATT = E[\tau_i \mid T_i = 1]$ | Average Treatment Effect on the Treated |
| $CATE(x) = E[\tau_i \mid X_i = x]$ | Conditional ATE (effect for subgroup with $X = x$) |

### Unconfoundedness (ignorability)

$$(Y(1), Y(0)) \perp T \mid X$$

After conditioning on observed covariates $X$, treatment assignment is as good as random. **Untestable**; sensitivity analysis is essential.

## Difference-in-Differences (DiD)

Setup: a treatment is applied to one group (treated) but not another (control) at a specific time, and you have data on both groups before and after.

DiD estimator:

$$\hat{\tau}_{DiD} = (\bar{Y}_{T, post} - \bar{Y}_{T, pre}) - (\bar{Y}_{C, post} - \bar{Y}_{C, pre})$$

In regression form:

$$Y_{it} = \alpha + \beta_1 T_i + \beta_2 \text{Post}_t + \beta_3 (T_i \times \text{Post}_t) + \epsilon_{it}$$

The interaction coefficient $\beta_3$ is the DiD estimate.

**Key assumption: parallel trends.** In the absence of treatment, the treated and control groups would have followed the same trend. Partially testable by examining pre-treatment trends.

## Instrumental Variables (IV)

Use an instrument $Z$ such that:

1. $Z$ affects $T$ (relevance).
2. $Z$ affects $Y$ only through $T$ (exclusion restriction).
3. $Z$ is independent of unobserved confounders (independence).

Two-stage least squares (2SLS): regress $T$ on $Z$, then regress $Y$ on the predicted $\hat{T}$. The coefficient estimates the **Local Average Treatment Effect (LATE)**: the effect for compliers (units whose treatment status was changed by the instrument).

Diagnostics: report the first-stage F-statistic; values below 10 indicate a weak instrument and biased IV estimates (Staiger-Stock).

## Regression Discontinuity Design (RDD)

Treatment is assigned when a running variable crosses a threshold $c$. Compare outcomes just above and just below $c$:

$$\hat{\tau}_{RDD} = \lim_{x \downarrow c} E[Y \mid X = x] - \lim_{x \uparrow c} E[Y \mid X = x]$$

Sharp RDD: treatment is perfectly determined by the cutoff. Fuzzy RDD: the cutoff changes the probability of treatment but does not determine it perfectly (use the cutoff as an instrument).

## Double / Debiased Machine Learning (DML)

When confounders are many, high-dimensional, or interact non-linearly, parametric regression is misspecified and ML must enter, but naive ML for the outcome model produces biased treatment effect estimates due to **regularization bias**: the ML model shrinks important confounders to fit better, biasing the residual partition.

DML (Chernozhukov et al. 2018) solves this with a "residualize then regress" strategy. For a partially linear model $Y = T \tau + g(X) + \varepsilon$ and propensity $T = m(X) + u$:

1. **Estimate nuisance functions** $\hat{g}(X)$ and $\hat{m}(X)$ on a holdout fold using any flexible ML (random forest, gradient boosting, neural network).
2. **Residualize.** Compute $\tilde{Y} = Y - \hat{g}(X)$ and $\tilde{T} = T - \hat{m}(X)$ on the other fold.
3. **Regress.** $\hat{\tau}$ is the OLS coefficient of $\tilde{Y}$ on $\tilde{T}$.
4. **Cross-fit.** Swap the folds and repeat; average the two estimates.

Why it works. The key property is **Neyman orthogonality**: the moment condition that defines $\hat{\tau}$ is insensitive to small errors in the nuisance functions. ML errors of order $o(n^{-1/4})$ in $\hat{g}$ and $\hat{m}$ still leave $\hat{\tau}$ asymptotically normal at the parametric $\sqrt{n}$ rate. Cross-fitting prevents the ML model from overfitting to the same data used for the second stage, which would reintroduce bias.

DML extends to interactive models (heterogeneous treatment effects), instrumental variables (DML-IV), and panel data (DML-DiD). The Microsoft `EconML` library implements all the variants with a uniform sklearn-style API.

```python
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

est = LinearDML(
    model_y=RandomForestRegressor(n_estimators=200),
    model_t=RandomForestClassifier(n_estimators=200),
    discrete_treatment=True,
    cv=5,
)
est.fit(Y, T, X=X, W=W)  # X = effect modifiers, W = controls only
print(est.effect(X)[:10])      # CATE per unit
print(est.effect_inference(X).summary_frame().head())
```

## Causal forests

When the heterogeneous effect $\tau(x) = E[Y(1) - Y(0) \mid X = x]$ is the object of interest, **causal forests** (Athey, Wager 2019) extend random forests to estimate $\tau(x)$ with valid confidence intervals.

Two innovations vs vanilla random forest:

- **Honesty.** Each tree is built using one subsample for splitting decisions and a different subsample for computing leaf-node estimates. This breaks the dependence between split selection and the estimate, and is what gives causal forests their valid confidence intervals.
- **Splitting on treatment-effect heterogeneity.** Trees split to maximize the difference between estimated treatment effects in the child nodes, rather than minimizing prediction error.

The leaf-level estimate at $x$ is a local averaging of treatment effects from the subsample of training data falling into the same leaves as $x$. Confidence intervals are constructed via the infinitesimal jackknife.

```r
library(grf)
cf <- causal_forest(X = covariates, Y = outcome, W = treatment, num.trees = 5000)
cate_hat <- predict(cf, X_new)$predictions
ci <- predict(cf, X_new, estimate.variance = TRUE)
# 95% CI: cate_hat +/- 1.96 * sqrt(ci$variance.estimates)
```

In Python, use `econml.dml.CausalForestDML` or `causalml.inference.tree.CausalRandomForestRegressor`. Both implement honest splitting.

When to use causal forests:

- You believe heterogeneity is real and want to characterize who benefits most.
- You need confidence intervals on CATE, not just point estimates.
- The covariate space is moderate (fewer than a few hundred features); higher-dimensional cases need DML or doubly-robust learners.

When not to use causal forests: very high-dimensional covariates, very small samples (under 1000), or when the back-door criterion fails (use IV-style methods instead).

## Concrete examples: correlation vs causation

| Observation | Naive interpretation | Causal reality | Why it matters |
|-------------|---------------------|----------------|----------------|
| Customers who called support have higher churn | Calling support causes churn | Unhappy customers both call support and churn | Reducing support calls would not reduce churn |
| Insured drivers with telematics have fewer accidents | Telematics causes safer driving | Safe drivers self-select into telematics | The program benefit is smaller than naive comparison suggests |
| Regions with more police have more crime | Police cause crime | High-crime areas get more police (reverse causation) | Reducing police would not reduce crime |
| Students who use tutoring score higher | Tutoring causes higher scores | Motivated students both seek tutoring and study more | The tutoring effect is smaller than the raw comparison suggests |

## Python implementation

```python
# Difference-in-Differences
import statsmodels.formula.api as smf
did_model = smf.ols("outcome ~ treated * post", data=df).fit()
print(did_model.summary())

# Instrumental Variables (2SLS)
from linearmodels.iv import IV2SLS
iv_model = IV2SLS(
    dependent=df["outcome"],
    exog=df[["const"] + control_vars],
    endog=df["treatment"],
    instruments=df["instrument"],
).fit()
print(iv_model.summary)

# Regression Discontinuity
cutoff, bandwidth = 700, 50
rdd_data = df.query("@cutoff - @bandwidth <= running_var <= @cutoff + @bandwidth").copy()
rdd_data["above_cutoff"] = (rdd_data["running_var"] >= cutoff).astype(int)
rdd_data["centered"]    = rdd_data["running_var"] - cutoff
rdd_model = smf.ols("outcome ~ above_cutoff * centered", data=rdd_data).fit()
print(rdd_model.summary())

# DoWhy (full pipeline)
from dowhy import CausalModel
model = CausalModel(
    data=df, treatment="treatment", outcome="outcome",
    common_causes=["age", "tenure", "risk_score"],
)
identified = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(identified,
                                 method_name="backdoor.propensity_score_weighting")
refute = model.refute_estimate(identified, estimate,
                               method_name="placebo_treatment_refuter")

# DML
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
est = LinearDML(model_y=RandomForestRegressor(),
                model_t=RandomForestClassifier(),
                discrete_treatment=True, cv=5)
est.fit(df["outcome"], df["treatment"], X=df[effect_modifiers], W=df[controls])

# Causal forest
from econml.dml import CausalForestDML
cf = CausalForestDML(model_y=RandomForestRegressor(),
                     model_t=RandomForestClassifier(),
                     discrete_treatment=True, n_estimators=2000)
cf.fit(df["outcome"], df["treatment"], X=df[features])
cate = cf.effect(df[features])
```

Key libraries: `dowhy` (full pipeline + refutation), `econml` (DML, causal forests, IV), `linearmodels` (IV2SLS, panel methods), `causalml` (Uber, uplift modeling), `statsmodels` (DiD, OLS).

## R implementation

```r
# DiD
did_model <- lm(outcome ~ treated * post, data = df)

# IV
library(AER)
iv_model <- ivreg(outcome ~ treatment + controls | instrument + controls, data = df)
summary(iv_model, diagnostics = TRUE)

# RDD
library(rdrobust)
rd <- rdrobust(y = df$outcome, x = df$running_var, c = cutoff)

# Causal forest
library(grf)
cf <- causal_forest(X, Y, W, num.trees = 5000)
predict(cf, X_new, estimate.variance = TRUE)

# DAGs
library(dagitty)
dag <- dagitty('dag { Age -> Treatment; Age -> Outcome; Treatment -> Outcome }')
adjustmentSets(dag, "Treatment", "Outcome")
```

## Gotchas and lessons learned

1. **Correlation vs causation is the central problem, not a slogan.** Confounders are pervasive; always ask "what else could explain this?".
2. **No method eliminates the need for assumptions.** Be explicit; conduct sensitivity analyses.
3. **Do not control for colliders or post-treatment variables.** Draw a DAG before deciding what to control for.
4. **Internal vs external validity.** RDD gives the effect at the cutoff; IV gives LATE for compliers. Be honest about the population.
5. **DiD requires careful parallel-trends checking.** Plot pre-treatment trends; consider synthetic control alternatives.
6. **Weak instruments cause more harm than good.** First-stage F above 10.
7. **Domain knowledge is not optional.** Choosing confounders, instruments, and judging assumptions requires substantive knowledge.
8. **Heterogeneous effects are the norm.** Use CATE methods (causal forests, DML) to expose them.
9. **Pre-register the analysis plan.** Reduces researcher degrees of freedom.
10. **Start with the simplest method.** RCT, then DiD, then RDD, then matching/regression, then IV, then DML/causal forest as the data and confounding pattern dictates.

## Resources

- `docs/theory/08_experimentation_causal_primer.md` for the theory companion.
- `projects/causal-uplift-marketing/` for a portfolio worked example.
- Hernan, Robins, *Causal Inference: What If*, free online.
- Cunningham, *Causal Inference: The Mixtape*, free online.
- Huntington-Klein, *The Effect: An Introduction to Research Design and Causality*.
- Pearl, *Causality: Models, Reasoning, and Inference*, 2nd ed., 2009.
- Angrist, Pischke, *Mostly Harmless Econometrics*.
- Brady Neal, "Introduction to Causal Inference from a Machine Learning Perspective", free course notes.

## References

- Imbens, Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*, 2015.
- Pearl, *Causality*, 2009.
- Athey, Imbens, "Machine Learning Methods for Estimating Heterogeneous Causal Effects", 2016.
- Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins, "Double/Debiased Machine Learning for Treatment and Structural Parameters", Econometrics Journal 2018.
- Athey, Wager, "Estimation and inference of heterogeneous treatment effects using random forests", Journal of the American Statistical Association 2019.
- Wager, Athey, "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests", JASA 2018.
- Pearl, "Causal diagrams for empirical research", Biometrika 1995.
