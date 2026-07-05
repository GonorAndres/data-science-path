# Potential Outcomes Framework

## The setup

For each unit $i$, two potential outcomes:

- $Y_i(1)$: outcome if treated.
- $Y_i(0)$: outcome if not treated.

We observe only one. The fundamental problem of causal inference: we never see the counterfactual.

Observed outcome:

$$
Y_i = T_i Y_i(1) + (1 - T_i) Y_i(0)
$$

## What we can estimate

- **Individual Treatment Effect:** $\tau_i = Y_i(1) - Y_i(0)$. Never observed directly.
- **Average Treatment Effect (ATE):** $E[\tau_i]$. Observable from an RCT.
- **Conditional Average Treatment Effect (CATE):** $\tau(x) = E[\tau_i | X_i = x]$. The target of uplift modeling.
- **Average Treatment Effect on the Treated (ATT):** relevant for observational studies.

## Randomization

Random assignment ensures:

$$
(Y(1), Y(0)) \perp T
$$

So $E[Y | T = 1] - E[Y | T = 0] = E[Y(1)] - E[Y(0)] = \text{ATE}$.

Criteo data was collected from an existing campaign; treatment is randomized within certain contexts.

## The key identifying assumption

Even with randomization at the population level, for CATE we need:

**Unconfoundedness (a.k.a. ignorability):**

$$
(Y(1), Y(0)) \perp T | X
$$

Given features $X$, treatment is as-if random. This holds trivially under randomization.

**Overlap:**

$$
0 < P(T = 1 | X) < 1
$$

For every covariate region, there is some probability of treatment. Violated if the treatment was targeted based on a variable we did not observe.

**SUTVA (Stable Unit Treatment Value Assumption):**

One unit's treatment does not affect another unit's outcome. Violated when users interact (network effects, marketplace crowding). In marketing this is usually reasonable.

## Observational vs experimental

For observational data (non-randomized), unconfoundedness is a strong assumption. Methods to address:

- **Propensity score weighting.** Weight each unit by $1/P(T | X)$.
- **Matching.** Pair treated and control with similar covariates.
- **Instrumental variables.** Find a variable that affects treatment but not outcome directly.

For this project, Criteo's quasi-randomized design means we can proceed without these complications, but document the caveat.

## Cross-links

- `02_ab_testing_math.md`
- `03_uplift_modeling.md`
- `docs/theory/08_experimentation_causal_primer.md`
