# Observational Causal Inference

When randomization is impossible (ethics, legal, cost, infeasibility), causal claims must come from observational data plus an identifying assumption. This essay covers the canonical methods, their assumptions, how to check them, and the modern ML extensions (DML, causal forests, synthetic control).

## Potential outcomes framework

For each unit $i$ and treatment indicator $W_i \in \{0, 1\}$, define potential outcomes:

- $Y_i(1)$: outcome if $i$ received treatment.
- $Y_i(0)$: outcome if $i$ did not.

Only one is observed: $Y_i = W_i Y_i(1) + (1 - W_i) Y_i(0)$. The individual treatment effect is $\tau_i = Y_i(1) - Y_i(0)$, never directly observable.

Estimands:

- **Average Treatment Effect (ATE).** $\tau_{\mathrm{ATE}} = E[Y(1) - Y(0)]$.
- **ATE on the Treated (ATT).** $\tau_{\mathrm{ATT}} = E[Y(1) - Y(0) \mid W = 1]$. The effect among those who actually got treated.
- **Conditional ATE (CATE).** $\tau(x) = E[Y(1) - Y(0) \mid X = x]$. Heterogeneity by covariates.

Without additional assumptions, none of these is identified from observational data. Each method below pairs an estimand with an identifying assumption.

## Difference-in-differences (DiD)

Setup: a treatment is applied to one group at a known time $T^*$. Compare the change in outcome for the treated group vs an untreated comparison group.

$$
\hat\tau_{\mathrm{DiD}} = (\bar Y_{T,\,\text{post}} - \bar Y_{T,\,\text{pre}}) - (\bar Y_{C,\,\text{post}} - \bar Y_{C,\,\text{pre}}).
$$

### Parallel trends assumption

Identification requires that absent treatment, treated and control groups would have followed parallel trends. This is untestable for the post period, but you can test it on the pre-period:

1. Plot pre-period means by time for both groups.
2. Run a placebo DiD using a fake treatment date inside the pre-period; the estimate should be near zero.
3. Test for differential pre-trends with a regression of $Y$ on group $\times$ time interactions.

If pre-trends differ, parallel trends post-treatment is implausible. Consider a triple-differences design or matching.

### Two-way fixed effects

The DiD generalization is

$$
Y_{it} = \alpha_i + \lambda_t + \tau \cdot D_{it} + \epsilon_{it}
$$

with unit and time fixed effects. With staggered treatment adoption, recent literature (Goodman-Bacon 2021) shows that two-way FE can produce biased estimates when treatment effects vary over time; modern alternatives are Callaway-Sant'Anna and de Chaisemartin-d'Haultfoeuille estimators.

## Propensity score matching (PSM)

Estimate the probability of treatment given covariates: $e(X) = P(W = 1 \mid X)$. Match each treated unit to a control unit with similar propensity.

### Identifying assumptions

- **Ignorability (no unobserved confounders).** $(Y(1), Y(0)) \perp W \mid X$. Untestable; argue from domain knowledge.
- **Overlap (common support).** $0 < e(X) < 1$ for all $X$ in the support. Testable: plot the distribution of $\hat e(X)$ in treated vs control; trim regions with no overlap.

### Checking balance

Compute the standardized mean difference (SMD) for each covariate before and after matching:

$$
\mathrm{SMD}_j = \frac{\bar X_j^T - \bar X_j^C}{\sqrt{(s_T^2 + s_C^2)/2}}.
$$

Rule of thumb: $|\mathrm{SMD}| < 0.1$ for all $j$ post-matching means balance is acceptable. If a covariate remains unbalanced, refine the matching (caliper, exact matching, or include the covariate as a regressor in the outcome model).

### Caliper matching

Discard pairs with $|e(X^T) - e(X^C)| > c$ for some caliper $c$ (often $0.2 \cdot \mathrm{SD}(\hat e)$). Trades sample size for match quality.

## Inverse Propensity Weighting (IPW)

Weight each unit by $1/\hat e(X_i)$ if treated and $1/(1-\hat e(X_i))$ if control:

$$
\hat\tau_{\mathrm{IPW}} = \frac{1}{n}\sum_i \left[\frac{W_i Y_i}{\hat e(X_i)} - \frac{(1 - W_i) Y_i}{1 - \hat e(X_i)}\right].
$$

Reweights the observational sample so treatment is as if random. Unstable when $\hat e$ approaches 0 or 1: a single observation with $\hat e = 0.01$ and $W = 1$ contributes weight 100, dominating the estimate.

### Trimming and stabilization

- **Trimming.** Drop observations with $\hat e \notin [0.05, 0.95]$. Loses sample but improves stability.
- **Stabilized weights.** Multiply by the marginal treatment probability:

$$
w_i = \frac{P(W = W_i)}{\hat e(X_i)^{W_i} (1 - \hat e(X_i))^{1 - W_i}}.
$$

Reduces variance without bias.

## Instrumental variables (IV)

Find a variable $Z$ that affects treatment but affects the outcome only through treatment. Identification rests on three assumptions:

- **Relevance.** $\mathrm{Cov}(Z, W) \neq 0$. Testable: first-stage regression $W = \pi_0 + \pi_1 Z + \nu$; require first-stage F-statistic above 10 (Stock, Yogo 2005). Weak instruments produce biased and badly behaved estimates.
- **Exclusion restriction.** $Z$ affects $Y$ only through $W$. Untestable; argue from design.
- **Monotonicity.** $Z$ does not push some units toward treatment and others away. Required to interpret the estimate as a Local Average Treatment Effect (LATE) on the compliers.

The two-stage least squares (2SLS) estimator regresses $W$ on $Z$ to get $\hat W$, then regresses $Y$ on $\hat W$.

Classic example: randomly assigned Vietnam draft lottery as IV for military service's effect on earnings (Angrist 1990). Valid instruments are rare in product analytics; the most common ones are encouragement designs (random offer of treatment) and natural experiments (policy rollouts).

## Regression discontinuity (RDD)

When treatment assignment depends sharply on a threshold of a continuous running variable (exam score for scholarship, income cutoff for a benefit), units just above and just below the threshold are nearly identical except for treatment.

### Identification: continuity assumption

The conditional expectation of potential outcomes is continuous at the threshold; any jump in observed $Y$ is the treatment effect.

### Bandwidth selection

Use observations within a window $[c - h, c + h]$ around the threshold $c$. Wider $h$ gives more data but more bias; narrower $h$ gives less bias but more variance. Imbens-Kalyanaraman (2012) provide an optimal bandwidth formula; the `rdrobust` package implements it.

### McCrary density test

If units can manipulate the running variable to land on the favorable side of the threshold (students nudging their exam score up), the design is invalid. McCrary (2008) tests for a discontinuity in the density of the running variable at the threshold; a significant jump invalidates RDD.

## DAGs and identification criteria

A directed acyclic graph (DAG) encodes causal assumptions. Nodes are variables, edges are direct causal effects. Pearl's framework (Pearl 2009) provides graphical criteria for when an effect is identified from observational data.

### Back-door criterion

A set $Z$ satisfies the back-door criterion for $W \to Y$ if:

1. No node in $Z$ is a descendant of $W$.
2. $Z$ blocks every path from $W$ to $Y$ that starts with an arrow into $W$.

If such $Z$ exists, $P(Y \mid \mathrm{do}(W)) = \sum_z P(Y \mid W, Z = z) P(Z = z)$. Adjustment by regression, matching, or weighting on $Z$ identifies the causal effect.

### Front-door criterion

When $W$ and $Y$ share an unobserved confounder but $W$ affects $Y$ only through a measured mediator $M$, the effect is identified by the front-door formula. Used in mediation analysis when the back door is blocked by latent confounding.

DAGitty (web tool) and the `dagitty` R package automate the checking.

## Double Machine Learning (DML)

Chernozhukov et al. (2018). When confounder set $X$ is high-dimensional, parametric adjustment is impossible and naive ML adjustments are biased due to regularization.

DML's idea: orthogonalize.

1. Fit $\hat g(X) = E[Y \mid X]$ with any ML model. Compute $\tilde Y = Y - \hat g(X)$.
2. Fit $\hat m(X) = E[W \mid X]$ with any ML model. Compute $\tilde W = W - \hat m(X)$.
3. Regress $\tilde Y$ on $\tilde W$ via OLS. The coefficient is the ATE.

Cross-fitting (estimate $\hat g$ and $\hat m$ on one fold, residualize on another) avoids overfitting bias. The result is $\sqrt{n}$-consistent and asymptotically normal even when the nuisance estimators converge slower than $n^{-1/2}$.

DML is the modern default for observational ATE estimation with many controls. Implementations: `econml` (Microsoft), `DoubleML` (R/Python).

## Causal forests

Athey & Wager (2019). Tree-based estimator for the CATE $\tau(x)$.

- **Honest splits.** Each tree partitions training data into two halves: one used for choosing splits, the other for estimating leaf values. Avoids the over-fitting bias of single-sample splits.
- **Asymptotic normality.** Pointwise confidence intervals for $\hat\tau(x)$ are available.
- **No parametric form.** Captures arbitrary heterogeneity in treatment effect.

Implementation: `grf` (R) and `econml.dml.CausalForestDML` (Python). Use when heterogeneity is the question (which customers respond positively to the campaign?) and the data have hundreds of thousands of rows or more.

## Synthetic control

Abadie, Diamond, Hainmueller (2010). For comparative case studies with a single treated unit (a city, a state, a country adopting a policy):

1. Construct a synthetic version of the treated unit as a weighted average of donor (untreated) units; weights chosen to match the treated unit's pre-period outcome trajectory.
2. After treatment, the difference between treated and synthetic is the effect.

Inference is by permutation: re-run the procedure pretending each donor was treated; compare the actual treated effect to the placebo distribution.

Strong identifying assumption: pre-period fit being good implies the synthetic would have continued to track the treated unit absent treatment. Weak when pre-period is short or noisy.

## Workflow for an observational study

1. Draw the DAG. Identify the set of confounders and the identification strategy.
2. Choose the method that matches the design (DiD if you have pre-period and a control group, RDD if there is a sharp threshold, IV if there is a credible instrument, DML if you have many controls).
3. State the identifying assumption and how you will defend it.
4. Implement and check diagnostics: parallel trends, overlap, balance, first-stage F.
5. Report effect, confidence interval, and a sensitivity analysis (Rosenbaum bounds, e-values for unobserved confounding).
6. Pre-register an analysis plan when feasible; observational studies are uniquely vulnerable to specification searching.

## Linking to the portfolio

- `causal-uplift-marketing`: applies CATE estimation (T-learner, S-learner, R-learner, causal forest) to a marketing intervention.
- `timeseries-retail-forecast`: counterfactual forecasting can be framed as synthetic control over a single store treated with a promotion.
- `recsys-movielens`: off-policy evaluation of recommender policies uses IPW on logged propensities.

## References

- Imbens, Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*, 2015.
- Pearl, *Causality*, 2nd ed., 2009.
- Chernozhukov et al., "Double/Debiased Machine Learning for Treatment and Structural Parameters", Econometrics Journal, 2018.
- Athey, Wager, "Estimation and Inference of Heterogeneous Treatment Effects using Random Forests", JASA, 2019.
- Abadie, Diamond, Hainmueller, "Synthetic Control Methods for Comparative Case Studies", JASA, 2010.
- Goodman-Bacon, "Difference-in-differences with variation in treatment timing", Journal of Econometrics, 2021.
- Imbens, Kalyanaraman, "Optimal Bandwidth Choice for the Regression Discontinuity Estimator", Review of Economic Studies, 2012.
- McCrary, "Manipulation of the running variable in the regression discontinuity design", Journal of Econometrics, 2008.
- Stock, Yogo, "Testing for Weak Instruments in Linear IV Regression", in *Identification and Inference for Econometric Models*, 2005.
