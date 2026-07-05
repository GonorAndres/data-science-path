# Experimentation and Causal Inference Primer

> **This essay has been split.** As of 2026-04-25, the material lives in two dedicated essays:
>
> - [08a_experimentation.md](08a_experimentation.md): RCTs, power (with per-arm vs total clarification), SRM, sequential testing (SPRT, mSPRT), CUPED, network effects, cluster-robust SEs.
> - [08b_observational_causal.md](08b_observational_causal.md): potential outcomes, DiD with parallel-trends testing, PSM with balance diagnostics, IPW with stabilized weights, IV (relevance, exclusion, monotonicity), RDD (continuity, McCrary density), back-door and front-door criteria, DML, causal forests, synthetic control.
>
> The original combined essay is preserved below for back-compatibility. New work should cite 08a or 08b.

## Correlation vs causation: the practical stake

An observational dataset can show that X and Y move together. It cannot tell you whether X causes Y, Y causes X, or a common cause drives both. For any business decision based on a prediction ("will this campaign improve retention?"), you need a causal claim.

## The potential outcomes framework

For each unit $i$, there are two potential outcomes:

- $Y_i(1)$: outcome if treated.
- $Y_i(0)$: outcome if not treated.

The individual treatment effect is $\tau_i = Y_i(1) - Y_i(0)$. You never see both; only one was realized. This is the fundamental problem of causal inference.

We care about averages:

- Average Treatment Effect (ATE): $E[Y(1) - Y(0)]$.
- ATE on the Treated (ATT): $E[Y(1) - Y(0) | W = 1]$.
- Conditional Average Treatment Effect (CATE): $\tau(x) = E[Y(1) - Y(0) | X = x]$.

## Randomized controlled trial (the gold standard)

Randomly assign units to treatment or control. Randomization ensures:

$$
Y(1), Y(0) \perp W
$$

So $E[Y | W = 1] - E[Y | W = 0]$ estimates the ATE without bias.

The classical A/B test. In product, marketing, pricing, and UX, this is the default.

### Power analysis

Before running an A/B test, compute required sample size for the effect you want to detect.

$$
n \approx \frac{2 (z_{\alpha/2} + z_{\beta})^2 \sigma^2}{\delta^2}
$$

$\delta$ is the minimum detectable effect (MDE). $\sigma$ is the outcome's standard deviation. $z$ are standard normal quantiles for significance $\alpha$ and power $\beta$.

Running an underpowered test wastes resources and risks false negatives. Always do power analysis first.

### Sample ratio mismatch (SRM)

If you expect 50-50 split and observe 52-48, something is wrong with the randomization. Detect with a chi-square test; never analyze an SRM-failed experiment.

### Guardrail metrics

Primary metric answers the question. Guardrails verify the test did not harm something important (latency, error rates, revenue per session if not the primary). Always define both.

### Multiple comparisons

Running many tests increases false-positive rate. Control via Bonferroni, Benjamini-Hochberg, or sequential testing procedures (SPRT, mSPRT).

### Novelty and primacy effects

New features often see an initial CTR spike (novelty) or drop (primacy). Measure for at least 2 to 4 weeks to observe the steady state.

### Network effects

When users interact (social network, marketplace), treatment of one user affects another. Randomizing at user level biases estimates. Cluster-randomize at the communication boundary (ego network, geography, time block).

## Observational causal inference (when you cannot randomize)

Randomization is not always feasible (ethics, legal, cost, infeasibility). Observational methods approximate the ATE under identifying assumptions.

### Difference-in-differences (DiD)

Compare the change over time in a treated group vs an untreated group. Assumes parallel trends: absent treatment, both would have moved together.

$$
\text{ATE} \approx (\bar{Y}_{T,1} - \bar{Y}_{T,0}) - (\bar{Y}_{C,1} - \bar{Y}_{C,0})
$$

Useful for policy evaluations (tax law changed; stores got remodeled; regulation rolled out by state).

### Propensity score matching

Estimate the probability of being treated given observed covariates: $e(X) = P(W=1 | X)$. Match treated units to control units with similar propensity scores. Compare outcomes in matched pairs.

Assumes ignorability: no unobserved confounders. Also assumes overlap: every covariate region has both treated and control units.

### Inverse Propensity Score Weighting (IPSW)

Weight each unit by $1/e(X_i)$ if treated and $1/(1-e(X_i))$ if control. Reweights the observational sample so treatment is as if random.

Unstable if propensities approach zero or one. Trim or clip.

### Instrumental variables (IV)

Find a variable Z that affects treatment but affects the outcome only through treatment. Use Z to identify causal effect.

Classic example: randomly assigned Vietnam draft lottery as IV for military service's effect on earnings.

Hard in practice: valid instruments are rare.

### Regression discontinuity (RDD)

When treatment assignment depends sharply on a threshold of a continuous variable (exam score, income cutoff), compare units just above to just below. Assumes units near the threshold are locally comparable.

### Synthetic control

Build a synthetic version of the treated unit as a weighted average of untreated units. Compare treated to synthetic post-treatment.

Popular for single-treated-unit studies (a city rolled out a program; a state passed a law).

## Uplift modeling

Beyond ATE, estimate individual-level treatment effects. Which customers respond positively to the intervention? Which are indifferent? Which respond negatively?

Methods:

- **Two-model approach (T-learner).** Train one model on treated, one on control. Predict both, difference is CATE.
- **Single-model approach (S-learner).** Include treatment as a feature; predict outcome for treatment=1 and treatment=0, difference is CATE.
- **Transformed-outcome (R-learner).** Transform the outcome using propensity scores to learn CATE directly.
- **Uplift trees.** Split on features that maximize the difference in treatment effects between child nodes.

Evaluation: Qini curve, uplift at top deciles. Calibrated against a randomized holdout.

## Common mistakes in practice

- **Peeking.** Checking the test significance before the planned end date and stopping early inflates false-positive rate.
- **Ignoring interference.** Network effects or marketplace crowding invalidate unit-level randomization.
- **Confusing correlation with causation in observational analysis.** Without an identifying assumption (DiD, RDD, IV), regression coefficients do not have causal meaning.
- **Accepting negative results silently.** Failed experiments are signal; document them.
- **Not controlling for pre-period outcome.** CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction uses the pre-period to tighten estimates. Cheap to add, often halves required sample size.

## Experiment review checklist

Before running:

- Hypothesis stated, with primary and secondary metrics defined.
- Power analysis: MDE, expected $\sigma$, required sample size, duration.
- Guardrails defined.
- Randomization unit chosen; clustering considered.
- Analysis plan pre-registered (what analysis, what decision rule).

During:

- Daily SRM check.
- Guardrail metrics monitored; kill switch ready.

After:

- Primary metric tested.
- Guardrails checked.
- Segments investigated.
- Results documented: effect size, CI, interpretation, action.

## Linking to the portfolio

- `causal-uplift-marketing`: the whole project is this topic.
- `timeseries-retail-forecast`: forecast counterfactuals when promotions are applied; causal framing helps interpret model coefficients.
- `recsys-movielens`: offline evaluation uses historical data; note that counterfactual policy evaluation is its own subfield (inverse propensity weighting on logging propensities).
- `mlops-model-service`: production A/B infrastructure is a typical ML Engineer responsibility; reference this essay.

## References

- Imbens, Rubin, *Causal Inference for Statistics, Social, and Biomedical Sciences*, 2015.
- Pearl, *Causality*, 2009.
- Athey, Imbens, "Machine Learning Methods for Estimating Heterogeneous Causal Effects", 2016.
- Kohavi, Tang, Xu, *Trustworthy Online Controlled Experiments*, 2020.
- Deng, Xu, Kohavi, Walker, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (CUPED), 2013.
- Chernozhukov et al., "Double/Debiased Machine Learning for Treatment and Structural Parameters", 2018.

## Exercises

1. An A/B test runs for 4 weeks. The treatment group shows a lift of 2 percent with $p = 0.04$. Your boss asks to call it significant. What questions do you ask before agreeing?
2. Compute the required sample size per arm for a test with $\alpha = 0.05$, power $= 0.80$, baseline conversion $= 0.10$, MDE $= 0.01$, two-sided.
3. State the parallel-trends assumption for DiD and describe one empirical test for it.
4. You estimate a propensity-score model with logistic regression. After matching, the SMD for one covariate is still $0.15$. What do you do?
5. Explain the exclusion restriction for IV and give one example where it plausibly holds and one where it is suspect.
