# Experimentation and A/B Testing

This essay covers randomized controlled trials at depth: setup, power, monitoring, sequential testing, network effects, and variance reduction. Observational causal inference is in the companion essay 08b.

## Why experimentation is the gold standard

An observational dataset can show that X and Y move together. It cannot tell you whether X causes Y. A randomized controlled trial (RCT) breaks the dependence between treatment assignment and confounders by construction: when the treatment indicator $W$ is independent of potential outcomes, $E[Y \mid W=1] - E[Y \mid W=0]$ is an unbiased estimate of the average treatment effect.

In product, marketing, pricing, and UX, the A/B test is the working tool. The actuarial reader will find the same logic in pricing experiments and rate-relativity tests.

## SUTVA: the often-violated foundation

The Stable Unit Treatment Value Assumption (Rubin 1980) has two parts:

1. **No interference.** Unit $i$'s outcome depends on $i$'s treatment, not on others' treatments.
2. **No hidden versions.** All treated units receive the same treatment.

SUTVA fails when users interact (social network), share resources (marketplace, ads, drivers), or exchange information. The fix is cluster randomization: randomize at the level of the interference boundary (ego network, geography, time block) and analyze accordingly with cluster-robust standard errors.

## Randomization unit and SRM

Choose the randomization unit before the test. For most product tests, user is the right unit. For network-effect-heavy products (marketplaces, social), the unit is a cluster (region, time slice, audience pool).

After randomization, check that the realized assignment matches the planned proportions. A 50/50 plan that comes out 52/48 with $n = 100{,}000$ is suspicious. Run a chi-square test of the observed counts against expected:

$$
\chi^2 = \sum_g \frac{(O_g - E_g)^2}{E_g}.
$$

A failure (Sample Ratio Mismatch, SRM) signals a bug in the assignment system, the logging, or the analysis filter. **Never analyze an SRM-failed experiment.** Fix the pipeline and rerun.

## Power analysis (with the per-arm vs total distinction)

Before running an A/B test, compute the required sample size for the effect you want to detect. For a two-sample test with equal-size arms, equal variance $\sigma^2$, and minimum detectable effect $\delta$, the **per-arm** sample size is

$$
n_{\text{per arm}} = \frac{2 (z_{1-\alpha/2} + z_{1-\beta})^2 \sigma^2}{\delta^2}.
$$

The total sample size is $2 n_{\text{per arm}}$. The factor of 2 in the numerator is the SE of a difference of two means with equal arms; do not confuse it with the per-arm vs total convention. Practitioners regularly halve their sample size by misreading this formula.

### Worked example

You want to detect a 1% lift in conversion rate, baseline 10%, with $\alpha = 0.05$ two-sided and power $1 - \beta = 0.80$.

- $\delta = 0.01$.
- $\sigma^2 = p(1-p) = 0.10 \cdot 0.90 = 0.09$.
- $z_{0.975} = 1.96$, $z_{0.80} = 0.84$.
- $n_{\text{per arm}} = 2 \cdot (1.96 + 0.84)^2 \cdot 0.09 / 0.01^2 = 2 \cdot 7.84 \cdot 0.09 / 0.0001 \approx 14{,}112$.

So you need about 14,100 users per arm, 28,200 total. Plug into your daily traffic to get the experiment duration.

## Multiple comparisons in experiment families

If your team runs 20 experiments per quarter at $\alpha = 0.05$, you will see one false positive per quarter on pure chance. Two corrections:

- **Bonferroni** ($p < \alpha / m$). Conservative, controls family-wise error rate. Use when any false positive is unacceptable (regulatory, financial settlement).
- **Benjamini-Hochberg FDR**. Less conservative, controls expected proportion of false positives. Use for screening (campaign optimization, large-scale feature testing).

Within a single experiment, applying multiple-comparison correction across guardrail metrics is debated. The pragmatic answer: declare the primary metric in advance and treat guardrails as one-sided sanity checks rather than confirmatory tests.

## Sequential testing

Continuous monitoring of a fixed-horizon test inflates Type I error: stopping the moment $p < 0.05$ shows up gives a false-positive rate well above 5%. Two principled fixes:

### Sequential probability ratio test (SPRT)

Wald (1945). After each new observation, compute the likelihood ratio under $H_1$ vs $H_0$. Reject $H_0$ if the ratio exceeds an upper boundary; accept if below a lower boundary; otherwise continue. Boundaries are chosen so that overall Type I and Type II rates match targets.

SPRT is optimal for a fixed alternative hypothesis. The catch: in product testing the true effect is unknown; misspecified $H_1$ degrades performance.

### Mixture SPRT (mSPRT)

Robbins (1970), popularized for industry by Optimizely (Johari et al. 2017). Replaces the point alternative with a prior over effect sizes, integrating out the unknown. Anytime-valid: you can peek at the test as often as you like without inflating Type I error. The cost is a slightly larger required sample size compared to a fixed-horizon test.

Production A/B platforms (Optimizely, Statsig, GrowthBook) use mSPRT-style anytime-valid inference because product managers will peek regardless.

## Network and interference effects

When users interact, treating one user can spill over to a control user (a treated user posts content that a control user sees). Effect estimates under user-level randomization are biased. Two fixes:

- **Cluster randomization.** Identify clusters (ego networks, geographies, time windows) and randomize at the cluster level. Analyze with cluster-robust SEs or hierarchical models. Trades sample size for unbiasedness.
- **Switchback experiments.** Alternate the entire system between treatment and control over time blocks. Used by Lyft, DoorDash for marketplace effects.

## CUPED: variance reduction with pre-period data

Deng, Xu, Kohavi, Walker (2013). Use pre-experiment data on the same outcome to remove its predictable component before computing the treatment effect.

Define $Y$ as the outcome and $X$ as a pre-period covariate with $\mathrm{Cov}(X, Y) \neq 0$. The CUPED-adjusted outcome is

$$
\tilde Y = Y - \theta(X - \bar X), \qquad \theta = \frac{\mathrm{Cov}(X, Y)}{\mathrm{Var}(X)}.
$$

By randomization, $X$ is independent of treatment, so the difference of $\tilde Y$ between arms is still an unbiased ATE estimate, but the variance is reduced by a factor $1 - \rho^2$ where $\rho = \mathrm{Corr}(X, Y)$.

### Worked numerical example

Suppose pre-period engagement and post-period engagement have $\rho = 0.7$. Variance reduction is $1 - 0.49 = 0.51$. Required sample size shrinks by the same factor: a test that previously needed 20,000 users per arm now needs about 10,200. CUPED is essentially free; if you have pre-period data, use it.

## Novelty and primacy effects

New features often see a CTR spike (novelty: users click out of curiosity) or a drop (primacy: users have not learned the new UI). The transient effect can mask the long-run treatment effect. Mitigations:

- Run the test for at least 2 to 4 weeks.
- Plot the daily treatment effect over the experiment; check for a stabilizing trend.
- Use a long-run holdout: a small population that never sees the change, monitored for months.

## Cluster-robust standard errors

When SUTVA fails or you cluster-randomize, observations within a cluster are correlated. Naive standard errors are too small; tests are anti-conservative.

The fix: cluster-robust (sandwich) standard errors, where the meat of the sandwich is summed over clusters rather than rows. In Python, statsmodels OLS with `cov_type='cluster', cov_kwds={'groups': cluster_id}`. In R, `lmtest::coeftest(model, vcov = sandwich::vcovCL(model, cluster=...))`.

Rule of thumb: number of clusters should be at least 30 to 40 for cluster-robust inference to be reliable. With fewer clusters, use wild cluster bootstrap.

## Guardrail metrics

Primary metric answers the business question. Guardrails verify the test did not harm something important: latency, error rate, secondary revenue streams, retention. Define both before launch. Stopping rules on guardrails should be pre-registered.

## Experiment review checklist

Before:

- Hypothesis, primary and secondary metrics declared.
- Power analysis: MDE, expected $\sigma$, per-arm $n$, total $n$, duration.
- Guardrails defined with thresholds.
- Randomization unit chosen; clustering considered.
- Analysis plan pre-registered (which test, which decision rule, which subgroups).

During:

- Daily SRM check.
- Daily guardrail monitoring.
- Sequential procedure if peeking; otherwise no early decisions.

After:

- Primary metric tested at the planned stopping point.
- Guardrails checked.
- Pre-registered subgroups investigated; ad-hoc subgroup hunts disclosed as exploratory.
- Effect size, confidence interval, and decision documented.

## Linking to the portfolio

- `causal-uplift-marketing`: experimentation is the rigor backbone for the uplift estimates.
- `mlops-model-service`: production A/B infrastructure is a typical ML Engineer responsibility.
- `streamlit-exec-dashboard`: surfaces guardrail metrics during a live test.

## References

- Kohavi, Tang, Xu, *Trustworthy Online Controlled Experiments*, Cambridge, 2020.
- Deng, Xu, Kohavi, Walker, "Improving the Sensitivity of Online Controlled Experiments by Utilizing Pre-Experiment Data" (CUPED), WSDM 2013.
- Johari, Koomen, Pekelis, Walsh, "Peeking at A/B Tests: Why it matters, and what to do about it" (mSPRT), KDD 2017.
- Wald, "Sequential Tests of Statistical Hypotheses", Annals of Mathematical Statistics, 1945.
- Cameron, Miller, "A Practitioner's Guide to Cluster-Robust Inference", J. Human Resources, 2015.
- Athey, Imbens, "The Econometrics of Randomized Experiments", Handbook of Field Experiments, 2017.
