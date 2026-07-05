# A/B Testing Math

## Setup

Two groups: treatment and control. Each has its own observed conversion rate $p_T$ and $p_C$. We ask: is $p_T - p_C > 0$?

## Test statistic

Two-proportion z-test:

$$
Z = \frac{\hat{p}_T - \hat{p}_C}{\sqrt{\hat{p}(1 - \hat{p})(\frac{1}{n_T} + \frac{1}{n_C})}}
$$

where $\hat{p}$ is the pooled proportion.

Under $H_0: p_T = p_C$, Z is approximately standard normal for large n.

## Power analysis

To detect a minimum effect size $\delta$ with significance $\alpha$ and power $1 - \beta$, required sample size per group:

$$
n \approx \left(\frac{z_{\alpha/2} + z_\beta}{\delta}\right)^2 \cdot 2 \hat{p}(1 - \hat{p})
$$

Rough rule of thumb at 5 percent significance and 80 percent power:

$$
n \approx \frac{16 \hat{p}(1 - \hat{p})}{\delta^2}
$$

If baseline conversion is 5 percent and you want to detect a 0.5 pp lift, you need about 12,000 users per group.

## Multiple comparisons

Running many tests inflates the false positive rate. Corrections:

- **Bonferroni.** Multiply p-values by number of tests. Conservative.
- **Benjamini-Hochberg.** Controls false discovery rate. More powerful.
- **Sequential testing.** Tests designed for peeking: mSPRT, always-valid p-values.

## Sample Ratio Mismatch (SRM)

If treatment and control should have equal volume but do not, something is wrong with the randomization. Chi-square test:

$$
\chi^2 = \sum \frac{(O - E)^2}{E}
$$

A failing SRM invalidates the test.

## CUPED variance reduction

Using pre-period outcomes to reduce variance:

$$
Y' = Y - \theta (X - \bar{X})
$$

where $X$ is a pre-period covariate correlated with $Y$. Can cut required sample size in half.

## Connecting to uplift

A/B test measures ATE. Uplift modeling estimates CATE, so you can target users with higher expected uplift. The A/B test validates the uplift-targeted campaign against untargeted control.

## Cross-links

- `03_uplift_modeling.md`
- `04_sensitivity_and_assumptions.md`
- `docs/theory/08_experimentation_causal_primer.md`
