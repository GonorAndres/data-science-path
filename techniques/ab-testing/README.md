# A/B Testing

## What it is

A/B testing (also called split testing or randomized controlled experimentation) is the gold standard method for determining whether a change -- to a website, a marketing campaign, a pricing strategy, or a product feature -- actually *causes* an improvement in a metric you care about. You randomly split your population into two groups: the **control** group (which gets the current experience, "A") and the **treatment** group (which gets the new experience, "B"). By comparing outcomes between the groups, you can attribute any observed difference to the treatment rather than to confounding factors. A/B testing turns gut feelings into evidence.

## Why it matters

- **Causal evidence**: Observational data can only show correlations. A/B testing, through randomization, provides causal evidence. If the treatment group performs better, the treatment *caused* it (assuming the test was properly designed).
- **De-risks decisions**: Launching a major change without testing is a gamble. A/B testing lets you try the change on a small fraction of users first and measure the impact before committing.
- **Insurance applications**: Testing different pricing strategies, renewal communication templates, claims process changes, or agent incentive structures. In insurance, the cost of a bad decision (mispricing, increased lapse) can be enormous, making testing especially valuable.
- **Quantifies effect size**: Beyond "did it work?", A/B testing tells you *how much* it worked. A 2% lift in conversion might not be worth the engineering effort; a 15% lift changes the business.
- **Builds a testing culture**: Organizations that A/B test regularly make better decisions because they replace "the highest-paid person's opinion" (HiPPO) with data.
- **Statistical rigor**: A/B testing forces you to formalize your hypotheses, choose metrics, calculate required sample sizes, and interpret results correctly -- all core skills for a data scientist.

## When to use it

- You want to test a change to a website, app, email, or marketing campaign and measure its impact on a specific metric.
- You can randomly assign units (users, policyholders, claims) to treatment and control groups.
- You have enough traffic or sample size to detect the effect size you care about.
- The metric of interest can be measured within a reasonable time frame.
- You want to settle a debate between two (or more) design options with data rather than opinions.
- You are launching a new pricing strategy and want to test it on a subset before full rollout.
- You want to optimize a process (claims processing, underwriting workflow) and measure the impact of changes.

## How it works (intuitive)

### The medical trial analogy

A/B testing is borrowed from clinical trials. When testing a new drug, researchers randomly assign patients to receive either the drug or a placebo. If the drug group improves more than the placebo group, and the difference is unlikely to be due to chance, the drug works.

In the digital world, the "drug" is your change (new button color, new pricing, new email subject line), the "patients" are your users or customers, and the "improvement" is your business metric (conversion rate, click-through rate, retention).

### Step-by-step process

**Step 1: Formulate a hypothesis**

Be specific:
- **Null hypothesis ($H_0$)**: There is no difference between A and B. The treatment has no effect.
- **Alternative hypothesis ($H_1$)**: There is a difference (two-sided) or the treatment improves the metric (one-sided).

Example: "Changing the renewal email subject line from 'Your policy is expiring' to 'Protect what matters: renew today' will increase email open rates."

**Step 2: Choose your metric**

Define a **primary metric** (the one you are trying to improve) and **guardrail metrics** (metrics that should not get worse).

| Metric type | Example |
|-------------|---------|
| Primary | Email open rate |
| Guardrail | Unsubscribe rate (should not increase) |
| Guardrail | Conversion rate downstream (should not decrease) |

**Step 3: Calculate sample size**

Before running the test, determine how many observations you need. This depends on:

- **Baseline rate** ($p_0$): The current value of your metric (e.g., 20% open rate).
- **Minimum detectable effect (MDE)**: The smallest improvement worth detecting (e.g., 2 percentage points, from 20% to 22%).
- **Significance level** ($\alpha$): Probability of a false positive (concluding the treatment works when it does not). Standard: 0.05.
- **Power** ($1 - \beta$): Probability of detecting a true effect. Standard: 0.80.

The formula for sample size per group (for proportions):

$$n = \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot [p_0(1-p_0) + p_1(1-p_1)]}{(p_1 - p_0)^2}$$

where $p_1 = p_0 + \text{MDE}$, $Z_{\alpha/2} \approx 1.96$ for $\alpha = 0.05$ (two-sided), and $Z_{\beta} \approx 0.84$ for power = 0.80.

Alternatively, for continuous metrics (e.g., revenue per user):

$$n = \frac{(Z_{\alpha/2} + Z_{\beta})^2 \cdot 2\sigma^2}{\delta^2}$$

where $\sigma^2$ is the variance of the metric and $\delta$ is the minimum detectable difference.

**Step 4: Randomize and run**

Randomly assign units to A (control) or B (treatment). Ensure:
- **Random assignment**: No systematic differences between groups before the treatment.
- **No interference**: One unit's assignment should not affect another unit's outcome (the Stable Unit Treatment Value Assumption, or SUTVA).
- **Consistent experience**: Each unit stays in their assigned group for the entire test.

**Step 5: Analyze the results**

After collecting enough data, perform the appropriate statistical test:

| Metric type | Test | When to use |
|-------------|------|-------------|
| Proportion (e.g., conversion rate) | Z-test or chi-squared test | Comparing two proportions |
| Continuous mean (e.g., revenue) | Two-sample t-test or Welch's t-test | Comparing two means |
| Count data (e.g., claims) | Poisson test or negative binomial | Comparing rates |
| Time-to-event (e.g., days to renew) | Log-rank test | Comparing survival curves |

**Step 6: Interpret and decide**

- If the p-value < $\alpha$, reject $H_0$: the difference is statistically significant.
- Check the confidence interval for the effect size. Is it practically meaningful?
- Check guardrail metrics. Even if the primary metric improved, a test that increases churn is not worth it.
- Make the decision: roll out, iterate, or abandon.

### The courtroom analogy

A/B testing follows courtroom logic:
- **Null hypothesis** = "Innocent until proven guilty." We assume the treatment has no effect until the evidence says otherwise.
- **p-value** = The probability of seeing evidence this extreme if the defendant were truly innocent. A very low p-value means the evidence is hard to explain under innocence.
- **Significance level ($\alpha$)** = The threshold for conviction. At $\alpha = 0.05$, we convict (reject $H_0$) if there is less than a 5% chance the evidence arose by chance.
- **Type I error** = Convicting an innocent person (false positive).
- **Type II error** = Acquitting a guilty person (false negative, missing a real effect).

## Key concepts

### Hypothesis testing framework

| Term | Symbol | Meaning |
|------|--------|---------|
| Significance level | $\alpha$ | Probability of Type I error (false positive). Usually 0.05. |
| Power | $1 - \beta$ | Probability of detecting a true effect. Usually 0.80. |
| p-value | $p$ | Probability of observing data this extreme under $H_0$. |
| Effect size | $\delta$ | The magnitude of the difference between A and B. |
| Confidence interval | CI | Range of plausible values for the true effect. |

### Two-sample t-test

For comparing means:

$$t = \frac{\bar{X}_B - \bar{X}_A}{\sqrt{\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}}}$$

where $\bar{X}$ is the sample mean, $s^2$ is the sample variance, and $n$ is the sample size.

Degrees of freedom (Welch-Satterthwaite approximation):

$$df = \frac{\left(\frac{s_A^2}{n_A} + \frac{s_B^2}{n_B}\right)^2}{\frac{(s_A^2/n_A)^2}{n_A - 1} + \frac{(s_B^2/n_B)^2}{n_B - 1}}$$

### Chi-squared test for proportions

For comparing conversion rates:

$$\chi^2 = \sum_{i} \frac{(O_i - E_i)^2}{E_i}$$

where $O_i$ are observed counts and $E_i$ are expected counts under $H_0$ (no difference).

### Power analysis

Power is the probability of correctly rejecting $H_0$ when $H_1$ is true:

$$\text{Power} = P(\text{reject } H_0 | H_1 \text{ is true}) = 1 - \beta$$

The four quantities are interrelated: $n$, $\alpha$, $\beta$ (power), and $\delta$ (effect size). Fix any three and you can solve for the fourth. In practice, you fix $\alpha = 0.05$, power $= 0.80$, and the MDE, then solve for $n$.

### Multiple testing correction

Running multiple tests simultaneously inflates the false positive rate. If you test 20 variations at $\alpha = 0.05$, you expect one false positive by chance.

**Bonferroni correction**: Divide $\alpha$ by the number of tests.

$$\alpha_{\text{adjusted}} = \frac{\alpha}{m}$$

where $m$ is the number of comparisons. This is conservative -- it reduces Type I errors but increases Type II errors (misses real effects).

**Benjamini-Hochberg (FDR)**: Instead of controlling the probability of *any* false positive, control the *proportion* of discoveries that are false (the False Discovery Rate). Less conservative than Bonferroni.

1. Sort p-values: $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$
2. Find the largest $k$ such that $p_{(k)} \leq \frac{k}{m} \cdot \alpha$
3. Reject all hypotheses $1, 2, \ldots, k$

**Guideline**: Use Bonferroni for a small number of critical tests. Use FDR for exploratory analyses with many comparisons.

### The peeking problem

Checking results before the planned sample size is reached inflates the false positive rate. Every time you "peek" and consider stopping if the result looks significant, you are effectively running a test at a lower significance level than you think.

**Example**: If your planned test requires 10,000 observations but you check after every 1,000, your effective $\alpha$ might be 0.15 instead of 0.05.

**Solutions**:
- **Pre-commit**: Decide the sample size in advance and do not look until it is reached.
- **Sequential testing**: Use methods designed for continuous monitoring (see below).

### Sequential testing

Sequential testing methods (also called "always-valid" tests) allow you to monitor results continuously without inflating the false positive rate.

**Group sequential designs**: Pre-specify interim analysis points (e.g., at 25%, 50%, 75% of planned sample size) with adjusted significance boundaries (e.g., O'Brien-Fleming or Pocock boundaries).

**Always-valid confidence sequences**: Provide confidence intervals that are valid at any stopping time. These are becoming more popular in tech companies.

Key methods:
- **O'Brien-Fleming boundaries**: Very conservative early, close to standard $\alpha$ at the final analysis. Practical for planned interim looks.
- **Spending functions** (Lan-DeMets): More flexible allocation of the "alpha budget" across interim analyses.
- **Mixture sequential probability ratio test (mSPRT)**: Used by some tech companies for always-on monitoring.

### Practical effect size (Cohen's conventions)

For continuous metrics, Cohen's $d$ standardizes the effect size:

$$d = \frac{\bar{X}_B - \bar{X}_A}{s_{\text{pooled}}}$$

| $d$ | Interpretation |
|-----|---------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

For proportions, use the absolute difference or the relative lift:
- Absolute: $p_B - p_A$ (e.g., 22% - 20% = 2 percentage points)
- Relative: $\frac{p_B - p_A}{p_A}$ (e.g., $\frac{0.02}{0.20} = 10\%$ relative lift)

## Python implementation

```python
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import TTestIndPower, NormalIndPower
import statsmodels.stats.api as sms

# --- Sample size calculation for proportions ---
def sample_size_proportions(p0, mde, alpha=0.05, power=0.80):
    """
    Calculate required sample size per group for a two-proportion z-test.

    Parameters
    ----------
    p0 : float
        Baseline conversion rate (control group).
    mde : float
        Minimum detectable effect (absolute, e.g., 0.02 for 2pp).
    alpha : float
        Significance level.
    power : float
        Desired power.

    Returns
    -------
    int
        Required sample size per group.
    """
    p1 = p0 + mde
    effect_size = sms.proportion_effectsize(p0, p1)
    analysis = NormalIndPower()
    n = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='two-sided'
    )
    return int(np.ceil(n))

# --- Sample size calculation for means ---
def sample_size_means(delta, sigma, alpha=0.05, power=0.80):
    """
    Required sample size per group for a two-sample t-test.

    Parameters
    ----------
    delta : float
        Minimum detectable difference in means.
    sigma : float
        Estimated standard deviation.
    """
    effect_size = delta / sigma
    analysis = TTestIndPower()
    n = analysis.solve_power(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        alternative='two-sided'
    )
    return int(np.ceil(n))

# --- Two-proportion z-test ---
def test_proportions(successes_a, n_a, successes_b, n_b, alpha=0.05):
    """
    Two-proportion z-test for comparing conversion rates.
    """
    count = np.array([successes_b, successes_a])
    nobs = np.array([n_b, n_a])
    z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')

    p_a = successes_a / n_a
    p_b = successes_b / n_b
    lift = (p_b - p_a) / p_a

    print(f"Control rate:   {p_a:.4f}")
    print(f"Treatment rate: {p_b:.4f}")
    print(f"Absolute diff:  {p_b - p_a:.4f}")
    print(f"Relative lift:  {lift:.2%}")
    print(f"Z-statistic:    {z_stat:.4f}")
    print(f"P-value:        {p_value:.4f}")
    print(f"Significant:    {'Yes' if p_value < alpha else 'No'} (alpha={alpha})")

    return z_stat, p_value

# --- Two-sample t-test ---
def test_means(group_a, group_b, alpha=0.05):
    """
    Welch's two-sample t-test for comparing means.
    """
    t_stat, p_value = stats.ttest_ind(group_b, group_a, equal_var=False)

    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)

    print(f"Control mean:   {mean_a:.4f}")
    print(f"Treatment mean: {mean_b:.4f}")
    print(f"Difference:     {mean_b - mean_a:.4f}")
    print(f"T-statistic:    {t_stat:.4f}")
    print(f"P-value:        {p_value:.4f}")
    print(f"Significant:    {'Yes' if p_value < alpha else 'No'} (alpha={alpha})")

    return t_stat, p_value

# --- Multiple testing correction ---
from statsmodels.stats.multitest import multipletests

p_values = [0.01, 0.04, 0.03, 0.15, 0.50]

# Bonferroni
reject_bonf, pvals_bonf, _, _ = multipletests(p_values, method='bonferroni')

# Benjamini-Hochberg (FDR)
reject_bh, pvals_bh, _, _ = multipletests(p_values, method='fdr_bh')

# --- Chi-squared test ---
def test_chi_squared(contingency_table):
    """
    Chi-squared test on a 2x2 contingency table.
    contingency_table: [[control_success, control_failure],
                        [treatment_success, treatment_failure]]
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"Chi-squared: {chi2:.4f}")
    print(f"P-value:     {p_value:.4f}")
    print(f"DOF:         {dof}")
    return chi2, p_value
```

Key libraries:
- **scipy.stats**: t-tests, chi-squared tests, and other statistical tests.
- **statsmodels**: Proportion tests, power analysis, multiple testing correction.
- **numpy** for numerical computations.
- **plotly** or **seaborn** for visualizing results and confidence intervals.

## R implementation (if applicable)

```r
# Power analysis
library(pwr)
pwr.2p.test(h = ES.h(p1 = 0.22, p2 = 0.20),
            sig.level = 0.05,
            power = 0.80,
            alternative = "two.sided")

# Two-proportion test
prop.test(c(successes_b, successes_a), c(n_b, n_a))

# Two-sample t-test
t.test(group_b, group_a, var.equal = FALSE)

# Chi-squared test
chisq.test(matrix(c(s_a, f_a, s_b, f_b), nrow = 2))

# Multiple testing correction
p.adjust(p_values, method = "BH")   # Benjamini-Hochberg
p.adjust(p_values, method = "bonferroni")
```

**Python vs. R tradeoff**: Both handle A/B testing equally well. R's `pwr` package is excellent for power analysis. Python's `statsmodels` integrates better with data science workflows. For sequential testing, R has more mature packages (`gsDesign`, `rpact`).

## Gotchas & lessons learned

### 1. The peeking problem is real and widespread
Checking results daily "just to see how it's going" inflates your false positive rate. If you peeked 10 times during a test at $\alpha = 0.05$, your effective $\alpha$ could be 0.15-0.25. Either commit to a fixed sample size or use sequential testing methods.

### 2. Sample size is usually larger than you think
A 1-percentage-point lift from 5% to 6% conversion requires roughly 14,000 observations per group at 80% power and 5% significance. For small effect sizes, you need massive samples. Do the power analysis *before* running the test.

### 3. Novelty and primacy effects
Users may react differently to a change simply because it is new (novelty effect) or because they are used to the old experience (primacy effect). Consider running the test long enough for these effects to wear off -- at least 2-4 weeks for most products.

### 4. Metric sensitivity
Choose a metric that is sensitive enough to detect the change. Revenue per user has high variance and requires huge samples. Conversion rate (binary) is often more tractable. But do not optimize for a proxy metric that does not actually drive business value.

### 5. Survivorship bias in analysis
If users can drop out of the test (e.g., they unsubscribe), analyzing only survivors introduces bias. Analyze on an intent-to-treat basis: include everyone who was randomized, regardless of whether they completed the experience.

### 6. Practical significance vs. statistical significance
A test can be statistically significant (p < 0.05) but practically meaningless (0.01% lift). Always report confidence intervals on the effect size and evaluate whether the effect is large enough to justify the cost of implementation.

### 7. Simpson's paradox in segment analysis
A test might show no overall effect, but a strong positive effect in one segment and a negative effect in another. Post-hoc segment analysis is useful for generating hypotheses but should not be used to "find" significant results. Pre-register any planned subgroup analyses.

### 8. Network effects and interference
In social products or when users interact (insurance agents serving multiple customers), one user's treatment can affect another user's outcome, violating the independence assumption. This requires cluster-randomized designs (randomize at the agent level, not the customer level).

### 9. Multiple metrics = multiple testing
If you test one change against 10 metrics, on average 0.5 metrics will be "significant" by chance at $\alpha = 0.05$. Define a primary metric and apply multiple testing corrections to secondary metrics.

### 10. The cost of not testing
It is tempting to skip A/B testing for "obvious" improvements. But research consistently shows that most A/B tests show no significant effect or even a *negative* effect. Without testing, you would have shipped those negative changes and never known.

## Resources

- **Kohavi, Tang & Xu (2020)**: "Trustworthy Online Controlled Experiments" -- the definitive book on A/B testing by the Microsoft experimentation team.
- **Evan Miller's tools**: [evanmiller.org/ab-testing/](https://www.evanmiller.org/ab-testing/) -- sample size calculator and sequential testing tools.
- **Statsmodels documentation**: [statsmodels.org](https://www.statsmodels.org/) -- power analysis and statistical tests.
- **scipy.stats**: [scipy docs](https://docs.scipy.org/doc/scipy/reference/stats.html) -- statistical test implementations.
- **R pwr package**: [CRAN - pwr](https://cran.r-project.org/package=pwr) -- power analysis.
- **Google's course**: "A/B Testing" on Udacity (by Google) -- free and practical.
- **Optimizely knowledge base**: Excellent articles on experimentation pitfalls and best practices.
- **Sequential testing**: Howard et al. (2021), "Time-uniform, nonparametric, nonasymptotic confidence sequences" for the theory behind always-valid inference.
- **DataCamp**: "Customer Analytics and A/B Testing in Python."
