# Bayesian Credibility

## What it is

Bayesian credibility (also called credibility theory) is the actuarial method for blending an individual risk's own experience with the broader population's experience to produce a better estimate. The central question is: "How much should we trust this particular policyholder's (or group's) claims history versus the overall portfolio average?" The answer is a credibility-weighted estimate, where the weight $Z$ (the credibility factor) ranges from 0 to 1. When $Z = 1$, you trust the individual experience completely. When $Z = 0$, you ignore the individual data and use the population mean. In between, you blend the two. This framework is one of the foundational tools in actuarial science and appears prominently on credentialing exams (SOA and CAS).

## Why it matters

- **Pricing fairness**: An insurer cannot simply charge every policyholder the same rate, nor can it rely solely on one policyholder's tiny claims history. Credibility theory provides the principled middle ground.
- **Small data, big decisions**: Many actuarial segments (a small employer group, a niche product line, a new territory) have sparse experience. Credibility tells you exactly how much to lean on that sparse data.
- **Connection to Bayesian statistics**: Buhlmann credibility is mathematically equivalent to Bayesian posterior estimation under certain assumptions. Understanding credibility deepens your Bayesian intuition and vice versa.
- **Exam relevance**: Credibility is a core topic on SOA Exam STAM (Short-Term Actuarial Mathematics) and CAS Exam MAS-I. The formulas and concepts here are directly testable.
- **Practical use in reserving, pricing, and experience rating**: Credibility-weighted estimates appear in loss reserving (blending a company's development pattern with industry patterns), experience rating (adjusting premiums for large accounts), and assumption setting (mortality, lapse, morbidity studies).

## When to use it

- You are setting rates for a group that has some claims experience but not enough to be fully credible on its own.
- You are performing an experience study (mortality, lapse, morbidity) and need to blend your company's experience with industry tables.
- You are adjusting a large commercial account's premium based on their claims history (experience rating / retrospective rating).
- You need a credibility-weighted IBNR estimate that blends your company's development factors with industry benchmarks.
- You want a theoretically grounded way to shrink individual estimates toward a population mean (this is the same concept as "shrinkage" in statistics and "regularization" in machine learning).

## How it works (intuitive)

### The big picture: the "new teacher" analogy

Imagine a new teacher joins a school. Before seeing any student work, the principal has a general expectation of student performance based on school-wide averages (the **population mean**). After one semester, the new teacher's class has an average test score. Should the principal evaluate the teacher purely on this one-semester result?

No. One semester is not enough data. The class might have been unusually strong or weak by chance. The principal should blend the class result with the school-wide average, weighting the class result more heavily if the class was large (more data) and the school-wide results were highly variable across teachers (more reason to think individual differences are real).

This is exactly what credibility theory does:

$$\hat{\mu}_i = Z \cdot \bar{X}_i + (1 - Z) \cdot \mu$$

where:
- $\hat{\mu}_i$ = the credibility-weighted estimate for risk $i$
- $\bar{X}_i$ = the observed experience of risk $i$ (the teacher's class average)
- $\mu$ = the overall population mean (the school-wide average)
- $Z$ = the credibility factor (how much weight to give the individual experience)

### What determines Z?

The credibility factor $Z$ depends on the balance between two sources of variation:

1. **Process variance** ($v$): The randomness *within* a single risk. Even a perfectly average driver will have some years with claims and some without. This is the inherent noise in any individual's experience.

2. **Variance of hypothetical means** ($a$): The genuine differences *between* risks. Some drivers truly are riskier than others. This captures the real spread in underlying risk levels across the population.

The key insight:

$$Z = \frac{n}{n + k} \quad \text{where } k = \frac{v}{a}$$

- If risks are very different from each other ($a$ is large relative to $v$), then $k$ is small, and $Z$ is close to 1 even with moderate data. Interpretation: individual experience is informative because there is real heterogeneity to detect.
- If all risks are basically the same ($a$ is small), then $k$ is large, and you need a huge amount of data before $Z$ approaches 1. Interpretation: individual experience is mostly noise.

### Connection to Bayesian updating

Under the Normal-Normal conjugate model:
- Prior: $\theta_i \sim N(\mu, a)$ -- each risk's true mean comes from a population distribution.
- Likelihood: $\bar{X}_i \mid \theta_i \sim N(\theta_i, v/n_i)$ -- the observed mean has sampling variability.
- Posterior mean: $E[\theta_i \mid \bar{X}_i] = Z_i \cdot \bar{X}_i + (1 - Z_i) \cdot \mu$ where $Z_i = \frac{n_i}{n_i + v/a}$.

This is *identical* to the Buhlmann credibility formula. Credibility theory is Bayesian estimation in disguise.

## Key concepts

### Buhlmann credibility

The simplest and most important credibility model. Assumes:
- Each risk $i$ has a true underlying parameter $\theta_i$.
- Given $\theta_i$, the observations $X_{i1}, X_{i2}, \ldots, X_{in_i}$ are independent with the same mean and variance.
- The $\theta_i$ values themselves are drawn from some population distribution.

**Core formulas:**

| Quantity | Formula | Meaning |
|----------|---------|---------|
| Expected value of process variance | $v = E[\text{Var}(X \mid \theta)]$ | Average within-risk variance |
| Variance of hypothetical means | $a = \text{Var}[E(X \mid \theta)]$ | Variance of true underlying means across risks |
| Buhlmann credibility parameter | $k = v / a$ | How many observations you need for "full" credibility |
| Credibility factor | $Z = n / (n + k)$ | Weight on individual experience |
| Credibility-weighted estimate | $\hat{\mu}_i = Z \cdot \bar{X}_i + (1 - Z) \cdot \hat{\mu}$ | Blended estimate |

**Interpreting $k$:**
- $k$ is the number of observations at which $Z = 0.50$ (equal weight to individual and population).
- If $k = 10$, then with 10 observations you are 50% credible, with 100 observations you are about 91% credible ($Z = 100/110$), etc.

### Buhlmann-Straub credibility

An extension of Buhlmann that allows each observation to carry a different **weight** (exposure). This is critical in insurance because a policy covering a large fleet of vehicles represents more exposure than a personal auto policy.

**Modifications:**

- Each observation $X_{ij}$ has an associated weight $w_{ij}$ (e.g., earned premium, number of car-years, number of member-months).
- The conditional variance is inversely proportional to the weight: $\text{Var}(X_{ij} \mid \theta_i) = v / w_{ij}$.
- Credibility factor: $Z_i = \frac{w_{i\cdot}}{w_{i\cdot} + k}$ where $w_{i\cdot} = \sum_j w_{ij}$ is the total weight for risk $i$.

Everything else proceeds identically to Buhlmann, but now the "number of observations" is replaced by the total weight.

### Estimating $v$ and $a$ from data

In practice, $v$ and $a$ are unknown and must be estimated. The standard non-parametric estimators are:

**Within-risk variance ($\hat{v}$):**

$$\hat{v} = \frac{1}{r} \sum_{i=1}^{r} \frac{1}{n_i - 1} \sum_{j=1}^{n_i} (X_{ij} - \bar{X}_{i\cdot})^2$$

This is the average of the within-risk sample variances (pooled over all risks).

**Between-risk variance ($\hat{a}$):**

$$\hat{a} = \frac{\sum_{i=1}^{r} n_i (\bar{X}_{i\cdot} - \bar{X}_{\cdot\cdot})^2 - (r-1)\hat{v}}{\sum_{i=1}^{r} n_i - \frac{\sum_{i=1}^{r} n_i^2}{\sum_{i=1}^{r} n_i}}$$

If $\hat{a}$ comes out negative (which can happen when the between-risk variability is very small relative to within-risk noise), it is typically truncated to 0, which gives $Z = 0$ (no credibility for individual experience).

### Partial credibility (classical / limited fluctuation)

An older, simpler approach (still tested on exams). Instead of deriving $Z$ from variance components, you set a "full credibility standard" -- the number of expected claims $n_0$ needed for the observed experience to be fully credible (within a specified percentage of the true value with a given confidence level).

$$n_0 = \left(\frac{z_{\alpha/2}}{r}\right)^2 \cdot \left(1 + \frac{\text{CV}^2}{\text{frequency}}\right)$$

where $r$ is the desired accuracy (e.g., within 5%) and $z_{\alpha/2}$ is the normal critical value (e.g., 1.96 for 95% confidence).

For frequency only (Poisson claims), the standard simplifies to:

$$n_0 = \left(\frac{z_{\alpha/2}}{r}\right)^2$$

Example: For 95% confidence and 5% accuracy, $n_0 = (1.96/0.05)^2 = 1{,}537$ expected claims.

The partial credibility factor is then:

$$Z = \min\left(1, \sqrt{\frac{n}{n_0}}\right)$$

The square root formula is a convention (not derived from optimality) that ensures $Z$ approaches 1 smoothly.

### The credibility spectrum

| $Z$ value | Interpretation | Action |
|-----------|---------------|--------|
| $Z = 0$ | No credibility; individual experience is pure noise | Use population mean only |
| $0 < Z < 1$ | Partial credibility | Blend individual and population |
| $Z = 1$ | Full credibility; enough data to trust individual experience entirely | Use individual mean only |

In practice, $Z$ rarely reaches 1 exactly. Most actuarial applications live in the partial credibility zone.

## Python implementation

There is no single dominant Python library for credibility theory, but the calculations are straightforward with `numpy` and `pandas`.

```python
import numpy as np
import pandas as pd

def buhlmann_credibility(data, risk_col, obs_col):
    """
    Compute Buhlmann credibility estimates.

    Parameters:
    -----------
    data : DataFrame with columns [risk_col, obs_col]
    risk_col : column identifying the risk group
    obs_col : column with the observed values

    Returns:
    --------
    DataFrame with credibility-weighted estimates for each risk.
    """
    grouped = data.groupby(risk_col)[obs_col]

    # Number of observations per risk
    n_i = grouped.count()
    # Mean per risk
    xbar_i = grouped.mean()
    # Within-risk variance per risk
    s2_i = grouped.var(ddof=1)

    r = len(n_i)  # number of risks

    # Grand mean (overall average)
    mu_hat = data[obs_col].mean()

    # v-hat: average within-risk variance
    v_hat = s2_i.mean()

    # a-hat: between-risk variance (Buhlmann estimator)
    n_total = n_i.sum()
    numerator = (n_i * (xbar_i - mu_hat)**2).sum() - (r - 1) * v_hat
    denominator = n_total - (n_i**2).sum() / n_total
    a_hat = max(0, numerator / denominator)

    # k and Z
    k = v_hat / a_hat if a_hat > 0 else float('inf')
    Z_i = n_i / (n_i + k)

    # Credibility-weighted estimates
    cred_est = Z_i * xbar_i + (1 - Z_i) * mu_hat

    result = pd.DataFrame({
        'n': n_i,
        'observed_mean': xbar_i,
        'Z': Z_i,
        'credibility_estimate': cred_est,
    })

    return result, {'v_hat': v_hat, 'a_hat': a_hat, 'k': k, 'grand_mean': mu_hat}
```

```python
# Buhlmann-Straub extension (weighted)
def buhlmann_straub(data, risk_col, obs_col, weight_col):
    """
    Buhlmann-Straub credibility with exposure weights.
    """
    grouped = data.groupby(risk_col)

    # Weighted mean per risk
    def weighted_mean(g):
        return np.average(g[obs_col], weights=g[weight_col])

    xbar_i = grouped.apply(weighted_mean)
    w_i = grouped[weight_col].sum()

    # Grand weighted mean
    mu_hat = np.average(data[obs_col], weights=data[weight_col])

    # Estimate v and a (Buhlmann-Straub estimators)
    # ... (analogous to Buhlmann but weight-adjusted)

    # Then Z_i = w_i / (w_i + k) and blend as before
    pass  # Full implementation follows the same structure
```

## R implementation (if applicable)

The `actuar` package in R provides built-in credibility functions:

```r
library(actuar)

# Buhlmann credibility
fit <- cm(~ risk_group, data = claims_data, ratios = loss_ratio, weights = exposure)
summary(fit)

# Extract credibility factors
fit$unbiased  # a-hat, v-hat
fit$nodes$risk_group$credibility  # Z for each group
```

The `cm()` function handles Buhlmann, Buhlmann-Straub, and hierarchical credibility models (e.g., risk groups nested within territories).

## Gotchas & lessons learned

1. **Negative $\hat{a}$ estimates**: The between-risk variance estimator can go negative when within-risk noise overwhelms between-risk differences. Truncate to 0 and interpret this as "no evidence of real differences between risks." Do not use a negative $a$ in the formula.

2. **Credibility is not confidence**: $Z = 0.60$ does not mean "60% confidence." It means "give 60% weight to the individual experience and 40% weight to the population." These are conceptually different.

3. **Choosing the complement of credibility**: The formula $\hat{\mu}_i = Z \cdot \bar{X}_i + (1 - Z) \cdot M$ requires choosing what $M$ is. Common choices:
   - Overall portfolio mean (Buhlmann).
   - Industry benchmark.
   - A prior derived from a regression model (this links credibility to GLMs).
   The choice of $M$ matters more than the choice of $Z$ when $Z$ is small.

4. **Limited fluctuation vs. Buhlmann**: Exam questions will specify which framework to use. Limited fluctuation (classical) credibility uses the square-root formula and a fixed standard; Buhlmann uses variance components. They give different $Z$ values for the same data.

5. **Exposure matters**: In Buhlmann-Straub, two observations with very different exposures should not be treated equally. Always use the weighted version when exposures vary across observations.

6. **Hierarchical credibility**: When data has a natural hierarchy (policyholder within territory within state), you can apply credibility at each level. The `actuar` package in R handles multi-level hierarchical credibility models.

7. **Connection to random effects models**: Buhlmann credibility is mathematically equivalent to a random-intercept model. The credibility-weighted estimate equals the Best Linear Unbiased Predictor (BLUP) from a linear mixed model. If you understand one, you understand the other.

8. **Full credibility thresholds are arbitrary**: The 1,082 expected claims standard (for Poisson frequency, 90% confidence, 5% accuracy) is a convention. Different parameters give different thresholds. Always state your assumptions.

9. **Do not double-count**: If you use credibility-weighted rates *and* separately apply a Bayesian prior adjustment, you may be shrinking the estimate twice. Be clear about what adjustments have already been applied.

10. **Credibility for trending**: When applying trend to historical data, consider whether the credibility should be applied to the trended or untrended values. The standard approach is to apply credibility to the already-trended experience.

## Resources

- [Klugman, Panjer, Willmot -- Loss Models: From Data to Decisions](https://www.wiley.com/en-us/Loss+Models%3A+From+Data+to+Decisions) -- Standard actuarial exam reference for credibility.
- [Mahler & Dean -- Foundations of Casualty Actuarial Science (Credibility chapter)](https://www.casact.org/) -- CAS-focused treatment.
- [SOA Exam STAM Study Notes -- Credibility section](https://www.soa.org/)
- [Buhlmann & Gisler -- A Course in Credibility Theory and its Applications](https://link.springer.com/book/10.1007/3-540-29273-X) -- The definitive theoretical reference.
- [actuar R package documentation](https://cran.r-project.org/web/packages/actuar/vignettes/credibility.pdf) -- Credibility modeling in R.
- [Wikipedia: Credibility theory](https://en.wikipedia.org/wiki/Credibility_theory) -- Decent overview with formulas.
- [CAS Monograph on Credibility](https://www.casact.org/monograph/cas-monograph-no-4) -- Practical applications.
