# Loss Reserving

## What it is

Loss reserving is the process of estimating the total amount an insurer will ultimately pay for claims that have already occurred but are not yet fully settled. When an accident happens today, the insurer may not know the final cost for years -- medical treatments continue, lawsuits develop, and late-reported claims trickle in. The difference between what has been paid so far and the projected ultimate cost is the **reserve**. A major component of this reserve is **IBNR** (Incurred But Not Reported): claims that have happened but have not yet been reported to the insurer. Loss reserving is arguably the most critical actuarial function in property & casualty insurance, directly affecting an insurer's financial statements, solvency, and regulatory standing.

## Why it matters

- **Financial accuracy**: Reserves are the largest liability on a P&C insurer's balance sheet. Getting them wrong -- either too high or too low -- distorts the company's financial picture.
- **Solvency**: Regulators require adequate reserves to ensure the company can pay future claims. Under-reserving can lead to insolvency; over-reserving ties up capital unnecessarily.
- **Pricing feedback**: The accuracy of loss reserves feeds back into the ratemaking process. If reserves consistently develop adversely (actual payments exceed estimates), it signals that current rates are inadequate.
- **Exam and professional relevance**: Loss reserving is a core topic on CAS exams and is central to the work of reserving actuaries. The methods covered here (Chain Ladder, Bornhuetter-Ferguson, Cape Cod) are the bread and butter of the profession.
- **Data science application**: Modern reserving increasingly incorporates machine learning for micro-level reserve estimation, but the traditional aggregate methods remain the foundation.

## When to use it

- You are a reserving actuary estimating the IBNR reserve for an insurance portfolio.
- You need to project ultimate losses for financial reporting (GAAP, IFRS 17, statutory).
- You are evaluating reserve adequacy for a prospective acquisition or reinsurance treaty.
- You are analyzing historical claims development patterns for pricing or business planning.
- You want to understand how long it takes claims to fully develop in a given line of business.
- You are building a reserve range or distribution (stochastic reserving) for risk management.

## How it works (intuitive)

### The loss triangle

The fundamental data structure in loss reserving is the **loss triangle** (also called a development triangle or run-off triangle). It organizes historical claims data by two dimensions:

1. **Accident period** (rows): When the loss-causing event occurred (accident year, accident quarter).
2. **Development period** (columns): How much time has elapsed since the accident period.

Each cell contains a cumulative measure (cumulative paid losses or cumulative incurred losses).

**Example: Cumulative Paid Loss Triangle ($000s)**

| Accident Year | Dev 1 | Dev 2 | Dev 3 | Dev 4 | Dev 5 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2020 | 1,000 | 1,500 | 1,700 | 1,800 | 1,830 |
| 2021 | 1,100 | 1,650 | 1,870 | 1,980 | |
| 2022 | 1,200 | 1,800 | 2,040 | | |
| 2023 | 1,050 | 1,575 | | | |
| 2024 | 1,300 | | | | |

The triangle shape arises because the most recent accident years have not had time to develop fully. The goal of loss reserving is to **fill in the lower-right portion of the triangle** -- projecting what the cumulative losses will look like when they are fully developed.

### The Chain Ladder method: an analogy

Think of the Chain Ladder like predicting a child's adult height by looking at growth patterns. If you know that, historically, children who were 100 cm tall at age 5 grew to 120 cm by age 8 (a factor of 1.20), and to 170 cm at adulthood (cumulative factor of 1.70), then you can apply these historical growth factors to a new child who is currently 105 cm at age 5 and estimate their adult height.

The Chain Ladder method does exactly this for claims:
1. Calculate historical **development factors** (link ratios) from the triangle.
2. Apply those factors to incomplete accident years to project ultimate losses.

### The three methods, compared

| Method | Key idea | Best when |
|--------|----------|-----------|
| **Chain Ladder** | Historical development patterns will continue | Stable, mature portfolio with consistent patterns |
| **Bornhuetter-Ferguson** | Blend historical patterns with an independent loss estimate | Immature accident years, volatile experience |
| **Cape Cod** | Like BF but derives the expected loss ratio from the data itself | You do not have a strong prior on expected losses |

## Key concepts

### Chain Ladder (Development Factor Method)

**Step 1: Compute age-to-age development factors (link ratios).**

For each development period transition (e.g., Dev 1 to Dev 2), compute:

$$f_d = \frac{\sum_i C_{i,d+1}}{\sum_i C_{i,d}}$$

where $C_{i,d}$ is the cumulative loss for accident year $i$ at development period $d$. The sum is over all accident years where both $C_{i,d}$ and $C_{i,d+1}$ are observed.

This is the volume-weighted average development factor.

**Step 2: Compute cumulative development factors (CDFs).**

The CDF to ultimate from development period $d$ is:

$$\text{CDF}_d = f_d \times f_{d+1} \times \cdots \times f_{\text{ultimate}}$$

A **tail factor** is often applied at the end to account for development beyond the observed triangle.

**Step 3: Project ultimate losses.**

$$\hat{U}_i = C_{i,d_i} \times \text{CDF}_{d_i}$$

where $d_i$ is the latest observed development period for accident year $i$.

**Step 4: Compute IBNR.**

$$\text{IBNR}_i = \hat{U}_i - C_{i,d_i}$$

Total IBNR = sum across all accident years.

### Bornhuetter-Ferguson (BF) method

The BF method addresses a weakness of Chain Ladder: for immature accident years, the Chain Ladder projection is entirely driven by one or two early data points, which can be volatile.

BF introduces an **a priori expected ultimate loss** ($E_i$), typically derived from:
- An expected loss ratio times earned premium.
- A pricing model estimate.
- Industry benchmarks.

**Formula:**

$$\hat{U}_i^{BF} = C_{i,d_i} + E_i \times \left(1 - \frac{1}{\text{CDF}_{d_i}}\right)$$

In words: Ultimate = Reported-to-date + (Expected unreported portion).

The term $(1 - 1/\text{CDF}_d)$ is the **percentage of ultimate losses that are still unreported** at development period $d$.

**Interpretation:**
- For mature accident years (where $\text{CDF}$ is close to 1), BF gives nearly the same result as Chain Ladder because almost all losses are already reported.
- For immature accident years, BF leans heavily on the a priori estimate $E_i$, which stabilizes the projection.

BF is a blend: the reported portion uses actual data; the unreported portion uses the prior expectation.

### Cape Cod method (Stanard-Buhlmann)

The Cape Cod method is a variant of BF where the a priori expected loss ratio is not externally specified but is estimated from the triangle data itself.

**Step 1: Compute the used-up premium.**

For each accident year $i$ with earned premium $P_i$ and development factor $\text{CDF}_{d_i}$:

$$\text{Used-up Premium}_i = P_i / \text{CDF}_{d_i}$$

This represents how much of the premium has been "used up" by claims that have had a chance to be reported.

**Step 2: Estimate the expected loss ratio (ELR).**

$$\hat{\text{ELR}} = \frac{\sum_i C_{i,d_i}}{\sum_i \text{Used-up Premium}_i}$$

**Step 3: Apply the BF formula using the Cape Cod ELR.**

$$\hat{U}_i^{CC} = C_{i,d_i} + P_i \times \hat{\text{ELR}} \times \left(1 - \frac{1}{\text{CDF}_{d_i}}\right)$$

### IBNR components

IBNR is often decomposed into:

1. **Pure IBNR**: Claims that have occurred but not yet been reported.
2. **IBNER** (Incurred But Not Enough Reserved): Claims that have been reported but whose current case reserves are inadequate.
3. **Development on known claims**: Changes in the value of claims already in the system.

In practice, "IBNR" in financial statements usually encompasses all three components.

### Tail factors

The observed triangle has a finite number of development periods. For long-tailed lines (like workers' compensation or medical malpractice), significant development occurs beyond the triangle's horizon. A **tail factor** extends the CDF from the last observed development period to ultimate.

Common approaches:
- Fit a curve (exponential, inverse power) to the observed development factors and extrapolate.
- Use industry benchmarks for tail behavior.
- Apply a judgmental tail factor based on line-of-business knowledge.

### Reserve diagnostics

- **Residual analysis**: Plot residuals from the Chain Ladder model to check for patterns (calendar-year effects, outlier accident years).
- **Actual vs. Expected**: Compare one-year-later actual development against the prior year's projection.
- **Incurred vs. Paid consistency**: Run the methods on both paid and incurred triangles; large discrepancies signal case reserve adequacy issues.
- **Calendar-year effects**: Diagonal patterns in the triangle often indicate inflation changes, operational shifts, or one-time events.

## Mack chain ladder and stochastic reserving

The deterministic Chain Ladder produces a point estimate of the ultimate but says nothing about uncertainty. A reserve estimate of 100 million MXN with a standard error of 5 million is a different financial reality than the same estimate with a standard error of 30 million. Mack (1993) showed that under three explicit assumptions, the standard error of the Chain Ladder reserve has a closed form, with no parametric distributional assumption.

### Mack's three assumptions

For cumulative losses $C_{i,d}$:

1. **Independence of accident years.** $C_{i,\cdot}$ and $C_{j,\cdot}$ are independent for $i \neq j$. Catastrophes that hit multiple accident years (a long-running asbestos exposure, a medical-billing scandal that develops over a decade) violate this.
2. **Markov property of development.** The expected next-period cumulative loss is proportional to the current cumulative loss with a development-period-specific factor:
   $$
   E[C_{i, d+1} \mid C_{i, 1}, \ldots, C_{i, d}] = f_d \cdot C_{i, d}
   $$
   This is exactly the Chain Ladder assumption, made explicit.
3. **Variance proportional to cumulative.** $\text{Var}[C_{i, d+1} \mid C_{i, 1}, \ldots, C_{i, d}] = \sigma_d^2 \cdot C_{i, d}$. The variance of next-period development scales linearly with the current cumulative; large-loss accident years have proportionally larger development noise.

Under these assumptions Mack derives an unbiased estimator $\hat{\sigma}_d^2$ of the per-period variance and a closed form for the **standard error of the ultimate** for each accident year:

$$
\text{SE}(\hat{U}_i)^2 = \hat{U}_i^2 \sum_{d = d_i}^{D-1} \frac{\hat{\sigma}_d^2 / \hat{f}_d^2}{\hat{C}_{i, d}} + \hat{U}_i^2 \sum_{d = d_i}^{D-1} \frac{\hat{\sigma}_d^2 / \hat{f}_d^2}{\sum_{j=1}^{D-d} C_{j, d}}
$$

The two terms separate **process variance** (random claim development going forward) from **estimation variance** (uncertainty in the development factors themselves). The total reserve standard error aggregates across accident years with appropriate covariance terms because the same development factors $\hat{f}_d$ are used for every accident year.

In practice: report Mack's IBNR point estimate plus its standard error, plus the implied 75th, 95th, and 99th percentiles assuming a lognormal distribution. CAS Exam 7 and IFRS 17 risk-adjustment work both expect Mack-style standard errors as the baseline.

### What Mack did not show

Mack did not derive a full distribution for the IBNR; he provided the first two moments. Higher percentiles require an assumed distribution (lognormal is conventional but not justified by the model itself) or a bootstrap.

## ODP bootstrap reserving

The Over-Dispersed Poisson (ODP) bootstrap (England, Verrall 1999, 2002) produces a full predictive distribution of the IBNR by combining a GLM-based reformulation of the Chain Ladder with a residual bootstrap. It is the most widely used stochastic reserving method in P&C practice.

### Why ODP

The development factors estimated from a triangle have **over-dispersion**: their empirical variance exceeds what a Poisson model would predict, because the underlying claims process has more randomness than pure counting. ODP introduces a dispersion parameter $\phi$ that scales the variance:

$$
\text{Var}(X_{i, d}) = \phi \cdot \mu_{i, d}
$$

where $X_{i, d}$ is the incremental loss and $\mu_{i, d}$ its mean. A Poisson GLM with log link, accident-year and development-period factors, and a dispersion parameter is mathematically equivalent to the volume-weighted Chain Ladder for the point estimate; the ODP framing gives access to GLM machinery for residuals and inference.

### The ODP bootstrap algorithm

```
1. Fit the ODP GLM to the observed triangle. Recover fitted incremental losses
   mu_hat_{i,d} and Pearson residuals
       r_{i,d} = (X_{i,d} - mu_hat_{i,d}) / sqrt(mu_hat_{i,d})
2. Estimate the dispersion phi_hat from the Pearson residuals.
3. For b = 1, ..., B (typically B = 5000 to 10000):
   3a. Resample the residuals with replacement to form r*_{i,d}.
   3b. Construct a pseudo-triangle:
           X*_{i,d} = mu_hat_{i,d} + r*_{i,d} * sqrt(mu_hat_{i,d})
   3c. Refit the Chain Ladder on X*_{i,d} to obtain new development factors
       and ultimate projections.
   3d. Add a process-variance step: simulate the unobserved cells from a
       Gamma (or ODP) distribution with mean from the projection and the
       estimated dispersion.
   3e. Sum the simulated unobserved cells to get a single bootstrap IBNR sample.
4. The empirical distribution of the B IBNR samples is the predictive distribution.
   Compute mean, SE, 75th, 95th, 99th quantiles.
```

Step 3d is critical and often missed by toy implementations. Without process variance, the bootstrap captures only the uncertainty in the development factors (estimation variance) and dramatically understates total reserve risk.

The `chainladder` Python library implements this as `cl.BootstrapODPSample`. The R `ChainLadder` package implements it as `BootChainLadder` with `process.distr = "gamma"` or `"od.pois"`. Both produce the full distribution; reporting the 75th and 99th percentiles is standard for IFRS 17 risk-adjustment work and reinsurance pricing.

## IFRS 17 connection

IFRS 17 (effective 2023, applied across most jurisdictions including Mexico via CNSF) reframes insurance liabilities into three components:

- **Best Estimate Liability (BEL).** The probability-weighted expected present value of future cash flows. The Chain Ladder, BF, or Cape Cod ultimate, less paid-to-date and discounted, is the BEL for non-life claims.
- **Risk Adjustment (RA).** The compensation the entity requires for bearing uncertainty in non-financial risks. Operationally this is a percentile of the ultimate distribution above the BEL, often the 75th or 85th percentile from a stochastic reserving model. Mack standard errors and ODP bootstrap quantiles feed directly into the RA calculation.
- **Contractual Service Margin (CSM).** The unearned profit at recognition, released over the coverage period. Less relevant for short-tailed P&C; central for life and long-duration contracts.

The reserving actuary's IFRS 17 deliverable is therefore not just a single best-estimate number but a distribution of ultimates with documented percentiles. Mack and ODP are the auditor-defensible methods for producing it; full Bayesian reserving (Wuthrich-Merz, Stan, Hamiltonian Monte Carlo on a hierarchical model) is the next-generation approach for groups with the actuarial-DS skill set to maintain it.

## Individual claims ML reserving

The triangle methods aggregate claims and lose information. Two claims that both report 50,000 MXN paid at month 6 may have very different ultimate trajectories: one is a soft-tissue auto claim that closes at 60,000; the other is a complex bodily injury claim that develops to 500,000 over five years. The triangle treats them identically.

Wuthrich (2018) and Gabrielli-Wuthrich (2018) propose **individual claims reserving**: model the ultimate for each claim conditional on its features. Per-claim features include claim type, jurisdiction, policy limit, claimant age, days from accident to first report, body parts injured, attorney involved, prior development pattern of the same claim. The model is a per-claim regressor:

$$
E[\text{Ultimate}_i \mid \text{features}_i, \text{paid}_{i, t}, t]
$$

Architectures used in practice:

- **Gradient-boosted trees** (XGBoost, LightGBM) on engineered features and lagged payment patterns. The pragmatic baseline.
- **Sequence models** (LSTM, transformer) on the per-claim payment sequence, treating each claim as a time series. Higher capacity, more data needed.
- **Hybrid Cox-style models** that separate the time-to-closure from the size given closure.

The aggregate reserve is the sum of per-claim predicted ultimates minus paid-to-date. Validation requires holding out a recent accident year and comparing the per-claim predicted ultimate to the realized ultimate as it develops.

When to use individual claims ML over triangle methods. The crossover happens when (a) per-claim feature data is reliably captured at first notice and (b) the portfolio has heterogeneous claim types whose triangle aggregation hides material variability. Workers compensation and bodily injury portfolios benefit most; pure-property portfolios with homogeneous fast-closing claims see little improvement.

The two approaches complement rather than replace. Many large insurers use the Chain Ladder as the primary regulatory reserve and individual claims ML as an internal management metric to flag claims that the regulatory reserve under-projects.

## Python implementation

The `chainladder` library is the standard Python package for loss reserving.

```python
import chainladder as cl
import pandas as pd

# Load sample data
triangle = cl.load_sample('raa')  # Reinsurance Association of America dataset
print(triangle)

# Chain Ladder
cl_model = cl.Chainladder().fit(triangle)
print("Ultimate losses:\n", cl_model.ultimate_)
print("IBNR:\n", cl_model.ibnr_)

# Bornhuetter-Ferguson
# Requires an a priori expected ultimate (apriori parameter)
bf_model = cl.BornhuetterFerguson(apriori=0.70).fit(
    triangle,
    sample_weight=triangle.latest_diagonal * 0 + 50_000  # earned premium proxy
)
print("BF Ultimate:\n", bf_model.ultimate_)
print("BF IBNR:\n", bf_model.ibnr_)

# Cape Cod
cc_model = cl.CapeCod().fit(
    triangle,
    sample_weight=triangle.latest_diagonal * 0 + 50_000
)
print("Cape Cod Ultimate:\n", cc_model.ultimate_)

# Development factor selection
dev = cl.Development().fit(triangle)
print("Age-to-age factors:\n", dev.ldf_)
print("Cumulative factors:\n", dev.cdf_)

# Tail factor
tail = cl.TailCurve(curve='exponential').fit(dev.transform(triangle))
print("Tail factor:\n", tail.tail_)

# Stochastic reserving (Mack method for standard errors)
mack = cl.MackChainladder().fit(triangle)
print("Reserve standard errors:\n", mack.summary_)
```

## R implementation (if applicable)

The `ChainLadder` package is the standard R package.

```r
library(ChainLadder)

# Load sample triangle
data(RAA)
print(RAA)

# Chain Ladder
cl_result <- MackChainLadder(RAA, est.sigma = "Mack")
summary(cl_result)
plot(cl_result)

# Development factors
ata(RAA)  # Age-to-age factors

# Bornhuetter-Ferguson (requires expected losses)
expected_losses <- c(rep(40000, 10))  # a priori for each accident year
bf_result <- MackChainLadder(RAA, est.sigma = "Mack")
# BF is available via the BootChainLadder function for stochastic BF

# Bootstrap reserving for prediction intervals
boot_result <- BootChainLadder(RAA, R = 10000, process.distr = "gamma")
summary(boot_result)
plot(boot_result)
quantile(boot_result, c(0.75, 0.95, 0.99))
```

## Gotchas & lessons learned

1. **Chain Ladder assumes stable development patterns**: If operational changes, legal environment shifts, or claims handling practices have changed, historical development factors may not apply to recent accident years. Always investigate diagonal (calendar-year) effects.

2. **Garbage in, garbage out**: The triangle must be constructed consistently. Changes in case reserving philosophy, data system migrations, or portfolio composition can corrupt the development pattern. Document all known data issues.

3. **Paid vs. incurred triangles tell different stories**: Paid triangles are less susceptible to case reserve manipulation but develop more slowly. Incurred triangles are more responsive but can be distorted by case reserve changes. Run both and reconcile.

4. **Tail factor selection is judgment-heavy**: For long-tailed lines, the tail factor can represent a significant portion of total reserves. Small changes in the tail assumption have outsized impacts. Always sensitivity-test the tail.

5. **BF is more stable but depends on the prior**: The BF method stabilizes immature years by anchoring to an expected loss ratio. If that expected loss ratio is wrong, BF will be wrong for the immature years. The Chain Ladder is "data-driven" but volatile for immature years. Neither method dominates; they complement each other.

6. **Negative IBNR is possible**: If actual development is more favorable than expected, IBNR can come out negative for some accident years. This is legitimate (it means prior estimates were too conservative) but should be scrutinized.

7. **Do not average blindly**: Volume-weighted average development factors (the standard) give more weight to large accident years. Simple averages give equal weight. Excluding outlier years or using medians is sometimes appropriate. The choice of averaging method is part of the actuary's judgment.

8. **Calendar-year effects**: Inflation, court decisions, or catastrophes can create diagonal patterns in the triangle. These violate the Chain Ladder assumption that development depends only on accident year and development lag, not on calendar year.

9. **Stochastic methods give reserve ranges, not just point estimates**: The Mack method provides standard errors under the Chain Ladder framework. The bootstrap method gives a full distribution. Always accompany point estimates with uncertainty measures.

10. **Segmentation vs. aggregation**: Running the triangle at a more granular level (by coverage, territory, claim type) can reveal different development patterns, but granular triangles are noisier. Find the right balance.

## Resources

- [Friedland -- Estimating Unpaid Claims Using Basic Techniques (CAS)](https://www.casact.org/sites/default/files/2021-01/estimating-unpaid-claims.pdf) -- Free, comprehensive CAS monograph. The single best starting resource.
- [Wuthrich & Merz -- Stochastic Claims Reserving Methods in Insurance](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470772836) -- Advanced stochastic methods.
- [chainladder-python documentation](https://chainladder-python.readthedocs.io/) -- Python implementation with examples.
- [ChainLadder R package](https://cran.r-project.org/web/packages/ChainLadder/) -- R implementation.
- [CAS Exam 7 Study Materials](https://www.casact.org/) -- Detailed treatment of all reserving methods.
- [England & Verrall (2002) -- Stochastic Claims Reserving in General Insurance](https://www.actuaries.org.uk/) -- Foundational paper on stochastic reserving.
- [Mack (1993) -- Distribution-free Calculation of the Standard Error of Chain Ladder Reserve Estimates](https://www.casact.org/) -- The Mack method paper.
- Wuthrich, "Machine learning in individual claims reserving", Scandinavian Actuarial Journal 2018.
- Gabrielli, Wuthrich, "An Individual Claims History Simulation Machine", Risks 2018.
- IFRS 17 Insurance Contracts, IFRS Foundation 2017 (effective 2023).
- CNSF, "Disposiciones de caracter general aplicables a las instituciones de seguros y de fianzas" (Mexican implementation of IFRS 17, 2023 onward).
