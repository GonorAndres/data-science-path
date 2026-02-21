# Extreme Value Theory

## What it is

Extreme Value Theory (EVT) is the branch of statistics that deals with the probabilistic behavior of unusually large (or small) values. While most of statistics focuses on the "center" of a distribution -- means, medians, and typical observations -- EVT focuses on the **tails**. It provides rigorous mathematical tools for answering questions like: "What is the probability of a $500 million hurricane loss?" or "What is the once-in-200-year flood level?" EVT is built on limit theorems (analogous to the Central Limit Theorem but for maxima instead of averages) that dictate which distributions arise for extreme observations, regardless of the underlying data-generating process. This universality makes EVT remarkably powerful for modeling risks that live in the tails.

## Why it matters

- **Tail risk is existential for insurers**: An insurer can absorb many small losses, but a single catastrophic event can threaten solvency. EVT provides the mathematical tools to quantify this tail risk.
- **Reinsurance pricing**: Reinsurance covers the tails of loss distributions (excess-of-loss, catastrophe covers). Pricing these contracts requires estimating probabilities of extreme events with almost no historical precedent.
- **Capital modeling and solvency**: Regulatory frameworks (Solvency II, ORSA, RBC) require insurers to hold capital sufficient to survive extreme adverse scenarios (e.g., 1-in-200 year events). EVT underpins these calculations.
- **Catastrophe modeling**: Natural catastrophe risk (hurricanes, earthquakes, floods) involves modeling events that are rare, high-severity, and have heavy-tailed distributions. EVT provides the theoretical foundation.
- **Beyond insurance**: EVT applies to financial risk management (VaR, Expected Shortfall), environmental science (flood return levels), engineering (structural failure under extreme loads), and network traffic modeling.

## When to use it

- You need to estimate the probability or magnitude of events more extreme than anything in your historical data.
- You are pricing reinsurance layers that attach at high loss levels (e.g., losses excess of $10 million).
- You need to estimate return levels (e.g., "the 100-year flood level" or "the 250-year loss").
- Your data has a heavy tail and standard distributions (normal, lognormal) underestimate extreme quantiles.
- You are building a catastrophe model or an operational risk model.
- You need to compute risk measures like Value-at-Risk (VaR) or Expected Shortfall (CVaR/TVaR) at high confidence levels (99.5%, 99.9%).
- You are working with data where the tail behavior is qualitatively different from the body of the distribution.

## How it works (intuitive)

### Why the tail is special

The Central Limit Theorem (CLT) tells us that the *average* of many observations converges to a normal distribution regardless of the underlying distribution. EVT provides an analogous result for the *maximum* (or minimum): the distribution of the maximum of many observations converges to one of only three possible distributions, regardless of the underlying distribution. This is called the **Fisher-Tippett-Gnedenko theorem**, and it is the foundational result of EVT.

### An analogy: the tallest person in the room

If you put 100 random adults in a room and measure the tallest person's height, then repeat this experiment many times, the distribution of those maximum heights follows a pattern predicted by EVT. The key question is: how heavy is the upper tail?

- **Light tail** (heights are bounded or drop off quickly): The maximum converges to a Gumbel distribution. Even extreme values are not that far from the average.
- **Heavy tail** (incomes, insurance losses): The maximum converges to a Frechet distribution. Extreme values can be vastly larger than typical ones.
- **Bounded tail** (e.g., test scores on a 0-100 scale): The maximum converges to a Weibull distribution. There is a hard upper limit.

### Two main approaches

**1. Block Maxima approach:**
- Divide the data into blocks (e.g., years).
- Take the maximum observation from each block.
- Fit a Generalized Extreme Value (GEV) distribution to these block maxima.

This is like asking: "What does the worst year look like?"

**2. Peaks Over Threshold (POT) approach:**
- Choose a high threshold $u$.
- Model the exceedances above $u$ (i.e., $X - u$ for observations $X > u$).
- Fit a Generalized Pareto Distribution (GPD) to these exceedances.

This is like asking: "Given that a loss exceeds $1 million, how much worse can it get?"

The POT approach is generally preferred in practice because it uses data more efficiently (multiple extreme events per year contribute, not just the single maximum).

## Key concepts

### Generalized Extreme Value (GEV) distribution

The GEV unifies the three types of extreme value distributions:

$$F(x) = \exp\left\{-\left[1 + \xi\left(\frac{x - \mu}{\sigma}\right)\right]^{-1/\xi}\right\}$$

for $1 + \xi(x - \mu)/\sigma > 0$, where:

| Parameter | Symbol | Meaning |
|-----------|--------|---------|
| Location | $\mu$ | Shifts the distribution left or right (like the mean) |
| Scale | $\sigma > 0$ | Stretches or compresses the distribution (like standard deviation) |
| Shape | $\xi$ | Controls the tail behavior -- the most important parameter |

**The shape parameter $\xi$ determines the tail type:**

| $\xi$ value | Distribution type | Tail behavior | Example domains |
|-------------|-------------------|---------------|-----------------|
| $\xi > 0$ | Frechet (Type II) | Heavy tail, power-law decay. No finite upper bound. | Insurance losses, financial returns |
| $\xi = 0$ | Gumbel (Type I) | Light tail, exponential decay. | Temperature extremes, log-normal maxima |
| $\xi < 0$ | Reverse Weibull (Type III) | Bounded upper tail. Finite upper endpoint at $\mu - \sigma/\xi$. | Bounded physical quantities |

For insurance and financial applications, $\xi > 0$ (heavy tails) is almost always the case.

### Generalized Pareto Distribution (GPD)

For the POT approach, exceedances over a threshold $u$ follow the GPD:

$$F_u(y) = P(X - u \leq y \mid X > u) = 1 - \left(1 + \frac{\xi y}{\tilde{\sigma}}\right)^{-1/\xi}$$

for $y > 0$ and $1 + \xi y / \tilde{\sigma} > 0$, where:
- $\tilde{\sigma} = \sigma + \xi(u - \mu)$ is the scale parameter (depends on the threshold)
- $\xi$ is the same shape parameter as the GEV

The GPD has only two parameters (scale and shape), making it easier to estimate than the GEV when the threshold is known.

**Key result (Pickands-Balkema-de Haan theorem):** For a wide class of distributions, the distribution of exceedances over a sufficiently high threshold converges to a GPD. This is the theoretical justification for the POT approach.

### Return levels and return periods

A **return level** $x_T$ is the value expected to be exceeded on average once every $T$ periods (e.g., the 100-year flood level):

$$P(X > x_T) = \frac{1}{T}$$

For the GEV:

$$x_T = \mu - \frac{\sigma}{\xi}\left[1 - \left(-\log(1 - 1/T)\right)^{-\xi}\right]$$

For the GPD (threshold exceedances):

$$x_T = u + \frac{\tilde{\sigma}}{\xi}\left[\left(\frac{T \cdot n_u}{n}\right)^{\xi} - 1\right]$$

where $n_u$ is the number of exceedances above $u$ and $n$ is the total number of observations.

**Return level plot**: A graph of $x_T$ vs. $\log(T)$. For heavy-tailed distributions ($\xi > 0$), this curve is convex (return levels grow faster than logarithmically). For light-tailed distributions ($\xi = 0$), the curve is linear.

### Threshold selection for POT

Choosing $u$ is the most important practical decision in the POT approach. Too low, and the GPD approximation is poor (the threshold is not extreme enough). Too high, and you have too few exceedances to estimate the parameters reliably.

**Diagnostic tools:**

1. **Mean Residual Life (MRL) plot**: Plot the mean excess $e(u) = E[X - u \mid X > u]$ vs. $u$. For a GPD, the MRL is linear in $u$:

   $$e(u) = \frac{\tilde{\sigma} + \xi u}{1 - \xi}$$

   Choose $u$ where the MRL plot becomes approximately linear.

2. **Parameter stability plot**: Fit the GPD at multiple thresholds and plot $\xi$ and the modified scale $\tilde{\sigma}^*$ vs. $u$. Choose $u$ where both parameters stabilize.

3. **Rule of thumb**: Use the top 5-10% of observations. For insurance claims data, this often corresponds to a natural "kink" in the severity distribution where the tail begins.

### Risk measures from EVT

| Risk Measure | Formula (GPD) | Interpretation |
|-------------|---------------|----------------|
| **VaR at level $p$** | $u + \frac{\tilde{\sigma}}{\xi}\left[\left(\frac{1-p}{\bar{F}(u)}\right)^{-\xi} - 1\right]$ | Loss not exceeded with probability $p$ |
| **Expected Shortfall (TVaR)** | $\text{VaR}_p + \frac{\tilde{\sigma} + \xi(\text{VaR}_p - u)}{1 - \xi}$ | Average loss given that VaR is exceeded (requires $\xi < 1$) |

Expected Shortfall (also called Tail Value-at-Risk or Conditional VaR) is the preferred risk measure for solvency purposes because it accounts for the severity of tail losses, not just their threshold.

### Tail index estimation

The **tail index** is $1/\xi$ for heavy-tailed distributions. Larger $\xi$ means heavier tails. Key estimators:

- **Hill estimator**: Non-parametric estimator of $\xi$ for heavy-tailed distributions. Uses the top $k$ order statistics:

  $$\hat{\xi}_{\text{Hill}} = \frac{1}{k} \sum_{i=1}^{k} \log X_{(n-i+1)} - \log X_{(n-k)}$$

  where $X_{(1)} \leq \cdots \leq X_{(n)}$ are the order statistics.

- **Pickands estimator**: Works for all sign of $\xi$.
- **Maximum Likelihood**: Fit the GEV or GPD directly via MLE.

## Python implementation

**Key libraries:**
- `scipy.stats.genextreme`: GEV distribution.
- `scipy.stats.genpareto`: GPD distribution.
- `pyextremes`: Comprehensive EVT package (block maxima, POT, diagnostics, return level plots).

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# --- Block Maxima approach with GEV ---
# Suppose 'annual_maxima' contains the maximum loss for each year
annual_maxima = np.array([12.3, 15.1, 8.7, 22.4, 11.0, 18.9, 9.2, 30.5, 14.6, 10.8,
                          25.3, 13.7, 16.2, 19.8, 11.5, 28.1, 12.0, 17.4, 21.6, 14.0])

# Fit GEV (scipy uses shape parameter c = -xi)
c, loc, scale = stats.genextreme.fit(annual_maxima)
xi = -c  # Convert to standard EVT convention
print(f"Shape (xi): {xi:.4f}, Location (mu): {loc:.2f}, Scale (sigma): {scale:.2f}")

# Return level: 100-year event
p = 1 - 1/100
return_level_100 = stats.genextreme.ppf(p, c, loc, scale)
print(f"100-year return level: {return_level_100:.2f}")

# --- POT approach with GPD ---
losses = np.random.pareto(a=2.5, size=5000) * 100  # Simulated heavy-tailed losses

# Choose threshold (e.g., 95th percentile)
threshold = np.percentile(losses, 95)
exceedances = losses[losses > threshold] - threshold
print(f"Threshold: {threshold:.2f}, Number of exceedances: {len(exceedances)}")

# Fit GPD
xi_gpd, loc_gpd, scale_gpd = stats.genpareto.fit(exceedances, floc=0)
print(f"GPD Shape (xi): {xi_gpd:.4f}, Scale (sigma): {scale_gpd:.2f}")

# VaR at 99.5%
p_var = 0.995
n = len(losses)
n_u = len(exceedances)
prob_exceed = n_u / n
var_995 = threshold + (scale_gpd / xi_gpd) * ((((1 - p_var) / prob_exceed) ** (-xi_gpd)) - 1)
print(f"VaR(99.5%): {var_995:.2f}")

# Expected Shortfall (TVaR)
tvar_995 = var_995 / (1 - xi_gpd) + (scale_gpd - xi_gpd * threshold) / (1 - xi_gpd)
print(f"TVaR(99.5%): {tvar_995:.2f}")
```

```python
# Using pyextremes for comprehensive EVT analysis
from pyextremes import EVA

# Create EVA model from a pandas Series of losses (with datetime index)
model = EVA(data=loss_series)

# Block maxima approach
model.get_extremes(method="BM", block_size="365.25D")
model.fit_model()
model.plot_diagnostic(alpha=0.95)

# POT approach
model.get_extremes(method="POT", threshold=threshold)
model.fit_model()

# Return level plot
model.plot_return_values(return_period=np.arange(2, 500, 1), alpha=0.95)
```

## R implementation (if applicable)

R has excellent EVT packages:

```r
# evd: Extreme Value Distributions
library(evd)

# Fit GEV to block maxima
gev_fit <- fgev(annual_maxima)
summary(gev_fit)
plot(gev_fit)  # Diagnostic plots

# Return level plot
rl <- rlevel.gev(gev_fit, return.period = 100)
print(rl)

# Fit GPD to exceedances (POT)
gpd_fit <- fpot(losses, threshold = threshold)
summary(gpd_fit)
plot(gpd_fit)

# evir: another EVT package
library(evir)

# Mean Residual Life plot (threshold selection diagnostic)
meplot(losses)

# Hill plot (tail index estimation)
hill(losses)

# Shape parameter stability plot
shape(losses)

# ismev: Introduction to Statistical Modeling of Extreme Values
library(ismev)
gev_result <- gev.fit(annual_maxima)
gev.diag(gev_result)

gpd_result <- gpd.fit(losses, threshold = threshold)
gpd.diag(gpd_result)

# POT package
library(POT)
gpd_fit_pot <- fitgpd(losses, threshold = threshold, est = "mle")
summary(gpd_fit_pot)
retlev(gpd_fit_pot, npy = 365, period = 100)
```

## Gotchas & lessons learned

1. **Threshold selection is the hardest part of POT**: There is no perfect automatic method. Use the MRL plot and parameter stability plot together, and try multiple thresholds to assess sensitivity. Document your choice and its justification.

2. **Heavy tails mean high uncertainty**: When $\xi > 0$ (heavy tails), parameter estimates -- especially the shape parameter -- have wide confidence intervals. The heavier the tail, the more uncertain the estimates. This is inherent to the problem, not a flaw of the methodology.

3. **Beware of $\xi > 1$**: If $\xi > 1$, the mean of the distribution does not exist (infinite expected value). If $\xi > 0.5$, the variance does not exist. These situations arise in extreme insurance applications (catastrophe losses). Standard summary statistics become meaningless; use quantiles and tail risk measures instead.

4. **Stationarity assumption**: EVT assumes the underlying process is stationary (the tail behavior does not change over time). For insurance data, this may be violated by climate change (increasing hurricane intensity), inflation (increasing claim severity over time), or regulatory changes. Consider detrending or using non-stationary EVT models.

5. **Small sample sizes in the tail**: By definition, extreme events are rare. You may have only 30-50 exceedances above a high threshold. Estimates are inherently imprecise. Always report confidence intervals.

6. **GEV vs. GPD consistency**: The shape parameter $\xi$ should be the same whether estimated from block maxima (GEV) or exceedances (GPD). If they disagree substantially, something is wrong (likely a threshold issue or non-stationarity).

7. **Do not extrapolate beyond reason**: EVT gives a principled framework for extrapolation, but estimating the 1000-year event from 20 years of data is stretching the method to its limits. The confidence intervals will be enormous. Be transparent about this uncertainty.

8. **Clustering of extremes**: In many real-world processes (storms, financial crises), extreme events cluster in time. This violates the independence assumption. Use declustering techniques (e.g., identify independent clusters and use cluster maxima) before applying EVT.

9. **GPD location parameter**: When fitting the GPD to exceedances $(X - u)$, the location parameter should be fixed at 0 (since exceedances are defined as the amount above the threshold). Some software fits a location parameter by default; fix it to 0.

10. **Model checking**: QQ plots against the fitted GEV/GPD are essential. If the points deviate systematically from the 45-degree line, the model is misspecified. Also examine the return level plot: do the observed data fall within the confidence bands?

11. **Connection to reinsurance**: The expected loss to a reinsurance layer $[d, d+l]$ can be computed directly from the GPD:

$$E[\min(X-d, l) \mid X > d] = \frac{\tilde{\sigma}_d}{1-\xi}\left[1 - \left(1 + \frac{\xi l}{\tilde{\sigma}_d}\right)^{1-1/\xi}\right]$$

This is one of the most practical applications of EVT in actuarial work.

## Resources

- [Coles -- An Introduction to Statistical Modeling of Extreme Values](https://link.springer.com/book/10.1007/978-1-4471-3675-0) -- The standard introductory textbook. Clear, practical, and accessible.
- [Embrechts, Kluppelberg, Mikosch -- Modelling Extremal Events for Insurance and Finance](https://link.springer.com/book/10.1007/978-3-642-33483-2) -- The comprehensive theoretical reference for actuarial applications.
- [McNeil, Frey, Embrechts -- Quantitative Risk Management](https://press.princeton.edu/books/hardcover/9780691166278/quantitative-risk-management) -- EVT chapters in the context of financial and insurance risk management.
- [pyextremes documentation](https://georgebv.github.io/pyextremes/) -- Python EVT package.
- [evd R package](https://cran.r-project.org/web/packages/evd/) -- Comprehensive R EVT package.
- [ismev R package](https://cran.r-project.org/web/packages/ismev/) -- Companion to Coles' textbook.
- [POT R package](https://cran.r-project.org/web/packages/POT/) -- Peaks Over Threshold methods in R.
- [Reiss & Thomas -- Statistical Analysis of Extreme Values](https://link.springer.com/book/10.1007/978-3-7643-7399-3) -- Comprehensive reference with applications.
- [Swiss Re Technical Publication -- Natural Catastrophes](https://www.swissre.com/) -- Industry application of EVT in catastrophe modeling.
- SOA/CAS research on extreme value methods for insurance applications.
