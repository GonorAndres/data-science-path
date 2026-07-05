---
tags: [time-series, volatility, actuarial, finance]
status: draft
created: 2026-04-25
updated: 2026-04-25
---

# GARCH and Volatility Modeling

## What it is

GARCH (Generalized Autoregressive Conditional Heteroscedasticity) is a family of models for time-series whose **conditional variance** evolves over time. Where ARIMA models the conditional mean and assumes the variance is constant, GARCH explicitly models the variance as autoregressive in past squared shocks and past variances. The framework underlies modern market-risk modeling, capital calculations under Solvency II and Mexican CNSF, and any pricing problem where heavy tails matter.

## Why it matters

- **Volatility clustering.** Real financial and insurance loss series exhibit periods of high variance followed by periods of low variance; OLS residuals are not iid even after the conditional mean is correctly modeled. GARCH captures the clustering.
- **Heavy tails.** Empirical returns and large-loss distributions have far more mass in the tails than a Gaussian; GARCH with t-innovations is a parsimonious heavy-tail model that is auditable and understood by regulators.
- **VaR and expected shortfall.** Capital requirements and reinsurance pricing depend on tail percentiles. A constant-variance model massively understates these in volatile periods; GARCH adapts.
- **CAS / SOA fellowship-relevant.** Volatility modeling is on Exam SRM, MAS-II, and the FRM curriculum.

## The empirical motivation: volatility clustering

Plot daily log-returns of a stock index, an FX pair, or daily aggregate insurance losses on a noisy line of business. Two facts:

1. The unconditional distribution has fatter tails than a Gaussian (excess kurtosis).
2. The autocorrelation of returns is near zero, but the autocorrelation of **squared** returns is large and slowly decaying. Volatile days cluster together.

OLS residuals from any conditional-mean model on such series are **heteroscedastic**: their variance depends on time. Standard inference (t-tests, confidence intervals) is invalidated. The residuals look uncorrelated but their squares are autocorrelated, which is exactly the diagnostic signature GARCH addresses.

## ARCH(q): the precursor

Engle (1982) introduced the ARCH model:

$$
y_t = \mu + \varepsilon_t,\qquad \varepsilon_t = \sigma_t z_t,\qquad z_t \sim \text{iid}(0, 1)
$$

with conditional variance

$$
\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2
$$

The variance at time $t$ depends on past squared shocks. **Engle's LM test** for ARCH effects regresses squared residuals on their lags and tests joint significance.

Limitations of ARCH:

- Many lags $q$ are usually needed to capture persistent volatility, leading to over-parameterized models.
- Estimated $\alpha_i$ are often non-monotonic in lag, lacking a clear economic interpretation.

## GARCH(p, q): Bollerslev 1986

Bollerslev added an autoregressive term in past variances:

$$
\sigma_t^2 = \omega + \sum_{i=1}^q \alpha_i \varepsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2
$$

The GARCH(1,1) is the workhorse:

$$
\sigma_t^2 = \omega + \alpha\, \varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2
$$

Three parameters cover the same volatility dynamics that ARCH would need 10 to 20 lags for. The intuition: variance has memory ($\beta$ near 1 means yesterday's variance carries forward) and reacts to shocks ($\alpha$ governs the size of the reaction).

### Persistence and stationarity

The **persistence** is $\alpha + \beta$. For covariance-stationarity:

$$
\alpha + \beta < 1
$$

The unconditional variance is

$$
\sigma^2 = \frac{\omega}{1 - \alpha - \beta}
$$

In equity markets $\alpha + \beta$ is typically 0.95 to 0.99: shocks decay slowly. As $\alpha + \beta \to 1$ the model approaches **integrated GARCH** (IGARCH), where shocks are permanent. Estimated persistence above 1 means the model is misspecified or the data has structural breaks.

### Half-life of a shock

A shock decays at rate $(\alpha + \beta)^h$ over $h$ steps. The half-life is

$$
h_{1/2} = \frac{\log 0.5}{\log(\alpha + \beta)}
$$

For $\alpha + \beta = 0.97$ on daily data, the half-life is 23 days. Useful for setting reasonable forecast horizons and for stress-testing how long a market shock persists in capital projections.

## GJR-GARCH: leverage effect

Equity returns exhibit asymmetric volatility: bad news increases volatility more than good news of the same magnitude (Black 1976, "leverage effect"). GJR-GARCH (Glosten, Jagannathan, Runkle 1993) adds an indicator term:

$$
\sigma_t^2 = \omega + \alpha\, \varepsilon_{t-1}^2 + \gamma\, \mathbb{1}[\varepsilon_{t-1} < 0]\, \varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2
$$

If $\gamma > 0$ negative shocks contribute $\alpha + \gamma$ to next-period variance while positive shocks contribute only $\alpha$. Empirically $\gamma$ is consistently positive in equity data; GJR-GARCH almost always outperforms vanilla GARCH on stock indices.

For insurance losses the asymmetry typically points the other way: large losses (positive shocks in the loss series) increase variance more than benign months. The same model class fits with $\gamma$ on the positive-shock indicator.

## EGARCH: log specification

Nelson (1991) modeled $\log \sigma_t^2$ directly:

$$
\log \sigma_t^2 = \omega + \alpha (|z_{t-1}| - \mathbb{E}|z_{t-1}|) + \gamma z_{t-1} + \beta \log \sigma_{t-1}^2
$$

Two advantages:

- No positivity constraints on parameters: the log specification automatically keeps $\sigma_t^2 > 0$.
- Asymmetry is built in via the $\gamma z_{t-1}$ term, which is signed.

EGARCH is the second most-cited GARCH variant after GJR. Estimation is slightly more delicate but `arch` in Python and `rugarch` in R both handle it.

## Estimation: MLE with t-innovations

The Gaussian assumption $z_t \sim \mathcal{N}(0, 1)$ is convenient but underestimates tails. Standard practice is to replace it with a Student t with degrees of freedom $\nu$ estimated jointly:

$$
z_t \sim t_\nu / \sqrt{\nu / (\nu - 2)}
$$

(scaled to unit variance). Smaller $\nu$ means heavier tails; financial data typically estimates $\nu \in [4, 10]$. Skewed-t (Hansen 1994) adds a skewness parameter for further tail asymmetry.

The likelihood is computed by recursing the variance forward, evaluating the t-density at each standardized residual $z_t = \varepsilon_t / \sigma_t$, and summing log-likelihoods. Maximization is over $(\omega, \alpha, \beta, \gamma, \nu)$ subject to positivity and stationarity constraints; numerical optimization with analytical gradients is standard.

```python
import numpy as np
import pandas as pd
from arch import arch_model

# returns: a pandas Series of daily log-returns
returns = pd.read_parquet("returns.parquet")["log_ret"] * 100  # scale to percent

# GJR-GARCH(1,1) with skewed-t innovations and a constant mean
model = arch_model(returns, mean="Constant", vol="GARCH",
                   p=1, o=1, q=1, dist="skewt")
res = model.fit(disp="off")
print(res.summary())

# 30-day-ahead conditional variance forecast
forecast = res.forecast(horizon=30, reindex=False)
sigma_path = np.sqrt(forecast.variance.values[-1])
```

The `arch` library is the de facto Python implementation and supports GARCH, EGARCH, GJR, FIGARCH, HAR, and a range of innovation distributions.

## Applications

### VaR and expected shortfall

For a horizon $h$ and confidence level $\alpha$, **Value at Risk** is the $\alpha$-quantile of the predictive return distribution and **Expected Shortfall** is the conditional expectation in the tail beyond VaR. Under a fitted GARCH-skewt model:

```python
from scipy.stats import t as student_t

q_alpha = student_t.ppf(0.01, df=res.params["nu"])  # 1% quantile of standardized
mu_h = res.params["mu"]
sigma_h = sigma_path[0]  # 1-day-ahead conditional vol

VaR_99 = -(mu_h + q_alpha * sigma_h)        # in percent
ES_99  = -(mu_h + sigma_h * student_t.expect(lambda x: x,
                                             args=(res.params["nu"],),
                                             ub=q_alpha) / 0.01)
```

Backtesting (Kupiec, Christoffersen tests) checks whether the realized exceedance rate matches the nominal $1 - \alpha$. Regulators expect both unconditional and independence backtests.

### Capital modeling

Internal-model capital under Solvency II is a 99.5 percent VaR over a one-year horizon. GARCH provides the volatility input for the multivariate loss-and-asset stress simulation; the cross-asset dependence is supplied by a copula on the standardized residuals (so-called copula-GARCH or DCC-GARCH for time-varying correlations).

### Cat bond pricing and reinsurance

Reinsurance treaties with attachment points well above expected loss are priced from the tail of the conditional loss distribution. A GARCH-skewt model on the underlying loss series produces tail percentiles at any forecast horizon and integrates with attachment, exhaustion, and reinstatement provisions to produce a layer price.

### Heavy-tailed claim severity

For a claim-severity series with volatility clustering (e.g., daily aggregate large losses on a property book during hurricane season), a GARCH model on log-severity captures the time-varying variance and feeds the catastrophe-loss simulation engine.

## Common gotchas

1. **Persistence near unity.** Estimated $\hat\alpha + \hat\beta > 0.99$ is a warning sign: structural breaks, regime changes, or a misspecified mean equation. Test for breaks with Bai-Perron; consider Markov-switching GARCH.
2. **Mean equation misspecification.** A neglected AR or seasonal effect in the mean shows up as ARCH effects in the residuals. Test for ARCH after fitting the mean, not before.
3. **Out-of-sample volatility forecasts collapse.** GARCH forecasts revert toward unconditional variance over long horizons. For multi-step VaR use simulation, not the closed-form one-step forecast iterated.
4. **Daily vs intraday scaling.** Annualized volatility from daily data is $\sigma_{\text{daily}} \sqrt{252}$; from monthly data $\sigma_{\text{monthly}} \sqrt{12}$. Mixing scales is a frequent mistake in actuarial reports.
5. **Multivariate is a different beast.** Joint volatility modeling across assets (DCC-GARCH, BEKK) explodes in parameters. Use simpler factor structures (CAPM-residual GARCH) when you have many series.

## Resources

- `techniques/state-space-and-kalman/README.md` for the conditional-mean modeling that should precede GARCH on the residuals.
- `techniques/extreme-value-theory/README.md` for the unconditional tail modeling that complements GARCH.
- `docs/theory/11_time_series_foundations.md` for the broader time-series context.

## References

- Engle, "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation", Econometrica 1982.
- Bollerslev, "Generalized Autoregressive Conditional Heteroscedasticity", Journal of Econometrics 1986.
- Nelson, "Conditional Heteroskedasticity in Asset Returns: A New Approach" (EGARCH), Econometrica 1991.
- Glosten, Jagannathan, Runkle, "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks" (GJR), Journal of Finance 1993.
- McNeil, Frey, Embrechts, *Quantitative Risk Management: Concepts, Techniques and Tools*, Princeton 2015.
- Tsay, *Analysis of Financial Time Series*, 3rd ed., Wiley 2010.
- Sheppard et al., `arch` Python package, github.com/bashtage/arch.
- Ghalanos, `rugarch` R package.
