# Decomposition and Stationarity

## Why decompose

Before modeling, understand the structure: trend, seasonality, residual. For retail daily sales:

- **Weekly seasonality.** Strong and dominant (weekends vs weekdays).
- **Annual seasonality.** Christmas, back-to-school, Super Bowl, payday cycles.
- **Trend.** Generally slow; reflects category shifts, population, market share.
- **Promotional spikes.** Not strictly seasonal but structurally predictable from covariates.

STL (Seasonal-Trend decomposition with Loess) handles this well:

```python
from statsmodels.tsa.seasonal import STL
stl = STL(series, period=7).fit()
stl.plot()
```

For this project, STL is used for EDA, not as a modeling component. The modeling stack (LightGBM with lag features) learns these components implicitly.

## Stationarity

Weakly stationary: mean, variance, autocovariance constant over time. Most practical retail series are not stationary:

- Trend in total demand (inflation, population).
- Non-constant variance (Christmas weeks have higher variance).
- Structural breaks (new store opens, category repricing).

### Tests

- **ADF (Augmented Dickey-Fuller).** Null: non-stationary (unit root). Rejection suggests stationarity.
- **KPSS.** Null: stationary. Rejection suggests non-stationary.
- Use both; disagreements point to trend-stationary or near-unit-root cases.

### Implications

Classical ARIMA needs stationarity after differencing. Prophet handles non-stationary series directly via its decomposition. LightGBM is agnostic, but lag features transform the problem into a roughly stationary supervised learning task.

## Differencing

If using ARIMA:

- First-order difference: $\Delta Y_t = Y_t - Y_{t-1}$.
- Seasonal difference: $\Delta_7 Y_t = Y_t - Y_{t-7}$.
- Sometimes both.

After differencing, re-test stationarity.

For this project, differencing is not applied directly; the LightGBM approach uses raw sales and lag features, which implicitly models the relationship without requiring stationarity.

## Cross-links

- `docs/theory/11_time_series_foundations.md`: broader theory.
- `02_classical_vs_ml_forecasts.md`: why we chose LightGBM over ARIMA.
