# Time Series Analysis (ARIMA, Exponential Smoothing, STL Decomposition)

## What it is

Time series analysis is a collection of techniques for modeling and forecasting data that is collected sequentially over time. Unlike cross-sectional data (where each observation is independent), time series data has **temporal ordering** and **autocorrelation** -- past values carry information about future values. This README covers three core approaches: **ARIMA** (AutoRegressive Integrated Moving Average) for flexible linear forecasting, **Exponential Smoothing** (ETS) for weighted-average forecasting with trend and seasonality components, and **STL Decomposition** for breaking a time series into its constituent parts (trend, seasonal, residual).

## Why it matters

Time series problems are everywhere in actuarial and business contexts: forecasting claim volumes, projecting premium income, predicting customer churn rates over time, modeling economic indicators for reserving, and budgeting. Unlike standard regression (where you predict Y from X), time series forecasting predicts **future values of the same variable** using its own past behavior. Understanding stationarity, seasonality, and autocorrelation is essential -- applying standard regression to time series data without accounting for these properties leads to spurious relationships, overconfident predictions, and unreliable intervals.

## When to use it

- **STL Decomposition**: As a first step to understand any time series -- separate trend from seasonality from noise. Also useful for seasonal adjustment before applying other models.
- **Exponential Smoothing (ETS)**: When the time series has a clear trend and/or seasonal pattern, when you need reliable prediction intervals, or when you want an automated method (many implementations auto-select the best ETS variant).
- **ARIMA**: When the time series has complex autocorrelation structure that is not purely trend + seasonality. When you want a flexible, general-purpose linear model. When the series may require differencing to become stationary.
- **Seasonal ARIMA (SARIMA)**: When the series has both non-seasonal and seasonal autocorrelation (e.g., monthly data with yearly seasonality).

## How it works (intuitive)

### STL Decomposition

STL (Seasonal and Trend decomposition using Loess) breaks a time series $y_t$ into three components:

$$y_t = T_t + S_t + R_t$$

Where:
- $T_t$ = **Trend**: The long-term direction (upward, downward, flat).
- $S_t$ = **Seasonal**: The repeating pattern within each cycle (e.g., claims spike every winter).
- $R_t$ = **Remainder (residual)**: What is left after removing trend and seasonality.

**How it works**:

1. Start with an initial estimate of the trend (using a moving average).
2. Subtract the trend to get the detrended series.
3. Estimate the seasonal component from the detrended series by averaging values at the same seasonal position across cycles.
4. Subtract the seasonal component to update the trend estimate.
5. Iterate until convergence.

STL uses **Loess** (locally weighted scatterplot smoothing) for both the trend and seasonal estimation, which makes it robust to outliers.

**Analogy**: Imagine peeling layers off an onion. The outer layer is the overall trend (the series is generally going up or down). The next layer is the seasonal pattern (it oscillates regularly). The core is the residual -- the random, unexplained variation.

### Exponential Smoothing (ETS)

Exponential smoothing creates forecasts as **weighted averages of past observations**, where the weights decay exponentially -- recent observations get much more weight than distant ones.

**Simple Exponential Smoothing** (no trend, no seasonality):

$$\hat{y}_{t+1} = \alpha y_t + (1-\alpha) \hat{y}_t$$

Where $\alpha \in (0,1)$ is the smoothing parameter. High $\alpha$ means the forecast reacts quickly to recent changes. Low $\alpha$ means the forecast is more stable.

**Analogy**: It is like a running average that pays more attention to what happened recently. If $\alpha = 0.8$, today's observation contributes 80% and the historical forecast 20%. If $\alpha = 0.2$, it is the reverse.

**Holt's method** adds a trend component:

$$\text{Level: } l_t = \alpha y_t + (1-\alpha)(l_{t-1} + b_{t-1})$$
$$\text{Trend: } b_t = \beta(l_t - l_{t-1}) + (1-\beta)b_{t-1}$$
$$\text{Forecast: } \hat{y}_{t+h} = l_t + h \cdot b_t$$

**Holt-Winters** adds seasonality (additive or multiplicative):

For **additive seasonality** (seasonal swings have constant magnitude):

$$\hat{y}_{t+h} = l_t + h \cdot b_t + s_{t+h-m}$$

For **multiplicative seasonality** (seasonal swings grow proportionally with the level):

$$\hat{y}_{t+h} = (l_t + h \cdot b_t) \cdot s_{t+h-m}$$

Where $m$ is the seasonal period (e.g., 12 for monthly data with yearly seasonality).

The **ETS framework** (Error, Trend, Seasonal) provides a systematic taxonomy of 30 exponential smoothing models based on three choices:
- **Error**: Additive (A) or Multiplicative (M)
- **Trend**: None (N), Additive (A), Additive damped (Ad), Multiplicative (M), Multiplicative damped (Md)
- **Seasonal**: None (N), Additive (A), Multiplicative (M)

### ARIMA

ARIMA stands for AutoRegressive Integrated Moving Average. It models the time series after making it **stationary** (constant mean and variance over time).

**Three components**:

**AR (AutoRegressive, order p)**: The current value depends linearly on its past $p$ values.

$$y_t = \phi_1 y_{t-1} + \phi_2 y_{t-2} + \ldots + \phi_p y_{t-p} + \epsilon_t$$

**Analogy**: Your mood today depends on your mood yesterday and the day before. If $\phi_1 = 0.7$, then 70% of today's value is "inherited" from yesterday.

**I (Integrated, order d)**: The number of times you difference the series to make it stationary. First differencing: $y'_t = y_t - y_{t-1}$. If the series has a linear trend, first differencing removes it.

**Analogy**: Instead of looking at the stock price, you look at the daily change in stock price. If the changes are roughly constant (stationary), you only needed one round of differencing ($d=1$).

**MA (Moving Average, order q)**: The current value depends linearly on past $q$ forecast errors.

$$y_t = \epsilon_t + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q}$$

**Analogy**: Today's sales are normal, plus a correction for how wrong your forecasts were recently. If yesterday's forecast was too high (positive error), today's value adjusts downward.

**Combined ARIMA(p, d, q)**:

$$y'_t = c + \phi_1 y'_{t-1} + \ldots + \phi_p y'_{t-p} + \epsilon_t + \theta_1 \epsilon_{t-1} + \ldots + \theta_q \epsilon_{t-q}$$

Where $y'_t$ is the differenced series.

### The Box-Jenkins Methodology

The classical approach to building an ARIMA model:

1. **Visualize**: Plot the series. Look for trend, seasonality, outliers, structural breaks.
2. **Stationarize**: Apply differencing and/or transformations (log, Box-Cox) to make the series stationary. Use the ADF (Augmented Dickey-Fuller) test to check.
3. **Identify**: Use the ACF (AutoCorrelation Function) and PACF (Partial AutoCorrelation Function) plots to determine $p$ and $q$:
   - ACF shows the correlation between $y_t$ and $y_{t-k}$ for lags $k = 1, 2, 3, \ldots$
   - PACF shows the "direct" correlation between $y_t$ and $y_{t-k}$, removing the effect of intermediate lags.
   - **AR(p) signature**: PACF cuts off after lag $p$; ACF decays gradually.
   - **MA(q) signature**: ACF cuts off after lag $q$; PACF decays gradually.
   - **ARMA**: Both ACF and PACF decay gradually.
4. **Estimate**: Fit the ARIMA model using maximum likelihood.
5. **Diagnose**: Check that residuals are white noise (no remaining autocorrelation). Use the Ljung-Box test and residual ACF plot.
6. **Forecast**: Generate point forecasts and prediction intervals.

**Seasonal ARIMA (SARIMA)** adds seasonal terms: ARIMA$(p,d,q) \times (P,D,Q)_m$, where $m$ is the seasonal period, and $P, D, Q$ are the seasonal AR, differencing, and MA orders.

### Stationarity

A time series is **stationary** if its statistical properties (mean, variance, autocorrelation) do not change over time. ARIMA requires stationarity (after differencing).

**Tests for stationarity**:
- **Augmented Dickey-Fuller (ADF) test**: Null hypothesis = series has a unit root (non-stationary). p-value < 0.05 suggests stationarity.
- **KPSS test**: Null hypothesis = series is stationary. p-value < 0.05 suggests non-stationarity.
- Use both tests together for confidence.

## Key concepts

### ACF and PACF Interpretation Guide

| Pattern | ACF | PACF | Suggested Model |
|---------|-----|------|----------------|
| AR(p) | Decays exponentially or sinusoidal | Cuts off after lag p | ARIMA(p,d,0) |
| MA(q) | Cuts off after lag q | Decays exponentially or sinusoidal | ARIMA(0,d,q) |
| ARMA(p,q) | Decays after lag q | Decays after lag p | ARIMA(p,d,q) |
| White noise | All near zero | All near zero | No model needed |

### Information Criteria for Model Selection

Rather than relying solely on ACF/PACF (which is subjective), use information criteria to compare candidate models:

- **AIC**: $-2\ln L + 2k$ -- Penalizes complexity lightly. Tends to select larger models.
- **AICc**: Corrected AIC for small samples. Preferred in practice.
- **BIC**: $-2\ln L + k\ln n$ -- Penalizes complexity more. Tends to select simpler models.

Lower is better. The `auto_arima` function in `pmdarima` automates this search.

### Forecast Accuracy Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | $\frac{1}{h}\sum_{i=1}^{h}\|e_i\|$ | Same units as data |
| RMSE | $\sqrt{\frac{1}{h}\sum_{i=1}^{h}e_i^2}$ | Penalizes large errors |
| MAPE | $\frac{100}{h}\sum_{i=1}^{h}\|e_i/y_i\|$ | Percentage, but undefined when $y_i = 0$ |
| MASE | $\frac{1}{h}\sum_{i=1}^{h}\|e_i\| / \text{MAE}_{\text{naive}}$ | Scale-free; MASE > 1 means worse than naive forecast |

## Python implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima

# --- STL Decomposition ---
stl = STL(series, period=12, robust=True)
result = stl.fit()
result.plot()

# Access components
trend = result.trend
seasonal = result.seasonal
residual = result.resid

# --- Stationarity Testing ---
adf_result = adfuller(series, autolag='AIC')
print(f"ADF Statistic: {adf_result[0]:.4f}, p-value: {adf_result[1]:.4f}")

kpss_result = kpss(series, regression='c', nlags='auto')
print(f"KPSS Statistic: {kpss_result[0]:.4f}, p-value: {kpss_result[1]:.4f}")

# --- ACF/PACF Plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(series.diff().dropna(), lags=40, ax=ax1)
plot_pacf(series.diff().dropna(), lags=40, ax=ax2)

# --- Exponential Smoothing (Holt-Winters) ---
hw_model = ExponentialSmoothing(
    train, trend='add', seasonal='mul', seasonal_periods=12
).fit(optimized=True)
hw_forecast = hw_model.forecast(steps=12)

# --- ARIMA (manual) ---
sarima_model = SARIMAX(
    train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False, enforce_invertibility=False
).fit(disp=False)
print(sarima_model.summary())
sarima_model.plot_diagnostics(figsize=(12, 8))

# --- Auto ARIMA (automated) ---
auto_model = auto_arima(
    train, seasonal=True, m=12,
    stepwise=True, trace=True,
    error_action='ignore', suppress_warnings=True
)
print(auto_model.summary())
forecast = auto_model.predict(n_periods=12)
```

## R implementation (if applicable)

```r
library(forecast)
library(tseries)

# STL Decomposition
ts_data <- ts(values, frequency = 12, start = c(2020, 1))
stl_result <- stl(ts_data, s.window = "periodic")
plot(stl_result)

# Stationarity test
adf.test(ts_data)
kpss.test(ts_data)

# Exponential Smoothing
hw_model <- hw(ts_data, seasonal = "multiplicative", h = 12)
plot(hw_model)

# Auto ARIMA
auto_model <- auto.arima(ts_data, seasonal = TRUE, stepwise = FALSE)
summary(auto_model)
checkresiduals(auto_model)
forecast_result <- forecast(auto_model, h = 12)
plot(forecast_result)
```

## Gotchas & lessons learned

1. **Always plot the data first.** No amount of statistical testing replaces a visual inspection. Look for trend, seasonality, outliers, structural breaks (e.g., COVID-19 effects), and level shifts.

2. **Stationarity is a prerequisite for ARIMA, not optional.** Fitting ARIMA to a non-stationary series gives meaningless results. Test with ADF and KPSS. When in doubt, difference.

3. **Do not over-difference.** Differencing once removes a linear trend. Differencing twice removes a quadratic trend. Differencing more than twice is rarely justified and introduces noise. If the ADF test says the differenced series is stationary, stop.

4. **Log-transform right-skewed series before modeling.** If the series values span orders of magnitude (common for financial data), take the log first. This stabilizes variance and often makes the series more normal.

5. **Seasonal differencing and regular differencing are different.** For monthly data with yearly seasonality: regular differencing is $y_t - y_{t-1}$; seasonal differencing is $y_t - y_{t-12}$. You may need both.

6. **Exponential smoothing is often as good as ARIMA in practice.** The M-competition results (empirical forecasting competitions) consistently show that ETS methods perform comparably to ARIMA. Do not dismiss them as "too simple."

7. **Auto ARIMA is useful but not infallible.** Always check the residual diagnostics. Auto ARIMA can miss seasonal patterns if the seasonal period is not specified correctly, or if the series has structural breaks.

8. **Prediction intervals widen over time.** Unlike point forecasts (which are often reasonably accurate for a few steps ahead), prediction intervals grow rapidly. Be honest about uncertainty in long-horizon forecasts.

9. **Cross-validation for time series is NOT random k-fold.** Use **time series cross-validation** (expanding window or rolling window) to respect temporal ordering. Never let future data leak into the training set.

10. **Beware of the "calendar effect."** Monthly data can be distorted by the varying number of days per month. Consider adjusting for trading days or using daily data when precision matters.

11. **In actuarial contexts, combine time series with domain knowledge.** A pure statistical forecast of claim volumes might miss known regulatory changes, product launches, or market shifts. Use the statistical forecast as a baseline and adjust with expert judgment.

## Resources

- **Hyndman & Athanasopoulos, "Forecasting: Principles and Practice" (3rd ed.)** -- Free online at [otexts.com/fpp3](https://otexts.com/fpp3/). The best introductory time series forecasting textbook. Uses R but concepts apply universally.
- **statsmodels Time Series documentation**: [statsmodels.org/stable/tsa.html](https://www.statsmodels.org/stable/tsa.html)
- **pmdarima documentation**: [alkaline-ml.com/pmdarima](https://alkaline-ml.com/pmdarima/) -- Python port of R's `auto.arima`.
- **Box, Jenkins, Reinsel & Ljung, "Time Series Analysis" (5th ed.)** -- The original ARIMA reference. Rigorous and comprehensive.
- **Rob Hyndman's blog**: [robjhyndman.com](https://robjhyndman.com/) -- Practical insights from one of the world's leading forecasting researchers.
