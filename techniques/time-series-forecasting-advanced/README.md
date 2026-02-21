# Advanced Time Series Forecasting

## What it is

Advanced time series forecasting goes beyond basic ARIMA and exponential smoothing to tackle the messy realities of real-world temporal data: multiple seasonality patterns, external drivers, multivariate relationships, and structural breaks. This guide covers four powerful approaches: **SARIMA** (Seasonal ARIMA), **Prophet** (Meta/Facebook's additive decomposition model), **VAR** (Vector Autoregression for multivariate series), and **State Space Models** (the flexible framework underlying many of the above). Each has strengths for different forecasting scenarios, and understanding when to reach for which tool is half the battle.

## Why it matters

Time series data is everywhere in actuarial and insurance work:

- **Claims frequency and severity**: Forecasting monthly claim counts and average severity for reserving and budgeting.
- **Premium volume**: Projecting written premium for business planning and capital allocation.
- **Loss ratios**: Tracking and forecasting combined ratios over time.
- **Exposure growth**: Predicting policy count trajectories for capacity planning.
- **Mortality and morbidity trends**: Life and health actuaries forecast trend rates for pricing and valuation.
- **Economic indicators**: Interest rates, inflation, unemployment -- all time series that affect insurance liabilities.
- **Staffing and operations**: Predicting call volumes, claim submission spikes (e.g., after a storm), and resource needs.

Basic models often fail when data has multiple seasonal patterns (daily and weekly and yearly), external regressors (weather, marketing campaigns), or complex trend changes. Advanced methods handle these gracefully.

## When to use it

- **SARIMA**: You have a single time series with clear seasonal patterns and want a principled statistical model. Good for moderate-length series (50+ observations) with stable seasonality.
- **Prophet**: You have daily or sub-daily data with multiple seasonalities (weekly + yearly), holiday effects, and occasional trend changes. You want a model that is easy to tune and handles missing data gracefully.
- **VAR**: You have multiple related time series (e.g., claims frequency, severity, and premium volume) and want to capture how they influence each other over time.
- **State space models**: You want maximum flexibility -- time-varying parameters, missing data handling, or you need to combine multiple components (trend, season, regression) in a principled probabilistic framework.

## How it works (intuitive)

### SARIMA: Seasonal ARIMA

**ARIMA recap**: ARIMA(p, d, q) models a time series as a combination of:
- **AR (Autoregressive)**: The current value depends on its own past values (e.g., "this month's claims count depends on last month's and the month before").
- **I (Integrated)**: Differencing the series to make it stationary (removing trends).
- **MA (Moving Average)**: The current value depends on past forecast errors.

**SARIMA** adds a seasonal component: SARIMA(p, d, q)(P, D, Q, s), where the uppercase letters are the seasonal AR, differencing, and MA orders, and s is the seasonal period (e.g., 12 for monthly data with yearly seasonality).

**Analogy**: Think of ARIMA as a model that remembers the recent past ("yesterday's claims predict today's"). SARIMA adds a longer memory: "claims this January are also related to claims last January, and January two years ago."

**Step-by-step**:
1. **Plot and inspect** the series for trends and seasonality.
2. **Test for stationarity** (ADF test). If non-stationary, difference the series (d=1 or D=1 for seasonal differencing).
3. **Examine ACF and PACF plots** to identify candidate p, q, P, Q values.
4. **Fit candidate models** and compare using AIC/BIC.
5. **Check residuals**: They should look like white noise (no autocorrelation, roughly normal).
6. **Forecast** and compute prediction intervals.

Alternatively, use `auto_arima` from `pmdarima` to automate the order selection.

### Prophet: Additive Decomposition

Prophet, developed by Meta, decomposes a time series into three (or four) additive components:

$$y(t) = g(t) + s(t) + h(t) + \epsilon_t$$

Where:
- $g(t)$ = **trend** (piecewise linear or logistic growth)
- $s(t)$ = **seasonality** (Fourier series for weekly, yearly, or custom periods)
- $h(t)$ = **holiday/event effects** (spikes or dips on specific dates)
- $\epsilon_t$ = **noise** (everything else)

**How it works**:

1. **Trend**: Prophet fits a piecewise linear trend with automatic changepoint detection. It finds points where the trend growth rate changes (e.g., when a new product launched or a regulation changed). You can also specify known changepoints.

2. **Seasonality**: Instead of rigid seasonal indices, Prophet models seasonality using Fourier series -- sums of sine and cosine functions. This can capture complex seasonal patterns smoothly. You specify how many Fourier terms (higher = more flexible seasonality).

3. **Holidays**: You provide a list of holidays/events with dates. Prophet estimates a separate effect for each one. In insurance, this could be storms, regulatory changes, or open enrollment periods.

4. **External regressors**: You can add external variables (temperature, marketing spend, economic indicators) as additional regressors.

**Analogy**: Prophet treats a time series like a recipe with separate ingredients. The trend is the base temperature of the oven (gradually changing). Seasonality is a recurring pattern like daily temperature cycles. Holidays are one-off spikes. You can understand and adjust each ingredient separately.

**Why Prophet is popular**:
- Handles missing data and outliers gracefully
- Works well out of the box with minimal tuning
- Intuitive parameters that non-statisticians can understand
- Uncertainty intervals built in
- Easy to add domain knowledge (known trend changes, holidays)

### VAR: Vector Autoregression

VAR models multiple time series simultaneously, allowing each series to depend on its own past values AND the past values of all other series in the system.

For two series $y_{1,t}$ and $y_{2,t}$:

$$y_{1,t} = c_1 + \phi_{11} y_{1,t-1} + \phi_{12} y_{2,t-1} + \epsilon_{1,t}$$
$$y_{2,t} = c_2 + \phi_{21} y_{1,t-1} + \phi_{22} y_{2,t-1} + \epsilon_{2,t}$$

**Analogy**: In a VAR model, each time series is like a person in a conversation -- what they say depends on what they said earlier AND what the other people said. Claims frequency might depend on its own history and also on recent changes in premium volume or exposure counts.

**When VAR shines**:
- You believe multiple series influence each other (Granger causality).
- You want to analyze impulse responses: "If claims frequency spikes this month, how does severity respond over the next 6 months?"
- You need to forecast multiple related series coherently (not independently).

**Key steps**:
1. Ensure all series are stationary (difference if needed -- VARIMA is the integrated variant).
2. Select the lag order using AIC/BIC.
3. Fit the VAR model.
4. Check residuals for remaining autocorrelation.
5. Use Granger causality tests to check if one series actually helps predict another.
6. Compute impulse response functions and forecast error variance decomposition.

### State Space Models

State space models are the most flexible framework. They model an observed time series as a function of hidden (latent) states that evolve over time.

**Observation equation**: $y_t = Z_t \alpha_t + \epsilon_t$
**State transition equation**: $\alpha_t = T_t \alpha_{t-1} + R_t \eta_t$

Where $\alpha_t$ is the hidden state (e.g., the "true" level and trend of the series, unobserved), and $y_t$ is what you actually observe (noisy measurements).

**Analogy**: Imagine you are tracking a submarine. You cannot see it directly (hidden state), but you get sonar pings (observations) that are noisy. The Kalman filter (the algorithm that fits state space models) estimates the submarine's true position and trajectory from the noisy pings.

**Why they matter**: Exponential smoothing, ARIMA, structural time series models, dynamic linear models, and even some forms of Prophet can all be expressed as state space models. They provide:
- Principled handling of missing data
- Time-varying parameters (coefficients that change over time)
- Combined trend + seasonal + regression components
- Exact likelihood computation via the Kalman filter

## Key concepts

### Stationarity

A stationary time series has constant statistical properties (mean, variance) over time. Most time series models require stationarity. Tools:

- **ADF test** (Augmented Dickey-Fuller): Tests the null hypothesis that a unit root exists (non-stationary). p-value < 0.05 suggests stationarity.
- **KPSS test**: Tests the null that the series IS stationary. Useful as a complement to ADF.
- **Differencing**: Subtract the previous value ($y_t - y_{t-1}$) to remove trends. Seasonal differencing ($y_t - y_{t-s}$) removes seasonal patterns.

### Seasonality handling

| Method | Seasonality approach | Strengths |
|--------|---------------------|-----------|
| SARIMA | Seasonal AR/MA terms + seasonal differencing | Theoretically principled, captures complex seasonal dynamics |
| Prophet | Fourier series decomposition | Flexible, handles multiple seasonalities easily |
| VAR | External seasonal dummies or pre-differencing | Keeps multivariate structure |
| State space | Seasonal component in state vector | Time-varying seasonal patterns |

### Forecast evaluation metrics

**MAPE (Mean Absolute Percentage Error)**:
$$\text{MAPE} = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

Intuitive (expressed as a percentage) but problematic when actual values are near zero (division by zero) and asymmetric (penalizes over-forecasts and under-forecasts differently).

**MASE (Mean Absolute Scaled Error)**:
$$\text{MASE} = \frac{\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|}{\frac{1}{T-s}\sum_{t=s+1}^{T}|y_t - y_{t-s}|}$$

Scales the forecast errors by the errors of a naive seasonal forecast. MASE < 1 means your model beats the seasonal naive baseline. Recommended by Hyndman and others as the most robust general-purpose forecast metric.

**RMSE (Root Mean Squared Error)**:
$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

Penalizes large errors more. Use when large forecast errors are disproportionately costly.

**Coverage**: What fraction of actual values fall within your prediction intervals? If you claim 95% intervals, roughly 95% of actuals should fall inside.

### Multi-step forecasting strategies

- **Recursive (iterated)**: Forecast one step ahead, then use that forecast as input to forecast two steps ahead, and so on. Errors accumulate.
- **Direct**: Train separate models for each forecast horizon (one model for h=1, another for h=2, etc.). Avoids error accumulation but requires more models.
- **Rectify**: Use recursive forecasts but correct for known bias at each horizon.

## Python implementation

### SARIMA with statsmodels and pmdarima

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
import matplotlib.pyplot as plt

# Automatic order selection
auto_model = auto_arima(
    y_train,
    seasonal=True,
    m=12,                  # Seasonal period (12 for monthly)
    d=None,                # Auto-detect differencing order
    D=None,                # Auto-detect seasonal differencing
    max_p=3, max_q=3,
    max_P=2, max_Q=2,
    stepwise=True,
    trace=True,
    error_action='ignore',
    suppress_warnings=True
)
print(auto_model.summary())

# Manual SARIMA fit
model = SARIMAX(
    y_train,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12),
    enforce_stationarity=False,
    enforce_invertibility=False
)
results = model.fit(disp=False)
print(results.summary())

# Forecast
forecast = results.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
conf_int = forecast.conf_int(alpha=0.05)
```

### Prophet

```python
from prophet import Prophet
import pandas as pd

# Prophet requires columns named 'ds' (date) and 'y' (value)
df = pd.DataFrame({'ds': dates, 'y': values})

# Basic model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,  # Flexibility of trend changes (lower = smoother)
    seasonality_prior_scale=10.0   # Flexibility of seasonality
)

# Add holidays
holidays = pd.DataFrame({
    'holiday': 'hurricane_season',
    'ds': pd.to_datetime(['2023-08-01', '2024-08-01']),
    'lower_window': 0,
    'upper_window': 90  # Effect lasts 90 days after the date
})
model = Prophet(holidays=holidays)

# Add external regressors
model.add_regressor('temperature')
model.add_regressor('marketing_spend')

model.fit(df_train)

# Forecast
future = model.make_future_dataframe(periods=365)
future['temperature'] = temperature_forecast  # Must provide future regressor values
future['marketing_spend'] = spend_forecast
forecast = model.predict(future)

# Plot components
model.plot_components(forecast)
```

### VAR with statsmodels

```python
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests

# Prepare multivariate data (DataFrame with multiple columns)
# e.g., columns: ['claims_freq', 'severity', 'premium_volume']
model = VAR(df_stationary)

# Select lag order
lag_order = model.select_order(maxlags=12)
print(lag_order.summary())

# Fit
results = model.fit(lag_order.aic)
print(results.summary())

# Granger causality test
granger_results = grangercausalitytests(
    df_stationary[['claims_freq', 'severity']],
    maxlag=4,
    verbose=True
)

# Impulse response functions
irf = results.irf(periods=24)
irf.plot()

# Forecast
forecast = results.forecast(df_stationary.values[-results.k_ar:], steps=12)
```

### State space models with statsmodels

```python
from statsmodels.tsa.structural import UnobservedComponents

# Structural time series model (local linear trend + seasonal)
model = UnobservedComponents(
    y_train,
    level='local linear trend',
    seasonal=12,
    freq_seasonal=[{'period': 12, 'harmonics': 4}]
)
results = model.fit(disp=False)
print(results.summary())

# Extract components
fig = results.plot_components()

# Forecast
forecast = results.get_forecast(steps=24)
```

## Gotchas & lessons learned

### Stationarity is not optional
SARIMA and VAR assume stationarity. If you skip this step, your parameter estimates are unreliable and your forecasts will diverge. Always check with ADF and KPSS tests, and difference as needed. Prophet handles non-stationarity internally.

### Overfitting in-sample does not mean good forecasts
A model with many parameters can fit historical data perfectly but produce terrible forecasts. Use proper time series cross-validation (rolling origin or expanding window), never random k-fold.

### Time series cross-validation is different
You cannot randomly shuffle time series data. Always train on the past and test on the future. Use `TimeSeriesSplit` from scikit-learn or implement rolling-origin evaluation.

### Prophet is not magic
Prophet works remarkably well for daily data with clear seasonal patterns. It struggles with:
- Very short series (< 2 years of daily data)
- High-frequency data with complex dynamics (use ARIMA or state space)
- Series where external regressors dominate (consider regression-based approaches)
- Abrupt structural breaks that are not pre-specified

### VAR requires all series to be stationary and comparable in scale
Mixing a series in dollars (millions) with one in percentages (0-1) without standardizing will cause problems. Also, VAR is not suitable for very long lag structures or high-dimensional systems (many series).

### Prediction intervals widen over time
This is a feature, not a bug. Uncertainty grows with forecast horizon. Report prediction intervals alongside point forecasts. If stakeholders only want a single number, explain that the further out you forecast, the less certain that number is.

### Seasonality can change over time
Monthly claims might show stronger seasonality in some years than others (e.g., due to changing business mix). State space models and Prophet handle this better than SARIMA, which assumes fixed seasonal patterns.

### Watch out for calendar effects
Business days per month, leap years, and weekday distributions within a month can create spurious patterns. Adjust for these when working with monthly or weekly aggregated data.

### External regressors require future values
If your model includes temperature as a regressor, you need temperature forecasts (or scenarios) for the forecast period. This introduces additional uncertainty that is often ignored.

## Resources

- [statsmodels time series documentation](https://www.statsmodels.org/stable/tsa.html) -- SARIMAX, VAR, and state space models
- [Prophet documentation](https://facebook.github.io/prophet/) -- Official documentation with excellent tutorials
- [pmdarima documentation](https://alkaline-ml.com/pmdarima/) -- Auto ARIMA for Python
- [Hyndman & Athanasopoulos - Forecasting: Principles and Practice (3rd ed)](https://otexts.com/fpp3/) -- Free online textbook, the gold standard for applied time series
- [Rob Hyndman's blog](https://robjhyndman.com/hyndsight/) -- Practical time series wisdom
- [Durbin & Koopman (2012) - Time Series Analysis by State Space Methods](https://global.oup.com/academic/product/time-series-analysis-by-state-space-methods-9780199641178) -- Definitive reference for state space models
- [Lutkepohl (2005) - New Introduction to Multiple Time Series Analysis](https://link.springer.com/book/10.1007/978-3-540-27752-1) -- The VAR bible
- [scikit-learn TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) -- Time series cross-validation
