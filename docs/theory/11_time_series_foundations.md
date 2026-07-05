# Time Series Foundations

## What makes time series different

A time series is an ordered sequence of observations. The order matters: swapping rows changes the data. Standard ML assumptions (IID observations, random splits, symmetric errors) break.

Key concepts:

- **Trend.** Long-run direction.
- **Seasonality.** Repeating patterns (daily, weekly, yearly).
- **Cyclicality.** Non-fixed-period oscillations (business cycle).
- **Noise.** Unpredictable variation.
- **Autocorrelation.** Today's value depends on previous values.
- **Stationarity.** Statistical properties constant over time (mean, variance, autocorrelation).

## Decomposition

Additive:

$$
Y_t = T_t + S_t + R_t
$$

Multiplicative (for data with variance proportional to level):

$$
Y_t = T_t \times S_t \times R_t
$$

Algorithmic options: STL (Seasonal-Trend decomposition with Loess), X-13, classical moving average.

Useful for exploratory analysis and for detrending or deseasonalizing before modeling.

## Stationarity

Strictly stationary: joint distribution invariant under time shifts. Weakly stationary: mean, variance, and autocovariance are constant over time.

Many time-series models assume (weak) stationarity. Check with:

- Rolling mean and variance plots.
- Augmented Dickey-Fuller (ADF) test: null hypothesis is non-stationarity.
- KPSS test: null is stationarity. Use both; disagreements are informative.

Fix non-stationarity via differencing:

$$
\Delta Y_t = Y_t - Y_{t-1}
$$

Seasonal differencing:

$$
\Delta_s Y_t = Y_t - Y_{t-s}
$$

## Classical models

### ARIMA(p, d, q)

- AR($p$): Y depends linearly on last $p$ values.
- I($d$): Differenced $d$ times to achieve stationarity.
- MA($q$): Residuals depend on last $q$ noise terms.

SARIMA adds seasonal components.

Fit via maximum likelihood (statsmodels.SARIMAX). Diagnose with ACF and PACF plots.

### Exponential smoothing (ETS)

Error / Trend / Seasonal decomposition with state-space form. Simple variants: Simple Exponential Smoothing, Holt's linear, Holt-Winters.

Fast, interpretable, strong baseline.

### Prophet

Bayesian additive model: trend + seasonality + holidays + error. Robust to missing data and outliers; easy to specify.

Strengths: quick to use, good defaults, handles holidays and changepoints naturally.
Weaknesses: less principled than ARIMA, can miss subtle structure; loses accuracy at scale vs ML.

## ML-based forecasting

### Feature-engineered approach

Convert time series to supervised learning: features are lagged values and time-based covariates (day of week, month, holiday flags, weather, promotions). Train a regressor (LightGBM is the standard).

This approach:

- Scales to thousands or millions of series trained jointly.
- Handles exogenous variables cleanly.
- Wins many Kaggle forecasting competitions (M5, M6).
- Requires careful temporal cross-validation.

### Deep learning

- **LSTM / GRU.** Recurrent networks. Older now, but still competitive on some datasets.
- **N-BEATS.** Fully connected networks with backward-forward residual blocks. Strong on M4.
- **N-HITS.** Hierarchical variant of N-BEATS.
- **Temporal Fusion Transformer (TFT).** Attention-based, handles static and time-varying covariates.
- **PatchTST.** Transformer treating time-series as patches.

Deep learning shines with large datasets and many related series; for a handful of series, classical or LightGBM usually suffices.

## Hierarchical forecasting and reconciliation

These two terms are often conflated. **Hierarchical forecasting** is the *problem*: you have multi-level structure (SKU within store within region) and want forecasts that add up consistently across levels. **Forecast reconciliation** is one *family of solution techniques* that takes independent forecasts at every level and post-processes them to satisfy the hierarchy constraints.

### Approaches

- **Bottom-up (BU).** Forecast at the finest level, sum up. Accurate at leaves, noisy at the top.
- **Top-down (TD).** Forecast at the top, disaggregate by historical proportions. Smooth top, but leaves may miss local dynamics.
- **Middle-out.** Forecast at an intermediate level.
- **OLS reconciliation.** Forecast every level independently, project onto the coherent subspace by ordinary least squares.
- **MinT reconciliation.** Wickramasuriya, Athanasopoulos, Hyndman (2019). Choose the reconciliation that minimizes the forecast-error variance under unbiasedness. Current best practice; available in `hts` (R) and `hierarchicalforecast` (Nixtla, Python).

## Temporal cross-validation

Random k-fold is wrong for time series. Use:

- **Expanding window.** Training set grows; validation slides forward.
- **Sliding window.** Both training and validation windows are fixed size.
- **Blocked CV.** Non-overlapping blocks; useful for seasonal CV.

Evaluate on multiple horizons (1-step, 7-step, 28-step); don't just report one number.

## Evaluation metrics

- **MAE.** Scale-dependent, robust to outliers.
- **RMSE.** Scale-dependent, penalizes large errors more.
- **MAPE.** Scale-free percent error; fails for zero actuals.
- **sMAPE.** Symmetric MAPE; still has issues near zero.
- **WAPE.** Weighted (by total volume) MAPE; common in retail.
- **MASE.** Scaled relative to a naive baseline.
- **Quantile loss / pinball loss.** For probabilistic forecasts.
- **CRPS (Continuous Ranked Probability Score).** For full distributional forecasts.

Pick the metric before looking at results; back-fit metrics to data is a classic mistake.

## Probabilistic forecasting

A point forecast alone is rarely enough. Decisions depend on uncertainty:

- **Quantile regression.** Model trained to predict specific quantiles (e.g., 0.1, 0.5, 0.9).
- **Conformal prediction.** Distribution-free prediction intervals with coverage guarantees. Wrap any point forecaster.
- **Parametric distributions.** Model mean and variance (Gaussian), or use negative binomial for counts.
- **Ensemble spread.** Variability across a model ensemble as uncertainty proxy.

## Special cases

### Intermittent demand

Many zeros punctuated by occasional demand (spare parts). Standard models struggle.

- **Croston's method.** Separately forecast demand size and inter-demand intervals.
- **TSB, SBA.** Variants of Croston with bias corrections.
- **Hurdle models.** Zero-inflated models.

### Outliers and structural breaks

Identify (sigma-clipping, isolation forest, Prophet's automatic changepoints). Decide: replace with imputed value, add indicator feature, or model them explicitly.

### Missing timestamps

A missing day is not the same as a zero. Reindex to a complete time grid, then decide the imputation strategy (carry-forward, zero, interpolate, or model the missingness).

## Forecasting workflow

1. **Understand the data.** Plot the series. Look at decomposition. Check for stationarity, seasonality, outliers.
2. **Build a baseline.** Naive (last value carried forward) or seasonal naive (same value as last season). Everything later must beat this.
3. **Try classical models.** ETS, ARIMA, Prophet. Usually strong and fast.
4. **Try ML with lag features.** LightGBM is the common jump in accuracy.
5. **Deep learning if warranted.** Many series, complex covariates, long horizons.
6. **Reconcile hierarchy.** If applicable.
7. **Add intervals.** Quantile regression or conformal prediction.
8. **Evaluate on true holdout.** Walk-forward CV with realistic horizons.
9. **Monitor in production.** Bias, calibration, interval coverage.

## Common mistakes

1. Random k-fold on time-series data. Leakage inflates metrics.
2. Using future information as features. "Promotion schedule for next month" is fine if it is known; "actual sales next week" is not.
3. Fitting seasonality with inadequate history. A weekly seasonality needs multiple full weeks. Annual seasonality needs multiple years.
4. Tuning only on one horizon. Models optimized for 1-step ahead often degrade at 28-step.
5. Ignoring intervals. Point forecasts without uncertainty mislead operations.
6. Overfitting to a single holdout. Walk-forward validation is more trustworthy.

## Linking to the portfolio

- `timeseries-retail-forecast`: applies this entire workflow on M5 Walmart data.
- `technique/temporal-fusion-transformer`: deep dive on TFT.
- Future: predictive maintenance technique could extend.

## GARCH and volatility modeling

Many financial and actuarial time series exhibit **volatility clustering**: large absolute returns are followed by large absolute returns, regardless of sign. Linear models (ARMA) capture conditional mean dynamics but assume constant variance; they miss this structure entirely.

### ARCH motivation

Engle (1982). Model the conditional variance as autoregressive in past squared residuals:

$$
\sigma_t^2 = \alpha_0 + \sum_{i=1}^q \alpha_i \epsilon_{t-i}^2.
$$

### GARCH(p, q)

Bollerslev (1986). Add lagged conditional variance terms:

$$
\sigma_t^2 = \alpha_0 + \sum_{i=1}^q \alpha_i \epsilon_{t-i}^2 + \sum_{j=1}^p \beta_j \sigma_{t-j}^2.
$$

GARCH(1,1) suffices for most financial return series. Parameters typically satisfy $\alpha_1 + \beta_1 < 1$ (stationarity) with $\beta_1$ near 0.9 and $\alpha_1$ near 0.05 to 0.1.

### Asymmetric variants

- **GJR-GARCH** (Glosten, Jagannathan, Runkle 1993). Adds a leverage term so negative shocks raise volatility more than positive shocks of equal size:

$$
\sigma_t^2 = \alpha_0 + \alpha_1 \epsilon_{t-1}^2 + \gamma I_{\{\epsilon_{t-1} < 0\}} \epsilon_{t-1}^2 + \beta_1 \sigma_{t-1}^2.
$$

- **EGARCH** (Nelson 1991). Models $\log \sigma_t^2$ to enforce positivity without parameter constraints, and captures asymmetry naturally.

### Applications

- Market risk: VaR and expected-shortfall calculation requires conditional volatility forecasts.
- Actuarial capital modeling: economic scenario generators feed GARCH-driven equity-return paths into solvency simulations.
- Pairs trading and option pricing.

### Code

```python
from arch import arch_model
am = arch_model(returns, vol="GARCH", p=1, q=1)
res = am.fit(disp="off")
forecast = res.forecast(horizon=10)
```

## State-space ETS and the Kalman filter

### State-space ETS

The Hyndman-Koehler-Snyder-Grose (2008) framework rewrites the ETS family in a unified state-space form:

$$
y_t = w(\boldsymbol{x}_{t-1}) + r(\boldsymbol{x}_{t-1}) \epsilon_t, \qquad \boldsymbol{x}_t = f(\boldsymbol{x}_{t-1}) + g(\boldsymbol{x}_{t-1}) \epsilon_t,
$$

where $\boldsymbol{x}_t$ is the unobserved state (level, trend, seasonal components). Each ETS variant ETS(error, trend, seasonal) corresponds to specific choices of $w, r, f, g$. The state-space form yields exact likelihoods, prediction intervals, and a clean path to the Kalman filter for the linear Gaussian cases. `statsforecast` uses this representation.

### Kalman filter

For the linear Gaussian state-space model

$$
\boldsymbol{x}_t = F \boldsymbol{x}_{t-1} + \boldsymbol{w}_t, \qquad y_t = H \boldsymbol{x}_t + v_t,
$$

with $\boldsymbol{w}_t \sim \mathcal{N}(0, Q)$ and $v_t \sim \mathcal{N}(0, R)$, the Kalman filter (Kalman 1960) computes optimal one-step-ahead state estimates recursively:

- **Prediction.** $\hat{\boldsymbol{x}}_{t \mid t-1} = F \hat{\boldsymbol{x}}_{t-1 \mid t-1}$, $P_{t \mid t-1} = F P_{t-1 \mid t-1} F^\top + Q$.
- **Update.** $K_t = P_{t \mid t-1} H^\top (H P_{t \mid t-1} H^\top + R)^{-1}$; $\hat{\boldsymbol{x}}_{t \mid t} = \hat{\boldsymbol{x}}_{t \mid t-1} + K_t(y_t - H \hat{\boldsymbol{x}}_{t \mid t-1})$.

The Kalman smoother runs backward to refine state estimates with future observations. Use the Kalman filter over ARIMA when you need:

- Time-varying coefficients.
- Online updating as new data arrives.
- Direct handling of missing observations.
- Multi-source observation fusion.

### Bayesian Structural Time Series (BSTS)

Scott & Varian (2014). Decompose

$$
y_t = \mu_t + \tau_t + \beta^\top x_t + \epsilon_t,
$$

with $\mu_t$ a trend component, $\tau_t$ a seasonal component, $x_t$ regression covariates, all expressed in state-space form. Inference is by Markov chain Monte Carlo. `bsts` (R) and `tfp.sts` (TensorFlow Probability) implement this.

The canonical use case is **CausalImpact**: counterfactual prediction for a treated time series, using a fitted BSTS on the pre-period as the counterfactual generator (Brodersen et al. 2015). For an actuarial reader, this is the right tool when you need the post-intervention impact of a marketing campaign or a regulatory change in a single time series.

## Foundation time-series models

By 2026 the time-series world has its own foundation-model paradigm: pretrain a large transformer on heterogeneous time series, deploy zero-shot on new series.

- **TimeGPT** (Nixtla, 2023). Closed-source API. First production-grade time-series foundation model.
- **Chronos** (Amazon, 2024). Tokenizes numeric values and trains a T5-style encoder-decoder. Open weights.
- **Moirai** (Salesforce, 2024). Patch-based encoder, multivariate. Open weights.
- **Lag-Llama** (2024). Decoder-only with lagged inputs. Open weights.

These models forecast unseen series with no fine-tuning, often beating tuned local models when the series is short. Cost: each prediction is a transformer forward pass, expensive vs ARIMA. Best fit: many short series, no time to tune per series, and zero-shot accuracy is good enough to justify the cost. For long single series with rich history, classical or feature-engineered ML still wins.

## Changepoint detection

When the data-generating process shifts at unknown time points, standard models smear the change across the boundary. Two approaches:

### Bayesian Online Changepoint Detection (BOCPD)

Adams, MacKay (2007). Maintain a posterior over the run length (time since the last changepoint). At each new observation, update the posterior; high probability on short run lengths signals a recent change. Online and probabilistic.

### PELT (Pruned Exact Linear Time)

Killick, Fearnhead, Eckley (2012). Offline algorithm: minimize the sum of segment costs plus a penalty per changepoint. With pruning, runs in linear time. The `ruptures` (Python) and `changepoint` (R) libraries implement PELT and several alternatives.

Use changepoint detection to:

- Segment a series before fitting a model per segment.
- Trigger retraining alerts in production (essay 13).
- Identify regime shifts in financial or actuarial time series.

## References

- Hyndman, Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed. (free online at otexts.com/fpp3).
- Makridakis, Spiliotis, Assimakopoulos, "The M4 Competition: 100,000 time series and 61 forecasting methods", 2020.
- Makridakis et al., "The M5 accuracy competition: Results, findings, and conclusions", 2022.
- Lim et al., "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting", 2020.
- Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting", 2020.
- Bollerslev, "Generalized Autoregressive Conditional Heteroskedasticity", Journal of Econometrics, 1986.
- Engle, "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation", Econometrica, 1982.
- McNeil, Frey, Embrechts, *Quantitative Risk Management: Concepts, Techniques and Tools*, Princeton, 2015.
- Hyndman, Koehler, Snyder, Grose, *Forecasting with Exponential Smoothing: The State Space Approach*, Springer, 2008.
- Kalman, "A New Approach to Linear Filtering and Prediction Problems", J. Basic Engineering, 1960.
- Scott, Varian, "Predicting the Present with Bayesian Structural Time Series", Int. J. Mathematical Modelling and Numerical Optimisation, 2014.
- Brodersen et al., "Inferring causal impact using Bayesian structural time-series models" (CausalImpact), Annals of Applied Statistics, 2015.
- Wickramasuriya, Athanasopoulos, Hyndman, "Optimal Forecast Reconciliation for Hierarchical and Grouped Time Series Through Trace Minimization" (MinT), JASA, 2019.
- Adams, MacKay, "Bayesian Online Changepoint Detection", 2007.
- Killick, Fearnhead, Eckley, "Optimal Detection of Changepoints With a Linear Computational Cost" (PELT), JASA, 2012.
- Ansari et al., "Chronos: Learning the Language of Time Series", 2024.
- Garza, Mergenthaler-Canseco, "TimeGPT-1", 2023.

## Exercises

1. A time series has $\alpha + \beta = 0.97$ in a GARCH(1,1) fit. What does this imply about volatility persistence?
2. You fit an ARIMA(2,1,1) to monthly sales data. The KPSS test rejects stationarity on the differenced series. What do you do?
3. Explain the difference between forecast reconciliation and hierarchical forecasting. Which methods belong to each?
4. Your Kalman filter shows a large innovation covariance $R$. What does this imply about your observation model?
5. You want to detect a structural break in weekly insurance claims after a new regulation took effect. Name two methods.
