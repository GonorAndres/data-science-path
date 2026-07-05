# Classical vs ML Forecasts

## The spectrum

| Method | Strengths | Weaknesses | Good for |
|--------|-----------|------------|----------|
| Seasonal naive | Cheap, hard to beat on short horizons | No covariates | Baseline only |
| Exponential smoothing (ETS) | Interpretable, per-series | No cross-series learning | Small, stable series |
| ARIMA / SARIMA | Principled, stationarity diagnostics | Covariate limited, per-series | Clean academic series |
| Prophet | Handles holidays, missing data, trend changes | Not always most accurate | Many series, fast iteration |
| LightGBM global | Cross-series learning, covariates | Needs feature engineering | Large-scale retail, logistics |
| N-BEATS / TFT / NHiTS | Strong on many datasets | Requires scale, complex tuning | Enterprise forecasting teams |
| LLM zero-shot forecasting | No training | Expensive, experimental | Research, cold start |

## Why LightGBM for M5

M5 has 42,840 series. Classical models need 42,840 fits; LightGBM fits one global model on all series treated as rows with series-id as a feature. Benefits:

- **Shared signal.** If two SKUs correlate, the model learns it. Per-series ARIMA cannot.
- **Feature ingestion.** Price, promotion, calendar features go in as columns.
- **Faster training and re-training.** One LightGBM model in 1 hour beats 42,840 ARIMAs in a day.
- **Exogenous robustness.** Modeling promotions and holidays is natural in a supervised framework.
- **Competition pedigree.** M5 winners used LightGBM.

## Feature design

Features for the global LightGBM:

### Temporal

- Day of week, month, quarter.
- Day of month, week of year.
- Is weekend, is holiday.
- SNAP calendar flags (for M5 specifically).

### Lag

- Lag 7, 14, 28 days.
- Lag 365 (annual).

### Rolling

- Mean, std, max, min over past 7, 28, 56 days.
- Rolling max over past 365 days.

### Price

- Current price.
- Price change over past week (momentum).
- Price percentile vs own series history.

### Series-level

- Category, department, store, state (encoded).
- Item popularity rank.

### Promotional

- Event type, days-to-event, days-since-event.

## Objective function

Options:

- **MSE.** Default. Predicts conditional mean.
- **Poisson or Tweedie.** For count or semi-count data with right-skewed distribution.
- **Quantile.** For probabilistic forecasts; one model per quantile.

For M5, Tweedie with power 1.1 or 1.2 works well because sales are non-negative, heavy-tailed, and include zeros.

## Hyperparameters that matter

- `num_leaves`, `max_depth`, `min_child_samples`.
- `learning_rate`, `num_iterations`.
- `feature_fraction`, `bagging_fraction`.
- `reg_alpha`, `reg_lambda`.
- Objective-specific (`tweedie_variance_power`).

Tune with Optuna on a single validation fold to keep cost reasonable.

## Per-series vs global: when to use which

- **Per-series** when: few series (less than 100), different structural behaviors, business requires per-series explanations.
- **Global** when: many series (thousands+), covariates, similar behaviors expected.
- **Hybrid** is also possible: global for most series, per-series for top-K high-volume items.

## Cross-links

- `03_temporal_cv.md`: validation.
- `04_hierarchical_forecasting.md`: reconciliation after the base model.
- `docs/theory/11_time_series_foundations.md`: broader theory.
- `docs/theory/05_feature_engineering_cookbook.md`: feature patterns.
