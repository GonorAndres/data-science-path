# Hierarchical Forecasting

## The hierarchy

M5 has a natural hierarchy:

```
Total (US)
|-- State (CA, TX, WI)
|   |-- Store (10 stores total)
|   |   |-- Category (Foods, Hobbies, Household)
|   |   |   |-- Department
|   |   |   |   |-- Item (3,049 items)
```

Stakeholders consume different levels: finance uses the top; replenishment uses the bottom; merchandising uses middle.

## The reconciliation problem

If you forecast each level independently, the numbers will not add up. Store-level forecasts summed up may not equal the state forecast. This is confusing at best and business-breaking at worst.

Three reconciliation strategies:

### 1. Bottom-up

Forecast at the finest level (SKU-store-day), sum up for higher levels.

Pros: self-consistent; aggregates are clean.
Cons: noise at the leaves compounds; top-level accuracy suffers.

### 2. Top-down

Forecast the aggregate, disaggregate using historical proportions.

Pros: top-level is accurate (the main forecast is done directly there).
Cons: leaves are smooth and miss local dynamics.

### 3. Middle-out

Forecast at an intermediate level, go up (sum) and down (proportion).

Pros: balance.
Cons: choice of middle level is ad hoc.

## Optimal reconciliation (MinT)

Hyndman et al. (2018) proposed a unified framework. Forecast at every level independently, then find the set of reconciled forecasts that:

- Preserves the additive constraints exactly.
- Minimizes the trace of the forecast error covariance matrix.

The solution is a linear projection of the base forecasts. The critical input is an estimate of the error covariance. Common choices:

- **OLS.** Assume errors independent, equal variance. Simple but rarely good.
- **WLS.** Weighted by inverse variance per level.
- **MinT with shrinkage.** Shrink the full covariance matrix toward a diagonal.

MinT consistently beats bottom-up and top-down on competition data. Library: `hts` in R, `statsforecast.HierarchicalForecast` in Python.

## Empirical rule of thumb for M5

- Bottom-up WAPE at bottom level: baseline.
- Top-down improves top-level WAPE by 2 to 5 pp.
- MinT improves middle-level WAPE by 1 to 3 pp over bottom-up, with top- and bottom-level unchanged or improved.

## Implementation in this project

1. Train one global LightGBM across all series (SKU-store level).
2. Generate forecasts for all series for the next 28 days.
3. Sum up to each hierarchy level to get base forecasts at each level.
4. Apply MinT reconciliation using the training-fold residuals as the covariance estimate.
5. Report WAPE per level before and after reconciliation.

## Evaluation gotcha

Evaluate reconciled forecasts at each level separately. A single aggregate WAPE hides where reconciliation helped.

## Cross-links

- `01_decomposition_and_stationarity.md`: series structure.
- `02_classical_vs_ml_forecasts.md`: the underlying model.
- `03_temporal_cv.md`: validation.
- `docs/theory/11_time_series_foundations.md`: broader context.
