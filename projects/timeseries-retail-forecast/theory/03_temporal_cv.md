# Temporal Cross-Validation

## Why random k-fold is wrong here

A random split would put rows from June 2020 in training and June 2019 in validation, letting the model "peek at the future." The model appears great; production fails. Always split by time.

## The expanding-window pattern

Walk-forward:

```
Fold 1: train on Jan-Jun 2020, validate on Jul 2020
Fold 2: train on Jan-Jul 2020, validate on Aug 2020
Fold 3: train on Jan-Aug 2020, validate on Sep 2020
...
```

Each fold trains on all history up to the fold boundary and validates on the next period.

## The sliding-window pattern

Alternative when distant history adds noise more than signal:

```
Fold 1: train on Jan-Jun 2020, validate on Jul 2020
Fold 2: train on Feb-Jul 2020, validate on Aug 2020
Fold 3: train on Mar-Aug 2020, validate on Sep 2020
...
```

Fixed-size training window. Useful when the underlying distribution drifts.

## M5 specifics

The M5 competition used a canonical split: train through May 2016, predict next 28 days. For CV, we create multiple such splits further back in history:

```
Validation 1: train through Apr 2016, predict May 1-28
Validation 2: train through Mar 2016, predict Apr 1-28
Validation 3: train through Feb 2016, predict Mar 1-28
...
```

Five or six such folds give a robust mean and std of WAPE.

## Horizon-specific metrics

One horizon is not enough. A model strong at 1-step can be weak at 28-step. Report:

- WAPE at horizon 1 day.
- WAPE at horizon 7 days.
- WAPE at horizon 14 days.
- WAPE at horizon 28 days.

Average across horizons for a single number if required.

## The "last-day-only" trap

It is tempting to train on t-1 features and predict t. But the features at t are not available at prediction time (you are predicting the next 28 days from today). Features must be known at the moment of prediction.

Solution: recursive prediction. Start with day 1, predict. Use that prediction (or the known actual if available) to update lag features for day 2. And so on.

Or: direct prediction. Train separate models per horizon (28 models). Expensive but avoids error compounding.

Or: the hybrid. One model with horizon as a feature. Efficient and performs well.

## Validation metric

WAPE (weighted absolute percentage error):

$$
\text{WAPE} = \frac{\sum_t |y_t - \hat{y}_t|}{\sum_t |y_t|}
$$

Scale-free, robust to zero actuals, weighted by volume. Industry standard.

## Cross-links

- `02_classical_vs_ml_forecasts.md`: model choices.
- `04_hierarchical_forecasting.md`: reconciliation of forecasts.
- `docs/theory/03_cross_validation_strategies.md`: broader CV theory.
