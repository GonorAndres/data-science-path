# Ensembling Basics

## Why ensembles work

Different models make different errors. Averaging or combining their predictions cancels independent errors and preserves shared signal.

## Methods

### Simple averaging

Arithmetic mean of model predictions. Works surprisingly well if models are comparable in strength.

### Weighted averaging

Weight each model proportional to its CV score (or inverse error). Often improves 0.1 to 0.5 percent over simple averaging.

### Rank averaging

Instead of averaging probabilities, average ranks. Useful when models' outputs are on different scales.

### Stacking

Train a second-level meta-model on the out-of-fold predictions of base models. More powerful; more prone to overfitting.

- Base layer: diverse models (LightGBM, XGBoost, neural nets).
- Meta layer: simple (linear regression, logistic regression).
- Use proper CV at both layers.

### Blending

Train base models; make predictions on a held-out set; train a meta-model on those. Simpler than stacking; less data-efficient.

## Tips

- **Diversity matters more than raw accuracy.** Two mediocre diverse models beat one strong model.
- **Check correlation of model predictions.** Highly correlated models do not help in ensembles.
- **Keep track of OOF predictions** from every model; you can always ensemble later.
- **Overfitting at the meta layer** is the biggest risk. Simple meta-models are safer.

## When to start

Usually at the end of the competition, after individual models are strong. Starting too early wastes time on fragile ensembles.

## Cross-links

- `02_cv_strategy_for_competitions.md`
- `01_competition_strategy.md`
