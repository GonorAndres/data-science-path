# CV Strategy for Competitions

## The principle

A robust CV is the most valuable asset in a Kaggle competition. It lets you decide whether a change is real improvement or noise. Public LB is noisy and limited.

## Matching CV to data structure

- **IID data.** Stratified k-fold. Safe default.
- **Time series.** Walk-forward or blocked time splits. Never random.
- **Grouped data.** GroupKFold on the group key (user ID, image ID, patient ID).
- **Imbalanced classes.** StratifiedKFold ensures positives in every fold.

## Matching CV to evaluation metric

If evaluation is rank-based (AUC, NDCG), rank-preserving transformations of the CV metric matter less. If metric is threshold-based (F1 at a fixed threshold), the threshold choice interacts with CV.

## CV-LB gap

Track CV score and public LB side by side. A widening gap (CV improves, LB flat or drops) usually means:

- You are overfitting to the CV structure.
- You are exploiting something in the train distribution absent from the test.

If the gap is consistent, private LB will follow CV. If it is inconsistent, something is leaking.

## Number of folds

- 5 is standard.
- 10 for smaller datasets (< 100k rows) or more stable estimates.
- Hold out an additional 10 to 20 percent as a "secret test" if you distrust CV alone.

## Seed sensitivity

Run CV with 2-3 different seeds; if fold standard deviation is high, metrics are noisy. Increase folds or collect more data (by augmentation or additional validation sets).

## Cross-links

- `docs/theory/03_cross_validation_strategies.md`
- `01_competition_strategy.md`
