# Leakage Checklist

A concise checklist before submitting in a Kaggle competition. See `docs/theory/04_data_leakage_patterns.md` for the broader treatment.

## Before any submission

- [ ] Features computed with explicit time cutoff (no future information).
- [ ] Preprocessing (scalers, imputers, encoders) fit inside CV folds.
- [ ] Oversampling (SMOTE) applied only to training fold.
- [ ] Feature selection done inside CV, not on full data.
- [ ] Grouped CV used if data has group structure (user ID, patient ID).
- [ ] Target encoding with smoothing and CV-based aggregation.
- [ ] No file names or paths encoding the label.
- [ ] No row IDs correlated with the target.

## Warning signs

- **Too-good-to-be-true CV.** Audit every feature.
- **CV ~ LB gap widening.** Something is leaking.
- **A single feature explains > 80 percent SHAP.** Suspect.
- **Training error near zero.** Either massive overfitting or leakage.

## Sanity test

Drop the suspect feature and retrain. If performance barely changes, the feature was redundant or leaky. If performance collapses, the feature was load-bearing; check it again for leakage.

## Cross-links

- `docs/theory/04_data_leakage_patterns.md`
- `02_cv_strategy_for_competitions.md`
