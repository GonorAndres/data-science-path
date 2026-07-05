# Data Leakage: The Silent Killer of Portfolio Projects

## Definition

Data leakage is any situation where information not available at prediction time influences training or validation metrics. The result: offline metrics look great, production performance collapses, and you cannot tell why without rebuilding the pipeline.

Leakage is the single biggest cause of "my model was 0.92 offline and flat online" stories.

## The canonical test

At the moment of prediction (in production), what information is available?

- The answer defines the legal feature set.
- Anything else is leakage.
- Every feature should pass this check before entering a training pipeline.

## Leakage patterns

### 1. Target leakage from derived features

The feature is a direct or indirect function of the target that happens after the target is realized.

**Example:** predicting default from features that include "was contacted by collections". Collections only happens post-default. Including this feature gives near-perfect accuracy offline and zero online.

**Detection:** audit features for business logic. Ask: is this recorded before or after the event we are predicting?

**Fix:** exclude the feature. Regenerate your feature set with strict point-in-time cutoffs.

### 2. Train-test contamination via preprocessing

Fitting scalers, imputers, or encoders on the full dataset (before splitting) leaks test-distribution statistics into train.

**Example:** `StandardScaler().fit(X_full)`. The mean and std are computed over training plus test. Your test set's scaling is partially informed by its own data.

**Fix:** use `sklearn.pipeline.Pipeline` with the scaler as a step, and call `fit` only inside cross-validation folds.

### 3. Time leakage in features

The feature was generated using data from after the prediction timestamp.

**Example:** "lifetime purchase count as of today" used to predict a churn that happened six months ago. The feature incorporates purchases made after the churn window.

**Fix:** point-in-time feature stores. For each training row, compute features using only data available before the row's timestamp.

### 4. Time leakage in splits

Random k-fold on time-ordered data.

**Example:** forecasting next week's sales with 5-fold k-fold. Rows from next week end up in the training set of some folds.

**Fix:** `TimeSeriesSplit`. Never shuffle time.

### 5. Group leakage

Same entity appears in both train and validation.

**Example:** predicting loan default with rows at the loan level, same customer has five loans. Customer features implicit in loan outcomes leak through the shared customer.

**Fix:** `GroupKFold` on customer ID.

### 6. Duplicate leakage

Near-duplicates (same row with minor noise) in both train and test.

**Example:** image data augmented offline, then split randomly. Augmented variants of the same image land in both splits.

**Fix:** deduplicate before splitting. Augment only inside the training fold, not before.

### 7. Leakage via resampling

Oversampling (SMOTE) before splitting generates synthetic rows based on all data including test.

**Fix:** oversample inside the training fold only.

### 8. ID or proxy leakage

An identifier encodes the target.

**Example:** customer ID has a sequential structure where high IDs are recent signups who are more likely to churn. The ID itself becomes a proxy target.

**Fix:** drop or hash IDs. Audit for suspicious signal in identifier columns.

### 9. Information-rich metadata

File names, creation timestamps, folder structure encoding the label.

**Example:** MRI scans stored in `/positive/` and `/negative/` folders. A model learns the file path before the image.

**Fix:** randomize file paths. Strip metadata from input features.

### 10. Leakage in NLP / LLM datasets

The train and test come from overlapping web corpora. For evaluation benchmarks, pretrained LLMs may have seen the test questions during pretraining.

**Fix:** use contamination-aware benchmarks or held-out private test sets. For your own NLP task, ensure no exact-duplicate text across train and test.

### 11. Validation set used for feature selection

You ran correlation-based feature selection on the full dataset, picked the top 50 features, then trained on those. The selection step saw test data.

**Fix:** do feature selection only within cross-validation training folds.

### 12. Time leakage via joined tables

Feature engineering with a join that pulls in an "as-of" snapshot of a dimension table. If the dimension was updated after the target event, you have leakage.

**Example:** joining customer status table (updated monthly) to transaction table. Transactions from January joined to February's customer status is leakage.

**Fix:** point-in-time joins. Dimension tables need effective-dated rows so you can snapshot to any historical moment.

## Detection heuristics

Beyond the prediction-time check, these warning signs often indicate leakage:

- **Too-good-to-be-true offline metric.** Your baseline should start reasonable, not at 0.99.
- **A single feature explains almost everything.** Check the top SHAP feature. If it is suspiciously predictive, audit.
- **Training error is near zero and validation is too.** On difficult real-world tasks, both should have headroom.
- **Shuffling the target breaks the model** (this is expected). **Shuffling a suspicious feature barely changes performance** (this is also expected). **Shuffling the label but keeping a leaky feature still gives high accuracy** (this is the leakage flag).

## Prevention workflow

1. Write a feature generation function that accepts a timestamp and produces features as of that timestamp. No global state.
2. Split data strictly (by time, by group, by entity) before any feature engineering that could leak.
3. Fit preprocessors inside CV folds via a pipeline.
4. Deduplicate rows before splitting; do not rely on exact-match dedup (near-duplicates matter).
5. Use a feature store with enforced point-in-time semantics for production ML.
6. Document every feature's source and availability in a data dictionary.

## Debugging a leakage suspicion

If you suspect leakage:

1. Train a model with the suspect feature removed. If performance drops massively, that feature was load-bearing.
2. Check correlation of the suspect feature with the target after shuffling everything else. If still very high, suspect.
3. Audit the business definition of the feature: when is it computed, from what data, with what cutoff.
4. Rebuild the feature with explicit timestamp gating and retrain. If metrics drop to realistic levels, the original feature was leaky.

## Consequences of undetected leakage

- Offline metrics mislead stakeholders, set expectations the team cannot meet.
- The model rolls out, production A/B is flat or negative.
- Trust in the team erodes; the next model launch gets more scrutiny.
- In regulated industries (finance, health), leakage can trigger regulatory findings.

## Linking to the portfolio

Every project in `projects/` has a section in its `theory/` folder specifically naming the leakage risks and the controls applied:

- `nlp-reviews-multilingual`: near-duplicate dedup before splitting (same review re-posted).
- `timeseries-retail-forecast`: point-in-time features, TimeSeriesSplit.
- `recsys-movielens`: leave-last-out per user to avoid user-level leakage and temporal leakage simultaneously.
- `causal-uplift-marketing`: propensity scores computed on features available before treatment assignment.
- `mlops-model-service`: production logging to verify feature distributions match training.

## References

- Kaufman et al., "Leakage in Data Mining: Formulation, Detection, and Avoidance", 2012.
- Nisbet, Miner, Yale, *Handbook of Statistical Analysis and Data Mining Applications*, 2018, chapter on leakage.
- Kapoor, Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science", 2022.
- Feast documentation on point-in-time correctness, feast.dev.

## Exercises

1. A churn model has AUC $0.98$ on the test set. Name three concrete leakage patterns you would investigate first.
2. What is preprocessing leakage and how do you fix it using a sklearn `Pipeline`?
3. A model predicting credit default uses zip code as a feature. Is this leakage? Is it an ethical concern? Are these the same question?
4. Explain temporal leakage in a feature join and give a SQL example of how a point-in-time join prevents it.
5. You notice that shuffling the target label barely changes the model AUC. What does this tell you?
