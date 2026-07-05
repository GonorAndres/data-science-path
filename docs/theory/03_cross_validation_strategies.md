# Cross-Validation Strategies That Survive Contact with Reality

## Why cross-validate

A single train-test split gives one noisy estimate of generalization. Cross-validation averages multiple estimates and lowers the variance of that measurement. It also lets you use every observation for both training and evaluation (at the cost of fitting $k$ models).

The danger: if the splitting procedure does not match the structure of the data, CV gives a confident wrong answer.

## k-fold cross-validation (the default)

Split the training data into $k$ equal folds (typically $k = 5$ or $10$). For each fold $i$:

- Train on the other $k-1$ folds.
- Evaluate on fold $i$.
- Record the metric.

Average the $k$ metrics. The variance of the average is an indicator of estimation stability.

## Stratified k-fold

For classification, stratify by the target so each fold has the same class proportions. Essential for imbalanced data: without stratification, a fold might contain zero positive examples.

## Grouped k-fold

When observations are not independent (same user appearing in multiple rows; same patient in multiple visits; same image augmented many ways), random splits leak across train and validation. Use `GroupKFold` so all rows from the same group go to the same fold.

Without this, validation metrics are inflated because the model effectively sees the same entity at both train and evaluation.

## Stratified Group k-fold

Combines both: groups go together, and class proportions are balanced. Scikit-learn's `StratifiedGroupKFold`.

## Time series cross-validation

Never shuffle time. Two common schemes:

### Expanding window (walk-forward)

```
Fold 1: Train [t0, t1], Validate [t1, t2]
Fold 2: Train [t0, t2], Validate [t2, t3]
Fold 3: Train [t0, t3], Validate [t3, t4]
```

The training set grows; the validation window slides forward.

### Sliding window

```
Fold 1: Train [t0, t1], Validate [t1, t2]
Fold 2: Train [t1, t2], Validate [t2, t3]
Fold 3: Train [t2, t3], Validate [t3, t4]
```

Both training and validation windows have fixed size. Useful when the relationship between features and target changes slowly over time and you do not want ancient history dominating the training set.

Scikit-learn's `TimeSeriesSplit` implements expanding window.

## Nested cross-validation

Tuning hyperparameters on the same CV used for reporting inflates the reported metric: you pick the hyperparameters that happened to win on those specific folds.

Nested CV has two loops:

- **Outer loop** ($k_{\text{out}}$ folds): splits data for final evaluation.
- **Inner loop** ($k_{\text{in}}$ folds): within each outer training fold, tune hyperparameters.

Expensive: $k_{\text{out}} \times k_{\text{in}}$ full model fits. Usually only needed for academic papers or small-data, high-stakes settings. In practice, most teams use a single CV for tuning and a separate held-out test set for final reporting.

## Leave-one-out CV (LOO-CV)

Extreme case: $k = n$. Each observation is its own validation fold. Low bias in the error estimate but very high variance and computationally painful. Useful only when $n$ is small (dozens of observations).

## Random sub-sampling CV

Shuffle, pick a fraction as validation, train on the rest, repeat. Less common than k-fold because observations can appear in multiple validation folds (or none).

## Choosing k

- $k = 5$: standard balance of bias and variance of the error estimate.
- $k = 10$: lower bias, slightly higher variance, more compute.
- $k = n$ (LOO): minimal bias, maximal variance, use only for tiny datasets.

## Confidence intervals on CV metrics

The $k$ fold metrics are not independent (training sets across folds share most of their data), so the naive $\sqrt{k}$ standard error formula understates uncertainty. For reporting, use:

- Point estimate: mean of fold metrics.
- Standard error: fold standard deviation divided by $\sqrt{k}$ (informal, biased low).
- Better: bootstrap on top of CV (resample folds, recompute the mean).

### The Bengio-Grandvalet result

Bengio and Grandvalet (2004) proved that **no general unbiased estimator exists for the variance of $k$-fold cross-validation**. The dependence between folds depends on the learning algorithm and the data distribution; you cannot construct a closed-form variance estimate that works without further assumptions.

What this implies for practice:

- Do not trust a single CV run's "standard error" as a real confidence interval. Treat it as a stability indicator only.
- Run **repeated $k$-fold CV** with different random seeds (5x5 CV is a common choice: 5 repeats of 5-fold). Report the spread across seeds rather than the within-run fold SE.
- For model comparison, use a paired test across the same fold splits (e.g., paired Wilcoxon across repeats) rather than comparing two means with independent SEs.
- When the comparison really matters (paper, regulatory filing), report the full distribution of metric values, not a point estimate.

## Purged and embargoed CV

The default time-series splits (expanding or sliding window) assume features and labels at the train-test boundary are independent. In financial and insurance time-series this assumption frequently fails: a label assigned to a row at time $t$ can depend on information that overlaps in time with rows in the training fold (overlapping holding periods, aggregated windows, claims developing across fold boundaries).

Lopez de Prado's **purged + embargoed CV** addresses both:

- **Purging.** Remove from the training fold any rows whose feature window or label window overlaps in time with rows in the test fold.
- **Embargo.** Add a gap of $h$ time steps after the test fold during which training rows are also removed. This prevents post-test-fold rows from leaking through serial correlation in features.

Pseudocode:

```
function purged_embargo_split(times, labels, test_indices, embargo_h):
    # times[i] = (start_i, end_i): the time interval the row's information covers
    test_times = [times[i] for i in test_indices]
    test_start = min(t.start for t in test_times)
    test_end   = max(t.end   for t in test_times)

    train_indices = []
    for j, t_j in enumerate(times):
        if j in test_indices: continue
        # Purge: skip rows whose interval overlaps the test interval
        if t_j.end >= test_start and t_j.start <= test_end:
            continue
        # Embargo: skip rows starting within h steps after test_end
        if 0 <= (t_j.start - test_end) < embargo_h:
            continue
        train_indices.append(j)
    return train_indices, test_indices
```

Use this whenever the row-level label window is non-trivially long relative to the fold size. In actuarial and reserving settings, claim-level features that develop over months and labels at ultimate make purging mandatory.

## Rolling-origin metric averaging

In rolling-origin time-series evaluation each "origin" is a forecast made at a specific cutoff date, producing a vector of predictions for horizons $h = 1, 2, \ldots, H$. There are two distinct ways to average the resulting metric grid, and they answer different questions.

- **Average across origins, per horizon.** For each horizon $h$, average the metric over all origins. This gives a fair estimate of expected error at horizon $h$. Report as a curve over $h$. Use this when the downstream consumer cares about specific horizons (e.g., a 4-week-ahead inventory plan).
- **Average across horizons, per origin.** For each origin, average the metric across horizons; then average across origins. This gives a single scalar but mixes horizon difficulty with origin difficulty. Two methods that perform very differently at $h = 1$ vs $h = 12$ can look identical under this scheme.

For a fair comparison between models, prefer the per-horizon averaging and report the full curve. Report the per-origin scalar only as a supplementary summary, and only after confirming the methods do not have crossing horizon profiles. Conflating the two is the most common error in published time-series benchmarks.

## Handling class imbalance in CV

If positive class is rare (under 5 percent), ensure every fold has enough positives for meaningful validation. Stratified k-fold helps; if still insufficient, use a different metric (PR-AUC, F-beta with $\beta$ tuned to the business).

## When random splits are just wrong

| Data structure | Wrong | Right |
|----------------|-------|-------|
| Same user has multiple rows | KFold | GroupKFold |
| Rows are ordered in time | KFold | TimeSeriesSplit |
| Classes are imbalanced | KFold | StratifiedKFold |
| Same image augmented multiple times | KFold | GroupKFold on image_id |
| Multiple sensors per patient | KFold | GroupKFold on patient_id |
| Strong seasonality | TimeSeriesSplit on random chunks | TimeSeriesSplit aligned to full seasonal cycles |

## Reporting CV properly

- State the number of folds and the split type.
- Report mean and standard deviation across folds.
- If tuning: state whether tuning was on the same folds or nested.
- If time-series: show a diagram of train/validate windows.

Example:

> "We used 5-fold GroupKFold on `store_id`, tuning LightGBM hyperparameters on an inner 3-fold split. Reported WAPE mean 9.2 percent with standard deviation 0.6 across outer folds."

This is more informative than "WAPE was 9.2 percent on CV" and tells a reviewer what confidence to assign.

## Common CV bugs

1. Feature engineering before the split. Target encoding, mean imputation, scaling fitted on the full dataset leak test information into train.

   **Fix:** wrap preprocessing into a `sklearn.pipeline.Pipeline` or explicitly fit on each training fold and apply to the validation fold.

2. Oversampling (SMOTE) before the split. Oversampled synthetic rows of test examples leak information.

   **Fix:** oversample inside each training fold.

3. Data leakage via time. Features that contain future information (e.g., total user lifetime purchases) leak into any time-based prediction.

   **Fix:** compute features using only data from before the prediction timestamp.

4. Tuning on the test set. The test set becomes a validation set the moment you look at it during model selection.

   **Fix:** split off a test set, do not touch it until final reporting.

## CV in practice for the portfolio

- `nlp-reviews-multilingual`: Stratified k-fold on labels; no group leakage because each review is independent.
- `timeseries-retail-forecast`: TimeSeriesSplit on weekly increments, plus hierarchy-aware averaging of metrics.
- `recsys-movielens`: Leave-last-out per user (for each user, most recent interaction is the test, rest is train). Not a traditional CV but the canonical recommender evaluation scheme.
- `cv-defect-detection`: Stratified k-fold on defect class, with per-image augmentations applied only inside training folds.
- `causal-uplift-marketing`: Use CV-folded uplift estimates; report Qini curve with bootstrap confidence intervals.

Document the CV strategy in each project's `theory/02_implementation_notes.md` or equivalent.

## References

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Chapter 7.10.
- Arlot, Celisse, "A survey of cross-validation procedures for model selection", 2010.
- Hyndman, Athanasopoulos, *Forecasting: Principles and Practice*, 3rd ed., Chapter 5 (time-series CV).
- Bergmeir, Hyndman, Koo, "A note on the validity of cross-validation for evaluating autoregressive time series prediction", 2018.
- Bengio, Grandvalet, "No Unbiased Estimator of the Variance of K-Fold Cross-Validation", JMLR 2004.
- Lopez de Prado, *Advances in Financial Machine Learning*, Wiley 2018 (Chapter 7 on purged k-fold).

## Exercises

1. You have 5 years of monthly insurance claims data and want to estimate model generalization error. Which CV scheme do you use and why?
2. What does Bengio and Grandvalet (2004) say about the variance of $k$-fold CV error estimates, and what is the practical implication?
3. For a model trained on customer transactions, explain why random 5-fold CV would give optimistic estimates of real-world performance.
4. Define purging and embargoing in the context of financial time-series CV, and write pseudocode for the fold construction.
5. When does LOOCV have high variance despite having low bias?
