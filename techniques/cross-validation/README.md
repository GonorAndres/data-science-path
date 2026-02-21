# Cross-Validation

## What it is

Cross-validation is a resampling technique used to estimate how well a model will perform on unseen data. Instead of relying on a single train/test split -- which can be lucky or unlucky depending on which observations end up where -- cross-validation systematically rotates through multiple splits, training and evaluating the model on different portions of the data each time. The result is a more reliable and less noisy estimate of out-of-sample performance. It is the gold standard for model selection, hyperparameter tuning, and honest performance estimation.

## Why it matters

For actuarial and data science practitioners, cross-validation solves a fundamental problem: you need to know how your model will perform on future data, but future data does not exist yet. A single train/test split gives you one noisy estimate. Cross-validation gives you multiple estimates and their variability, which tells you both "how good is this model?" and "how certain am I about that?"

In actuarial work, where models must be validated before regulatory filing and where small datasets (e.g., commercial lines with few large accounts) are common, cross-validation provides the statistical rigor needed to justify model choices. It also guards against one of the most dangerous problems in predictive modeling: data leakage -- where information from the test set inadvertently contaminates the training process, producing overly optimistic performance estimates.

## When to use it

- **When selecting between competing models** -- compare cross-validated scores to choose the best model.
- **When tuning hyperparameters** -- use cross-validation within a grid search or random search.
- **When your dataset is too small for a dedicated holdout set** -- cross-validation makes efficient use of limited data.
- **When you need confidence intervals on model performance** -- the variance across folds tells you about estimation uncertainty.
- **When building production models** -- cross-validation during development, plus a final held-out test set for reporting.
- **When working with time-series data** -- you need time-aware cross-validation to prevent look-ahead bias.
- **When your data has group structure** -- policies from the same customer, students from the same school, etc.

## How it works (intuitive)

### The core idea

Imagine you have 1,000 observations and want to know how well your model generalizes. You could split into 800 training and 200 test observations. But what if the 200 test observations happen to be unusually easy or hard? Your performance estimate would be misleading.

Cross-validation fixes this by using every observation for both training and testing (just never at the same time).

### K-fold cross-validation

The most common form. Here is how it works:

1. **Shuffle** the data randomly.
2. **Split** it into $K$ equally sized groups (folds). Typically $K = 5$ or $K = 10$.
3. **For each fold $k$**:
   - Use fold $k$ as the validation set (hold it out).
   - Train the model on the remaining $K-1$ folds.
   - Evaluate the model on fold $k$ and record the score.
4. **Average** the $K$ scores to get the cross-validated performance estimate.

*Analogy*: Imagine a class of 30 students taking a practice exam. Instead of one exam, you run 5 rounds. Each round, 6 students sit out and watch while the other 24 study. Then the 6 take the exam. After 5 rounds, every student has taken the exam exactly once, and you have a comprehensive view of how well the class performs.

**Choosing K**:
- $K = 5$: Good default. Moderate bias, moderate variance. Fast to compute.
- $K = 10$: Slightly less biased (training on 90% of data instead of 80%). Common in practice.
- $K = n$ (Leave-One-Out): Lowest bias but highest variance and very slow. See below.
- Larger $K$ means less bias (training on more data each fold) but more variance (folds overlap more) and more computation.

### Stratified K-fold

Standard K-fold randomly assigns observations to folds without caring about the target variable. If your positive class rate is 5%, some folds might end up with 2% and others with 8%, introducing noise into your estimates.

Stratified K-fold ensures that each fold has approximately the same proportion of each class as the full dataset.

*Analogy*: If you are dividing a bag of mixed candy into 5 equal piles, regular K-fold just grabs handfuls randomly. Stratified K-fold makes sure each pile has the same ratio of chocolates to gummies as the original bag.

**When to use**: Always use stratified K-fold for classification problems, especially with imbalanced classes. There is almost no reason not to.

### Leave-One-Out Cross-Validation (LOOCV)

The extreme case: $K = n$, where each fold is a single observation. Train on $n-1$ observations, test on 1, and repeat $n$ times.

**Pros**:
- Minimum bias -- you train on nearly all the data every time.
- Deterministic -- no randomness from fold assignment.

**Cons**:
- Computationally expensive -- you train $n$ separate models.
- High variance -- each training set differs by only one observation, so the models are highly correlated, which can inflate variance of the performance estimate.
- Not practical for large datasets.

**When to use**: Small datasets (under a few hundred observations) where every data point is precious. Common in actuarial analyses of small portfolios.

### Time-series cross-validation

Standard K-fold randomly assigns observations to folds, which means future data can end up in the training set while past data is in the validation set. For time-series data, this is cheating -- you are letting the model peek into the future.

Time-series CV respects temporal order: the training set always comes before the validation set.

**Expanding window (growing window)**:
- Fold 1: Train on months 1-12, validate on month 13.
- Fold 2: Train on months 1-13, validate on month 14.
- Fold 3: Train on months 1-14, validate on month 15.
- ...and so on.

The training set grows with each fold. This mimics how the model would be used in practice: you retrain on all available historical data and predict the next period.

**Sliding window (fixed window)**:
- Fold 1: Train on months 1-12, validate on month 13.
- Fold 2: Train on months 2-13, validate on month 14.
- Fold 3: Train on months 3-14, validate on month 15.
- ...and so on.

The training window has a fixed size and slides forward. This is useful when older data becomes less relevant (e.g., consumer behavior changes over time, or insurance market conditions shift).

*Analogy*: Expanding window is like studying for an exam using everything you have ever learned. Sliding window is like only studying the last semester's material, assuming older material is outdated.

**When to use**: Any time your data has a temporal component -- claims data with accident dates, monthly policy snapshots, financial time series, etc. This is very common in actuarial work.

### Grouped cross-validation

When observations are not independent -- multiple claims from the same policyholder, multiple years of data for the same account -- you need grouped CV. Standard K-fold might put some claims from policyholder A in training and others in validation, leaking information about that policyholder.

Grouped CV ensures that all observations from the same group stay in the same fold.

*Analogy*: If you are testing whether students learned a subject, you would not put some of a student's test answers in training and others in validation. You keep each student entirely in one fold.

**When to use**: Whenever your data has a natural grouping structure that could cause leakage. In insurance: group by policyholder, policy number, or household. In medical studies: group by patient. In education: group by school or teacher.

### Nested cross-validation

When you use cross-validation for both hyperparameter tuning AND performance estimation, you need to be careful. If you use the same CV loop for both, you end up selecting hyperparameters that look good on the validation folds, which biases your performance estimate upward.

Nested CV uses two loops:
- **Outer loop**: Splits data into train and test folds for unbiased performance estimation.
- **Inner loop**: Within each outer training fold, runs another CV to tune hyperparameters.

This gives you an honest estimate of how well your tuning procedure works on new data.

*Analogy*: The outer loop is the final exam. The inner loop is the practice quizzes you take while studying. You would not count your practice quiz scores as your final grade.

## Key concepts

### Bias-variance tradeoff in CV

| K value | Training set size | Bias | Variance | Computation |
|---|---|---|---|---|
| K = 2 | 50% of data | High | Low | Fast |
| K = 5 | 80% of data | Moderate | Moderate | Moderate |
| K = 10 | 90% of data | Low | Moderate-High | Moderate |
| K = n (LOOCV) | n-1 of data | Very low | High | Slow |

### Standard error of CV estimate

The mean CV score has an associated standard error:

$$\text{SE} = \frac{s}{\sqrt{K}}$$

Where $s$ is the standard deviation of the $K$ fold scores. This helps you assess whether the difference between two models is statistically meaningful.

**Caveat**: This formula assumes the fold scores are independent, which they are not (the training sets overlap). The true uncertainty is often larger than this formula suggests.

### The one-standard-error rule

When selecting among models (e.g., different values of a regularization parameter), choose the simplest model whose CV score is within one standard error of the best model's score. This guards against overfitting during model selection.

$$\text{Selected model score} \geq \text{Best score} - 1 \times \text{SE of best score}$$

### Data leakage in cross-validation

Data leakage occurs when information from the validation fold influences the training process. Common sources:

1. **Preprocessing before splitting**: Computing scaling parameters, target encoding statistics, or feature selection on the full dataset before CV creates leakage.
2. **Temporal leakage**: Using future data to predict the past in time-series.
3. **Group leakage**: Same entity in both training and validation folds.
4. **Feature leakage**: Features derived from the target or from information not available at prediction time.

**The fix**: Put ALL preprocessing steps inside the CV loop using `sklearn.pipeline.Pipeline`.

## Python implementation

```python
import numpy as np
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut,
    TimeSeriesSplit, GroupKFold,
    cross_val_score, cross_validate,
    GridSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# --- Basic K-fold ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='roc_auc')
print(f"AUC: {scores.mean():.3f} +/- {scores.std():.3f}")

# --- Stratified K-fold (preferred for classification) ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

# --- Multiple metrics at once ---
results = cross_validate(
    model, X, y, cv=skf,
    scoring=['roc_auc', 'average_precision', 'neg_log_loss'],
    return_train_score=True
)
for metric in ['test_roc_auc', 'test_average_precision', 'test_neg_log_loss']:
    vals = results[metric]
    print(f"{metric}: {vals.mean():.3f} +/- {vals.std():.3f}")

# --- Leave-One-Out ---
loo = LeaveOneOut()
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

# --- Time-series CV ---
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')

# Visualize the splits
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold}: Train indices {train_idx[0]}-{train_idx[-1]}, "
          f"Val indices {val_idx[0]}-{val_idx[-1]}")

# --- Grouped CV ---
group_kfold = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=group_kfold, groups=policyholder_ids,
                          scoring='roc_auc')

# --- Pipeline inside CV (prevents leakage) ---
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(penalty='l1', solver='saga', max_iter=1000))
])
scores = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')

# --- Nested CV for unbiased performance estimation ---
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(
    LogisticRegression(solver='saga', max_iter=1000),
    param_grid, cv=inner_cv, scoring='roc_auc', refit=True
)
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='roc_auc')
print(f"Nested CV AUC: {nested_scores.mean():.3f} +/- {nested_scores.std():.3f}")

# --- Custom time-series CV with sliding window ---
class SlidingWindowCV:
    """Fixed-size sliding window for time-series cross-validation."""
    def __init__(self, train_size, test_size, step=1):
        self.train_size = train_size
        self.test_size = test_size
        self.step = step

    def split(self, X, y=None, groups=None):
        n = len(X)
        for start in range(0, n - self.train_size - self.test_size + 1, self.step):
            train_end = start + self.train_size
            test_end = train_end + self.test_size
            yield (np.arange(start, train_end), np.arange(train_end, test_end))

    def get_n_splits(self, X=None, y=None, groups=None):
        n = len(X)
        return (n - self.train_size - self.test_size) // self.step + 1

sliding_cv = SlidingWindowCV(train_size=365, test_size=30, step=30)
scores = cross_val_score(model, X, y, cv=sliding_cv, scoring='neg_mean_squared_error')
```

## R implementation (if applicable)

```r
library(caret)
library(rsample)

# --- K-fold with caret ---
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)
model <- train(target ~ ., data = train_data, method = "glmnet",
               trControl = ctrl, metric = "ROC")

# --- Stratified K-fold (default in caret for classification) ---
# caret automatically stratifies for classification tasks

# --- Time-series CV with rsample ---
library(rsample)
rolling_origin <- rolling_origin(
  data = ts_data,
  initial = 365,     # training window size
 assess = 30,       # test window size
  cumulative = TRUE   # expanding window (FALSE for sliding)
)

# --- Grouped CV ---
ctrl_grouped <- trainControl(
  method = "cv",
  number = 5,
  index = groupKFold(train_data$policyholder_id, k = 5)
)
```

## Gotchas & lessons learned

1. **Always use stratified K-fold for classification**: Standard K-fold can produce folds with very different class proportions, especially with imbalanced data. Stratified K-fold is almost always the right choice for classification. The only exception is when you are using grouped or time-series CV.

2. **Preprocessing MUST be inside the CV loop**: If you standardize features, perform target encoding, or select features on the full dataset before cross-validation, you have introduced data leakage. The validation fold has influenced the preprocessing. Use `sklearn.pipeline.Pipeline` to ensure all transformations are fit only on the training folds.

3. **Time-series data requires time-aware CV**: Using standard K-fold on time-series data produces wildly optimistic results because the model gets to train on future data and predict the past. Always use `TimeSeriesSplit` or a custom temporal split.

4. **Grouped CV is often overlooked**: If your dataset has multiple rows per entity (multiple claims per policy, multiple years per customer), standard CV can leak information. Group by the entity to get honest estimates.

5. **CV scores have uncertainty**: A CV score of 0.82 +/- 0.05 is not meaningfully different from 0.80 +/- 0.04. Do not over-interpret small differences. Use the one-standard-error rule for model selection.

6. **Nested CV for honest performance**: If you use the same CV loop to tune hyperparameters and estimate performance, your estimate is biased upward. Use nested CV (outer loop for estimation, inner loop for tuning) when you need an unbiased performance number for reporting.

7. **Shuffling matters**: With `shuffle=True`, K-fold randomly assigns observations to folds. With `shuffle=False`, observations go to folds in order. For non-time-series data, always shuffle. For time-series data, never shuffle.

8. **Repeated CV reduces variance**: Running K-fold multiple times with different random splits (e.g., 5x10-fold = 50 model fits) reduces the variance of your performance estimate at the cost of more computation. Use `RepeatedStratifiedKFold` in scikit-learn.

9. **K=5 or K=10 is almost always fine**: Research shows that 5-fold and 10-fold CV provide a good bias-variance tradeoff for the performance estimate. LOOCV is rarely worth the computational cost.

10. **Beware of data leakage from feature engineering**: If you created features using the entire dataset (e.g., target encoding means computed on all data), those features contain information from the validation folds. Recompute data-dependent features within each CV fold.

## Resources

- **scikit-learn cross-validation guide**: https://scikit-learn.org/stable/modules/cross_validation.html
- **An Introduction to Statistical Learning (ISLR), Chapter 5**: Clear treatment of CV, bootstrap, and the bias-variance tradeoff.
- **Hastie, Tibshirani, Friedman -- ESL, Chapter 7**: Model assessment and selection, including nested CV.
- **Hyndman & Athanasopoulos -- Forecasting: Principles and Practice**: https://otexts.com/fpp3/ -- time-series cross-validation.
- **Bergmeir & Benitez (2012)**: "On the use of cross-validation for time series predictor evaluation" -- when standard CV can be okay for time series and when it cannot.
- **Cawley & Talbot (2010)**: "On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation" -- explains why nested CV is necessary.
- **caret R package**: https://topepo.github.io/caret/ -- comprehensive model training framework with built-in CV.
