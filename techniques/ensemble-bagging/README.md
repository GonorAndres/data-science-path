# Ensemble Methods: Bagging & Random Forest

## What it is

Bagging (Bootstrap Aggregating) is an ensemble technique that builds multiple copies of a model on different **bootstrap samples** of the training data and combines their predictions to produce a more stable, accurate result. **Random Forest** extends bagging by also randomizing which features each tree can consider at each split, further reducing correlation between the individual trees. The core insight is that combining many "weak" or "noisy" models produces a "strong" model with lower variance -- the individual errors cancel out when you average many independent predictions. Random Forest is one of the most widely used machine learning algorithms because it is powerful, robust, requires minimal tuning, and provides useful diagnostics like feature importance and out-of-bag error estimates.

## Why it matters

Single decision trees are intuitive and interpretable but prone to overfitting -- small changes in the training data can produce dramatically different trees. Bagging and Random Forest solve this problem by averaging over many trees, smoothing out the instability. For actuarial and data science practitioners, Random Forest offers a strong baseline model that often performs surprisingly well without extensive hyperparameter tuning. It handles non-linear relationships, interactions, and mixed feature types naturally. It also provides built-in feature importance measures that can guide feature selection for simpler, more interpretable models (like GLMs) -- a common workflow in actuarial practice.

## When to use it

- **Strong baseline model**: When you want a model that performs well "out of the box" with minimal tuning. Random Forest is often the first non-linear model to try.
- **Feature importance analysis**: When you want to understand which features drive predictions, especially before building a simpler interpretable model.
- **When you have many features**: Random Forest handles hundreds of features without feature selection.
- **When you have non-linear relationships and interactions**: Trees naturally capture these without explicit feature engineering.
- **When you need robustness**: Random Forest is resistant to outliers, missing values (with imputation), and does not require feature scaling.
- **When training data is noisy**: Bagging reduces variance, which helps when the training data has a lot of noise.
- **When you do NOT need**: (a) Inference (p-values, confidence intervals) -- use GLMs. (b) Maximum predictive performance -- boosting methods (XGBoost, LightGBM) usually outperform. (c) Interpretable coefficients -- use linear/logistic regression.

## How it works (intuitive)

### Bootstrap Sampling

A **bootstrap sample** is created by randomly drawing $n$ observations **with replacement** from a dataset of size $n$. This means:
- Some observations appear multiple times.
- Some observations are left out entirely.
- On average, about 63.2% of unique observations appear in each bootstrap sample (the rest, about 36.8%, are left out).

**Analogy**: Imagine you have a bag of 100 colored marbles. You draw 100 marbles one at a time, putting each one back before drawing the next. Your sample of 100 draws will have some colors appearing multiple times and some not appearing at all. This is a bootstrap sample.

### Bagging: The Procedure

1. **Create $B$ bootstrap samples** from the training data (typically $B = 100$ to $500$).
2. **Train a separate decision tree** on each bootstrap sample. Each tree is grown deep (low bias, high variance).
3. **Combine predictions**:
   - **Regression**: Average the predictions from all $B$ trees.
   - **Classification**: Take the majority vote across all $B$ trees (or average the predicted probabilities).

**Why does this work?**

If each tree has variance $\sigma^2$ and the trees are **independent**, the variance of the average is $\sigma^2 / B$. Even if the trees are correlated (correlation $\rho$), the variance of the average is:

$$\rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

The first term ($\rho \sigma^2$) does not decrease with more trees -- it is the irreducible variance due to correlation. The second term decreases as $B$ grows. So the key to reducing variance further is to **reduce the correlation between trees**.

### Random Forest: Decorrelating the Trees

Random Forest adds one crucial modification: at each split in each tree, instead of considering all $p$ features, only a **random subset of $m$ features** is considered.

- For **classification**: Default $m = \sqrt{p}$.
- For **regression**: Default $m = p/3$.

This is called **feature randomization** or **random subspace method**.

**Why does this help?** Without feature randomization, if one feature is very strong, every tree will split on it first, making all trees similar (high correlation). By restricting the feature set, some trees are forced to use different features, creating more diversity.

**Analogy**: Imagine an election where every voter gets the full list of candidates. If one candidate is well-known, most voters will choose them, and the election is a landslide. But if each voter only sees a random subset of candidates, the results are more diverse and the collective vote captures more information about all candidates.

### Out-of-Bag (OOB) Error

Because each bootstrap sample leaves out about 36.8% of observations, these **out-of-bag** observations serve as a free validation set for that particular tree.

**OOB error estimation**:

1. For each observation $i$, identify all trees that did NOT include $i$ in their bootstrap sample (roughly $B/3$ trees).
2. Average the predictions of only those trees for observation $i$.
3. Compare to the true value. The overall OOB error is the average prediction error across all observations.

This is nearly equivalent to leave-one-out cross-validation but comes for free, without any additional computation.

### Feature Importance

Random Forest provides two types of feature importance:

**1. Impurity-based importance (Gini importance)**:
For each feature, sum the total reduction in impurity (Gini or MSE) caused by splits on that feature across all trees. Features that create large impurity reductions are "important."

- Fast to compute (comes free with model fitting).
- **Caveat**: Biased toward high-cardinality features (features with many unique values or categories get artificially high importance).

**2. Permutation importance**:
For each feature, randomly shuffle its values across observations and measure how much the model's performance drops. A large drop = important feature.

- Slower (requires re-evaluating the model for each feature).
- Unbiased -- treats all features fairly regardless of cardinality.
- Recommended over impurity-based importance.

Use `sklearn.inspection.permutation_importance` for permutation importance.

## Key concepts

### Bias-Variance Decomposition

| Component | Single Tree | Bagged Trees | Random Forest |
|-----------|-------------|--------------|---------------|
| Bias | Low (deep tree) | Low (same) | Low (same) |
| Variance | High (unstable) | Reduced ($\sigma^2/B$ ideally) | Further reduced (decorrelation) |
| Overfitting risk | High | Moderate | Low |

Bagging reduces variance without increasing bias. Random Forest reduces variance even further by decorrelating the trees.

### Key Hyperparameters

| Parameter | Description | Default | Effect |
|-----------|-------------|---------|--------|
| `n_estimators` | Number of trees ($B$) | 100 | More trees = lower variance, diminishing returns after ~300-500 |
| `max_features` | Features per split ($m$) | $\sqrt{p}$ (classif.) or $p/3$ (regr.) | Lower = more diversity, potentially higher bias |
| `max_depth` | Maximum tree depth | None (fully grown) | Shallower = more bias, less variance. Rarely needed. |
| `min_samples_leaf` | Minimum observations per leaf | 1 | Higher = smoother predictions, less overfitting |
| `min_samples_split` | Minimum observations to split a node | 2 | Higher = fewer splits, simpler trees |
| `max_samples` | Fraction of data per bootstrap | 1.0 | Lower = more diversity, useful for very large datasets |
| `class_weight` | Weight for each class | None | Set to `'balanced'` for imbalanced classes |

### When to Tune vs. When to Accept Defaults

Random Forest is one of the few algorithms where defaults work remarkably well:
- `n_estimators`: Just increase until OOB error plateaus (300-500 is usually enough).
- `max_features`: Default is usually close to optimal.
- `max_depth`: Leave as None unless you have a strong reason to limit it.
- `min_samples_leaf`: Increase if you see overfitting (5-20 is a reasonable range).

## Python implementation

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Random Forest Classifier ---
rf_clf = RandomForestClassifier(
    n_estimators=500,
    max_features='sqrt',
    min_samples_leaf=5,
    oob_score=True,       # Enable OOB error estimation
    random_state=42,
    n_jobs=-1             # Use all CPU cores
)
rf_clf.fit(X_train, y_train)

# OOB score (no need for separate validation!)
print(f"OOB Accuracy: {rf_clf.oob_score_:.4f}")

# Test accuracy
print(f"Test Accuracy: {rf_clf.score(X_test, y_test):.4f}")

# --- Feature Importance ---
# Impurity-based (fast but biased)
feat_imp = pd.Series(rf_clf.feature_importances_,
                      index=feature_names).sort_values(ascending=False)
feat_imp.head(15).plot(kind='barh')
plt.title('Gini Feature Importance')

# Permutation importance (unbiased, recommended)
perm_imp = permutation_importance(rf_clf, X_test, y_test,
                                   n_repeats=10, random_state=42, n_jobs=-1)
perm_imp_sorted = pd.Series(perm_imp.importances_mean,
                             index=feature_names).sort_values(ascending=False)
perm_imp_sorted.head(15).plot(kind='barh')
plt.title('Permutation Feature Importance')

# --- Random Forest Regressor ---
rf_reg = RandomForestRegressor(
    n_estimators=500,
    max_features=0.33,     # p/3 for regression
    min_samples_leaf=5,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
rf_reg.fit(X_train, y_train)
print(f"OOB R-squared: {rf_reg.oob_score_:.4f}")

# --- Plain Bagging (using any base estimator) ---
from sklearn.tree import DecisionTreeClassifier
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=500,
    max_samples=1.0,       # Full bootstrap sample
    max_features=1.0,      # Use all features (this is plain bagging, not RF)
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
bagging_clf.fit(X_train, y_train)
```

## R implementation (if applicable)

```r
library(randomForest)

# Random Forest
rf_model <- randomForest(
  target ~ ., data = train_data,
  ntree = 500, mtry = sqrt(ncol(train_data) - 1),
  importance = TRUE
)
print(rf_model)

# OOB error
plot(rf_model)

# Variable importance
importance(rf_model)
varImpPlot(rf_model)

# Prediction
predictions <- predict(rf_model, newdata = test_data)

# ranger (faster alternative)
library(ranger)
rf_ranger <- ranger(
  target ~ ., data = train_data,
  num.trees = 500, importance = "permutation"
)
```

## Gotchas & lessons learned

1. **Random Forest does NOT need feature scaling.** Trees split on threshold values, so the absolute scale of features does not matter. This is a significant advantage over methods like SVM, KNN, or neural networks.

2. **More trees is almost always better.** Unlike boosting (where too many iterations can overfit), Random Forest performance monotonically improves with more trees and plateaus. The only cost is computation time. There is no reason to tune `n_estimators` down -- just set it high enough that OOB error stabilizes (300-500 is usually sufficient).

3. **Do not use Gini importance for feature selection when features have different cardinalities.** A categorical feature with 50 categories will appear more "important" than a binary feature, even if the binary feature is truly more predictive. Use permutation importance instead.

4. **Random Forest tends to average out extreme predictions.** Because it averages many trees, it cannot predict values outside the range of the training data (for regression). It also struggles with trends and extrapolation.

5. **OOB error is your friend.** Use it as a quick, reliable estimate of generalization error without splitting your data into train/validation sets. Compare OOB error to cross-validation error to verify.

6. **For imbalanced classes**, set `class_weight='balanced'` or use `class_weight='balanced_subsample'` (which adjusts weights independently for each bootstrap sample).

7. **Random Forest is not interpretable.** Unlike a single decision tree, you cannot draw a simple flowchart of the decision process. Use feature importance and partial dependence plots to understand the model's behavior.

8. **Large datasets**: Random Forest can be slow to train on very large datasets because each tree sees the full dataset. Consider subsampling (`max_samples < 1.0`) or using `sklearn.ensemble.ExtraTreesClassifier` (Extremely Randomized Trees), which is faster because it does not search for optimal split thresholds.

9. **In actuarial workflows**, Random Forest is often used for feature screening: fit an RF, examine permutation importance, then select the top features for a more interpretable GLM. This combines the power of RF with the interpretability of GLMs.

10. **Watch out for data leakage in feature importance.** If you compute permutation importance on the training data, features that are overfit will appear important. Always compute permutation importance on a held-out set.

## Resources

- **Breiman, "Random Forests" (2001)** -- The original paper. Clear and readable.
- **scikit-learn Ensemble documentation**: [Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html#forests-of-randomized-trees)
- **An Introduction to Statistical Learning (ISLR)**, Chapter 8 -- Covers bagging and Random Forest clearly.
- **The Elements of Statistical Learning (ESL)**, Chapter 15 -- More rigorous treatment.
- **Probst, Wright & Boulesteix, "Hyperparameters and Tuning Strategies for Random Forest" (2019)** -- Practical guidance on when and what to tune.
- **StatQuest**: Josh Starmer's YouTube videos on Random Forest and bagging.
- **scikit-learn permutation importance**: [Documentation](https://scikit-learn.org/stable/modules/permutation_importance.html)
