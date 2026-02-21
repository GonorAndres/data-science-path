# Gradient Boosting Deep Dive: XGBoost, LightGBM, CatBoost

## What it is

XGBoost, LightGBM, and CatBoost are three highly optimized implementations of gradient boosting that have become the dominant algorithms for structured/tabular data in both competitions and production. While the core idea is the same (sequentially fit trees to residuals), each library introduces engineering innovations that make training faster, more memory-efficient, and more accurate. **XGBoost** pioneered regularized gradient boosting with column sampling and efficient parallel tree construction. **LightGBM** introduced histogram-based splitting and leaf-wise tree growth for dramatic speed improvements. **CatBoost** handles categorical features natively and uses ordered boosting to reduce overfitting from target leakage. This README focuses on practical tuning -- understanding the key hyperparameters, when to choose which library, and how to systematically optimize performance.

## Why it matters

If you work with tabular data (and as an actuarial data scientist, you almost certainly do), gradient boosting libraries will be your most powerful predictive tools. They consistently win or place in the top positions of data science competitions on structured data, and they are widely deployed in industry for pricing, fraud detection, churn prediction, credit scoring, and risk assessment. Knowing how to tune these models -- which parameters matter, what values to try, and how to avoid overfitting -- is among the most practically valuable skills in applied machine learning.

## When to use it

- **Structured/tabular data**: These libraries are the go-to choice for any prediction task with features in columns and observations in rows.
- **When you need maximum predictive accuracy**: On tabular data, gradient boosting typically outperforms random forests, neural networks, and linear models.
- **Classification and regression**: All three libraries handle both seamlessly.
- **Ranking problems**: XGBoost and LightGBM have built-in ranking objectives (LambdaMART).
- **When you have moderate-to-large datasets**: These libraries are optimized for datasets from thousands to millions of rows.
- **When you need fast training**: LightGBM is particularly fast and memory-efficient.
- **When you have many categorical features**: CatBoost handles them natively without manual encoding.

## How it works (intuitive)

### Review: Gradient Boosting Core

All three libraries implement the same core algorithm (see the ensemble-boosting README for the conceptual foundation):

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \nu \cdot h_m(x)$$

Each tree $h_m$ is fit to the negative gradient (pseudo-residuals) of the loss function. The innovations are in HOW the trees are built.

### XGBoost Innovations

**1. Regularized Objective Function**

XGBoost adds explicit regularization to the tree-building objective:

$$\text{Obj} = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{m=1}^{M} \Omega(h_m)$$

Where the regularization term for each tree is:

$$\Omega(h) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$

- $T$ = number of leaves in the tree.
- $w_j$ = weight (prediction) of leaf $j$.
- $\gamma$ = penalty for each additional leaf (pruning parameter).
- $\lambda$ = L2 regularization on leaf weights.

This is like Lasso/Ridge regression for tree weights, preventing any single leaf from making an extreme prediction.

**2. Second-Order Approximation**

XGBoost uses a second-order Taylor expansion of the loss function (using both gradient and Hessian), which provides better approximation and faster convergence than the first-order gradient alone.

$$\text{Obj} \approx \sum_{i=1}^{n} \left[ g_i h_m(x_i) + \frac{1}{2} h_i h_m(x_i)^2 \right] + \Omega(h_m)$$

Where $g_i = \partial L / \partial \hat{y}_i$ (gradient) and $h_i = \partial^2 L / \partial \hat{y}_i^2$ (Hessian).

**3. Column Subsampling**

Like Random Forest's feature randomization, XGBoost can sample a random subset of features for each tree or each split, adding diversity.

**4. Sparsity-Aware Splitting**

XGBoost efficiently handles missing values by learning the optimal direction (left or right) for missing values at each split, rather than requiring imputation.

### LightGBM Innovations

**1. Histogram-Based Splitting**

Instead of evaluating every possible split point (as in exact greedy XGBoost), LightGBM **bins** continuous features into discrete histograms (typically 255 bins). Splitting happens on bin boundaries.

- **Speed**: Evaluating 255 bin boundaries is much faster than evaluating thousands of unique values.
- **Memory**: Storing 8-bit bin indices instead of 64-bit floats reduces memory by 8x.
- **Regularization**: Binning acts as implicit regularization by reducing the granularity of splits.

**2. Leaf-Wise (Best-First) Tree Growth**

Standard gradient boosting grows trees **level-wise** (all nodes at depth $d$ are split before moving to depth $d+1$). LightGBM grows trees **leaf-wise** -- it always splits the leaf with the highest gain, regardless of depth.

Leaf-wise growth produces deeper, more asymmetric trees that capture complex patterns with fewer leaves. However, it can overfit more easily, so limiting `num_leaves` is critical.

**3. Gradient-Based One-Side Sampling (GOSS)**

Not all data points contribute equally to the gradient. Points with large gradients (large errors) are more informative. GOSS keeps all large-gradient points and randomly samples a fraction of small-gradient points, speeding up training while maintaining accuracy.

**4. Exclusive Feature Bundling (EFB)**

High-dimensional sparse features (e.g., one-hot encoded categoricals) are often mutually exclusive (never non-zero at the same time). EFB bundles them into a single feature, reducing the effective number of features.

### CatBoost Innovations

**1. Native Categorical Feature Handling**

CatBoost converts categorical features to numeric using **ordered target statistics**, avoiding the need for one-hot encoding, label encoding, or target encoding as preprocessing.

For each observation, CatBoost computes the target statistic using only observations that came BEFORE it in a random permutation, preventing target leakage:

$$\hat{x}_k^i = \frac{\sum_{j < i, x_j^k = x_i^k} y_j + a \cdot p}{\sum_{j < i, x_j^k = x_i^k} 1 + a}$$

Where $a$ is a smoothing parameter and $p$ is the global prior (overall mean).

**2. Ordered Boosting**

Standard gradient boosting has a subtle form of target leakage: the model predictions used to compute gradients at round $m$ were trained on the same data. CatBoost uses **ordered boosting** with random permutations to ensure that gradients for each observation are computed from models that never saw that observation.

**3. Symmetric (Oblivious) Decision Trees**

CatBoost uses oblivious trees by default: at each level, the same split condition is applied to all nodes. This creates simpler, more regularized trees that are also faster for inference.

### Comparison Table

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Tree growth | Level-wise (default) | Leaf-wise | Symmetric/oblivious |
| Splitting | Exact or histogram | Histogram | Ordered |
| Speed | Fast | Fastest | Moderate |
| Memory | Moderate | Low | Moderate |
| Categorical handling | Manual encoding needed | Integer encoding + special mode | Native (best) |
| Missing values | Built-in (learns direction) | Built-in | Built-in |
| GPU support | Yes | Yes | Yes (strong) |
| Default performance | Good | Good | Often best out-of-box |
| Overfitting risk | Moderate | Higher (leaf-wise) | Lower (ordered boosting) |

## Key concepts

### The Hyperparameter Hierarchy

Not all hyperparameters are equally important. Here is a practical priority order:

**Priority 1 -- Most Impact:**
1. `learning_rate` (eta): Controls step size. Start at 0.05-0.1.
2. `n_estimators` (num_boost_round): Number of trees. Set high and use early stopping.
3. `max_depth` / `num_leaves`: Controls tree complexity. The MOST important regularization knob.

**Priority 2 -- Moderate Impact:**
4. `min_child_weight` / `min_data_in_leaf`: Minimum observations per leaf.
5. `subsample` / `bagging_fraction`: Fraction of data per tree (stochastic boosting).
6. `colsample_bytree` / `feature_fraction`: Fraction of features per tree.

**Priority 3 -- Fine-Tuning:**
7. `reg_alpha` (L1) / `reg_lambda` (L2): Regularization on leaf weights.
8. `gamma` / `min_split_gain`: Minimum loss reduction for a split.
9. `max_bin`: Number of histogram bins (LightGBM).

### Hyperparameter Mapping Across Libraries

| Concept | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| Learning rate | `eta` / `learning_rate` | `learning_rate` | `learning_rate` |
| Number of trees | `n_estimators` / `num_boost_round` | `n_estimators` / `num_iterations` | `iterations` |
| Tree depth | `max_depth` (3-10) | `max_depth` (-1 = no limit) | `depth` (6-10) |
| Leaf count | N/A (set max_depth) | `num_leaves` (31) | N/A (set depth) |
| Min samples per leaf | `min_child_weight` | `min_data_in_leaf` | `min_data_in_leaf` |
| Row sampling | `subsample` | `bagging_fraction` | `subsample` |
| Column sampling | `colsample_bytree` | `feature_fraction` | `rsm` |
| L1 regularization | `reg_alpha` / `alpha` | `lambda_l1` | `l2_leaf_reg` (L2 only) |
| L2 regularization | `reg_lambda` / `lambda` | `lambda_l2` | `l2_leaf_reg` |
| Min split gain | `gamma` | `min_split_gain` | N/A |

### Early Stopping

The single most important practical technique. Set `n_estimators` very high (e.g., 10,000) and let early stopping decide when to stop:

```python
# XGBoost
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          early_stopping_rounds=50,
          verbose=100)

# LightGBM
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

# CatBoost
model.fit(X_train, y_train,
          eval_set=(X_val, y_val),
          early_stopping_rounds=50,
          verbose=100)
```

**Rule of thumb for patience**: Set `early_stopping_rounds` to 5-10% of your maximum trees. For 5000 max trees, use patience of 50-100.

### Systematic Tuning Strategy

Rather than tuning all parameters at once, follow this structured approach:

**Step 1**: Fix `learning_rate = 0.1`, use early stopping to find a reasonable `n_estimators`.

**Step 2**: Tune tree structure (`max_depth` or `num_leaves`, `min_child_weight`).
```python
# Grid for max_depth (XGBoost/CatBoost)
max_depth: [3, 4, 5, 6, 7, 8]

# Grid for num_leaves (LightGBM) -- approximate 2^max_depth
num_leaves: [15, 31, 63, 127, 255]
```

**Step 3**: Tune sampling parameters (`subsample`, `colsample_bytree`).
```python
subsample: [0.6, 0.7, 0.8, 0.9, 1.0]
colsample_bytree: [0.6, 0.7, 0.8, 0.9, 1.0]
```

**Step 4**: Tune regularization (`reg_alpha`, `reg_lambda`, `gamma`).
```python
reg_alpha: [0, 0.01, 0.1, 1, 10]
reg_lambda: [0, 0.01, 0.1, 1, 10]
```

**Step 5**: Reduce `learning_rate` to 0.01-0.05, increase `n_estimators` with early stopping.

**For automated tuning**, use `Optuna` or `hyperopt`:

```python
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
    }
    model = xgb.XGBClassifier(**params, n_estimators=5000, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model.score(X_val, y_val)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
print(study.best_params)
```

### Handling Categorical Features

| Approach | XGBoost | LightGBM | CatBoost |
|----------|---------|----------|----------|
| One-hot encoding | Works (but high cardinality is expensive) | Works | Not recommended |
| Label/ordinal encoding | Works well | Works well | Not recommended |
| Native support | Limited (experimental) | Yes (`categorical_feature` param) | Yes (best, just specify `cat_features`) |

**CatBoost** handles categoricals best. If using XGBoost/LightGBM with high-cardinality categoricals, consider target encoding (with proper cross-validation to avoid leakage).

### Loss Functions for Common Tasks

| Task | XGBoost | LightGBM | CatBoost |
|------|---------|----------|----------|
| Binary classification | `binary:logistic` | `binary` | `Logloss` |
| Multi-class | `multi:softmax` | `multiclass` | `MultiClass` |
| Regression (MSE) | `reg:squarederror` | `regression` | `RMSE` |
| Regression (MAE) | `reg:absoluteerror` | `mae` | `MAE` |
| Poisson regression | `count:poisson` | `poisson` | `Poisson` |
| Tweedie regression | `reg:tweedie` | `tweedie` | `Tweedie` |
| Gamma regression | `reg:gamma` | `gamma` | N/A |
| Ranking | `rank:pairwise` | `lambdarank` | `YetiRank` |

**For actuarial pricing**: Use `count:poisson` for frequency, `reg:gamma` for severity, or `reg:tweedie` for pure premium. Set `tweedie_variance_power` between 1 and 2.

## Python implementation

```python
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import pandas as pd

# Split off validation set for early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

# --- XGBoost ---
xgb_model = xgb.XGBClassifier(
    n_estimators=5000,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    gamma=0.1,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=50
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=100)
print(f"XGBoost best iteration: {xgb_model.best_iteration}")
print(f"XGBoost test score: {xgb_model.score(X_test, y_test):.4f}")

# --- LightGBM ---
lgb_model = lgb.LGBMClassifier(
    n_estimators=5000,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    min_data_in_leaf=20,
    bagging_fraction=0.8,
    bagging_freq=1,
    feature_fraction=0.8,
    lambda_l1=0.1,
    lambda_l2=1.0,
    min_split_gain=0.01,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
)
print(f"LightGBM best iteration: {lgb_model.best_iteration_}")
print(f"LightGBM test score: {lgb_model.score(X_test, y_test):.4f}")

# --- CatBoost ---
cat_features = [i for i, col in enumerate(X_train.columns)
                if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category']

cb_model = CatBoostClassifier(
    iterations=5000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    subsample=0.8,
    rsm=0.8,
    cat_features=cat_features,
    eval_metric='Logloss',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)
cb_model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
print(f"CatBoost best iteration: {cb_model.best_iteration_}")
print(f"CatBoost test score: {cb_model.score(X_test, y_test):.4f}")

# --- Feature Importance (all three) ---
# XGBoost
xgb_imp = pd.Series(xgb_model.feature_importances_,
                      index=feature_names).sort_values(ascending=False)

# LightGBM (use 'gain' for split-based importance)
lgb_imp = pd.Series(lgb_model.feature_importances_,
                      index=feature_names).sort_values(ascending=False)

# CatBoost
cb_imp = pd.Series(cb_model.feature_importances_,
                     index=feature_names).sort_values(ascending=False)

# For more reliable importance, use SHAP
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)

# --- Actuarial Example: Poisson Frequency Model ---
freq_model = xgb.XGBRegressor(
    objective='count:poisson',
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    early_stopping_rounds=50
)
# Weight by exposure (sample_weight) or use offset
freq_model.fit(X_tr, y_tr,
               sample_weight=exposure_tr,
               eval_set=[(X_val, y_val)],
               sample_weight_eval_set=[exposure_val],
               verbose=100)
```

## R implementation (if applicable)

```r
library(xgboost)
library(lightgbm)
library(catboost)

# XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)

params <- list(
  objective = "binary:logistic",
  eta = 0.05, max_depth = 5,
  subsample = 0.8, colsample_bytree = 0.8,
  eval_metric = "logloss"
)
xgb_model <- xgb.train(
  params, dtrain, nrounds = 5000,
  watchlist = list(val = dtest),
  early_stopping_rounds = 50, verbose = 1
)
xgb.importance(model = xgb_model) |> xgb.plot.importance()

# LightGBM
lgb_train <- lgb.Dataset(data = as.matrix(X_train), label = y_train)
lgb_val <- lgb.Dataset(data = as.matrix(X_test), label = y_test)

lgb_model <- lgb.train(
  list(objective = "binary", metric = "binary_logloss",
       learning_rate = 0.05, num_leaves = 31),
  lgb_train, nrounds = 5000,
  valids = list(val = lgb_val),
  early_stopping_rounds = 50
)
```

## Gotchas & lessons learned

1. **Always use early stopping.** This is the single most important practical tip. Set `n_estimators` high and let the validation performance decide when to stop. Without early stopping, you WILL overfit.

2. **LightGBM's `num_leaves` is more important than `max_depth`.** Since LightGBM grows leaf-wise, depth is less meaningful. Control complexity with `num_leaves`. A good starting point: `num_leaves = 31` (equivalent to a balanced tree of depth 5).

3. **`num_leaves` should be less than $2^{\text{max\_depth}}$.** If `max_depth = 7`, then `num_leaves` should be less than 128. Setting `num_leaves` equal to $2^{\text{max\_depth}}$ makes the constraint redundant.

4. **CatBoost is the best default for datasets with many categorical features.** One-hot encoding high-cardinality categoricals (e.g., ZIP code with 40,000 values) is wasteful. CatBoost handles this natively and often outperforms.

5. **Use SHAP values for interpretability, not built-in feature importance.** Built-in importance measures (gain, split count) can be inconsistent across libraries and biased. SHAP provides theoretically grounded, consistent feature attributions.

6. **Learning rate and number of trees are coupled.** When you reduce the learning rate, you need more trees. A common workflow: tune with `learning_rate = 0.1` for speed, then halve the learning rate and double the trees for the final model.

7. **Watch for target leakage with categorical encoding.** If you target-encode categoricals outside the cross-validation loop, you introduce leakage. CatBoost avoids this with ordered encoding. For other libraries, target-encode within the CV folds.

8. **LightGBM is fastest but most prone to overfitting.** The leaf-wise growth strategy can create deep, narrow trees that overfit. Be disciplined about `num_leaves`, `min_data_in_leaf`, and early stopping.

9. **For insurance/actuarial applications**, use the appropriate objective function: `count:poisson` for frequency, `reg:gamma` for severity, `reg:tweedie` for pure premium. These enforce the correct distributional assumptions (non-negative predictions, appropriate mean-variance relationship).

10. **GPU training provides massive speedup for large datasets.** Set `tree_method='gpu_hist'` (XGBoost), `device='gpu'` (LightGBM), or `task_type='GPU'` (CatBoost). On datasets with >100K rows, GPU training is 5-10x faster.

11. **Monitor for train-test divergence.** Plot training and validation loss curves side by side. If they diverge early, you need more regularization. If they converge but validation loss is high, you need less regularization or more features.

12. **Ensemble the three libraries for extra performance.** A simple average of XGBoost, LightGBM, and CatBoost predictions is a surprisingly strong final model because the libraries have different inductive biases (different tree growth strategies, different binning methods).

## Decision Guide: Which Library?

```
Start here
  |
  v
Do you have many categorical features (>10 or high cardinality)?
  |-- YES --> CatBoost (native categorical handling)
  |-- NO  --> Continue
  v
Is training speed critical (millions of rows, many iterations)?
  |-- YES --> LightGBM (fastest histogram-based implementation)
  |-- NO  --> Continue
  v
Do you need maximum robustness with minimal tuning?
  |-- YES --> CatBoost (best defaults, ordered boosting)
  |-- NO  --> Continue
  v
Default choice: XGBoost (most mature, best documented, widest community)
  |
  v
For maximum performance: Average all three.
```

## Resources

- **XGBoost documentation**: [xgboost.readthedocs.io](https://xgboost.readthedocs.io/)
- **LightGBM documentation**: [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io/)
- **CatBoost documentation**: [catboost.ai](https://catboost.ai/)
- **Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System" (2016)** -- The XGBoost paper.
- **Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" (2017)** -- The LightGBM paper.
- **Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features" (2018)** -- The CatBoost paper.
- **Optuna documentation**: [optuna.org](https://optuna.org/) -- Best hyperparameter optimization framework.
- **SHAP documentation**: [shap.readthedocs.io](https://shap.readthedocs.io/) -- For interpreting gradient boosting models.
- **Kaggle "How to Win a Data Science Competition"**: Coursera course by top Kagglers. Covers gradient boosting tuning extensively.
- **Laurae's XGBoost/LightGBM parameter notes**: Community benchmark of hyperparameter effects.
