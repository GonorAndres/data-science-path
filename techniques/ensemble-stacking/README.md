# Ensemble Methods: Stacking & Blending

## What it is

Stacking (Stacked Generalization) is an ensemble technique that combines the predictions of multiple **diverse base models** by training a **meta-learner** to optimally weight and combine them. Unlike bagging (same algorithm, different data samples) and boosting (same algorithm, sequential correction), stacking uses **different algorithms** as base models and learns how to best combine their outputs. The key insight is that different models have different strengths -- a logistic regression might capture linear trends well, a random forest might handle non-linear patterns, and a KNN might excel in dense regions. A meta-learner trained on the base models' predictions can learn to trust each model where it performs best. **Blending** is a simpler variant of stacking that uses a single holdout set instead of cross-validation to generate meta-features.

## Why it matters

Stacking is one of the most reliable ways to squeeze extra predictive performance out of a set of models. Nearly every winning solution in Kaggle competitions and data science hackathons uses some form of stacking. For actuarial practitioners, stacking is valuable when you have built several candidate models (GLM, Random Forest, XGBoost) and want to combine them into a single prediction that is better than any individual model. It also provides a principled framework for model combination -- rather than ad-hoc averaging, the meta-learner learns the optimal combination from data.

## When to use it

- **When you have multiple good but different models**: Stacking shines when the base models are diverse (e.g., a linear model, a tree-based model, and a neural network). If all models are similar, stacking adds little.
- **When you need maximum predictive performance**: Stacking typically improves accuracy by 1-5% over the best individual model, which can matter in competitions or high-stakes applications.
- **When you are combining predictions from different teams or approaches**: In practice, different team members may build different models. Stacking provides a principled way to combine them.
- **When interpretability is secondary**: Stacking produces a complex pipeline that is harder to explain than a single model.
- **After you have exhausted individual model tuning**: Stacking is a "finishing move" -- first optimize your individual models, then stack them.

## How it works (intuitive)

### The Stacking Concept

**Analogy**: Imagine you are a coach assembling a relay team. Each runner (base model) has different strengths: one is great at starts, another at maintaining speed, another at finishing. As the coach (meta-learner), you learn how to assign each runner to the leg where they contribute most. You do not just pick the fastest runner for every leg -- you learn the optimal combination.

Another analogy: Think of stacking like hiring a panel of expert consultants (base models) and then hiring a decision-maker (meta-learner) who has watched how each consultant performs on past cases and learned when to trust each one.

### Step-by-Step: Stacking with Cross-Validation

The critical challenge in stacking is avoiding **data leakage**. If you train the base models on the same data you use to generate meta-features, the meta-learner will learn to trust overfit predictions, which defeats the purpose. The solution is **cross-validated predictions**.

**Level 0 (Base Models):**

1. Choose $K$ folds for cross-validation (e.g., $K = 5$).
2. For each base model $m$ and each fold $k$:
   a. Train model $m$ on the $K-1$ training folds.
   b. Generate predictions for the held-out fold $k$.
3. Stitch together the held-out predictions to create a full set of **out-of-fold (OOF) predictions** for each base model.
4. Also train each base model on the FULL training set and generate predictions for the test set (these will be used for final test predictions).

At this point, for $M$ base models and $n$ training observations, you have:
- Training meta-features: An $n \times M$ matrix of OOF predictions.
- Test meta-features: An $n_{\text{test}} \times M$ matrix of predictions on the test set.

**Level 1 (Meta-Learner):**

5. Train the meta-learner on the training meta-features (OOF predictions as features, original target as the label).
6. Predict on the test meta-features.

**The meta-learner is typically simple**: logistic regression (for classification) or ridge regression (for regression). A complex meta-learner risks overfitting the OOF predictions.

### Visual Diagram

```
Training Data
     |
     v
 +-------+-------+-------+
 | Base  | Base  | Base  |    Level 0: Diverse base models
 | Model | Model | Model |    (each trained with K-fold CV)
 |  (1)  |  (2)  |  (3)  |
 +---+---+---+---+---+---+
     |       |       |
     v       v       v
 [OOF_1] [OOF_2] [OOF_3]    Out-of-fold predictions
     |       |       |
     +---+---+---+---+
         |
         v
   +-----+-----+
   | Meta-      |    Level 1: Meta-learner
   | Learner    |    (trained on OOF predictions)
   +-----+------+
         |
         v
   Final Prediction
```

### Blending (Simplified Stacking)

Blending is a simpler alternative:

1. Split training data into two parts: a "training set" (e.g., 80%) and a "blending set" (e.g., 20%).
2. Train base models on the training set.
3. Generate predictions on the blending set.
4. Train the meta-learner on these predictions.

**Advantages**: Simpler to implement, faster (no cross-validation).
**Disadvantages**: Uses less data (the meta-learner only sees 20% of observations), so less robust than CV-based stacking.

### Multi-Level Stacking

You can extend stacking to multiple levels:

```
Level 0: Base models --> OOF predictions (meta-features)
Level 1: Level-1 models trained on meta-features --> Level-1 OOF predictions
Level 2: Final meta-learner trained on Level-1 OOF predictions
```

In practice, 2 levels (one layer of base models + one meta-learner) is usually sufficient. Adding more levels has diminishing returns and increases overfitting risk.

## Key concepts

### Diversity is Key

Stacking works best when base models are **diverse** -- they should make errors on different observations. Combining three random forests with slightly different hyperparameters adds almost no value. Combining a linear model, a tree-based model, and a distance-based model adds significant value.

**Sources of diversity**:
- **Different algorithms**: Logistic regression, Random Forest, XGBoost, KNN, SVM.
- **Different feature subsets**: One model uses all features, another uses only the top 10.
- **Different transformations**: One model uses raw features, another uses PCA-reduced features.
- **Different hyperparameters**: Different regularization strengths, tree depths, etc.

### Why Cross-Validated OOF Predictions?

If you train a base model on the full training set and then predict on the same training set, those predictions will be overfit. A complex model (like a deep random forest) will have near-perfect training predictions, making the meta-learner overvalue it. Cross-validated OOF predictions ensure that each training prediction was made by a model that never saw that observation, giving honest estimates of each model's accuracy.

### Common Meta-Learners

| Meta-Learner | When to Use |
|-------------|-------------|
| Logistic Regression | Classification. Simple, fast, rarely overfits. |
| Ridge Regression | Regression. Handles correlated base model predictions. |
| Lasso Regression | When you want automatic selection of useful base models. |
| Linear Regression | Simplest option, but can overfit if base model predictions are correlated. |
| XGBoost (shallow) | When base model interactions matter, but use with caution (overfitting risk). |

**Rule of thumb**: Use a simple meta-learner. The diversity comes from the base models, not the meta-learner.

### Adding Original Features to the Meta-Learner

Sometimes it helps to include the original features alongside the base model predictions as inputs to the meta-learner. This gives the meta-learner more information to work with. However:
- It increases overfitting risk (more features).
- It makes the meta-learner more complex.
- The benefit depends on whether the base models have already captured all the information in the original features.

Try both approaches and compare cross-validated performance.

### Averaging vs. Stacking

A simpler alternative to stacking is just **averaging** the base model predictions (or taking a weighted average). This works surprisingly well in practice:

$$\hat{y} = \frac{1}{M} \sum_{m=1}^{M} \hat{y}_m \quad \text{(simple average)}$$

$$\hat{y} = \sum_{m=1}^{M} w_m \hat{y}_m, \quad \sum w_m = 1 \quad \text{(weighted average)}$$

**When does stacking beat averaging?**
- When base models have very different accuracy levels (stacking learns to downweight weaker models).
- When base models are strong in different regions of the feature space (stacking learns this locally).
- When there are interactions between base model predictions.

**When is averaging sufficient?**
- When base models have similar accuracy.
- When you want simplicity and stability.
- When you have limited data (stacking's cross-validation can be unstable with small datasets).

## Python implementation

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np

# --- Using scikit-learn's StackingClassifier ---
# Define base models (Level 0)
base_models = [
    ('lr', Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, max_iter=1000))
    ])),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=8,
                                   random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                      max_depth=4, random_state=42)),
    ('knn', Pipeline([
        ('scaler', StandardScaler()),
        ('clf', KNeighborsClassifier(n_neighbors=10))
    ]))
]

# Define meta-learner (Level 1)
meta_learner = LogisticRegression(C=1.0, max_iter=1000)

# Build stacking ensemble
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_learner,
    cv=5,                    # 5-fold CV for generating OOF predictions
    stack_method='predict_proba',  # Use probabilities as meta-features
    passthrough=False,       # Set True to include original features
    n_jobs=-1
)
stacking_clf.fit(X_train, y_train)
print(f"Stacking Test Accuracy: {stacking_clf.score(X_test, y_test):.4f}")

# Compare to individual models
for name, model in base_models:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} Test Accuracy: {score:.4f}")

# --- Manual Stacking (for more control) ---
from sklearn.model_selection import KFold

def get_oof_predictions(model, X_train, y_train, X_test, n_folds=5):
    """Generate out-of-fold predictions for stacking."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_train = np.zeros(len(X_train))
    oof_test_folds = np.zeros((len(X_test), n_folds))

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr = y_train[train_idx]

        model.fit(X_tr, y_tr)
        oof_train[val_idx] = model.predict_proba(X_val)[:, 1]
        oof_test_folds[:, fold_idx] = model.predict_proba(X_test)[:, 1]

    oof_test = oof_test_folds.mean(axis=1)
    return oof_train, oof_test

# Generate OOF predictions for each base model
meta_train = []
meta_test = []
for name, model in base_models:
    oof_tr, oof_te = get_oof_predictions(model, X_train, y_train, X_test)
    meta_train.append(oof_tr)
    meta_test.append(oof_te)
    print(f"{name} OOF generated")

# Stack into meta-feature matrices
X_meta_train = np.column_stack(meta_train)
X_meta_test = np.column_stack(meta_test)

# Train meta-learner
meta_model = LogisticRegression(C=1.0, max_iter=1000)
meta_model.fit(X_meta_train, y_train)
print(f"Stacking Accuracy: {meta_model.score(X_meta_test, y_test):.4f}")

# --- Simple Averaging (baseline comparison) ---
avg_pred = np.mean(meta_test, axis=0)
avg_labels = (avg_pred > 0.5).astype(int)
avg_accuracy = np.mean(avg_labels == y_test)
print(f"Simple Averaging Accuracy: {avg_accuracy:.4f}")
```

## R implementation (if applicable)

```r
library(caret)
library(caretEnsemble)

# Define base models
control <- trainControl(method = "cv", number = 5,
                         savePredictions = "final",
                         classProbs = TRUE)

model_list <- caretList(
  target ~ ., data = train_data,
  trControl = control,
  methodList = c("glm", "rf", "gbm", "knn")
)

# Stack with meta-learner
stack_model <- caretStack(
  model_list,
  method = "glm",  # Meta-learner
  metric = "Accuracy"
)
summary(stack_model)
```

## Gotchas & lessons learned

1. **Data leakage is the biggest risk.** If you generate meta-features by predicting on the same data the base model was trained on, you will get overly optimistic results and the meta-learner will overfit. Always use cross-validated out-of-fold predictions.

2. **Diversity matters more than individual model quality.** A stack of three mediocre but diverse models often beats a stack of three excellent but similar models. Include models with fundamentally different inductive biases.

3. **Keep the meta-learner simple.** The meta-learner has very few features (one per base model), so a complex meta-learner easily overfits. Logistic regression or ridge regression is usually the best choice.

4. **Use `predict_proba` instead of `predict` for meta-features (classification).** Probabilities contain more information than hard class labels. A model that predicts 0.51 vs. 0.99 conveys different confidence levels that the meta-learner can exploit.

5. **Stacking has diminishing returns.** Adding a 4th model to a stack of 3 typically helps less than adding a 3rd to a stack of 2. Beyond 5-7 base models, improvements are usually marginal.

6. **The improvement is often small but consistent.** Expect 1-5% improvement in accuracy or AUC over the best individual model. In competitions, this edge matters. In production, weigh it against the added complexity.

7. **Watch for overfitting with small datasets.** If your dataset has fewer than a few thousand observations, the cross-validated OOF predictions can be noisy, and the meta-learner may overfit to this noise. Consider simpler averaging instead.

8. **Stacking adds latency and complexity to deployment.** In production, you need to maintain and serve all base models plus the meta-learner. Consider whether the marginal accuracy improvement justifies this operational overhead.

9. **Use consistent cross-validation folds across all base models.** Every base model should use the same fold splits when generating OOF predictions. This ensures that the meta-learner sees a consistent set of predictions.

10. **For actuarial work**, stacking can be useful in model validation: if a stacked model significantly outperforms a single GLM, it suggests that the GLM is missing important non-linear patterns or interactions. This diagnostic insight can be as valuable as the stacked model itself.

## Resources

- **Wolpert, "Stacked Generalization" (1992)** -- The original stacking paper. Establishes the theoretical framework.
- **Breiman, "Stacked Regressions" (1996)** -- Practical extension with regularized meta-learners.
- **scikit-learn Stacking documentation**: [StackingClassifier](https://scikit-learn.org/stable/modules/ensemble.html#stacked-generalization)
- **Kaggle, "Introduction to Ensembling/Stacking"** -- Practical tutorials from top competitors.
- **MLWave blog, "Kaggle Ensembling Guide"** -- Comprehensive practical guide to stacking in competitions.
- **An Introduction to Statistical Learning (ISLR)**, Chapter 8 -- General ensemble methods background.
- **Zhou, "Ensemble Methods: Foundations and Algorithms" (2012)** -- Textbook covering stacking theory in depth.
