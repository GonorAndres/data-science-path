# Ensemble Methods: Boosting (AdaBoost & Gradient Boosting Concept)

## What it is

Boosting is an ensemble technique that builds models **sequentially**, where each new model focuses on correcting the mistakes of the previous ones. Unlike bagging (which builds independent models in parallel and averages them), boosting creates a chain of models where each one learns from the errors of its predecessors. The two foundational boosting algorithms are **AdaBoost** (Adaptive Boosting), which re-weights misclassified observations to force subsequent models to focus on hard cases, and **Gradient Boosting**, which fits each new model to the **residual errors** (gradients) of the combined ensemble so far. Boosting converts many weak learners (models only slightly better than random guessing) into a single strong learner, and consistently produces among the most accurate predictions in machine learning.

## Why it matters

Boosting methods -- particularly gradient boosting -- dominate structured/tabular data competitions and real-world applications. XGBoost, LightGBM, and CatBoost (covered in the gradient-boosting-deep-dive technique folder) are all implementations of gradient boosting. Before diving into those libraries, it is essential to understand the underlying concepts: how sequential learning works, what residual fitting means, and how the learning rate controls the bias-variance tradeoff. For actuarial practitioners, boosting offers the best predictive performance on tabular data (pricing, claim prediction, fraud detection), though at the cost of interpretability compared to GLMs.

## When to use it

- **Maximum predictive accuracy on tabular data**: Boosting (especially gradient boosting) consistently outperforms other methods on structured data.
- **Classification and regression**: Both AdaBoost and Gradient Boosting handle both tasks.
- **When you can tolerate a black-box model**: Boosting models are complex ensembles that sacrifice interpretability for performance.
- **When you have sufficient data**: Boosting is data-hungry; with very small datasets, simpler models (logistic regression, single tree) may be preferable.
- **When overfitting is not extreme**: Boosting can overfit if pushed too far (unlike Random Forest, which is more self-regulating). Proper use of early stopping and learning rate is important.
- **After establishing a baseline**: Boosting is typically used after simpler models (GLM, Random Forest) to push performance further.

## How it works (intuitive)

### AdaBoost (Adaptive Boosting)

AdaBoost was one of the first successful boosting algorithms (Freund & Schapire, 1997). The core idea: give every training observation a weight. Start with equal weights. After fitting a weak learner, **increase the weights of misclassified observations** so the next learner focuses more on the hard cases.

**Step by step (binary classification):**

1. **Initialize weights**: $w_i = 1/n$ for all $n$ observations.

2. **For each round** $t = 1, 2, \ldots, T$:
   a. Fit a **weak learner** $h_t(x)$ to the weighted training data. A "weak learner" is typically a decision stump (a tree with only one split).
   b. Compute the **weighted error rate**: $\epsilon_t = \sum_{i: h_t(x_i) \neq y_i} w_i$
   c. Compute the **learner weight**: $\alpha_t = \frac{1}{2} \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)$. Models with lower error get higher weight.
   d. **Update observation weights**:
      - Correctly classified: $w_i \leftarrow w_i \cdot e^{-\alpha_t}$ (decrease weight)
      - Misclassified: $w_i \leftarrow w_i \cdot e^{\alpha_t}$ (increase weight)
   e. Normalize weights so they sum to 1.

3. **Final prediction**: $H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$

**Analogy**: Imagine a classroom where a series of substitute teachers each give a quiz. After each quiz, the students who got questions wrong have those questions highlighted for the next teacher to focus on. Each teacher is only slightly better than random, but together they cover each other's blind spots. The final "grade" is a weighted vote of all teachers, where teachers who performed well overall get more say.

**Key property**: AdaBoost is equivalent to **forward stagewise additive modeling** with an exponential loss function. This connection to loss minimization leads naturally to the more general gradient boosting framework.

### Gradient Boosting

Gradient Boosting generalizes the boosting concept: instead of re-weighting observations, it fits each new model to the **negative gradient of the loss function** -- which, for squared-error loss, simplifies to the residuals.

**The core idea**: Build the model additively:

$$F_M(x) = F_0(x) + \sum_{m=1}^{M} \nu \cdot h_m(x)$$

Where:
- $F_0(x)$ is the initial prediction (e.g., the mean of the target).
- $h_m(x)$ is the $m$-th weak learner, fitted to correct the errors of $F_{m-1}$.
- $\nu$ is the **learning rate** (shrinkage factor, typically 0.01-0.3).

**Step by step (regression with squared-error loss):**

1. **Initialize**: $F_0(x) = \bar{y}$ (predict the mean).

2. **For each round** $m = 1, 2, \ldots, M$:
   a. Compute the **pseudo-residuals**: $r_{im} = y_i - F_{m-1}(x_i)$. These are the errors the current ensemble is making.
   b. Fit a decision tree $h_m(x)$ to the pseudo-residuals $\{(x_i, r_{im})\}$.
   c. Update the ensemble: $F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$.

3. **Final prediction**: $F_M(x) = F_0 + \nu \sum_{m=1}^{M} h_m(x)$.

**Analogy**: Imagine you are sculpting a statue. Your first rough cut gets the general shape (the initial prediction). Then you look at what is wrong -- the nose is too big, the ears are too small -- and make small corrections (the residuals). Each round of corrections addresses the remaining errors. The learning rate is like using a fine chisel instead of a sledgehammer -- you make small, careful adjustments to avoid over-correcting.

**Why "gradient"?** For general loss functions (not just squared error), the pseudo-residuals are the **negative gradient of the loss function** with respect to the prediction:

$$r_{im} = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} \bigg|_{F = F_{m-1}}$$

For squared error loss $L = \frac{1}{2}(y - F)^2$, the gradient is $-(y - F) = -(residual)$, so the negative gradient IS the residual. For other losses (absolute error, log-loss, Huber), the pseudo-residuals take different forms but the algorithm structure is identical.

### Sequential vs. Parallel Learning

This is the fundamental difference between bagging and boosting:

| Property | Bagging (Random Forest) | Boosting |
|----------|------------------------|----------|
| Learning | Parallel (trees are independent) | Sequential (each tree depends on previous) |
| What each tree learns | The full signal from its bootstrap sample | The residual error from the previous ensemble |
| Combining | Simple average / majority vote | Weighted sum (each tree contributes partially) |
| Main effect | Reduces variance | Reduces bias (and variance with learning rate) |
| Risk of overfitting | Low | Moderate (can overfit if too many rounds) |

## Key concepts

### Learning Rate (Shrinkage)

The learning rate $\nu$ controls how much each new tree contributes:

$$F_m(x) = F_{m-1}(x) + \nu \cdot h_m(x)$$

- **Small $\nu$** (0.01-0.1): Each tree makes a tiny correction. Requires more trees ($M$) but generalizes better. This is the recommended approach.
- **Large $\nu$** (0.3-1.0): Each tree makes a big correction. Fewer trees needed but higher risk of overfitting.

**The tradeoff**: $\nu$ and $M$ are inversely related. A smaller learning rate with more trees almost always outperforms a larger learning rate with fewer trees, at the cost of more computation.

**Analogy**: Walking toward a target. A large learning rate is like taking big steps -- you get there fast but might overshoot. A small learning rate is like shuffling forward -- slower but more precise.

### Loss Functions

The choice of loss function determines what the boosting algorithm optimizes:

| Task | Loss Function | Pseudo-Residual |
|------|--------------|----------------|
| Regression | Squared error: $\frac{1}{2}(y-F)^2$ | $y - F$ (residual) |
| Regression | Absolute error: $\|y-F\|$ | $\text{sign}(y - F)$ |
| Regression | Huber loss | Blends squared and absolute |
| Classification | Log-loss (cross-entropy) | $y - p$ where $p$ is predicted probability |
| Ranking | Pairwise ranking loss | Gradient of ranking metric |

### Regularization in Boosting

Boosting has several mechanisms to prevent overfitting:

1. **Learning rate** ($\nu$): Shrinks each tree's contribution.
2. **Number of trees** ($M$): Stop adding trees when validation error starts increasing.
3. **Tree constraints**: Limit `max_depth` (typically 3-8), `min_samples_leaf`, `max_features`.
4. **Subsampling (Stochastic Gradient Boosting)**: Use only a fraction (e.g., 50-80%) of the training data for each tree. Introduced by Friedman (1999). Reduces overfitting AND speeds up training.
5. **Early stopping**: Monitor performance on a validation set and stop when it stops improving.

### Early Stopping

The most important practical regularization technique:

1. Set `n_estimators` to a large number (e.g., 5000).
2. Provide a validation set.
3. Monitor the validation loss after each tree is added.
4. Stop when the validation loss has not improved for `early_stopping_rounds` consecutive rounds.

This automatically finds the optimal number of trees without manual tuning.

### Bias-Variance Dynamics in Boosting

As you add more trees:

| Phase | Bias | Variance | Total Error |
|-------|------|----------|-------------|
| Few trees | High | Low | High (underfitting) |
| Moderate trees | Decreasing | Slightly increasing | Decreasing (sweet spot) |
| Too many trees | Very low | High | Increasing (overfitting) |

The learning rate slows this progression. A smaller learning rate pushes the overfitting point further out, giving you more room to add trees.

## Python implementation

```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np

# --- AdaBoost ---
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # Decision stumps
    n_estimators=200,
    learning_rate=0.1,
    random_state=42
)
ada.fit(X_train, y_train)
print(f"AdaBoost Test Accuracy: {ada.score(X_test, y_test):.4f}")

# Training curve (error vs. number of estimators)
ada_train_errors = []
ada_test_errors = []
for y_pred_train, y_pred_test in zip(
    ada.staged_predict(X_train), ada.staged_predict(X_test)
):
    ada_train_errors.append(1 - np.mean(y_pred_train == y_train))
    ada_test_errors.append(1 - np.mean(y_pred_test == y_test))

plt.plot(ada_train_errors, label='Train')
plt.plot(ada_test_errors, label='Test')
plt.xlabel('Boosting Round'); plt.ylabel('Error Rate')
plt.legend(); plt.title('AdaBoost Training Curve')

# --- Gradient Boosting Classifier ---
gb_clf = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=10,
    subsample=0.8,           # Stochastic gradient boosting
    validation_fraction=0.1,  # Hold out 10% for early stopping
    n_iter_no_change=20,     # Early stopping patience
    random_state=42
)
gb_clf.fit(X_train, y_train)
print(f"Best n_estimators: {gb_clf.n_estimators_}")
print(f"Test Accuracy: {gb_clf.score(X_test, y_test):.4f}")

# --- Gradient Boosting Regressor ---
gb_reg = GradientBoostingRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    min_samples_leaf=10,
    subsample=0.8,
    loss='huber',             # Robust to outliers
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)
gb_reg.fit(X_train, y_train)

# Feature importance
feat_imp = pd.Series(gb_clf.feature_importances_,
                      index=feature_names).sort_values(ascending=False)
feat_imp.head(15).plot(kind='barh')
plt.title('Gradient Boosting Feature Importance')

# Partial dependence plots
from sklearn.inspection import PartialDependenceDisplay
fig, ax = plt.subplots(figsize=(12, 4))
PartialDependenceDisplay.from_estimator(gb_clf, X_train,
                                         features=[0, 1, (0, 1)], ax=ax)
```

## R implementation (if applicable)

```r
library(gbm)
library(adabag)

# AdaBoost
library(adabag)
ada_model <- boosting(target ~ ., data = train_data, boos = TRUE,
                      mfinal = 200, coeflearn = "Freund")

# Gradient Boosting
gb_model <- gbm(
  target ~ ., data = train_data,
  distribution = "bernoulli",  # or "gaussian" for regression
  n.trees = 1000,
  interaction.depth = 4,
  shrinkage = 0.05,
  bag.fraction = 0.8,
  cv.folds = 5
)

# Optimal number of trees
best_trees <- gbm.perf(gb_model, method = "cv")
print(best_trees)

# Variable importance
summary(gb_model, n.trees = best_trees)

# Prediction
predictions <- predict(gb_model, newdata = test_data, n.trees = best_trees,
                        type = "response")
```

## Gotchas & lessons learned

1. **Boosting CAN overfit (unlike Random Forest).** Always use early stopping or cross-validation to determine the optimal number of trees. Watching the training error alone is misleading -- it will keep decreasing even as test error increases.

2. **Low learning rate + many trees > high learning rate + few trees.** This is one of the most robust findings in boosting. Set the learning rate to 0.01-0.1 and let early stopping find the right number of trees.

3. **Tree depth should be shallow.** Unlike Random Forest (which uses deep trees), boosting works best with shallow trees (depth 3-8). Each tree is meant to capture a small piece of the pattern. Deep trees in boosting overfit quickly.

4. **Stochastic gradient boosting (subsample < 1.0) almost always helps.** Using 50-80% of the data for each tree reduces overfitting and speeds up training, similar to how dropout works in neural networks.

5. **Boosting is sensitive to noisy labels.** Because boosting focuses on hard-to-classify observations, mislabeled data points get increasingly high weight, pulling the model toward noise. Clean your labels before boosting.

6. **AdaBoost is a special case of gradient boosting with exponential loss.** Understanding this connection helps you see why gradient boosting is more general and flexible.

7. **Feature importance from boosting can be misleading.** Like Random Forest, impurity-based importance is biased toward high-cardinality features. Use permutation importance or SHAP values for more reliable results.

8. **Boosting does not extrapolate.** Like all tree-based methods, boosting cannot predict values outside the range of the training data. If your test data has feature values beyond the training range, predictions at those extremes will be capped.

9. **For actuarial applications**: Gradient boosting models often outperform GLMs in predictive accuracy but are harder to explain to regulators. A common workflow is to use boosting to identify important features and interactions, then encode those insights into a more interpretable GLM.

10. **The interaction between `max_depth` and `learning_rate` matters.** Deeper trees capture more complex interactions per step, so they need a lower learning rate to avoid overfitting. A common starting point: `max_depth=3, learning_rate=0.1` or `max_depth=6, learning_rate=0.05`.

## Resources

- **Friedman, "Greedy Function Approximation: A Gradient Boosting Machine" (2001)** -- The foundational gradient boosting paper. Introduces stochastic gradient boosting and partial dependence plots.
- **Freund & Schapire, "A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting" (1997)** -- The AdaBoost paper.
- **scikit-learn Ensemble documentation**: [Gradient Tree Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)
- **An Introduction to Statistical Learning (ISLR)**, Chapter 8.2.3 -- Clear boosting introduction.
- **The Elements of Statistical Learning (ESL)**, Chapter 10 -- Rigorous treatment of boosting and gradient boosting.
- **StatQuest**: Josh Starmer's YouTube videos on AdaBoost and Gradient Boost are outstanding visual explanations.
- **Natekin & Knoll, "Gradient Boosting Machines, A Tutorial" (2013)** -- Accessible survey paper.
