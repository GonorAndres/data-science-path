# Classification (Logistic Regression, Decision Trees, SVM, KNN)

## What it is

Classification is a supervised learning task where the goal is to assign observations to **discrete categories** (classes) based on input features. Unlike regression which predicts a number, classification predicts a label -- "will the customer churn or not?", "is this claim fraudulent?", "which risk tier does this policyholder belong to?". This README covers four foundational classifiers: **Logistic Regression** (a linear model that outputs probabilities), **Decision Trees** (rule-based splits), **Support Vector Machines** (maximum-margin separators), and **K-Nearest Neighbors** (instance-based voting). Each draws a different kind of **decision boundary** through feature space, and understanding their strengths and weaknesses is essential for choosing the right model for a given problem.

## Why it matters

Classification problems are everywhere in actuarial work and data science: fraud detection, underwriting decisions, customer churn prediction, claim triage, credit scoring, and medical diagnosis. As a practitioner, you need to understand not just whether a model classifies correctly, but **how confident** it is (probability calibration), **where** it makes errors (confusion matrix), and **why** it draws the boundaries it does (interpretability). Logistic regression, in particular, is a regulatory favorite because of its transparency, while tree-based methods serve as the building blocks for the powerful ensemble methods (Random Forest, XGBoost) covered elsewhere in this repository.

## When to use it

- **Logistic Regression**: Binary or multi-class classification when you need interpretable coefficients, probability estimates, or when the relationship between features and log-odds is approximately linear. Regulatory environments (insurance pricing, credit scoring) often mandate interpretable models.
- **Decision Trees**: When you need a highly interpretable model that non-technical stakeholders can visualize and understand. Also useful for exploring data and understanding feature interactions before building more complex models.
- **Support Vector Machines (SVM)**: When you have a moderate-sized dataset, possibly with many features, and you want a powerful classifier that can capture non-linear boundaries via kernels. Less common in production actuarial work but excellent for text classification and image-related tasks.
- **K-Nearest Neighbors (KNN)**: When the decision boundary is complex and irregular, when you have a moderate-sized dataset, or when you want a simple non-parametric baseline. Works well when similar observations cluster together in feature space.

## How it works (intuitive)

### Logistic Regression

Logistic regression is the classification analog of linear regression. Instead of predicting a continuous number, it predicts the **probability** that an observation belongs to a particular class.

**The key insight**: You cannot directly use a linear equation to predict probabilities because a line extends from $-\infty$ to $+\infty$, while probabilities must stay between 0 and 1. The solution is to model the **log-odds** (logit) instead:

$$\ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x_1 + \ldots + \beta_p x_p$$

Then invert to get the probability:

$$p = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \ldots + \beta_p x_p)}}$$

This is the **sigmoid function**, which squashes any real number into the (0, 1) range.

**Analogy**: Think of the linear part $\beta_0 + \beta_1 x_1 + \ldots$ as a "score" that can be any number. The sigmoid function is like a dimmer switch that converts this score to a probability: very negative scores get mapped near 0, very positive scores near 1, and scores near zero get mapped near 0.5.

**Fitting**: Instead of minimizing squared errors (like OLS), logistic regression maximizes the **log-likelihood** -- it finds coefficients that make the observed class labels most probable. This is equivalent to minimizing **binary cross-entropy loss**.

**Decision boundary**: Logistic regression draws a linear decision boundary (a straight line in 2D, a plane in 3D, a hyperplane in higher dimensions). Points on one side are classified as class 0, points on the other side as class 1.

### Decision Trees

A decision tree classifies observations by asking a sequence of yes/no questions about the features, creating a flowchart-like structure.

**Step by step:**

1. Start with all data at the root node.
2. Find the feature and threshold that **best separates** the classes (e.g., "Is age > 35?").
3. Split the data into two groups based on this question.
4. Repeat recursively for each group until a stopping criterion is met (max depth, minimum samples per leaf, etc.).
5. Each leaf node assigns the majority class of the observations that ended up there.

**How does the tree decide the "best" split?** It uses an impurity measure:

- **Gini Impurity**: $G = 1 - \sum_{k} p_k^2$, where $p_k$ is the proportion of class $k$ in the node. A pure node (all one class) has Gini = 0.
- **Entropy**: $H = -\sum_{k} p_k \log_2(p_k)$. Also measures impurity. The tree picks the split that produces the largest **reduction** in impurity (information gain).

**Analogy**: Imagine sorting a mixed pile of red and blue balls into bins. Each split is like a sieve that separates balls based on some property (weight, size). A good sieve creates bins that are mostly one color. You keep sieving until each bin is nearly pure.

**Decision boundary**: Trees create rectangular, axis-aligned decision boundaries. Each split is parallel to one feature axis.

### Support Vector Machines (SVM)

SVM finds the hyperplane that **maximizes the margin** between the two classes -- it is not just any separating line, but the one with the widest possible buffer zone.

**Step by step:**

1. In 2D, imagine two classes of points that can be separated by a line. There are infinitely many possible separating lines.
2. SVM picks the line that is **farthest from the nearest points of each class**. These nearest points are called **support vectors**.
3. The distance from the line to the nearest support vector on either side is the **margin**. Maximizing this margin leads to better generalization.

**What if the data is not linearly separable?** Two strategies:

- **Soft-margin SVM**: Allow some points to be on the wrong side of the boundary, controlled by the parameter $C$. High $C$ = fewer violations (tighter fit, risk of overfitting). Low $C$ = more violations allowed (simpler boundary, risk of underfitting).
- **Kernel trick**: Project the data into a higher-dimensional space where it IS linearly separable. Common kernels:
  - **Linear**: No transformation.
  - **RBF (Radial Basis Function)**: Creates non-linear, blob-like decision boundaries. Controlled by the parameter $\gamma$ (how far the influence of a single training example reaches).
  - **Polynomial**: Creates polynomial decision boundaries.

**Analogy**: Imagine two groups of marbles on a table that you cannot separate with a straight ruler. If you could lift the table and let gravity pull the marbles into different levels, a flat cut through the air might separate them. The kernel trick does this mathematically -- lifting the data into higher dimensions where a flat separator works.

### K-Nearest Neighbors (KNN)

KNN is the simplest classifier conceptually: to classify a new point, look at its $k$ nearest neighbors in the training data and take a majority vote.

**Step by step:**

1. Choose $k$ (e.g., 5).
2. For a new observation, compute the distance to every training point.
3. Find the $k$ closest training points.
4. Assign the class that the majority of those $k$ neighbors belong to.

**Distance metrics**: The most common is Euclidean distance, but Manhattan distance, Minkowski distance, or domain-specific metrics are also used.

**Analogy**: You move to a new neighborhood and want to know if it is "urban" or "suburban." You ask your 5 nearest neighbors. If 4 say "urban," you classify your area as urban.

**Decision boundary**: KNN creates complex, non-linear decision boundaries that closely follow the data distribution. The boundary becomes smoother as $k$ increases.

## Key concepts

### Confusion Matrix and Metrics

For a binary classifier, the confusion matrix is:

|  | Predicted Positive | Predicted Negative |
|--|-------------------|--------------------|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

Key metrics derived from it:

| Metric | Formula | Intuition |
|--------|---------|-----------|
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall correctness (misleading with imbalanced classes) |
| Precision | $\frac{TP}{TP + FP}$ | Of those predicted positive, how many are correct? |
| Recall (Sensitivity) | $\frac{TP}{TP + FN}$ | Of the actual positives, how many did we catch? |
| Specificity | $\frac{TN}{TN + FP}$ | Of the actual negatives, how many did we correctly identify? |
| F1 Score | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | Harmonic mean of precision and recall |
| AUC-ROC | Area under the ROC curve | Probability that the model ranks a random positive higher than a random negative |

### Probability Calibration

Many classifiers output scores that are not true probabilities. Calibration ensures that when a model says "70% chance of churn," roughly 70% of such predictions actually churn.

- **Logistic Regression** is generally well-calibrated by nature.
- **Decision Trees** tend to output extreme probabilities (near 0 or 1) and are often poorly calibrated.
- **SVM** outputs are not probabilities at all (use Platt scaling to convert).
- **KNN** outputs can be interpreted as probabilities (proportion of neighbors in each class) but can be noisy.

Calibration methods:
- **Platt Scaling**: Fit a logistic regression on the model's output scores.
- **Isotonic Regression**: Non-parametric calibration. More flexible but needs more data.

Use `sklearn.calibration.CalibratedClassifierCV` or plot calibration curves with `sklearn.calibration.calibration_curve`.

### Multi-Class Strategies

When you have more than two classes:

- **One-vs-Rest (OvR)**: Train one binary classifier per class (class k vs. all others). Predict the class with the highest confidence. Default for logistic regression and SVM in scikit-learn.
- **One-vs-One (OvO)**: Train one classifier for every pair of classes. $\binom{K}{2}$ classifiers. Predict by majority vote. Default for SVM with non-linear kernels.
- **Softmax (Multinomial)**: Extend logistic regression to output a probability distribution over all classes simultaneously. More principled than OvR.

### Decision Boundary Comparison

| Classifier | Boundary Shape | Flexibility |
|-----------|---------------|-------------|
| Logistic Regression | Linear (hyperplane) | Low -- can add polynomial features for non-linearity |
| Decision Tree | Axis-aligned rectangles | Medium -- captures interactions naturally |
| SVM (linear) | Linear (hyperplane) | Low |
| SVM (RBF) | Smooth, non-linear blobs | High -- controlled by $\gamma$ and $C$ |
| KNN | Irregular, data-following | High -- controlled by $k$ |

### Key Hyperparameters

| Classifier | Parameter | Effect |
|-----------|-----------|--------|
| Logistic Regression | $C$ (inverse regularization) | Higher C = less regularization, more complex boundary |
| Decision Tree | `max_depth` | Deeper = more complex, risk of overfitting |
| Decision Tree | `min_samples_leaf` | Higher = simpler, more stable predictions |
| SVM | $C$ | Higher C = harder margin, fits training data more tightly |
| SVM (RBF) | $\gamma$ | Higher gamma = tighter, more complex boundary |
| KNN | $k$ | Higher k = smoother boundary, lower variance |

## Python implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

# Always scale for SVM and KNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(C=1.0, penalty='l2', solver='lbfgs', max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
dt.fit(X_train, y_train)  # Trees do not need scaling
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=feature_names, class_names=class_names, filled=True)

# SVM with RBF kernel
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train_scaled, y_train)

# KNN
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train_scaled, y_train)

# Evaluate all
for name, model, X_eval in [('LR', lr, X_test_scaled),
                              ('DT', dt, X_test),
                              ('SVM', svm, X_test_scaled),
                              ('KNN', knn, X_test_scaled)]:
    y_pred = model.predict(X_eval)
    print(f"\n{name}:")
    print(classification_report(y_test, y_pred))
```

## R implementation (if applicable)

```r
# Logistic Regression
glm_model <- glm(target ~ ., data = train, family = binomial)
summary(glm_model)
pred_prob <- predict(glm_model, newdata = test, type = "response")

# Decision Tree (rpart)
library(rpart)
library(rpart.plot)
tree_model <- rpart(target ~ ., data = train, method = "class",
                    control = rpart.control(maxdepth = 5, minsplit = 20))
rpart.plot(tree_model)

# SVM (e1071)
library(e1071)
svm_model <- svm(target ~ ., data = train, kernel = "radial",
                 cost = 1, gamma = 0.1, probability = TRUE)

# KNN (class)
library(class)
knn_pred <- knn(train_features, test_features, train_labels, k = 5)
```

## Gotchas & lessons learned

1. **Do not use accuracy with imbalanced classes.** If 95% of claims are non-fraudulent, a model that always predicts "not fraud" has 95% accuracy but is useless. Use precision, recall, F1, and AUC-ROC instead.

2. **Always scale features for SVM and KNN.** These methods use distance calculations. If one feature ranges from 0 to 1 and another from 0 to 100,000, the second feature dominates. Decision trees do not need scaling.

3. **Decision trees overfit easily.** An unpruned tree will memorize the training data. Always set `max_depth`, `min_samples_leaf`, or use cost-complexity pruning (`ccp_alpha`).

4. **SVM does not natively output probabilities.** Setting `probability=True` in scikit-learn enables Platt scaling, but this adds computation and can be slow. If you only need the class label (not probability), leave it off.

5. **KNN is slow at prediction time.** For each new point, it must compute distances to ALL training points. With large datasets, use approximate nearest neighbor methods (e.g., `FAISS`, `Annoy`, or `KDTree`/`BallTree` in scikit-learn).

6. **Check class balance in each fold during cross-validation.** Use `StratifiedKFold` (the default in scikit-learn for classifiers) to ensure each fold has roughly the same class distribution.

7. **Logistic regression coefficients are log-odds ratios.** A coefficient of 0.5 means a one-unit increase in that feature increases the log-odds by 0.5, or equivalently multiplies the odds by $e^{0.5} \approx 1.65$. This is a different interpretation than linear regression coefficients.

8. **The "curse of dimensionality" kills KNN.** In high-dimensional spaces, all points become approximately equidistant, making the notion of "nearest neighbor" meaningless. Use dimensionality reduction (PCA) before applying KNN in high-dimensional settings.

9. **Probability calibration matters for business decisions.** If you use predicted probabilities to set thresholds (e.g., "flag claims with >30% fraud probability for review"), poorly calibrated probabilities lead to wrong threshold decisions. Always check calibration plots.

10. **For actuarial applications**, logistic regression often wins on interpretability grounds. You can explain to a regulator or stakeholder exactly how each factor contributes to the prediction. Trees are useful for initial exploration, and SVMs/KNN are better suited for internal prediction tasks where interpretability is secondary.

## Resources

- **scikit-learn documentation**: [Supervised Learning](https://scikit-learn.org/stable/supervised_learning.html)
- **An Introduction to Statistical Learning (ISLR)**, Chapters 4 (Classification), 8 (Trees), 9 (SVM) -- Free PDF available.
- **Stanford CS229 Lecture Notes**: Covers logistic regression, SVM, and kernels with clear mathematical exposition.
- **StatQuest YouTube channel**: Josh Starmer's visual explanations of logistic regression, decision trees, SVM, and KNN are outstanding for building intuition.
- **scikit-learn Calibration tutorial**: [Probability Calibration](https://scikit-learn.org/stable/modules/calibration.html)
- **Actuarial applications**: SOA Predictive Analytics resources discuss classification in insurance contexts.
