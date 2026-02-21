# Model Evaluation

## What it is

Model evaluation is the systematic process of measuring how well a predictive model performs. It answers the fundamental question: "Can I trust this model's predictions?" Different metrics illuminate different aspects of performance -- accuracy, discrimination, calibration, ranking ability, and error magnitude. Choosing the right evaluation metric is just as important as choosing the right algorithm, because a metric defines what "good" means for your model. A model that maximizes one metric might perform poorly on another, and the right choice depends entirely on the business problem.

## Why it matters

For actuarial and data science practitioners, model evaluation is not optional -- it is the foundation of model governance. In insurance, a pricing model that looks great on one metric but fails on another can lead to adverse selection, regulatory scrutiny, or financial loss. Actuaries are also uniquely positioned to understand evaluation metrics because many have deep roots in actuarial science: the Gini coefficient used in insurance pricing is closely related to the AUC from machine learning, and concepts like lift charts have direct analogs in traditional actuarial practice.

Understanding evaluation metrics also protects you from common traps. A model with 99% accuracy sounds impressive until you realize that simply predicting "no claim" for every policy would achieve the same accuracy when only 1% of policies have claims. The right metric exposes this.

## When to use it

- **After training any predictive model** -- always evaluate before drawing conclusions.
- **When comparing competing models** -- metrics provide an objective basis for selection.
- **When tuning hyperparameters** -- the metric you optimize during tuning defines what the model learns.
- **When deciding on a classification threshold** -- different thresholds produce different precision/recall tradeoffs.
- **When presenting model results to stakeholders** -- choose metrics that align with business objectives.
- **When monitoring deployed models** -- track metrics over time to detect model degradation.
- **When dealing with imbalanced classes** -- standard accuracy is misleading; you need better metrics.

## How it works (intuitive)

### The confusion matrix: the foundation

For binary classification (e.g., claim vs no claim, fraud vs not fraud), every prediction falls into one of four categories:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actually Positive** | True Positive (TP) | False Negative (FN) |
| **Actually Negative** | False Positive (FP) | True Negative (TN) |

*Analogy*: Think of a smoke detector. A true positive is when it correctly detects a fire. A false positive is a false alarm (no fire, but the alarm went off). A false negative is the dangerous case: there is a fire, but the alarm stayed silent. A true negative is a quiet night with no fire and no alarm.

From this matrix, most classification metrics are derived.

### Accuracy (and why it is often misleading)

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

The proportion of correct predictions. Simple and intuitive, but deeply flawed for imbalanced datasets. If 99% of policies have no claim, predicting "no claim" for everyone gives 99% accuracy while being completely useless for identifying claims.

**Rule of thumb**: Only use accuracy when classes are roughly balanced. In most actuarial applications, they are not.

### Precision and recall

**Precision** (Positive Predictive Value): Of all the cases I predicted as positive, how many actually were?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall** (Sensitivity, True Positive Rate): Of all the actual positives, how many did I catch?

$$\text{Recall} = \frac{TP}{TP + FN}$$

*Analogy*: Precision is about quality -- "when I say yes, how often am I right?" Recall is about completeness -- "of all the actual yeses, how many did I find?"

There is almost always a tradeoff between the two. A fraud detection model that flags every transaction as fraudulent has perfect recall (catches all fraud) but terrible precision (overwhelms investigators with false alarms).

### The precision-recall curve

Plot precision vs recall at every possible classification threshold. The curve shows the tradeoff between catching more positives (higher recall) and maintaining prediction quality (higher precision).

- A model with a curve that hugs the top-right corner is excellent.
- The area under the precision-recall curve (PR-AUC) summarizes performance. Random model: PR-AUC equals the prevalence of the positive class.
- **Use PR curves when the positive class is rare** (fraud, large claims). They are more informative than ROC curves in imbalanced settings.

### ROC curve and AUC

The Receiver Operating Characteristic (ROC) curve plots the True Positive Rate (recall) against the False Positive Rate at every threshold:

$$\text{TPR} = \frac{TP}{TP + FN}, \quad \text{FPR} = \frac{FP}{FP + TN}$$

The Area Under the ROC Curve (AUC) ranges from 0 to 1:
- **AUC = 0.5**: Random guessing (the diagonal line). The model has no discriminative ability.
- **AUC = 1.0**: Perfect discrimination. The model perfectly separates positives from negatives.
- **AUC = 0.7-0.8**: Acceptable discrimination for many applications.
- **AUC = 0.8-0.9**: Good discrimination.
- **AUC > 0.9**: Excellent (and sometimes suspiciously good -- check for data leakage).

*Intuition*: AUC equals the probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example. If you pick a random fraudulent claim and a random non-fraudulent claim, AUC is the probability that the model assigns a higher fraud score to the fraudulent one.

### Gini coefficient: the actuarial version

The Gini coefficient is directly related to AUC:

$$\text{Gini} = 2 \times \text{AUC} - 1$$

- Gini = 0: No discrimination (random model).
- Gini = 1: Perfect discrimination.

**Important: Actuarial Gini vs ML Gini**

In actuarial science, the "Gini coefficient" typically refers to this discrimination metric for models (also called the Somers' D or the area between the ROC curve and the diagonal).

In machine learning and decision tree literature, "Gini" often refers to the Gini impurity used to measure node purity in decision trees:

$$\text{Gini impurity} = 1 - \sum_{k=1}^{K} p_k^2$$

These are completely different concepts that happen to share a name. When someone in an actuarial context says "the model has a Gini of 0.45," they mean $2 \times \text{AUC} - 1 = 0.45$, so AUC = 0.725. When someone building a random forest says "Gini," they mean node impurity. Always clarify which is meant.

### Lift and gain charts

**Lift chart**: If I use the model to select the top 10% riskiest policies, how much more likely are they to have a claim compared to a random 10% sample? Lift = (percentage of positives captured) / (percentage of population selected).

**Cumulative gains chart**: Sort all observations by predicted probability (highest first). Plot the percentage of population examined (x-axis) against the percentage of total positives captured (y-axis).

- A perfect model would capture all positives in the smallest possible slice of the population.
- A random model follows the diagonal (examining 20% of the population captures 20% of positives).
- The gap between your model's curve and the diagonal shows the model's added value.

*Actuarial use*: Lift charts are invaluable for explaining model value to business stakeholders. "By targeting the top 20% riskiest policies identified by our model, we capture 65% of all claims" is a powerful statement.

### Log loss (cross-entropy)

For probabilistic predictions, log loss measures how close the predicted probabilities are to the true outcomes:

$$\text{Log Loss} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]$$

- Log loss heavily penalizes confident wrong predictions. Predicting 0.99 for an event that does not happen is much worse than predicting 0.6.
- Lower is better. Perfect predictions give log loss = 0.
- **Use log loss when you care about the quality of predicted probabilities**, not just ranking. This is critical in insurance pricing where predicted probabilities directly become premium components.

### RMSE and MAE (regression metrics)

For regression problems (predicting claim amounts, premiums, etc.):

**Root Mean Squared Error (RMSE)**:

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$

**Mean Absolute Error (MAE)**:

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Key difference**: RMSE penalizes large errors more heavily (because errors are squared). MAE treats all errors linearly.

*Analogy*: If you are predicting insurance claim amounts, RMSE will strongly penalize the cases where you predicted $1,000 but the actual claim was $100,000. MAE will penalize that case too, but proportionally rather than quadratically.

- **Use RMSE** when large errors are particularly costly (you really want to avoid being way off on any single prediction).
- **Use MAE** when you want a metric that is robust to outliers and easier to interpret ("on average, we are off by $X").

### Threshold selection

Classification models typically output probabilities. You must choose a threshold to convert probabilities into binary predictions. The default threshold of 0.5 is rarely optimal.

**Approaches to threshold selection**:
1. **Maximize F1-score**: Find the threshold that maximizes the harmonic mean of precision and recall.
2. **Youden's J statistic**: Maximize $\text{TPR} - \text{FPR}$ (the point farthest from the ROC diagonal).
3. **Cost-based**: If you know the relative cost of false positives vs false negatives, choose the threshold that minimizes expected cost.
4. **Business constraint**: "We can only investigate 100 cases per month" -- set the threshold that selects approximately 100 cases.

## Key concepts

### Metrics summary table

| Metric | Type | Range | Best Value | Use When |
|---|---|---|---|---|
| Accuracy | Classification | [0, 1] | 1 | Balanced classes only |
| Precision | Classification | [0, 1] | 1 | Cost of false positives is high |
| Recall | Classification | [0, 1] | 1 | Cost of false negatives is high |
| F1-score | Classification | [0, 1] | 1 | Need balance of precision and recall |
| AUC-ROC | Classification | [0, 1] | 1 | General discrimination ability |
| Gini | Classification | [0, 1] | 1 | Actuarial model comparison |
| PR-AUC | Classification | [prev, 1] | 1 | Imbalanced classes |
| Log loss | Classification | [0, inf) | 0 | Probability quality matters |
| RMSE | Regression | [0, inf) | 0 | Large errors are costly |
| MAE | Regression | [0, inf) | 0 | Robust to outliers |

### The F1-score

The harmonic mean of precision and recall:

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN}$$

More generally, the $F_\beta$ score weights recall $\beta$ times as much as precision:

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

### Class imbalance effects on metrics

| Metric | Robust to imbalance? | Notes |
|---|---|---|
| Accuracy | No | Dominated by majority class |
| Precision | Partially | Still meaningful but sensitive to threshold |
| Recall | Partially | Still meaningful |
| AUC-ROC | Partially | Can look optimistic with severe imbalance |
| PR-AUC | Yes | Directly reflects rare-event detection |
| Log loss | Partially | Can be dominated by majority class predictions |

## Python implementation

```python
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    log_loss, f1_score,
    mean_squared_error, mean_absolute_error
)
import matplotlib.pyplot as plt

# --- Confusion matrix ---
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))

# --- ROC curve and AUC ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
gini = 2 * auc - 1

fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'Model (AUC={auc:.3f}, Gini={gini:.3f})')
ax.plot([0, 1], [0, 1], 'k--', label='Random')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()

# --- Precision-Recall curve ---
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

fig, ax = plt.subplots()
ax.plot(recall, precision, label=f'Model (PR-AUC={pr_auc:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curve')
ax.legend()

# --- Log loss ---
ll = log_loss(y_test, y_prob)
print(f"Log loss: {ll:.4f}")

# --- Optimal threshold (Youden's J) ---
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_threshold = thresholds[best_idx]
print(f"Optimal threshold (Youden's J): {best_threshold:.3f}")

# --- Lift chart ---
def plot_cumulative_gains(y_true, y_prob, ax=None):
    """Plot cumulative gains chart."""
    if ax is None:
        fig, ax = plt.subplots()
    # Sort by predicted probability descending
    order = np.argsort(-y_prob)
    y_sorted = np.array(y_true)[order]
    cumulative_gains = np.cumsum(y_sorted) / y_sorted.sum()
    population_pct = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    ax.plot(population_pct, cumulative_gains, label='Model')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('Proportion of population')
    ax.set_ylabel('Proportion of positives captured')
    ax.set_title('Cumulative Gains Chart')
    ax.legend()
    return ax

# --- Regression metrics ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
mae = mean_absolute_error(y_test, y_pred_reg)
print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")
```

## R implementation (if applicable)

```r
library(pROC)
library(caret)

# --- ROC and AUC ---
roc_obj <- roc(y_test, y_prob)
auc_val <- auc(roc_obj)
gini_val <- 2 * auc_val - 1
plot(roc_obj, main = paste("AUC =", round(auc_val, 3)))

# --- Confusion matrix with many metrics ---
cm <- confusionMatrix(factor(y_pred), factor(y_test), positive = "1")
print(cm)

# --- Lift chart ---
library(ROCR)
pred_obj <- prediction(y_prob, y_test)
perf_lift <- performance(pred_obj, "tpr", "rpp")  # cumulative gains
plot(perf_lift, main = "Cumulative Gains Chart")
abline(0, 1, lty = 2)
```

## Gotchas & lessons learned

1. **Never use accuracy alone for imbalanced problems**: In insurance, claim rates are typically 5-15%. A model that predicts "no claim" for everyone achieves 85-95% accuracy. Always use AUC, Gini, precision-recall, or log loss for imbalanced classification.

2. **AUC can be misleadingly high with severe imbalance**: With 0.1% positive rate, even mediocre models can achieve AUC > 0.9 because the false positive rate stays very low across most thresholds. PR-AUC gives a more honest picture in extreme imbalance.

3. **ROC AUC is threshold-independent**: AUC measures discrimination across all thresholds. If you care about performance at a specific operating point, look at precision/recall/F1 at that threshold instead.

4. **Gini naming confusion**: Always clarify whether "Gini" means the discrimination metric (2*AUC - 1) or the tree impurity measure. In an actuarial context, it almost always means the former. In an ML engineering context, it might mean either.

5. **Log loss penalizes overconfidence**: If your model predicts 0.001 probability for an event that actually happens, the log loss penalty for that single observation is enormous (-log(0.001) = 6.9). This is why model calibration matters when you evaluate with log loss.

6. **RMSE vs MAE depends on your cost function**: If the business cost of error grows quadratically with error magnitude, use RMSE. If it grows linearly, use MAE. In insurance claim amount prediction, large claims (tail risk) often have disproportionate business impact, which argues for RMSE or even quantile-based metrics.

7. **Always report multiple metrics**: No single metric tells the whole story. Report at least: a discrimination metric (AUC/Gini), a calibration metric (log loss or Brier score), and a threshold-dependent metric (F1 at your chosen threshold).

8. **Test set metrics only**: Final performance numbers must come from a held-out test set that was never used for training or hyperparameter tuning. Cross-validation metrics are for model selection; test set metrics are for reporting.

9. **Stratified sampling for evaluation**: When splitting data into train/test, use stratified sampling to ensure the positive class proportion is similar in both sets. This is especially important with rare events.

10. **Threshold selection is a business decision**: The "optimal" threshold depends on the relative costs of false positives and false negatives. In fraud detection, missing a $50,000 fraud (false negative) is much worse than investigating a $200 legitimate transaction (false positive). Encode these costs explicitly.

## Resources

- **scikit-learn metrics documentation**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **An Introduction to Statistical Learning (ISLR), Chapter 4.4**: Classification metrics and the confusion matrix.
- **The Relationship Between Precision-Recall and ROC Curves (Davis & Goadrich, 2006)**: Why PR curves matter for imbalanced data.
- **CAS Monograph on Predictive Modeling**: Covers Gini, lift charts, and double-lift charts in actuarial context.
- **Google's ML Crash Course -- Classification metrics**: https://developers.google.com/machine-learning/crash-course/classification
- **pROC R package**: https://cran.r-project.org/web/packages/pROC/ -- comprehensive ROC analysis in R.
