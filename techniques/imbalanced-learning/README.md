# Imbalanced Learning

## What it is

Imbalanced learning refers to the collection of techniques used when your classification target has a highly skewed distribution -- one class vastly outnumbers the other(s). In a fraud detection dataset, maybe 0.5% of transactions are fraudulent. In a lapse prediction model, maybe 3% of policyholders lapse. Standard machine learning algorithms are designed to minimize overall error, which means they can achieve 99.5% accuracy by simply predicting "not fraud" for every single transaction -- and be completely useless. Imbalanced learning techniques modify the data, the algorithm, or the evaluation process so that the minority class gets the attention it deserves.

## Why it matters

Class imbalance is the norm, not the exception, in actuarial and insurance data science:

- **Fraud detection**: Fraudulent claims are rare (often <1% of total claims), but catching them is the entire point of the model.
- **Lapse/churn prediction**: Most policyholders renew; the ones who do not are the signal you are trying to capture.
- **Catastrophe/extreme event modeling**: Rare events are rare by definition, but they drive the tail risk that actuaries care most about.
- **Claim litigation prediction**: Only a small fraction of claims end up in litigation, but those claims are disproportionately expensive.
- **Underwriting referral**: Most applications sail through automated underwriting; the ones flagged for manual review are the minority class.

If you do not handle imbalance, your model will look great on accuracy but fail at the task that matters -- identifying the rare, high-impact events.

## When to use it

- Your target variable has a minority class representing less than ~10-20% of observations, and the minority class is the one you care about detecting.
- Your model achieves high overall accuracy but terrible recall on the minority class.
- You are building a fraud, churn, default, or rare-event prediction model.
- Precision-recall tradeoffs are more relevant than raw accuracy in your business context.
- Your model's confusion matrix reveals it is barely predicting the positive class at all.
- You need to tune the balance between false positives and false negatives to match business costs (e.g., the cost of investigating a false fraud alert vs. the cost of missing real fraud).

## How it works (intuitive)

### Why standard models fail on imbalanced data

Imagine you are training a model to identify counterfeit bills in a stack of 10,000 bills, where only 50 are counterfeit. A lazy model that says "every bill is genuine" gets 99.5% accuracy. The loss function (e.g., cross-entropy) does not penalize this lazy model much because the vast majority of predictions are correct.

The model is not "broken" -- it is doing exactly what it was told: minimize overall errors. The problem is that we did not tell it what we actually want: find the counterfeits, even at the cost of some false alarms.

### The three families of solutions

There are three main approaches, and they are not mutually exclusive:

#### 1. Data-level methods (resampling)

Change the training data to make the classes more balanced:

**Random Oversampling**: Duplicate random instances from the minority class. Simple but can lead to overfitting because the model memorizes repeated examples.

**Random Undersampling**: Remove random instances from the majority class. Simple and fast, but throws away potentially useful data. Works surprisingly well when you have lots of data.

**SMOTE (Synthetic Minority Over-sampling Technique)**: Instead of duplicating existing minority samples, create new synthetic ones. For each minority sample, SMOTE:
1. Finds its K nearest minority-class neighbors.
2. Picks one neighbor at random.
3. Creates a new point somewhere on the line between the original and the neighbor.

**Analogy**: If you have 50 photos of counterfeit bills, SMOTE does not just photocopy them. It creates new "plausible counterfeit" images by blending features of existing counterfeits. The new examples are similar to real counterfeits but not exact duplicates.

**ADASYN (Adaptive Synthetic Sampling)**: Like SMOTE, but it generates more synthetic samples for minority instances that are harder to classify (those near the decision boundary or surrounded by majority-class neighbors). This focuses the synthetic generation where it is most needed.

**Analogy**: ADASYN is like a teacher who gives extra practice problems specifically to students who are struggling, rather than giving equal extra practice to everyone.

#### 2. Algorithm-level methods

Modify the learning algorithm to pay more attention to the minority class:

**Class weighting**: Tell the algorithm that misclassifying a minority sample is more costly than misclassifying a majority sample. Most scikit-learn classifiers accept a `class_weight` parameter. Setting `class_weight='balanced'` automatically sets weights inversely proportional to class frequencies.

**Cost-sensitive learning**: Define an explicit cost matrix: the cost of a false negative (missing fraud) might be $10,000 while the cost of a false positive (investigating a legitimate claim) might be $100. The algorithm then minimizes expected total cost, not error count.

#### 3. Threshold tuning

Instead of using the default 0.5 probability threshold for classification, tune the threshold to match your business needs:

- Lower the threshold to catch more positives (higher recall, lower precision).
- Raise the threshold to reduce false alarms (lower recall, higher precision).
- Use the precision-recall curve or the F-beta score to find the optimal threshold.

### Why accuracy is misleading

On a dataset where 1% of cases are positive:
- A model predicting "all negative" achieves 99% accuracy.
- A model predicting 80% of positives correctly (recall=0.80) with 10% false positive rate might have 90% accuracy.

The second model is vastly more useful, but accuracy says the first model is "better." This is why imbalanced classification demands different evaluation metrics.

## Key concepts

### Evaluation metrics for imbalanced data

**Confusion matrix terms**:
- **True Positive (TP)**: Correctly identified positive (caught the fraud)
- **False Positive (FP)**: Incorrectly flagged as positive (investigated a legitimate claim)
- **True Negative (TN)**: Correctly identified negative
- **False Negative (FN)**: Missed a positive (fraud slipped through)

**Precision** (of those you flagged, how many were actually positive):

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall / Sensitivity** (of all actual positives, how many did you catch):

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score** (harmonic mean of precision and recall):

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

**F-beta Score** (weighted harmonic mean, where beta controls the emphasis):

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{(\beta^2 \cdot \text{Precision}) + \text{Recall}}$$

- $\beta = 2$: Emphasizes recall (use when missing positives is very costly, e.g., fraud)
- $\beta = 0.5$: Emphasizes precision (use when false alarms are very costly)

**Area Under the Precision-Recall Curve (AUPRC)**: Summarizes the precision-recall tradeoff across all thresholds. More informative than AUC-ROC for imbalanced data because it focuses on the positive class.

**Matthews Correlation Coefficient (MCC)**: A balanced measure that accounts for all four quadrants of the confusion matrix. Ranges from -1 to +1, where +1 is perfect, 0 is random, and -1 is perfectly wrong.

$$\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$$

**AUC-ROC**: Still useful but can be overly optimistic on imbalanced data because the ROC curve's false positive rate is dominated by the large negative class.

### SMOTE details

Given a minority class instance $\mathbf{x}_i$:

1. Find its $k$ nearest neighbors among minority class instances (typically $k=5$).
2. Choose one neighbor $\mathbf{x}_{nn}$ randomly.
3. Generate a synthetic point: $\mathbf{x}_{\text{new}} = \mathbf{x}_i + \lambda \cdot (\mathbf{x}_{nn} - \mathbf{x}_i)$, where $\lambda \sim \text{Uniform}(0, 1)$.

SMOTE variants:
- **Borderline-SMOTE**: Only oversample minority instances near the decision boundary.
- **SMOTE-ENN**: Combine SMOTE with Edited Nearest Neighbors (clean up noisy samples after oversampling).
- **SMOTE-Tomek**: Combine SMOTE with Tomek Links removal to clean the boundary between classes.

### ADASYN details

ADASYN adapts the number of synthetic samples generated for each minority instance based on the local density of majority class instances:

1. Compute the ratio $r_i = \frac{\Delta_i}{k}$ where $\Delta_i$ is the number of majority-class neighbors among the $k$ nearest neighbors of minority instance $i$.
2. Normalize: $\hat{r}_i = r_i / \sum r_i$.
3. Generate $g_i = \hat{r}_i \times G$ synthetic samples for instance $i$, where $G$ is the total number of synthetic samples to generate.

Instances surrounded by many majority-class neighbors (harder to classify) get more synthetic samples.

## Python implementation

### Resampling with imbalanced-learn

```python
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, f1_score,
                             precision_recall_curve, average_precision_score)

# Basic SMOTE
smote = SMOTE(sampling_strategy='auto', k_neighbors=5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ADASYN
adasyn = ADASYN(sampling_strategy='auto', n_neighbors=5, random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# Combined approach: undersample majority + oversample minority
pipeline = ImbPipeline([
    ('under', RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
    ('smote', SMOTE(sampling_strategy=1.0, random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])
```

### Class weighting

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# Automatic balanced weighting
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

# Custom weighting (e.g., misclassifying fraud costs 50x more)
lr = LogisticRegression(class_weight={0: 1, 1: 50}, random_state=42)

# For XGBoost: use scale_pos_weight
import xgboost as xgb
ratio = (y_train == 0).sum() / (y_train == 1).sum()
xgb_clf = xgb.XGBClassifier(scale_pos_weight=ratio, random_state=42)
```

### Threshold tuning

```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

# Get predicted probabilities
y_proba = model.predict_proba(X_test)[:, 1]

# Find the threshold that maximizes F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"Precision at optimal: {precisions[optimal_idx]:.3f}")
print(f"Recall at optimal: {recalls[optimal_idx]:.3f}")
print(f"F1 at optimal: {f1_scores[optimal_idx]:.3f}")

# Apply the tuned threshold
y_pred_tuned = (y_proba >= optimal_threshold).astype(int)
```

### Proper cross-validation with resampling

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, average_precision_score

# CRITICAL: Resampling must happen INSIDE cross-validation
# Never resample before splitting -- this leaks information
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {
    'f1': make_scorer(f1_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'average_precision': make_scorer(average_precision_score, needs_proba=True)
}

results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
```

## Gotchas & lessons learned

### NEVER resample before train-test split
This is the number one mistake. If you apply SMOTE to your entire dataset before splitting, synthetic samples generated from the minority class will leak information between your training and test sets. The test set will contain synthetic points that are interpolations of training points. Your evaluation metrics will be wildly optimistic. **Always resample inside the cross-validation loop or only on the training set.**

### SMOTE does not work well with categorical features
SMOTE interpolates between points in continuous feature space. Interpolating between "male" (0) and "female" (1) to get 0.43 is meaningless. Use SMOTE-NC (for mixed numeric-categorical) or encode categoricals carefully before applying SMOTE.

### Oversampling can cause overfitting
Especially random oversampling (which just duplicates minority examples), but even SMOTE can overfit if the original minority samples are noisy. Monitor your validation metrics -- if training F1 is 0.95 and validation F1 is 0.60, you are overfitting.

### Start with class weighting before resampling
Class weighting (`class_weight='balanced'`) is simpler, does not change your dataset size, does not create synthetic data, and works well with tree-based models. Try it first. Only move to SMOTE/ADASYN if class weighting is insufficient.

### Evaluate on the ORIGINAL class distribution
Always evaluate your model on the original, imbalanced test set. Resampling the test set gives you misleading metrics that do not reflect real-world performance.

### The precision-recall tradeoff is a business decision
There is no universally "correct" threshold. In fraud detection, you might accept low precision (many false alarms) for high recall (catch almost all fraud). In an automated underwriting triage, you might want high precision (when you flag, you are usually right) even if recall is lower. This tradeoff should be driven by the business cost of false positives vs. false negatives.

### Extreme imbalance (>100:1) needs special care
When the minority class is less than 1%, even SMOTE may not help much. Consider:
- Anomaly detection approaches (one-class SVM, Isolation Forest) instead of classification
- Cost-sensitive learning with explicit cost matrices
- Ensemble methods designed for imbalance (BalancedRandomForest, EasyEnsemble)

### Calibration after resampling
Resampling distorts the predicted probabilities. A model trained on 50/50 resampled data will overestimate the probability of the positive class in production (where the true prevalence might be 1%). If you need calibrated probabilities (e.g., for expected value calculations in pricing), recalibrate after resampling using Platt scaling or isotonic regression.

### Use stratified sampling everywhere
Always use `StratifiedKFold` for cross-validation and `stratify=y` in `train_test_split`. Without stratification, some folds might have zero positive examples, making evaluation meaningless.

## Resources

- [imbalanced-learn documentation](https://imbalanced-learn.org/stable/) -- The primary Python library for resampling methods
- [Chawla et al. (2002) - SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/abs/1106.1813) -- The original SMOTE paper
- [He et al. (2008) - ADASYN: Adaptive Synthetic Sampling](https://ieeexplore.ieee.org/document/4633969)
- [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics)
- [Google Developers - Classification: Imbalanced Datasets](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data)
- [Machine Learning Mastery - Imbalanced Classification](https://machinelearningmastery.com/what-is-imbalanced-classification/)
- [Lemaaitre, Nogueira, Aridas (2017) - Imbalanced-learn: A Python toolbox](https://jmlr.org/papers/v18/16-365.html)
- [He & Garcia (2009) - Learning from Imbalanced Data](https://ieeexplore.ieee.org/document/5128907) -- Comprehensive survey paper
