# Anomaly Detection

## What it is

Anomaly detection (also called outlier detection) is the task of identifying observations that deviate significantly from the expected pattern in a dataset. These unusual observations -- anomalies, outliers, novelties -- might represent fraud, equipment failures, data entry errors, rare diseases, or genuinely unusual but legitimate behavior. Unlike standard classification where you have labeled examples of both classes, anomaly detection often works with little or no labeled anomaly data. You typically know what "normal" looks like and want to flag anything that does not fit that pattern.

Anomaly detection sits in a unique space between supervised and unsupervised learning. Sometimes you have labels (known fraud cases), sometimes you have none (you just know something "seems off"), and sometimes you have labels only for the normal class. The techniques span statistical methods, distance-based methods, and machine learning approaches, each with different strengths.

## Why it matters

For actuarial and data science practitioners, anomaly detection is directly relevant to several critical applications:

**Fraud detection**: Insurance fraud costs the industry billions annually. Anomaly detection can flag suspicious claims -- those that look different from the typical claims pattern -- for investigation. A claim with an unusual combination of characteristics (high severity, short time to report, specific provider, specific claimant history) might be perfectly legitimate, but it warrants a closer look.

**Data quality**: Before building any predictive model, you need clean data. Anomaly detection identifies data entry errors, system glitches, and measurement problems that could corrupt your model. A policy with a negative premium or a vehicle with 0 km driven is not an insight -- it is a data error.

**Risk monitoring**: Identifying policyholders or portfolios whose behavior suddenly deviates from the norm can signal emerging risks. A sudden spike in claims from a particular territory or product line is an anomaly worth investigating.

**Reserving and financial reporting**: Unusual claim development patterns can indicate problems -- either operational issues (claims handling delays) or genuine shifts in loss experience.

## When to use it

- **When looking for fraud** -- insurance claims fraud, payment fraud, identity fraud.
- **When checking data quality** before model building -- identifying corrupt records, duplicates, or impossible values.
- **When monitoring production systems** -- detecting when model inputs or outputs drift from expected ranges.
- **When you have very few (or no) labeled examples of the anomalous class** -- standard classification needs labeled data for both classes.
- **When the definition of "anomaly" is vague or evolving** -- you cannot train a classifier if you do not know what fraud looks like yet.
- **When doing exploratory data analysis** -- understanding the unusual observations in your data often reveals important insights.
- **When building actuarial experience studies** -- identifying outlier claims that might distort aggregate statistics.

## How it works (intuitive)

### Statistical methods: The simple guards

**Z-score method**: For each variable, compute how many standard deviations an observation is from the mean. Flag observations with extreme z-scores (typically |z| > 3).

$$z_i = \frac{x_i - \mu}{\sigma}$$

*Analogy*: If the average adult height is 170 cm with a standard deviation of 10 cm, someone who is 210 cm tall (z-score = 4) is clearly unusual.

**Pros**: Simple, fast, interpretable. **Cons**: Assumes normal distribution. Sensitive to the very outliers it is trying to detect (the mean and standard deviation are pulled by outliers). Only works for single variables at a time.

**Modified z-score (using median and MAD)**: Replaces the mean with the median and the standard deviation with the Median Absolute Deviation (MAD), making it robust to outliers:

$$\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$

$$z_{\text{modified}} = \frac{0.6745 \cdot (x_i - \text{median}(x))}{\text{MAD}}$$

The constant 0.6745 scales MAD to be comparable to the standard deviation for normal distributions.

**IQR method (Tukey's fences)**: Compute the interquartile range (Q3 - Q1). Flag observations below Q1 - 1.5*IQR or above Q3 + 1.5*IQR as outliers. This is the logic behind box plot whiskers.

*Analogy*: You look at the middle 50% of the data and draw a fence at 1.5 times the width of that middle band on each side. Anything outside the fence is flagged.

**Pros**: Non-parametric, does not assume normality. Robust to outliers. **Cons**: Still univariate. The 1.5 multiplier is arbitrary (some use 3 for extreme outliers).

### Isolation Forest: The random path approach

Isolation Forest is one of the most popular ML-based anomaly detection methods. Its key insight: **anomalies are easier to isolate than normal points.**

**How it works**:
1. Randomly select a feature.
2. Randomly select a split value between the feature's minimum and maximum.
3. Repeat, recursively partitioning the data.
4. Count how many splits it takes to isolate each observation into its own partition.

Anomalies -- points that are far from the crowd -- get isolated in fewer splits. Normal points, which are packed together, require more splits to separate.

*Analogy*: Imagine dropping a series of random vertical and horizontal lines onto a scatter plot. The point that sits far from the cluster will be isolated quickly (just one or two lines might do it). Points in the dense center need many lines before each one is separated. The number of lines needed is the "anomaly score."

The anomaly score is based on the average path length across many random trees:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

Where:
- $E[h(x)]$ is the average path length for observation $x$ across all trees.
- $c(n)$ is the expected path length for a normal observation in a dataset of size $n$.
- $s$ close to 1: definitely anomalous (very short paths).
- $s$ close to 0.5: normal (average path length).
- $s$ close to 0: not anomalous at all.

**Pros**: Fast (linear time complexity), handles high-dimensional data, no distance metrics needed, works well in practice. **Cons**: The `contamination` parameter (expected proportion of outliers) must be set, and the results are sensitive to it.

### Local Outlier Factor (LOF): The neighborhood approach

LOF compares the density of the neighborhood around each point to the density of its neighbors' neighborhoods. A point in a sparse region surrounded by points in dense regions is anomalous.

**How it works**:
1. For each point, find its $k$ nearest neighbors.
2. Compute the "local reachability density" -- how dense the neighborhood is around each point.
3. Compare each point's density to the density of its neighbors.
4. If a point's density is much lower than its neighbors', it is an outlier.

*Analogy*: Imagine a party where most people are in tight clusters of conversation. LOF identifies the person standing alone in the middle of the room (global outlier) AND the person standing at the edge of a tight group (local outlier). The key word is "local" -- LOF can detect outliers relative to their local neighborhood, even when the data has clusters of different densities.

$$\text{LOF}(x) = \frac{\sum_{o \in N_k(x)} \frac{\text{lrd}(o)}{\text{lrd}(x)}}{|N_k(x)|}$$

Where $\text{lrd}(x)$ is the local reachability density of $x$ and $N_k(x)$ is the set of $k$ nearest neighbors.

- LOF close to 1: Normal (density similar to neighbors).
- LOF >> 1: Anomalous (density much lower than neighbors).

**Pros**: Detects local outliers that global methods miss. Works with clusters of varying density. **Cons**: Slow for large datasets ($O(n^2)$ for finding neighbors). Sensitive to the choice of $k$. Not designed for very high-dimensional data (distances become less meaningful -- the curse of dimensionality again).

### One-Class SVM: The boundary approach

One-Class SVM learns a decision boundary that encloses the normal data points. New observations that fall outside this boundary are classified as anomalies.

**How it works**:
1. Map the data into a high-dimensional feature space using a kernel function (typically RBF/Gaussian).
2. Find the smallest hypersphere (or hyperplane) that encloses most of the data.
3. Points outside this boundary are anomalies.

*Analogy*: Imagine wrapping a rubber band around a cluster of points on a table. The rubber band defines the "normal" region. Any point outside the rubber band is flagged. The kernel trick allows this wrapping to happen in complex, non-linear shapes.

**Key parameter** -- `nu`: An upper bound on the fraction of training observations that can be outside the boundary (and also a lower bound on the fraction of support vectors). In practice, it approximates the expected contamination rate.

**Pros**: Can learn complex non-linear boundaries. Theoretically principled. **Cons**: Sensitive to kernel choice and parameters. Does not scale well to very large datasets. Requires standardized features.

### Unsupervised vs semi-supervised anomaly detection

| Approach | Training data | Assumption | Example |
|---|---|---|---|
| Unsupervised | No labels at all | Anomalies are rare and different | Isolation Forest, LOF |
| Semi-supervised | Only normal labels | You know what "normal" looks like | One-Class SVM, autoencoder |
| Supervised | Labels for both classes | You have enough labeled anomalies | Random forest, XGBoost |

In practice, fraud detection often starts as unsupervised (you do not know what fraud looks like), then evolves to semi-supervised (you have a clean dataset of non-fraudulent claims), and eventually becomes supervised (you have accumulated enough confirmed fraud cases).

## Key concepts

### The contamination parameter

Most anomaly detection algorithms require you to specify (or estimate) the proportion of anomalies in the dataset. This is called the `contamination` parameter.

- **Too low**: You miss real anomalies (false negatives).
- **Too high**: You flag too many normal observations (false positives).

*Actuarial context*: In insurance fraud, contamination is typically 1-10% depending on the line of business and definition of fraud. For data quality checking, it might be much lower (0.1-1%).

**Setting contamination**:
- If you have domain knowledge, use it (e.g., "we estimate 5% of claims are fraudulent").
- If you have no idea, start with 5-10% and adjust based on the results.
- You can also examine the distribution of anomaly scores and set a threshold based on a clear gap or elbow in the distribution.

### Evaluation without labels

Evaluating anomaly detection is fundamentally hard because you often do not have ground truth labels. Strategies:

1. **Domain expert review**: Show flagged anomalies to investigators or underwriters. Track the "hit rate" -- what fraction of flagged cases are actually anomalous.
2. **Synthetic anomalies**: Inject known anomalies into your data and check if the algorithm detects them.
3. **Stability analysis**: Run the algorithm multiple times with different parameters. Observations consistently flagged across settings are more likely to be true anomalies.
4. **If you have labels**: Use precision, recall, F1, and AUC-ROC (but remember the class is severely imbalanced, so PR-AUC is usually more informative).

### Multivariate anomalies

Some anomalies are only visible when considering multiple variables together. A 25-year-old with 30 years of driving experience is anomalous, but neither "age = 25" nor "years_driving = 30" is individually unusual. This is why multivariate methods (Isolation Forest, LOF, One-Class SVM) are essential -- they detect combinations that univariate methods miss.

### Feature scaling

Most distance-based anomaly detection methods (LOF, One-Class SVM) are sensitive to feature scales. Always standardize features before applying these methods. Isolation Forest is an exception -- since it uses random splits on individual features, it is relatively scale-invariant.

## Python implementation

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# --- Statistical methods ---

# Z-score
from scipy import stats
z_scores = np.abs(stats.zscore(df[numeric_cols]))
outliers_zscore = (z_scores > 3).any(axis=1)

# Modified z-score (robust)
def modified_zscore(x):
    median = np.median(x)
    mad = np.median(np.abs(x - median))
    return 0.6745 * (x - median) / (mad + 1e-10)

mz = np.abs(modified_zscore(df['claim_amount']))
outliers_mz = mz > 3.5

# IQR method
Q1 = df['claim_amount'].quantile(0.25)
Q3 = df['claim_amount'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers_iqr = (df['claim_amount'] < lower) | (df['claim_amount'] > upper)

# --- Isolation Forest ---
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.05,       # Expected 5% anomalies
    max_samples='auto',
    random_state=42
)
iso_forest.fit(X_train_scaled)

# Predictions: -1 = anomaly, 1 = normal
iso_labels = iso_forest.predict(X_test_scaled)
iso_scores = iso_forest.decision_function(X_test_scaled)  # Higher = more normal

# --- Local Outlier Factor ---
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=False  # Use True if you want to predict on new data
)
lof_labels = lof.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
lof_scores = -lof.negative_outlier_factor_  # Higher = more anomalous

# LOF for novelty detection (predict on new data)
lof_novelty = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
lof_novelty.fit(X_train_scaled)  # Fit on "normal" training data
lof_novel_labels = lof_novelty.predict(X_test_scaled)

# --- One-Class SVM ---
oc_svm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.05  # Upper bound on fraction of outliers
)
oc_svm.fit(X_train_scaled)
svm_labels = oc_svm.predict(X_test_scaled)
svm_scores = oc_svm.decision_function(X_test_scaled)

# --- Visualization: Anomaly scores distribution ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(iso_scores, bins=50, edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--', label='Threshold')
axes[0].set_title('Isolation Forest Scores')
axes[0].legend()

axes[1].hist(lof_scores, bins=50, edgecolor='black')
axes[1].set_title('LOF Scores')

axes[2].hist(svm_scores, bins=50, edgecolor='black')
axes[2].axvline(x=0, color='red', linestyle='--', label='Threshold')
axes[2].set_title('One-Class SVM Scores')
axes[2].legend()

plt.tight_layout()
plt.show()

# --- Ensemble approach: combine multiple methods ---
def ensemble_anomaly_detection(X, contamination=0.05):
    """Flag observations as anomalous if multiple methods agree."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso = IsolationForest(contamination=contamination, random_state=42)
    lof = LocalOutlierFactor(contamination=contamination, novelty=False)

    iso_labels = iso.fit_predict(X_scaled)
    lof_labels = lof.fit_predict(X_scaled)

    # Consensus: anomaly only if both methods agree
    consensus = ((iso_labels == -1) & (lof_labels == -1)).astype(int)
    return consensus

# --- Evaluation with labels (if available) ---
from sklearn.metrics import (
    classification_report, precision_recall_curve,
    average_precision_score
)

# Convert -1/1 to 1/0 for compatibility with sklearn
y_pred_binary = (iso_labels == -1).astype(int)
print(classification_report(y_true, y_pred_binary, target_names=['Normal', 'Anomaly']))

# PR-AUC is more informative than ROC-AUC for anomaly detection
pr_auc = average_precision_score(y_true, -iso_scores)  # Negate: higher score = more anomalous
print(f"PR-AUC: {pr_auc:.3f}")
```

## R implementation (if applicable)

```r
library(isotree)     # Isolation Forest
library(dbscan)      # LOF
library(e1071)       # One-Class SVM

# --- Isolation Forest ---
iso_model <- isolation.forest(X_train, ntrees = 200, sample_size = 256)
iso_scores <- predict(iso_model, X_test)  # Higher = more anomalous

# --- Local Outlier Factor ---
lof_scores <- lof(X_scaled, minPts = 20)
# lof_scores > 1 suggests an outlier

# --- One-Class SVM ---
svm_model <- svm(X_train, type = "one-classification", kernel = "radial",
                  nu = 0.05, gamma = 1 / ncol(X_train))
svm_pred <- predict(svm_model, X_test)  # TRUE = normal, FALSE = anomaly

# --- Statistical: IQR method ---
outlier_iqr <- function(x, k = 1.5) {
  q <- quantile(x, probs = c(0.25, 0.75))
  iqr <- q[2] - q[1]
  x < (q[1] - k * iqr) | x > (q[2] + k * iqr)
}
```

## Gotchas & lessons learned

1. **The contamination parameter is critical and often guessed**: Setting `contamination=0.05` means you expect 5% anomalies. If the true rate is 1%, you will get many false positives. If the true rate is 10%, you will miss many. When possible, estimate contamination from domain knowledge or past investigations. When not possible, examine the score distribution and look for a natural gap.

2. **Anomaly detection is not classification**: Do not expect anomaly detection to achieve the same precision and recall as a supervised classifier. Without labeled examples, the algorithm is essentially making educated guesses about what is unusual. Use the output as a prioritization tool (investigate the most anomalous cases first), not as a definitive label.

3. **Scale your features (except for Isolation Forest)**: LOF, One-Class SVM, and most distance-based methods are heavily affected by feature scales. A feature measured in millions will dominate the distance calculations. Standardize first. Isolation Forest is relatively robust to scaling because it uses random splits on individual features.

4. **High dimensionality hurts distance-based methods**: In very high dimensions, all points become roughly equidistant (the curse of dimensionality). LOF and One-Class SVM can degrade. Consider dimensionality reduction (PCA) before applying these methods, or use Isolation Forest, which handles high dimensions better.

5. **Normal data is not always well-behaved**: Many methods assume that normal data forms a single, roughly elliptical cluster. If your normal data has multiple distinct clusters (e.g., different customer segments), global methods might flag points in small clusters as anomalies. Use local methods (LOF) or cluster first and detect anomalies within each cluster.

6. **Seasonality and temporal patterns create false anomalies**: A claim amount that is normal in December (holiday season) might look anomalous in February. If your data has temporal patterns, account for them before anomaly detection (e.g., detect anomalies within each time period, or use features that capture seasonal effects).

7. **Ensemble methods are more robust**: Individual anomaly detectors have different strengths and blind spots. Combining multiple methods (e.g., flag as anomalous only if both Isolation Forest and LOF agree) reduces false positives and produces more trustworthy results.

8. **Interpret, do not just flag**: Identifying an anomaly is step one. Understanding WHY it is anomalous is equally important. Pair anomaly detection with SHAP or similar interpretability tools to explain what makes each flagged observation unusual. An investigator needs to know "this claim is flagged because the combination of low policy tenure and high claim severity is unusual," not just "this claim is flagged."

9. **Isolation Forest's random nature**: Isolation Forest uses random splits, so results can vary between runs. Set `random_state` for reproducibility, and consider averaging scores across multiple runs for more stable results.

10. **Beware of confusing rare with anomalous**: A 90-year-old policyholder is rare but not necessarily anomalous. A legitimate catastrophe claim is rare but not fraudulent. Anomaly detection flags unusual patterns, not incorrect or malicious ones. The output is a starting point for investigation, not a verdict.

11. **Updating anomaly detection models**: As your data evolves (new products, new customer segments, changing fraud tactics), the definition of "normal" changes. Retrain anomaly detection models regularly. In fraud detection, adversaries actively adapt to avoid detection, so your model must adapt too.

12. **For fraud detection, combine rules and ML**: In practice, the best fraud detection systems combine expert-defined rules (e.g., "flag any claim over $50,000 filed within 30 days of policy inception") with ML-based anomaly detection. Rules catch known patterns; ML catches novel patterns.

## Resources

- **scikit-learn anomaly detection**: https://scikit-learn.org/stable/modules/outlier_detection.html
- **Liu, Ting, Zhou (2008)**: "Isolation Forest" -- the original paper. https://doi.org/10.1109/ICDM.2008.17
- **Breunig et al. (2000)**: "LOF: Identifying Density-Based Local Outliers" -- the original LOF paper.
- **Scholkopf et al. (2001)**: "Estimating the Support of a High-Dimensional Distribution" -- One-Class SVM.
- **PyOD library**: https://pyod.readthedocs.io/ -- a comprehensive Python library with 30+ anomaly detection algorithms. Much broader than scikit-learn's offerings.
- **Aggarwal (2017)**: "Outlier Analysis" -- comprehensive textbook covering theory and practice.
- **CAS Fraud Detection resources**: https://www.casact.org/ -- actuarial perspective on fraud analytics.
- **Chandola, Banerjee, Kumar (2009)**: "Anomaly Detection: A Survey" -- broad survey of anomaly detection methods and applications.
