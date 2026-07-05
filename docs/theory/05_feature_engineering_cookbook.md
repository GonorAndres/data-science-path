# Feature Engineering Cookbook

Feature engineering is often the single highest-leverage move in tabular ML. A well-chosen feature can do more than any model upgrade. This essay is a catalog of patterns that work across domains, with notes on when each applies.

## General principles

- Features encode domain knowledge. Start from "what does a domain expert know that my raw columns do not?"
- Every feature should pass the point-in-time check (see `04_data_leakage_patterns.md`).
- Simpler features with clear semantics beat opaque clever features of similar predictive power. They are easier to debug, monitor, and explain.

## Numeric features

### Binning (discretization)

Convert continuous values to bins. Useful when:

- The relationship with the target is highly non-linear and you use linear models.
- You want to handle outliers robustly (top and bottom bins absorb them).
- The domain already has standard bins (age groups, income brackets).

Quantile binning (equal-frequency) handles skewed distributions better than equal-width.

### Log, sqrt, box-cox transforms

Useful when the feature spans several orders of magnitude or is heavily right-skewed (prices, counts, income, duration).

For tree models, monotonic transforms do not matter (trees are scale-invariant). For linear models and neural networks, they do.

### Scaling

Standardization ($z$-score) for linear models and neural networks. Min-max for bounded distributions. Tree models need no scaling.

### Interaction features

Products, ratios, differences between two features. Examples:

- `revenue / orders` = average order value.
- `days_since_last_order - days_since_first_order` = customer tenure.
- `price * units` = line-item revenue.

Linear models benefit massively from explicit interactions. Trees pick some up but not always.

## Categorical features

### One-hot encoding

Default for low-cardinality categoricals (under 20 to 50 levels). Simple, honest, high-dimensional.

For tree-based models, one-hot is less efficient than ordinal + split-finding on categories (LightGBM handles this natively with `categorical_feature=`).

### Frequency encoding

Replace the category with how often it appears. Preserves signal about rarity.

### Target (mean) encoding

Replace the category with the mean target value in that category. Leakage risk: must be computed on training folds only, and usually with smoothing for rare categories:

$$
\text{enc}(c) = \frac{n_c \cdot \bar{y}_c + m \cdot \bar{y}}{n_c + m}
$$

where $n_c$ is the category size, $\bar{y}_c$ is the category mean, $\bar{y}$ is the global mean, and $m$ is a smoothing hyperparameter.

### Entity embeddings

Learned dense vectors per category. Popular for high-cardinality categoricals (zip code, product ID, hotel, doctor). Embeddings are trained jointly with a neural network or extracted from a pretrained one.

### Hashing trick

Hash the category to a fixed number of buckets. Useful at very high cardinality where one-hot is infeasible and you cannot afford to learn embeddings.

## Datetime features

Decompose a timestamp into multiple features:

- Year, month, day of month, day of week, hour, minute.
- Is weekend, is holiday, is month start, is quarter end.
- Days since a reference date.
- Cyclical encoding: $\sin(2\pi \cdot \text{hour}/24)$ and $\cos(\cdot)$ so the model sees midnight and 23:00 as close.

For holiday features, use a calendar aware of the relevant country (Mexican holidays differ from US).

## Aggregation features

For a row belonging to an entity (user, store, product), bring in summary statistics of related records:

- Past 7-day spend per user.
- Rolling 30-day average order size.
- Count of distinct products purchased.
- Standard deviation of inter-event times.

Use a feature store or explicit windowed aggregation in SQL. Always respect time cutoffs.

## Ratio and rate features

Divisions communicate a lot:

- Conversion rate (clicks / impressions).
- Return rate (returns / purchases).
- Frequency (events / days observed).

Handle zero denominators with `NULLIF` in SQL or pandas.

## Text features

### Bag of words / TF-IDF

Baseline for text classification. Sparse but fast. Fine for short-text, well-defined-vocabulary tasks.

### Pretrained embeddings

Word2Vec, GloVe (older). Sentence-transformers (modern default). Each document becomes a dense vector; feed to downstream classifier.

### LLM feature extraction

Pass text through a transformer, take the `[CLS]` token or pooled representation, feed to a downstream model. High quality, expensive.

### Text statistics

Length, number of words, average word length, number of punctuation, ratio of capital letters, emoji count. Often useful even for sophisticated models.

## Image features

### Pretrained CNN or ViT embeddings

Pass the image through a pretrained ResNet, EfficientNet, or ViT; take the penultimate layer. Use as features for a downstream classifier or regressor.

### Geometric and color features

Histograms, color moments, HOG, SIFT. Rarely beats learned embeddings but useful when compute is constrained.

## Geographic features

- Haversine distance between two lat-lon points.
- Reverse geocoding: assign zip code, colonia, delegacion.
- Drive-time distance (requires a routing API).
- H3 or S2 hexagonal index for spatial joins and aggregation.

## Graph features

When the data has relationships (social network, user-item, org chart):

- Degree (number of connections).
- Clustering coefficient.
- PageRank.
- Community membership (Louvain, Leiden).
- Embeddings from node2vec or a GNN.

## Time-series specific features

- Lags: target value 1, 7, 28 periods ago.
- Rolling statistics: mean, std, max over a window.
- Differences: first difference, seasonal difference.
- Fourier components: for seasonal decomposition.
- Trend: fitted slope over a window.

Care with windowing: each feature must use only past data relative to the target timestamp.

## Anti-patterns

### Over-engineering

More features is not always better. Irrelevant features add noise, dilute tree splits, and inflate computational cost.

### Feature engineering on the test set

Any aggregation, encoding, or imputation fitted on the full dataset is a leakage hazard.

### Copying from academic papers verbatim

Papers report features that work on clean benchmarks. Real production data rarely matches those conditions. Validate every feature on your own data.

### Keeping features that are unavailable in production

A feature that requires a human manual annotation has no place in an automated pipeline.

## Feature importance and pruning

After training, drop features with low importance (SHAP values, permutation importance, XGBoost gain). Retrain and compare metrics.

Pruning benefits:

- Smaller models, faster inference.
- Less surface area for distribution shift.
- Easier monitoring.

Pruning risks:

- Low aggregate importance can hide local importance (a feature that matters only for a small segment).
- Correlated features can split importance; removing one reallocates it to the other.

## Documentation

Maintain a feature catalog: name, source table, computation, availability timestamp, expected range, freshness.

Modern tools: Feast, Tecton, Hopsworks. For small projects, a markdown file or a CSV is enough.

## Linking to the portfolio

- `nlp-reviews-multilingual`: TF-IDF baseline plus sentence-transformer embeddings plus length features.
- `timeseries-retail-forecast`: lags, rolling stats, holiday flags, promotional calendars, Fourier terms for seasonality.
- `recsys-movielens`: user and item embeddings from matrix factorization plus content features (genres, release year).
- `cv-defect-detection`: ResNet embeddings from a pretrained backbone plus image-level statistics as auxiliary input.
- `causal-uplift-marketing`: pre-treatment behavior aggregates, tenure features, promotional response history.

## Weight of Evidence and Information Value

Weight of Evidence (WoE) and Information Value (IV) are the canonical encoding pair from credit scoring and actuarial pricing. They convert a categorical (or binned numeric) feature into a single numeric column that is monotonic in the target log-odds.

For a binary target $y \in \{0, 1\}$ and a feature with categories indexed by $c$, define the share of positives and negatives in category $c$:

$$
P_c = \frac{\#\{y=1, \text{cat}=c\}}{\#\{y=1\}}, \qquad N_c = \frac{\#\{y=0, \text{cat}=c\}}{\#\{y=0\}}.
$$

Then

$$
\mathrm{WoE}_c = \log\frac{P_c}{N_c}, \qquad \mathrm{IV} = \sum_c (P_c - N_c) \cdot \mathrm{WoE}_c.
$$

WoE is the per-category encoded value. IV is a single scalar measuring the feature's predictive strength.

### Interpretation table for IV

| IV range | Interpretation |
|----------|----------------|
| $< 0.02$ | Useless for prediction. |
| $0.02$ to $0.1$ | Weak. |
| $0.1$ to $0.3$ | Medium. |
| $0.3$ to $0.5$ | Strong. |
| $> 0.5$ | Suspicious; check for leakage or target encoding contamination. |

### Python implementation

```python
import numpy as np
import pandas as pd

def woe_iv(df, feature, target):
    grouped = df.groupby(feature)[target].agg(['sum', 'count'])
    grouped.columns = ['pos', 'total']
    grouped['neg'] = grouped['total'] - grouped['pos']
    pos_total = grouped['pos'].sum()
    neg_total = grouped['neg'].sum()
    grouped['p_share'] = grouped['pos'] / pos_total
    grouped['n_share'] = grouped['neg'] / neg_total
    # add small epsilon to avoid log(0)
    eps = 1e-6
    grouped['woe'] = np.log((grouped['p_share'] + eps) / (grouped['n_share'] + eps))
    iv = ((grouped['p_share'] - grouped['n_share']) * grouped['woe']).sum()
    return grouped['woe'].to_dict(), iv
```

Always compute WoE on the training fold only and apply the learned mapping to validation and test. Re-fitting WoE on the test set is leakage.

## James-Stein and empirical-Bayes target encoding

Smoothed target encoding (essay's earlier section) is the credibility formula in disguise. Rewrite

$$
\hat y_c = Z_c \bar y_c + (1 - Z_c) \bar y, \qquad Z_c = \frac{n_c}{n_c + k},
$$

with $k$ chosen by empirical Bayes:

$$
\hat k = \frac{\widehat{E[\sigma_c^2]}}{\widehat{\operatorname{Var}}(\bar y_c)}.
$$

This is precisely Buhlmann credibility: $k$ is the ratio of within-class variance to between-class variance. The James-Stein estimator generalizes this shrinkage to vector-valued targets.

When $n_c$ is large, $Z_c \to 1$ and the encoding equals the raw category mean. When $n_c$ is small, $Z_c \to 0$ and the encoding falls back to the global mean. The single hyperparameter $k$ controls how aggressive the shrinkage is.

For an actuarial reader, the bridge is direct: target encoding and credibility are the same estimator with different vocabulary.

## Out-of-fold target encoding (the right pattern)

Smoothing alone is necessary but not sufficient. If the encoding is fitted on the full training set including the row whose label is being encoded, the encoded feature contains the label and produces overly optimistic CV. The fix is **out-of-fold (OOF) encoding**:

```text
for each fold f in CV:
    train_fold = data outside fold f
    for each category c:
        compute hat_y_c on train_fold only
    apply mapping to fold f
encoded_train = concatenation of fold-encoded blocks

# for the test set, fit the encoder on the full training set once
encoded_test = mapping(test) using full-training-set fit
```

The concatenated `encoded_train` has the property that no row's encoded value used its own label. Pass this column into your CV pipeline as if it were any ordinary numeric feature.

The category-encoders library (`MEstimateEncoder`, `TargetEncoder`) implements OOF correctly when used with sklearn's `Pipeline` inside `cross_val_score`. Check its documentation; subtle misuse defeats the protection.

## Count vs frequency encoding

These two encodings are often conflated.

- **Count encoding.** Replace category $c$ with $n_c$, the number of training rows in that category. Useful when category size carries signal independent of the target.
- **Frequency encoding.** Replace category $c$ with $n_c / N$, the relative share. Same information rescaled.

Count is preferred when downstream model treats the value as a magnitude (XGBoost using gain). Frequency is preferred for linear models or when you want the encoding bounded. Both are leakage-free because they only use feature-side counts, not labels.

## Monotonic constraints for tree models

When business logic requires a feature to have a monotonic effect on the prediction (claim frequency increases with driver age past 65; default probability decreases with income), tree models can violate the constraint due to noise or interactions. Both XGBoost and LightGBM accept monotonic constraints.

```python
# XGBoost: vector with one entry per feature in [-1, 0, 1]
# +1 means strictly increasing, -1 strictly decreasing, 0 unconstrained
import xgboost as xgb
model = xgb.XGBRegressor(monotone_constraints=(1, 0, -1, 0, 0))

# LightGBM: same convention
import lightgbm as lgb
model = lgb.LGBMRegressor(monotone_constraints=[1, 0, -1, 0, 0])
```

Use sparingly: each constraint trades flexibility for interpretability. In regulated pricing or credit settings, monotonicity is often a hard requirement from the regulator, not a preference.

## Worked pipeline example: leaky vs non-leaky

Wrong (leakage during CV):

```python
# fits scaler on the full X before splitting
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
# now CV uses X_scaled, but each fold's scaler "saw" the validation rows
```

Right (sklearn Pipeline keeps scaler inside CV):

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0)),
])
scores = cross_val_score(pipe, X, y, cv=5)
```

The Pipeline refits the scaler inside each fold; the validation rows never inform the scaler that scores them. The same pattern applies to imputers, target encoders, and any preprocessing that estimates parameters from data.

## References

- Kuhn, Johnson, *Feature Engineering and Selection*, 2019.
- Guyon, Elisseeff, "An Introduction to Variable and Feature Selection", 2003.
- Guo, Berkhahn, "Entity Embeddings of Categorical Variables", 2016.
- Micci-Barreca, "A Preprocessing Scheme for High-Cardinality Categorical Attributes in Classification and Prediction Problems", 2001 (target encoding).

## Exercises

1. Compute the WoE for a binary target where the bad rate in a bin is $0.15$ and the overall bad rate is $0.05$.
2. Describe the James-Stein smoothed target encoding formula and relate the smoothing parameter to Buhlmann credibility.
3. You are encoding a high-cardinality categorical (10,000 unique values) for a tree model. Compare one-hot encoding, target encoding, and entity embeddings. When is each preferred?
4. Why must target encoding be computed inside the cross-validation loop and not before? Show with a concrete example what goes wrong if it is done outside.
5. A tree model shows a non-monotonic relationship between policyholder age and claim probability. Actuarial domain knowledge says this should be monotonic. What do you do?
- Siddiqi, *Credit Risk Scorecards: Developing and Implementing Intelligent Credit Scoring*, 2nd ed., 2017 (WoE and IV).
- Efron, Morris, "Stein's Paradox in Statistics", Scientific American, 1977 (James-Stein intuition).
- Buhlmann, Gisler, *A Course in Credibility Theory and its Applications*, Springer, 2005.
