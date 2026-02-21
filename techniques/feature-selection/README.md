# Feature Selection

## What it is

Feature selection is the process of identifying which variables in your dataset actually contribute useful information to your model -- and discarding the rest. It is the disciplined act of saying "less is more." If feature engineering is about creating the right ingredients, feature selection is about removing the ones that add noise, redundancy, or computational cost without improving (and sometimes hurting) your model's performance. The goal is to find the smallest set of features that captures the signal in your data while ignoring the noise.

## Why it matters

In actuarial and data science work, you often start with dozens or hundreds of potential predictors. Policy databases might have 200+ columns. Not all of them matter. Including irrelevant or redundant features can:

- **Increase overfitting**: More features means more opportunities for the model to memorize noise in the training data rather than learning genuine patterns.
- **Slow down training**: More features means larger matrices, longer computation, and more memory. This matters when you need to retrain models frequently or work within production latency constraints.
- **Reduce interpretability**: A model with 150 features is much harder to explain to an underwriter or regulator than one with 15.
- **Trigger the curse of dimensionality**: As the number of features grows relative to the number of observations, the data becomes increasingly sparse, and models struggle to find reliable patterns.

For actuaries specifically, regulatory and governance requirements often demand that you can justify every variable in your model. Feature selection gives you the evidence to do that.

## When to use it

- **After feature engineering**, when you have created many candidate features and need to narrow down.
- **When your model is overfitting** -- removing noisy features is one of the most effective remedies.
- **When you need an interpretable model** for regulatory filings, audit, or stakeholder communication.
- **When training time is a constraint** and you need to reduce the input dimensionality.
- **When you suspect multicollinearity** is distorting your coefficient estimates (especially in linear/GLM models).
- **When working with wide datasets** (many columns relative to rows), common in genomics, text mining, or detailed insurance transaction data.
- **Before deploying a model to production**, to minimize the number of data feeds and transformations required.

## How it works (intuitive)

Feature selection methods fall into three families. Think of them as three different strategies for deciding which players make the team.

### Filter methods: The tryout stats

Filter methods evaluate each feature independently (or in pairs), without training a model. They use statistical measures to score features and then keep the top-ranked ones.

*Analogy*: You are a football coach looking at individual player stats -- speed, accuracy, stamina -- before you even run a practice game. You cut players with the worst stats.

**Correlation (for numeric features)**

Pearson correlation measures the linear relationship between a feature and the target. Features with correlation near zero are likely not useful for linear models.

$$r_{xy} = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2 \cdot \sum_{i=1}^{n}(y_i - \bar{y})^2}}$$

- $r$ close to +1 or -1: strong linear relationship.
- $r$ close to 0: no linear relationship (but there could be a non-linear one).
- Also check feature-to-feature correlation. If two features are highly correlated with each other (e.g., $r > 0.9$), one of them is probably redundant.

**Mutual information (for any feature type)**

Mutual information measures how much knowing a feature reduces uncertainty about the target. Unlike correlation, it captures non-linear relationships.

$$I(X; Y) = \sum_{x} \sum_{y} p(x, y) \log \frac{p(x, y)}{p(x) \, p(y)}$$

- MI = 0 means the feature and target are independent.
- Higher MI means the feature carries more information about the target.

*Analogy*: Correlation asks "do these two things move in a straight line together?" Mutual information asks the broader question "does knowing one thing tell me anything at all about the other?"

**Chi-squared test (for categorical features, categorical target)**

The chi-squared test checks whether a categorical feature is statistically independent of a categorical target. A large chi-squared statistic means the feature and target are likely related.

$$\chi^2 = \sum_{i} \sum_{j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Where $O_{ij}$ is the observed count and $E_{ij}$ is the expected count if the feature and target were independent.

**Variance threshold**

The simplest filter: remove any feature whose variance is below a threshold. A feature that is nearly constant (e.g., 99.9% of values are the same) carries almost no information.

### Wrapper methods: The practice games

Wrapper methods actually train models with different subsets of features and see which combination performs best. They treat the model as a black box and search for the best feature subset.

*Analogy*: Instead of looking at individual stats, you run actual practice games with different team combinations and see which lineup wins the most.

**Recursive Feature Elimination (RFE)**

1. Train a model with all features.
2. Rank features by importance (e.g., coefficient magnitude in linear models, or feature importance in trees).
3. Remove the least important feature.
4. Repeat until you reach the desired number of features.

RFE is like elimination rounds on a cooking show -- each round, the weakest contestant goes home.

**Forward selection**

1. Start with zero features.
2. Try adding each available feature one at a time.
3. Keep the one that improves performance the most.
4. Repeat until adding more features no longer helps.

**Backward elimination**

1. Start with all features.
2. Try removing each feature one at a time.
3. Remove the one whose absence hurts performance the least.
4. Repeat until removing any feature would significantly hurt performance.

Wrapper methods are more thorough than filters but much more computationally expensive. With $p$ features, the full search space is $2^p$ possible subsets.

### Embedded methods: Built-in selection

Embedded methods perform feature selection as part of the model training process itself. The model learns which features matter while it learns the parameters.

*Analogy*: Instead of separate tryouts, the coach evaluates players during actual games and naturally gives more playing time to the ones who perform well.

**L1 regularization (Lasso)**

Lasso regression adds a penalty that pushes some coefficients exactly to zero. Features with zero coefficients are effectively removed. (See the regularization README for full details.)

$$\hat{\beta} = \arg\min_{\beta} \left[ \sum_{i=1}^{n}(y_i - X_i\beta)^2 + \lambda \sum_{j=1}^{p}|\beta_j| \right]$$

**Tree-based feature importance**

Decision trees, random forests, and gradient boosting models track how much each feature contributes to reducing prediction error. Features that are never (or rarely) used in splits have low importance and can be dropped.

- **Gini importance / impurity-based**: Measures how much each feature reduces impurity across all splits in the tree ensemble.
- **Permutation importance**: Measures how much model performance degrades when a feature's values are randomly shuffled. More reliable than impurity-based importance, especially with correlated features.

## Key concepts

### The curse of dimensionality

As the number of features grows, the volume of the feature space explodes exponentially. This creates several problems:

1. **Sparsity**: Data points become increasingly far apart, making it harder to find meaningful neighbors or clusters.
2. **Overfitting**: With many features relative to observations, models can find spurious correlations that do not generalize.
3. **Computational cost**: Training time and memory grow with the number of features.

*Intuition*: Imagine searching for a friend in a one-dimensional space (a road) -- easy. In two dimensions (a field) -- harder. In a hundred dimensions -- nearly impossible with the same number of data points. You need exponentially more data as dimensions increase.

**Rule of thumb**: You generally want at least 10-20 observations per feature for stable modeling. If you have 1,000 observations and 500 features, you are in dangerous territory.

### Variance-bias tradeoff in feature selection

- **Too few features**: High bias. The model is too simple and misses important patterns.
- **Too many features**: High variance. The model fits noise and performs poorly on new data.
- **Just right**: The model captures the real patterns without overfitting to noise.

### Multicollinearity

When features are highly correlated with each other, it does not necessarily hurt prediction accuracy, but it destabilizes coefficient estimates in linear models. Feature selection can help by keeping one representative from each group of correlated features.

Variance Inflation Factor (VIF) quantifies multicollinearity:

$$\text{VIF}_j = \frac{1}{1 - R_j^2}$$

Where $R_j^2$ is the R-squared from regressing feature $j$ on all other features. VIF > 5-10 suggests problematic multicollinearity.

### Summary of methods

| Method | Type | Pros | Cons |
|---|---|---|---|
| Correlation | Filter | Fast, interpretable | Only detects linear relationships |
| Mutual information | Filter | Detects non-linear relationships | Slower to compute, requires binning |
| Chi-squared | Filter | Good for categorical vs categorical | Only for non-negative features |
| Variance threshold | Filter | Very fast, obvious | Only removes near-constant features |
| RFE | Wrapper | Considers feature interactions | Computationally expensive |
| Forward/backward | Wrapper | Finds good subsets | Very expensive, greedy (not globally optimal) |
| Lasso (L1) | Embedded | Automatic, principled | Assumes linear model; unstable with correlated features |
| Tree importance | Embedded | Works with non-linear models | Impurity importance biased toward high-cardinality features |
| Permutation importance | Embedded/Post-hoc | Model-agnostic, reliable | Slow for large datasets |

## Python implementation

```python
import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_classif, mutual_info_classif,
    chi2, RFE, SequentialFeatureSelector
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
from sklearn.inspection import permutation_importance

# --- Filter: Variance threshold ---
sel_var = VarianceThreshold(threshold=0.01)
X_filtered = sel_var.fit_transform(X)

# --- Filter: Mutual information ---
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_ranking = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# --- Filter: SelectKBest with chi-squared ---
sel_chi2 = SelectKBest(chi2, k=20)
X_chi2 = sel_chi2.fit_transform(X, y)
selected_features = X.columns[sel_chi2.get_support()]

# --- Wrapper: Recursive Feature Elimination ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf, n_features_to_select=15, step=1)
rfe.fit(X, y)
selected = X.columns[rfe.support_]

# --- Wrapper: Sequential forward selection ---
sfs = SequentialFeatureSelector(rf, n_features_to_select=15, direction='forward', cv=5)
sfs.fit(X, y)
selected = X.columns[sfs.get_support()]

# --- Embedded: Lasso ---
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X, y)
selected = X.columns[lasso.coef_ != 0]

# --- Embedded: Tree-based importance ---
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_features = importances.nlargest(20).index

# --- Permutation importance (more reliable) ---
perm_imp = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
perm_ranking = pd.Series(perm_imp.importances_mean, index=X.columns).sort_values(ascending=False)
```

## R implementation (if applicable)

```r
library(caret)
library(Boruta)  # Wrapper method using random forests

# --- Filter: correlation ---
cor_matrix <- cor(train_data[, numeric_cols])
high_cor <- findCorrelation(cor_matrix, cutoff = 0.90)
train_reduced <- train_data[, -high_cor]

# --- Wrapper: Boruta (finds all relevant features) ---
boruta_result <- Boruta(target ~ ., data = train_data, doTrace = 2)
selected_features <- getSelectedAttributes(boruta_result, withTentative = FALSE)

# --- Embedded: glmnet (Lasso) ---
library(glmnet)
cv_lasso <- cv.glmnet(as.matrix(X), y, alpha = 1, family = "binomial")
coefs <- coef(cv_lasso, s = "lambda.1se")
selected <- rownames(coefs)[coefs[, 1] != 0]
```

## Gotchas & lessons learned

1. **Filter methods miss interactions**: Correlation and mutual information evaluate features one at a time. A feature that is useless alone might be highly predictive in combination with another. Wrapper and embedded methods can catch this; filters cannot.

2. **Impurity-based importance is biased**: In random forests, features with more unique values (high cardinality) tend to get higher impurity-based importance scores, even if they are not truly more predictive. Use permutation importance instead for a fairer ranking.

3. **Feature selection must happen inside cross-validation**: If you select features on the full training set and then run cross-validation, you have leaked information. The feature selection step should be part of the CV pipeline (e.g., using `sklearn.pipeline.Pipeline`). This is one of the most common sources of optimistic performance estimates.

4. **Lasso with correlated features is unstable**: When two features are highly correlated, Lasso will arbitrarily pick one and zero out the other. Run it multiple times or use Elastic Net (which mixes L1 and L2) for more stable selection with correlated features.

5. **Do not confuse statistical significance with predictive usefulness**: A feature can be statistically significant (low p-value) but have negligible predictive power. Conversely, a feature that is not significant in isolation might be useful in a multivariate model.

6. **Removing correlated features blindly can hurt**: If features A and B are correlated at 0.95, removing one seems safe. But if they capture slightly different aspects of the same phenomenon, keeping both might still help prediction. Test both options.

7. **Domain knowledge should guide feature selection**: Statistical methods are tools, not oracles. If actuarial experience tells you that a feature matters (e.g., vehicle age in motor pricing), keep it even if the data does not show strong univariate correlation. The relationship might be non-linear or interaction-dependent.

8. **Stability of selection matters**: Run your feature selection on different bootstrap samples of your data. If a feature gets selected in 9 out of 10 runs, it is probably genuinely important. If it gets selected in 3 out of 10, treat it with suspicion. This is called "selection stability."

9. **p >> n is a red flag**: When you have more features than observations, most methods will struggle. In this regime, strong regularization (Lasso, Elastic Net) or aggressive filtering is essential. Consider techniques like PCA or UMAP for dimensionality reduction as an alternative.

10. **The best feature selection is often a combination**: Use filters for a quick initial screen (remove obviously useless features), then use embedded methods (tree importance or Lasso) for a more refined selection, then validate with wrapper methods or cross-validated performance.

## Resources

- **scikit-learn feature selection guide**: https://scikit-learn.org/stable/modules/feature_selection.html
- **An Introduction to Variable and Feature Selection (Guyon & Elisseeff, 2003)**: Classic survey paper. Available at: https://www.jmlr.org/papers/v3/guyon03a.html
- **Feature Engineering and Selection (Kuhn & Johnson)**: https://bookdown.org/max/FES/ -- chapters on filter and wrapper methods.
- **Boruta package for R**: https://cran.r-project.org/web/packages/Boruta/ -- implements an all-relevant feature selection algorithm.
- **Permutation importance explanation (scikit-learn)**: https://scikit-learn.org/stable/modules/permutation_importance.html
- **CAS Monograph on Predictive Modeling**: Covers variable selection in actuarial pricing context.
