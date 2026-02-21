# SHAP and Model Interpretability

## What it is

SHAP (SHapley Additive exPlanations) is a method for explaining individual predictions of any machine learning model. It assigns each feature a "contribution score" for every single prediction, answering the question: "How much did each feature push this prediction higher or lower compared to the average?" SHAP is grounded in Shapley values from cooperative game theory -- a principled mathematical framework for fairly distributing a collective payout among contributors. In the context of ML, the "payout" is the prediction, and the "contributors" are the features. SHAP has become the de facto standard for model interpretability in industry because it provides both local explanations (why this specific prediction) and global explanations (which features matter overall).

## Why it matters

For actuarial and data science practitioners, interpretability is not a nice-to-have -- it is a requirement. Regulators demand that you explain why a policyholder was charged a certain premium. Business stakeholders need to understand what drives model predictions to trust and act on them. And modelers themselves need interpretability to debug models, detect bias, and build confidence that the model is learning real patterns rather than artifacts.

Traditional linear models are inherently interpretable (each coefficient has a clear meaning), but modern ML models (gradient boosting, neural networks) are black boxes. SHAP cracks open those black boxes without sacrificing model performance. It bridges the gap between "accurate but opaque" and "interpretable but limited."

In insurance specifically, SHAP explanations can:
- Justify individual pricing decisions to regulators.
- Identify which risk factors drive the difference between two policyholders.
- Detect if the model is using prohibited variables (even indirectly, through correlated features).
- Communicate model behavior to underwriters who need to trust and override model recommendations.

## When to use it

- **After training any complex model** (gradient boosting, random forest, neural network) where coefficients are not directly interpretable.
- **When you need to explain individual predictions** -- "Why was this claim flagged as fraudulent?"
- **When you need global feature importance** that is more reliable than impurity-based importance.
- **When presenting model results to non-technical stakeholders** who need to understand what drives the model.
- **When debugging a model** -- unexpected SHAP patterns often reveal data leakage or feature engineering errors.
- **When checking for fairness** -- SHAP can reveal whether protected characteristics (age, gender, race) have undue influence, even when they are not direct model inputs.
- **When regulatory requirements demand model transparency** -- common in insurance, banking, and healthcare.
- **When comparing how different models use the same features** -- SHAP provides a common language across model types.

## How it works (intuitive)

### The game theory foundation

Imagine three friends -- Alice, Bob, and Carol -- work together on a project that earns a $300 bonus. How should they split the money fairly?

Shapley's insight (1953): Consider every possible combination of teammates. For each combination, measure the marginal contribution of each person -- how much more does the team earn when that person joins versus when they are absent? Average these marginal contributions across all possible orderings of team formation. The result is a fair allocation.

In SHAP, the "players" are features, the "game" is a single prediction, and the "payout" is the difference between the actual prediction and the average prediction:

$$f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j(x)$$

Where:
- $f(x)$ is the model's prediction for observation $x$.
- $\phi_0$ is the base value (the average prediction across the dataset).
- $\phi_j(x)$ is the SHAP value for feature $j$ -- how much feature $j$ contributed to pushing this prediction away from the average.

*Analogy*: Think of predicting an insurance premium. The average premium across all policies is $1,200 (the base value). For a specific policyholder, the model predicts $1,800. SHAP breaks down the $600 difference: age contributed +$200, driving record contributed +$300, vehicle type contributed +$150, and location contributed -$50. The contributions add up exactly to the prediction.

### Computing SHAP values

The exact Shapley value for feature $j$ is:

$$\phi_j = \sum_{S \subseteq N \setminus \{j\}} \frac{|S|! \, (|N| - |S| - 1)!}{|N|!} \left[ f(S \cup \{j\}) - f(S) \right]$$

Where:
- $N$ is the set of all features.
- $S$ is a subset of features that does not include $j$.
- $f(S \cup \{j\}) - f(S)$ is the marginal contribution of feature $j$ to subset $S$.

This formula considers every possible subset of features and measures how much adding feature $j$ changes the prediction. The factorials ensure each ordering is weighted equally.

**The computational problem**: With $p$ features, there are $2^p$ possible subsets. For 20 features, that is over a million subsets. For 100 features, it is more than the number of atoms in the universe.

**Solution**: SHAP uses efficient approximations:
- **TreeSHAP**: An exact, polynomial-time algorithm for tree-based models (random forest, XGBoost, LightGBM). This is fast and exact.
- **KernelSHAP**: A model-agnostic approximation that works for any model but is slower.
- **DeepSHAP**: An approximation for deep neural networks based on DeepLIFT.
- **LinearSHAP**: Exact values for linear models (the SHAP values are simply the feature contributions, which are just `coefficient * feature_value` in a linear model).

### Local vs global interpretability

**Local interpretability**: Explaining a single prediction. "Why did the model predict a 15% churn probability for this customer?" SHAP waterfall and force plots answer this.

**Global interpretability**: Understanding the model's overall behavior. "Which features matter most across all predictions?" SHAP summary plots (beeswarm) and importance bar charts answer this.

The beauty of SHAP is that global explanations are built by aggregating local explanations. You do not need different tools for different levels of explanation.

### SHAP visualization types

**Waterfall plot**: Shows how each feature contributes to a single prediction, starting from the base value and adding or subtracting each feature's contribution.

Reading a waterfall plot:
- Start at the base value (average prediction) at the bottom.
- Each bar shows one feature's contribution: red bars push the prediction higher, blue bars push it lower.
- The final value at the top is the actual prediction.

**Beeswarm plot (summary plot)**: Shows every SHAP value for every feature across all observations. Each dot is one observation.

Reading a beeswarm plot:
- Y-axis: Features ranked by importance (most important at top).
- X-axis: SHAP value (how much the feature pushes the prediction).
- Color: The actual feature value (red = high, blue = low).
- Spread: How variable the feature's effect is across observations.

For example, if "age" shows red dots (high age) on the right (positive SHAP), it means higher age pushes predictions upward.

**Force plot**: A compact horizontal visualization showing how features push a single prediction from the base value to the final prediction. Red features push the prediction higher; blue features push it lower.

**Dependence plot**: Shows the relationship between a feature's value (x-axis) and its SHAP value (y-axis), optionally colored by an interacting feature. This reveals non-linear effects and interactions.

Reading a dependence plot:
- If the pattern is a straight line, the feature has a linear effect.
- If the pattern curves, the feature has a non-linear effect.
- If the dots split into bands of different colors, there is an interaction with the coloring feature.

**Bar plot (mean absolute SHAP)**: The simplest global importance measure. For each feature, compute the mean of |SHAP values| across all observations. Larger values mean the feature has more influence on average.

### Model-agnostic vs model-specific

| Approach | Method | Speed | Accuracy |
|---|---|---|---|
| Model-specific | TreeSHAP (for trees) | Fast | Exact |
| Model-specific | LinearSHAP (for linear) | Very fast | Exact |
| Model-specific | DeepSHAP (for neural nets) | Moderate | Approximate |
| Model-agnostic | KernelSHAP | Slow | Approximate |

**When possible, use the model-specific method.** TreeSHAP for XGBoost/LightGBM/Random Forest is fast and exact. Only fall back to KernelSHAP when you have a model type that does not have a dedicated SHAP algorithm.

## Key concepts

### Shapley value properties (axioms)

Shapley values are the ONLY attribution method that satisfies all four of these fairness properties simultaneously:

1. **Efficiency**: The SHAP values sum to the difference between the prediction and the base value. Nothing is left unaccounted for.

$$\sum_{j=1}^{p} \phi_j = f(x) - E[f(X)]$$

2. **Symmetry**: If two features contribute identically in every coalition, they get the same SHAP value.

3. **Dummy (Null player)**: A feature that never changes the prediction (regardless of what other features are present) gets a SHAP value of zero.

4. **Additivity (Linearity)**: For a model that is the sum of two sub-models, the SHAP values are the sum of the SHAP values from each sub-model.

These axioms are what make SHAP values mathematically principled rather than ad hoc.

### Interaction effects

SHAP can decompose feature contributions into main effects and interaction effects:

$$\phi_{ij} = \text{SHAP interaction value between features } i \text{ and } j$$

The main effect is $\phi_{ii}$, and the interaction effect between $i$ and $j$ is $\phi_{ij}$ (for $i \neq j$). The sum of all interactions for feature $i$ equals its SHAP value:

$$\phi_i = \phi_{ii} + \sum_{j \neq i} \phi_{ij}$$

### SHAP vs other interpretability methods

| Method | Local | Global | Consistent | Model-agnostic |
|---|---|---|---|---|
| SHAP | Yes | Yes (aggregated) | Yes | Yes (KernelSHAP) |
| LIME | Yes | No (ad hoc) | No | Yes |
| Permutation importance | No | Yes | No | Yes |
| Partial dependence plots | No | Yes | N/A | Yes |
| Tree feature importance | No | Yes | No | No (trees only) |
| Coefficients (linear models) | Yes | Yes | Yes | No (linear only) |

### Base value

The base value $E[f(X)]$ is typically the mean prediction across the training dataset. All SHAP explanations are relative to this base value. For a binary classifier predicting probability, the base value might be 0.08 (the average claim rate). A policyholder with a prediction of 0.15 has SHAP values that sum to 0.15 - 0.08 = 0.07.

## Python implementation

```python
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# --- Train a model ---
model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# --- Create SHAP explainer (TreeSHAP for tree models) ---
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # shape: (n_samples, n_features)

# For newer SHAP API:
explanation = explainer(X_test)  # Returns an Explanation object

# --- Waterfall plot (single observation) ---
shap.waterfall_plot(explanation[0])  # Explain the first test observation

# --- Force plot (single observation) ---
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# --- Beeswarm plot (global view) ---
shap.summary_plot(shap_values, X_test, plot_type="dot")

# --- Bar plot (mean |SHAP|) ---
shap.summary_plot(shap_values, X_test, plot_type="bar")

# --- Dependence plot (one feature) ---
shap.dependence_plot("driver_age", shap_values, X_test,
                      interaction_index="vehicle_type")

# --- KernelSHAP for non-tree models ---
from sklearn.neural_network import MLPClassifier
nn_model = MLPClassifier(hidden_layer_sizes=(50, 25)).fit(X_train, y_train)

# Use a sample of training data as background
background = shap.sample(X_train, 100)
kernel_explainer = shap.KernelExplainer(nn_model.predict_proba, background)
kernel_shap_values = kernel_explainer.shap_values(X_test[:50])  # Slow -- limit samples

# --- SHAP interaction values (TreeSHAP only) ---
shap_interaction = explainer.shap_interaction_values(X_test)
# shape: (n_samples, n_features, n_features)

# --- Practical workflow for a pricing model ---
# 1. Train model
# 2. Compute SHAP values on validation set
# 3. Check beeswarm for global story
# 4. Check dependence plots for each key rating factor
# 5. Use waterfall plots to explain individual policy pricing
# 6. Check for unexpected patterns (data leakage, proxy discrimination)
```

## R implementation (if applicable)

```r
library(xgboost)
library(SHAPforxgboost)

# --- Train XGBoost model ---
dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
model <- xgb.train(params = list(objective = "binary:logistic", max_depth = 5),
                    data = dtrain, nrounds = 200)

# --- Compute SHAP values ---
shap_values <- shap.values(xgb_model = model, X_train = as.matrix(X_test))

# --- Summary plot (beeswarm) ---
shap.plot.summary(shap_values)

# --- Dependence plot ---
shap.plot.dependence(
  data_long = shap_values$shap_score,
  x = "driver_age",
  color_feature = "vehicle_type"
)

# --- Force plot for single observation ---
shap.plot.force_plot(
  shap_values$shap_score[1, ],
  base_value = shap_values$mean_shap_score
)
```

## Gotchas & lessons learned

1. **KernelSHAP is slow**: For tree models, always use `TreeExplainer`, not `KernelExplainer`. TreeSHAP is orders of magnitude faster and gives exact values. KernelSHAP on a dataset of 10,000 observations with 100 features can take hours.

2. **SHAP values depend on the background dataset**: KernelSHAP and some TreeSHAP configurations require a "background" dataset that represents the baseline. Different background datasets can give slightly different SHAP values. Use a representative sample of your training data.

3. **Correlated features complicate interpretation**: If two features are highly correlated, SHAP must split the credit between them. The split can be unstable and counterintuitive. A feature might get a low SHAP value not because it is unimportant, but because its information is shared with a correlated feature.

4. **SHAP values are not causal**: A high SHAP value for "zip_code" does not mean the zip code causes higher risk. It means the model uses zip code information to make predictions. Causal inference requires different tools.

5. **Beeswarm plots can be overwhelming**: With 100+ features, a beeswarm plot becomes unreadable. Show only the top 15-20 features. Use `max_display=20` in `shap.summary_plot()`.

6. **Waterfall plots are individual**: A waterfall plot explains ONE prediction. Do not generalize from a single example. Always complement individual explanations with global summaries.

7. **SHAP values are in the prediction space**: For a logistic regression predicting log-odds, SHAP values are in log-odds space. For a model predicting probabilities, they are in probability space. For a model predicting dollar amounts, they are in dollar space. Always check what space you are in.

8. **TreeSHAP has two variants**: `feature_perturbation="tree_path_dependent"` (default, fast, can attribute to features the model never saw) vs `feature_perturbation="interventional"` (uses background data, more aligned with causal interpretation). For most practical purposes, the default is fine.

9. **SHAP importance differs from permutation importance**: SHAP mean(|SHAP|) measures average contribution to predictions. Permutation importance measures how much performance degrades when a feature is shuffled. They answer different questions and can give different rankings.

10. **Use SHAP for model debugging**: Before trusting a model, check SHAP values for patterns that should not exist. If "policy_number" has high SHAP importance, you probably have data leakage. If "gender" has high importance in a jurisdiction where gender rating is prohibited, you have a compliance problem.

11. **SHAP for actuarial rate reviews**: SHAP provides a natural decomposition of premium into component parts: "base premium + age effect + territory effect + vehicle effect + ..." This is exactly the kind of breakdown actuaries need for rate filings and premium audits.

## Resources

- **SHAP documentation**: https://shap.readthedocs.io/en/latest/
- **Lundberg & Lee (2017)**: "A Unified Approach to Interpreting Model Predictions" -- the original SHAP paper. https://arxiv.org/abs/1705.07874
- **Lundberg et al. (2020)**: "From local explanations to global understanding with explainable AI for trees" -- TreeSHAP and its applications. https://www.nature.com/articles/s42256-019-0138-9
- **Christoph Molnar -- Interpretable Machine Learning**: https://christophm.github.io/interpretable-ml-book/ -- excellent free book covering SHAP, LIME, PDP, and more.
- **SHAP GitHub repository**: https://github.com/shap/shap -- code, tutorials, and examples.
- **SHAPforxgboost R package**: https://cran.r-project.org/web/packages/SHAPforxgboost/ -- SHAP visualization for XGBoost in R.
- **Shapley (1953)**: "A Value for n-Person Games" -- the original game theory paper that started it all.
