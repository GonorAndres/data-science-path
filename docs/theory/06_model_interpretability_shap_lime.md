# Model Interpretability with SHAP, LIME, and Friends

## Why interpretability matters

- **Debugging.** An unexplainable model is hard to fix. If you do not know why it predicts, you cannot reason about why it fails.
- **Stakeholder trust.** Business partners adopt models they understand. A black box with no explanation gets second-guessed.
- **Regulatory requirements.** Finance and insurance regulators (CNBV in Mexico, EU AI Act) increasingly require model explanation. Adverse action letters for credit decisions require feature-level reasons.
- **Fairness auditing.** You cannot audit what you cannot see. Interpretability is the entry point to bias analysis.

## Global vs local interpretability

- **Global.** How the model behaves on average: which features matter overall, what is the sign of their effect.
- **Local.** Why the model made a specific prediction for a specific row.

Both are needed. Global tells you the model's logic; local tells you about a particular decision.

## Simple approaches

### Linear regression coefficients

For linear models, the coefficient is the direct answer (with care for scale and correlation). Standardize features first if you want coefficients comparable to each other.

### Logistic regression odds ratios

$\exp(\beta_j)$ is the multiplicative effect on odds of a one-unit increase in $X_j$, holding other features constant. Interpretable out of the box.

### Decision tree paths

A single tree can be traced from root to leaf to explain any prediction. Decision paths are human-readable. The tradeoff: single trees are usually weaker than ensembles.

### Feature importance (tree ensembles)

`feature_importances_` in scikit-learn and XGBoost reports average gain per feature across all splits. Rough guide but flawed:

- High-cardinality features get inflated importance (more split opportunities).
- Correlated features split importance; both look weaker than they are.

## Permutation importance

Shuffle a feature's values in the test set, remeasure performance. The drop equals that feature's contribution to predictions.

Pros: model-agnostic, uses test data (what we actually care about).
Cons: shuffling breaks feature correlations, can underestimate important features when correlated with others.

## Partial dependence plots (PDP)

For a feature $X_j$, show how the average prediction changes as $X_j$ varies, holding other features at their empirical distribution.

Intuition: "what happens to the prediction if we move this feature from low to high?" Useful for seeing non-linear effects.

Caveats:

- Averaging can hide subgroup effects (feature matters only for some rows).
- Assumes independence between $X_j$ and other features; this can produce unrealistic combinations.

## Individual Conditional Expectation (ICE) plots

Like PDP but shows one line per row instead of an average. Captures heterogeneity: if the lines fan out, the effect depends on other features.

Read PDP and ICE together. PDP gives the average story; ICE shows variance around it.

## LIME (Local Interpretable Model-agnostic Explanations)

For a single prediction:

1. Generate perturbations of the input row.
2. Get the model's predictions for each perturbation.
3. Weight perturbations by similarity to the original row.
4. Fit a simple (linear) model to the predictions.
5. Report the linear model's coefficients as the local explanation.

Pros: model-agnostic, simple intuition.
Cons: can be unstable (different random perturbations give different explanations); domain of validity unclear; sensitive to the choice of kernel and neighborhood size.

## SHAP (Shapley Additive exPlanations)

Based on Shapley values from cooperative game theory. Each feature's contribution to a prediction is the average marginal contribution across all possible feature subsets.

Mathematically:

$$
\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ v(S \cup \{j\}) - v(S) \right]
$$

where $v(S)$ is the model's prediction with only features in $S$ known.

Key properties:

- **Efficiency.** Feature contributions sum to the difference between the prediction and the baseline.
- **Symmetry.** If two features always contribute equally, their Shapley values are equal.
- **Null player.** A feature that never contributes gets zero.
- **Additivity.** Shapley values of a linear combination of models equal the same linear combination of Shapley values.

These axioms make SHAP the strongest theoretical grounding for feature attribution.

### TreeSHAP

Efficient exact computation of Shapley values for tree ensembles (XGBoost, LightGBM, Random Forest). The original Lundberg et al. (2018) result establishes worst-case complexity $O(T L D^2)$ where $T$ is the number of trees, $L$ is the maximum number of leaves, and $D$ is the maximum depth. Fast enough for batch computation; orders of magnitude faster than KernelSHAP on tree models.

### KernelSHAP

Model-agnostic approximation. Runs many weighted predictions, fits a weighted linear model. Slower but works for any model.

### DeepSHAP

Extension for deep networks using backpropagation of contributions through layers.

## Using SHAP in practice

### Summary plot

Scatter of SHAP values per feature across many rows. Each point is one row's SHAP value for that feature, colored by the feature's value.

Read it as:

- Feature at the top: most important globally.
- Wide spread: large effect range.
- Color pattern: does high feature value push prediction up or down?

### Dependence plot

For one feature, plot SHAP value vs feature value, colored by an interacting feature. Reveals non-linearities and interactions.

### Waterfall / force plot

For a single row, show the additive contribution of each feature starting from the baseline, ending at the prediction.

## Interpretability for neural networks

### Integrated Gradients

Computes the integral of gradients along a path from a baseline to the input. Attributes each input dimension's contribution.

### Grad-CAM (for CNNs)

Uses gradients flowing into the last convolutional layer to produce a coarse localization map of where the model looked to make its decision. Essential for computer-vision debugging.

### Attention visualization (for transformers)

Plot attention weights between tokens. Useful but contested: Jain and Wallace (2019) argue "attention is not explanation"; Wiegreffe and Pinter (2019) reply "attention is not not explanation", showing that under the right experimental setup attention is informative. The literature is unsettled. Treat attention maps as a debugging aid, not a faithful causal account.

## Conditional vs interventional vs marginal SHAP

SHAP is defined relative to a value function $v(S)$, the expected model output when only features in subset $S$ are observed. There are three canonical choices, and they materially change the answer.

### Conditional (observational) value function

$$
v(S) = E[f(X) \mid X_S = x_S].
$$

The expectation is over the joint distribution of the missing features given the observed ones. Aas, Jullum, Loland (2021) show that KernelSHAP with empirical conditional sampling estimates this version. The interpretation: "the model's expected prediction given what we can see about this row".

### Interventional (do-calculus) value function

$$
v(S) = E[f(X) \mid \mathrm{do}(X_S = x_S)] = E_{X_{\bar S}}[f(x_S, X_{\bar S})].
$$

Here the missing features are sampled from their marginal distribution, severing their dependence on the observed features. Janzing, Minorics, Blobaum (2020) argue this is the right notion when the goal is causal contribution. TreeSHAP's `feature_perturbation="interventional"` mode and the standard tree-path algorithm with marginal sampling implement this.

### Marginal (path-dependent) value function

TreeSHAP's path-dependent algorithm (the original Lundberg 2017 default) computes a quantity that is neither pure conditional nor pure interventional; it weights paths by the empirical training distribution along the tree. Often close to interventional in practice but not identical.

### Why this matters

When features are correlated, the three value functions can give very different rankings. A feature that has high conditional impact (knowing it tells you about other features that drive the prediction) may have low interventional impact (intervening on it directly does not change the prediction). For a regulator-facing role, **state which value function you used and why**. The default is rarely innocent.

## Accumulated Local Effects (ALE) plots

PDP averages predictions across the marginal distribution of other features, including unrealistic combinations. When features are correlated, this produces misleading curves. ALE (Apley & Zhu 2020) fixes this by averaging the local effect within small slices of the feature.

For feature $X_j$:

1. Bin $X_j$ into $K$ intervals.
2. For each interval $k$, compute the average finite-difference effect $f(\text{upper}, X_{-j}) - f(\text{lower}, X_{-j})$ using only rows with $X_j$ in interval $k$.
3. Accumulate (cumulative sum across bins) and center.

The result is a curve that respects the joint distribution: no extrapolation into empty regions of feature space. Use ALE as the default in any setting where features are non-trivially correlated. PDP remains useful when features are nearly independent.

## Anchors

Ribeiro et al. (2018) propose Anchors as a complement to LIME: instead of a local linear surrogate, find a minimal set of feature conditions (an "anchor") that, when held fixed, guarantees the same prediction with high probability across perturbations. Anchors are rule-form explanations ("if income > 50K and age < 40, the model predicts approve with 95% precision"). Easier to communicate than LIME coefficients; harder to compute.

## Counterfactual explanations

Wachter et al. (2017) frame the explanation question as "what is the smallest change to this input that would flip the prediction?" The answer is the closest counterfactual on the decision boundary. The DiCE framework (Mothilal, Sharma, Tan 2020) generalizes this to diverse counterfactuals subject to actionability and feasibility constraints.

Counterfactuals translate naturally to user-facing explanations: "your application would have been approved if your debt-to-income ratio were below 0.35". They are also the foundation of GDPR's right to explanation and underlie the "reason codes" required in US adverse-action letters.

## Regulatory reason codes

For credit decisions in the US (Equal Credit Opportunity Act, Regulation B) and analogous insurance regulations, an adverse-action letter must list the principal reasons for denial. SHAP values map naturally to reason codes:

1. Compute SHAP values for the denied applicant.
2. Sort features by SHAP value (most negative first, since denial is the unfavorable outcome).
3. Translate each feature into a regulator-approved reason category (delinquencies in last 24 months, debt-to-income ratio, length of credit history, etc.).
4. Report the top 4 to 5 reasons.

For a Mexican-context analogue, CONDUSEF and CNBV expect similar transparency in scoring-driven credit decisions; the same SHAP-to-reason-code translation works.

A few cautions:

- The choice of value function (conditional, interventional, marginal) affects which feature shows up as a reason. Document the choice.
- A SHAP-derived reason code is a **statistical** explanation, not a causal one. The reader can rebuild the decision but cannot infer that intervening on the feature would change the outcome.
- Proxy features (zip code as a proxy for race) will appear as legitimate reasons. The fairness audit (essay 07) is a separate mandatory check.

## Pitfalls

- **Correlated features.** Shapley distributes credit; permutation importance shuffles one at a time. Both struggle with correlation.
- **Overinterpretation.** A SHAP summary plot is not a causal diagram. Features correlated with the target have high SHAP whether or not they cause it.
- **Regulatory risk.** Feature importance does not equal legal compliance. If a feature is a proxy for a protected attribute (zip code proxies for race), SHAP will happily show it is important; you still need a fairness audit (see `07_fairness_and_responsible_ai.md`).
- **Speed.** TreeSHAP is fast; KernelSHAP and DeepSHAP can be painfully slow. Budget accordingly.

## Reporting template

When presenting a model's interpretability:

1. Top 10 features by global SHAP importance.
2. Two or three dependence plots for the most business-critical features.
3. One or two waterfall plots for example rows (one typical, one outlier).
4. A written paragraph translating the top findings to business language.
5. Caveats: mention correlation, potential proxies, and limitations.

## Linking to the portfolio

- `nlp-reviews-multilingual`: SHAP for the TF-IDF baseline; LIME for transformer predictions.
- `timeseries-retail-forecast`: TreeSHAP on LightGBM to show lag and promotion effects.
- `recsys-movielens`: show per-user top features driving recommendations.
- `cv-defect-detection`: Grad-CAM heatmaps overlaid on images.
- `causal-uplift-marketing`: SHAP explains the uplift model's feature contributions.

## References

- Lundberg, Lee, "A Unified Approach to Interpreting Model Predictions", NeurIPS 2017.
- Ribeiro, Singh, Guestrin, "Why Should I Trust You? Explaining the Predictions of Any Classifier", KDD 2016 (LIME).
- Molnar, *Interpretable Machine Learning*, 2nd ed., 2022 (free online).
- Sundararajan, Taly, Yan, "Axiomatic Attribution for Deep Networks" (Integrated Gradients), 2017.
- Jain, Wallace, "Attention is not Explanation", NAACL 2019.

## Exercises

1. What is the difference between a marginal SHAP value and a conditional SHAP value for a feature $X_j$ when $X_j$ is correlated with $X_k$?
2. A PDP for feature "income" shows a positive relationship with churn, but you suspect it is driven by correlation with "tenure". Which plot do you use to check, and why?
3. Explain why ALE plots are preferred over PDP when features are correlated.
4. You need to write an adverse-action letter for a declined credit application. How do you convert SHAP values into reason codes?
5. What does the Shapley efficiency axiom guarantee about the sum of SHAP values?
- Wiegreffe, Pinter, "Attention is not not Explanation", EMNLP 2019.
- Janzing, Minorics, Blobaum, "Feature relevance quantification in explainable AI: a causal problem", AISTATS 2020.
- Aas, Jullum, Loland, "Explaining individual predictions when features are dependent: More accurate approximations to Shapley values", Artificial Intelligence, 2021.
- Lundberg et al., "Consistent Individualized Feature Attribution for Tree Ensembles", arXiv:1802.03888, 2018.
- Apley, Zhu, "Visualizing the Effects of Predictor Variables in Black Box Supervised Learning Models" (ALE), JRSS-B, 2020.
- Ribeiro, Singh, Guestrin, "Anchors: High-Precision Model-Agnostic Explanations", AAAI 2018.
- Wachter, Mittelstadt, Russell, "Counterfactual Explanations Without Opening the Black Box", Harvard JOLT, 2017.
- Mothilal, Sharma, Tan, "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations" (DiCE), FAccT 2020.
