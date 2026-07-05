# Uplift Modeling

## Meta-learners

### T-learner (two-model)

Train two models:

- $\mu_1(x) = E[Y | X, T = 1]$ on treated data.
- $\mu_0(x) = E[Y | X, T = 0]$ on control data.

Predict uplift:

$$
\hat{\tau}(x) = \hat{\mu}_1(x) - \hat{\mu}_0(x)
$$

**Pros.** Simple. Reuses existing classifier tech.
**Cons.** Two models have different biases; their difference is noisy. Poor when one class (usually treated) is small.

### S-learner (single-model)

Train one model on all data with treatment as a feature:

$$
\mu(x, t) = E[Y | X, T = t]
$$

Predict uplift:

$$
\hat{\tau}(x) = \hat{\mu}(x, 1) - \hat{\mu}(x, 0)
$$

**Pros.** Single model, often more stable.
**Cons.** Model may learn to ignore T if other features are strong; underestimates uplift.

### R-learner

Target the residuals:

$$
\tilde{Y} = Y - \hat{m}(x), \quad \tilde{T} = T - \hat{e}(x)
$$

where $\hat{m}(x) = E[Y | X]$ and $\hat{e}(x) = P(T | X)$ (the propensity score). Then fit:

$$
\tau(x) = \arg\min_\tau \sum_i (\tilde{Y}_i - \tilde{T}_i \tau(x_i))^2 \cdot \text{weights}
$$

**Pros.** More statistically principled; often best offline.
**Cons.** More complex, needs propensity and outcome models.

### X-learner

For imbalanced treatment. Uses treated/control to impute counterfactual outcomes, then fits two models on imputed uplifts.

### Tree-based direct methods

- **Uplift trees.** Split criterion that maximizes the difference in treatment effect across child nodes.
- **Causal forests.** Generalized random forests for treatment effect estimation.

Libraries: `causalml` (Uber), `scikit-uplift`, `econml` (Microsoft), `grf` (R).

## Evaluation

### Qini curve

Rank units by predicted uplift descending. For the top X percent:

$$
Q(X) = \frac{n_T^X}{n_T} \cdot Y_T^X - \frac{n_C^X}{n_C} \cdot Y_C^X
$$

Plots cumulative uplift vs percent targeted. Area under the curve is the Qini coefficient.

### Uplift @ K

Take the top K percent by predicted uplift. Compute the observed treatment effect on this subset:

$$
\text{uplift@K} = \bar{Y}_{T, \text{topK}} - \bar{Y}_{C, \text{topK}}
$$

Business-friendly: "targeting the top 10 percent yields 2x the uplift of random."

### Comparison baseline

Random targeting gives a straight line from origin to total uplift. Any useful model's Qini curve should lie above this.

## Practical tips

- **Cross-validation.** Train/test split respecting group (no user in both).
- **Confidence intervals.** Bootstrap over users.
- **Interpretation.** Feature importance on the uplift prediction (not on the outcome).
- **Negative uplift.** Some segments respond negatively. Do not treat them.

## Cross-links

- `02_ab_testing_math.md`
- `04_sensitivity_and_assumptions.md`
- `docs/theory/08_experimentation_causal_primer.md`
