# Propensity Score Matching

## What it is

Propensity Score Matching (PSM) is a statistical technique for estimating the causal effect of a treatment, intervention, or exposure when you cannot run a randomized experiment. The core idea is to create a "pseudo-experiment" from observational data by matching each treated unit with one or more untreated units that had a similar probability (propensity) of receiving the treatment. By comparing outcomes between these matched pairs, you approximate what a randomized experiment would have produced -- under certain assumptions. PSM is one of the most widely used methods in the causal inference toolkit, bridging the gap between "we observed a difference" and "the treatment caused a difference."

## Why it matters

- **Most business data is observational**: You cannot always randomize. Insurance companies cannot randomly assign customers to receive discounts; hospitals do not randomly deny treatments. Yet you still need to estimate causal effects from the data you have.
- **Insurance applications**: Did the loyalty discount reduce lapse rates, or did loyal customers just happen to get the discount? Did the wellness program reduce claims, or were healthier people more likely to enroll? PSM helps disentangle treatment effects from selection effects.
- **Policy evaluation**: Government and corporate policies are rarely randomized. PSM is a standard tool for evaluating the effect of policy changes (new underwriting guidelines, new claims processes, regulatory changes) from administrative data.
- **Complements A/B testing**: When A/B testing is infeasible (ethical, logistical, or political constraints), PSM provides a second-best alternative for causal estimation.
- **Actuarial relevance**: Actuaries often need to assess the impact of interventions on loss experience (e.g., "Did implementing telematics reduce auto claims severity?"). PSM provides a rigorous framework for this.

## When to use it

- You have observational data with a binary treatment variable (treated vs. untreated).
- Randomization was not possible, and you suspect that treated and untreated groups differ on important characteristics (confounders).
- You believe that all relevant confounders are observed in your data (the "no unobserved confounders" assumption -- see Gotchas).
- You want to estimate the Average Treatment Effect on the Treated (ATT) or the Average Treatment Effect (ATE).
- You have a reasonable sample size with sufficient overlap between treated and untreated groups.
- You prefer a transparent, interpretable approach over more complex causal methods (though at the cost of stronger assumptions).

## How it works (intuitive)

### The "apples to apples" analogy

Suppose you want to know if attending a tutoring program improves test scores. You cannot just compare tutored students to non-tutored students because motivated students (who would have scored well anyway) are more likely to attend tutoring. The comparison is apples to oranges.

PSM fixes this by finding, for each tutored student, a non-tutored student who looks just like them -- same baseline test scores, same demographics, same motivation level (if you can measure it). Now you are comparing apples to apples: tutored students vs. their matched non-tutored "twins." Any remaining difference in test scores can be attributed to the tutoring itself.

### Step-by-step process

**Step 1: Identify the treatment and the confounders**

- **Treatment**: The binary variable (e.g., received loyalty discount = 1, did not = 0).
- **Confounders**: Variables that affect both the treatment assignment AND the outcome. For example, tenure (long-tenured customers are both more likely to receive the discount and less likely to lapse).

The key insight: you need to identify and include all variables that influence both the treatment and the outcome. Missing a confounder biases your results.

**Step 2: Estimate the propensity score**

The propensity score is the probability of receiving treatment, given the observed confounders:

$$e(X_i) = P(\text{Treatment}_i = 1 | X_i)$$

Typically estimated using logistic regression:

$$\log \frac{e(X_i)}{1 - e(X_i)} = \beta_0 + \beta_1 X_{i1} + \cdots + \beta_p X_{ip}$$

You can also use more flexible models (random forests, gradient boosting) to estimate propensity scores, but logistic regression is the standard.

The propensity score is a **dimension reduction device**: instead of matching on many confounders simultaneously (the curse of dimensionality), you match on a single number -- the propensity score. Rosenbaum and Rubin (1983) proved that if the confounders suffice for unconfoundedness, then matching on the propensity score also suffices.

**Step 3: Match treated and untreated units**

Using the estimated propensity scores, match each treated unit to one or more untreated units with similar scores. Common matching algorithms:

| Method | How it works | Pros | Cons |
|--------|-------------|------|------|
| **Nearest neighbor** | Match each treated unit to the untreated unit with the closest propensity score | Simple, intuitive | Can match distant pairs if overlap is poor |
| **Caliper matching** | Like nearest neighbor, but only allow matches within a specified distance (caliper) | Prevents bad matches | May leave treated units unmatched |
| **Radius matching** | Match to all untreated units within the caliper | Uses more data | Varying number of matches per treated unit |
| **Kernel matching** | Weighted average of all untreated units, with weights inversely proportional to distance | Uses all data, smooth | Less intuitive |
| **Mahalanobis distance + PS** | Match on both propensity score and key covariates using Mahalanobis distance | More exact on key vars | Higher dimensionality |

**Step 4: Check covariate balance**

After matching, the treated and untreated groups should look similar on all confounders. This is the most critical diagnostic step. Check balance using:

- **Standardized mean differences (SMD)**: For each covariate, compute the difference in means between treated and untreated groups, standardized by the pooled standard deviation. A rule of thumb: $|SMD| < 0.1$ is good balance.

$$SMD = \frac{\bar{X}_{\text{treated}} - \bar{X}_{\text{untreated}}}{\sqrt{\frac{s_{\text{treated}}^2 + s_{\text{untreated}}^2}{2}}}$$

- **Variance ratios**: The ratio of variances should be close to 1 (between 0.5 and 2.0).
- **Visual checks**: Histograms or density plots of propensity scores and key covariates, before and after matching.

If balance is poor, revise the propensity score model (add interactions, non-linear terms) or try a different matching method.

**Step 5: Estimate the treatment effect**

On the matched sample, compute the difference in outcomes:

$$\hat{\tau}_{ATT} = \frac{1}{n_{\text{treated}}} \sum_{i \in \text{treated}} \left[ Y_i - Y_{j(i)} \right]$$

where $Y_{j(i)}$ is the outcome of the matched untreated unit for treated unit $i$.

For standard errors, use:
- Bootstrap standard errors (most common and robust).
- Abadie-Imbens standard errors (accounts for the matching estimation step).

**Step 6: Sensitivity analysis**

Since the key assumption (no unobserved confounders) is untestable, assess how sensitive your results are to potential hidden confounders. Rosenbaum bounds or the E-value quantify how strong an unobserved confounder would need to be to overturn your conclusion.

### The dating app analogy

Think of PSM like a dating app matching system. The "treatment" is whether someone uses a premium feature. The "outcome" is whether they get a date. You cannot just compare premium users to free users because premium users might be more attractive or more active. So you compute a "propensity to go premium" score for everyone, then match each premium user with a free user who had the same likelihood of going premium but chose not to. Now you can fairly compare outcomes.

## Key concepts

### Propensity score theorem (Rosenbaum & Rubin, 1983)

If treatment assignment is strongly ignorable given confounders $X$ (i.e., all confounders are observed):

$$(Y(1), Y(0)) \perp T | X$$

then it is also strongly ignorable given the propensity score:

$$(Y(1), Y(0)) \perp T | e(X)$$

This means matching on the scalar propensity score $e(X)$ is sufficient for removing confounding, even when $X$ is high-dimensional.

### Average Treatment Effect (ATE) vs. Average Treatment Effect on the Treated (ATT)

| Estimand | Definition | Interpretation |
|----------|-----------|----------------|
| ATE | $E[Y(1) - Y(0)]$ | Average effect for the entire population |
| ATT | $E[Y(1) - Y(0) | T = 1]$ | Average effect for those who actually received treatment |

**ATT is more natural for PSM** because matching finds untreated "twins" for treated units. It answers: "Among those who received the treatment, what was the average effect?"

**ATE requires overlap in both directions**: treated units need untreated twins AND untreated units need treated twins. This is harder to achieve and often requires inverse probability weighting (IPW) rather than matching.

### Inverse Probability Weighting (IPW)

An alternative to matching that uses the propensity score as weights:

$$\hat{\tau}_{ATE}^{IPW} = \frac{1}{n} \sum_{i=1}^{n} \left[ \frac{T_i Y_i}{e(X_i)} - \frac{(1 - T_i) Y_i}{1 - e(X_i)} \right]$$

IPW uses all observations (no matching), but can be unstable when propensity scores are close to 0 or 1 (extreme weights). Trimming or stabilized weights help.

### Doubly Robust Estimation

Combines propensity score weighting with outcome regression. If *either* the propensity score model or the outcome model is correctly specified, the estimator is consistent. This provides an insurance policy against model misspecification.

### Common Support / Overlap

Matching only works in the region where both treated and untreated groups exist (common support). If a treated unit has a propensity score of 0.95 and no untreated unit has a score above 0.80, that treated unit has no good match and should be excluded.

$$0 < e(X) < 1 \quad \text{for all } X$$

Visually: plot the propensity score distributions for treated and untreated. They should overlap substantially.

### Key assumptions

| Assumption | Meaning | Testable? |
|-----------|---------|----------|
| **Unconfoundedness** (no hidden confounders) | All variables affecting both treatment and outcome are observed and included | No (this is the fundamental limitation) |
| **Overlap** (common support) | For every treated unit, there exist similar untreated units | Yes (check propensity score distributions) |
| **SUTVA** (no interference) | One unit's treatment does not affect another's outcome | Depends on context |
| **Correct propensity score model** | The logistic regression (or other model) correctly captures the treatment assignment mechanism | Partially (balance checks are an indirect test) |

## Python implementation

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# --- Step 1: Estimate propensity scores ---
def estimate_propensity_scores(df, treatment_col, covariate_cols):
    """
    Estimate propensity scores using logistic regression.
    """
    X = df[covariate_cols]
    T = df[treatment_col]

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, T)

    df = df.copy()
    df['propensity_score'] = model.predict_proba(X)[:, 1]

    return df, model

# --- Step 2: Nearest neighbor matching ---
def match_nearest_neighbor(df, treatment_col, caliper=None):
    """
    1:1 nearest neighbor matching on propensity score.
    """
    treated = df[df[treatment_col] == 1].copy()
    untreated = df[df[treatment_col] == 0].copy()

    # Fit nearest neighbor model on untreated propensity scores
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean')
    nn.fit(untreated[['propensity_score']])

    distances, indices = nn.kneighbors(treated[['propensity_score']])

    # Apply caliper if specified
    if caliper is not None:
        within_caliper = distances.flatten() <= caliper
        treated = treated[within_caliper]
        indices = indices[within_caliper]
        distances = distances[within_caliper]

    # Get matched untreated indices
    matched_untreated_idx = untreated.iloc[indices.flatten()].index

    # Combine matched sample
    matched_treated = treated.copy()
    matched_treated['match_distance'] = distances.flatten()
    matched_untreated = untreated.loc[matched_untreated_idx].copy()

    matched_df = pd.concat([matched_treated, matched_untreated])

    return matched_df, matched_treated, matched_untreated

# --- Step 3: Check covariate balance ---
def check_balance(df, treatment_col, covariate_cols):
    """
    Compute standardized mean differences for covariate balance.
    """
    treated = df[df[treatment_col] == 1]
    untreated = df[df[treatment_col] == 0]

    balance = []
    for col in covariate_cols:
        mean_t = treated[col].mean()
        mean_u = untreated[col].mean()
        std_t = treated[col].std()
        std_u = untreated[col].std()
        pooled_std = np.sqrt((std_t**2 + std_u**2) / 2)

        smd = (mean_t - mean_u) / pooled_std if pooled_std > 0 else 0
        var_ratio = (std_t**2) / (std_u**2) if std_u > 0 else np.inf

        balance.append({
            'covariate': col,
            'mean_treated': mean_t,
            'mean_untreated': mean_u,
            'smd': smd,
            'abs_smd': abs(smd),
            'variance_ratio': var_ratio
        })

    balance_df = pd.DataFrame(balance)
    return balance_df

# --- Step 4: Estimate ATT ---
def estimate_att(treated_outcomes, matched_untreated_outcomes):
    """
    Estimate ATT with bootstrap confidence interval.
    """
    att = np.mean(treated_outcomes - matched_untreated_outcomes)

    # Bootstrap CI
    n_boot = 1000
    boot_atts = []
    n = len(treated_outcomes)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, size=n)
        boot_att = np.mean(treated_outcomes.iloc[idx].values -
                          matched_untreated_outcomes.iloc[idx].values)
        boot_atts.append(boot_att)

    ci_lower = np.percentile(boot_atts, 2.5)
    ci_upper = np.percentile(boot_atts, 97.5)

    print(f"ATT: {att:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

    return att, ci_lower, ci_upper

# --- Using the causalinference library ---
# pip install causalinference
from causalinference import CausalModel

# Y: outcome, T: treatment, X: confounders
model = CausalModel(Y, T, X)
model.est_propensity_s()    # Estimate propensity score with variable selection
model.trim_s()              # Trim non-overlapping observations
model.stratify_s()          # Stratify on propensity score
model.est_via_matching()    # Matching estimator
model.est_via_weighting()   # IPW estimator
model.est_via_ols()         # OLS with propensity score adjustment

print(model.estimates)
```

Key libraries:
- **causalinference**: Clean API for propensity score estimation, matching, IPW, and OLS adjustment. [causalinference docs](https://causalinferenceinpython.org/)
- **DoWhy**: Microsoft's causal inference library with a graphical approach. [DoWhy docs](https://www.pywhy.org/dowhy/)
- **scikit-learn**: Logistic regression for propensity score estimation.
- **statsmodels**: For regression-based outcome analysis on matched data.

## R implementation (if applicable)

```r
# install.packages("MatchIt")
# install.packages("cobalt")
library(MatchIt)
library(cobalt)

# Propensity score matching
m <- matchit(treatment ~ age + tenure + premium + claims_history,
             data = df,
             method = "nearest",
             distance = "glm",     # logistic regression
             caliper = 0.2,        # 0.2 SD of propensity score
             ratio = 1)            # 1:1 matching

summary(m)

# Covariate balance
bal.tab(m, thresholds = c(m = 0.1))  # SMD threshold of 0.1
love.plot(m, abs = TRUE)             # Love plot (visual balance check)

# Extract matched data and estimate effect
matched_data <- match.data(m)
lm(outcome ~ treatment, data = matched_data, weights = weights)
```

**Python vs. R tradeoff**: R's `MatchIt` package is the gold standard for PSM -- it has more matching methods, better diagnostics, and excellent documentation. The `cobalt` package provides the best balance checking tools. Python's `causalinference` and `DoWhy` are good but less feature-rich for matching specifically. For PSM, R has a significant advantage.

## Gotchas & lessons learned

### 1. The unobserved confounder problem
The biggest limitation: PSM assumes no unobserved confounders. If there is a variable that affects both treatment and outcome that you did not include, your estimate is biased. This assumption is untestable. Always conduct a sensitivity analysis (Rosenbaum bounds) to assess how robust your conclusion is.

### 2. Propensity score estimation is a means, not an end
Do not evaluate the propensity score model by its predictive accuracy (AUC). The goal is not to predict treatment perfectly -- it is to achieve covariate balance. A model with moderate AUC but excellent balance is better than one with high AUC but poor balance.

### 3. Always check balance, always
Balance checking is not optional. If post-matching SMDs are above 0.1 for key covariates, your analysis is compromised. Iterate on the propensity score model until balance is achieved. A love plot (dot plot of SMDs before and after matching) is the standard diagnostic.

### 4. Poor overlap = poor estimates
If the propensity score distributions of treated and untreated groups barely overlap, matching is forced to pair dissimilar units. The result is biased and imprecise. Check overlap visually and trim observations outside the common support.

### 5. Matching without replacement vs. with replacement
Matching without replacement (each untreated unit can be a match for at most one treated unit) ensures unique matches but can lead to poor matches when there are many treated units relative to untreated ones. Matching with replacement allows reuse and typically improves balance but requires adjusting standard errors.

### 6. The caliper choice is subjective
A common recommendation: caliper = 0.2 * standard deviation of the logit propensity score. But there is no universal "right" caliper. Too tight excludes too many treated units; too loose allows bad matches.

### 7. PSM can increase bias when overlap is poor
Counterintuitively, King & Nielsen (2019) showed that PSM can increase imbalance and bias compared to other matching methods (like Mahalanobis distance matching or coarsened exact matching) in some settings. Consider alternatives, especially when overlap is limited.

### 8. Standard errors need special treatment
Naive standard errors from a t-test on matched data underestimate uncertainty because they ignore the matching estimation step. Use bootstrap standard errors or Abadie-Imbens standard errors.

### 9. The propensity score paradox
Including variables that predict treatment but not outcome (instrumental variables) in the propensity score model can actually increase variance without reducing bias. Include variables that affect the outcome, not just variables that predict treatment.

### 10. PSM is one tool, not the only tool
Consider alternatives: IPW (uses all data), doubly robust estimation (insurance against misspecification), regression adjustment, or coarsened exact matching (CEM). PSM is popular because of its intuitive appeal, but it is not always the best estimator.

## Resources

- **Rosenbaum & Rubin (1983)**: "The Central Role of the Propensity Score in Observational Studies for Causal Effects" -- the foundational paper.
- **King & Nielsen (2019)**: "Why Propensity Scores Should Not Be Used for Matching" -- important critique. [Link](https://gking.harvard.edu/publications/why-propensity-scores-should-not-be-used-formatching)
- **MatchIt R package**: [CRAN - MatchIt](https://cran.r-project.org/package=MatchIt) -- the best implementation.
- **cobalt R package**: [CRAN - cobalt](https://cran.r-project.org/package=cobalt) -- balance diagnostics.
- **causalinference Python**: [causalinferenceinpython.org](https://causalinferenceinpython.org/)
- **DoWhy**: [pywhy.org/dowhy](https://www.pywhy.org/dowhy/) -- Microsoft's causal inference framework.
- **Cunningham (2021)**: "Causal Inference: The Mixtape" -- free textbook with excellent PSM chapter. [Link](https://mixtape.scunning.com/)
- **Hernan & Robins**: "Causal Inference: What If" -- free textbook on causal inference methods. [Link](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/)
- **Actuarial application**: Austin (2011), "An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies" -- excellent tutorial paper.
