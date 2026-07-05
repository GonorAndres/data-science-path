# Sensitivity and Assumptions

## What could go wrong

Uplift modeling is only as good as its assumptions. If any fail, the estimated uplift is biased.

## Assumption 1: Unconfoundedness

$(Y(1), Y(0)) \perp T | X$. With randomization this holds. In observational data, it requires that all confounders are in $X$.

**Sensitivity test.** Rosenbaum's sensitivity analysis: how strong would an unobserved confounder need to be to overturn the conclusion? Tools: `dowhy` library's refutation methods.

## Assumption 2: Overlap

$0 < P(T | X) < 1$. Every covariate region has both treated and control examples.

**Test.** Plot propensity scores. If you see scores near 0 or 1 with many units, overlap fails there. Trim the dataset or narrow the target population.

## Assumption 3: SUTVA

One unit's treatment does not affect another's outcome.

**Where it breaks.** Social network effects (sending a coupon to you influences your friend). Marketplace effects (treating some buyers changes prices for others).

**Test.** Hard to test directly. Randomize at a higher level (clusters, regions) if network effects are suspected.

## Assumption 4: Stable treatment definition

Treatment is well-defined and consistent across units.

**Where it breaks.** "Email campaign" could be slightly different per variant. "Discount" varies by amount. Document the exact treatment definition.

## Model-level robustness checks

### 1. Refutation tests

Add a placebo (random) treatment. The estimated uplift should be close to zero.

### 2. Bootstrap confidence intervals

Sample with replacement, refit the model, recompute uplift. The bootstrap distribution gives a confidence interval for the Qini coefficient.

### 3. Cross-validation of uplift

Split data into folds by time or user group. Train on one fold, test on another. Uplift metrics should be stable.

### 4. Business sanity check

Rank users by predicted uplift. Inspect the top few and bottom few. Do they look like persuadables vs sleeping dogs? If the predicted-persuadable group is full of users who already bought, the model is wrong.

## Reporting

Every uplift result should include:

- The estimate (uplift@K, Qini coefficient).
- A confidence interval (bootstrap).
- A sanity check (placebo refutation).
- A caveat section listing the assumptions that could fail.

## Cross-links

- `01_potential_outcomes.md`
- `03_uplift_modeling.md`
- `docs/theory/08_experimentation_causal_primer.md`
