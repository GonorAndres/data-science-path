# Ratemaking Basics

## What it is

Ratemaking is the actuarial process of determining the price (premium rate) an insurer should charge for insurance coverage. It is the bridge between historical loss experience and future pricing. The fundamental equation is deceptively simple: the premium must be enough to cover expected losses, expenses, and profit. But the devil is in the details -- historical losses must be adjusted for trends, development, and on-leveling before they can inform future rates. Ratemaking is governed by the principle that "rates shall not be excessive, inadequate, or unfairly discriminatory," a standard enshrined in insurance regulation across most jurisdictions.

## Why it matters

- **Core actuarial function**: Ratemaking (also called pricing) is the most visible and commercially important thing actuaries do. Getting it right determines whether a company is competitive and solvent.
- **Regulatory compliance**: Rates must be filed with and approved by regulators in most states and countries. The actuary must be able to defend every assumption and adjustment.
- **Business viability**: Rates that are too low lead to underwriting losses and potential insolvency. Rates that are too high drive away customers, leading to adverse selection (only the worst risks stay).
- **Connects to reserving and experience studies**: Ratemaking uses the same data as reserving (loss triangles, development factors) and the same validation tools as experience studies (A/E ratios). It is the forward-looking complement to these backward-looking analyses.
- **Exam topic**: Ratemaking is a major component of CAS exams (Exam 5 in particular) and is covered in SOA exams for general insurance.

## When to use it

- You are a pricing actuary preparing a rate filing for personal auto, homeowners, workers' compensation, or any other P&C line.
- You are reviewing the adequacy of existing rates for a renewal.
- You need to calculate a rate indication (how much rates should change from current levels).
- You are building a predictive model for insurance pricing and need to understand the traditional actuarial pipeline that the model must be consistent with.
- You want to understand why two policyholders with different characteristics are charged different premiums.

## How it works (intuitive)

### The ratemaking pipeline

Think of ratemaking as a pipeline that transforms raw historical data into a future rate:

```
Raw historical data
  --> Adjust for development (IBNR)
  --> Adjust for trend (inflation, frequency changes)
  --> On-level to current rate level
  --> Calculate loss ratio or pure premium
  --> Layer on expenses and profit
  --> Indicated rate change
```

Each step removes a distortion from the historical data so that the final result reflects what losses will look like *next year under current conditions*.

### An analogy: pricing a lemonade stand

Imagine you are pricing lemonade for next summer. Last year, you spent $80 on lemons and $20 on cups (total cost = $100) and sold 200 cups (average cost = $0.50/cup).

But you need to adjust:
1. **Development**: Some bills from last summer have not arrived yet (maybe a late lemon delivery invoice). You estimate the true cost was $85 on lemons.
2. **Trend**: Lemons are getting 5% more expensive each year. Next year's lemon cost will be about $89.25.
3. **On-leveling**: You raised cup prices partway through last summer. To compare fairly, you need to figure out what total revenue would have been if you had charged the current price all year.
4. **Expenses and profit**: You want to cover your $20 in fixed costs (cups, stand rental) and earn a 10% profit margin.

Working through these adjustments gives you the price you should charge per cup next summer. Insurance ratemaking does exactly this, but for losses instead of lemons.

## Key concepts

### The fundamental insurance equation

$$\text{Premium} = \text{Losses} + \text{LAE} + \text{UW Expenses} + \text{UW Profit}$$

Or, in ratio form:

$$1.00 = \text{Loss Ratio} + \text{LAE Ratio} + \text{Expense Ratio} + \text{Profit Margin}$$

where each ratio is expressed as a fraction of premium.

The **permissible loss ratio** (also called the "target loss ratio") is:

$$\text{PLR} = 1.00 - \text{Expense Ratio} - \text{Profit Margin}$$

### Two main ratemaking methods

#### Pure Premium Method

Calculates the rate directly from the per-exposure cost of losses.

$$\text{Indicated Rate} = \frac{\text{Projected Ultimate Losses + LAE per Exposure}}{1.0 - V - Q_T}$$

where:
- Projected Ultimate Losses + LAE per Exposure = the **pure premium** (expected loss cost per unit of exposure)
- $V$ = variable expense ratio (commissions, premium taxes, expressed as % of premium)
- $Q_T$ = target underwriting profit margin

**Steps to compute the pure premium:**

1. Start with historical incurred losses.
2. **Develop to ultimate** using development factors (from loss reserving).
3. **Trend to the future** using trend factors.
4. Divide by earned exposures to get the trended ultimate pure premium.

#### Loss Ratio Method

Compares expected losses to current premium to determine the needed rate change.

$$\text{Indicated Rate Change} = \frac{\text{Experience Loss Ratio (adjusted)}}{\text{Permissible Loss Ratio}} - 1.0$$

where:
- Experience Loss Ratio = (Trended, Developed Losses) / (On-Leveled Earned Premium)
- Permissible Loss Ratio = $1.0 - V - Q_T$

If the indicated rate change is +10%, current rates need to increase by 10%.

**When to use which:**

| Feature | Pure Premium | Loss Ratio |
|---------|-------------|------------|
| Needs current rate level? | No | Yes (requires on-leveling) |
| Output | Dollar rate per exposure | Percentage rate change |
| Better for | New products, pricing from scratch | Rate revisions, renewals |
| Sensitivity to exposure changes | Less sensitive | More sensitive (through premium) |

### Loss development

Historical losses are not yet fully developed (see Loss Reserving README). Ratemaking uses the same development factors:

$$\text{Developed Losses}_i = \text{Reported Losses}_i \times \text{CDF to Ultimate}_i$$

Losses are typically developed on both a paid and incurred basis; the actuary selects the more appropriate or uses a blend.

### Trend factors

Historical losses occurred under past conditions. Future losses will reflect different conditions due to:
- **Claim severity trend**: Medical inflation, auto repair cost increases, legal environment changes.
- **Claim frequency trend**: Changes in driving patterns, workplace safety improvements, fraud levels.

The **trend factor** adjusts historical losses to the cost level expected during the future policy period.

$$\text{Trend Factor} = (1 + t)^n$$

where:
- $t$ = annual trend rate (e.g., 4% severity trend)
- $n$ = number of years from the historical period to the midpoint of the future policy period.

**Example**: If the historical accident year is 2022, the projected effective date is January 1, 2025, and the policy period is one year, then the midpoint of the future policy period is July 1, 2025.

$$n = 2025.5 - 2022.5 = 3.0 \text{ years}$$
$$\text{Trend Factor} = (1.04)^{3.0} = 1.1249$$

Historical losses from 2022 are multiplied by 1.1249 to project them to 2025 cost levels.

**Frequency and severity trends are applied separately** because they may differ in magnitude and direction:

$$\text{Total Trend Factor} = (1 + t_{\text{freq}})^n \times (1 + t_{\text{sev}})^n$$

### On-leveling

If the insurer changed rates during the historical period, the earned premium in each year reflects a mix of old and new rates. **On-leveling** restates historical premium as if today's rate level had been in effect the entire time.

**Parallelogram method (extension of exposures):**

If rates changed from $R_1$ to $R_2$ partway through a policy year, the earned premium at current rate level is:

$$\text{On-Level Premium}_i = \text{Earned Premium}_i \times \text{On-Level Factor}_i$$

$$\text{On-Level Factor}_i = \frac{\text{Current Rate Level}}{\text{Average Rate Level during Year } i}$$

The average rate level during a year depends on when rate changes took effect and how policies are distributed over the year. The parallelogram method uses geometric areas to compute the average.

### Rate indication formula (putting it all together)

**Loss Ratio Method:**

$$\text{Indicated Change} = \frac{W}{PLR} - 1.0$$

where:

$$W = \frac{\sum_i \text{Developed Trended Losses}_i}{\sum_i \text{On-Leveled Premium}_i}$$

**Pure Premium Method:**

$$\text{Indicated Rate} = \frac{\sum_i \text{Developed Trended Losses}_i / \sum_i \text{Exposures}_i}{1.0 - V - Q_T}$$

### Expense loading

Expenses fall into two categories:

| Type | Examples | Treatment |
|------|----------|-----------|
| **Variable expenses** ($V$) | Commissions, premium taxes, contingent commissions | Expressed as % of premium; loaded multiplicatively |
| **Fixed expenses** ($F$) | Underwriting salaries, rent, IT systems | Expressed as $ per policy or per exposure; loaded additively |

The fully loaded rate is:

$$\text{Rate} = \frac{\text{Pure Premium} + F}{1.0 - V - Q_T}$$

### Credibility in ratemaking

When the insurer's own experience is not fully credible (see Bayesian Credibility README), the indicated rate change is blended with a complement:

$$\text{Credibility-Weighted Indication} = Z \times \text{Company Indication} + (1 - Z) \times \text{Complement}$$

Common complements:
- Industry loss costs (e.g., ISO advisory rates).
- Trended prior indication.
- A loss ratio of 1.0 (no change).

### Loss Costs vs. Rates

- **Loss cost**: The expected cost of losses per unit of exposure, before any expense or profit loading. Advisory organizations (ISO, NCCI) publish loss costs.
- **Rate**: The final price charged to the policyholder, including loss cost + expenses + profit.
- **Rate = Loss Cost / Permissible Loss Ratio** (approximately).

Individual companies apply their own expense loads and profit margins to the advisory loss cost.

## Python implementation

```python
import pandas as pd
import numpy as np

def pure_premium_indication(losses, lae, exposures, dev_factors, trend_factors,
                            variable_expense_ratio, fixed_expense_per_exposure,
                            profit_target):
    """
    Calculate indicated rate using the Pure Premium method.

    Parameters:
    -----------
    losses : array of historical incurred losses by accident year
    lae : array of loss adjustment expenses by accident year
    exposures : array of earned exposures by accident year
    dev_factors : array of CDF-to-ultimate for each accident year
    trend_factors : array of trend factors for each accident year
    variable_expense_ratio : V (as decimal, e.g., 0.25 for 25%)
    fixed_expense_per_exposure : F (dollar amount per exposure)
    profit_target : Q_T (as decimal, e.g., 0.05 for 5%)

    Returns:
    --------
    dict with indicated rate, pure premium, and supporting details.
    """
    # Develop losses to ultimate
    developed_losses = np.array(losses) * np.array(dev_factors)
    developed_lae = np.array(lae) * np.array(dev_factors)

    # Trend losses to future cost level
    trended_losses = developed_losses * np.array(trend_factors)
    trended_lae = developed_lae * np.array(trend_factors)

    # Pure premium
    total_trended = trended_losses.sum() + trended_lae.sum()
    total_exposures = np.array(exposures).sum()
    pure_premium = total_trended / total_exposures

    # Indicated rate
    indicated_rate = (pure_premium + fixed_expense_per_exposure) / (
        1.0 - variable_expense_ratio - profit_target
    )

    return {
        'pure_premium': round(pure_premium, 2),
        'fixed_expense_per_exposure': fixed_expense_per_exposure,
        'variable_expense_ratio': variable_expense_ratio,
        'profit_target': profit_target,
        'indicated_rate': round(indicated_rate, 2),
        'permissible_loss_ratio': round(1.0 - variable_expense_ratio - profit_target, 4),
    }


def loss_ratio_indication(losses, lae, on_level_premium, dev_factors, trend_factors,
                          variable_expense_ratio, profit_target):
    """
    Calculate indicated rate change using the Loss Ratio method.
    """
    developed_losses = np.array(losses) * np.array(dev_factors)
    developed_lae = np.array(lae) * np.array(dev_factors)
    trended = (developed_losses + developed_lae) * np.array(trend_factors)

    experience_lr = trended.sum() / np.array(on_level_premium).sum()
    permissible_lr = 1.0 - variable_expense_ratio - profit_target

    indicated_change = experience_lr / permissible_lr - 1.0

    return {
        'experience_loss_ratio': round(experience_lr, 4),
        'permissible_loss_ratio': round(permissible_lr, 4),
        'indicated_rate_change': round(indicated_change, 4),
        'indicated_rate_change_pct': f"{indicated_change * 100:.1f}%",
    }


# Example: trend factor calculation
def trend_factor(annual_rate, years):
    """Compute compound trend factor."""
    return (1 + annual_rate) ** years
```

**Useful Python libraries:**
- `chainladder`: Loss development factors (integrates with ratemaking workflow).
- `statsmodels`: Trend analysis (linear regression on log-transformed losses for exponential trend).
- `scikit-learn` / `xgboost` / `lightgbm`: Predictive models for class-level ratemaking (GLMs, GBMs).
- `pandas`: Data manipulation throughout the pipeline.

## R implementation (if applicable)

```r
# Basic ratemaking calculations in R
library(dplyr)

# On-leveling with the parallelogram method
on_level_factor <- function(current_rate_level, avg_rate_level) {
  current_rate_level / avg_rate_level
}

# Trend factor
trend_factor <- function(annual_trend, n_years) {
  (1 + annual_trend)^n_years
}

# Pure premium indication
pure_premium_indication <- function(developed_trended_losses, exposures,
                                     variable_expense_ratio, fixed_expense,
                                     profit_target) {
  pure_premium <- sum(developed_trended_losses) / sum(exposures)
  indicated_rate <- (pure_premium + fixed_expense) /
    (1.0 - variable_expense_ratio - profit_target)
  return(indicated_rate)
}

# For GLM-based class plans:
library(statmod)  # tweedie
library(tweedie)
# See the Tweedie Compound Models README for GLM-based ratemaking
```

## Gotchas & lessons learned

1. **On-leveling is easy to mess up**: If you forget to on-level premium, you are comparing trended losses to premium at a mix of historical rate levels. This produces a biased loss ratio. Always verify that premium is on-leveled to the current rate level.

2. **Trend selection is judgment, not just curve-fitting**: Historical trends can be volatile. A 3-year trend may differ dramatically from a 10-year trend. Consider whether recent patterns reflect temporary shocks (e.g., COVID reducing auto frequency in 2020) or structural changes.

3. **Frequency and severity trends should be analyzed separately**: Aggregate loss trends can mask offsetting movements. Frequency might be declining while severity is increasing rapidly. Understanding the components is essential.

4. **Development factor selection interacts with trend**: If you use paid losses (which develop slower), you need a longer development period, which means a larger trend factor. If you use incurred losses, development is shorter but can be distorted by case reserve changes. The two approaches should give consistent results.

5. **Expense ratio is not constant**: Variable expense ratios change with premium volume. Fixed expenses per policy change with inflation and operational efficiency. Use the most current and forward-looking expense assumptions.

6. **Do not double-count adjustments**: If losses are already trended in the development triangle, do not apply trend again in the ratemaking formula. Keep a clear audit trail of which adjustments have been applied.

7. **Rate adequacy vs. rate competitiveness**: An actuarially indicated rate might be correct but uncompetitive. Business decisions about which risks to write at what price are management decisions, but the actuary must document the indicated rate regardless.

8. **Catastrophe loading**: In property lines, catastrophe losses (hurricanes, earthquakes) are rare but large. Do not include them in the normal trend/development process. Use separate catastrophe models (AIR, RMS, CoreLogic) to estimate expected annual catastrophe losses and add them to the rate.

9. **Large loss treatment**: Individual large losses can distort the experience period. Common treatments include capping individual losses at a threshold (e.g., $500K), then adding back a load for expected excess losses based on severity distributions. This stabilizes the experience data.

10. **Credibility matters for small segments**: If you are ratemaking for a territory or class with thin experience, the indicated rate change for that segment is unreliable. Apply credibility weighting to blend with a complement (state-wide indication, industry loss cost).

## Resources

- [Werner & Modlin -- Basic Ratemaking (CAS)](https://www.casact.org/sites/default/files/2021-02/werner_modlin_ratemaking.pdf) -- Free CAS monograph. The single best starting resource for P&C ratemaking. Chapters cover every topic above in detail.
- [CAS Exam 5 Study Materials](https://www.casact.org/) -- Detailed treatment of all ratemaking methods.
- [Casualty Actuarial Society -- Statement of Principles Regarding Property and Casualty Insurance Ratemaking](https://www.casact.org/) -- Professional standards for ratemaking.
- [ISO (Verisk) -- Advisory Loss Costs](https://www.verisk.com/) -- Industry benchmark loss costs.
- [Friedland -- Estimating Unpaid Claims](https://www.casact.org/) -- For the loss development component of ratemaking.
- [chainladder-python documentation](https://chainladder-python.readthedocs.io/) -- Python implementation of development factor methods.
