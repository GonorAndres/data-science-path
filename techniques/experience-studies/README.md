# Experience Studies

## What it is

An experience study is the actuarial process of comparing what actually happened (actual experience) with what was expected to happen (expected experience) for key insurance assumptions. Insurers set products based on assumptions about mortality rates, lapse rates, morbidity incidence, claim frequency, and more. Over time, they must validate these assumptions against real data to determine whether they are still accurate. The primary output of an experience study is the **Actual-to-Expected (A/E) ratio**: if your mortality table predicted 100 deaths and you observed 95, your A/E ratio is 0.95 (or 95%), meaning actual mortality was 5% lower than expected. Experience studies are how the insurance industry keeps its assumptions honest.

## Why it matters

- **Assumption validation**: Every insurance product is priced using assumptions (mortality, lapse, morbidity, expense). If those assumptions drift from reality, the product can become unprofitable or overpriced. Experience studies catch the drift.
- **Regulatory and professional requirements**: Actuaries are required by Actuarial Standards of Practice (ASOPs) to base assumptions on relevant experience and to update them periodically. Experience studies are the mechanism.
- **Financial reporting**: Under IFRS 17, GAAP (ASC 944), and statutory frameworks, actuaries must unlock or update assumptions when experience deviates materially from expectations. Experience studies provide the evidence.
- **Credibility applications**: The results of experience studies are often credibility-weighted (see the Bayesian Credibility README) to blend company-specific experience with industry benchmarks.
- **Career skill**: Experience studies are bread-and-butter actuarial work, especially for life and health actuaries. Understanding this process is essential for assumption-setting roles, valuation, and pricing.

## When to use it

- You are an actuary reviewing whether your company's mortality assumptions are still appropriate for a block of life insurance.
- You need to determine if policyholder lapse rates have changed, particularly in response to interest rate movements.
- You are setting morbidity assumptions for a disability or long-term care product.
- You are performing an annual or triennial assumption review for financial reporting.
- You want to investigate whether a specific demographic segment, product, or distribution channel has different experience than the overall portfolio.
- You are contributing to an industry-wide experience study (e.g., SOA intercompany studies).

## How it works (intuitive)

### The core concept: A/E analysis

Think of an experience study like grading your own predictions. At the start of the year, you predicted that 100 out of 10,000 policyholders would die. At the end of the year, you count the actual deaths. If 92 died, your A/E ratio is 92/100 = 0.92. Your predictions were 8% too conservative (actual mortality was lower than expected).

But this simple calculation hides important details:
- Not all 10,000 policyholders were in force for the entire year. Some joined mid-year, some lapsed, some died. You need to calculate **exposures** correctly.
- Different policyholders have different expected rates (based on age, gender, underwriting class, policy duration). You need to apply the correct expected rate to each policyholder's actual exposure.

### Step-by-step process

**Step 1: Define the study parameters.**
- **Decrement of interest**: What are you studying? Mortality, lapse, morbidity incidence, claim frequency?
- **Study period**: Calendar years covered (e.g., 2020-2024).
- **Population**: Which policies are in scope? All life insurance, or just term? Individual or group?
- **Basis of expected**: What table or assumption are you comparing against? (e.g., 2017 CSO mortality table, company's current pricing assumptions.)

**Step 2: Calculate exposures.**

Exposure measures how long each policyholder was "at risk" of experiencing the decrement during the study period. This is crucial because a policyholder who was in force for only 3 months should contribute less to the study than one who was in force for 12 months.

**Exact exposure method (policy-year basis):**
For each policy, calculate the fraction of the year during which the policy was active and exposed to the decrement. A policy in force from January 1 to September 30 contributes 0.75 policy-years of exposure.

**Census-based (actuarial estimate) exposure:**
When exact dates are not available, approximate exposures using beginning-of-period and end-of-period counts:

$$\text{Exposure} \approx \frac{\text{BOY count} + \text{EOY count}}{2}$$

Or, more precisely:

$$\text{Exposure} = \text{BOY count} + \frac{1}{2} \times \text{Entries}} - \frac{1}{2} \times \text{Withdrawals (non-decrement)}$$

The assumption is that entries and withdrawals occur uniformly throughout the year. Decrements (deaths, for a mortality study) are handled differently depending on the convention:

- **Exact exposure**: Decrement occurs at the exact date; exposure ends at that date.
- **Actuarial exposure**: Decrement is assumed to occur mid-year; the individual contributes a half-year of exposure.
- **Balducci hypothesis**: Decrements are uniformly distributed across the year of age.

**Step 3: Calculate expected decrements.**

For each policyholder (or cell), multiply the exposure by the expected rate:

$$E_i = \text{Exposure}_i \times q_i^{\text{expected}}$$

where $q_i^{\text{expected}}$ is the expected decrement rate for the policyholder's age, gender, duration, underwriting class, etc.

Sum across all policyholders to get total expected decrements: $E = \sum_i E_i$.

**Step 4: Count actual decrements.**

Count the actual number of decrements (deaths, lapses, etc.) during the study period: $A = \sum_i A_i$.

**Step 5: Compute the A/E ratio.**

$$\text{A/E} = \frac{A}{E} = \frac{\text{Actual decrements}}{\text{Expected decrements}}$$

- A/E > 1.00: Actual experience is worse than expected (more deaths, more lapses than the table predicts).
- A/E < 1.00: Actual experience is better than expected.
- A/E = 1.00: Actual experience matches expectations perfectly.

**Step 6: Analyze by segments.**

Break down the A/E ratio by relevant dimensions:
- Age group
- Gender
- Policy duration (select vs. ultimate period for mortality)
- Product type
- Distribution channel
- Underwriting class
- Calendar year (to detect trends)
- Face amount band (for mortality studies with anti-selection concerns)

This segmentation reveals where the assumptions are accurate and where they need adjustment.

**Step 7: Apply credibility.**

If your company's experience is not fully credible (not enough data to be statistically reliable), blend it with industry experience using credibility weighting:

$$q^{\text{adjusted}} = Z \times q^{\text{company A/E}} + (1 - Z) \times q^{\text{industry}}$$

See the Bayesian Credibility README for details on computing $Z$.

### A visual analogy

Imagine a weather forecaster who predicted rain on 100 days last year. An experience study would check how many of those 100 days it actually rained (say, 85). The A/E ratio for rain predictions would be 0.85. You would then drill down: were the predictions better in summer than winter? Better for the coast than inland? This segmentation reveals where the model works and where it needs recalibration.

## Key concepts

### Exposure calculation formulas

**Exact (seriatim) exposure for mortality:**

For policyholder $i$ with entry date $t_{\text{in}}$ and exit date $t_{\text{out}}$ during the study period $[T_0, T_1]$:

$$e_i = \frac{\min(t_{\text{out}}, T_1) - \max(t_{\text{in}}, T_0)}{365.25}$$

expressed in years. If the policyholder died, $t_{\text{out}}$ is the death date.

**Central death rate vs. initial rate:**

| Rate | Formula | Context |
|------|---------|---------|
| Central death rate ($m_x$) | $m_x = D_x / E_x^c$ where $E_x^c$ is central exposure | Exposure excludes time after death. Used in life table construction. |
| Initial rate ($q_x$) | $q_x = D_x / E_x^0$ where $E_x^0$ is initial (exact) exposure | Exposure includes the full period for each life at risk. Used for A/E comparisons. |
| Relationship | $q_x \approx 1 - e^{-m_x}$ (under UDD: $q_x = m_x / (1 + 0.5 \cdot m_x)$) | Conversion formulas between the two. |

### Statistical significance testing

An A/E ratio of 1.05 for a small block might be pure random noise. For a large block, it is almost certainly meaningful. Use the normal approximation (valid when expected counts are large):

$$\text{Test statistic} = \frac{A - E}{\sqrt{E}} \sim N(0, 1) \text{ approximately}$$

This assumes actual decrements follow a Poisson distribution with mean $E$, so $\text{Var}(A) \approx E$.

A 95% confidence interval for the A/E ratio:

$$\text{A/E} \pm 1.96 \times \frac{\sqrt{A}}{E}$$

The standard error uses $\sqrt{A}$ (not $\sqrt{E}$) in the numerator because $A$ is the observed Poisson count.

### Mortality study specifics

**Select and ultimate period:**
- **Select period**: The years immediately after underwriting, when mortality is low because unhealthy applicants were screened out.
- **Ultimate period**: After the select period wears off (typically 15-25 years for life insurance), mortality reflects the general insured population.
- Experience studies should analyze these periods separately because the A/E patterns differ.

**Duration analysis:**
Policy duration (time since issue) is a key predictor:
- Duration 1 (first policy year): Mortality is lowest due to recent underwriting.
- Durations 2-15: Mortality gradually increases as the "selection effect" wears off.
- Durations 15+: Mortality stabilizes at the ultimate level.

**Smoking status and underwriting class:**
Modern experience studies stratify by preferred/standard/substandard classes and smoker status. These strata can have very different A/E patterns.

### Lapse study specifics

**Lapse rate definition:**

$$\text{Lapse rate} = \frac{\text{Number of policies lapsed or surrendered}}{\text{Number of policies exposed}}$$

**Key dimensions for lapse studies:**
- **Policy duration**: Lapse rates are typically highest in the first 1-3 policy years and decrease with duration (the "lapse and re-entry" effect).
- **Interest rate sensitivity**: For products with cash value (whole life, universal life), lapse rates depend on the difference between the policy's credited rate and market rates.
- **Premium jump**: For term life insurance, lapse rates spike dramatically at the end of the level premium period when premiums increase.
- **Shock lapse**: A sudden, large increase in lapse rates, often at the end of a surrender charge period or premium guarantee period.

### Exposure by amount vs. by count

| Measure | A/E by count | A/E by amount |
|---------|-------------|--------------|
| Numerator | Number of actual deaths | Sum of face amounts for actual deaths |
| Denominator | Number of expected deaths | Sum of (face amount x expected rate) for all exposed |
| Use case | Tests the frequency of the decrement | Tests the severity (are higher face amount policies dying at different rates?) |
| Anti-selection detection | Limited | High face amount policies with above-average mortality suggest anti-selection |

Amount-weighted studies are important for detecting anti-selection, where individuals who know they are in poor health maintain larger policies.

## Python implementation

```python
import pandas as pd
import numpy as np
from scipy import stats

def compute_ae_ratio(data, actual_col, exposure_col, expected_rate_col):
    """
    Compute A/E ratio from seriatim data.

    Parameters:
    -----------
    data : DataFrame with one row per policyholder
    actual_col : binary column (1 = decrement occurred, 0 = no decrement)
    exposure_col : exposure in years
    expected_rate_col : expected annual decrement rate

    Returns:
    --------
    dict with A/E ratio, actual count, expected count, and confidence interval.
    """
    actual = data[actual_col].sum()
    expected = (data[exposure_col] * data[expected_rate_col]).sum()

    ae_ratio = actual / expected if expected > 0 else np.nan

    # 95% confidence interval (normal approximation)
    se = np.sqrt(actual) / expected if expected > 0 else np.nan
    ci_lower = ae_ratio - 1.96 * se
    ci_upper = ae_ratio + 1.96 * se

    return {
        'actual': actual,
        'expected': round(expected, 2),
        'ae_ratio': round(ae_ratio, 4),
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
        'n_policies': len(data),
        'total_exposure': round(data[exposure_col].sum(), 2),
    }


def ae_by_segment(data, segment_col, actual_col, exposure_col, expected_rate_col):
    """
    Compute A/E ratio for each segment.
    """
    results = []
    for segment, group in data.groupby(segment_col):
        result = compute_ae_ratio(group, actual_col, exposure_col, expected_rate_col)
        result['segment'] = segment
        results.append(result)

    return pd.DataFrame(results).set_index('segment')


# Example usage
# Suppose we have seriatim data with columns:
# policy_id, age, gender, duration, exposure_years, died (0/1), expected_qx
#
# ae_overall = compute_ae_ratio(data, 'died', 'exposure_years', 'expected_qx')
# ae_by_age = ae_by_segment(data, 'age_group', 'died', 'exposure_years', 'expected_qx')
```

**Useful Python libraries:**
- `pandas`: Data manipulation for seriatim records.
- `lifelines`: Survival analysis (Kaplan-Meier, Cox PH) for non-parametric exposure analysis.
- `scipy.stats`: Statistical tests (chi-squared goodness of fit for A/E significance).
- `pymort`: Python package for accessing SOA mortality tables.

```python
# Using lifelines for exposure-based survival analysis
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()
kmf.fit(durations=data['exposure_years'],
        event_observed=data['died'],
        label='Company Experience')
kmf.plot_survival_function()
```

## R implementation (if applicable)

R is widely used for experience studies, especially in life insurance:

```r
# Basic A/E calculation in R
library(dplyr)

ae_results <- data %>%
  group_by(age_group, gender) %>%
  summarise(
    actual = sum(died),
    expected = sum(exposure_years * expected_qx),
    ae_ratio = actual / expected,
    n_policies = n(),
    .groups = 'drop'
  ) %>%
  mutate(
    se = sqrt(actual) / expected,
    ci_lower = ae_ratio - 1.96 * se,
    ci_upper = ae_ratio + 1.96 * se
  )

# Using the actuar package for exposure calculation
library(actuar)

# Using the survival package for Kaplan-Meier
library(survival)
km_fit <- survfit(Surv(exposure_years, died) ~ gender, data = data)
plot(km_fit)
```

The SOA also publishes experience study datasets and tools, and many companies use proprietary systems (e.g., AXIS, MoSes, Prophet) that have built-in experience study modules.

## Gotchas & lessons learned

1. **Exposure calculation errors are the most common mistake**: Getting the exposure right is the hardest and most important part. Double-check:
   - Are you handling mid-year entries and exits correctly?
   - Are you using the right convention for decrements (exact, actuarial, Balducci)?
   - Are you correctly truncating exposures at the study period boundaries?
   - Are you excluding policies that should not be in scope?

2. **Multiple decrements complicate things**: In a lapse study, what do you do with policyholders who died? They are not lapses, but they left the exposed population. Use multiple-decrement methodology: treat non-study decrements as censored observations that reduce exposure.

3. **A/E ratios near 1.00 do not mean the assumption is perfect**: The A/E ratio is an aggregate measure. The assumption could be too high for young ages and too low for old ages, averaging to 1.00 overall. Always examine A/E by segment.

4. **Trends matter**: An A/E ratio of 0.95 for the full study period is less informative than seeing that it was 1.05 in year 1 and 0.85 in year 5 (mortality is improving over time). Always plot A/E by calendar year to detect trends.

5. **Small cell sizes produce unreliable A/E ratios**: An A/E ratio of 2.00 based on 4 actual deaths vs. 2 expected is meaningless noise. Apply credibility weighting or group small cells together. Always report confidence intervals.

6. **Anti-selection and data quality**: High A/E ratios in certain segments may reflect genuine risk differences or data quality issues (miscoded underwriting class, incorrect dates). Investigate before concluding that assumptions need updating.

7. **Expected basis matters enormously**: The same actual experience will produce different A/E ratios depending on which expected table you use. Always clearly document the expected basis and version.

8. **Lapse studies are messier than mortality studies**: Death is unambiguous. "Lapse" can include voluntary surrenders, non-forfeiture options, reduced paid-up, 1035 exchanges, and more. Define the decrement clearly and consistently.

9. **Amount-weighted vs. count-weighted**: If your mortality A/E is 1.00 by count but 1.15 by amount, larger policies are dying at a higher rate. This is a red flag for anti-selection and should trigger further investigation.

10. **Updating assumptions is judgment, not just math**: An A/E ratio of 0.90 does not automatically mean you should reduce your assumed rates by 10%. Consider: Is the experience credible? Is there a trend? Are there known one-time events? Is the expected basis itself changing? Assumption updates require actuarial judgment informed by the data.

## Resources

- [SOA Experience Studies](https://www.soa.org/experience-studies/) -- Industry-wide mortality, lapse, and morbidity experience studies. Essential reference data.
- [SOA 2017 CSO Mortality Tables](https://www.soa.org/resources/experience-studies/2015/2017-cso-tables/) -- The current standard individual life mortality tables.
- [Actuarial Mathematics by Bowers, Gerber, Hickman, Jones, Nesbitt](https://www.amazon.com/Actuarial-Mathematics-Newton-Bowers/dp/0938959468) -- Textbook treatment of exposure theory and multiple decrements.
- [ASOP No. 25 -- Credibility Procedures](https://www.actuarialstandardsboard.org/) -- Actuarial Standard of Practice on applying credibility.
- [ASOP No. 23 -- Data Quality](https://www.actuarialstandardsboard.org/) -- Standards for data used in actuarial analysis.
- [pymort Python package](https://github.com/actuarialopensource/pymort) -- Access SOA mortality tables in Python.
- [lifelines documentation](https://lifelines.readthedocs.io/) -- Survival analysis and exposure calculation in Python.
- [Dick London -- Survival Models and Their Estimation](https://actexmadriver.com/) -- Classic actuarial text on survival model estimation from experience data.
