# Customer Lifetime Value (CLV)

## What it is

Customer Lifetime Value (CLV, sometimes written as LTV or CLTV) is an estimate of the total net profit a business can expect from a customer over the entire duration of their relationship. It answers a deceptively simple question: "How much is this customer worth?" The answer is not just their last purchase or their annual spend -- it is the present value of all future cash flows they will generate, minus the costs of serving them. CLV is the single most important metric in customer-centric businesses because it connects marketing, product, and finance into one number.

## Why it matters

- **Marketing ROI**: If you know a customer is worth $2,000 over their lifetime, you can justify spending $500 to acquire them. Without CLV, you are flying blind on acquisition budgets.
- **Customer prioritization**: Not all customers are equal. CLV tells you who to invest in (high-CLV customers) and who to manage for efficiency (low-CLV customers).
- **Insurance-specific value**: In insurance, CLV maps directly to the actuarial concept of the **present value of future premiums minus expected claims and expenses**. An actuary computing the embedded value of a book of business is essentially computing aggregate CLV. The BG/NBD model for non-contractual CLV is structurally similar to decrement models actuaries use daily.
- **Strategic planning**: CLV informs pricing (how much discount can you offer to retain a high-value customer?), product development (which features increase CLV?), and channel strategy (which acquisition channels bring high-CLV customers?).
- **Valuation**: Startup valuations and business acquisitions often depend on CLV estimates. If you can credibly estimate the CLV of your customer base, you can value the business.

## When to use it

- You are setting acquisition budgets and need to know the maximum allowable cost-per-acquisition (CPA).
- You are building a retention strategy and want to quantify the dollar impact of reducing churn by X%.
- You want to compare the long-term value of customers acquired through different channels.
- You are designing loyalty or discount programs and need to estimate the ROI.
- You are doing actuarial reserving or embedded value calculations and want to connect with data science approaches.
- You need to segment customers by predicted future value (not just past behavior like RFM).
- You are forecasting future revenue and want a bottom-up model based on customer-level projections.

## How it works (intuitive)

### The apartment tenant analogy

Imagine you own an apartment building. Each tenant pays monthly rent. Some tenants stay for years; others leave after a few months. The "lifetime value" of a tenant is the total rent they will pay over their entire tenancy, minus your costs (maintenance, vacancy periods, etc.), discounted back to today's dollars.

CLV for customers works the same way:
1. Estimate how long the customer will stay (retention/survival).
2. Estimate how much they will spend per period (monetary value).
3. Subtract the cost of serving them (variable costs, support, etc.).
4. Discount future cash flows to present value (because a dollar next year is worth less than a dollar today).

### The three levels of CLV estimation

**Level 1: Historical CLV (simple sum)**

Just add up what the customer has spent so far. This is not really "lifetime" value -- it is backward-looking. But it is a useful starting point.

$$CLV_{historical} = \sum_{t=1}^{T} (Revenue_{t} - Cost_{t})$$

**Level 2: Simple predictive CLV (formula-based)**

Use average retention rate and average revenue per period to project forward:

$$CLV = \frac{\bar{m}}{1 + d - r}$$

where $\bar{m}$ is the average margin per period, $d$ is the discount rate per period, and $r$ is the retention rate per period. This assumes constant retention and constant margin -- strong assumptions, but useful for back-of-envelope calculations.

**Level 3: Probabilistic CLV (model-based)**

Use statistical models to estimate each customer's individual probability of being "alive," their expected future transaction rate, and their expected transaction value. This is where the BG/NBD and Gamma-Gamma models come in.

### Contractual vs. non-contractual settings

This distinction is critical and often glossed over:

| Setting | Definition | Examples | Key feature |
|---------|-----------|----------|-------------|
| **Contractual** | Customer actively decides to leave (cancels) | Subscriptions, insurance policies, gym memberships | You observe churn directly |
| **Non-contractual** | Customer just stops showing up -- no cancellation | Retail, e-commerce, restaurants | You never observe churn directly; you must infer it |

In **contractual** settings, you know exactly when a customer leaves, so the problem reduces to predicting renewal/churn probability and future spend.

In **non-contractual** settings, you never know for sure if a customer has churned or is just between purchases. This is why probabilistic models like BG/NBD are necessary -- they model the latent "alive/dead" state.

Insurance is interesting because it is typically **contractual** (policies have defined terms and renewal dates), but some product lines (like on-demand insurance) are moving toward non-contractual.

### The BG/NBD + Gamma-Gamma framework

This is the most widely used probabilistic CLV framework for non-contractual settings. It has two components:

**BG/NBD model (Buy-till-you-Die)**: Models the transaction process.
- Each customer has an individual purchase rate $\lambda$ (drawn from a Gamma distribution).
- Each customer has an individual dropout probability $p$ after each transaction (drawn from a Beta distribution).
- While "alive," the customer transacts according to a Poisson process with rate $\lambda$.
- At some point, the customer "dies" (becomes permanently inactive).
- The model uses the customer's observed Recency and Frequency to estimate their probability of being alive and their expected future transactions.

**Gamma-Gamma model**: Models the monetary value.
- Each customer has an individual average transaction value $\nu$ (drawn from a Gamma distribution).
- The model estimates each customer's expected average transaction value based on their observed spending.
- It assumes that monetary value is independent of transaction frequency (this assumption should be tested).

**Combined CLV**:

$$E[CLV_i] = E[\text{Future Transactions}_i] \times E[\text{Average Transaction Value}_i] \times \text{Margin\%}$$

discounted to present value.

### The actuarial present value connection

For actuaries, CLV is structurally identical to the present value of a life annuity with decrements:

$$APV = \sum_{t=1}^{\omega} v^t \cdot {}_tp_x \cdot b_t$$

where:
- $v = \frac{1}{1+d}$ is the discount factor
- ${}_tp_x$ is the probability of surviving (staying as a customer) to period $t$
- $b_t$ is the benefit (margin) at period $t$
- $\omega$ is the maximum possible lifetime

In actuarial notation, the simple CLV formula $CLV = \frac{\bar{m}}{1+d-r}$ is equivalent to a perpetuity-due with a "decrement" rate of $1-r$.

## Key concepts

### Simple CLV formulas

**For contractual/subscription businesses with constant retention**:

$$CLV = \sum_{t=0}^{T} \frac{m \cdot r^t}{(1+d)^t} = m \cdot \sum_{t=0}^{T} \left(\frac{r}{1+d}\right)^t$$

As $T \to \infty$:

$$CLV = \frac{m}{1 + d - r}$$

where:
- $m$ = margin per period (revenue minus variable costs)
- $r$ = retention rate per period (e.g., 0.85 means 85% of customers renew each period)
- $d$ = discount rate per period (e.g., 0.01 for 1% monthly or 0.10 for 10% annual)

**Example**: If a customer pays $100/month with 80% margin, retains at 90% annually, and the annual discount rate is 10%:

$$CLV = \frac{\$80}{1 + 0.10 - 0.90} = \frac{\$80}{0.20} = \$400$$

### Discounted CLV with time-varying retention

When retention varies by period (it usually does -- first-year churn is higher):

$$CLV = \sum_{t=0}^{T} \frac{m_t \cdot S(t)}{(1+d)^t}$$

where $S(t) = \prod_{s=1}^{t} r_s$ is the survival function (cumulative retention to period $t$).

### BG/NBD model parameters

The BG/NBD model has four parameters:

| Parameter | Distribution | Meaning |
|-----------|-------------|---------|
| $r, \alpha$ | $\lambda \sim \text{Gamma}(r, \alpha)$ | Purchase rate heterogeneity across customers |
| $a, b$ | $p \sim \text{Beta}(a, b)$ | Dropout probability heterogeneity across customers |

The model takes as input each customer's:
- **Frequency** ($x$): number of repeat purchases (not counting the first)
- **Recency** ($t_x$): time of last purchase (in periods since first purchase)
- **T**: total observation time (time since first purchase to end of observation)

Key outputs:
- $P(\text{alive} | x, t_x, T)$: probability the customer is still active
- $E[\text{future transactions} | x, t_x, T]$: expected number of transactions in a future period

### Gamma-Gamma model parameters

| Parameter | Distribution | Meaning |
|-----------|-------------|---------|
| $p, q, v$ | $\nu \sim \text{Gamma}(p, v)$ and $z_i | \nu \sim \text{Gamma}(q, \nu)$ | Individual-level spending heterogeneity |

Input: each customer's average transaction value and frequency.
Output: $E[M | x, \bar{m}_x]$: expected average transaction value given observed behavior.

**Critical assumption**: Monetary value must be independent of frequency. Test this before fitting the model. If the correlation between frequency and average monetary value is strong (|r| > 0.3), the Gamma-Gamma model is not appropriate and you may need alternative approaches.

### Key CLV metrics

| Metric | Formula | Interpretation |
|--------|---------|---------------|
| CLV-to-CAC ratio | $\frac{CLV}{CAC}$ | Should be > 3:1 for a healthy business |
| Payback period | Time until cumulative margin exceeds CAC | How quickly you recover the acquisition cost |
| Margin per period | Revenue - Variable costs | The "benefit" in the CLV calculation |
| Retention elasticity | $\frac{\partial CLV}{\partial r} \cdot \frac{r}{CLV}$ | How sensitive CLV is to retention improvements |

### Sensitivity of CLV to retention

A classic result: small improvements in retention have outsized effects on CLV. At $r = 0.90$ and $d = 0.10$:

$$CLV = \frac{m}{0.20} = 5m$$

At $r = 0.95$:

$$CLV = \frac{m}{0.15} = 6.67m$$

A 5-percentage-point improvement in retention increased CLV by 33%. This is why retention is the most powerful lever for growing customer value.

## Python implementation

```python
# Using the lifetimes library (pip install lifetimes)
# Note: lifetimes is sometimes maintained under the btyd name
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import pandas as pd

# --- Step 1: Prepare RFM summary from transaction data ---
rfm = summary_data_from_transaction_data(
    transactions,
    customer_id_col='customer_id',
    datetime_col='transaction_date',
    monetary_value_col='amount',
    observation_period_end='2024-12-31'
)
# rfm columns: frequency, recency, T, monetary_value

# --- Step 2: Fit BG/NBD model ---
bgf = BetaGeoFitter(penalizer_coef=0.01)
bgf.fit(rfm['frequency'], rfm['recency'], rfm['T'])

# Predict expected transactions in the next 12 months
rfm['predicted_purchases_12m'] = bgf.conditional_expected_number_of_purchases_up_to_time(
    12, rfm['frequency'], rfm['recency'], rfm['T']
)

# Predict probability of being alive
rfm['prob_alive'] = bgf.conditional_probability_alive(
    rfm['frequency'], rfm['recency'], rfm['T']
)

# --- Step 3: Fit Gamma-Gamma model ---
# Filter to customers with frequency > 0 (repeat purchasers)
returning = rfm[rfm['frequency'] > 0]

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(returning['frequency'], returning['monetary_value'])

# Predict expected average transaction value
rfm_returning = returning.copy()
rfm_returning['expected_avg_value'] = ggf.conditional_expected_average_profit(
    returning['frequency'], returning['monetary_value']
)

# --- Step 4: Compute CLV ---
clv = ggf.customer_lifetime_value(
    bgf,
    returning['frequency'],
    returning['recency'],
    returning['T'],
    returning['monetary_value'],
    time=12,           # 12 months projection
    discount_rate=0.01  # monthly discount rate
)

# --- Simple contractual CLV formula ---
def simple_clv(margin_per_period, retention_rate, discount_rate, periods=None):
    """
    Simple CLV for contractual settings.

    If periods is None, uses the infinite-horizon formula.
    """
    if periods is None:
        return margin_per_period / (1 + discount_rate - retention_rate)
    else:
        clv = 0
        for t in range(periods):
            clv += margin_per_period * (retention_rate ** t) / ((1 + discount_rate) ** t)
        return clv
```

Key libraries:
- **lifetimes** (or **btyd**): BG/NBD, Gamma-Gamma, and CLV computation. [lifetimes docs](https://lifetimes.readthedocs.io/)
- **pymc** or **stan**: For Bayesian CLV models with more flexibility than lifetimes.
- **pandas** for data preparation (summary_data_from_transaction_data).

## R implementation (if applicable)

```r
# install.packages("BTYDplus")  # or BTYD
library(BTYDplus)

# Prepare data: customer-level summary with x (frequency), t.x (recency), T.cal
# Fit BG/NBD model
params <- bgnbd.EstimateParameters(cal.cbs)

# Expected future transactions
bgnbd.ConditionalExpectedTransactions(params, T.star = 12, x, t.x, T.cal)

# Probability alive
bgnbd.PAlive(params, x, t.x, T.cal)
```

**Python vs. R tradeoff**: The `lifetimes` Python library has a cleaner API and better documentation. The R `BTYD` / `BTYDplus` packages are more academically grounded and offer additional model variants (Pareto/NBD, Modified BG/NBD). For production, Python integrates better with ML pipelines. For research, R has more model variants.

## Gotchas & lessons learned

### 1. The biggest gotcha: contractual vs. non-contractual confusion
Using the simple formula $CLV = m / (1+d-r)$ for a non-contractual business (where you cannot observe churn directly) is a common mistake. You need a probabilistic model for non-contractual settings.

### 2. Discount rate choice matters enormously
The discount rate $d$ can swing CLV by 2-3x. A 5% annual rate gives a very different answer than 15%. For corporate planning, use the company's weighted average cost of capital (WACC). For marketing decisions, use a rate that reflects the opportunity cost of the marketing budget.

### 3. Margin, not revenue
CLV should be computed on margin (revenue minus variable costs), not revenue. If a customer spends $100 but costs you $70 to serve, their margin is $30. Computing CLV on $100 overstates their value by more than 3x.

### 4. The Gamma-Gamma independence assumption
The Gamma-Gamma model assumes that monetary value is independent of purchase frequency. This is often violated -- customers who buy more frequently may buy cheaper items. Check the correlation. If it is strong, consider segmenting customers and fitting separate models, or using a model that relaxes this assumption.

### 5. Calibration/holdout validation
Always split your data into a calibration period (used for fitting) and a holdout period (used for validation). Compare predicted transactions in the holdout period to actual transactions. The `lifetimes` library has built-in functions for this.

### 6. Cold-start problem
For new customers with zero or one transaction, the BG/NBD model has very little information and will default to population averages. Supplement with demographic or behavioral features if available. Some practitioners use a two-stage approach: segment-level CLV for new customers, individual-level CLV for customers with enough history.

### 7. Recency vs. calendar time
In the BG/NBD model, "recency" is the time of last purchase measured from the *first* purchase, not from today. This is different from the RFM recency definition (days since last purchase from today). The `lifetimes` library handles this conversion, but be careful if you are preparing data manually.

### 8. Negative CLV customers exist
Some customers cost more to serve than they generate in revenue (e.g., frequent claimants in insurance, serial returners in e-commerce). CLV can be negative. Do not filter these out -- they represent real economic drag and deserve their own strategy (e.g., repricing, service-level adjustment).

### 9. Time horizon matters
An infinite-horizon CLV is a theoretical construct. In practice, cap the projection at a reasonable horizon (3-5 years for most businesses, matching the strategic planning horizon). This also reduces the sensitivity to the discount rate.

### 10. CLV is not static
Customer behavior changes. Recompute CLV periodically (quarterly or monthly) and track how individual CLV estimates evolve. A customer whose predicted CLV is declining is a candidate for intervention.

## Resources

- **Fader, Hardie, Lee (2005)**: "Counting Your Customers the Easy Way: An Alternative to the Pareto/NBD Model" -- the foundational BG/NBD paper. [Link](https://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)
- **Fader & Hardie (2013)**: "The Gamma-Gamma Model of Monetary Value" -- the companion paper for spending prediction. [Link](https://www.brucehardie.com/notes/025/gamma_gamma.pdf)
- **lifetimes library** (Python): [lifetimes docs](https://lifetimes.readthedocs.io/) -- the go-to Python implementation.
- **BTYD R package**: [CRAN - BTYD](https://cran.r-project.org/package=BTYD)
- **Peter Fader's Wharton course**: "Customer Analytics" on Coursera -- taught by the inventor of BG/NBD.
- **Daniel McCarthy's work**: Advanced CLV models with covariates and contractual settings. [dannymccarthy.com](https://www.dannymccarthy.com)
- **Actuarial connection**: For the link between CLV and actuarial present values, see any life contingencies textbook (e.g., Bowers et al., "Actuarial Mathematics").
- **DataCamp**: "Predicting Customer Lifetime Value" course.
