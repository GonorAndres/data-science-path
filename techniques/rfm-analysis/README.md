# RFM Analysis

## What it is

RFM Analysis is a customer segmentation technique that scores every customer on three behavioral dimensions: **Recency** (how recently they transacted), **Frequency** (how often they transact), and **Monetary** (how much they spend). By combining these three scores, you create simple but powerful customer segments -- from your best "Champions" to dormant customers you might be losing. It is one of the oldest and most practical frameworks in direct marketing and customer analytics, and it remains widely used because it is easy to explain, fast to compute, and surprisingly effective at prioritizing outreach.

## Why it matters

For an actuarial or data science practitioner, RFM sits at the intersection of customer behavior and business strategy:

- **Insurance cross-sell and upsell**: Identifying which policyholders are most engaged (high R, high F) tells you who is most receptive to hearing about additional products -- home insurance for an auto policyholder, or an umbrella policy for someone with both.
- **Retention targeting**: Customers whose recency score is declining are showing early warning signs of churn. RFM gives you a lightweight early-warning system before you build a full churn model.
- **Resource allocation**: Marketing budgets are finite. RFM helps you avoid the classic mistake of spreading spend equally across all customers. Instead, you can concentrate on the segments with the highest expected return.
- **Foundation for advanced models**: RFM features (days since last purchase, purchase count, total spend) are often the most predictive features in downstream ML models for CLV, churn, or next-best-action systems.
- **Actuarial parallel**: Actuaries are used to segmenting risk pools. RFM is essentially the same idea applied to revenue pools -- you are grouping customers by their observed behavior to predict future value.

## When to use it

- You have transactional data with customer IDs, transaction dates, and transaction amounts.
- You need a quick, interpretable segmentation without building a full ML pipeline.
- You want to prioritize a marketing campaign (email, direct mail, agent outreach) by customer value.
- You are building a cross-sell or upsell strategy for insurance products and need to identify engaged policyholders.
- You need baseline customer segments to compare against a more sophisticated clustering or CLV model.
- Your stakeholders want something they can understand in a 10-minute meeting -- RFM segments have intuitive names ("Champions", "At Risk", "Lost").
- You are doing exploratory analysis on a new customer dataset and want a fast read on the customer base composition.

## How it works (intuitive)

### The core idea

Think of RFM like grading students on three subjects. Each customer gets a grade (score) on Recency, Frequency, and Monetary. Then you combine the grades to decide which "honors program" (segment) each customer belongs to.

### Step-by-step process

**Step 1: Compute the raw RFM values**

For each customer, calculate three numbers from the transaction history:

| Metric | What you compute | Example |
|--------|-----------------|---------|
| Recency (R) | Days since the customer's most recent transaction, measured from a reference date (usually today or the end of the analysis period) | Customer A last purchased 5 days ago, Customer B last purchased 120 days ago |
| Frequency (F) | Count of transactions in the analysis period | Customer A made 12 purchases, Customer B made 2 |
| Monetary (M) | Total (or average) spend in the analysis period | Customer A spent $2,400 total, Customer B spent $150 |

**Step 2: Score each dimension**

The most common approach is **quintile-based scoring**: sort all customers by each metric and divide them into 5 equal-sized groups (quintiles), assigning scores 1 through 5.

- For **Recency**, a *lower* number of days is better (more recent = more engaged), so the most recent quintile gets a score of 5, the least recent gets 1.
- For **Frequency**, a *higher* count is better, so the most frequent quintile gets 5.
- For **Monetary**, a *higher* spend is better, so the top-spending quintile gets 5.

After scoring, every customer has a triplet like (R=4, F=3, M=5).

**Step 3: Create segments**

You can either:

1. **Concatenate scores** into an "RFM cell" (e.g., "435") and define named segments by grouping cells. For instance, all customers with R >= 4 and F >= 4 might be called "Champions."
2. **Sum the scores** into a single composite RFM score (e.g., 4+3+5 = 12 out of 15) and segment by score bands.
3. **Use a lookup table** that maps score combinations to business-friendly segment names.

**Step 4: Act on the segments**

Each segment gets a different treatment:

| Segment | Typical RFM profile | Suggested action |
|---------|---------------------|------------------|
| Champions | R=5, F=5, M=5 | Reward loyalty, ask for referrals, early access to new products |
| Loyal Customers | R=4-5, F=4-5, M=3-5 | Upsell, cross-sell, loyalty program enrollment |
| Potential Loyalists | R=4-5, F=2-3, M=2-3 | Nurture with targeted offers to increase frequency |
| At Risk | R=2-3, F=3-5, M=3-5 | Win-back campaigns -- these were once valuable |
| Hibernating | R=1-2, F=1-2, M=1-2 | Low-cost reactivation or deprioritize |
| Lost | R=1, F=1, M=1 | Minimal investment unless strategic |

### The restaurant analogy

Imagine you own a restaurant and you keep a guest book:
- **Recency**: Did this guest come in last week, or haven't you seen them in six months?
- **Frequency**: Do they come every week, or was it a one-time visit?
- **Monetary**: Do they order the tasting menu with wine pairings, or just a coffee?

A guest who was here last week (R=5), comes every week (F=5), and always orders the tasting menu (M=5) is your VIP. You personally greet them, maybe send them a holiday card. A guest who came once six months ago and ordered a coffee? You probably won't send them the VIP invite.

RFM is exactly this intuition, scaled to thousands or millions of customers.

## Key concepts

### RFM metric definitions

$$R_i = d_{ref} - d_{last,i}$$

where $d_{ref}$ is the reference date and $d_{last,i}$ is the date of customer $i$'s most recent transaction.

$$F_i = \sum_{t=1}^{T} \mathbb{1}[\text{customer } i \text{ transacted at time } t]$$

$$M_i = \sum_{t=1}^{T} v_{i,t}$$

where $v_{i,t}$ is the value of customer $i$'s transaction at time $t$. Some practitioners use the average transaction value $\bar{M}_i = M_i / F_i$ instead of the total.

### Quintile scoring

For each metric, customers are ranked and divided into 5 equal bins:

$$\text{Score}_i^{(k)} = \text{quintile rank of customer } i \text{ on metric } k \in \{R, F, M\}$$

Scores range from 1 (worst) to 5 (best). Note that for Recency, "best" means the fewest days since last transaction, so the ranking is inverted.

### Alternative scoring methods

| Method | How it works | When to use |
|--------|-------------|-------------|
| Quintile (equal-frequency) | 5 bins with ~20% of customers each | Default; works well when distributions are roughly uniform |
| Custom percentile | Use different cutoffs (e.g., top 10%, next 20%, etc.) | When you want more granularity at the top or bottom |
| Fixed thresholds | Business-defined rules (e.g., "recency < 30 days = 5") | When business context dictates what "good" means |
| K-means on RFM values | Cluster on the raw (possibly log-transformed) R, F, M values | When you want data-driven segments rather than rule-based |
| Pareto-based | 80/20 rule: top 20% of spenders get the highest score | When the distribution is extremely skewed |

### Composite scoring

**Additive**: $\text{RFM}_{total} = R + F + M$ (range 3-15 with quintile scores)

**Weighted**: $\text{RFM}_{weighted} = w_R \cdot R + w_F \cdot F + w_M \cdot M$ where weights reflect business priorities. A retention-focused business might weight R heavily; a revenue-focused business might weight M.

**Concatenated**: Treat the RFM score as a string "RFM" = "435", giving $5^3 = 125$ possible cells.

### Segment definitions (common scheme)

| Segment name | R score | F score | M score |
|-------------|---------|---------|---------|
| Champions | 5 | 5 | 4-5 |
| Loyal | 4-5 | 4-5 | 3-5 |
| Potential Loyalists | 4-5 | 2-3 | 2-3 |
| Recent Customers | 5 | 1 | 1-2 |
| Promising | 3-4 | 1-2 | 1-2 |
| Needs Attention | 3 | 3 | 3 |
| About to Sleep | 2-3 | 2-3 | 2-3 |
| At Risk | 1-2 | 4-5 | 4-5 |
| Can't Lose Them | 1-2 | 5 | 5 |
| Hibernating | 1-2 | 1-2 | 1-2 |
| Lost | 1 | 1 | 1 |

The exact definitions vary by business. The important thing is that the names are meaningful to your stakeholders.

### Insurance-specific applications

- **Cross-sell scoring**: Policyholders with high R and F (frequent interactions, recent activity) are prime targets for cross-sell. Their M score might reflect total premium or total claims-free years.
- **Renewal prioritization**: For commercial lines, R can be redefined as "days until renewal" and F as "number of policies held." This gives agents a prioritized renewal list.
- **Claims engagement**: For claims departments, R = recency of last claim report, F = frequency of communication with adjuster, M = claim severity. Segments help allocate adjuster attention.

## Python implementation

```python
import pandas as pd
import numpy as np
from datetime import datetime

# --- Step 1: Compute raw RFM values ---
def compute_rfm(df, customer_col='customer_id', date_col='transaction_date',
                amount_col='amount', reference_date=None):
    """
    Compute RFM values from a transaction-level dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction-level data with customer ID, date, and amount columns.
    customer_col : str
        Column name for customer identifier.
    date_col : str
        Column name for transaction date.
    amount_col : str
        Column name for transaction amount.
    reference_date : datetime or None
        Date to compute recency from. Defaults to max date in the data + 1 day.

    Returns
    -------
    pd.DataFrame
        Customer-level dataframe with Recency, Frequency, and Monetary columns.
    """
    df[date_col] = pd.to_datetime(df[date_col])

    if reference_date is None:
        reference_date = df[date_col].max() + pd.Timedelta(days=1)

    rfm = df.groupby(customer_col).agg(
        Recency=(date_col, lambda x: (reference_date - x.max()).days),
        Frequency=(date_col, 'nunique'),
        Monetary=(amount_col, 'sum')
    ).reset_index()

    return rfm


# --- Step 2: Score using quintiles ---
def score_rfm(rfm_df, n_bins=5):
    """
    Assign quintile-based RFM scores (1=worst, 5=best).
    Recency is reverse-scored (lower days = higher score).
    """
    rfm = rfm_df.copy()

    # Recency: lower is better, so labels are reversed
    rfm['R_score'] = pd.qcut(rfm['Recency'], q=n_bins, labels=range(n_bins, 0, -1))
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'), q=n_bins, labels=range(1, n_bins+1))
    rfm['M_score'] = pd.qcut(rfm['Monetary'].rank(method='first'), q=n_bins, labels=range(1, n_bins+1))

    # Convert to int
    for col in ['R_score', 'F_score', 'M_score']:
        rfm[col] = rfm[col].astype(int)

    rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    rfm['RFM_cell'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

    return rfm


# --- Step 3: Map to named segments ---
def assign_segments(rfm_df):
    """Assign human-readable segment names based on R and F scores."""
    def _segment(row):
        r, f = row['R_score'], row['F_score']
        if r >= 4 and f >= 4:
            return 'Champions'
        elif r >= 3 and f >= 3:
            return 'Loyal'
        elif r >= 4 and f <= 2:
            return 'Recent Customers'
        elif r <= 2 and f >= 4:
            return 'At Risk'
        elif r <= 2 and f <= 2:
            return 'Lost / Hibernating'
        else:
            return 'Needs Attention'

    rfm_df['Segment'] = rfm_df.apply(_segment, axis=1)
    return rfm_df
```

Key libraries:
- **pandas** for data manipulation and `pd.qcut` for quintile binning.
- **plotly** or **seaborn** for heatmaps of RFM segments.
- **lifetimes** (now called `btyd`) for connecting RFM to probabilistic CLV models.

## R implementation (if applicable)

The `rfm` package in R provides a complete RFM workflow:

```r
# install.packages("rfm")
library(rfm)

# rfm_table_order() computes RFM from order-level data
rfm_result <- rfm_table_order(
  data = orders_df,
  customer_id = customer_id,
  order_date = order_date,
  revenue = amount,
  analysis_date = as.Date("2024-01-01")
)

# Visualizations
rfm_heatmap(rfm_result)
rfm_bar_chart(rfm_result)
rfm_histograms(rfm_result)
rfm_order_dist(rfm_result)
```

**Python vs. R tradeoff**: Python gives more control over scoring logic and integrates better with ML pipelines. The R `rfm` package is faster for exploratory analysis and comes with built-in visualizations.

## Gotchas & lessons learned

### 1. Recency scoring direction
The most common mistake is scoring Recency the wrong way. Lower recency (fewer days since last purchase) is *better*, so the scoring must be inverted relative to Frequency and Monetary. If you use `pd.qcut` naively, the highest recency values (worst customers) get the highest score.

### 2. Tied values break quintiles
When many customers have the same value (e.g., lots of customers with exactly 1 purchase), `pd.qcut` will fail because it cannot create 5 distinct bins. Use `rank(method='first')` before binning, or switch to `pd.cut` with custom edges.

### 3. Skewed distributions
Monetary values are almost always right-skewed (a few whales, many small spenders). Quintile scoring handles this naturally because it uses ranks, not raw values. But if you use k-means on raw RFM values, apply a log transform first or the clusters will be dominated by the monetary dimension.

### 4. Time window matters
The choice of analysis period dramatically affects results. A 3-month window favors frequent, recent buyers. A 2-year window captures more of the customer lifecycle. Match the window to your business cycle -- for annual insurance renewals, a 12-18 month window makes sense.

### 5. RFM is descriptive, not predictive
RFM tells you what customers *have done*, not what they *will do*. A customer scored (5,5,5) today could churn tomorrow. For prediction, use RFM as input features to a classification or CLV model.

### 6. Segment boundaries are arbitrary
The difference between an R score of 3 and 4 is just a quintile boundary. A customer right on the edge of two segments could flip with one more transaction. Do not over-interpret segment membership for individual customers; focus on segment-level patterns and strategies.

### 7. Watch for returns and refunds
If your transaction data includes returns (negative amounts), decide upfront whether to include them in the Monetary calculation. Excluding returns inflates M; including them is more accurate but can create negative M values for some customers.

### 8. B2B vs. B2C differences
In B2B or commercial insurance, one "customer" (a company) might have multiple decision-makers and irregular but large transactions. Standard quintile scoring may not work well. Consider adjusting Frequency to count distinct interaction types or using deciles instead of quintiles.

### 9. Static vs. dynamic segmentation
RFM scores computed once are a snapshot. For ongoing use, recompute monthly or quarterly and track how customers move between segments over time. This "segment migration" analysis is more valuable than any single snapshot.

### 10. Do not ignore the "Lost" segment entirely
It is tempting to write off low-RFM customers. But in insurance, a "Lost" customer might have let a policy lapse and could be reactivated at renewal time with a targeted offer. Always check if there is a cost-effective reactivation play before abandoning a segment.

## Resources

- **Original paper**: Arthur Hughes, "Strategic Database Marketing" -- the book that popularized RFM in direct marketing.
- **Putler blog**: [RFM Analysis Guide](https://www.putler.com/rfm-analysis/) -- practical walkthrough with visuals.
- **Python rfm package**: [github.com/joaolcorreia/RFM-analysis](https://github.com/joaolcorreia/RFM-analysis)
- **R rfm package**: [CRAN - rfm](https://cran.r-project.org/package=rfm)
- **lifetimes library** (Python): Connects RFM to BG/NBD and Gamma-Gamma CLV models. [lifetimes docs](https://lifetimes.readthedocs.io/)
- **Google Analytics RFM**: [Think with Google - RFM Segmentation](https://www.thinkwithgoogle.com/)
- **DataCamp**: "Customer Segmentation in Python" course covers RFM as a foundational technique.
