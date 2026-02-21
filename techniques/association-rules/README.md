# Association Rules (Market Basket Analysis)

## What it is

Association rule mining is a technique for discovering interesting relationships (associations) between items in large datasets. The classic application is **market basket analysis**: given a database of customer transactions, what products tend to be purchased together? The output is a set of rules like "customers who buy bread and butter also tend to buy milk," quantified by metrics that tell you how strong, frequent, and surprising each association is. Beyond retail, association rules apply wherever you have sets of co-occurring items -- insurance product bundles, medical co-diagnoses, website page co-visits, or claims with co-occurring damage types.

## Why it matters

- **Cross-sell and bundling**: The most direct business application. If you know that auto insurance customers who also have renter's insurance are 3x more likely to add an umbrella policy, you can target that cross-sell proactively. Association rules make these patterns explicit.
- **Insurance product design**: Understanding which coverage types naturally cluster together informs product bundling. If home + auto + umbrella frequently co-occur, design a bundle with a discount.
- **Claims analysis**: In insurance claims, association rules can reveal patterns like "claims involving water damage and electrical failure tend to co-occur," which helps adjusters anticipate related damages and improves fraud detection.
- **Recommendation systems**: Association rules are the simplest form of collaborative filtering. "Customers who bought X also bought Y" is an association rule in action.
- **Operational efficiency**: Understanding item associations helps with inventory placement (put associated items near each other), staffing (bundle purchases may need different handling), and promotional strategy.
- **Simplicity and interpretability**: Unlike black-box recommendation algorithms, association rules produce human-readable patterns that business stakeholders can immediately understand and act on.

## When to use it

- You have transactional data where each transaction contains a set of items (products, services, coverage types).
- You want to identify which items are frequently bought together.
- You are designing cross-sell campaigns and need data-driven product pairing recommendations.
- You want to optimize insurance product bundles based on observed purchasing patterns.
- You are analyzing claims data to find patterns of co-occurring damage types or coverage triggers.
- You need an interpretable recommendation engine that does not require user ratings or complex ML infrastructure.
- You want to explore a dataset for interesting patterns before building a more sophisticated model.

## How it works (intuitive)

### The shopping cart analogy

Imagine standing at a grocery store checkout, watching carts go by. After watching 1,000 carts, you notice that whenever someone buys pasta, they almost always buy tomato sauce too. That is an association rule: {pasta} -> {tomato sauce}. Now imagine systematically checking every possible item combination -- that is what the Apriori and FP-Growth algorithms do.

### The three key metrics

Before diving into algorithms, understand the three metrics that quantify each rule:

**Support**: How often does this combination appear in the dataset?

$$\text{Support}(A \Rightarrow B) = \frac{|\text{transactions containing both A and B}|}{|\text{total transactions}|}$$

Think of support as "popularity." A rule with 0.01 support means 1% of all transactions contain both items. Low support means the pattern is rare; high support means it is common.

**Confidence**: When A appears, how often does B also appear?

$$\text{Confidence}(A \Rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} = P(B | A)$$

Think of confidence as "reliability." A confidence of 0.80 means that 80% of the time someone buys A, they also buy B. But confidence can be misleading -- if B is already very popular (everyone buys milk), then {anything} -> {milk} will have high confidence just because milk is common.

**Lift**: Is the co-occurrence more than you would expect by chance?

$$\text{Lift}(A \Rightarrow B) = \frac{\text{Confidence}(A \Rightarrow B)}{\text{Support}(B)} = \frac{P(A \cap B)}{P(A) \cdot P(B)}$$

Think of lift as "surprise factor." Lift > 1 means A and B appear together more often than expected if they were independent. Lift = 1 means no association. Lift < 1 means A and B *avoid* each other (negative association). Lift is the most useful metric because it accounts for the baseline popularity of both items.

### Additional metrics

**Conviction**: Measures the degree to which A appearing without B would be unexpected:

$$\text{Conviction}(A \Rightarrow B) = \frac{1 - \text{Support}(B)}{1 - \text{Confidence}(A \Rightarrow B)}$$

Conviction ranges from 0 to infinity. A conviction of 1 means A and B are independent. Higher conviction means stronger dependence. Unlike lift, conviction is asymmetric -- the direction of the rule matters.

**Leverage**: The difference between observed co-occurrence and expected co-occurrence under independence:

$$\text{Leverage}(A \Rightarrow B) = \text{Support}(A \cup B) - \text{Support}(A) \cdot \text{Support}(B)$$

### The Apriori algorithm

The naive approach to finding association rules is to check every possible item combination. With $n$ items, there are $2^n$ possible itemsets -- computationally explosive. The Apriori algorithm makes this tractable using a simple principle:

**Apriori principle**: If an itemset is infrequent (support below the threshold), then all its supersets must also be infrequent.

In other words, if nobody buys {pasta, saffron, truffles} together, then you do not need to check {pasta, saffron, truffles, lobster}. This lets the algorithm prune the search space dramatically.

**Step-by-step**:
1. Set a minimum support threshold (e.g., 0.01).
2. Find all individual items that meet minimum support (frequent 1-itemsets).
3. Combine frequent 1-itemsets to generate candidate 2-itemsets. Check which meet minimum support.
4. Combine frequent 2-itemsets to generate candidate 3-itemsets. Check support.
5. Repeat until no more frequent itemsets can be found.
6. For each frequent itemset, generate rules and filter by minimum confidence and/or lift.

### FP-Growth algorithm

FP-Growth (Frequent Pattern Growth) is a faster alternative to Apriori. Instead of repeatedly scanning the database and generating candidates, it:
1. Builds a compressed data structure called an FP-tree that stores the transaction database in memory.
2. Mines frequent patterns directly from the tree using a divide-and-conquer approach.

FP-Growth avoids the expensive candidate generation step and is typically 2-10x faster than Apriori on large datasets. In practice, use FP-Growth unless your dataset is small enough that Apriori runs in seconds.

### The newspaper analogy

Imagine you run a newspaper and want to know which articles people read together. Checking every possible pair of articles from your 500-article archive would mean $\binom{500}{2} = 124,750$ pairs, and three-article combinations would be over 20 million. Apriori says: "If articles A and B are each read by less than 1% of readers, do not bother checking if A, B, and C are read together." FP-Growth says: "Let me build a compact summary of reading patterns and extract the popular combinations directly, without enumerating all possibilities."

## Key concepts

### Formal definitions

Let $I = \{i_1, i_2, \ldots, i_n\}$ be a set of $n$ items. Let $D = \{T_1, T_2, \ldots, T_m\}$ be a database of $m$ transactions, where each transaction $T_j \subseteq I$.

An **association rule** is an implication $A \Rightarrow B$ where $A, B \subset I$ and $A \cap B = \emptyset$.

### Metric summary table

| Metric | Formula | Range | Interpretation |
|--------|---------|-------|---------------|
| Support | $\frac{|A \cup B|}{|D|}$ | [0, 1] | How common is this combination? |
| Confidence | $\frac{\text{Support}(A \cup B)}{\text{Support}(A)}$ | [0, 1] | Given A, how likely is B? |
| Lift | $\frac{\text{Support}(A \cup B)}{\text{Support}(A) \cdot \text{Support}(B)}$ | [0, +inf) | How much more likely than chance? |
| Conviction | $\frac{1 - \text{Support}(B)}{1 - \text{Confidence}(A \Rightarrow B)}$ | [0, +inf) | How dependent is A on B? |
| Leverage | $\text{Supp}(A \cup B) - \text{Supp}(A) \cdot \text{Supp}(B)$ | [-0.25, 0.25] | Absolute difference from independence |

### Typical threshold choices

| Setting | Min support | Min confidence | Min lift |
|---------|------------|---------------|----------|
| Retail (many items, low frequency) | 0.005 - 0.01 | 0.20 - 0.40 | > 1.5 |
| Insurance (fewer products, higher frequency) | 0.01 - 0.05 | 0.30 - 0.50 | > 2.0 |
| Web clickstream | 0.001 - 0.01 | 0.10 - 0.30 | > 1.5 |

These are starting points. Adjust based on your data size and the number of actionable rules you want.

### Apriori complexity

- Time complexity: $O(2^n)$ in the worst case, but pruning makes it practical for datasets with up to ~10,000 unique items.
- The minimum support threshold is the primary knob for controlling runtime. Lower support = more itemsets to check = slower.

### FP-Growth complexity

- Typically $O(|D| \cdot |I|)$ for tree construction, with mining time depending on the tree structure.
- Much faster than Apriori for dense datasets or low support thresholds.

### Insurance-specific rule patterns

| Rule | Business interpretation |
|------|----------------------|
| {Auto} -> {Home} | Auto policyholders often also buy home insurance. Target auto-only customers for home cross-sell. |
| {Home, Auto} -> {Umbrella} | Bundle buyers are prime candidates for umbrella liability. |
| {Young driver endorsement} -> {Higher deductible} | Families with young drivers tend to choose higher deductibles to manage premium. |
| {Water damage claim} -> {Mold remediation claim} | Water claims frequently lead to mold claims -- anticipate and inspect proactively. |
| {Comprehensive, Collision} -> {Roadside assistance} | Customers with full coverage tend to add roadside assistance. |

## Python implementation

```python
# Using mlxtend (pip install mlxtend)
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

# --- Step 1: Prepare transaction data ---
# Input: list of lists, where each inner list is a transaction
transactions = [
    ['auto', 'home', 'umbrella'],
    ['auto', 'renters'],
    ['auto', 'home'],
    ['auto', 'home', 'umbrella', 'life'],
    ['home', 'renters'],
    # ... more transactions
]

# Encode to binary matrix
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# --- Step 2: Find frequent itemsets ---
# Option A: Apriori
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# Option B: FP-Growth (faster)
frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)

# --- Step 3: Generate association rules ---
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.5)

# --- Step 4: Filter and sort ---
# Top rules by lift
top_rules = rules.sort_values('lift', ascending=False).head(20)

# Rules for a specific antecedent
auto_rules = rules[rules['antecedents'].apply(lambda x: 'auto' in x)]

# Print nicely
for _, row in top_rules.iterrows():
    ant = ', '.join(list(row['antecedents']))
    con = ', '.join(list(row['consequents']))
    print(f"{ant} -> {con}  "
          f"(support={row['support']:.3f}, "
          f"confidence={row['confidence']:.3f}, "
          f"lift={row['lift']:.2f})")
```

Key libraries:
- **mlxtend**: The standard Python library for association rule mining. Implements Apriori, FP-Growth, and rule generation. [mlxtend docs](http://rasbt.github.io/mlxtend/)
- **pandas** for data manipulation and one-hot encoding.
- **networkx** for visualizing rules as a network graph (items as nodes, rules as edges).

## R implementation (if applicable)

```r
# install.packages("arules")
# install.packages("arulesViz")
library(arules)
library(arulesViz)

# Read transactions
txn <- read.transactions("transactions.csv", format = "basket", sep = ",")

# Find rules
rules <- apriori(txn,
                 parameter = list(supp = 0.01, conf = 0.3, minlen = 2))

# Inspect top rules by lift
inspect(sort(rules, by = "lift")[1:20])

# Visualize
plot(rules, method = "graph", engine = "htmlwidget")
plot(rules, method = "scatter", measure = c("support", "confidence"),
     shading = "lift")
```

**Python vs. R tradeoff**: R's `arules` package is the most mature and feature-rich implementation, with excellent visualization via `arulesViz`. Python's `mlxtend` is simpler and integrates better with pandas/scikit-learn workflows. For pure association rule mining, R's ecosystem is arguably stronger. For integrating rules into a larger ML pipeline, Python is more practical.

## Gotchas & lessons learned

### 1. High confidence does not mean high lift
A rule {anything} -> {milk} will have high confidence if 90% of customers buy milk, but the lift will be close to 1 (no surprise). Always look at lift alongside confidence. Confidence alone is misleading for popular items.

### 2. Support threshold is critical
Setting minimum support too low floods you with thousands of rules, most of which are noise. Setting it too high misses interesting niche patterns. Start with a moderate threshold and adjust. The "right" number of rules depends on how many you can actually act on.

### 3. The rule direction matters
{bread} -> {butter} and {butter} -> {bread} are different rules with potentially different confidence values. The antecedent is what the customer already has; the consequent is what you recommend. Choose the direction that matches your business use case.

### 4. Correlation does not imply causation (again)
Just because {umbrella policy} and {home insurance} co-occur does not mean buying home insurance causes people to buy umbrella policies. It might be that both are marketed together, or that certain customer segments tend to buy comprehensive coverage in general.

### 5. Transaction definition matters
What counts as a "transaction"? For a grocery store, it is a single checkout. For insurance, it could be all policies a customer holds at a point in time, or all products purchased in the first year. Different definitions produce different rules.

### 6. Rare items create noise
Items purchased by very few customers generate unstable rules. A rule based on 3 transactions might have a lift of 50 but is not reliable. Filter rules by both minimum support AND minimum number of transactions containing the itemset.

### 7. Too many items = too many rules
If you have 10,000 unique items, the number of potential rules is astronomical. Reduce dimensionality first: aggregate items into categories, or filter to the top N most common items before mining.

### 8. Redundant rules
Many generated rules are redundant. {A, B} -> {C} and {A} -> {C} might both appear if B adds nothing. Prune redundant rules by removing any rule whose confidence does not improve when adding items to the antecedent.

### 9. Time dimension is ignored
Standard association rules do not consider time. They treat the entire transaction database as static. If purchasing patterns change over time (seasonality, trends), consider running association analysis on time-windowed subsets.

### 10. Actionability over statistical significance
A rule with lift of 1.2 might be statistically significant with enough data, but is it worth acting on? Focus on rules that are both statistically strong (high lift, reasonable support) and actionable (you can actually change the product placement, send a targeted offer, or redesign a bundle).

## Resources

- **Agrawal & Srikant (1994)**: "Fast Algorithms for Mining Association Rules" -- the original Apriori paper.
- **Han, Pei & Yin (2000)**: "Mining Frequent Patterns without Candidate Generation" -- the FP-Growth paper.
- **mlxtend documentation**: [http://rasbt.github.io/mlxtend/](http://rasbt.github.io/mlxtend/) -- Python implementation.
- **arules R package**: [CRAN - arules](https://cran.r-project.org/package=arules) -- comprehensive R implementation.
- **arulesViz**: [CRAN - arulesViz](https://cran.r-project.org/package=arulesViz) -- visualization for association rules.
- **DataCamp**: "Market Basket Analysis in Python" course.
- **Practical recommendation**: Michael Hahsler's website has excellent tutorials on association rules in R: [https://michael.hahsler.net/](https://michael.hahsler.net/)
- **Insurance application**: Search for "insurance product bundling analytics" for industry-specific case studies.
