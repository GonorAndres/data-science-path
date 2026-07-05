# Cohort and Funnel SQL

## The pattern

Cohort and funnel analyses dominate product analytics. Core pattern: segment users, measure behavior per segment, compare.

## Signup-cohort retention

1. Define cohort: `DATE_TRUNC('month', first_purchase_date)`.
2. For each user, list all subsequent purchase dates.
3. Compute months-since-cohort.
4. Aggregate: per cohort and per months-since, count distinct active users.
5. Normalize by cohort size.

```sql
WITH cohort AS (
  SELECT
    customer_id,
    DATE_TRUNC('month', MIN(order_date)) AS cohort_month
  FROM orders
  GROUP BY 1
),
activity AS (
  SELECT DISTINCT
    o.customer_id,
    DATE_TRUNC('month', o.order_date) AS activity_month
  FROM orders o
),
pairs AS (
  SELECT
    c.cohort_month,
    a.activity_month,
    DATEDIFF('month', c.cohort_month, a.activity_month) AS months_since,
    c.customer_id
  FROM cohort c
  JOIN activity a USING (customer_id)
),
cohort_size AS (
  SELECT cohort_month, COUNT(*) AS n FROM cohort GROUP BY 1
)
SELECT
  p.cohort_month,
  p.months_since,
  COUNT(DISTINCT p.customer_id) AS retained,
  1.0 * COUNT(DISTINCT p.customer_id) / cs.n AS retention_pct
FROM pairs p
JOIN cohort_size cs USING (cohort_month)
WHERE p.months_since BETWEEN 0 AND 12
GROUP BY 1, 2, cs.n
ORDER BY 1, 2;
```

## Funnel

For event data with multiple milestones, a funnel shows the drop-off between stages.

### Loose funnel (any order)

```sql
WITH steps AS (
  SELECT
    COUNT(DISTINCT CASE WHEN stage = 'view' THEN user_id END) AS viewed,
    COUNT(DISTINCT CASE WHEN stage = 'cart' THEN user_id END) AS carted,
    COUNT(DISTINCT CASE WHEN stage = 'checkout' THEN user_id END) AS checkedout,
    COUNT(DISTINCT CASE WHEN stage = 'purchase' THEN user_id END) AS purchased
  FROM events
)
SELECT
  *,
  carted * 100.0 / NULLIF(viewed, 0) AS view_to_cart,
  checkedout * 100.0 / NULLIF(carted, 0) AS cart_to_checkout,
  purchased * 100.0 / NULLIF(checkedout, 0) AS checkout_to_purchase
FROM steps;
```

### Strict funnel (order enforced)

Each step's timestamp must come after the previous step's timestamp.

```sql
WITH first_times AS (
  SELECT
    user_id,
    MIN(CASE WHEN stage = 'view' THEN event_at END) AS t_view,
    MIN(CASE WHEN stage = 'cart' THEN event_at END) AS t_cart,
    MIN(CASE WHEN stage = 'checkout' THEN event_at END) AS t_checkout,
    MIN(CASE WHEN stage = 'purchase' THEN event_at END) AS t_purchase
  FROM events
  GROUP BY user_id
)
SELECT
  COUNT(*) FILTER (WHERE t_view IS NOT NULL) AS viewed,
  COUNT(*) FILTER (WHERE t_cart > t_view) AS carted,
  COUNT(*) FILTER (WHERE t_checkout > t_cart) AS checkedout,
  COUNT(*) FILTER (WHERE t_purchase > t_checkout) AS purchased
FROM first_times;
```

DuckDB supports `FILTER`, as do PostgreSQL 9.4+ and BigQuery (with `COUNTIF`).

## LTV

Simple LTV: cumulative revenue per customer over N days/months.

```sql
WITH purchases AS (
  SELECT
    customer_id,
    MIN(order_date) AS first_date,
    SUM(order_revenue) AS total_revenue
  FROM orders
  GROUP BY 1
)
SELECT
  DATE_TRUNC('quarter', first_date) AS cohort,
  AVG(total_revenue) AS avg_ltv,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY total_revenue) AS median_ltv
FROM purchases
GROUP BY 1;
```

Advanced LTV uses horizon cuts (90-day, 365-day) and survival-adjusted estimates.

## Olist limitations

The Olist dataset has no session or pageview events. We cannot build a classical view-cart-checkout-purchase funnel. We build:

- Order status funnel: created, approved, shipped, delivered.
- Customer-level first-order-to-return funnel.

These are flagged in each mart's documentation.

## Cross-links

- `02_window_functions_deep_dive.md`
- `docs/theory/15_sql_for_data_scientists.md`
