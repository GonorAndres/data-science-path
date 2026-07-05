CREATE OR REPLACE TABLE mart_retention_curve AS
WITH cohort AS (
  SELECT customer_unique_id, DATE_TRUNC('month', first_purchase_at) AS cohort_month
  FROM int_customer_orders
),
activity AS (
  SELECT DISTINCT
    c.customer_unique_id,
    DATE_TRUNC('month', r.order_purchase_at) AS activity_month
  FROM int_order_revenue r
  JOIN stg_customers sc ON sc.customer_id = r.customer_id
  JOIN cohort c ON c.customer_unique_id = sc.customer_unique_id
),
pairs AS (
  SELECT
    c.cohort_month,
    a.activity_month,
    DATE_DIFF('month', c.cohort_month, a.activity_month) AS months_since,
    c.customer_unique_id
  FROM cohort c
  JOIN activity a USING (customer_unique_id)
),
cohort_size AS (
  SELECT cohort_month, COUNT(DISTINCT customer_unique_id) AS n FROM cohort GROUP BY 1
)
SELECT
  p.cohort_month,
  p.months_since,
  COUNT(DISTINCT p.customer_unique_id) AS retained,
  1.0 * COUNT(DISTINCT p.customer_unique_id) / cs.n AS retention_pct
FROM pairs p
JOIN cohort_size cs USING (cohort_month)
WHERE p.months_since BETWEEN 0 AND 12
GROUP BY 1, 2, cs.n
ORDER BY 1, 2;
