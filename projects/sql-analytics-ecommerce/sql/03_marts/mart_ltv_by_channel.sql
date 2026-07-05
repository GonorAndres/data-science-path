CREATE OR REPLACE TABLE mart_ltv_by_channel AS
WITH first_payment AS (
  SELECT
    r.customer_id,
    p.payment_type AS first_channel
  FROM int_order_revenue r
  JOIN stg_payments p USING (order_id)
  QUALIFY ROW_NUMBER() OVER (PARTITION BY r.customer_id ORDER BY r.order_purchase_at) = 1
),
ltv AS (
  SELECT c.customer_unique_id, c.lifetime_revenue, c.n_orders
  FROM int_customer_orders c
),
link AS (
  SELECT
    sc.customer_unique_id,
    fp.first_channel,
    l.lifetime_revenue,
    l.n_orders
  FROM first_payment fp
  JOIN stg_customers sc ON sc.customer_id = fp.customer_id
  JOIN ltv l USING (customer_unique_id)
)
SELECT
  first_channel,
  COUNT(*) AS n_customers,
  AVG(lifetime_revenue) AS avg_ltv,
  MEDIAN(lifetime_revenue) AS median_ltv,
  AVG(n_orders) AS avg_orders_per_customer
FROM link
GROUP BY 1
ORDER BY avg_ltv DESC;
