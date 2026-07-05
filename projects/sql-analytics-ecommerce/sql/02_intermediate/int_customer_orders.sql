CREATE OR REPLACE TABLE int_customer_orders AS
SELECT
  c.customer_unique_id,
  MIN(r.order_purchase_at) AS first_purchase_at,
  MAX(r.order_purchase_at) AS last_purchase_at,
  COUNT(r.order_id) AS n_orders,
  SUM(r.order_revenue) AS lifetime_revenue
FROM int_order_revenue r
JOIN stg_customers c USING (customer_id)
GROUP BY 1;
