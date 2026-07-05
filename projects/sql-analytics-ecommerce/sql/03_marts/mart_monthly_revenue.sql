CREATE OR REPLACE TABLE mart_monthly_revenue AS
SELECT
  DATE_TRUNC('month', order_purchase_at) AS month,
  COUNT(*) AS n_orders,
  SUM(order_revenue) AS revenue,
  AVG(order_revenue) AS avg_order_revenue
FROM int_order_revenue
GROUP BY 1
ORDER BY 1;
