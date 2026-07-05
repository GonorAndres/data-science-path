CREATE OR REPLACE TABLE int_order_revenue AS
SELECT
  o.order_id,
  o.customer_id,
  o.order_status,
  o.order_purchase_at,
  SUM(i.price + i.freight_value) AS order_revenue,
  COUNT(*) AS n_items
FROM stg_orders o
LEFT JOIN stg_order_items i USING (order_id)
WHERE o.order_status = 'delivered'
GROUP BY 1, 2, 3, 4;
