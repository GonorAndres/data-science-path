CREATE OR REPLACE TABLE mart_rolling_revenue AS
WITH daily AS (
  SELECT
    CAST(order_purchase_at AS DATE) AS day,
    SUM(order_revenue) AS revenue
  FROM int_order_revenue
  GROUP BY 1
)
SELECT
  day,
  revenue,
  AVG(revenue) OVER (
    ORDER BY day
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS revenue_7d,
  AVG(revenue) OVER (
    ORDER BY day
    ROWS BETWEEN 27 PRECEDING AND CURRENT ROW
  ) AS revenue_28d
FROM daily
ORDER BY day;
