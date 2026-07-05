CREATE OR REPLACE TABLE mart_product_performance AS
WITH rev AS (
  SELECT
    p.product_category_name,
    i.product_id,
    SUM(i.price) AS product_revenue,
    COUNT(*) AS n_sales
  FROM stg_order_items i
  JOIN stg_products p USING (product_id)
  JOIN int_order_revenue r USING (order_id)
  GROUP BY 1, 2
)
SELECT
  product_category_name,
  product_id,
  product_revenue,
  n_sales,
  RANK() OVER (PARTITION BY product_category_name ORDER BY product_revenue DESC) AS rank_in_category
FROM rev
ORDER BY product_category_name, rank_in_category;
