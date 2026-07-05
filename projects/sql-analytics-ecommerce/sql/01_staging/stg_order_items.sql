CREATE OR REPLACE TABLE stg_order_items AS
SELECT
  order_id,
  order_item_id,
  product_id,
  seller_id,
  TRY_CAST(shipping_limit_date AS TIMESTAMP) AS shipping_limit_at,
  CAST(price AS DOUBLE) AS price,
  CAST(freight_value AS DOUBLE) AS freight_value
FROM read_csv_auto('data/olist_order_items_dataset.csv');
