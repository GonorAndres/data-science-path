CREATE OR REPLACE TABLE stg_orders AS
SELECT
  order_id,
  customer_id,
  order_status,
  TRY_CAST(order_purchase_timestamp AS TIMESTAMP) AS order_purchase_at,
  TRY_CAST(order_approved_at AS TIMESTAMP) AS order_approved_at,
  TRY_CAST(order_delivered_carrier_date AS TIMESTAMP) AS order_delivered_carrier_at,
  TRY_CAST(order_delivered_customer_date AS TIMESTAMP) AS order_delivered_customer_at,
  TRY_CAST(order_estimated_delivery_date AS TIMESTAMP) AS order_estimated_delivery_at
FROM read_csv_auto('data/olist_orders_dataset.csv');
