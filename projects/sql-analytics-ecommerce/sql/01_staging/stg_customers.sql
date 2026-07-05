CREATE OR REPLACE TABLE stg_customers AS
SELECT
  customer_id,
  customer_unique_id,
  customer_zip_code_prefix,
  customer_city,
  customer_state
FROM read_csv_auto('data/olist_customers_dataset.csv');
