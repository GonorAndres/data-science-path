CREATE OR REPLACE TABLE stg_products AS
SELECT
  product_id,
  product_category_name,
  CAST(product_weight_g AS DOUBLE) AS product_weight_g,
  CAST(product_length_cm AS DOUBLE) AS product_length_cm,
  CAST(product_height_cm AS DOUBLE) AS product_height_cm,
  CAST(product_width_cm AS DOUBLE) AS product_width_cm
FROM read_csv_auto('data/olist_products_dataset.csv');
