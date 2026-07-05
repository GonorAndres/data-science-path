CREATE OR REPLACE TABLE mart_customer_cohorts AS
SELECT
  customer_unique_id,
  DATE_TRUNC('month', first_purchase_at) AS cohort_month,
  n_orders,
  lifetime_revenue
FROM int_customer_orders;
