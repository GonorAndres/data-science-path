"""Smoke test for the SQL pipeline scaffolding."""

from pathlib import Path


def test_all_sql_files_exist():
    project = Path(__file__).resolve().parent.parent
    expected = [
        "sql/01_staging/stg_orders.sql",
        "sql/01_staging/stg_customers.sql",
        "sql/01_staging/stg_order_items.sql",
        "sql/01_staging/stg_products.sql",
        "sql/01_staging/stg_payments.sql",
        "sql/02_intermediate/int_order_revenue.sql",
        "sql/02_intermediate/int_customer_orders.sql",
        "sql/03_marts/mart_monthly_revenue.sql",
        "sql/03_marts/mart_customer_cohorts.sql",
        "sql/03_marts/mart_retention_curve.sql",
        "sql/03_marts/mart_rolling_revenue.sql",
        "sql/03_marts/mart_product_performance.sql",
        "sql/03_marts/mart_ltv_by_channel.sql",
    ]
    for relative_path in expected:
        assert (project / relative_path).exists(), relative_path
