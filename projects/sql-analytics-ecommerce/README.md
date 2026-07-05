# sql-analytics-ecommerce

## Resumen (ES)

**Problema:** analizar el dataset publico de Olist (e-commerce brasileno) con SQL de calidad de warehouse: retencion por cohorte, embudo de conversion, LTV por canal, revenue rolling. Estructura tipo dbt con pruebas y documentacion.

**Metodo:** DuckDB como warehouse local (cero setup), dbt-lite para organizar modelos, SQL limpio con CTEs y window functions. Un dashboard complementario en Streamlit (ver `streamlit-exec-dashboard`).

**Resultado objetivo:** 15 queries analiticas bien documentadas, lineage visible, tests de esquema corriendo, dashboard final para stakeholders.

## Summary (EN)

**Problem:** analyze the public Olist (Brazilian e-commerce) dataset with warehouse-grade SQL: cohort retention, conversion funnel, channel LTV, rolling revenue. dbt-style structure with tests and documentation.

**Method:** DuckDB as the local warehouse (zero setup), dbt-lite for model organization, clean SQL with CTEs and window functions. Complementary Streamlit dashboard (see `streamlit-exec-dashboard`).

**Target result:** 15 well-documented analytical queries, visible lineage, schema tests passing, final dashboard for stakeholders.

---

## Why this project

SQL mastery is the single biggest gap in most data science portfolios. A recruiter who sees a project with only Jupyter notebooks filters for reasonable SQL skills via interview; a recruiter who sees a project with a real SQL pipeline assumes baseline competence.

This project:

- Uses DuckDB so anyone can run it with `pip install duckdb` and a single data file. No warehouse account needed.
- Follows dbt patterns (staging, intermediate, marts) without requiring a full dbt deployment.
- Produces the queries behind the `streamlit-exec-dashboard` project's charts.

## Dataset

Olist Brazilian E-commerce (Kaggle). Public. 100k orders 2016-2018 with customers, products, sellers, payments, reviews, geolocation.

https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce

## Structure

```
sql-analytics-ecommerce/
├── README.md
├── theory/
│   ├── 00_problem_framing.md
│   ├── 01_dimensional_modeling.md
│   ├── 02_window_functions_deep_dive.md
│   ├── 03_cohort_and_funnel_sql.md
│   └── 04_performance_patterns.md
├── data_download.py
├── sql/
│   ├── 01_staging/
│   │   ├── stg_orders.sql
│   │   ├── stg_customers.sql
│   │   ├── stg_products.sql
│   │   ├── stg_order_items.sql
│   │   └── stg_payments.sql
│   ├── 02_intermediate/
│   │   ├── int_customer_orders.sql
│   │   └── int_order_revenue.sql
│   ├── 03_marts/
│   │   ├── mart_monthly_revenue.sql
│   │   ├── mart_customer_cohorts.sql
│   │   ├── mart_retention_curve.sql
│   │   ├── mart_product_performance.sql
│   │   ├── mart_ltv_by_channel.sql
│   │   ├── mart_review_sentiment.sql
│   │   └── mart_rolling_revenue.sql
│   └── tests/
│       ├── not_null_customer_id.sql
│       ├── unique_order_id.sql
│       └── positive_revenue.sql
├── src/
│   ├── __init__.py
│   ├── pipeline.py              # orchestrate DuckDB queries in order
│   └── tests.py                 # run SQL tests
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── Makefile
```

## Quickstart

```bash
pip install -r requirements.txt
python data_download.py         # downloads Olist CSVs into data/
python -m src.pipeline          # runs all SQL models in order
python -m src.tests             # runs SQL tests
```

Output: a `data/olist.duckdb` file with all staging / intermediate / mart tables materialized.

## Key queries

| Query | Purpose | Complexity |
|-------|---------|------------|
| `mart_monthly_revenue` | Total revenue by month | Basic aggregation |
| `mart_customer_cohorts` | Signup cohorts | GROUP BY + date_trunc |
| `mart_retention_curve` | Cohort x month retention | Self-join + window |
| `mart_ltv_by_channel` | Revenue per customer over 12 months | CTEs + window |
| `mart_rolling_revenue` | 7-day and 28-day rolling revenue | Window with range |
| `mart_product_performance` | Rank products by revenue in category | Window ranking |
| `mart_review_sentiment` | Average review score by product | Join + agg |

## Decisions and trade-offs

| Decision | Alternative | Why |
|----------|-------------|-----|
| DuckDB | PostgreSQL, BigQuery | Zero setup, one-file warehouse, same SQL. |
| dbt-lite (custom runner) | Full dbt | dbt adds deps; this project demos the ideas without the framework weight. |
| SQL only in `sql/`, Python only as orchestrator | Jupyter with SQL magic | SQL files are reusable and testable; notebooks are not. |
| Olist | Other public e-commerce | Olist has reviews and geolocation, richer for analytics. |

## References

- `docs/theory/15_sql_for_data_scientists.md`
- dbt docs: docs.getdbt.com
- Kimball, *The Data Warehouse Toolkit*.
