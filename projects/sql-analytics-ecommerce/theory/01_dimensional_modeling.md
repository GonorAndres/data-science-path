# Dimensional Modeling (for this project)

## Layers

We follow the dbt-style three-layer pattern:

1. **Staging.** Raw tables cleaned and renamed. One source file -> one staging model. Types enforced, obvious errors fixed.
2. **Intermediate.** Shared logic used by multiple marts. A customer-orders join; an order-revenue calculation.
3. **Marts.** Stakeholder-facing tables. One per analytical question.

Benefits:

- Marts are simple because they sit on top of intermediate layers that encapsulate complexity.
- Unit-level tests live at the staging layer (non-null, unique, referential integrity).
- Marts depend on intermediate; intermediate on staging; staging on sources. DAG is explicit.

## Stars and snowflakes (conceptual)

Classical data warehousing uses:

- **Fact tables.** Quantitative observations (orders, transactions).
- **Dimension tables.** Descriptive attributes (customer, product, date).

For Olist:

- Fact: `order_items` (one row per line item with price, freight, product, seller).
- Dimensions: `customers`, `products`, `sellers`, `dates`.

A fully dimensional schema is not built here; we simulate it via CTEs and joins inside marts. In a larger project with many stakeholders, an explicit star schema pays off.

## Surrogate keys

Original data uses natural keys (`customer_id` as a string). In a warehouse, we would assign surrogate integer keys for joins. Here we keep natural keys for simplicity and document the trade-off.

## Slowly changing dimensions

Olist has mostly static dimensions. A production system tracks address changes, product attribute changes, seller status changes. Type 2 SCDs preserve history with effective-date ranges. Out of scope here but noted.

## Cross-links

- `02_window_functions_deep_dive.md`
- `03_cohort_and_funnel_sql.md`
- `docs/theory/15_sql_for_data_scientists.md`
- Kimball, *The Data Warehouse Toolkit* for deeper reading.
