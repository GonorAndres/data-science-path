# Window Functions Deep Dive

## The idea

A window function computes a value for each row using a specified window of rows. Unlike GROUP BY, it does not collapse rows.

General syntax:

```sql
FUNCTION(arg) OVER (
  PARTITION BY col_a
  ORDER BY col_b
  RANGE BETWEEN ... AND ...
)
```

## Common functions

- `ROW_NUMBER()`: unique sequential integer per partition.
- `RANK()`: rank with ties; gaps after ties.
- `DENSE_RANK()`: rank with ties; no gaps.
- `LAG(col, n)`, `LEAD(col, n)`: value from n rows before/after.
- `FIRST_VALUE(col)`, `LAST_VALUE(col)`: boundary values.
- `NTH_VALUE(col, n)`: value from n-th row in the window.
- `SUM(col)`, `AVG(col)`, `MIN(col)`, `MAX(col)`, `COUNT(col)`: running aggregates.
- `PERCENT_RANK()`, `CUME_DIST()`, `NTILE(n)`: distribution functions.

## Frame clauses

`ROWS BETWEEN` vs `RANGE BETWEEN`:

- `ROWS`: row-count frame. `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` is exactly 7 rows.
- `RANGE`: value-based. `RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW` depends on date gaps.

For irregular time series, `RANGE` gives calendar-based windows; `ROWS` gives row-count windows. Pick based on the intent.

## Patterns used in this project

### Running total

```sql
SELECT
  order_date,
  revenue,
  SUM(revenue) OVER (ORDER BY order_date) AS running_revenue
FROM orders;
```

### Rolling 7-day mean

```sql
SELECT
  order_date,
  AVG(revenue) OVER (
    ORDER BY order_date
    RANGE BETWEEN INTERVAL 6 DAY PRECEDING AND CURRENT ROW
  ) AS revenue_7d
FROM orders;
```

### Month-over-month delta

```sql
SELECT
  month,
  revenue,
  revenue - LAG(revenue) OVER (ORDER BY month) AS delta,
  (revenue - LAG(revenue) OVER (ORDER BY month))
    * 1.0 / NULLIF(LAG(revenue) OVER (ORDER BY month), 0) AS pct_change
FROM monthly;
```

### Top-N per category

```sql
SELECT * FROM (
  SELECT
    category, product, revenue,
    ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC) AS r
  FROM products
) t WHERE r <= 5;
```

### Sessionization

```sql
WITH e AS (
  SELECT *,
    CASE WHEN event_at - LAG(event_at) OVER (PARTITION BY user_id ORDER BY event_at)
              > INTERVAL '30 minutes' THEN 1 ELSE 0 END AS is_new_session
  FROM events
),
sessioned AS (
  SELECT *,
    SUM(is_new_session) OVER (PARTITION BY user_id ORDER BY event_at) AS session_id
  FROM e
)
SELECT user_id, session_id, MIN(event_at), MAX(event_at)
FROM sessioned
GROUP BY user_id, session_id;
```

## Performance

- Window functions often add sorts. Ensure the underlying data has useful ordering (indexes, clustering keys).
- Aggregations over unbounded partitions can be expensive; in a warehouse, clustering on the PARTITION BY column helps.
- Avoid computing the same window expression multiple times; use a CTE.

## Cross-links

- `03_cohort_and_funnel_sql.md`: window functions applied to cohort retention.
- `04_performance_patterns.md`: when window functions are slow and what to do.
- `docs/theory/15_sql_for_data_scientists.md`: broader context.
