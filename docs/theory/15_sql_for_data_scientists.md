# SQL for Data Scientists: Beyond SELECT

## Why SQL still matters most

Every production DS job touches SQL. Data engineers build pipelines in SQL; analysts live in SQL; DS teams extract training data via SQL. Recruiters in Mexico screen SQL in the first technical round, every time.

Modern DS is Python-heavy, but SQL is closer to the data: it runs where the data lives, it is auditable, and it forces you to think about sets, not loops.

## The core dialects

- **PostgreSQL.** Open-source, feature-rich, the baseline for most greenfield projects.
- **MySQL / MariaDB.** Popular in older stacks. Fewer window-function niceties until version 8.
- **BigQuery.** GCP's warehouse. Columnar, serverless, SQL with extensions.
- **Snowflake.** Cloud warehouse. Similar to BigQuery in role.
- **Redshift.** AWS warehouse.
- **DuckDB.** In-process analytics database. Fast, single-file, excellent for local work.

Learn the ANSI SQL core; know where each dialect deviates.

## Query performance mindset

SQL is declarative: you describe what you want, not how to compute it. The engine's query planner decides the plan. Three levers you control:

1. **Schema.** Table design, data types, partitioning.
2. **Indexes.** What is indexed, in what order.
3. **Query shape.** How you write the query (joins, subqueries, CTEs).

Use `EXPLAIN ANALYZE` (PostgreSQL), `EXPLAIN` (BigQuery console), or the equivalent to see the actual plan. A hot path that takes 30 seconds is usually a missing index or a poorly chosen join order.

## Key patterns for data scientists

### Window functions

The single most valuable SQL feature for DS. Use for running totals, lag/lead differences, rankings within groups, moving averages, first/last record per group.

General syntax:

```sql
function_name(arg) OVER (
  PARTITION BY col1
  ORDER BY col2
  ROWS BETWEEN N PRECEDING AND M FOLLOWING
)
```

Worked example, per-customer running revenue, 3-row centered average, and rank:

```sql
SELECT
  customer_id,
  order_date,
  revenue,
  SUM(revenue) OVER (
    PARTITION BY customer_id
    ORDER BY order_date
    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
  ) AS running_total,
  AVG(revenue) OVER (
    PARTITION BY customer_id
    ORDER BY order_date
    ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING
  ) AS centered_3row_avg,
  ROW_NUMBER() OVER (
    PARTITION BY customer_id
    ORDER BY revenue DESC
  ) AS revenue_rank
FROM orders;
```

The combination of `PARTITION BY`, `ORDER BY`, and frame clause covers most analytical SQL.

### Common Table Expressions (CTEs)

Name intermediate results with `WITH`. Improves readability for complex queries.

```sql
WITH monthly AS (
  SELECT date_trunc('month', order_date) AS month, sum(amount) AS revenue
  FROM orders
  GROUP BY 1
),
deltas AS (
  SELECT month, revenue, revenue - lag(revenue) OVER (ORDER BY month) AS delta
  FROM monthly
)
SELECT * FROM deltas WHERE delta > 0;
```

Recursive CTEs (`WITH RECURSIVE`) handle hierarchies (org charts, family trees, graph traversals).

### Anti-joins

"A where not in B" in relational algebra. Three idiomatic forms:

```sql
-- NOT EXISTS (preferred)
SELECT * FROM users u WHERE NOT EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);

-- LEFT JOIN ... IS NULL
SELECT u.* FROM users u LEFT JOIN orders o ON u.id = o.user_id WHERE o.user_id IS NULL;

-- EXCEPT / MINUS (set operation, may return distinct rows)
SELECT id FROM users EXCEPT SELECT user_id FROM orders;
```

Avoid `NOT IN` when the subquery can return NULL.

### Funnel and cohort analysis

Covered in detail in the SQL drills. The pattern:

1. Compute per-user milestones.
2. Aggregate or pivot.
3. Report conversion between milestones.

### Gaps and islands

Identify consecutive runs of events separated by gaps. The trick: subtract a row number (as days or hours) from a date; consecutive dates share a group identifier.

### Pivot without PIVOT clause

`SUM(CASE WHEN condition THEN value ELSE 0 END) AS new_col`. Portable across dialects.

## Join types

- **INNER.** Rows with matches on both sides.
- **LEFT.** All left rows; right columns are NULL when no match.
- **RIGHT.** Mirror of LEFT. Usually written as LEFT with sides swapped for readability.
- **FULL OUTER.** All rows from both, NULLs where no match.
- **CROSS.** Cartesian product. Use carefully.
- **ANTI JOIN (semi-emulated).** Rows with no match on the other side.
- **SEMI JOIN.** Rows with at least one match (but no duplication). Expressed via EXISTS in standard SQL.

## Subqueries: when and when not

- Uncorrelated subqueries: fine, usually the planner optimizes them.
- Correlated subqueries (reference the outer query): can be slow. Often equivalent to a window function, which is faster.
- Scalar subqueries in SELECT: readable but do not scale; prefer a JOIN.

## Performance patterns

### Columnar warehouses (BigQuery, Snowflake, Redshift)

- Queries scan only the columns you select. SELECT * is expensive.
- Filtering on partitioned columns dramatically reduces cost.
- Clustering on commonly filtered columns speeds queries further.
- Materialized views pre-aggregate common queries.

### Row-oriented databases (PostgreSQL, MySQL)

- Indexes matter more. A missing index on a WHERE-column is catastrophic on a large table.
- Composite indexes must match the leftmost columns of filter and ordering.
- Partial indexes (WHERE clauses in the index) narrow memory footprint.

### General tips

- Filter early, join late. Narrow down rows before joining, not after.
- Push filter conditions into join conditions (`a.x = b.x AND b.y = 5` beats filtering b after the join).
- Avoid functions on indexed columns in WHERE (`WHERE date(created_at) = '2026-04-23'` kills the index; use `created_at >= '2026-04-23' AND created_at < '2026-04-24'`).
- `SELECT DISTINCT` is sometimes papering over a join fanout; investigate.

## Date and time handling

- Store dates and timestamps in native types, not strings.
- Time zones matter. `timestamptz` in PostgreSQL; UTC is usually safe.
- `DATE_TRUNC` for grouping to period boundaries.
- Calendar math (`INTERVAL '1 month'`) handles month-length variation.
- BigQuery's `GENERATE_DATE_ARRAY` is useful for zero-filled time series.

## Nulls

Three-valued logic. `NULL = NULL` is `UNKNOWN`, not `TRUE`. Consequences:

- `WHERE col = NULL` returns nothing; use `IS NULL`.
- `NOT IN (subquery with nulls)` returns nothing; use `NOT EXISTS`.
- Aggregations skip NULL (COUNT, SUM, AVG).
- `COUNT(*)` counts rows including NULLs; `COUNT(col)` skips NULLs in col.

## SQL for feature engineering

Many features can be computed more cheaply in SQL than in pandas. Examples:

```sql
-- Rolling 7-day spend per user
SELECT
  user_id,
  date,
  SUM(amount) OVER (
    PARTITION BY user_id
    ORDER BY date
    RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW
  ) AS rolling_7d_spend
FROM orders;

-- Days since last transaction
SELECT
  user_id,
  transaction_date,
  transaction_date - LAG(transaction_date) OVER (PARTITION BY user_id ORDER BY transaction_date) AS days_since_last
FROM transactions;
```

For large datasets, computing features in the warehouse beats pulling rows into pandas.

## dbt and the analytics engineering shift

dbt is a framework for SQL transformations: model files, lineage, tests, documentation. It has become the standard at product companies.

Core concepts:

- **Models.** SQL files that define tables or views.
- **Tests.** Assertions on models (not_null, unique, accepted_values, relationships).
- **Sources.** Raw tables.
- **Documentation.** YAML descriptions, auto-published.
- **Lineage.** Auto-generated DAG.

For a DS portfolio, include at least one project with dbt structure (`sql-analytics-ecommerce` does this).

## Modern SQL capabilities worth knowing

- **Array and struct types.** BigQuery, PostgreSQL. Complex nested data.
- **JSON functions.** Extract fields from JSON columns.
- **MATCH_RECOGNIZE** (SQL:2016). Pattern matching in row sequences. Powerful for session and funnel analysis.
- **LISTAGG / STRING_AGG.** Concatenate grouped values.
- **APPROX_COUNT_DISTINCT, APPROX_QUANTILES.** Fast approximate analytics.

## Reading SQL interview questions

Interviewers love asking:

1. "Given this table, write X" (the standard drill).
2. "This query is slow, why?" (performance reasoning).
3. "There are two queries doing the same thing; which is faster and why?" (reading query plans).
4. "How would you design the schema for X?" (schema modeling).

Prepare for all four types.

## Common mistakes

1. Forgetting `GROUP BY` columns when mixing aggregates and non-aggregates.
2. `JOIN` with incorrect ON clause producing cross product.
3. Using integer division and getting zero (PostgreSQL `5 / 2 = 2`; cast to float).
4. String comparisons case-insensitive by default in some dialects, not in others.
5. Using `DISTINCT` everywhere to paper over duplication (usually a join or aggregation bug).
6. Hardcoding dates in WHERE clauses (fails next year).
7. Running `EXPLAIN` on a tiny sample and assuming the plan holds at scale.

## Linking to the portfolio

- `sql-analytics-ecommerce`: warehouse-grade SQL case study with dbt structure.
- `genai-agent-tool-use`: the agent's primary tool queries an SQL database.
- `career/interview_prep/sql_drills.md`: 40 questions with solutions.

## MERGE and UPSERT

When loading slowly changing dimensions or updating a feature store, you need "insert if new, update if existing". Two idioms.

### PostgreSQL `INSERT ... ON CONFLICT`

```sql
INSERT INTO customer_features (customer_id, ltv, updated_at)
VALUES (123, 4500.00, NOW())
ON CONFLICT (customer_id) DO UPDATE
SET ltv = EXCLUDED.ltv,
    updated_at = EXCLUDED.updated_at;
```

The `EXCLUDED` pseudo-table refers to the row that would have been inserted.

### BigQuery / Snowflake `MERGE`

```sql
MERGE INTO customer_features T
USING staging_customer_features S
ON T.customer_id = S.customer_id
WHEN MATCHED AND S.updated_at > T.updated_at THEN
  UPDATE SET ltv = S.ltv, updated_at = S.updated_at
WHEN NOT MATCHED THEN
  INSERT (customer_id, ltv, updated_at)
  VALUES (S.customer_id, S.ltv, S.updated_at);
```

`MERGE` is the SQL standard form; `INSERT ON CONFLICT` is PostgreSQL's older but more concise alternative. Both are atomic per-row.

## Lateral joins

A lateral join lets the right side of the join reference columns from the left side. Canonical use case: "for each row in A, run a parameterized subquery."

### PostgreSQL `LATERAL`

For each customer, fetch the last 3 orders:

```sql
SELECT c.customer_id, c.name, recent.*
FROM customers c
LEFT JOIN LATERAL (
  SELECT order_id, order_date, total
  FROM orders o
  WHERE o.customer_id = c.customer_id
  ORDER BY o.order_date DESC
  LIMIT 3
) recent ON TRUE;
```

Without LATERAL, this requires a window-function pattern with `ROW_NUMBER() <= 3` and is harder to read.

### BigQuery `UNNEST`

BigQuery's array columns and `UNNEST` give a similar pattern. For nested-array columns:

```sql
SELECT c.customer_id, item
FROM customers c, UNNEST(c.recent_items) AS item;
```

## Window frames: ROWS vs RANGE vs GROUPS

The frame clause inside `OVER` accepts three modes, and the difference matters when there are ties in the `ORDER BY` column.

- **ROWS BETWEEN ... .** Frame is a count of physical rows. With ties, only the listed rows enter; tied rows are kept separate.
- **RANGE BETWEEN ... .** Frame is a range of values. With ties, all rows tied with the boundary enter together.
- **GROUPS BETWEEN ... .** Frame is a count of distinct ordering values (peer groups). Newer (SQL:2011), supported in PostgreSQL 11+.

Worked example. Suppose three orders happen on the same date:

| order_id | date       | amount |
|----------|------------|--------|
| 1        | 2026-01-01 | 100    |
| 2        | 2026-01-02 | 50     |
| 3        | 2026-01-02 | 80     |
| 4        | 2026-01-02 | 70     |
| 5        | 2026-01-03 | 90     |

Running total over `ORDER BY date`:

- `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`: rows are processed one at a time. Row 2 sees 100+50=150, row 3 sees 100+50+80=230, etc.
- `RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`: rows tied on date are treated as a peer group. All three Jan 2 rows see 100+50+80+70=300.

Use ROWS for monotone-by-row computations (rank-style); use RANGE for time-bucket aggregates where ties should accumulate together.

## DISTINCT ON (PostgreSQL)

Cleanest way to get the first row per group:

```sql
SELECT DISTINCT ON (customer_id) customer_id, order_date, total
FROM orders
ORDER BY customer_id, order_date DESC;
```

The `DISTINCT ON (cols)` keeps only the first row for each unique combination of `cols`, where "first" is defined by the `ORDER BY`. Equivalent to a `ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date DESC) = 1` filter, but shorter and often faster.

Not in standard SQL; use `ROW_NUMBER` for portability.

## System-versioned (temporal) tables

The SQL:2011 temporal-tables feature lets the database automatically maintain row-level history.

```sql
CREATE TABLE customer_state (
  customer_id BIGINT,
  ltv NUMERIC,
  valid_from TIMESTAMP NOT NULL,
  valid_to TIMESTAMP NOT NULL,
  PERIOD FOR system_time (valid_from, valid_to)
) WITH SYSTEM VERSIONING;
```

Updates to a versioned table preserve the previous row in a history table; queries can be issued "AS OF" a past timestamp:

```sql
SELECT ltv FROM customer_state FOR system_time AS OF '2026-01-15' WHERE customer_id = 42;
```

Critical for **point-in-time feature retrieval**: when training a model, you must reconstruct each feature as it was known at the prediction time. System-versioned tables give this for free; otherwise you must add `valid_from`/`valid_to` columns and write the AS-OF join by hand. Available in SQL Server, MariaDB; emulated in PostgreSQL via extensions.

## EXPLAIN ANALYZE: reading a query plan

`EXPLAIN ANALYZE` runs the query and reports the actual plan with timings. Worked example:

```sql
EXPLAIN ANALYZE
SELECT u.id, u.email, COUNT(o.id) AS orders
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at > '2026-01-01'
GROUP BY u.id, u.email;
```

Sample output:

```
HashAggregate  (cost=12340.00..12390.00 rows=5000 width=40) (actual time=85.2..86.1 rows=4823 loops=1)
  Group Key: u.id, u.email
  ->  Hash Right Join  (cost=420.00..11850.00 rows=98000 width=32) (actual time=2.1..72.4 rows=97431 loops=1)
        Hash Cond: (o.user_id = u.id)
        ->  Seq Scan on orders o  (cost=0.00..8500.00 rows=500000 width=16) (actual time=0.01..28.9 rows=500000 loops=1)
        ->  Hash  (cost=380.00..380.00 rows=5000 width=24) (actual time=2.0..2.0 rows=4823 loops=1)
              ->  Index Scan using users_created_at_idx on users u  (cost=0.42..380.00 rows=5000 width=24) (actual time=0.02..1.4 rows=4823 loops=1)
                    Index Cond: (created_at > '2026-01-01'::timestamp)
Planning time: 0.42 ms
Execution time: 86.5 ms
```

What to look for:

- **Seq Scan vs Index Scan.** A Seq Scan on a large table inside a hot path is usually a missing index. Here `orders` is sequentially scanned because there is no filter on it; that is fine for a join but worth checking if the query selects only a subset.
- **Hash Join vs Nested Loop.** Hash joins win on large unsorted inputs; nested loops win on small inner relations with index access. The optimizer chooses based on row estimates.
- **Estimated rows vs actual rows.** Big mismatches (`rows=5000` estimate vs `actual rows=4823`) are usually fine; mismatches like 5,000 estimate vs 5,000,000 actual signal stale statistics. Run `ANALYZE` to refresh.
- **Loops.** A nested loop with `loops=10000` and inner cost 1 ms each is 10 seconds, even if the plan looks cheap per iteration.
- **Buffers.** Add `BUFFERS` to see read and hit counts; high read counts on a hot path mean the working set does not fit in cache.

## References

- Tozer, *SQL for Data Analysis*, 2021.
- Celko, *SQL for Smarties* (advanced patterns).
- Markus Winand, *SQL Performance Explained*, use-the-index-luke.com (free).
- Getting Started with dbt, docs.getdbt.com.
- Mode Analytics SQL tutorial, mode.com/sql-tutorial.

## Exercises

1. Write a SQL query using a lateral join to get the 3 most recent claims for each policyholder.
2. Explain the difference between `ROWS BETWEEN 6 PRECEDING AND CURRENT ROW` and `RANGE BETWEEN 6 PRECEDING AND CURRENT ROW` in a window function on a date column with ties.
3. A slowly-changing-dimension type 2 table tracks policyholder address history. Write a query using a temporal join to find the correct address at a given effective date.
4. Read this `EXPLAIN ANALYZE` output: `Seq Scan on claims (cost=0.00..15000.00 rows=1000000) (actual time=0.1..8200.0 rows=1000000)`. What does this tell you, and what is the fix?
5. Write a `MERGE` statement to upsert monthly claim aggregates into a summary table, inserting new rows and updating existing ones.
