# Performance Patterns

## DuckDB specifics

- Columnar storage; queries scan only needed columns.
- Automatically parallel across cores.
- `CREATE TABLE AS SELECT ...` materializes; `CREATE VIEW` is virtual.
- Reads Parquet directly (`SELECT * FROM 'path.parquet'`).
- Supports `ATTACH DATABASE` for multiple stores.

For a project of this size (100k orders) everything runs in seconds. Performance patterns matter more at terabyte scale in BigQuery / Snowflake / Redshift.

## Warehouse patterns (BigQuery / Snowflake / Redshift)

### 1. Filter early

```sql
-- bad
SELECT *
FROM big_table a
JOIN other b USING (id)
WHERE a.created_at > '2026-01-01';

-- good
SELECT *
FROM (SELECT * FROM big_table WHERE created_at > '2026-01-01') a
JOIN other b USING (id);
```

Though modern query planners do push down filters, being explicit avoids surprises.

### 2. Avoid SELECT *

Columnar stores charge by columns scanned. Select only what you need.

### 3. Partition and cluster

Partition on a time column (ingestion date, event date) for time-series data. Cluster on commonly filtered columns (user_id, region).

BigQuery: `PARTITION BY`, `CLUSTER BY`.
Snowflake: clustering keys.
Redshift: distribution and sort keys.

### 4. Avoid functions on indexed columns in WHERE

```sql
-- bad (index on created_at cannot be used)
WHERE DATE(created_at) = '2026-04-23';

-- good
WHERE created_at >= '2026-04-23' AND created_at < '2026-04-24';
```

### 5. Prefer EXISTS over NOT IN

`NOT IN` is broken by NULLs and often slower.

### 6. Approximate functions when exact is not needed

```sql
SELECT APPROX_COUNT_DISTINCT(user_id) FROM events;
```

10x to 100x faster with ~1 percent error.

### 7. Materialize expensive CTEs

A CTE that is referenced multiple times gets recomputed in most dialects. Materialize to a temp table or use a materialized view.

### 8. Join order and hints

Query planners usually pick good join orders. If not, use query hints (BigQuery `@BROADCAST`, Snowflake hints) sparingly.

## When to give up on SQL and move to Python

- Recursive algorithms that are not naturally expressed in SQL (graph algorithms).
- Custom ML model scoring (though BigQuery ML and Snowpark cover more ground yearly).
- Text processing beyond regex.

Most analytics stays in SQL.

## Testing

Every mart should have tests at the mart level:

- **Row count.** Fails if empty or if row count shrinks unexpectedly.
- **Null checks.** Primary keys and foreign keys must not be null.
- **Uniqueness.** Primary keys must be unique.
- **Range checks.** Revenue positive, dates in expected window, status values from a controlled vocabulary.

This project implements these via `src/tests.py` running simple SQL assertions.

## Cross-links

- `docs/theory/15_sql_for_data_scientists.md`
- Use the Index Luke by Markus Winand.
