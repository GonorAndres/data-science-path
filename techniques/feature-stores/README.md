---
tags: [feature-store, mlops, data-engineering]
status: complete
created: 2026-04-23
updated: 2026-04-25
---

# Feature Stores

## What it is

A feature store is a database layer specifically engineered for ML features. It enforces three invariants:

1. **Dual serving.** The same feature can be retrieved historically (for training) and in real time (for inference) with consistent semantics.
2. **Point-in-time correctness.** When constructing a training set, every feature value reflects only data known as of the label's timestamp; no future leakage.
3. **Reuse.** Features are defined once and consumed by many models, owned by the team that produces the source data, with documented lineage.

It is not just a database. It is a contract between data producers and model consumers, plus the infrastructure that enforces the contract.

## The training-serving consistency contract

The single most insidious failure mode in production ML is **training-serving skew**: the feature passed to the model at inference does not equal the feature the model was trained on, because the two are computed by different code paths.

Concrete examples of how skew arises:

- A SQL aggregation in the training pipeline computes "average transaction in the last 30 days" using daily-snapshot tables; the online service approximates the same feature with a Redis sliding-window counter that resets at midnight UTC. The two diverge by 5 to 20 percent depending on the time of day.
- The training pipeline uses a one-hot encoder fit on the full historical vocabulary; the online encoder uses a stale fit from last month's data and silently maps new categories to "unknown".
- Training imputes missing income with the median over the training set; production imputes with a hardcoded constant from a config file last updated two years ago.

Each of these costs accuracy that no model improvement can recover. The feature store solves the problem by making the feature definition the **single source of truth**: one transformation, applied identically in both batch (offline materialization) and streaming or on-demand (online retrieval) paths. The DAG between raw data and feature is checked into version control alongside the model code.

When a feature store is missing the workaround is rigorous code reuse: the same Python function fed by `pandas` at training and by a serving wrapper at inference, plus a test that asserts the two produce identical outputs on a fixed input. This works for small teams; it does not scale to dozens of models sharing features.

## Core concepts

| Concept | Definition | Example |
|---------|------------|---------|
| Entity | The primary key features attach to | `customer_id`, `policy_id`, `claim_id` |
| Feature view | A named group of features defined over an entity at a time window | `customer_spend_30d` containing `total_spend_30d`, `n_transactions_30d`, `avg_ticket_30d` |
| Data source | Where the raw data lives | A BigQuery table, a Kafka topic, a Parquet path on GCS |
| Materialization | The act of computing features from sources and writing them to the store | Daily Airflow DAG that runs the SQL transformation |
| Offline store | The historical feature record, append-only, used for training | BigQuery, Snowflake, Parquet on GCS |
| Online store | The fresh-only feature record, key-value, used for inference | Redis, DynamoDB, Bigtable |

A model's training pipeline declares the entities it needs and the feature views it consumes; the feature store joins them point-in-time correctly and produces a training matrix. The serving pipeline declares the same entities and feature views and gets the latest values for inference.

## Point-in-time joins

The historical join must answer: "for each row in my training labels (entity, label_timestamp), what was the value of each feature **as of** that timestamp?" Naively joining on entity alone leaks future information; joining on entity + exact timestamp drops every row that does not have a feature update at that exact moment.

The correct semantics is **as-of join**: for each (entity, label_timestamp), find the latest feature row for that entity with `feature_timestamp <= label_timestamp` and within an allowed staleness window.

Pseudocode:

```python
def asof_join(labels, features, entity, label_ts, feature_ts, max_age):
    out = []
    for row in labels:
        candidates = features[
            (features[entity] == row[entity]) &
            (features[feature_ts] <= row[label_ts]) &
            (row[label_ts] - features[feature_ts] <= max_age)
        ]
        if len(candidates) == 0:
            out.append({**row, **{f: None for f in feature_columns}})
        else:
            latest = candidates.sort_values(feature_ts).iloc[-1]
            out.append({**row, **latest[feature_columns].to_dict()})
    return pd.DataFrame(out)
```

In Feast, BigQuery, or Snowflake the same logic is expressed in SQL as a windowed `LAST_VALUE` or via the database's native `ASOF JOIN`. Implementing this by hand in pandas works for prototyping; production volumes need a SQL engine.

## Feast end-to-end example

Feast is the open-source default. Other options (Tecton, Hopsworks, Sagemaker Feature Store) follow the same pattern with managed infrastructure.

### 1. Define the entity

```python
# entities.py
from feast import Entity, ValueType

customer = Entity(
    name="customer_id",
    value_type=ValueType.INT64,
    description="Unique customer identifier",
)
```

### 2. Define a feature view

```python
# feature_views.py
from feast import FeatureView, Field, BigQuerySource
from feast.types import Float32, Int64
from datetime import timedelta

source = BigQuerySource(
    table="my_project.analytics.customer_spend_daily",
    timestamp_field="event_ts",
)

customer_spend_30d = FeatureView(
    name="customer_spend_30d",
    entities=[customer],
    ttl=timedelta(days=2),  # max staleness in the online store
    schema=[
        Field(name="total_spend_30d",   dtype=Float32),
        Field(name="n_transactions_30d", dtype=Int64),
        Field(name="avg_ticket_30d",    dtype=Float32),
    ],
    source=source,
    online=True,
)
```

### 3. Apply and materialize

```bash
feast apply
feast materialize-incremental $(date -u +%Y-%m-%dT00:00:00)
```

`apply` registers the definitions in the registry. `materialize-incremental` computes the latest feature values for each entity and writes them to the online store (Redis or similar) up to the specified timestamp.

### 4. Retrieve features for training

```python
from feast import FeatureStore
store = FeatureStore(repo_path=".")

training_df = store.get_historical_features(
    entity_df=labels_df,  # has customer_id and event_timestamp columns
    features=[
        "customer_spend_30d:total_spend_30d",
        "customer_spend_30d:n_transactions_30d",
        "customer_spend_30d:avg_ticket_30d",
    ],
).to_df()
```

The point-in-time join happens server-side in BigQuery; the result is a training matrix where each row's features reflect the customer's state as of `event_timestamp` and nothing after.

### 5. Retrieve features for online inference

```python
features = store.get_online_features(
    features=[
        "customer_spend_30d:total_spend_30d",
        "customer_spend_30d:n_transactions_30d",
        "customer_spend_30d:avg_ticket_30d",
    ],
    entity_rows=[{"customer_id": 12345}],
).to_dict()
```

The same feature names. The same definition. Returned in single-digit milliseconds from Redis.

## Architecture patterns

### Push vs pull materialization

- **Pull (batch).** A scheduled job (Airflow, Dagster, dbt) reads the source, applies the feature definition, and writes to the store. Latency from raw event to materialized feature is on the order of an hour to a day. Default for slowly changing features.
- **Push (streaming).** A Kafka or Pub/Sub consumer applies the same feature definition to events as they arrive and writes to the online store directly. Latency on the order of seconds. Required for features where freshness matters: real-time fraud signals, session-based recommendation features, dynamic pricing inputs.

Production systems combine both. A feature view declares a streaming source for the online path and a batch source for backfill and offline training. The feature store ensures both paths produce identical values.

### Streaming feature updates with Kafka

```
Raw events  ->  Kafka topic  ->  Stream processor (Flink, Beam, or Faust)
                                    |
                                    v
                            applies feature transformation
                                    |
                                    v
                             Online store (Redis)

Offline materialization  ->  same transformation in batch  ->  Offline store (BigQuery)
```

The transformation function is shared between the stream processor and the batch job. Many feature stores compile a single feature definition into both runtimes; alternatively, write the transformation in shared Python and execute it from both jobs.

## When you need a feature store, and when you do not

Reach for a feature store when:

- More than two or three models share features and the duplication is causing skew incidents.
- Online inference requires sub-100ms feature lookups across many features per request.
- Compliance requires point-in-time auditability ("show me what features the model saw on day X for customer Y").
- The organization has dedicated feature owners (data engineers responsible for upstream signals) who need a contract with downstream model consumers.

Skip the feature store when:

- A single team owns one or two models end to end and feature reuse is not yet a problem.
- Inference is fully batch (predictions written to a table on a schedule) and a SQL view in the warehouse already provides correct point-in-time joins.
- Latency requirements allow recomputing features at request time from a primary database.

The premature feature store is a classic over-engineering trap. Start with shared SQL views plus a Python module of feature transformations; promote to a real feature store when training-serving skew incidents become recurring.

## Common gotchas

1. **TTL too short.** If the online store TTL expires before the materialization job refreshes the feature, online lookups return null and the model gets a default value at inference. Set TTL conservatively and monitor null rates.
2. **Backfill cost.** Materializing a feature view back over years of history can be expensive in BigQuery scan costs. Partition by date, materialize incrementally, and budget the backfill.
3. **Schema evolution.** Adding a feature is safe; renaming or removing one breaks every consumer. Treat the feature view as a public API: deprecate first, remove later.
4. **Time-zone bugs in `event_timestamp`.** Mixing UTC and local time in the entity dataframe and the source table causes silent off-by-one-day joins. Standardize on UTC end to end.
5. **Streaming vs batch divergence.** Even when the transformation function is shared, streaming aggregations have to handle late-arriving events while batch jobs assume completeness. Document the agreed semantics.

## Resources

- `docs/theory/13_mlops_architecture_patterns.md` for where the feature store fits in the broader stack.
- `techniques/data-contracts-and-validation/README.md` for the producer-side discipline.

## References

- Feast project, feast.dev.
- Tecton blog and architecture papers, tecton.ai.
- Hopsworks documentation, docs.hopsworks.ai.
- Sculley et al., "Hidden Technical Debt in Machine Learning Systems", NeurIPS 2015 (the canonical reference for training-serving skew).
- Lakshmanan, Robinson, Munn, *Machine Learning Design Patterns*, O'Reilly 2020.
