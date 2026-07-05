---
tags: [mlops, deployment, production]
status: review
created: 2026-04-23
updated: 2026-04-23
---

# MLOps Fundamentals

## What it is

The operational discipline for ML in production: versioning, reproducibility, CI/CD, serving, monitoring, governance.

## Why it matters

A trained model has no business value until it serves real traffic reliably. MLOps is the bridge.

## Core practices

### Reproducibility

- Pin dependencies (requirements.txt, lock files).
- Version data (DVC, lakeFS).
- Version models (MLflow, Sagemaker registry).
- Containerize runtime (Docker).

### CI/CD for ML

- Code CI: lint, test, type check.
- Data validation: schema, distributions.
- Model validation: offline metrics vs production baseline.
- Deploy: blue-green or canary.
- Rollback on alert.

### Serving patterns

- Batch: scheduled runs, write to table.
- Online: HTTP API, low latency.
- Streaming: event-driven, Kafka or PubSub.
- Edge: on-device.

### Monitoring

- Operational: latency, error rate.
- Data quality: drift, missingness.
- Model: prediction distribution, online vs offline metrics.

### Feature stores

Share features across training and serving with point-in-time correctness. Feast, Tecton.

## When to adopt each practice

Small project: pin deps, track MLflow experiments, Docker for serving.
Medium team: add CI/CD, feature store, staging environment.
Large: full platform, registries, automated retraining, governance committee.

## Gotchas

- Training-serving skew: features computed differently online vs offline.
- Silent model decay: no monitoring means you notice only when the business does.
- Stale data: pipelines break; feature freshness drops without anyone noticing.
- Shadow deploys without A/B analysis: "we shipped it and it works" is not a measurement.

## Resources

- `docs/theory/13_mlops_architecture_patterns.md`.
- Huyen. *Designing Machine Learning Systems*. 2022.
- Google MLOps guide: cloud.google.com/resources/mlops-whitepaper.
- See `projects/mlops-model-service` for a reference implementation.
