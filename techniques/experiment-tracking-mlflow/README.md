---
tags: [mlflow, experiment-tracking, mlops]
status: draft
created: 2026-04-23
updated: 2026-04-23
---

# Experiment Tracking with MLflow

## What it is

MLflow records:

- Hyperparameters per run.
- Metrics (during and after training).
- Artifacts (models, plots, reports).
- Source code version.
- A full audit trail.

Plus a model registry for versioning and stage promotion.

## Why it matters

Without experiment tracking, you cannot compare yesterday's run with today's. The same dataset tuned by different people produces different numbers with no way to reconcile. Tracking is the baseline of reproducibility.

## When to use it

- Any ML project with more than 3 trial runs.
- Teams where multiple people train models.
- Regulated industries (audit trail).

## Core concepts

- **Experiment.** Namespace for related runs.
- **Run.** A single training invocation.
- **Tag.** Metadata labels (author, purpose).
- **Registered model.** A named model versioned over time.
- **Stage.** dev, staging, production, archived.

## Alternatives

- **Weights & Biases.** Commercial, slicker UI, paid.
- **Neptune.** Similar space.
- **ClearML.** Open-source + commercial.
- **Internal.** Large companies often build their own.

MLflow is the de facto open-source default.

## Gotchas

- Default file-based backend is fine local but not multi-user. Use PostgreSQL + S3 (or GCS) for production.
- Logging large artifacts slows runs; log summaries instead.
- Autolog is magic but opaque; know what it logs.

## Resources

- MLflow docs: mlflow.org.
- `docs/theory/13_mlops_architecture_patterns.md`.
