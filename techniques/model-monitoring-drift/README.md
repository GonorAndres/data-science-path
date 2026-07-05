---
tags: [monitoring, drift, mlops]
status: draft
created: 2026-04-23
updated: 2026-04-23
---

# Model Monitoring and Drift Detection

## What it is

Post-deployment observability for ML models. Beyond uptime and latency (standard SRE), ML needs:

- **Data drift.** Input distribution shift.
- **Concept drift.** Input-output relationship shift.
- **Label drift.** Distribution of outcomes shifts.
- **Performance decay.** Offline metrics degrading over time.

## Why it matters

A healthy service can silently serve a decaying model. Without monitoring, you discover the problem from business dashboards weeks late.

## Drift metrics

- **Population Stability Index (PSI).** Common on tabular.
- **Kolmogorov-Smirnov.** Continuous distributions.
- **KL or JS divergence.** Distributions.
- **Chi-square.** Categorical.
- **Maximum Mean Discrepancy.** Multivariate, kernel-based.
- **Embedding drift.** For high-dimensional inputs (text, images), compare embedding distributions.

## Thresholds

Heuristic for PSI:

- < 0.1: no significant drift.
- 0.1 to 0.25: moderate; investigate.
- > 0.25: significant; retrain or act.

Thresholds are context-specific; set from your own baseline.

## Tools

- **Evidently AI.** Open-source; dashboards, tests.
- **WhyLogs.** Open-source; profiling.
- **Arize, Fiddler, WhyLabs, Aporia.** Commercial.
- **Vertex AI Model Monitoring, Sagemaker Model Monitor.** Cloud-native.

## Gotchas

- High-cardinality features produce unstable PSI.
- Seasonality triggers false drift. Compare to a seasonally-matched reference.
- Label drift alone is not actionable; context matters.

## Resources

- Rabanser, Gunnemann, Lipton. "Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift". 2019.
- Evidently AI docs: docs.evidentlyai.com.
- `docs/theory/13_mlops_architecture_patterns.md`.
