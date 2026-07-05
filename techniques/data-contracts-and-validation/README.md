---
tags: [data-quality, validation, contracts]
status: draft
created: 2026-04-23
updated: 2026-04-23
---

# Data Contracts and Validation

## What it is

Formal specification of the shape and constraints of datasets, with automated checks. Think of it as types for tables.

## Why it matters

ML pipelines silently break when upstream data changes. A new NULL category, a renamed column, a shifted distribution. Without contracts, the break manifests as a model that silently underperforms.

Data contracts catch changes at the source, before they propagate downstream.

## When to use it

- Production pipelines with multiple teams.
- Regulated industries.
- Model serving that depends on feature stability.

## Tools

- **Great Expectations.** Schema + distribution checks, with HTML reports.
- **Pandera.** Pandas-specific; decorator-based.
- **dbt tests.** SQL-based; integrates with dbt models.
- **Soda.** Rule-based data quality.
- **WhyLogs.** Profiling + drift.

## Patterns

- **Schema validation.** Column names, types, nullability.
- **Distribution checks.** Numeric ranges, categorical membership.
- **Cross-column checks.** Referential integrity, date ordering.
- **Uniqueness checks.** Primary keys are unique.
- **Row count bounds.** Too few or too many rows is a red flag.

## Gotchas

- Over-strict schemas block legitimate upstream evolution.
- Too-loose schemas do not catch real breakage.
- Alert fatigue: 50 checks with frequent false positives get muted.

## Resources

- Great Expectations docs: docs.greatexpectations.io.
- Pandera docs: pandera.readthedocs.io.
- dbt docs: docs.getdbt.com.
