---
tags: [embeddings, categorical, tabular]
status: draft
created: 2026-04-23
updated: 2026-04-23
---

# Entity Embeddings

## What it is

A technique for encoding high-cardinality categorical variables (user IDs, product IDs, zip codes) into dense vectors learned jointly with a downstream model.

## Why it matters

One-hot encoding blows up dimensionality. Target encoding leaks. Entity embeddings give dense representations that capture semantic similarity (e.g., zip codes in the same region land near each other in vector space).

## When to use it

- Tabular data with high-cardinality categoricals.
- Recommenders (user and item embeddings).
- Any deep net ingesting categorical IDs.

Skip for low-cardinality categoricals (one-hot is fine).

## Key concepts

- **Embedding table.** Learnable lookup; one row per category.
- **Embedding dimension.** Hyperparameter; rule of thumb: min(50, (cardinality + 1) / 2).
- **Initialization.** Random; sometimes pretrained from another task.

## Python implementation

See `example.py` for a simple embedding + MLP model on synthetic data.

## Resources

- Guo, Berkhahn. "Entity Embeddings of Categorical Variables". 2016.
- `docs/theory/05_feature_engineering_cookbook.md`.
- fast.ai tabular tutorial.
