---
tags: [recommenders, collaborative-filtering, ranking]
status: review
created: 2026-04-23
updated: 2026-04-23
---

# Recommender Systems

## What it is

Systems that predict what users will want next. Two main paradigms:

- **Collaborative filtering.** Learn from user-item interactions.
- **Content-based.** Learn from item features.

## Why it matters

Recommenders drive engagement at Netflix, Spotify, Amazon, Mercado Libre, TikTok. Every product company has one or will want one.

## When to use it

- Catalog of items + some interaction signal.
- Need to personalize at scale.
- Cold-start tolerable (or have content features to handle it).

Skip if the catalog is tiny (manual curation wins) or if cold-start dominates (every user and item is new; too little signal).

## Methods

1. **Popularity.** Baseline.
2. **User-based or item-based CF.** Similarity-driven, memory-based.
3. **Matrix factorization.** ALS on implicit feedback, BPR.
4. **Two-tower neural.** Separate encoders for users and items; dot-product for scoring.
5. **Sequential.** GRU4Rec, SASRec, BERT4Rec. Model the sequence of interactions.
6. **Graph-based.** PinSage, LightGCN.
7. **LLM-augmented.** Prompt a model with user history; still early.

## Metrics

Offline: Recall@K, NDCG@K, coverage, diversity.
Online: CTR, conversion, revenue per session.

## Python implementation

See `example.py` for implicit ALS on MovieLens-100k.

## Gotchas

- Random time splits leak future into the past. Use leave-last-out per user.
- Implicit feedback treats zeros ambiguously. Use confidence-weighted ALS or BPR.
- Position bias: top slots always get more clicks. Control via IPW.
- Filter bubbles. Include exploration.

## Resources

- `docs/theory/12_recommender_system_foundations.md`.
- Koren, Bell, Volinsky (matrix factorization).
- Covington et al. (YouTube deep neural networks).
- `implicit` Python library.
