# Problem Framing: MovieLens Recommender

## What we are predicting

Given a user's past movie ratings, predict the next movies they will watch (or rate). Output: for each user, a ranked list of candidate movies they have not rated yet.

This is a **next-item prediction** problem framed as a **top-K retrieval** task: we care about recall at the top of the list, not absolute rating prediction.

## Why not predict star ratings directly?

Historical star-rating prediction (RMSE-style) has fallen out of favor for two reasons:

1. Business never used the raw number. They used ranked recommendations. RMSE of the rating is a poor proxy for ranking quality.
2. Implicit feedback (views, clicks, purchases) is 100x more abundant than explicit ratings. Models that can exploit implicit signals win in production.

We predict "did the user interact with this movie" as a binary task with negative sampling, and rank by model score.

## Metrics

- **Recall@K:** of the user's actual next interactions, what fraction appear in our top-K? Primary metric.
- **NDCG@K:** ranked-quality measure; higher weight to earlier positions.
- **MRR:** mean reciprocal rank of the first correct item.
- **Coverage:** what fraction of the catalog is ever recommended across all users.
- **Diversity:** how different are recommended items within a list.

## Stakeholders (hypothetical)

- **Product:** wants higher engagement (CTR, time spent).
- **Content team:** wants diverse coverage (promote long-tail titles).
- **User research:** wants recommendations to feel relevant and non-spammy.

In a real product, these tensions are resolved via re-ranking and policy rules on top of the scoring model.

## Non-goals

- Not predicting exact rating values.
- Not modeling session dynamics (next click in a single session).
- Not incorporating external data (poster images, plot synopsis, tags beyond MovieLens tags).

## Cross-links

- `docs/theory/12_recommender_system_foundations.md`: broader recsys theory.
- `01_collab_vs_content.md`: method tradeoffs.
