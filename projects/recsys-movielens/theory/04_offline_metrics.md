# Offline Metrics for Recommenders

## The mismatch

Offline metrics measure the model's ability to reproduce held-out interactions. Online metrics measure the business outcome (CTR, revenue, retention). They correlate loosely, not strongly. Always A/B test in production.

But offline metrics are necessary for rapid iteration. This doc covers the standard set.

## Leave-last-out evaluation

For each user:

- Sort interactions by timestamp.
- Last interaction is the test item.
- Remaining interactions are training data.

This mimics temporal deployment. An alternative (random 80/20 per user) is easier to implement but less realistic; we use leave-last-out as the canonical split.

## Recall@K

$$
\text{Recall@K}(u) = \frac{|\text{relevant items in top-K}|}{|\text{relevant items total}|}
$$

With leave-last-out, there is exactly one relevant item per user, so Recall@K is either 0 or 1. Average across users:

$$
\text{Recall@K} = \frac{1}{|\mathcal{U}|} \sum_{u \in \mathcal{U}} \mathbb{1}[\text{test item}_u \in \text{top-K}_u]
$$

This is often called **Hit Rate@K** in that case.

## NDCG@K

Normalized Discounted Cumulative Gain. Weights earlier positions higher:

$$
\text{DCG@K} = \sum_{i=1}^{K} \frac{rel_i}{\log_2(i + 1)}
$$

Normalize by the ideal DCG (perfect ranking).

With one relevant item at position $r$: $\text{NDCG@K} = 1 / \log_2(r + 1)$ if $r \leq K$, else 0.

## MRR (Mean Reciprocal Rank)

$$
\text{MRR} = \frac{1}{|\mathcal{U}|} \sum_{u} \frac{1}{\text{rank}_u}
$$

$rank_u$ is the position of the relevant item in the ranked list. 1 if top, 0 if not found.

## Coverage

What fraction of the catalog is ever in any user's top-K?

$$
\text{Coverage@K} = \frac{|\bigcup_u \text{top-K}_u|}{|\mathcal{I}|}
$$

Low coverage indicates a model that always recommends the same popular items.

## Diversity

Intra-list diversity: average pairwise dissimilarity among items in a user's top-K list. Uses item embeddings or features (genre overlap).

Higher is usually better, but too high indicates noise.

## Novelty

How rare are the recommended items:

$$
\text{Novelty}(u) = -\frac{1}{K} \sum_{i \in \text{top-K}_u} \log_2 p(i)
$$

$p(i)$ is the item's global popularity. Higher is more novel.

## Popularity bias diagnostic

Compute the share of recommendations that fall in the top-10-percent most-popular items. A model that puts 80 percent of its top-K in the head of the popularity distribution is biased; if business wants long-tail exposure, this is a flag.

## A@K / P@K per user

Some users get perfect recommendations; some get none. Report the distribution, not just the mean.

## Evaluation protocol for this project

1. Leave-last-out split on MovieLens 25M.
2. For each user, score all candidate items (movies they have not rated).
3. Report Recall@20, NDCG@20, MRR, Coverage@20, Novelty.
4. Compare across popularity baseline, ALS, two-tower.

## Common mistakes

- Random split on time-ordered data. Train sees future; metric inflated.
- Testing on items that are in the training set. Shrink the candidate pool to items not yet seen.
- Reporting only aggregate metrics. The distribution matters.
- Ignoring catalog coverage. A recommender that repeats popular titles gets good Recall but poor novelty.

## Cross-links

- `00_problem_framing.md`: the business framing.
- `02_matrix_factorization.md`: ALS.
- `03_two_tower_architecture.md`: neural.
- `docs/theory/12_recommender_system_foundations.md`: broader framing.
