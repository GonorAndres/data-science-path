# Two-Tower Architecture

## Why two-tower

Matrix factorization embeds raw user and item IDs. It cannot use metadata (age, genre, release year) or interaction context (time of day).

Two-tower addresses this by replacing the ID lookup with a small neural network per side:

```
user features   ->  user tower  ->  user embedding (dim k)
                                         |
                                         v
                                   dot product  ->  score
                                         ^
                                         |
item features   ->  item tower  ->  item embedding (dim k)
```

The score is still a dot product, which means we can serve top-K retrieval via ANN at production speed. But embeddings now incorporate features.

## Training

Each training example: (user features, positive item features, label=1). Plus $K$ negatives per positive (label=0), sampled from the item catalog (uniform, or in-batch).

Loss: binary cross-entropy on the dot-product score, or sampled softmax for multi-class.

Optimizer: Adam, learning rate ~1e-3, batch ~1024+. Train 5 to 20 epochs. Validate on held-out user-item pairs.

## Embedding dimension

Start at 64. Larger ($k$ up to 256) can help if data is dense; smaller if sparse. Memory scales linearly.

## Architecture choices

User tower:

- Embedding layer for user ID.
- Optional embeddings for categorical features (age bucket, country).
- Concatenate + 1 to 3 dense layers with ReLU.
- Output: L2-normalized embedding.

Item tower:

- Similar pattern for item ID and metadata (genre one-hot, release year).
- L2-normalize output.

L2 normalization before dot product turns the score into cosine similarity. This helps training stability and works well with ANN indexes.

## Serving

At serving time:

- Precompute and index item embeddings in FAISS or ScaNN.
- Compute user embedding at request time.
- Query the ANN for top-K.

Typical latency: under 10 ms for the ANN lookup on millions of items.

Re-ranking (ranking model on the top 1,000 candidates) happens downstream.

## Negative sampling

The choice of negatives strongly affects training:

- **Uniform.** Simple, stable. Slight bias: popular items appear as both positives and negatives.
- **In-batch.** Use other users' positives as negatives for the current user. Efficient.
- **Popularity-biased.** Weight negatives by popularity; corrects for exposure bias.
- **Hard negatives.** Mine negatives that are "close" to positives in embedding space. Advanced technique; dramatically speeds up training but risks instability.

This project uses uniform sampling; popularity-biased is documented as an upgrade path.

## Pitfalls

- **Position bias.** Training on logged CTR data inherits exposure bias. Counterfactual correction is an advanced topic.
- **Freshness.** Item embeddings become stale. Retrain or use streaming updates.
- **Cold start.** Two-tower helps new items via metadata but still needs a feature-rich item tower.

## Cross-links

- `02_matrix_factorization.md`: the simpler approach.
- `04_offline_metrics.md`: evaluation.
- `docs/theory/12_recommender_system_foundations.md`: broader framing.
- Covington et al. 2016 (YouTube) is the canonical reference.
