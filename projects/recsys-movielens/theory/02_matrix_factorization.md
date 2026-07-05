# Matrix Factorization for Implicit Feedback

## The setup

Let $R$ be a users x items matrix. Entry $R_{ui}$ is 1 if user $u$ interacted with item $i$, else 0 (implicit feedback). We seek user embeddings $U \in \mathbb{R}^{|\mathcal{U}| \times k}$ and item embeddings $V \in \mathbb{R}^{|\mathcal{I}| \times k}$ such that:

$$
\hat{R}_{ui} = U_u^\top V_i
$$

approximates $R_{ui}$ (or rather, the user's preference for item $i$).

## ALS for implicit feedback (Hu, Koren, Volinsky 2008)

With implicit feedback, a zero is ambiguous (user did not see the item, or saw and disliked). The ALS formulation:

$$
\min_{U, V} \sum_{u, i} c_{ui} (p_{ui} - U_u^\top V_i)^2 + \lambda (\|U\|^2 + \|V\|^2)
$$

- $p_{ui} = 1$ if $R_{ui} > 0$ else 0 (preference).
- $c_{ui} = 1 + \alpha R_{ui}$ (confidence in the observation).
- $\lambda$: regularization strength.

Solved by alternating: fix $V$, solve $U$ via a least-squares problem per user; fix $U$, solve $V$.

Fast in practice; the `implicit` Python library is a strong implementation.

## Hyperparameters

- **factors** ($k$): 50 to 200. More factors fit more detail but overfit sparse data.
- **regularization** ($\lambda$): 0.01 to 10. Higher shrinks more.
- **alpha** ($\alpha$): 10 to 100. Controls confidence boost from observed interactions.
- **iterations**: 15 to 50.

Use CV or a held-out validation to tune. `implicit` provides sensible defaults.

## BPR (Bayesian Personalized Ranking)

Alternative for implicit feedback. Instead of minimizing squared error, maximize the probability that observed items rank higher than unobserved items per user. Loss is pairwise, sampled.

Often competitive with or slightly better than ALS on ranking metrics. Available in the `implicit` library.

## Embedding geometry

After training, user and item embeddings live in the same k-dimensional space. Cosine similarity or dot product measures relevance. ANN libraries (FAISS, ScaNN, HNSW) serve top-K in milliseconds for millions of items.

## Limitations

- Linear: captures only dot-product similarity.
- Cold start: new users have no embedding.
- No covariates: cannot condition on time of day, device, session.

Two-tower architectures (next theory doc) address these by embedding users and items from features, not just IDs.

## Cross-links

- `01_collab_vs_content.md`: positioning.
- `03_two_tower_architecture.md`: the neural extension.
- `04_offline_metrics.md`: how we measure.
- `docs/theory/12_recommender_system_foundations.md`: broader framing.
