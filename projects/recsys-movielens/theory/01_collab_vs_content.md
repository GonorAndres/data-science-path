# Collaborative vs Content-Based Recommenders

## Collaborative filtering

Uses only user-item interactions. "Users who watched X also watched Y."

- **Memory-based.** Compute similarity between users or items; recommend based on neighbors.
- **Model-based.** Factorize the interaction matrix into user and item embeddings.

Strengths:

- No need for item metadata.
- Captures subtle user preferences via latent factors.
- Scales to millions of users and items with matrix factorization and ANN.

Weaknesses:

- Cold start for new users and new items (no interaction history).
- Depends on interaction density; sparse data produces weak embeddings.
- Feedback loops: the model only learns from what has been served.

## Content-based filtering

Uses item features (genre, tags, text, images). "If you liked X, you might like Y because they are both sci-fi horror with strong female leads."

Strengths:

- Works for new items (no cold start for items).
- Explainable ("because you liked X").
- Diversifies by attribute.

Weaknesses:

- Cannot capture preferences that are not in the item features.
- Filter bubble: over-personalizes, under-explores.

## Hybrid

Most production recommenders combine both:

- Retrieve candidates via collaborative filtering.
- Score via content features.
- Re-rank with diversity constraints.
- Handle cold start via content-based fallback.

## This project's choice

MovieLens has:

- 25M interactions (dense enough for collaborative).
- Genre tags (content signal available).
- Tagging events (text-ish features).

We train three collaborative models (popularity, ALS, two-tower), and note in each theory doc how content features would extend the approach in a production setting.

## Cross-links

- `02_matrix_factorization.md`: the main collaborative approach.
- `03_two_tower_architecture.md`: the neural approach.
- `docs/theory/12_recommender_system_foundations.md`: broader framing.
