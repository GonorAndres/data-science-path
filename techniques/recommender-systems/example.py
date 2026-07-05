"""Tiny collaborative filtering example using `implicit` on synthetic data."""

import implicit
import numpy as np
from scipy.sparse import csr_matrix


def make_interactions(n_users: int = 200, n_items: int = 100, density: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_events = int(n_users * n_items * density)
    users = rng.integers(0, n_users, size=n_events)
    items = rng.integers(0, n_items, size=n_events)
    mat = csr_matrix(
        (np.ones(len(users)), (users, items)),
        shape=(n_users, n_items),
    )
    return mat


def main() -> None:
    mat = make_interactions()
    model = implicit.als.AlternatingLeastSquares(factors=16, iterations=10, regularization=0.01)
    model.fit(mat)
    user_ids, scores = model.recommend(userid=0, user_items=mat, N=5)
    print("Top 5 recommendations for user 0:")
    for i, s in zip(user_ids, scores):
        print(f"  item {i}: score {s:.4f}")


if __name__ == "__main__":
    main()
